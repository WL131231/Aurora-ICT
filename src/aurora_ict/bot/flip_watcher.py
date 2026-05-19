"""FlipWatcher — HTF FVG flip 실시간 감시기 (변경 7 핵심).

ICT 원칙: FVG zone 1회 touch = mitigation 인정 = 즉시 flip. wick 도 touch.

구조:
    - WS (Bybit V5 ticker) 가 primary 가격 소스.
    - WS 끊기면 REST polling (0.2s 간격) 으로 fallback.
    - WS 복구되면 polling stop.
    - tick 마다:
        * htf_map 전체에 대해 "밖 → 안" 전환 감지 → touch_count += 1
        * active_position.htf_flip_target zone 진입 → REST 가격 재확인 → flip 발동

정확성 원칙:
    - 0/None/NaN 가격 무시.
    - flip 발동 직전 REST 로 가격 재확인 (WS 이상치 거름).
    - flip 한 번 발동 후 동일 target ts_ms 재발동 방지.
    - 청산 → 진입 sequential, 모든 예외 흡수 + ERROR 로그 (봇 안 멈춤).
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from enum import StrEnum
from typing import TYPE_CHECKING

from aurora_ict.streams.bybit_ws import subscribe_ticker

if TYPE_CHECKING:
    from aurora_ict.bot.bot_ict_instance import BotIctInstance

logger = logging.getLogger(__name__)


class WatchMode(StrEnum):
    """가격 소스 상태 머신."""

    WS_ACTIVE = "ws_active"
    POLLING_ACTIVE = "polling_active"
    STOPPED = "stopped"


class FlipWatcher:
    """active_position.htf_flip_target 박혀있는 동안 가격을 tick 단위로 감시.

    Lifecycle:
        ``start()`` → 봇 start 시 한 번. WS task + polling supervisor task 가동.
        ``stop()``  → 봇 stop 시.
    """

    def __init__(
        self,
        bot: BotIctInstance,
        *,
        ws_url: str,
        polling_interval_sec: float,
        reconnect_max: int,
    ) -> None:
        self.bot = bot
        self._ws_url = ws_url
        self._polling_interval = polling_interval_sec
        self._reconnect_max = reconnect_max
        self.mode: WatchMode = WatchMode.STOPPED
        self._stop_event = asyncio.Event()
        self._ws_task: asyncio.Task[None] | None = None
        self._polling_task: asyncio.Task[None] | None = None
        # 이전 tick 가격 — "밖 → 안" 전환 감지용.
        self._last_price: float | None = None
        # 발동 직렬화 — handle_htf_flip 동시 진입 방지.
        self._flip_lock = asyncio.Lock()
        # 이미 flip 발동한 target ts_ms 재발동 방지.
        self._fired_target_ts: set[int] = set()

    # ----- lifecycle ----------------------------------------------------
    async def start(self) -> None:
        if self.mode is not WatchMode.STOPPED:
            return
        self._stop_event.clear()
        self.mode = WatchMode.WS_ACTIVE
        self._ws_task = asyncio.create_task(self._run_ws())
        logger.info("FlipWatcher 시작 — WS=%s", self._ws_url)

    async def stop(self) -> None:
        self.mode = WatchMode.STOPPED
        self._stop_event.set()
        for t in (self._ws_task, self._polling_task):
            if t is not None and not t.done():
                t.cancel()
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass
        self._ws_task = None
        self._polling_task = None
        logger.info("FlipWatcher 정지")

    # ----- WS / polling task --------------------------------------------
    async def _run_ws(self) -> None:
        """WS subscribe — 끊기면 polling 으로 스위치, 복구되면 polling stop."""
        try:
            await subscribe_ticker(
                self.bot.symbol,
                self._on_tick,
                ws_url=self._ws_url,
                reconnect_max=self._reconnect_max,
                on_fallback=self._activate_polling,
                stop_event=self._stop_event,
            )
            # subscribe_ticker 가 fallback 후 return — polling 이미 가동 중.
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("FlipWatcher WS task 예외: %s — polling 으로 스위치", e)
            await self._activate_polling()

    async def _activate_polling(self) -> None:
        """WS 포기 시 호출 — polling task 가동 (이미 가동 중이면 skip)."""
        if self._polling_task is not None and not self._polling_task.done():
            return
        self.mode = WatchMode.POLLING_ACTIVE
        self._polling_task = asyncio.create_task(self._run_polling())
        # WS 재시도 도 백그라운드 유지 — 복구되면 polling 중단.
        if self._ws_task is None or self._ws_task.done():
            self._ws_task = asyncio.create_task(self._retry_ws_until_alive())

    async def _retry_ws_until_alive(self) -> None:
        """polling 중에도 WS 재연결 시도 — 성공 시 polling 중단."""
        while not self._stop_event.is_set():
            await asyncio.sleep(10.0)
            if self._stop_event.is_set():
                return
            try:
                # 짧은 1회 connect 시도. 성공해서 첫 tick 들어오면 mode 가 WS_ACTIVE 로.
                await subscribe_ticker(
                    self.bot.symbol,
                    self._on_tick_ws_recovered,
                    ws_url=self._ws_url,
                    reconnect_max=1,
                    on_fallback=None,
                    stop_event=self._stop_event,
                )
                # subscribe_ticker 가 return 했다는 건 다시 끊김 → 다시 루프.
            except Exception:  # noqa: BLE001
                continue

    def _on_tick_ws_recovered(self, price: float, ts_ms: int) -> None:
        """WS 복구 첫 tick — polling 중단 후 일반 tick 처리로 위임."""
        if self.mode is WatchMode.POLLING_ACTIVE:
            logger.info("Bybit WS 복구 — polling 중단")
            self.mode = WatchMode.WS_ACTIVE
            if self._polling_task is not None and not self._polling_task.done():
                self._polling_task.cancel()
        # 일반 처리.
        return self._on_tick(price, ts_ms)  # type: ignore[return-value]

    async def _run_polling(self) -> None:
        """REST polling fallback — 0.2s 간격 fetch_ticker."""
        logger.warning("FlipWatcher polling 모드 시작 — interval=%.2fs", self._polling_interval)
        while not self._stop_event.is_set() and self.mode is WatchMode.POLLING_ACTIVE:
            try:
                price = await self._fetch_rest_price()
                if price is not None:
                    await self._handle_price(price, int(time.time() * 1000))
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("polling fetch 실패: %s", e)
            await asyncio.sleep(self._polling_interval)

    async def _fetch_rest_price(self) -> float | None:
        """REST 가격 조회 — ccxt fetch_ticker 'last' 사용."""
        client = self.bot.client
        fetcher = getattr(client, "fetch_ticker", None)
        if fetcher is None:
            return None
        try:
            t = await fetcher(self.bot.symbol)
        except Exception:  # noqa: BLE001 — 상위에서 로깅.
            raise
        if not isinstance(t, dict):
            return None
        for k in ("last", "close", "markPrice", "mark_price"):
            v = t.get(k)
            if isinstance(v, (int, float)) and v > 0 and not math.isnan(float(v)):
                return float(v)
        return None

    # ----- tick handler --------------------------------------------------
    def _on_tick(self, price: float, ts_ms: int) -> asyncio.Future | None:
        """WS subscribe_ticker callback — async 핸들러를 task 로 띄움."""
        try:
            return asyncio.ensure_future(self._handle_price(price, ts_ms))
        except RuntimeError:
            # 이벤트 루프 종료 중.
            return None

    async def _handle_price(self, price: float, ts_ms: int) -> None:
        """단일 tick 처리 — touch 누적 + flip 판정."""
        if price <= 0 or math.isnan(price) or math.isinf(price):
            return
        prev = self._last_price
        self._last_price = price

        # 1) htf_map 전체 touch 누적 ("밖 → 안" 전환만 +1).
        self._update_touch_counts(prev, price)

        # 2) active_position.htf_flip_target 진입 판정.
        pos = self.bot.active_position
        if pos is None or pos.htf_flip_target is None:
            return
        target = pos.htf_flip_target
        # 약화된 FVG 는 flip 안 함 — target clear.
        if target.touch_count >= self.bot.htf_fvg_max_touch_count:
            logger.info(
                "HTF flip target 약화 (touch=%d) — clear (%s zone=[%.2f,%.2f])",
                target.touch_count, target.tf, target.low, target.high,
            )
            pos.htf_flip_target = None
            return
        if not target.contains(price):
            return
        # 이미 같은 target 으로 발동했으면 skip.
        if target.ts_ms in self._fired_target_ts:
            return

        # 3) 발동 직전 REST 재확인 — WS 이상치 거름.
        async with self._flip_lock:
            # double-check (lock 잡는 동안 상태 변할 수 있음)
            pos = self.bot.active_position
            if pos is None or pos.htf_flip_target is None:
                return
            target = pos.htf_flip_target
            if target.ts_ms in self._fired_target_ts:
                return
            verify_price = price
            try:
                rest_price = await self._fetch_rest_price()
                if rest_price is not None:
                    verify_price = rest_price
            except Exception as e:  # noqa: BLE001
                logger.warning("flip REST 재확인 실패 (%s) — WS tick 가격 사용", e)
            if not target.contains(verify_price):
                logger.info(
                    "flip 발동 직전 REST 재확인 zone 이탈 — WS=%.4f REST=%.4f",
                    price, verify_price,
                )
                return
            self._fired_target_ts.add(target.ts_ms)
            try:
                await self.bot.handle_htf_flip(verify_price, ts_ms, target)
            except Exception as e:  # noqa: BLE001
                logger.exception("handle_htf_flip 실패: %s — 봇 유지", e)

    def _update_touch_counts(self, prev: float | None, curr: float) -> None:
        """htf_map 전체에 대해 "밖 → 안" 전환 시 touch_count += 1.

        prev=None (첫 tick) 이면 어떤 zone 안에 있어도 +1 하지 않음 — 초기 상태 모름.
        """
        if prev is None:
            return
        htf_map = self.bot._htf_fvg_map_cache  # type: ignore[attr-defined]
        if not htf_map:
            return
        for fv in htf_map:
            was_inside = fv.low <= prev <= fv.high
            now_inside = fv.low <= curr <= fv.high
            if (not was_inside) and now_inside:
                fv.touch_count += 1
                logger.debug(
                    "HTF FVG touch — %s zone=[%.2f,%.2f] count=%d (price %.4f→%.4f)",
                    fv.tf, fv.low, fv.high, fv.touch_count, prev, curr,
                )


__all__ = ["FlipWatcher", "WatchMode"]
