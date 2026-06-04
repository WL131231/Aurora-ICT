"""BotManager — 단일 BotIctInstance의 lifecycle + 모드 toggle 관리.

담당 책임:
- BotIctInstance 생성/소유 (또는 인스턴스 외부 주입)
- start / stop / 모드 전환 (demo ↔ live)
- 상태 스냅샷 제공 (UI / API 호출용)

모드 전환 흐름:
- 기존 인스턴스 정지 → settings 갱신 → 새 client로 새 BotIctInstance 생성

단일 인스턴스 전제 (멀티 심볼/유저는 상위 계층에서 처리).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from aurora_ict.bot.bot_ict_instance import (
    BotIctInstance,
    BotState,
    ExchangeClientProtocol,
)
from aurora_ict.config.settings import IctSettings, RunMode, get_settings

logger = logging.getLogger(__name__)


# Client factory — settings를 받아 client를 만들어 반환하는 async callable.
ClientFactory = Callable[[IctSettings], Awaitable[ExchangeClientProtocol]]


@dataclass(slots=True)
class BotStatus:
    """봇 상태 스냅샷 (UI/API 응답용)."""

    state: BotState
    run_mode: RunMode
    enabled: bool
    symbol: str
    has_credentials: bool
    has_active_position: bool
    last_setup_ts_ms: int


@dataclass(slots=True)
class BotManager:
    """단일 BotIctInstance의 lifecycle + 모드 전환을 관리.

    Attributes:
        client_factory: settings로부터 ExchangeClient를 만드는 async factory.
        settings: 사용할 설정. ``None``이면 ``get_settings()`` 사용.
    """

    client_factory: ClientFactory
    settings: IctSettings = field(default_factory=get_settings)
    _bot: BotIctInstance | None = field(default=None)
    _client: ExchangeClientProtocol | None = field(default=None)
    _start_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def start(self) -> BotStatus:
        """봇 기동. ``enabled=False``이거나 API 키가 없으면 ValueError.

        동시 호출 (UI 더블 클릭 / launcher race) 시 두 번째 호출이 새 client/bot 인스턴스
        생성해서 중복 실행 발생 — asyncio.Lock + state 재검사로 차단.
        """
        async with self._start_lock:
            if not self.settings.enabled:
                raise ValueError("봇이 disabled 상태 — settings.enabled=False")
            if not self.settings.has_credentials():
                raise ValueError(
                    f"API 키 없음 (run_mode={self.settings.run_mode.value})",
                )
            if self._bot is not None and self._bot.state is BotState.RUNNING:
                logger.info("이미 실행 중 — re-start 무시")
                return self.status()
            return await self._do_start()

    async def _do_start(self) -> BotStatus:
        # 이전 client 가 남아있으면 명시적으로 close — ccxt aiohttp session 누수 방지.
        await self._close_client()
        client = await self.client_factory(self.settings)
        self._client = client
        self._bot = BotIctInstance(
            client=client,
            symbol=self.settings.symbol,
            timeframe=self.settings.timeframe,
            leverage=self.settings.leverage,
            position_pct_base=self.settings.position_pct_base,
            position_pct_max=self.settings.position_pct_max,
            position_pct_step=self.settings.position_pct_step,
            min_rr=self.settings.min_rr,
            min_confluence=self.settings.min_confluence,
            high_rr_bypass_min_rr=self.settings.high_rr_bypass_min_rr,
            fvg_min_size_pct=self.settings.fvg_min_size_pct,
            step_interval_sec=self.settings.step_interval_sec,
            ohlcv_limit=self.settings.ohlcv_limit,
            # v0.4.30 이후 진입 완화 옵션 — settings 에서 명시 주입.
            setup_stale_bars=self.settings.setup_stale_bars,
            disable_time_filter=self.settings.disable_time_filter,
            # multi-TF (ICT 정통 HTF setup + LTF confirm) 옵션.
            multi_tf=self.settings.multi_tf,
            multi_tf_ltf_lookback=self.settings.multi_tf_ltf_lookback,
            enable_trail=self.settings.enable_trail,
            trail_buffer_ratio=self.settings.trail_buffer_ratio,
            use_market_entry=self.settings.use_market_entry,
            entry_limit_ttl_sec=self.settings.entry_limit_ttl_sec,
            min_sl_distance_pct=self.settings.min_sl_distance_pct,
            max_sl_distance_pct=self.settings.max_sl_distance_pct,
            max_entry_distance_pct=self.settings.max_entry_distance_pct,
            heartbeat_interval_sec=self.settings.heartbeat_interval_sec,
            htf_ltf_conflict_guard_ratio=self.settings.htf_ltf_conflict_guard_ratio,
            htf_ema_bias_enabled=self.settings.htf_ema_bias_enabled,
            htf_ema_bias_tf=self.settings.htf_ema_bias_tf,
            htf_ema_bias_period=self.settings.htf_ema_bias_period,
            # HTF FVG override (변형 7).
            htf_override_mode=self.settings.htf_override_mode,
            htf_fvg_tfs=self.settings.htf_fvg_tfs,
            # #SAFETY-1: 일일 손실 한도.
            daily_loss_limit_pct=self.settings.daily_loss_limit_pct,
        )
        # 거래소 측 leverage 를 settings 에 맞춤 — qty 계산 일치 보장.
        # 실패해도 봇 시작 자체는 진행. 다만 실패 시 실제 leverage 조회해서
        # bot.leverage 보정 (#LEV-1) — size 계산이 실제 거래소 값과 일치하게.
        # set_leverage 반환값으로 성공 판정 — aurora_adapter 가 실패 시 빈 dict
        # 또는 {"retCode": 110043} (이미 같은 값) 반환 (exception 안 던짐).
        # 따라서 try/except 만으로 판정 불가 → 반환 dict 검사.
        leverage_synced = False
        try:
            result = await client.set_leverage(
                self.settings.symbol, self.settings.leverage,
            )
            if isinstance(result, dict) and (
                result.get("alreadySet") is True
                or int(result.get("retCode", -1)) == 0
            ):
                leverage_synced = True
        except Exception as e:  # noqa: BLE001
            logger.warning("set_leverage 호출 자체 실패: %s", e)
        if not leverage_synced:
            actual = await client.fetch_actual_leverage(self.settings.symbol)
            if actual is not None and actual != self._bot.leverage:
                logger.warning(
                    "leverage mismatch — 의도 %dx, 거래소 실제 %dx. "
                    "size 계산을 거래소 값으로 보정 (수동 변경 권장).",
                    self._bot.leverage, actual,
                )
                self._bot.leverage = actual
        await self._bot.start()
        logger.info(
            "BotManager started — mode=%s symbol=%s leverage=%dx",
            self.settings.run_mode.value, self.settings.symbol, self.settings.leverage,
        )
        return self.status()

    async def stop(self) -> BotStatus:
        """봇 정지 + client close (aiohttp session 누수 방지)."""
        if self._bot is not None:
            await self._bot.stop()
        await self._close_client()
        return self.status()

    async def _close_client(self) -> None:
        """현재 client 의 내부 ccxt exchange (_ex) 를 명시적으로 close.

        ccxt async exchange 의 aiohttp ClientSession 이 GC 까지 살아있으면
        "Unclosed client session" WARNING 발생 → 명시 close 박아 해결.
        """
        if self._client is None:
            return
        client = self._client
        self._client = None
        # AuroraClientAdapter._client = Aurora CcxtClient. CcxtClient._ex = ccxt async exchange.
        inner = getattr(client, "_client", client)
        ex = getattr(inner, "_ex", None)
        if ex is None:
            return
        try:
            await ex.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("ccxt exchange close 실패 (무시): %s", e)

    async def set_run_mode(self, mode: RunMode) -> BotStatus:
        """모드 전환 — 실행 중이면 stop → settings 변경 → restart."""
        was_running = self._bot is not None and self._bot.state is BotState.RUNNING
        if was_running:
            await self.stop()
        self.settings.run_mode = mode
        logger.info("run_mode 변경 → %s", mode.value)
        if was_running:
            await self.start()
        return self.status()

    async def set_enabled(self, enabled: bool) -> BotStatus:
        """``enabled`` 토글. False면 stop, True면 start (필요 시)."""
        was_running = self._bot is not None and self._bot.state is BotState.RUNNING
        self.settings.enabled = enabled
        if not enabled and was_running:
            await self.stop()
        elif enabled and not was_running:
            await self.start()
        return self.status()

    def status(self) -> BotStatus:
        """현재 상태 스냅샷."""
        if self._bot is None:
            return BotStatus(
                state=BotState.STOPPED,
                run_mode=self.settings.run_mode,
                enabled=self.settings.enabled,
                symbol=self.settings.symbol,
                has_credentials=self.settings.has_credentials(),
                has_active_position=False,
                last_setup_ts_ms=0,
            )
        return BotStatus(
            state=self._bot.state,
            run_mode=self.settings.run_mode,
            enabled=self.settings.enabled,
            symbol=self._bot.symbol,
            has_credentials=self.settings.has_credentials(),
            has_active_position=self._bot.active_position is not None,
            last_setup_ts_ms=self._bot._last_setup_ts_ms,
        )

    @property
    def bot(self) -> BotIctInstance | None:
        """현재 BotIctInstance 노출 (debug/내부 접근용)."""
        return self._bot


def _make_status_dict(status: BotStatus) -> dict[str, Any]:
    """BotStatus → dict (UI/API 직렬화용)."""
    return {
        "state": status.state.value,
        "run_mode": status.run_mode.value,
        "enabled": status.enabled,
        "symbol": status.symbol,
        "has_credentials": status.has_credentials,
        "has_active_position": status.has_active_position,
        "last_setup_ts_ms": status.last_setup_ts_ms,
    }


__all__ = [
    "BotManager",
    "BotStatus",
    "ClientFactory",
]
