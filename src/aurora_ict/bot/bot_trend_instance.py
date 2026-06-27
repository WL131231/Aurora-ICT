"""BotTrendInstance — Cursus(Dual SuperTrend 추세형) 봇 인스턴스.

투트랙 2번째 봇. Origo(``BotIctInstance``, ICT 되돌림 단타)와 별도 라인으로, 같은
거래소 어댑터·매매기록(TradesStore)·복원 패턴을 재사용하되 전략은 ``dual_st``(Dual
SuperTrend 트레일링 중심)를 쓴다. 백테(``dst_trend_bt.py``, 1h 7페어 5년 + walk-forward
검증, 5대 의심 통과)로 확정한 우리원칙 버전.

전략(step):
    - 진입: 마감 1h 봉에서 ST1×2 & ST2×3 둘 다 정렬 돌파(``latest_signal``) → 시장가
      진입, 초기 SL = 트레일 ST(``trail_mult``, 기본 ×6).
    - 보유: 매 step 트레일 ST 를 추세 방향으로만 끌어(롱 max / 숏 min) 거래소 SL 갱신.
      가격이 SL 을 깨면 거래소가 자동 청산 → 다음 ``_sync_position_state`` 에서 qty 0
      감지 → 기록.
    - REVERSE: 반대 신호가 뜨면 시장가 청산(reduce_only) 후 역진입.
    - 고정 TP 없음 — ST 트레일이 손실=타이트/수익=추세 끝까지 → RR 비대칭.

``multi_user_manager`` 가 Origo 봇과 동일 인터페이스(start/stop/_run_loop/step)로 다룬다.
사용자 모델 선택값이 Cursus 면 이 인스턴스를, Origo 면 ``BotIctInstance`` 를 생성.

담당: 지영민 (투트랙 추세형 봇 — Cursus).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from aurora_ict.bot.bot_ict_instance import BotState, ExchangeClientProtocol
from aurora_ict.interfaces.trades_store import TradeEvent, TradeEventType, TradesStore
from aurora_ict.strategy.dual_st import (
    DualSTConfig,
    latest_signal,
    latest_trail_stop,
)
from aurora_ict.strategy.silver_bullet import Direction

logger = logging.getLogger(__name__)

# 모델명 — 매매 기록 [모델] 컬럼 + UI 표시. Origo 의 ORIGO_MODEL_NAME 과 대칭.
CURSUS_MODEL_NAME = "Cursus 1.0"
# 거래소 키 무효(잔고 조회 등) 연속 실패가 이 횟수면 봇 자동 정지(무한 재시도 차단).
_AUTH_FAIL_STOP_THRESHOLD = 5
# 일일 손실 한도 boundary — ICT 정통 일일 경계와 동일하게 NY local 자정 기준(Origo 차용).
_NY_TZ = ZoneInfo("America/New_York")


@dataclass(slots=True)
class _TrendPosition:
    """추세형 활성 포지션 — ST 트레일 스탑 + (옵션) 4분할 TP."""

    direction: Direction
    entry: float
    qty: float            # 현재 잔량(부분 TP 청산되면 감소)
    stop: float           # 현재 트레일 SL (추세 방향으로만 갱신)
    entry_ts_ms: int
    init_qty: float = 0.0  # 진입 시 전체 수량 — 부분 TP 청산 비중(0.25)의 기준
    tp_prices: list[float] = field(default_factory=list)  # 4분할 TP 가격(롱 위/숏 아래)
    tp_filled: list[bool] = field(default_factory=list)   # 각 TP 체결 여부

    # ICT UI(_build_position_payload) 호환 property — 추세형은 stop(트레일)만 쓰고
    # 고정 TP 없음. position API 가 stop_loss/take_profit/setup_ts_ms 를 읽어도 안전.
    @property
    def stop_loss(self) -> float:
        return self.stop

    @property
    def take_profit(self) -> float:
        return 0.0

    @property
    def setup_ts_ms(self) -> int:
        return self.entry_ts_ms


@dataclass(slots=True)
class BotTrendInstance:
    """Cursus 추세형 봇 — Dual SuperTrend 트레일링.

    Attributes:
        client: 거래소 어댑터(Origo 와 동일 ``ExchangeClientProtocol``).
        symbol: 거래 symbol (예 "BTC/USDT:USDT").
        timeframe: 신호 TF — 추세형은 1h(노이즈↓).
        leverage: 레버리지.
        size_pct: 진입 notional 사이즈(시드 대비, notional=equity×size_pct×leverage).
        cfg: DualST 설정(ST1/ST2/trail_mult). 라이브 시작 trail_mult=6.0(보수).
        step_interval_sec: step 폴링 간격. 1h 봉이라 잦게 돌아도 봉 미변경 시 무동작.
        ohlcv_limit: fetch 봉 수(ST/ATR 워밍업 위해 넉넉히).
        trades_data_dir / user_code / run_mode / alert_cb / notify_cb: 기록·알림(주입).
    """

    client: ExchangeClientProtocol
    symbol: str = "BTC/USDT:USDT"
    timeframe: str = "1h"
    leverage: int = 20
    size_pct: float = 0.9
    cfg: DualSTConfig = field(default_factory=DualSTConfig)
    step_interval_sec: int = 60
    ohlcv_limit: int = 300
    heartbeat_interval_sec: int = 300
    daily_loss_limit_pct: float = 0.0
    # 4분할 TP(파트너 6/27 복원 — 승률 우선). 진입가 ±tp_pcts(%) 도달 시 tp_frac 씩
    # 부분 익절, 잔량은 트레일 SL. enable=False 면 트레일 단독(net 우월하나 승률↓).
    enable_split_tp: bool = True
    tp_pcts: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04)
    tp_frac: float = 0.25
    trades_data_dir: Path | None = None
    user_code: str | None = None
    run_mode: str = "live"
    alert_cb: Any = None
    notify_cb: Any = None

    # 상태
    state: BotState = BotState.STOPPED
    active_position: _TrendPosition | None = None
    _task: asyncio.Task[None] | None = field(default=None)
    _trades_store: TradesStore | None = field(default=None)
    _symbol_meta: dict[str, float | None] = field(default_factory=dict)
    _auth_fail_streak: int = field(default=0)
    _sync_failure_streak: int = field(default=0)
    _last_heartbeat_ms: int = field(default=0)
    # 현재 포지션의 SL 이 거래소에 확정 등록됐는지. 진입/입양/복원 시 검증 결과로 세팅.
    # False 면 무방비 가능성 → step 에서 강제 재적용(_apply_protective_sl). 청산 시 리셋.
    _sl_synced: bool = field(default=False)
    # 일일 손실 한도 추적(Origo #SAFETY-1 패턴) — 청산 실현손익 누적, NY 자정 리셋.
    _today_realized_pnl_usdt: float = field(default=0.0)
    _today_date_str: str = field(default="")  # "YYYY-MM-DD" NY local
    _today_start_equity: float = field(default=0.0)
    # ICT(Origo) UI/엔드포인트 호환 stub — 추세형은 미사용하지만, status/judgment/
    # position/daily 등 ICT 전용 API 가 이 속성들을 읽어도 AttributeError(500) 안 나게
    # None/False 로 노출. #CURSUS hotfix 2026-06-25. (분기 조건이 모두 None/False 에서
    # 안전하게 빠지도록 설계됨 — 추세형은 pending/daily-limit 개념 미사용.)
    _pending_entry: Any = field(default=None)
    _daily_profit_hit: bool = field(default=False)
    _daily_limit_hit: bool = field(default=False)
    _htf_fvg_map_cache: Any = field(default=None)
    _last_setup_ts_ms: int = field(default=0)

    # ---- 생명주기 (BotIctInstance 와 동일 인터페이스) ----

    async def start(self) -> None:
        """봇 기동 — 심볼 메타·레버리지 설정, 거래소 포지션 복원, _run_loop task 생성."""
        if self.state is BotState.RUNNING:
            logger.info("BotTrendInstance(Cursus) %s 이미 실행 중", self.symbol)
            return
        if hasattr(self.client, "fetch_symbol_meta"):
            try:
                meta = await self.client.fetch_symbol_meta(self.symbol)
                self._symbol_meta = meta if isinstance(meta, dict) else {}
            except Exception as e:  # noqa: BLE001
                logger.warning("symbol meta 로드 실패 (%s): %s", self.symbol, e)
        try:
            await self.client.set_leverage(self.symbol, self.leverage)
        except Exception as e:  # noqa: BLE001
            logger.warning("set_leverage 실패 (%s): %s", self.symbol, e)
        await self._recover_position_from_exchange()
        # 무포지션이면 startup 고아 주문 청소(활성 포지션 있으면 SL 보존 위해 skip).
        if self.active_position is None:
            try:
                await self.client.cancel_all_orders(self.symbol)
            except Exception as e:  # noqa: BLE001
                logger.warning("startup 주문 청소 실패 (무시): %s", e)
        self.state = BotState.RUNNING
        self._task = asyncio.create_task(self._run_loop())
        logger.info("BotTrendInstance(Cursus) %s 시작 (trail x%.1f)", self.symbol, self.cfg.trail_mult)

    async def stop(self) -> None:
        """봇 정지 — _run_loop task cancel."""
        self.state = BotState.STOPPED
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("BotTrendInstance(Cursus) %s 정지", self.symbol)

    async def _run_loop(self) -> None:
        """step_interval_sec 마다 step() 호출 — BotIctInstance 패턴(에러 흡수+자동정지+heartbeat)."""
        while self.state is BotState.RUNNING:
            try:
                await self.step()
                self._auth_fail_streak = 0
                _bal_streak = getattr(self.client, "auth_fail_streak", 0)
                if _bal_streak >= _AUTH_FAIL_STOP_THRESHOLD:
                    logger.warning(
                        "%s 거래소 키 무효 %d회 — Cursus 봇 자동 정지. 키 재등록 필요.",
                        self.symbol, _bal_streak,
                    )
                    self.state = BotState.STOPPED
                    break
            except Exception as e:  # noqa: BLE001 — step 실패가 loop 를 죽이지 않게
                logger.exception("Cursus step 실패: %s", e)
            if self.heartbeat_interval_sec > 0:
                now_ms = int(time.time() * 1000)
                if now_ms - self._last_heartbeat_ms >= self.heartbeat_interval_sec * 1000:
                    logger.info(
                        "Cursus heartbeat — symbol=%s active_pos=%s",
                        self.symbol, self.active_position is not None,
                    )
                    self._last_heartbeat_ms = now_ms
            await asyncio.sleep(self.step_interval_sec)

    # ---- 핵심 step (DualST 트레일링) ----

    async def step(self) -> None:
        """단일 step — 1h OHLCV fetch → 거래소 동기화 → 트레일 SL 갱신/REVERSE → 진입."""
        df = await self._fetch_ohlcv()
        if df is None or len(df) < self.cfg.atr_period * 3:
            return
        price = float(df["close"].iloc[-1])
        sig = latest_signal(df, self.cfg)         # Direction | None
        trail = latest_trail_stop(df, self.cfg)   # float (NaN 가능)

        # 거래소 동기화 — active 인데 거래소 qty 0 이면 트레일 SL 체결로 청산됨.
        await self._sync_position_state(price)
        # 봇 모르는 거래소 포지션(재시작 사이·수동·force 진입) 입양 — active None 인데
        # 거래소에 포지션이 있으면 _recover 가 복원. 미인식(active_pos=False) 방지.
        # 트레일 stop 은 복원 후 다음 step 부터 latest_trail_stop 으로 갱신된다.
        if self.active_position is None:
            await self._recover_position_from_exchange(record=False)
            # 입양 직후 거래소 SL 즉시 박기(무방비 방지) — _recover 가 SL 못 읽을 수(sl=0).
            _ap = self.active_position
            if _ap is not None and not _isnan(trail):
                _ap.stop = self._liq_capped_sl(trail, _ap.entry, _ap.direction)
                await self._apply_protective_sl(_ap.stop, price)

        pos = self.active_position
        if pos is not None:
            # 4분할 TP 부분 익절(트레일/REVERSE 전) — 전량 익절되면 active 해제.
            if pos.tp_prices:
                await self._check_split_tp(price)
                if self.active_position is None:
                    return
                pos = self.active_position
            if not _isnan(trail):
                # 트레일 돌파 청산 — 가격이 트레일 스탑을 넘었으면(롱: price<=trail /
                # 숏: price>=trail) 추세 반전 = 청산 시점. SL 을 박아도 즉시청산가라
                # 거래소가 거부하므로 봇이 직접 reduce_only 청산(정상 트레일 청산 기록).
                breached = (
                    (pos.direction is Direction.LONG and price <= trail)
                    or (pos.direction is Direction.SHORT and price >= trail)
                )
                if breached:
                    await self._trail_exit(price)
                    return
                # 트레일 SL 갱신 — 추세 방향으로만(롱 max / 숏 min).
                new_stop = max(pos.stop, trail) if pos.direction is Direction.LONG \
                    else min(pos.stop, trail)
                new_stop = self._liq_capped_sl(new_stop, pos.entry, pos.direction)
                if not self._sl_synced:
                    # 복원/재시작 직후 SL 미확정(무방비 가능) — 검증+비상청산 경로로 확실히.
                    pos.stop = new_stop
                    await self._apply_protective_sl(new_stop, price)
                elif new_stop != pos.stop:
                    # 이미 SL 있음 — 갱신만(실패해도 기존 SL 유지, 다음 step 재시도).
                    pos.stop = new_stop
                    if not await self._set_sl_once(new_stop):
                        self._sl_synced = False  # 다음 step 강제 재적용
                        logger.warning(
                            "[%s] Cursus 트레일 SL 갱신 실패 — 다음 step 재적용", self.symbol,
                        )
            # 비상청산으로 포지션이 닫혔으면 이후 로직 skip.
            if self.active_position is None:
                return
            # REVERSE — 반대 신호.
            if sig is not None and sig is not pos.direction and not _isnan(trail):
                await self._reverse(sig, price, trail)
            return

        # 무포지션 — 진입 신호.
        if sig is not None and not _isnan(trail):
            # 일일 손실 한도 — 도달 시 신규 진입 차단(기존 포지션 트레일/청산은 유지).
            equity = await self._fetch_equity()
            self._maybe_reset_daily_pnl(equity)
            if self._is_daily_loss_limit_hit():
                if not self._daily_limit_hit:
                    logger.warning(
                        "[%s] Cursus 일일 손실 한도 %.1f%% 도달 — 신규 진입 차단",
                        self.symbol, self.daily_loss_limit_pct,
                    )
                    self._daily_limit_hit = True
                return
            await self._open(sig, price, trail)

    async def _open(self, direction: Direction, price: float, trail: float) -> None:
        """시장가 진입 + 초기 트레일 SL 설정 + ENTRY 기록."""
        qty = await self._calc_qty(price)
        if qty <= 0:
            logger.warning("[%s] qty 0 — 진입 skip (잔고/가격 확인)", self.symbol)
            return
        side = "buy" if direction is Direction.LONG else "sell"
        try:
            await self.client.place_order(symbol=self.symbol, side=side, qty=qty, price=None)
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] Cursus 진입 주문 실패: %s", self.symbol, e)
            return
        capped = self._liq_capped_sl(trail, price, direction)
        tp_prices = self._calc_tp_prices(price, direction)
        self.active_position = _TrendPosition(
            direction=direction, entry=price, qty=qty, stop=capped,
            entry_ts_ms=int(time.time() * 1000),
            init_qty=qty, tp_prices=tp_prices, tp_filled=[False] * len(tp_prices),
        )
        self._sl_synced = False
        self._record_trade(
            TradeEventType.ENTRY, direction=direction, price=price, qty=qty,
            reason=f"DualST 진입 trail={trail:.4f}",
        )
        # SL 등록을 검증 — 실패 시 재시도→비상청산(무SL 청산 직행 방지).
        await self._apply_protective_sl(capped, price)
        logger.info(
            "Cursus 진입 — %s %s price=%.4f qty=%.6f SL=%.4f",
            self.symbol, side, price, qty, trail,
        )

    async def _reverse(self, new_dir: Direction, price: float, trail: float) -> None:
        """반대 신호 — 현재 포지션 시장가 청산(reduce_only) 후 역진입."""
        pos = self.active_position
        if pos is None:
            return
        close_side = "sell" if pos.direction is Direction.LONG else "buy"
        try:
            await self.client.place_order(
                symbol=self.symbol, side=close_side, qty=pos.qty, price=None, reduce_only=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] Cursus REVERSE 청산 실패: %s", self.symbol, e)
            return
        self._record_trade(
            TradeEventType.FLIP_CLOSE, direction=pos.direction, price=price, qty=pos.qty,
            entry_for_pnl=pos.entry, reason="REVERSE 반대신호 청산",
        )
        self.active_position = None
        self._sl_synced = False
        # 청산 실제 반영 확인 — reduce_only 가 미체결인데 success 응답(false-positive)일 수
        # 있어, 역진입 전 거래소 잔량 0 을 확인. 남아있으면 역진입 보류(다음 step 재평가).
        try:
            ex = await self.client.fetch_position(self.symbol)
            if float((ex or {}).get("contracts", 0) or 0) > 0:
                logger.warning(
                    "[%s] Cursus REVERSE 청산 미반영(잔량 남음) — 역진입 보류", self.symbol,
                )
                return
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[%s] Cursus REVERSE 후 잔량 확인 실패 — 역진입 보류: %s", self.symbol, e,
            )
            return
        await self._open(new_dir, price, trail)

    async def _sync_position_state(self, price: float) -> None:
        """거래소 포지션 동기화 — active 인데 qty 0 이면 트레일 SL 체결(거래소 자동청산) 인지."""
        if self.active_position is None:
            return
        try:
            ex = await self.client.fetch_position(self.symbol)
        except Exception as e:  # noqa: BLE001
            self._sync_failure_streak += 1
            logger.warning(
                "[%s] fetch_position 실패 %d: %s", self.symbol, self._sync_failure_streak, e,
            )
            return
        self._sync_failure_streak = 0
        contracts = float((ex or {}).get("contracts", 0) or 0)
        if contracts <= 0:
            pos = self.active_position
            self._record_trade(
                TradeEventType.SL_HIT, direction=pos.direction, price=price, qty=pos.qty,
                entry_for_pnl=pos.entry, reason="트레일 SL 체결(거래소 자동청산)",
            )
            logger.info("Cursus 청산 감지 — 트레일 SL 체결 (%s)", self.symbol)
            self.active_position = None
            self._sl_synced = False
            return
        # 방향 불일치 감지 — 거래소 실제 방향 vs 봇이 믿는 방향. REVERSE 부분실패·입양
        # 오류 등으로 어긋나면 봇의 SL 이 엉뚱한 방향에 걸려 무방비 → 즉시 비상청산.
        pos = self.active_position
        side_raw = str((ex or {}).get("side", "") or "").lower()
        ex_dir = Direction.LONG if side_raw in ("long", "buy") else Direction.SHORT
        if side_raw and pos is not None and ex_dir is not pos.direction:
            logger.error(
                "[%s] Cursus 방향 불일치 — 봇=%s 거래소=%s. 비상청산.",
                self.symbol, pos.direction.value, ex_dir.value,
            )
            await self._emergency_close("방향 불일치(봇↔거래소) — 무방비 방지", price=price)

    async def _recover_position_from_exchange(self, *, record: bool = True) -> None:
        """봇 시작 시 거래소 활성 포지션 복원 — 재시작 중복 진입 방지.

        record=False 면 RECOVERED 매매기록을 남기지 않는다(step 중 조용한 재입양용).
        배포·재시작마다 step 입양이 RECOVERED 를 도배하던 문제 방지.
        """
        try:
            ex = await self.client.fetch_position(self.symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] Cursus 복원 fetch_position 실패: %s", self.symbol, e)
            return
        contracts = float((ex or {}).get("contracts", 0) or 0)
        if contracts <= 0:
            return
        side_raw = str((ex or {}).get("side", "") or "").lower()
        direction = Direction.LONG if side_raw in ("long", "buy") else Direction.SHORT
        entry = float((ex or {}).get("entryPrice") or (ex or {}).get("entry_price") or 0)
        sl = float((ex or {}).get("stopLossPrice") or (ex or {}).get("stop_loss") or 0)
        self.active_position = _TrendPosition(
            direction=direction, entry=entry, qty=contracts,
            stop=sl if sl > 0 else entry, entry_ts_ms=int(time.time() * 1000),
        )
        # 거래소에 SL 이 이미 있으면(sl>0) 확정, 없으면 무방비 → 다음 step 강제 적용.
        self._sl_synced = sl > 0
        if record:
            self._record_trade(
                TradeEventType.RECOVERED, direction=direction, price=entry, qty=contracts,
                reason="Cursus 봇 재시작 포지션 복원",
            )
        logger.info(
            "Cursus 복원 — %s %s entry=%.4f qty=%.6f sl=%.4f",
            self.symbol, direction.value, entry, contracts, sl,
        )

    # ---- 4분할 TP (파트너 6/27 복원 — 승률 우선) ----

    def _calc_tp_prices(self, entry: float, direction: Direction) -> list[float]:
        """진입가 기준 4분할 TP 가격 — 롱은 위(+%), 숏은 아래(-%). off 면 빈 리스트."""
        if not self.enable_split_tp or entry <= 0:
            return []
        if direction is Direction.LONG:
            return [entry * (1 + p) for p in self.tp_pcts]
        return [entry * (1 - p) for p in self.tp_pcts]

    async def _check_split_tp(self, price: float) -> None:
        """현재가가 미체결 TP 도달 시 reduce_only 시장가로 init_qty×tp_frac 부분 익절.

        SL 은 거래소 등록(트레일)이 보호하고, TP 만 봇이 모니터링(봇 다운 시 TP 미체결
        가능하나 SL 은 보존). 전량(모든 TP 체결 or 잔량 0) 청산되면 active 해제.

        Args:
            price: 현재가(최근 봉 종가).
        """
        pos = self.active_position
        if pos is None or not pos.tp_prices:
            return
        close_side = "sell" if pos.direction is Direction.LONG else "buy"
        for k, tp_px in enumerate(pos.tp_prices):
            if pos.tp_filled[k]:
                continue
            reached = (
                (pos.direction is Direction.LONG and price >= tp_px)
                or (pos.direction is Direction.SHORT and price <= tp_px)
            )
            if not reached:
                continue
            qty_part = min(pos.init_qty * self.tp_frac, pos.qty)
            if hasattr(self.client, "round_amount"):
                try:
                    qty_part = float(self.client.round_amount(self.symbol, qty_part))
                except Exception:  # noqa: BLE001
                    pass
            if qty_part <= 0:
                # 최소 수량 미달(소액 포지션 25%) — 분할 포기, 트레일이 청산.
                pos.tp_filled[k] = True
                continue
            try:
                await self.client.place_order(
                    symbol=self.symbol, side=close_side, qty=qty_part,
                    price=None, reduce_only=True,
                )
            except Exception as e:  # noqa: BLE001
                # 거부(최소수량 등)는 무한 재시도 도배 방지 위해 해당 TP 포기 —
                # 트레일 SL 이 잔량을 청산하므로 손실 위험은 없음.
                logger.error(
                    "[%s] Cursus 분할 TP%d 청산 실패(포기): %s", self.symbol, k + 1, e,
                )
                pos.tp_filled[k] = True
                continue
            pos.tp_filled[k] = True
            pos.qty = max(0.0, pos.qty - qty_part)
            self._record_trade(
                TradeEventType.TP_HIT, direction=pos.direction, price=tp_px,
                qty=qty_part, entry_for_pnl=pos.entry,
                reason=f"4분할 TP{k + 1}({self.tp_pcts[k] * 100:.0f}%)",
            )
            logger.info(
                "Cursus 분할 TP%d 체결 — %s @ %.4f qty=%.6f 잔량=%.6f",
                k + 1, self.symbol, tp_px, qty_part, pos.qty,
            )
        # 전량 익절(모든 TP 체결 or 잔량 소진)이면 포지션 종료.
        if pos.qty <= 1e-9 or all(pos.tp_filled):
            self.active_position = None
            self._sl_synced = False

    # ---- SL 안전망 (고배율 청산 방지 최우선) ----

    async def _set_sl_once(self, stop: float) -> bool:
        """거래소에 SL 1회 등록 + 성공 검증.

        어댑터는 SL 미적용 시 예외 대신 falsy(빈 dict)를 돌려줄 수 있으므로
        반환값을 진위로 검사한다. 예외도 실패(False).

        Args:
            stop: 등록할 SL 가격.

        Returns:
            True=등록 성공(truthy 응답). False=실패.
        """
        try:
            res = await self.client.set_position_tpsl(self.symbol, stop_loss=stop)
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] Cursus SL 설정 예외: %s", self.symbol, e)
            return False
        return bool(res)

    async def _apply_protective_sl(self, stop: float, price: float | None = None) -> bool:
        """SL 등록 + 검증 + 실패 시 1회 재시도 + 그래도 실패면 비상청산.

        고배율 봇 최우선 원칙(청산 방지) — 진입/입양/복원 직후 SL 이 거래소에
        확실히 박혔는지 보장한다. 2회 실패면 무방비 상태이므로 즉시 시장가 청산.

        Args:
            stop: 등록할 SL 가격.
            price: 현재가(비상청산 시 PnL 기록용). None 이면 entry 로 기록.

        Returns:
            True=SL 보장. False=2회 실패로 비상청산함(active_position=None).
        """
        for attempt in (1, 2):
            if await self._set_sl_once(stop):
                self._sl_synced = True
                return True
            logger.warning(
                "[%s] Cursus SL 미적용(시도 %d/2) — 재시도", self.symbol, attempt,
            )
        logger.error("[%s] Cursus SL 2회 적용 실패 — 무방비 방지 비상청산", self.symbol)
        await self._emergency_close("SL 적용 실패 — 무방비 방지 비상청산", price=price)
        return False

    async def _emergency_close(self, reason: str, price: float | None = None) -> None:
        """현재 포지션 reduce_only 시장가 즉시 청산 — 무방비/방향불일치 등 위험 상황.

        Args:
            reason: 청산 사유(기록·로그용).
            price: 현재가(PnL 기록용). None 이면 entry(=PnL 0)로 기록.
        """
        pos = self.active_position
        if pos is None:
            return
        close_px = price if (price is not None and price > 0) else pos.entry
        close_side = "sell" if pos.direction is Direction.LONG else "buy"
        try:
            await self.client.place_order(
                symbol=self.symbol, side=close_side, qty=pos.qty,
                price=None, reduce_only=True,
            )
            self._record_trade(
                TradeEventType.SYNC_CLOSE, direction=pos.direction, price=close_px,
                qty=pos.qty, entry_for_pnl=pos.entry, reason=reason,
            )
            logger.error("[%s] Cursus 비상청산 실행 — %s", self.symbol, reason)
        except Exception as e:  # noqa: BLE001
            logger.error(
                "[%s] Cursus 비상청산 실패(수동 확인 필요): %s", self.symbol, e,
            )
        self.active_position = None
        self._sl_synced = False

    async def _trail_exit(self, price: float) -> None:
        """가격이 트레일 스탑을 깬 정상 청산 — reduce_only 시장가 + SL_HIT 기록.

        SuperTrend 트레일이 가격 반대편으로 넘어간 상태(추세 반전)는 청산 시점이다.
        이때 거래소에 SL 을 박으면 즉시청산가라 거부(retCode 10001)되므로, 봇이
        직접 reduce_only 로 청산한다. 비상청산과 달리 정상 트레일 청산으로 기록.

        Args:
            price: 현재가(청산가 추정·PnL 기록용).
        """
        pos = self.active_position
        if pos is None:
            return
        close_side = "sell" if pos.direction is Direction.LONG else "buy"
        try:
            await self.client.place_order(
                symbol=self.symbol, side=close_side, qty=pos.qty,
                price=None, reduce_only=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] Cursus 트레일 청산 실패: %s", self.symbol, e)
            return
        self._record_trade(
            TradeEventType.SL_HIT, direction=pos.direction, price=price, qty=pos.qty,
            entry_for_pnl=pos.entry, reason="트레일 스탑 돌파 청산",
        )
        logger.info("Cursus 트레일 청산 — %s price=%.4f", self.symbol, price)
        self.active_position = None
        self._sl_synced = False

    # ---- 보조 ----

    async def _fetch_ohlcv(self) -> pd.DataFrame | None:
        """1h OHLCV fetch → DataFrame (시간 오름차순)."""
        try:
            rows = await self.client.fetch_ohlcv(self.symbol, self.timeframe, self.ohlcv_limit)
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] fetch_ohlcv 실패: %s", self.symbol, e)
            return None
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        return df

    async def _calc_qty(self, price: float) -> float:
        """notional 기반 수량 — qty = equity×size_pct×leverage / price (lot step 라운딩)."""
        equity = await self._fetch_equity()
        if equity <= 0 or price <= 0:
            return 0.0
        qty = equity * self.size_pct * self.leverage / price
        if hasattr(self.client, "round_amount"):
            try:
                qty = float(self.client.round_amount(self.symbol, qty))
            except Exception:  # noqa: BLE001
                pass
        return qty

    async def _fetch_equity(self) -> float:
        """USDT equity — fetch_balance 응답에서 유연 파싱(여러 ccxt 형태 대응)."""
        try:
            bal = await self.client.fetch_balance()
        except Exception as e:  # noqa: BLE001
            logger.warning("[%s] fetch_balance 실패: %s", self.symbol, e)
            return 0.0
        if not isinstance(bal, dict):
            return 0.0
        # ccxt 표준: bal["USDT"]["total"] / bal["total"]["USDT"] / bal["USDT"]["free"].
        usdt = bal.get("USDT")
        if isinstance(usdt, dict):
            val = usdt.get("total") or usdt.get("free") or 0
            if val:
                return float(val)
        total = bal.get("total")
        if isinstance(total, dict) and total.get("USDT"):
            return float(total["USDT"])
        return 0.0

    def _record_trade(
        self,
        event_type: TradeEventType,
        *,
        direction: Direction,
        price: float,
        qty: float,
        entry_for_pnl: float | None = None,
        reason: str = "",
        pnl_override: float | None = None,
    ) -> None:
        """매매 이벤트 1건 기록 (TradesStore lazy init) — model=Cursus 1.0 태그.

        BotIctInstance._record_trade 패턴. PnL 은 pnl_override(거래소 실현치) 우선,
        없으면 entry/price 기반 추정. trades_data_dir 미주입이면 기록 skip(로그만).
        """
        if self._trades_store is None:
            if self.trades_data_dir is None:
                return
            try:
                self._trades_store = TradesStore(self.trades_data_dir)
            except Exception as e:  # noqa: BLE001
                logger.warning("Cursus TradesStore 초기화 실패 — 기록 skip: %s", e)
                return
        pnl_usdt = pnl_override
        if pnl_usdt is None and entry_for_pnl is not None and qty > 0:
            sign = 1.0 if direction is Direction.LONG else -1.0
            pnl_usdt = sign * (price - entry_for_pnl) * qty
        event = TradeEvent(
            ts_ms=int(time.time() * 1000),
            event_type=event_type,
            symbol=self.symbol,
            direction="long" if direction is Direction.LONG else "short",
            price=price,
            qty=qty,
            pnl_usdt=pnl_usdt,
            reason=reason,
            mode=self.run_mode,
            model=CURSUS_MODEL_NAME,
        )
        # 청산 이벤트(pnl 산출됨)는 일일 누적 손익에 반영 — 손실 한도 판정용.
        # ENTRY 는 entry_for_pnl 없어 pnl None → 누적 제외. (SL_HIT/FLIP_CLOSE/SYNC_CLOSE)
        if pnl_usdt is not None:
            self._today_realized_pnl_usdt += pnl_usdt
        try:
            self._trades_store.record(event)
        except Exception as e:  # noqa: BLE001
            logger.warning("Cursus 매매 기록 실패: %s", e)
        logger.info(
            "매매 %s | %s %s price=%.4f qty=%.6f pnl=%s",
            event.event_type.value.upper(), self.symbol, event.direction, price, qty,
            f"{pnl_usdt:+.2f}" if pnl_usdt is not None else "—",
        )

    # ---- ICT(Origo) UI/엔드포인트 호환 메서드 stub (#CURSUS hotfix 2026-06-25) ----
    # status/daily_loss_limit/pending 등 ICT 전용 API 가 봇 메서드를 호출해도 500 안
    # 나게. 추세형은 일일 한도·pending limit entry 개념 미사용 → 비활성/빈 응답.

    def _maybe_reset_daily_pnl(self, equity_now: float | None) -> None:
        """NY local 자정 기준 일일 누적 손익 리셋 (Origo #SAFETY-1 패턴).

        새 거래일이 시작되면 ``_today_realized_pnl_usdt`` 를 0 으로,
        ``_today_start_equity`` 를 현재 가용자산으로 갱신한다. equity_now 가 None
        (잔고 조회 실패)이면 리셋을 보류해 baseline 오염을 막는다.

        Args:
            equity_now: 현재 가용 자산(USDT). 새 날짜 baseline.
        """
        ny_date = datetime.now(UTC).astimezone(_NY_TZ).strftime("%Y-%m-%d")
        if ny_date == self._today_date_str:
            return
        if equity_now is None:
            logger.warning(
                "[%s] Cursus daily PnL reset 보류 — 잔고 조회 실패 (NY %s)",
                self.symbol, ny_date,
            )
            return
        self._today_date_str = ny_date
        self._today_realized_pnl_usdt = 0.0
        self._today_start_equity = equity_now if equity_now > 0 else 0.0
        self._daily_limit_hit = False
        logger.info(
            "[%s] Cursus daily reset (NY %s) start_equity=%.2f",
            self.symbol, ny_date, self._today_start_equity,
        )

    def daily_loss_status(self) -> dict[str, Any]:
        """일일 손실 현황 — UI/엔드포인트용. 추세형은 수익 한도 미사용(추세 끝까지)."""
        start = self._today_start_equity
        pnl = self._today_realized_pnl_usdt
        pct = (-pnl / start * 100.0) if start > 0 else 0.0
        return {
            "limit_pct": self.daily_loss_limit_pct, "today_pnl_usdt": pnl,
            "today_pct": pct, "start_equity": start,
            "hit": self._is_daily_loss_limit_hit(),
            "date_ny": self._today_date_str, "profit_limit_pct": 0.0,
            "profit_hit": False,
        }

    def _is_daily_profit_limit_hit(self) -> bool:
        # 추세형은 수익 한도 미사용 — 추세를 끝까지 먹는 게 원칙(RR 비대칭).
        return False

    def _is_daily_loss_limit_hit(self) -> bool:
        """일일 손실 한도 초과 여부 — True 면 신규 진입 차단.

        ``daily_loss_limit_pct <= 0``(비활성) 또는 시작 equity 미정이면 False.
        """
        if self.daily_loss_limit_pct <= 0 or self._today_start_equity <= 0:
            return False
        loss_pct = -self._today_realized_pnl_usdt / self._today_start_equity * 100.0
        return loss_pct >= self.daily_loss_limit_pct

    async def cancel_pending_entry(self) -> bool:
        """ICT UI 호환 — 추세형은 pending limit entry 없음."""
        return False

    def _liq_capped_sl(self, stop: float, entry: float, direction: Direction) -> float:
        """SL 을 레버리지 청산가 안으로 캡 — 강제청산 전 SL 보장(#LIQ-CAP, Origo 차용).

        레버 높을수록 청산가가 entry 에 가까워, ST 트레일 SL(ATR×6)이 청산가 밖이면
        SL 도달 전 강제청산된다. SL 을 청산거리 80% 안으로 당겨 봇 SL 이 먼저 작동.
        """
        if self.leverage <= 0 or entry <= 0:
            return stop
        liq_dist = entry / self.leverage * 0.8
        if direction is Direction.LONG:
            return max(stop, entry - liq_dist)
        return min(stop, entry + liq_dist)

    def ensure_prefetch_started(self) -> None:
        """ICT UI 호환 — 추세형은 OHLCV prefetch 캐시 미사용(차트는 get_ohlcv_cached
        직접 fetch). ensure_bot_ready 가 호출하므로 no-op stub 필수(없으면 차트 경로 깨짐)."""
        return None

    async def get_ohlcv_cached(self, timeframe: str, limit: int) -> list[list[Any]]:
        """차트용 OHLCV — ICT UI(get_ohlcv_mu) 호환. 캐시 없이 거래소 직접 fetch.

        BotIctInstance 는 prefetch 캐시를 쓰지만 추세형은 캐시 미운영 — UI 차트
        표시를 위해 거래소에서 즉시 조회(실패 시 빈 리스트로 차트 비움).
        """
        try:
            return await self.client.fetch_ohlcv(self.symbol, timeframe, limit)
        except Exception:  # noqa: BLE001
            return []


def _isnan(x: float) -> bool:
    """NaN 안전 체크 (float('nan') != 자기자신)."""
    return x != x
