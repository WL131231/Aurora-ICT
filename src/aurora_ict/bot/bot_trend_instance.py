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
from pathlib import Path
from typing import Any

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


@dataclass(slots=True)
class _TrendPosition:
    """추세형 활성 포지션 — ST 트레일 스탑 기반(고정 TP 없음)."""

    direction: Direction
    entry: float
    qty: float
    stop: float           # 현재 트레일 SL (추세 방향으로만 갱신)
    entry_ts_ms: int


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

        pos = self.active_position
        if pos is not None:
            # 트레일 SL 갱신 — 추세 방향으로만(롱 max / 숏 min).
            if not _isnan(trail):
                new_stop = max(pos.stop, trail) if pos.direction is Direction.LONG \
                    else min(pos.stop, trail)
                if new_stop != pos.stop:
                    pos.stop = new_stop
                    try:
                        await self.client.set_position_tpsl(self.symbol, stop_loss=new_stop)
                    except Exception as e:  # noqa: BLE001
                        logger.warning("[%s] 트레일 SL 갱신 실패: %s", self.symbol, e)
            # REVERSE — 반대 신호.
            if sig is not None and sig is not pos.direction and not _isnan(trail):
                await self._reverse(sig, price, trail)
            return

        # 무포지션 — 진입 신호.
        if sig is not None and not _isnan(trail):
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
        self.active_position = _TrendPosition(
            direction=direction, entry=price, qty=qty, stop=trail,
            entry_ts_ms=int(time.time() * 1000),
        )
        try:
            await self.client.set_position_tpsl(self.symbol, stop_loss=trail)
        except Exception as e:  # noqa: BLE001
            logger.error("[%s] Cursus 진입 후 SL 설정 실패: %s", self.symbol, e)
        self._record_trade(
            TradeEventType.ENTRY, direction=direction, price=price, qty=qty,
            reason=f"DualST 진입 trail={trail:.4f}",
        )
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

    async def _recover_position_from_exchange(self) -> None:
        """봇 시작 시 거래소 활성 포지션 복원 — 재시작 중복 진입 방지."""
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
        self._record_trade(
            TradeEventType.RECOVERED, direction=direction, price=entry, qty=contracts,
            reason="Cursus 봇 재시작 포지션 복원",
        )
        logger.info(
            "Cursus 복원 — %s %s entry=%.4f qty=%.6f sl=%.4f",
            self.symbol, direction.value, entry, contracts, sl,
        )

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

    def daily_loss_status(self) -> dict[str, Any]:
        """ICT UI 호환 — 추세형은 일일 손실/수익 한도 미사용(비활성)."""
        return {
            "limit_pct": self.daily_loss_limit_pct, "today_pnl_usdt": 0.0,
            "today_pct": 0.0, "start_equity": 0.0, "hit": False,
            "date_ny": "", "profit_limit_pct": 0.0, "profit_hit": False,
        }

    def _is_daily_profit_limit_hit(self) -> bool:
        return False

    def _is_daily_loss_limit_hit(self) -> bool:
        return False

    async def cancel_pending_entry(self) -> bool:
        """ICT UI 호환 — 추세형은 pending limit entry 없음."""
        return False

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
