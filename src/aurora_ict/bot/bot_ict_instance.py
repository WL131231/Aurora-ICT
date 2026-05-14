"""Aurora-ICT bot instance — Bybit 대상 실매매 실행기.

처리 흐름:
1. 매 분 OHLCV (BTCUSDT 1m) fetch — Bybit
2. ``generate_ict_signal()`` 호출
3. ENTER_LONG / ENTER_SHORT 신호 시 → limit order (entry) + SL + TP placement
4. 진입 후 position 추적, SL/TP 트리거는 거래소가 처리 (Bybit conditional)
5. 청산이 감지되면 내부 position state 리셋

Exchange client는 외부 주입 — ``ExchangeClientProtocol``을 만족하면 됨.
Aurora 측 ``CCXTClient`` (Bybit)를 어댑터로 감싸 사용 (duck typing).
테스트에서는 mock으로 교체.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import pandas as pd

from aurora_ict.indicators.daily_bias import compute_daily_bias
from aurora_ict.indicators.structure import TrendDirection
from aurora_ict.signal.ict_signal import (
    ICTSignal,
    generate_ict_signal,
)
from aurora_ict.strategy.multi_tf_bias import (
    combine_bias,
    compute_bias_from_df,
    htf_pair,
)
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup

logger = logging.getLogger(__name__)


class ExchangeClientProtocol(Protocol):
    """Aurora 측 ``CCXTClient``와 호환되는 duck-typed 인터페이스."""

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int,
    ) -> list[list[Any]]: ...

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float | None = None,
        reduce_only: bool = False,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]: ...

    async def fetch_position(self, symbol: str) -> dict[str, Any] | None: ...

    async def fetch_balance(self) -> dict[str, Any]: ...


class BotState(StrEnum):
    """봇 상태."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass(slots=True)
class _ActivePosition:
    """현재 진입 중인 position state."""

    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    qty: float
    setup_ts_ms: int


@dataclass(slots=True)
class BotIctInstance:
    """ICT Silver Bullet 봇 instance.

    Attributes:
        client: Bybit client (Aurora 측 ``CCXTClient``를 어댑터로 감싼 것).
        symbol: 거래 symbol (e.g. "BTCUSDT").
        timeframe: OHLCV timeframe (표준 "1m").
        risk_per_trade_pct: 트레이드당 risk % (총 자산 기준).
        leverage: 사용 leverage.
        min_rr: 최소 RR (표준 2.0).
        step_interval_sec: step 호출 간격 (표준 60s).
        ohlcv_limit: fetch 봉 수 (표준 200).
    """

    client: ExchangeClientProtocol
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"
    # ICT 정통: risk 1% per trade, min_rr 1:3.
    risk_per_trade_pct: float = 1.0
    leverage: int = 5
    min_rr: float = 3.0
    step_interval_sec: int = 60
    ohlcv_limit: int = 200
    fvg_min_size_pct: float = 0.0005
    # FVG 이후 N 봉 안에 retest 없으면 진입 skip.
    setup_stale_bars: int = 5
    # LuxAlgo SB Strict mode — True 면 FVG mean threshold 까지 retrace 된 setup 만 진입.
    require_retrace: bool = False

    # Partial TP 분배 — TP1/TP2/TP3 비율 (합 = 1.0). ICT 정통 50/25/25.
    tp_distribution: tuple[float, float, float] = (0.5, 0.25, 0.25)

    state: BotState = field(default=BotState.STOPPED)
    active_position: _ActivePosition | None = field(default=None)
    _task: asyncio.Task[None] | None = field(default=None)
    _last_setup_ts_ms: int = field(default=0)  # 동일 setup 중복 진입 방지
    # HTF 봉 캐시 — (tf, last_ts_ms) → DataFrame. 같은 봉이면 재fetch 안 함.
    _htf_cache: dict[str, tuple[int, pd.DataFrame]] = field(default_factory=dict)

    async def start(self) -> None:
        """봇 기동 (background task 생성)."""
        if self.state is BotState.RUNNING:
            logger.info("BotIctInstance %s 이미 실행 중", self.symbol)
            return
        self.state = BotState.RUNNING
        self._task = asyncio.create_task(self._run_loop())
        logger.info("BotIctInstance %s 시작", self.symbol)

    async def stop(self) -> None:
        """봇 정지 (background task cancel)."""
        self.state = BotState.STOPPED
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("BotIctInstance %s 정지", self.symbol)

    async def _run_loop(self) -> None:
        """매 step_interval_sec 마다 step()을 호출하는 메인 루프."""
        while self.state is BotState.RUNNING:
            try:
                await self.step()
            except Exception as e:  # noqa: BLE001 — step 실패가 loop 전체를 죽이지 않도록
                logger.exception("step 실패: %s", e)
            await asyncio.sleep(self.step_interval_sec)

    async def step(self) -> ICTSignal:
        """단일 step — fetch + HTF/Daily bias + signal + execute. 생성된 signal 반환.

        Returns:
            계산된 ``ICTSignal`` (테스트/디버그 용도로 노출).
        """
        df = await self._fetch_ohlcv()
        htf_bias = await self._compute_htf_bias()
        daily_bias = await self._compute_daily_bias(df)
        bias = self._combine_with_daily(htf_bias, daily_bias)

        signal = generate_ict_signal(
            df,
            self.symbol,
            bias=bias,
            min_rr=self.min_rr,
            fvg_min_size_pct=self.fvg_min_size_pct,
            stale_bars=self.setup_stale_bars,
            require_retrace=self.require_retrace,
        )

        # 진입 중인 position이 있으면 신규 진입은 막고 상태만 동기화
        if self.active_position is not None:
            await self._sync_position_state()
            return signal

        if not signal.is_actionable or signal.setup is None:
            return signal

        # 동일 setup으로 재진입 방지 (중복 주문 X)
        if signal.setup.ts_ms == self._last_setup_ts_ms:
            return signal

        await self._execute_setup(signal.setup)
        self._last_setup_ts_ms = signal.setup.ts_ms
        return signal

    async def _fetch_ohlcv(self) -> pd.DataFrame:
        """OHLCV fetch + DataFrame 변환 (Trade TF)."""
        return await self._fetch_ohlcv_tf(self.timeframe, self.ohlcv_limit)

    async def _fetch_ohlcv_tf(self, tf: str, limit: int) -> pd.DataFrame:
        """임의 timeframe OHLCV fetch + DataFrame 변환."""
        rows = await self.client.fetch_ohlcv(self.symbol, tf, limit)
        df = pd.DataFrame(
            rows,
            columns=["ts_ms", "open", "high", "low", "close", "volume"],
        )
        df.index = pd.DatetimeIndex(pd.to_datetime(df["ts_ms"], unit="ms", utc=True))
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    async def _compute_htf_bias(self) -> TrendDirection | None:
        """Trade TF 의 상위 HTF1/HTF2 봉을 fetch 해 bias 산출.

        HTF 봉은 LTF 보다 훨씬 느리게 갱신되므로 마지막 봉 ts 기준 캐시 사용.

        Returns:
            - ``TrendDirection`` (UP/DOWN/NONE) — HTF 매핑이 있을 때.
            - ``None`` — Trade TF 가 HTF 매핑이 없을 때 (1m, 3m 등 또는 1w).
              이 경우 silver_bullet 이 같은 df 의 structure 로 자동 추정.
        """
        htf1_tf, htf2_tf = htf_pair(self.timeframe)
        if htf1_tf is None and htf2_tf is None:
            return None  # 매핑 없음 — 자동 추정 위임
        bias1 = await self._htf_bias_for_tf(htf1_tf) if htf1_tf else TrendDirection.NONE
        bias2 = await self._htf_bias_for_tf(htf2_tf) if htf2_tf else TrendDirection.NONE
        return combine_bias(bias1, bias2)

    async def _compute_daily_bias(self, ltf_df: pd.DataFrame) -> TrendDirection:
        """Daily 봉 fetch + 전일 H/L vs 현재가 비교로 daily bias 산출.

        Args:
            ltf_df: Trade TF DataFrame (마지막 close 가 현재가).

        Returns:
            - current > PDH → UP
            - current < PDL → DOWN
            - 범위 안 또는 일봉 부족 → NONE
        """
        if len(ltf_df) == 0:
            return TrendDirection.NONE
        # 일봉 20개로 PDH/PDL + 지난 주 월요일 (최대 13일 전) 까지 안전 커버.
        daily_df = await self._fetch_ohlcv_tf("1d", 20)
        if len(daily_df) < 2:
            return TrendDirection.NONE
        current_close = float(ltf_df["close"].iloc[-1])
        return compute_daily_bias(daily_df, current_close)

    @staticmethod
    def _combine_with_daily(
        htf: TrendDirection | None,
        daily: TrendDirection,
    ) -> TrendDirection | None:
        """HTF bias + Daily bias 결합.

        - HTF None (매핑 없음) → silver_bullet 자동 추정 위임 (None 그대로)
        - 둘 다 같은 방향 → 그 방향
        - 한쪽 NONE → 다른쪽 따름
        - 충돌 → NONE (진입 차단)
        """
        if htf is None:
            return None
        if htf is TrendDirection.NONE:
            return daily
        if daily is TrendDirection.NONE:
            return htf
        if htf == daily:
            return htf
        return TrendDirection.NONE

    async def _htf_bias_for_tf(self, tf: str) -> TrendDirection:
        """단일 HTF bias — 캐시된 봉 재사용, 새 봉 생겼을 때만 재fetch."""
        # 마지막 봉 ts 만 빠르게 보기 위해 limit=1 prefetch — 캐시 hit 시 skip
        # 단순화: 매 step 마다 limit=200 fetch (HTF 1d 라도 1m 1번이라 부담 작음)
        df = await self._fetch_ohlcv_tf(tf, 200)
        if len(df) == 0:
            return TrendDirection.NONE
        last_ts = int(df.index[-1].value // 10**6)
        cached = self._htf_cache.get(tf)
        if cached is not None and cached[0] == last_ts:
            df = cached[1]
        else:
            self._htf_cache[tf] = (last_ts, df)
        return compute_bias_from_df(df)

    async def _execute_setup(self, setup: SilverBulletSetup) -> None:
        """setup 한 건을 실제 주문으로 실행 (ICT 정통 partial TP).

        - entry limit order 등록 (SL 만, TP X — 별도 partial 처리)
        - tp1/tp2/tp3 각각 reduce_only limit (Bybit V5 multiple reduce-only 허용)
        - qty 는 tp_distribution 비율로 분배
        """
        side = "buy" if setup.direction is Direction.LONG else "sell"
        exit_side = "sell" if setup.direction is Direction.LONG else "buy"

        equity = await self._fetch_equity()
        qty = self._calc_qty(setup, equity)
        if qty <= 0:
            logger.warning("qty 계산 결과 0 이하 → skip: setup=%s", setup.ts_ms)
            return

        # tp 분배 — 마지막은 나머지로 보정 (rounding 손실 방지)
        pct1, pct2, _pct3 = self.tp_distribution
        qty1 = qty * pct1
        qty2 = qty * pct2
        qty3 = qty - qty1 - qty2

        logger.info(
            "Execute setup %s %s entry=%.4f sl=%.4f "
            "tp1=%.4f tp2=%.4f tp3=%.4f qty=%.4f (%.2f/%.2f/%.2f)",
            self.symbol, side, setup.entry, setup.stop_loss,
            setup.tp1, setup.tp2, setup.tp3, qty, qty1, qty2, qty3,
        )

        try:
            # Entry + SL — TP 는 partial 로 따로 등록
            await self.client.place_order(
                symbol=self.symbol,
                side=side,
                qty=qty,
                price=setup.entry,
                stop_loss=setup.stop_loss,
                take_profit=None,
            )
            # Partial TP 3개 (reduce_only limit)
            for tp_price, tp_qty in (
                (setup.tp1, qty1),
                (setup.tp2, qty2),
                (setup.tp3, qty3),
            ):
                if tp_qty <= 0:
                    continue
                await self.client.place_order(
                    symbol=self.symbol,
                    side=exit_side,
                    qty=tp_qty,
                    price=tp_price,
                    reduce_only=True,
                )
        except Exception as e:  # noqa: BLE001 — 주문 실패도 봇은 계속 돌아야 함
            logger.exception("place_order 실패: %s", e)
            return

        self.active_position = _ActivePosition(
            direction=setup.direction,
            entry=setup.entry,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            qty=qty,
            setup_ts_ms=setup.ts_ms,
        )

    async def _fetch_equity(self) -> float:
        """가용 자산 (USDT equity) 조회.

        Bybit V5 ccxt format = ``{"USDT": {"total": ...}, ...}``.
        fallback 1000.0 (테스트/오류 대비).
        """
        try:
            bal = await self.client.fetch_balance()
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_balance 실패: %s — fallback $1000", e)
            return 1000.0
        if not isinstance(bal, dict):
            return 1000.0
        # ccxt 표준 = {"USDT": {"total": float, "free": float, ...}, ...}
        usdt = bal.get("USDT")
        if isinstance(usdt, dict):
            total = usdt.get("total")
            if isinstance(total, (int, float)) and total > 0:
                return float(total)
        # Bybit direct format fallback
        total = bal.get("total")
        if isinstance(total, (int, float)) and total > 0:
            return float(total)
        return 1000.0

    def _calc_qty(self, setup: SilverBulletSetup, equity: float) -> float:
        """진입 qty 계산.

        risk_amount = equity × risk_per_trade_pct/100
        qty = risk_amount / |entry - stop_loss|
        """
        notional_risk = equity * (self.risk_per_trade_pct / 100.0)
        sl_dist = abs(setup.entry - setup.stop_loss)
        if sl_dist <= 0:
            return 0.0
        qty = notional_risk / sl_dist
        # Bybit BTC 최소 주문수량 0.001
        return max(qty, 0.001)

    async def _sync_position_state(self) -> None:
        """거래소 fetch_position()으로 현재 position 상태 동기화.

        SL/TP가 트리거되어 거래소 쪽에서 닫혔으면 ``active_position`` 리셋.
        """
        try:
            pos = await self.client.fetch_position(self.symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_position 실패: %s", e)
            return

        if pos is None or float(pos.get("contracts", 0) or 0) == 0:
            # 거래소 측 position 없음 → SL/TP hit으로 청산된 것으로 간주
            logger.info("position 종료 감지 (SL/TP hit 추정) — state reset")
            self.active_position = None


__all__ = [
    "BotIctInstance",
    "BotState",
    "ExchangeClientProtocol",
]
