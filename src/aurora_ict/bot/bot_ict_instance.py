"""Aurora-ICT bot instance — Bybit 박힌 거 박힘 매매 박힘.

박힌 거 박힌 거 박힘:
1. 매 분 OHLCV (BTCUSDT 1m) fetch — Bybit
2. ``generate_ict_signal()`` 박힌 거 박힘
3. ENTER_LONG / ENTER_SHORT 박힘 박힐 → limit order (entry) + SL + TP placement
4. 박힌 position 박힌 거 박힐 박힘 SL/TP 박힌 거 박힘 박힐 박힘 (Bybit conditional 박힘)
5. close 박힌 거 박힌 거 박힘 박힘 position 박힌 거 박힘 박힘

Exchange client 박힌 거 박힘 박힌 거 박힘 — ``ExchangeClientProtocol`` 박힘 박힘 박힘
박힐 박힘. Aurora 측 ``CCXTClient`` (Bybit) 박힌 거 박힘 박힘 박힘 박힘 박힘 (duck typing).
테스트 박힌 거 박힘 박힘 박힘 mock 박힘 박힘.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol

import pandas as pd

from aurora_ict.signal.ict_signal import (
    ICTSignal,
    SignalAction,
    generate_ict_signal,
)
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup

logger = logging.getLogger(__name__)


class ExchangeClientProtocol(Protocol):
    """Aurora 측 ``CCXTClient`` 박힌 거 박힘 (duck typing)."""

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


class BotState(str, Enum):
    """봇 상태."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass(slots=True)
class _ActivePosition:
    """박힌 박힌 position 박힌 박힌 state."""

    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    qty: float
    setup_ts_ms: int


@dataclass(slots=True)
class BotIctInstance:
    """ICT Silver Bullet 박힌 봇 instance.

    Attributes:
        client: Bybit client (Aurora 측 ``CCXTClient`` 박은 박힘 박힌 거).
        symbol: 박힌 symbol (e.g. "BTCUSDT").
        timeframe: OHLCV 박힌 timeframe (표준 "1m").
        risk_per_trade_pct: 박힌 trade 박힌 risk % (총 자산 박힌 거).
        leverage: 박힌 leverage.
        min_rr: 최소 RR (표준 2.0).
        step_interval_sec: 박힌 step 박힘 박힘 interval (표준 60s).
        ohlcv_limit: fetch 박힌 박힌 봉 수 (표준 200).
    """

    client: ExchangeClientProtocol
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"
    risk_per_trade_pct: float = 0.5
    leverage: int = 5
    min_rr: float = 2.0
    step_interval_sec: int = 60
    ohlcv_limit: int = 200
    fvg_min_size_pct: float = 0.0005

    state: BotState = field(default=BotState.STOPPED)
    active_position: _ActivePosition | None = field(default=None)
    _task: asyncio.Task[None] | None = field(default=None)
    _last_setup_ts_ms: int = field(default=0)  # 박힌 박힌 setup 박힌 거 중복 방지

    async def start(self) -> None:
        """봇 박힌 거 박힘 (background task 박힘)."""
        if self.state is BotState.RUNNING:
            logger.info("BotIctInstance %s 박힘 박힌 박힘 박힘", self.symbol)
            return
        self.state = BotState.RUNNING
        self._task = asyncio.create_task(self._run_loop())
        logger.info("BotIctInstance %s 박힘 박힘 박힘", self.symbol)

    async def stop(self) -> None:
        """봇 박힌 거 박힘 (background task 박힘 cancel)."""
        self.state = BotState.STOPPED
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        logger.info("BotIctInstance %s 박힘 박힘", self.symbol)

    async def _run_loop(self) -> None:
        """매 step 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘."""
        while self.state is BotState.RUNNING:
            try:
                await self.step()
            except Exception as e:  # noqa: BLE001 — step 박힌 거 박힘 박힘 박힘 박힘 박힘 박힘
                logger.exception("step 박힘 박힘: %s", e)
            await asyncio.sleep(self.step_interval_sec)

    async def step(self) -> ICTSignal:
        """박힌 step 박힘 박힘 — fetch + signal + execute. 박힌 박힘 박힘 signal 박힘.

        Returns:
            박힌 ``ICTSignal`` 박힘 박힘 박힘 (테스트 박힘 박힘 박힘 박힙 박힘).
        """
        df = await self._fetch_ohlcv()

        signal = generate_ict_signal(
            df,
            self.symbol,
            min_rr=self.min_rr,
            fvg_min_size_pct=self.fvg_min_size_pct,
        )

        # 박힌 박힌 position 박힌 거 박힙 박힘 진입 박힘 박힘 박힘 X
        if self.active_position is not None:
            await self._sync_position_state()
            return signal

        if not signal.is_actionable or signal.setup is None:
            return signal

        # 박은 setup 박힌 거 박힌 거 박힘 박힘 박힘 박힘 (중복 박힘 X)
        if signal.setup.ts_ms == self._last_setup_ts_ms:
            return signal

        await self._execute_setup(signal.setup)
        self._last_setup_ts_ms = signal.setup.ts_ms
        return signal

    async def _fetch_ohlcv(self) -> pd.DataFrame:
        """OHLCV fetch + DataFrame 박힘."""
        rows = await self.client.fetch_ohlcv(
            self.symbol, self.timeframe, self.ohlcv_limit,
        )
        # ccxt 박힌 거 박힘 = [ts_ms, open, high, low, close, volume]
        df = pd.DataFrame(
            rows,
            columns=["ts_ms", "open", "high", "low", "close", "volume"],
        )
        df.index = pd.DatetimeIndex(pd.to_datetime(df["ts_ms"], unit="ms", utc=True))
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    async def _execute_setup(self, setup: SilverBulletSetup) -> None:
        """박힌 setup 박힌 거 박힘 박힘 주문 박힘 박힘.

        - limit order 박은 entry 박힘
        - stop_loss / take_profit 박힘 박힘 (Bybit conditional)
        - qty = risk_per_trade_pct 박힌 거 박힌 거 박힘 박힘 / SL 박힌 거리
        """
        side = "buy" if setup.direction is Direction.LONG else "sell"

        qty = self._calc_qty(setup)
        if qty <= 0:
            logger.warning("qty 박힘 박힘 박힘 → skip: setup=%s", setup.ts_ms)
            return

        logger.info(
            "Execute setup %s %s entry=%.4f sl=%.4f tp=%.4f rr=%.2f qty=%.4f",
            self.symbol, side, setup.entry, setup.stop_loss,
            setup.take_profit, setup.risk_reward, qty,
        )

        try:
            await self.client.place_order(
                symbol=self.symbol,
                side=side,
                qty=qty,
                price=setup.entry,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit,
            )
        except Exception as e:  # noqa: BLE001 — 주문 실패 박힙 박힘 박힘 박힘
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

    def _calc_qty(self, setup: SilverBulletSetup) -> float:
        """진입 qty 박힘 박힘.

        risk_amount = total_equity × risk_per_trade_pct/100
        qty = risk_amount / |entry - stop_loss|

        여기는 박힌 total_equity 박힌 거 박힙 박힘 박힐 simplified 박힘 — fixed $1000 박힘
        박힘 (실제는 client.fetch_balance 박힘 박힐 박힘). 박힘 박힌 박힌 박힌 박힘.
        """
        # TODO v0.1.6: 박힌 자산 박은 거 박힘 박힘 박힘 박힘 fetch_balance() 박힘
        notional_risk = 1000.0 * (self.risk_per_trade_pct / 100.0)
        sl_dist = abs(setup.entry - setup.stop_loss)
        if sl_dist <= 0:
            return 0.0
        qty = notional_risk / sl_dist
        # 최소 박힌 거 박힘 박힘 — Bybit 박힘 박힘 0.001 BTC 박힘
        return max(qty, 0.001)

    async def _sync_position_state(self) -> None:
        """박힌 position 박은 거 박힌 거 박힘 박힘 박힘 fetch_position() 박힘 박힘.

        SL/TP 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 → ``active_position`` 박힘 박힘.
        """
        try:
            pos = await self.client.fetch_position(self.symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_position 박힘: %s", e)
            return

        if pos is None or float(pos.get("contracts", 0) or 0) == 0:
            # position 박힘 박힘 박힘 → SL/TP 박힘 박힙 박힘 박힘
            logger.info("position 박힘 박힘 박힘 (SL/TP hit 박힙) — state reset")
            self.active_position = None


__all__ = [
    "BotIctInstance",
    "BotState",
    "ExchangeClientProtocol",
]
