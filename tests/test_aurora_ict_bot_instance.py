"""BotIctInstance 박은 거 박힘 — Aurora-ICT v0.1.5."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance, BotState
from aurora_ict.signal.ict_signal import SignalAction
from aurora_ict.strategy.silver_bullet import Direction

NY = ZoneInfo("America/New_York")


def _bars_long_setup() -> list[tuple[float, float, float, float]]:
    """박힌 long setup 박힌 거 박힘 박힌 bars (NY 10:00 박힘 박힘 박힘)."""
    return [
        (100, 105, 99, 104),
        (104, 130, 103, 125),
        (125, 124, 100, 101),
        (101, 108, 95, 96),
        (96, 110, 95.5, 109),
        (109, 112, 92, 93),
        (93, 100, 92.5, 99),
        (99, 105, 98, 104),
        (104, 106, 100, 101),
        (101, 105, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 119, 108, 118),
        (118, 122, 115, 121),
    ]


def _ohlcv_rows(start_ny: datetime, bars: list[tuple[float, float, float, float]]) -> list[list[Any]]:
    """ccxt-style OHLCV rows: [ts_ms, o, h, l, c, v]."""
    rows = []
    for i, (o, h, l, c) in enumerate(bars):
        t = start_ny + timedelta(minutes=i)
        ts_ms = int(t.timestamp() * 1000)
        rows.append([ts_ms, o, h, l, c, 100.0])
    return rows


def _mock_client(ohlcv_rows: list[list[Any]]) -> AsyncMock:
    """Mock ExchangeClient 박힌 거."""
    client = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=ohlcv_rows)
    client.place_order = AsyncMock(return_value={"orderId": "TEST123"})
    client.fetch_position = AsyncMock(return_value=None)
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    return client


@pytest.mark.asyncio
async def test_step_no_signal_returns_no_action() -> None:
    """짧은 OHLCV → NO_ACTION."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = BotIctInstance(client=client)
    sig = await bot.step()
    assert sig.action is SignalAction.NO_ACTION
    client.place_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_step_executes_long_setup() -> None:
    """valid long setup 박힘 박힘 → place_order 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(
        client=client,
        symbol="BTCUSDT",
        min_rr=1.0,
        fvg_min_size_pct=0.001,
    )
    sig = await bot.step()
    assert sig.action is SignalAction.ENTER_LONG
    client.place_order.assert_awaited_once()
    call_kwargs = client.place_order.call_args.kwargs
    assert call_kwargs["symbol"] == "BTCUSDT"
    assert call_kwargs["side"] == "buy"
    assert call_kwargs["qty"] > 0
    assert call_kwargs["price"] > 0
    assert call_kwargs["stop_loss"] < call_kwargs["price"]
    assert call_kwargs["take_profit"] > call_kwargs["price"]
    # active position 박힘 박힘
    assert bot.active_position is not None
    assert bot.active_position.direction is Direction.LONG


@pytest.mark.asyncio
async def test_step_skip_when_position_exists() -> None:
    """active position 박힘 → place_order 박힘 X (박힘 박힘 fetch_position 박힘)."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    # 박힌 position 박힘 박힘 (active 박힘)
    client.fetch_position = AsyncMock(return_value={"contracts": 0.01})
    bot = BotIctInstance(
        client=client,
        min_rr=1.0,
        fvg_min_size_pct=0.001,
    )
    # active_position 박힘 박힘 박힘 박힘 박힘 박힘 박힘
    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    bot.active_position = _ActivePosition(
        direction=Direction.LONG,
        entry=100,
        stop_loss=95,
        take_profit=110,
        qty=0.01,
        setup_ts_ms=0,
    )
    await bot.step()
    # 박힘 박힙 place_order 박힘 X
    client.place_order.assert_not_awaited()
    # fetch_position 박힘 박힘 박힘
    client.fetch_position.assert_awaited()


@pytest.mark.asyncio
async def test_step_resets_position_on_close() -> None:
    """fetch_position 박힘 None / contracts=0 → active_position 박힘 None 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    # position closed (contracts=0)
    client.fetch_position = AsyncMock(return_value={"contracts": 0})
    bot = BotIctInstance(client=client, min_rr=1.0, fvg_min_size_pct=0.001)
    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    bot.active_position = _ActivePosition(
        direction=Direction.LONG,
        entry=100,
        stop_loss=95,
        take_profit=110,
        qty=0.01,
        setup_ts_ms=0,
    )
    await bot.step()
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_step_duplicate_setup_filtered() -> None:
    """같은 setup ts_ms 박힘 박힘 두 번째 박힘 박힘 박힘 X (중복 진입 박힘)."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(client=client, min_rr=1.0, fvg_min_size_pct=0.001)
    # 첫 step
    await bot.step()
    assert bot.active_position is not None
    # 박은 position 박힌 거 reset 박힘 → 박은 setup 박힘 박힘 박힘 박힘 박힙 박힘
    bot.active_position = None
    # 두 번째 step — 박은 OHLCV
    await bot.step()
    # 박은 setup_ts_ms 박힘 박힘 박힘 — place_order 박힘 박힘 1번
    assert client.place_order.await_count == 1


@pytest.mark.asyncio
async def test_start_stop_lifecycle() -> None:
    """start → state RUNNING, stop → state STOPPED."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    # step_interval 박힘 박힘 박힘 박힘 박힘 (test 박힘 박힘)
    bot = BotIctInstance(client=client, step_interval_sec=3600)
    assert bot.state is BotState.STOPPED
    await bot.start()
    assert bot.state is BotState.RUNNING
    await bot.stop()
    assert bot.state is BotState.STOPPED
    assert bot._task is None


@pytest.mark.asyncio
async def test_qty_calc_with_equity() -> None:
    """equity 박힌 거 박힙 박힘 qty 박힘 박힙 박힘 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(client=client, min_rr=1.0, fvg_min_size_pct=0.001)
    sig = await bot.step()
    assert sig.setup is not None
    # equity 박힌 거 박힘 박힘 qty 박힘 박힙 박힘 박힘 박힘 박힘
    qty_low = bot._calc_qty(sig.setup, equity=1000.0)
    qty_high = bot._calc_qty(sig.setup, equity=10000.0)
    assert qty_high > qty_low
    assert qty_low > 0


@pytest.mark.asyncio
async def test_fetch_equity_from_ccxt_balance() -> None:
    """ccxt format {'USDT': {'total': N}} 박힘 박힘 N 박힘 박힘."""
    client = _mock_client([])
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 5000.0}})
    bot = BotIctInstance(client=client)
    eq = await bot._fetch_equity()
    assert eq == 5000.0


@pytest.mark.asyncio
async def test_fetch_equity_fallback_on_error() -> None:
    """fetch_balance 박힙 박힙 박힘 fallback 1000."""
    client = _mock_client([])
    client.fetch_balance = AsyncMock(side_effect=RuntimeError("network"))
    bot = BotIctInstance(client=client)
    eq = await bot._fetch_equity()
    assert eq == 1000.0
