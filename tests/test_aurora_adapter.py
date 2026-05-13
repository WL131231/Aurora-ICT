"""AuroraClientAdapter — Aurora-ICT v0.1.7."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from aurora_ict.bot.aurora_adapter import AuroraClientAdapter


@pytest.mark.asyncio
async def test_fetch_ohlcv_timeframe_uppercase_for_hour_and_above() -> None:
    """Aurora client 는 1h+ 대문자만 인식 — adapter 가 자동 변환."""
    df = pd.DataFrame(
        [{"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}],
        index=pd.to_datetime(["2026-05-12 00:00"], utc=True),
    )
    inner = AsyncMock()
    inner.fetch_ohlcv = AsyncMock(return_value=df)
    adapter = AuroraClientAdapter(inner)

    # 소문자로 호출 → Aurora 에는 대문자로 전달
    for ui_tf, aurora_tf in [
        ("1h", "1H"), ("2h", "2H"), ("4h", "4H"),
        ("1d", "1D"), ("1w", "1W"),
    ]:
        await adapter.fetch_ohlcv("BTC/USDT:USDT", ui_tf, 10)
        called_args = inner.fetch_ohlcv.call_args
        assert called_args.args[1] == aurora_tf, (
            f"{ui_tf} 호출 시 Aurora 에 {aurora_tf} 전달되어야 함 "
            f"(받은 값: {called_args.args[1]})"
        )


@pytest.mark.asyncio
async def test_fetch_ohlcv_timeframe_minute_unchanged() -> None:
    """1m / 5m / 15m 은 소문자 그대로 전달."""
    df = pd.DataFrame(
        [{"open": 1, "high": 1, "low": 1, "close": 1, "volume": 1}],
        index=pd.to_datetime(["2026-05-12 00:00"], utc=True),
    )
    inner = AsyncMock()
    inner.fetch_ohlcv = AsyncMock(return_value=df)
    adapter = AuroraClientAdapter(inner)

    for tf in ("1m", "5m", "15m", "30m"):
        await adapter.fetch_ohlcv("BTC/USDT:USDT", tf, 10)
        assert inner.fetch_ohlcv.call_args.args[1] == tf


@pytest.mark.asyncio
async def test_fetch_ohlcv_dataframe_to_rows() -> None:
    """Aurora DataFrame 박힌 거 → ccxt raw rows."""
    df = pd.DataFrame([
        {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 50},
        {"open": 101, "high": 103, "low": 100, "close": 102, "volume": 60},
    ])
    df.index = pd.to_datetime(["2026-05-12 00:00", "2026-05-12 00:01"], utc=True)

    inner = AsyncMock()
    inner.fetch_ohlcv = AsyncMock(return_value=df)
    adapter = AuroraClientAdapter(inner)

    rows = await adapter.fetch_ohlcv("BTC/USDT:USDT", "1m", 100)
    assert len(rows) == 2
    assert rows[0][1] == 100.0  # open
    assert rows[0][2] == 102.0  # high
    assert rows[0][5] == 50.0   # volume


@pytest.mark.asyncio
async def test_fetch_ohlcv_empty_df() -> None:
    """빈 DataFrame → 빈 list."""
    inner = AsyncMock()
    inner.fetch_ohlcv = AsyncMock(return_value=pd.DataFrame())
    adapter = AuroraClientAdapter(inner)
    rows = await adapter.fetch_ohlcv("BTC/USDT:USDT", "1m", 100)
    assert rows == []


@pytest.mark.asyncio
async def test_place_order_dataclass_to_dict() -> None:
    """Aurora Order dataclass → dict 박힘."""

    @dataclass
    class FakeOrder:
        symbol: str
        side: str
        qty: float
        order_id: str

    inner = AsyncMock()
    inner.place_order = AsyncMock(return_value=FakeOrder(
        symbol="BTCUSDT", side="buy", qty=0.01, order_id="X1",
    ))
    adapter = AuroraClientAdapter(inner)
    result = await adapter.place_order(
        symbol="BTCUSDT", side="buy", qty=0.01, price=50000,
        stop_loss=49000, take_profit=52000,
    )
    assert result["symbol"] == "BTCUSDT"
    assert result["qty"] == 0.01
    assert result["order_id"] == "X1"


@pytest.mark.asyncio
async def test_fetch_position_dataclass_to_dict() -> None:
    """Aurora Position dataclass → dict + qty→contracts alias."""

    @dataclass
    class FakePosition:
        symbol: str
        side: str
        qty: float
        entry_price: float

    inner = AsyncMock()
    inner.fetch_position = AsyncMock(return_value=FakePosition(
        symbol="BTCUSDT", side="long", qty=0.05, entry_price=50000,
    ))
    adapter = AuroraClientAdapter(inner)
    pos = await adapter.fetch_position("BTCUSDT")
    assert pos is not None
    assert pos["symbol"] == "BTCUSDT"
    assert pos["qty"] == 0.05
    assert pos["contracts"] == 0.05  # ccxt alias 박힘


@pytest.mark.asyncio
async def test_fetch_position_none() -> None:
    """Position None → None."""
    inner = AsyncMock()
    inner.fetch_position = AsyncMock(return_value=None)
    adapter = AuroraClientAdapter(inner)
    pos = await adapter.fetch_position("BTCUSDT")
    assert pos is None


@pytest.mark.asyncio
async def test_fetch_balance_via_ex() -> None:
    """Aurora client._ex.fetch_balance() 박힘 박힘 박힘."""
    ex = AsyncMock()
    ex.fetch_balance = AsyncMock(return_value={"USDT": {"total": 5000.0}})
    inner = MagicMock()
    inner._ex = ex
    adapter = AuroraClientAdapter(inner)
    bal = await adapter.fetch_balance()
    assert bal == {"USDT": {"total": 5000.0}}


@pytest.mark.asyncio
async def test_fetch_balance_no_ex_returns_empty() -> None:
    """``_ex`` 박힘 X → 빈 dict."""
    inner = MagicMock(spec=[])  # no attrs
    adapter = AuroraClientAdapter(inner)
    bal = await adapter.fetch_balance()
    assert bal == {}


@pytest.mark.asyncio
async def test_fetch_balance_exception_returns_empty() -> None:
    """fetch_balance 박힘 박힘 박힙 → 빈 dict."""
    ex = AsyncMock()
    ex.fetch_balance = AsyncMock(side_effect=RuntimeError("network"))
    inner = MagicMock()
    inner._ex = ex
    adapter = AuroraClientAdapter(inner)
    bal = await adapter.fetch_balance()
    assert bal == {}


@pytest.mark.asyncio
async def test_adapter_can_be_used_with_bot_instance() -> None:
    """Adapter 박힌 거 박힘 BotIctInstance 박힘 박힘 박힘 박힘 박힘."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance

    df = pd.DataFrame([
        {"open": 100, "high": 102, "low": 99, "close": 101, "volume": 50},
    ])
    df.index = pd.to_datetime(["2026-05-12 00:00"], utc=True)

    inner = MagicMock()
    inner.fetch_ohlcv = AsyncMock(return_value=df)
    inner.fetch_position = AsyncMock(return_value=None)
    inner.place_order = AsyncMock(return_value={"orderId": "X"})
    inner._ex = MagicMock()
    inner._ex.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})

    adapter = AuroraClientAdapter(inner)
    bot = BotIctInstance(client=adapter)
    sig = await bot.step()
    # df 박힌 거 박힘 박힙 1개 박힙 → NO_ACTION
    assert sig.action.value == "no_action"


def _unused_ref() -> Any:
    """Make Any used."""
    return None
