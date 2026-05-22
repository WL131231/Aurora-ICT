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
    """Aurora Order dataclass → dict 변환."""

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
    )
    assert result["symbol"] == "BTCUSDT"
    assert result["qty"] == 0.01
    assert result["order_id"] == "X1"


@pytest.mark.asyncio
async def test_place_order_with_sl_tp_passed_to_client() -> None:
    """#LIVE-1 fix: entry + SL/TP → 본체 place_order 에 동봉 전달 (set_trading_stop X).

    Bybit create_order params 의 stopLoss/takeProfit 으로 entry 주문에 동봉되어
    체결 시 포지션에 자동 적용. 별도 set_trading_stop (포지션 API) 호출은 안 함.
    """
    inner = MagicMock()
    inner.place_order = AsyncMock(return_value={"orderId": "X"})
    ex = MagicMock()
    ex.private_post_v5_position_trading_stop = AsyncMock(return_value={"retCode": 0})
    inner._ex = ex
    adapter = AuroraClientAdapter(inner)
    await adapter.place_order(
        symbol="BTC/USDT:USDT", side="buy", qty=0.01,
        stop_loss=79000, take_profit=82000,
    )
    # 별도 set_trading_stop 호출 안 함 (entry 동봉으로 대체)
    ex.private_post_v5_position_trading_stop.assert_not_awaited()
    # 본체 place_order 에 SL/TP 동봉 전달
    kw = inner.place_order.call_args.kwargs
    assert kw["stop_loss"] == 79000
    assert kw["take_profit"] == 82000
    assert kw["reduce_only"] is False


@pytest.mark.asyncio
async def test_place_order_reduce_only_drops_sl_tp() -> None:
    """reduce_only=True (청산) 면 SL/TP 인자 와도 본체엔 None 전달 (동봉 X)."""
    inner = MagicMock()
    inner.place_order = AsyncMock(return_value={"orderId": "X"})
    inner._ex = MagicMock()
    adapter = AuroraClientAdapter(inner)
    await adapter.place_order(
        symbol="BTC/USDT:USDT", side="sell", qty=0.005, price=82000,
        reduce_only=True, stop_loss=79000,
    )
    kw = inner.place_order.call_args.kwargs
    assert kw["reduce_only"] is True
    assert kw["stop_loss"] is None
    assert kw["take_profit"] is None


@pytest.mark.asyncio
async def test_place_order_no_sl_passes_none() -> None:
    """SL/TP 인자 없으면 본체에 None 전달."""
    inner = MagicMock()
    inner.place_order = AsyncMock(return_value={"orderId": "X"})
    inner._ex = MagicMock()
    adapter = AuroraClientAdapter(inner)
    await adapter.place_order(symbol="BTC/USDT:USDT", side="buy", qty=0.01)
    kw = inner.place_order.call_args.kwargs
    assert kw["stop_loss"] is None
    assert kw["take_profit"] is None


@pytest.mark.asyncio
async def test_set_leverage_calls_bybit_api() -> None:
    """set_leverage 가 Bybit V5 set_leverage 호출 + 정확한 params."""
    inner = MagicMock()
    ex = MagicMock()
    ex.private_post_v5_position_set_leverage = AsyncMock(return_value={"retCode": 0})
    inner._ex = ex
    adapter = AuroraClientAdapter(inner)
    await adapter.set_leverage("BTC/USDT:USDT", 20)
    ex.private_post_v5_position_set_leverage.assert_awaited_once()
    params = ex.private_post_v5_position_set_leverage.call_args.args[0]
    assert params["symbol"] == "BTCUSDT"
    assert params["buyLeverage"] == "20"
    assert params["sellLeverage"] == "20"
    assert params["category"] == "linear"


@pytest.mark.asyncio
async def test_set_leverage_already_set_handled() -> None:
    """이미 같은 leverage 박혀있을 때 retCode 110043 박혀와도 OK 처리."""
    inner = MagicMock()
    ex = MagicMock()
    ex.private_post_v5_position_set_leverage = AsyncMock(
        side_effect=RuntimeError("bybit error 110043: leverage not modified"),
    )
    inner._ex = ex
    adapter = AuroraClientAdapter(inner)
    result = await adapter.set_leverage("BTC/USDT:USDT", 20)
    assert result.get("alreadySet") is True


@pytest.mark.asyncio
async def test_set_leverage_no_ex_returns_empty() -> None:
    inner = MagicMock(spec=[])
    adapter = AuroraClientAdapter(inner)
    result = await adapter.set_leverage("BTC/USDT:USDT", 20)
    assert result == {}


@pytest.mark.asyncio
async def test_place_order_returns_dict_on_success() -> None:
    """entry 주문 성공 시 dict (orderId 등) 정상 반환."""
    inner = MagicMock()
    inner.place_order = AsyncMock(return_value={"orderId": "X"})
    inner._ex = MagicMock()
    adapter = AuroraClientAdapter(inner)
    result = await adapter.place_order(
        symbol="BTC/USDT:USDT", side="buy", qty=0.01, stop_loss=79000,
    )
    assert result.get("orderId") == "X"


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
