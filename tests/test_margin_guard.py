"""가용액 확인 실패 시 진입 보류와 수량 축소를 검증한다. 담당: Codex.

결정론적 잔고와 실제 CCXT 수량 정밀도 함수만 사용하며 거래소에 접속하지 않는다.
"""

from __future__ import annotations

import asyncio
from typing import Any

import ccxt
import pytest

from aurora_ict.bot.margin_guard import available_usdt, cap_qty_to_available

SYMBOL = "BTC/USDT:USDT"


class BalanceClient:
    """합성 잔고를 반환하고 실제 CCXT 정밀도로 수량을 내린다."""

    def __init__(self, balance: Any, *, failure: Exception | None = None) -> None:
        self.balance = balance
        self.failure = failure
        self.balance_reads = 0
        self.exchange = ccxt.bybit()
        self.exchange.set_markets([{
            "id": "BTCUSDT", "symbol": SYMBOL, "base": "BTC", "quote": "USDT",
            "settle": "USDT", "spot": False, "swap": True, "type": "swap",
            "linear": True, "inverse": False, "precision": {"amount": 0.001},
        }])

    async def fetch_balance(self) -> Any:
        self.balance_reads += 1
        if self.failure is not None:
            raise self.failure
        return self.balance

    def round_amount(self, symbol: str, qty: float) -> float:
        return float(self.exchange.amount_to_precision(symbol, qty))


class AvailableClient(BalanceClient):
    """잔고가 아닌 명시적 가용액 계약의 우선 사용을 검증한다."""

    def __init__(self, available: Any) -> None:
        super().__init__({"USDT": {"free": 1000.0, "total": 1000.0}})
        self.available = available

    async def fetch_available_usdt(self) -> Any:
        return self.available


@pytest.mark.asyncio
@pytest.mark.parametrize(("balance", "expected"), [
    ({"USDT": {"total": 150.0, "free": 50.0}}, 50.0),
    ({"free": {"USDT": "50.0"}}, 50.0),
    ({"USDT": {"total": 100.0, "free": None}, "free": {"USDT": 10.0}}, 10.0),
    ({"USDT": {"free": 50.0}, "free": {"USDT": 20.0}}, 20.0),
    ({"USDT": {"free": 0, "total": 100.0}}, 0.0),
    ({"USDT": {"free": -5.0, "total": 100.0}}, 0.0),
    ({"USDT": {"total": 100.0}}, None),
    ({}, None),
    (None, None),
    ({"USDT": {"free": "bad"}}, None),
    ({"USDT": {"free": float("nan")}}, None),
    ({"USDT": {"free": float("inf")}}, None),
    ({"USDT": {"free": True}}, None),
    ({"USDT": {"free": 50.0}, "info": {"result": {"list": []}}}, None),
])
async def test_available_usdt_only_accepts_explicit_finite_free(
    balance: Any, expected: float | None,
) -> None:
    assert await available_usdt(BalanceClient(balance)) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize("value", [None, "bad", float("nan"), float("inf"), True])
async def test_unknown_adapter_amount_never_falls_back_to_ccxt_free(value: Any) -> None:
    client = AvailableClient(value)
    assert await available_usdt(client) is None
    assert client.balance_reads == 0


@pytest.mark.asyncio
async def test_adapter_amount_has_priority() -> None:
    client = AvailableClient(74.15)
    assert await available_usdt(client) == 74.15
    assert client.balance_reads == 0


@pytest.mark.asyncio
async def test_balance_fetch_failure_is_unknown() -> None:
    client = BalanceClient({}, failure=RuntimeError("synthetic network failure"))
    assert await available_usdt(client) is None
    assert await cap_qty_to_available(client, SYMBOL, 5.0, 100.0, 10) == 0.0


@pytest.mark.asyncio
async def test_cancellation_propagates() -> None:
    class CancelledClient:
        async def fetch_balance(self) -> dict[str, Any]:
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await cap_qty_to_available(CancelledClient(), SYMBOL, 5.0, 100.0, 10)


@pytest.mark.asyncio
@pytest.mark.parametrize(("free", "qty", "expected"), [
    (100.0, 1.0, 1.0),
    (10.0, 5.0, 0.9),
    (0.01, 5.0, 0.0),
    (0.0, 5.0, 0.0),
    (-5.0, 5.0, 0.0),
])
async def test_cap_uses_available_amount_and_real_lot_precision(
    free: float, qty: float, expected: float,
) -> None:
    client = BalanceClient({"USDT": {"total": 1000.0, "free": free}})
    result = await cap_qty_to_available(client, SYMBOL, qty, 100.0, 10)
    assert result == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(("qty", "price", "leverage"), [
    (0.0, 100.0, 10), (-1.0, 100.0, 10), (5.0, 0.0, 10),
    (5.0, 100.0, 0), (float("nan"), 100.0, 10), (5.0, float("inf"), 10),
])
async def test_invalid_order_inputs_never_fetch_balance_or_place_orders(
    qty: float, price: float, leverage: int,
) -> None:
    client = BalanceClient({"USDT": {"free": 1000.0}})
    assert await cap_qty_to_available(client, SYMBOL, qty, price, leverage) == 0.0
    assert client.balance_reads == 0


@pytest.mark.asyncio
async def test_numeric_strings_are_normalized_before_arithmetic() -> None:
    client = BalanceClient({"USDT": {"free": "10.0"}})
    assert await cap_qty_to_available(client, SYMBOL, "5", "100", "10") == 0.9


@pytest.mark.asyncio
async def test_notional_overflow_never_allows_original_quantity() -> None:
    client = AvailableClient(1e308)
    assert await cap_qty_to_available(client, SYMBOL, 5.0, 100.0, 10) == 0.0
