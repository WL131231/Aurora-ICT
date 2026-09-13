"""계획 수량 대비 극소액 진입 차단을 검증한다. 담당: Codex.

20% 하한은 유지하고, 가용액 미확인 시 원 수량으로 우회하지 않는지 확인한다.
합성 잔고와 실제 CCXT 내림 정밀도만 사용한다.
"""

from __future__ import annotations

import ccxt
import pytest

from aurora_ict.bot.margin_guard import cap_qty_to_available

SYMBOL = "BTC/USDT:USDT"


class SizeClient:
    """결정론적 잔고와 거래소 최소 수량 단위를 제공한다."""

    def __init__(self, free: float | None, step: float = 0.001) -> None:
        self.free = free
        self.exchange = ccxt.bybit()
        self.exchange.set_markets([{
            "id": "BTCUSDT", "symbol": SYMBOL, "base": "BTC", "quote": "USDT",
            "settle": "USDT", "spot": False, "swap": True, "type": "swap",
            "linear": True, "inverse": False, "precision": {"amount": step},
        }])

    async def fetch_balance(self) -> dict[str, dict[str, float]]:
        if self.free is None:
            raise RuntimeError("synthetic network failure")
        return {"USDT": {"free": self.free}}

    def round_amount(self, symbol: str, qty: float) -> float:
        return float(self.exchange.amount_to_precision(symbol, qty))


@pytest.mark.asyncio
@pytest.mark.parametrize(("free", "ratio", "expected"), [
    (1.0, 0.2, 0.0),
    (1.2, 0.2, 2.16),
    (1000.0, 0.2, 10.0),
    (1.0, 0.0, 1.8),
    (None, 0.2, 0.0),
    (None, 0.0, 0.0),
    (0.0, 0.2, 0.0),
    (-1.0, 0.2, 0.0),
])
async def test_minimum_ratio_and_unknown_balance(
    free: float | None, ratio: float, expected: float,
) -> None:
    qty = await cap_qty_to_available(
        SizeClient(free), SYMBOL, qty=10.0, price=10.0,
        leverage=20, min_qty_ratio=ratio,
    )
    assert qty == pytest.approx(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize(("step", "expected"), [(1.0, 2.0), (3.0, 0.0), (5.0, 0.0)])
async def test_ratio_uses_real_exchange_rounding(step: float, expected: float) -> None:
    qty = await cap_qty_to_available(
        SizeClient(1.2, step), SYMBOL, qty=10.0, price=10.0,
        leverage=20, min_qty_ratio=0.2,
    )
    assert qty == pytest.approx(expected)
