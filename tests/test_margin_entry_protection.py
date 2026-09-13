"""신규 진입 보류가 기존 포지션 보호를 막지 않는지 검증한다. 담당: Codex.

두 모델의 실제 메서드와 인메모리 주문 장부를 사용한다. 네트워크나 mock은 쓰지 않는다.
"""

from __future__ import annotations

from typing import Any

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
from aurora_ict.bot.bot_trend_instance import BotTrendInstance, _TrendPosition
from aurora_ict.indicators.fvg import FVG, FVGType
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup

SYMBOL = "BTC/USDT:USDT"
BAR = {"ts": 1_700_000_000_000.0, "low": 99.0, "high": 101.0}


class RecordingClient:
    """합성 잔고, 포지션과 주문 상태를 메모리에서만 관리한다."""

    def __init__(self, available: float | None, *, accept_sl: bool = True) -> None:
        self.available = available
        self.accept_sl = accept_sl
        self.available_reads = 0
        self.balance_reads = 0
        self.orders: list[dict[str, Any]] = []
        self.protection: list[dict[str, Any]] = []
        self.position: dict[str, Any] | None = None

    async def fetch_available_usdt(self) -> float | None:
        self.available_reads += 1
        return self.available

    async def fetch_balance(self) -> dict[str, dict[str, float]]:
        self.balance_reads += 1
        return {"USDT": {"total": 1000.0, "free": 1000.0}}

    async def fetch_ticker(self, symbol: str) -> float:
        return 100.0

    async def fetch_position(self, symbol: str) -> dict[str, Any] | None:
        return self.position

    async def place_order(
        self, symbol: str, side: str, qty: float, price: float | None,
        reduce_only: bool = False, **params: Any,
    ) -> dict[str, Any]:
        order = {"symbol": symbol, "side": side, "qty": qty, "price": price,
                 "reduce_only": reduce_only, **params}
        self.orders.append(order)
        filled = qty if price is None else 0.0
        if reduce_only:
            assert self.position is not None
            assert side != self.position["side"]
            assert qty <= self.position["contracts"]
            remaining = self.position["contracts"] - qty
            self.position = {**self.position, "contracts": remaining} if remaining else None
        elif filled:
            self.position = {"contracts": filled, "side": side, "entryPrice": 100.0}
        return {"orderId": str(len(self.orders)), "filled_qty": filled,
                "avg_fill_price": 100.0 if filled else 0.0}

    async def set_position_tpsl(self, symbol: str, **params: Any) -> dict[str, int]:
        self.protection.append({"symbol": symbol, **params})
        if not self.accept_sl:
            return {}
        if self.position is not None:
            self.position["stopLossPrice"] = params["stop_loss"]
        return {"retCode": 0}


def _setup() -> SilverBulletSetup:
    return SilverBulletSetup(
        ts_ms=1_700_000_000_000, direction=Direction.LONG, window="any",
        entry=100.0, stop_loss=98.0, take_profit=106.0, risk_reward=3.0,
        fvg=FVG(type=FVGType.BULLISH, idx=5, ts_ms=1_700_000_000_000,
                low=99.0, high=101.0),
    )


def _bot(model: str, client: RecordingClient, limit: bool) -> BotIctInstance | BotTrendInstance:
    if model == "origo":
        return BotIctInstance(client=client, symbol=SYMBOL, use_market_entry=not limit)
    return BotTrendInstance(client=client, symbol=SYMBOL, limit_entry=limit)


def _existing_position(bot: BotIctInstance | BotTrendInstance, client: RecordingClient) -> None:
    client.position = {"contracts": 1.0, "side": "buy", "entryPrice": 100.0}
    if isinstance(bot, BotIctInstance):
        bot.active_position = _ActivePosition(
            direction=Direction.LONG, entry=100.0, stop_loss=98.0,
            take_profit=106.0, qty=1.0, setup_ts_ms=1_700_000_000_000,
        )
    else:
        bot.active_position = _TrendPosition(
            direction=Direction.LONG, entry=100.0, qty=1.0, stop=98.0,
            entry_ts_ms=1_700_000_000_000, init_qty=1.0,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["origo", "cursus"])
@pytest.mark.parametrize("limit", [False, True], ids=["market", "limit"])
@pytest.mark.parametrize("available", [None, -5.0, 0.0, 1000.0])
async def test_both_entry_routes_require_known_positive_available_balance(
    model: str, limit: bool, available: float | None,
) -> None:
    client = RecordingClient(available)
    bot = _bot(model, client, limit)
    if isinstance(bot, BotIctInstance):
        await bot._execute_setup(_setup())
    else:
        await bot._open(Direction.LONG, 100.0, bar=BAR if limit else None)
    assert client.available_reads == 1
    assert client.balance_reads > 0
    if available is None or available <= 0:
        assert client.orders == []
        assert client.protection == []
        assert bot.active_position is None
        assert client.position is None
        pending = bot._pending_entry if isinstance(bot, BotIctInstance) else bot._pending_limit
        assert pending is None
    else:
        assert len(client.orders) == 1
        assert client.orders[0]["reduce_only"] is False
        assert (client.orders[0]["price"] is not None) is limit


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["origo", "cursus"])
@pytest.mark.parametrize("accept_sl", [True, False])
async def test_unknown_balance_does_not_block_sl_or_emergency_close(
    model: str, accept_sl: bool,
) -> None:
    client = RecordingClient(None, accept_sl=accept_sl)
    bot = _bot(model, client, False)
    _existing_position(bot, client)
    if isinstance(bot, BotIctInstance):
        ok = await bot._ensure_protective_sl(106.0, 2.0)
    else:
        ok = await bot._apply_protective_sl(98.0, 100.0)
    assert ok is accept_sl
    assert client.available_reads == 0
    assert client.balance_reads == 0
    assert len(client.protection) == (1 if accept_sl else 2)
    if accept_sl:
        assert bot.active_position is not None
        assert client.position["stopLossPrice"] == 98.0
        assert client.orders == []
    else:
        assert bot.active_position is None
        assert client.position is None
        assert len(client.orders) == 1
        assert client.orders[0]["reduce_only"] is True
        assert client.orders[0]["side"] == "sell"


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [False, True], ids=["market", "limit"])
async def test_cursus_reverse_closes_but_does_not_reenter_with_unknown_balance(limit: bool) -> None:
    client = RecordingClient(None)
    bot = BotTrendInstance(client=client, symbol=SYMBOL, limit_entry=limit)
    _existing_position(bot, client)
    await bot._reverse(Direction.SHORT, 100.0, bar=BAR if limit else None)
    assert len(client.orders) == 1
    assert client.orders[0]["reduce_only"] is True
    assert client.orders[0]["side"] == "sell"
    assert client.available_reads == 1
    assert bot.active_position is None
    assert bot._pending_limit is None
    assert client.position is None
