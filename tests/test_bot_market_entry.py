"""use_market_entry 옵션 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.indicators.fvg import FVG, FVGType
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup

NY = ZoneInfo("America/New_York")


def _ohlcv_rows(bars: list[tuple[float, float, float, float]]) -> list[list[Any]]:
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = []
    for i, (o, h, lo, c) in enumerate(bars):
        t = start + timedelta(minutes=i * 5)
        ts_ms = int(t.timestamp() * 1000)
        rows.append([ts_ms, o, h, lo, c, 100.0])
    return rows


def _mock_client() -> AsyncMock:
    client = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=_ohlcv_rows([(100, 101, 99, 100)] * 20))
    client.place_order = AsyncMock(return_value={"orderId": "T1"})
    client.fetch_position = AsyncMock(return_value=None)
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    client.modify_stop_loss = AsyncMock(return_value={"retCode": 0})
    return client


def _dummy_setup() -> SilverBulletSetup:
    fvg = FVG(type=FVGType.BULLISH, idx=5, ts_ms=12345, low=98, high=102)
    return SilverBulletSetup(
        ts_ms=12345,
        direction=Direction.LONG,
        window="any",
        entry=100.0,
        stop_loss=95.0,
        take_profit=115.0,
        risk_reward=3.0,
        fvg=fvg,
    )


@pytest.mark.asyncio
async def test_limit_entry_default_passes_setup_price() -> None:
    """use_market_entry=False (default) → place_order price=setup.entry."""
    client = _mock_client()
    bot = BotIctInstance(client=client, use_market_entry=False)
    await bot._execute_setup(_dummy_setup())
    # 첫 호출 (entry) — kwargs price 확인.
    first_call = client.place_order.await_args_list[0]
    assert first_call.kwargs["price"] == 100.0


@pytest.mark.asyncio
async def test_market_entry_passes_price_none() -> None:
    """use_market_entry=True → place_order price=None (시장가)."""
    client = _mock_client()
    bot = BotIctInstance(client=client, use_market_entry=True)
    await bot._execute_setup(_dummy_setup())
    first_call = client.place_order.await_args_list[0]
    assert first_call.kwargs["price"] is None


@pytest.mark.asyncio
async def test_market_entry_still_registers_partial_tps() -> None:
    """market entry 박혀도 partial TP 3개는 그대로 reduce_only limit 등록."""
    client = _mock_client()
    bot = BotIctInstance(client=client, use_market_entry=True)
    await bot._execute_setup(_dummy_setup())
    # 호출 4개 — entry + tp1 + tp2 + tp3
    assert client.place_order.await_count == 4
    # tp1/tp2/tp3 는 reduce_only=True
    for call in client.place_order.await_args_list[1:]:
        assert call.kwargs.get("reduce_only") is True
