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
async def test_market_entry_registers_single_tp() -> None:
    """정통 ICT 단일 TP (변형 5 정통화) — entry + reduce_only TP, 총 2건 호출."""
    client = _mock_client()
    bot = BotIctInstance(client=client, use_market_entry=True)
    await bot._execute_setup(_dummy_setup())
    # 호출 2건: entry + 단일 reduce_only TP
    assert client.place_order.await_count == 2
    # 첫 호출 = entry (reduce_only 미설정 또는 False)
    assert client.place_order.await_args_list[0].kwargs.get("reduce_only", False) is False
    # 두 번째 호출 = TP reduce_only
    assert client.place_order.await_args_list[1].kwargs.get("reduce_only") is True
