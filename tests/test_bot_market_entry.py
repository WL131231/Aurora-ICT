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
    client.fetch_ticker = AsyncMock(return_value=100.5)   # marketable limit 현재가
    # 즉시 체결 시뮬 (filled_qty/avg_fill_price 포함).
    client.place_order = AsyncMock(return_value={
        "orderId": "T1", "filled_qty": 1.0, "avg_fill_price": 100.5,
    })
    client.fetch_position = AsyncMock(return_value=None)
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    client.modify_stop_loss = AsyncMock(return_value={"retCode": 0})
    client.cancel_all_orders = AsyncMock(return_value=None)
    client.fetch_closed_positions = AsyncMock(return_value=[])
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
async def test_limit_entry_default_uses_marketable_limit() -> None:
    """use_market_entry=False (default) → marketable limit (현재가) + SL/TP 동봉 (#LIVE-1)."""
    client = _mock_client()
    bot = BotIctInstance(client=client, use_market_entry=False)
    await bot._execute_setup(_dummy_setup())
    first_call = client.place_order.await_args_list[0]
    # marketable limit = 현재가 (fetch_ticker), setup.entry(100.0) 아님
    assert first_call.kwargs["price"] == 100.5
    # SL/TP 가 entry 주문에 동봉 (setup 기준 고정 레벨)
    assert first_call.kwargs["stop_loss"] == 95.0
    assert first_call.kwargs["take_profit"] == 115.0


@pytest.mark.asyncio
async def test_market_entry_passes_price_none() -> None:
    """use_market_entry=True → place_order price=None (시장가)."""
    client = _mock_client()
    bot = BotIctInstance(client=client, use_market_entry=True)
    await bot._execute_setup(_dummy_setup())
    first_call = client.place_order.await_args_list[0]
    assert first_call.kwargs["price"] is None


@pytest.mark.asyncio
async def test_entry_registers_sl_tp_inline_single_call() -> None:
    """#LIVE-1 fix: entry 1회 호출에 SL/TP 동봉 (별도 reduce_only TP 주문 없음)."""
    client = _mock_client()
    bot = BotIctInstance(client=client, use_market_entry=True)
    await bot._execute_setup(_dummy_setup())
    # entry 1건만 — TP 는 동봉되어 별도 reduce_only 주문 없음
    assert client.place_order.await_count == 1
    call = client.place_order.await_args_list[0].kwargs
    assert call.get("reduce_only", False) is False
    assert call["stop_loss"] == 95.0
    assert call["take_profit"] == 115.0
