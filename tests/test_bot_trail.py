"""BotIctInstance enable_trail 통합 테스트."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
from aurora_ict.strategy.silver_bullet import Direction

NY = ZoneInfo("America/New_York")


def _ohlcv_rows(bars: list[tuple[float, float, float, float]]) -> list[list[Any]]:
    """ccxt-style OHLCV rows."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = []
    for i, (o, h, lo, c) in enumerate(bars):
        t = start + timedelta(minutes=i * 5)
        ts_ms = int(t.timestamp() * 1000)
        rows.append([ts_ms, o, h, lo, c, 100.0])
    return rows


def _mock_client(rows: list[list[Any]]) -> AsyncMock:
    client = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=rows)
    client.place_order = AsyncMock(return_value={"orderId": "T1"})
    client.fetch_position = AsyncMock(
        return_value={"contracts": 0.1, "symbol": "BTC/USDT:USDT"},
    )
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    client.modify_stop_loss = AsyncMock(return_value={"retCode": 0})
    return client


# ============================================================
# enable_trail OFF — trail 호출 없음
# ============================================================


@pytest.mark.asyncio
async def test_trail_disabled_no_modify_call() -> None:
    """enable_trail=False (default) → modify_stop_loss 호출 X."""
    rows = _ohlcv_rows([
        (100, 101, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 115, 108, 114),
        (114, 120, 113, 119),
        (119, 119, 105, 106),
        (106, 112, 105.5, 111),
    ])
    client = _mock_client(rows)
    bot = BotIctInstance(client=client, enable_trail=False)
    bot.active_position = _ActivePosition(
        direction=Direction.LONG,
        entry=100.0,
        stop_loss=95.0,
        take_profit=130.0,
        qty=0.1,
        setup_ts_ms=1,
    )
    await bot.step()
    assert client.modify_stop_loss.await_count == 0


# ============================================================
# enable_trail ON — trail 호출 + SL 갱신
# ============================================================


@pytest.mark.asyncio
async def test_trail_enabled_calls_modify_and_updates_sl() -> None:
    """enable_trail=True + 새 swing low > entry → modify_stop_loss 호출 + active_position SL 갱신."""
    rows = _ohlcv_rows([
        (100, 101, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 115, 108, 114),
        (114, 120, 113, 119),    # swing high (no influence)
        (119, 119, 105, 106),    # swing low @ 105 (idx=5)
        (106, 112, 105.5, 111),
        (111, 117, 110, 116),
    ])
    client = _mock_client(rows)
    bot = BotIctInstance(
        client=client,
        enable_trail=True,
        trail_buffer_ratio=0.001,
    )
    bot.active_position = _ActivePosition(
        direction=Direction.LONG,
        entry=100.0,
        stop_loss=95.0,
        take_profit=130.0,
        qty=0.1,
        setup_ts_ms=1,
    )
    await bot.step()
    assert client.modify_stop_loss.await_count == 1
    # SL 105 - 0.105 ≈ 104.895
    expected_new_sl = 105.0 - 105.0 * 0.001
    assert bot.active_position is None or bot.active_position.stop_loss == pytest.approx(expected_new_sl)


@pytest.mark.asyncio
async def test_trail_enabled_no_swing_no_modify() -> None:
    """단조 증가 → swing 없음 → modify X."""
    rows = _ohlcv_rows([(100 + i, 101 + i, 99 + i, 100 + i) for i in range(15)])
    client = _mock_client(rows)
    bot = BotIctInstance(client=client, enable_trail=True)
    bot.active_position = _ActivePosition(
        direction=Direction.LONG,
        entry=100.0,
        stop_loss=95.0,
        take_profit=130.0,
        qty=0.1,
        setup_ts_ms=1,
    )
    await bot.step()
    assert client.modify_stop_loss.await_count == 0


@pytest.mark.asyncio
async def test_trail_enabled_modify_failure_keeps_old_sl() -> None:
    """modify_stop_loss 실패 시 active_position SL 갱신 X."""
    rows = _ohlcv_rows([
        (100, 101, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 115, 108, 114),
        (114, 120, 113, 119),
        (119, 119, 105, 106),
        (106, 112, 105.5, 111),
    ])
    client = _mock_client(rows)
    client.modify_stop_loss = AsyncMock(side_effect=RuntimeError("API down"))
    bot = BotIctInstance(client=client, enable_trail=True)
    bot.active_position = _ActivePosition(
        direction=Direction.LONG,
        entry=100.0,
        stop_loss=95.0,
        take_profit=130.0,
        qty=0.1,
        setup_ts_ms=1,
    )
    await bot.step()
    # SL 갱신 X (실패 시)
    assert bot.active_position is not None
    assert bot.active_position.stop_loss == 95.0
