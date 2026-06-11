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
    client.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
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
async def test_sl_dist_mult_scales_sl_and_preserves_rr() -> None:
    """#EDGE-V2: sl_dist_mult=3 → SL 거리 3배 + TP 는 원 RR(3.0) 유지 비례 확장.

    원본 entry=100/sl=95(거리5)/tp=115(RR3) → sl=85(거리15), tp=145(15×3).
    """
    client = _mock_client()
    bot = BotIctInstance(client=client, sl_dist_mult=3.0)
    await bot._execute_setup(_dummy_setup())
    tpsl_kw = client.set_position_tpsl.await_args_list[0].kwargs
    assert tpsl_kw["stop_loss"] == pytest.approx(85.0)
    assert tpsl_kw["take_profit"] == pytest.approx(145.0)


@pytest.mark.asyncio
async def test_sl_dist_mult_short_direction() -> None:
    """#EDGE-V2: 숏 방향 — SL 은 위로 2배, TP 는 아래로 RR 유지."""
    client = _mock_client()
    bot = BotIctInstance(client=client, sl_dist_mult=2.0)
    fvg = FVG(type=FVGType.BEARISH, idx=5, ts_ms=12346, low=98, high=102)
    setup = SilverBulletSetup(
        ts_ms=12346, direction=Direction.SHORT, window="any",
        entry=100.0, stop_loss=102.0, take_profit=94.0, risk_reward=3.0, fvg=fvg,
    )
    await bot._execute_setup(setup)
    tpsl_kw = client.set_position_tpsl.await_args_list[0].kwargs
    # 거리 2→4: sl=104, tp=100-4×3=88
    assert tpsl_kw["stop_loss"] == pytest.approx(104.0)
    assert tpsl_kw["take_profit"] == pytest.approx(88.0)


@pytest.mark.asyncio
async def test_limit_entry_default_uses_setup_entry() -> None:
    """use_market_entry=False (default) → setup.entry (계획가) limit + SL/TP 동봉 (#LIVE-3).

    현재가 진입은 setup 타점 지나면 RR 망가짐 → 계획가(setup.entry) limit 으로
    가격이 retrace 시 체결, RR 보존.
    """
    client = _mock_client()
    bot = BotIctInstance(client=client, use_market_entry=False)
    await bot._execute_setup(_dummy_setup())
    first_call = client.place_order.await_args_list[0]
    # limit = setup.entry (계획가 100.0), 현재가(ticker 100.5) 아님
    assert first_call.kwargs["price"] == 100.0
    # #LIVE-4: SL/TP 는 entry 주문에 동봉하지 않음 (체결 후 set_position_tpsl 로)
    assert "stop_loss" not in first_call.kwargs
    assert "take_profit" not in first_call.kwargs
    # 즉시 체결 → set_position_tpsl 로 SL/TP 박음
    client.set_position_tpsl.assert_awaited_once()
    tpsl_kw = client.set_position_tpsl.await_args_list[0].kwargs
    assert tpsl_kw["stop_loss"] == 95.0
    assert tpsl_kw["take_profit"] == 115.0


@pytest.mark.asyncio
async def test_market_entry_passes_price_none() -> None:
    """use_market_entry=True → place_order price=None (시장가)."""
    client = _mock_client()
    bot = BotIctInstance(client=client, use_market_entry=True)
    await bot._execute_setup(_dummy_setup())
    first_call = client.place_order.await_args_list[0]
    assert first_call.kwargs["price"] is None


@pytest.mark.asyncio
async def test_entry_then_set_tpsl_after_fill() -> None:
    """#LIVE-4: entry 주문은 SL/TP 없이 1회, 체결 후 set_position_tpsl 로 SL/TP 설정."""
    client = _mock_client()
    bot = BotIctInstance(client=client, use_market_entry=True)
    await bot._execute_setup(_dummy_setup())
    # entry 1건만 (별도 reduce_only TP 주문 없음)
    assert client.place_order.await_count == 1
    call = client.place_order.await_args_list[0].kwargs
    assert call.get("reduce_only", False) is False
    # entry 주문엔 SL/TP 동봉 안 함
    assert "stop_loss" not in call
    assert "take_profit" not in call
    # 체결 후 set_position_tpsl 로 SL/TP conditional 설정
    client.set_position_tpsl.assert_awaited_once()
    tpsl_kw = client.set_position_tpsl.await_args_list[0].kwargs
    assert tpsl_kw["stop_loss"] == 95.0
    assert tpsl_kw["take_profit"] == 115.0
