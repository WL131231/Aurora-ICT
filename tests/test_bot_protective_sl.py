"""무SL 방지 — 보호 SL 보장 + 비상청산 (P1 / #LIVE-6) 단위 테스트.

계획가~체결 사이 가격 급변으로 계획 SL 이 현재가 너머로 가면 거래소가 거부(10001)
→ 무SL 포지션. 현재가 기준 보호 측 SL 재계산, 실패 시 청산을 검증.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
from aurora_ict.strategy.silver_bullet import Direction


def _client() -> AsyncMock:
    client = AsyncMock()
    client.fetch_ticker = AsyncMock(return_value=100.5)
    client.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    client.place_order = AsyncMock(return_value={"orderId": "C1"})
    client.fetch_position = AsyncMock(return_value=None)
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    return client


def _pos(direction: Direction, entry: float, sl: float, tp: float) -> _ActivePosition:
    return _ActivePosition(
        direction=direction, entry=entry, stop_loss=sl,
        take_profit=tp, qty=1.0, setup_ts_ms=0,
    )


# ============================================================
# 순수 함수
# ============================================================


def test_protective_sl_sides() -> None:
    assert BotIctInstance._protective_sl(Direction.SHORT, 100.0, 1.0) == 101.0
    assert BotIctInstance._protective_sl(Direction.LONG, 100.0, 1.0) == 99.0


def test_is_protective_sl() -> None:
    assert BotIctInstance._is_protective_sl(Direction.SHORT, 101.0, 100.0) is True
    assert BotIctInstance._is_protective_sl(Direction.SHORT, 99.0, 100.0) is False
    assert BotIctInstance._is_protective_sl(Direction.LONG, 99.0, 100.0) is True
    assert BotIctInstance._is_protective_sl(Direction.LONG, 101.0, 100.0) is False
    assert BotIctInstance._is_protective_sl(Direction.LONG, 0.0, 100.0) is False  # 무SL


# ============================================================
# _ensure_protective_sl
# ============================================================


@pytest.mark.asyncio
async def test_planned_sl_protective_kept() -> None:
    """계획 SL 이 이미 보호 측이면 그대로 적용 (LONG SL 95 < 현재가 100.5)."""
    client = _client()
    bot = BotIctInstance(client=client)
    bot.active_position = _pos(Direction.LONG, 100.0, 95.0, 115.0)
    ok = await bot._ensure_protective_sl(115.0, 5.0)
    assert ok is True
    kw = client.set_position_tpsl.await_args_list[0].kwargs
    assert kw["stop_loss"] == 95.0
    assert bot.active_position.stop_loss == 95.0
    client.place_order.assert_not_awaited()  # 청산 없음


@pytest.mark.asyncio
async def test_short_sl_wrong_side_recomputed() -> None:
    """SHORT SL 이 현재가 아래(거부 영역)면 현재가 기준 보호 측으로 재계산.

    현재가 100.5, 계획 SL 99(거부), 거리 1 → 재계산 SL = 100.5 + 1 = 101.5.
    """
    client = _client()
    bot = BotIctInstance(client=client)
    bot.active_position = _pos(Direction.SHORT, 100.0, 99.0, 95.0)
    ok = await bot._ensure_protective_sl(95.0, 1.0)
    assert ok is True
    kw = client.set_position_tpsl.await_args_list[0].kwargs
    assert kw["stop_loss"] == 101.5
    assert bot.active_position.stop_loss == 101.5


@pytest.mark.asyncio
async def test_sl_set_fails_twice_emergency_close() -> None:
    """set_position_tpsl 2회 실패 → 무SL 방치 금지: 포지션 reduce_only 청산."""
    client = _client()
    client.set_position_tpsl = AsyncMock(return_value=None)  # 항상 실패
    bot = BotIctInstance(client=client)
    bot.active_position = _pos(Direction.SHORT, 100.0, 99.0, 95.0)
    ok = await bot._ensure_protective_sl(95.0, 1.0)
    assert ok is False
    assert bot.active_position is None  # 청산됨
    close = client.place_order.await_args_list[-1].kwargs
    assert close["reduce_only"] is True
    assert close["side"] == "buy"  # SHORT 청산 = buy


@pytest.mark.asyncio
async def test_fallback_distance_when_zero() -> None:
    """sl_distance=0 (복구 등) → entry/현재가 대비 _FALLBACK_SL_PCT 적용.

    LONG, 현재가 100.0, fallback 0.5% → SL = 100 - 0.5 = 99.5.
    """
    client = _client()
    client.fetch_ticker = AsyncMock(return_value=100.0)
    bot = BotIctInstance(client=client)
    bot.active_position = _pos(Direction.LONG, 100.0, 0.0, 0.0)
    ok = await bot._ensure_protective_sl(None, 0.0)
    assert ok is True
    kw = client.set_position_tpsl.await_args_list[0].kwargs
    assert kw["stop_loss"] == 99.5


# ============================================================
# 복구 (P1-2)
# ============================================================


@pytest.mark.asyncio
async def test_recover_naked_position_applies_sl() -> None:
    """SL 없는(=0) 포지션 복구 시 보호 SL 적용."""
    client = _client()
    client.fetch_ticker = AsyncMock(return_value=100.0)
    client.fetch_position = AsyncMock(return_value={
        "contracts": 1.0, "side": "long", "entryPrice": 100.0,
        "stopLossPrice": 0, "takeProfitPrice": 0,
    })
    bot = BotIctInstance(client=client)
    await bot._recover_position_from_exchange()
    assert bot.active_position is not None
    client.set_position_tpsl.assert_awaited()
    kw = client.set_position_tpsl.await_args_list[0].kwargs
    # LONG, ref 100, fallback 0.5% → 99.5
    assert kw["stop_loss"] == 99.5
    assert bot.active_position.stop_loss == 99.5
