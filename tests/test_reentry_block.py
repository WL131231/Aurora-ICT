"""#REENTRY-BLOCK 2026-07-23: 사용자 TP 전 수동청산 후 동일 setup 재진입 차단 테스트.

사용자가 조기 익절(수동 청산)하면 봇이 같은 방향·같은 진입가 setup 을 곧바로
재진입해 사용자 의도를 무시했다. 청산된 setup 을 쿨다운 동안 차단.
mock 0 — 결정론적 합성.
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.indicators.fvg import FVG, FVGType
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup


def _bot(**kw) -> BotIctInstance:
    c = AsyncMock()
    c.fetch_position = AsyncMock(return_value=None)
    c.fetch_ticker = AsyncMock(return_value=100.0)
    c.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    c.place_order = AsyncMock(return_value={
        "orderId": "T", "filled_qty": 1.0, "avg_fill_price": 100.0})
    c.set_position_tpsl = AsyncMock(return_value=True)
    c.cancel_bot_orders = AsyncMock(return_value=0)
    c.position_opened_by_bot = AsyncMock(return_value=False)
    return BotIctInstance(client=c, **kw)


def _setup(direction: Direction, entry: float) -> SilverBulletSetup:
    fvg = FVG(type=FVGType.BULLISH, idx=5, ts_ms=1, low=98, high=102)
    return SilverBulletSetup(
        ts_ms=1, direction=direction, window="any",
        entry=entry, stop_loss=95.0, take_profit=115.0, risk_reward=3.0, fvg=fvg)


def test_arm_and_block_same_setup() -> None:
    """arm 후 같은 방향·진입가 1% 이내 setup → 차단(True)."""
    bot = _bot()
    bot._arm_reentry_block(Direction.LONG, 100.0, "test")
    assert bot._reentry_blocked(_setup(Direction.LONG, 100.0)) is True
    assert bot._reentry_blocked(_setup(Direction.LONG, 100.5)) is True   # 0.5% 이내
    assert bot._reentry_blocked(_setup(Direction.LONG, 105.0)) is False  # 5% 밖
    assert bot._reentry_blocked(_setup(Direction.SHORT, 100.0)) is False  # 반대방향 허용


def test_block_expires() -> None:
    """만료된 block 은 해제되고 통과."""
    bot = _bot()
    bot._arm_reentry_block(Direction.LONG, 100.0, "test")
    # 만료 시각을 과거로 강제.
    bdir, bentry, _ = bot._reentry_block
    bot._reentry_block = (bdir, bentry, int(time.time() * 1000) - 1000)
    assert bot._reentry_blocked(_setup(Direction.LONG, 100.0)) is False
    assert bot._reentry_block is None  # 해제됨


def test_block_off_when_sec_zero() -> None:
    """manual_close_reentry_block_sec=0 이면 arm 안 함(기능 off)."""
    bot = _bot(manual_close_reentry_block_sec=0)
    bot._arm_reentry_block(Direction.LONG, 100.0, "test")
    assert bot._reentry_block is None


@pytest.mark.asyncio
async def test_execute_setup_blocked_on_reentry() -> None:
    """차단 활성 시 _execute_setup 이 동일 setup 진입 안 함 (place_order 미호출)."""
    bot = _bot()
    bot._arm_reentry_block(Direction.LONG, 100.0, "test")
    await bot._execute_setup(_setup(Direction.LONG, 100.0))
    assert bot.active_position is None
    bot.client.place_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_reentry_block_cleared_on_new_entry() -> None:
    """반대 방향 등 다른 setup 으로 진입 성공하면 block 해제."""
    bot = _bot(disable_time_filter=True)
    bot._arm_reentry_block(Direction.LONG, 100.0, "test")
    # 반대 방향(SHORT)은 차단 안 됨 → 진입 성공 → block 해제.
    await bot._execute_setup(_setup(Direction.SHORT, 100.0))
    assert bot._reentry_block is None


def test_recovery_already_recorded_dedupe() -> None:
    """#REC-DEDUPE: 마지막 청산 이후 동일 방향·가격 ENTRY/RECOVERED 있으면 True."""
    from types import SimpleNamespace as E

    from aurora_ict.bot.bot_ict_instance import recovery_already_recorded
    from aurora_ict.interfaces.trades_store import TradeEventType as T

    sym = "BTC/USDT:USDT"
    # ENTRY(미청산) 존재 → 중복
    evs = [E(symbol=sym, event_type=T.ENTRY, direction="long", ts_ms=100, price=100.0)]
    assert recovery_already_recorded(evs, sym, "long", 100.0) is True
    # 그 뒤 청산 → 미청산 아님 → False
    evs.append(E(symbol=sym, event_type=T.SL_HIT, direction="long", ts_ms=200, price=95.0))
    assert recovery_already_recorded(evs, sym, "long", 100.0) is False
    # 청산 후 RECOVERED 재기록 존재 → 그 다음 복구는 중복 True
    evs.append(E(symbol=sym, event_type=T.RECOVERED, direction="long", ts_ms=300, price=100.0))
    assert recovery_already_recorded(evs, sym, "long", 100.0) is True
    # 다른 방향/먼 가격/다른 심볼 → False
    assert recovery_already_recorded(evs, sym, "short", 100.0) is False
    assert recovery_already_recorded(evs, sym, "long", 110.0) is False
    assert recovery_already_recorded(evs, "ETH/USDT:USDT", "long", 100.0) is False
