"""BotIctInstance multi_tf 모드 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.indicators.fvg import FVG, FVGType
from aurora_ict.strategy.htf_setup_tracker import HtfActiveSetup, HtfSetupTracker
from aurora_ict.strategy.ltf_entry_confirmer import ConfirmedEntry
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup

NY = ZoneInfo("America/New_York")


def _flat_rows(n: int) -> list[list[Any]]:
    """평탄한 OHLCV n 봉."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = []
    for i in range(n):
        t = start + timedelta(minutes=i * 5)
        ts_ms = int(t.timestamp() * 1000)
        rows.append([ts_ms, 100.0, 100.5, 99.5, 100.0, 50.0])
    return rows


def _mock_client(rows: list[list[Any]]) -> AsyncMock:
    client = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=rows)
    client.place_order = AsyncMock(return_value={"orderId": "TEST"})
    client.fetch_position = AsyncMock(return_value=None)
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    return client


@pytest.mark.asyncio
async def test_multi_tf_step_initializes_tracker() -> None:
    """multi_tf=True 첫 step 후 _htf_tracker 초기화 + htf_list 매칭."""
    client = _mock_client(_flat_rows(50))
    bot = BotIctInstance(
        client=client,
        timeframe="5m",
        multi_tf=True,
    )
    await bot.step()
    assert bot._htf_tracker is not None
    assert isinstance(bot._htf_tracker, HtfSetupTracker)
    # 5m → 7개 HTF (15m / 30m / 1h / 2h / 4h / 1d / 1w)
    assert bot._htf_tracker.htf_list() == (
        "15m", "30m", "1h", "2h", "4h", "1d", "1w",
    )


@pytest.mark.asyncio
async def test_multi_tf_no_active_setups_no_entry() -> None:
    """평탄한 봉 → setup 없음 → place_order 호출 X."""
    client = _mock_client(_flat_rows(50))
    bot = BotIctInstance(
        client=client,
        timeframe="5m",
        multi_tf=True,
    )
    await bot.step()
    assert bot.active_position is None
    assert client.place_order.await_count == 0


@pytest.mark.asyncio
async def test_multi_tf_step_returns_no_action_when_no_zone_match() -> None:
    """가격이 어느 HTF zone 안에도 없으면 NO_ACTION signal."""
    client = _mock_client(_flat_rows(50))
    bot = BotIctInstance(
        client=client,
        timeframe="5m",
        multi_tf=True,
    )
    signal = await bot.step()
    assert signal.action.value == "no_action"


def test_confirmed_to_setup_long() -> None:
    """ConfirmedEntry (LONG) → SilverBulletSetup 변환 — confluence_score=2."""
    fvg = FVG(type=FVGType.BULLISH, idx=10, ts_ms=999, low=100, high=110)
    confirmed = ConfirmedEntry(
        direction=Direction.LONG,
        entry=105.0,
        stop_loss=99.0,
        take_profit=129.0,
        ltf_fvg=fvg,
        htf_tf="1h",
        htf_setup_ts_ms=12345,
    )
    setup = BotIctInstance._confirmed_to_setup(confirmed)
    assert isinstance(setup, SilverBulletSetup)
    assert setup.direction is Direction.LONG
    assert setup.entry == 105.0
    assert setup.stop_loss == 99.0
    assert setup.take_profit == 129.0
    assert setup.confluence_score == 2
    assert setup.window == "multi_tf:1h"
    assert setup.ts_ms == 12345
    # RR = (129 - 105) / (105 - 99) = 24/6 = 4.0
    assert setup.risk_reward == pytest.approx(4.0)


def test_confirmed_to_setup_short() -> None:
    """ConfirmedEntry (SHORT) → SilverBulletSetup 변환."""
    fvg = FVG(type=FVGType.BEARISH, idx=10, ts_ms=999, low=90, high=100)
    confirmed = ConfirmedEntry(
        direction=Direction.SHORT,
        entry=95.0,
        stop_loss=101.0,
        take_profit=83.0,
        ltf_fvg=fvg,
        htf_tf="4h",
        htf_setup_ts_ms=99999,
    )
    setup = BotIctInstance._confirmed_to_setup(confirmed)
    assert setup.direction is Direction.SHORT
    assert setup.confluence_score == 2
    assert setup.window == "multi_tf:4h"
    # RR = (95 - 83) / (101 - 95) = 12/6 = 2.0
    assert setup.risk_reward == pytest.approx(2.0)


@pytest.mark.asyncio
async def test_multi_tf_active_position_skips_new_entry() -> None:
    """active_position 있으면 신규 진입 시도 X (sync_position_state 만)."""
    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    client = _mock_client(_flat_rows(50))
    bot = BotIctInstance(
        client=client,
        timeframe="5m",
        multi_tf=True,
    )
    bot.active_position = _ActivePosition(
        direction=Direction.LONG,
        entry=100.0,
        stop_loss=95.0,
        take_profit=115.0,
        qty=0.1,
        setup_ts_ms=111,
    )
    await bot.step()
    assert client.place_order.await_count == 0


@pytest.mark.asyncio
async def test_multi_tf_invalidates_sl_hit_setup() -> None:
    """SL 침범한 HTF setup 은 tracker 에서 제거됨."""
    client = _mock_client(_flat_rows(50))
    bot = BotIctInstance(
        client=client,
        timeframe="5m",
        multi_tf=True,
    )
    # 첫 step → tracker init
    await bot.step()
    assert bot._htf_tracker is not None
    # 가짜 LONG setup 박은 거 박은 거 박은 거 박은 거 박은 거: SL=95 박힘.
    fvg = FVG(type=FVGType.BULLISH, idx=2, ts_ms=1, low=98, high=102)
    setup = SilverBulletSetup(
        ts_ms=1,
        direction=Direction.LONG,
        window="1h",
        entry=100.0,
        stop_loss=95.0,
        take_profit=115.0,
        risk_reward=3.0,
        fvg=fvg,
    )
    bot._htf_tracker._active["1h"] = HtfActiveSetup(htf_tf="1h", setup=setup)
    # 평탄한 봉 close=100 박혔어도 step 안에서 SL 침범 체크는 됨 — 직접 호출.
    removed = bot._htf_tracker.invalidate_if_sl_hit(94.0)
    assert "1h" in removed
    assert "1h" not in bot._htf_tracker.get_active_setups()
