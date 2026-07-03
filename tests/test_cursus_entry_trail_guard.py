"""#ENTRY-TRAIL-GUARD — Cursus 진입 시 트레일 침범 검사 (7/2 반복진입 사고 회귀).

진입 신호(ST1·ST2 정렬)와 트레일 ST(×6)의 상태가 어긋나면 — 롱인데 트레일이
가격 위 — SL 등록이 불가능한 진입이 되어 [진입→SL거부→비상청산→재진입] 루프로
왕복 수수료를 소진했다(7/2 LINK 17회, 16분 노셔널 -12%, 누적 372건).
주문 전에 침범을 검사해 진입을 보류하는 가드를 검증한다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.bot_trend_instance import BotTrendInstance
from aurora_ict.strategy.silver_bullet import Direction


def _client() -> AsyncMock:
    client = AsyncMock()
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    client.fetch_ticker = AsyncMock(return_value=7.48)
    client.place_order = AsyncMock(return_value={"orderId": "C1"})
    client.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    client.modify_stop_loss = AsyncMock(return_value={"retCode": 0})
    client.fetch_position = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
async def test_open_skips_when_trail_above_price_for_long() -> None:
    """롱 진입인데 트레일이 가격 위(침범) → 주문 자체를 내지 않고 보류."""
    client = _client()
    bot = BotTrendInstance(client=client)

    # 7/2 사고 재현값 — price 7.48, trail 7.5235 (롱인데 SL 이 위).
    await bot._open(Direction.LONG, price=7.48, trail=7.5235)

    client.place_order.assert_not_awaited()
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_open_skips_when_trail_below_price_for_short() -> None:
    """숏 진입인데 트레일이 가격 아래(침범) → 보류."""
    client = _client()
    bot = BotTrendInstance(client=client)

    await bot._open(Direction.SHORT, price=7.48, trail=7.40)

    client.place_order.assert_not_awaited()
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_open_proceeds_when_trail_protective() -> None:
    """정상 케이스 — 롱 + 트레일이 가격 아래(보호 측)면 진입 진행."""
    client = _client()
    bot = BotTrendInstance(client=client)

    await bot._open(Direction.LONG, price=7.48, trail=7.20)

    assert client.place_order.await_count == 1
    assert bot.active_position is not None
    assert bot.active_position.direction is Direction.LONG
