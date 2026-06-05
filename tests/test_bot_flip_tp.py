"""HTF flip 재진입 시 거래소에 TP 가 함께 걸리는지 검증 (#FLIP-TP).

기존엔 flip 신규 진입이 ``modify_stop_loss`` 로 SL 만 걸어, TP conditional 이
거래소에 등록되지 않았다 (봇이 죽으면 TP 미실현 + SYNC_CLOSE 분류 오차).
일반 진입과 동일하게 ``_ensure_protective_sl`` 로 SL+TP 를 함께 적용하도록
수정한 것을 회귀 검증한다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
from aurora_ict.indicators.fvg import FVGType
from aurora_ict.strategy.htf_fvg_map import HtfFvgEntry
from aurora_ict.strategy.silver_bullet import Direction


def _client() -> AsyncMock:
    client = AsyncMock()
    # flip 신규 진입 = SHORT. 현재가는 trigger_price 부근.
    client.fetch_ticker = AsyncMock(return_value=99.0)
    client.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    client.place_order = AsyncMock(return_value={"orderId": "F1"})
    client.modify_stop_loss = AsyncMock(return_value={"retCode": 0})
    client.fetch_position = AsyncMock(return_value=None)
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    return client


@pytest.mark.asyncio
async def test_flip_entry_sets_take_profit_on_exchange() -> None:
    """LONG → SHORT flip 시 set_position_tpsl 가 TP 와 함께 호출된다.

    target.high(102) 가 SHORT SL, trigger_price(99) 가 entry → SL 은 entry 위
    (보호 측). _ensure_protective_sl 가 SL+TP 를 set_position_tpsl 로 동시 적용.
    """
    client = _client()
    bot = BotIctInstance(client=client)
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0,
        take_profit=115.0, qty=1.0, setup_ts_ms=0,
    )
    target = HtfFvgEntry(
        tf="1h", weight=10, type=FVGType.BEARISH,
        high=102.0, low=98.0, ts_ms=1000,
    )

    await bot.handle_htf_flip(trigger_price=99.0, ts_ms=2000, target=target)

    # flip 후 SHORT 신규 포지션이 살아있다.
    assert bot.active_position is not None
    assert bot.active_position.direction is Direction.SHORT
    # 거래소에 TP 가 함께 걸렸다 — set_position_tpsl take_profit != None.
    assert client.set_position_tpsl.await_count >= 1
    kw = client.set_position_tpsl.await_args_list[-1].kwargs
    assert kw["take_profit"] is not None
    assert kw["take_profit"] == bot.active_position.take_profit
    # SHORT TP 는 entry 아래여야 한다.
    assert bot.active_position.take_profit < bot.active_position.entry
    # 더 이상 modify_stop_loss(SL 만) 로 보호하지 않는다.
    client.modify_stop_loss.assert_not_awaited()


@pytest.mark.asyncio
async def test_flip_entry_order_omits_stop_loss() -> None:
    """#FLIP-SL: flip 신규 진입 주문에 SL/TP 를 동봉하지 않는다.

    일반 진입(#LIVE-4)과 동일하게, SL 을 진입 주문에 동봉하면 Bybit 10001
    (StopLoss 방향 검증)로 주문 자체가 거부 → 신규 진입 0 가 될 수 있다.
    SL/TP 는 체결 후 set_position_tpsl 로 따로 박는다. 따라서 flip 경로의
    모든 place_order 호출은 stop_loss/take_profit 없이 나가야 한다.
    """
    client = _client()
    bot = BotIctInstance(client=client)
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0,
        take_profit=115.0, qty=1.0, setup_ts_ms=0,
    )
    target = HtfFvgEntry(
        tf="1h", weight=10, type=FVGType.BEARISH,
        high=102.0, low=98.0, ts_ms=1000,
    )

    await bot.handle_htf_flip(trigger_price=99.0, ts_ms=2000, target=target)

    # 신규 진입 주문이 나갔고, 어떤 place_order 호출에도 SL/TP 동봉이 없다.
    assert client.place_order.await_count >= 1
    for call in client.place_order.await_args_list:
        assert call.kwargs.get("stop_loss") is None
        assert call.kwargs.get("take_profit") is None
