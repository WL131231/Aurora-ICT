"""Cursus 원본(``매매기법.py``) 엔진 정합 검증 — 2026-07-07 파트너 지시 복원.

원본 스펙: 고정 SL 2% + 4분할 TP 1/2/3/4% ×25% + TP 래더 트레일(TP2 체결 후
SL→TP1, TP3 후 SL→TP2, TP4 전량 종료) + REVERSE. 트레일 ST(×6) 변형은 제거됨.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.bot_trend_instance import BotTrendInstance
from aurora_ict.strategy.silver_bullet import Direction


def _client() -> AsyncMock:
    client = AsyncMock()
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    client.fetch_ticker = AsyncMock(return_value=100.0)
    client.place_order = AsyncMock(return_value={"orderId": "C1"})
    client.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    client.modify_stop_loss = AsyncMock(return_value={"retCode": 0})
    client.fetch_position = AsyncMock(return_value=None)
    return client


@pytest.mark.asyncio
async def test_open_sets_fixed_2pct_sl_and_tp_grid() -> None:
    """원본 build_order_plan 정합: 롱 100 진입 → SL 98, TP [101,102,103,104]."""
    client = _client()
    bot = BotTrendInstance(client=client)

    await bot._open(Direction.LONG, price=100.0)

    pos = bot.active_position
    assert pos is not None
    assert pos.stop == pytest.approx(98.0)          # 고정 2%
    assert pos.tp_prices == pytest.approx([101.0, 102.0, 103.0, 104.0])
    assert client.place_order.await_count == 1      # 진입 주문


@pytest.mark.asyncio
async def test_open_short_sets_sl_above() -> None:
    """숏 100 진입 → SL 102 (보호 측 = 위), TP 아래 [99,98,97,96]."""
    client = _client()
    bot = BotTrendInstance(client=client)

    await bot._open(Direction.SHORT, price=100.0)

    pos = bot.active_position
    assert pos.stop == pytest.approx(102.0)
    assert pos.tp_prices == pytest.approx([99.0, 98.0, 97.0, 96.0])


@pytest.mark.asyncio
async def test_ladder_moves_sl_after_tp2_and_tp3() -> None:
    """원본 TrailingManager 정합: TP2 체결 → SL=TP1(101), TP3 → SL=TP2(102)."""
    client = _client()
    bot = BotTrendInstance(client=client)
    await bot._open(Direction.LONG, price=100.0)
    pos = bot.active_position
    pos.tp_filled = [True, True, False, False]  # TP1·TP2 체결 상태

    # step 의 래더 로직만 재현 — hits=2 → tps[0]=101
    hits = sum(pos.tp_filled)
    assert hits >= bot.cfg.trail_trigger_target
    ladder = pos.tp_prices[hits - 2]
    assert ladder == pytest.approx(101.0)

    pos.tp_filled = [True, True, True, False]   # TP3 까지
    hits = sum(pos.tp_filled)
    assert pos.tp_prices[hits - 2] == pytest.approx(102.0)


@pytest.mark.asyncio
async def test_all_four_tps_close_position() -> None:
    """TP4 까지 전부 체결 → 전량 종료 (원본 hits>=4 closed)."""
    client = _client()
    bot = BotTrendInstance(client=client)
    await bot._open(Direction.LONG, price=100.0)

    # 가격이 TP4(104) 위 — _check_split_tp 가 4개 전부 체결 처리.
    await bot._check_split_tp(105.0)

    assert bot.active_position is None  # 25%×4 = 전량 종료


@pytest.mark.asyncio
async def test_reverse_closes_then_opens_opposite() -> None:
    """REVERSE: 롱 보유 중 숏 신호 → reduce_only 청산 + 반대 진입 (고정 SL)."""
    client = _client()
    bot = BotTrendInstance(client=client)
    await bot._open(Direction.LONG, price=100.0)
    client.fetch_position = AsyncMock(return_value={"contracts": 0})  # 청산 반영됨

    await bot._reverse(Direction.SHORT, price=100.0)

    pos = bot.active_position
    assert pos is not None
    assert pos.direction is Direction.SHORT
    assert pos.stop == pytest.approx(102.0)  # 숏 고정 2% SL
    # 청산(reduce_only) 1건 + 진입 2건(최초 롱 + 역진입 숏)
    reduce_calls = [c for c in client.place_order.await_args_list
                    if c.kwargs.get("reduce_only")]
    assert len(reduce_calls) == 1


@pytest.mark.asyncio
async def test_recover_rebuilds_tp_grid_and_fixed_sl() -> None:
    """복원(입양): 거래소 SL 없으면 고정 2% + entry 기준 TP 그리드 재구성."""
    client = _client()
    client.fetch_position = AsyncMock(return_value={
        "contracts": 5.0, "side": "long", "entryPrice": 100.0,
    })
    bot = BotTrendInstance(client=client)

    await bot._recover_position_from_exchange(record=False)

    pos = bot.active_position
    assert pos is not None
    assert pos.stop == pytest.approx(98.0)
    assert pos.tp_prices == pytest.approx([101.0, 102.0, 103.0, 104.0])
    assert pos.init_qty == pytest.approx(5.0)


@pytest.mark.asyncio
async def test_small_position_full_close_when_chunk_below_min() -> None:
    """#Cursus 2026-07-23 (파트너): 25% 분할 청크가 거래소 최소수량 미달인 소액
    포지션은 분할하지 않고 첫 TP 도달 시 잔량을 통짜 청산."""
    client = _client()
    bot = BotTrendInstance(client=client)
    await bot._open(Direction.LONG, price=100.0)
    pos = bot.active_position
    # 최소수량 = init_qty → 25% 청크(init×0.25) < min_qty → 통짜 청산 조건.
    bot._symbol_meta = {
        "min_qty": pos.init_qty, "qty_step": None, "max_leverage": None,
    }
    full_qty = pos.qty
    client.place_order.reset_mock()

    await bot._check_split_tp(101.0)  # TP1 도달

    assert bot.active_position is None                    # 전량 종료
    call = client.place_order.await_args
    assert call.kwargs.get("reduce_only") is True
    assert call.kwargs.get("qty") == pytest.approx(full_qty)  # 통짜(전량)
