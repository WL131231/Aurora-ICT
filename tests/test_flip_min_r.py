"""#FLIP-MIN-R (Origo 2.3) 검증 — flip 조기 절단 방지 게이트.

2026-07-30 근거:
    라이브 실측(flip 29건) flip 은 평균 **+0.61R** 에서 익절을 잘랐다(86%가 1R 미만).
    청산 후 실제 가격 추적 반사실 = 2R TP 선착 72% → 기대값 0.61R → 1.17R.
    라이브 정합 백테(126건)에서 최소 1.5R 이 net +239%, RR 1.20→1.94, MDD 764→533.

검증 항목:
    1. 이익이 최소 R 미만이면 flip 이 **실행되지 않는다**(주문 0건).
    2. 최소 R 이상이면 기존대로 flip 이 실행된다(회귀 없음).
    3. flip_min_r=0 이면 게이트 비활성 — 기존 동작 유지.
    4. R 계산은 **진입 시점 risk** 기준이라 trail/BE 로 SL 이 이동해도 흔들리지 않는다.
    5. 게이트가 막을 때 `_flip_done_for_ts` 를 세우지 않아 다음 tick 에 재평가된다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
from aurora_ict.indicators.fvg import FVGType
from aurora_ict.strategy.htf_fvg_map import HtfFvgEntry
from aurora_ict.strategy.silver_bullet import Direction


def _client() -> AsyncMock:
    c = AsyncMock()
    c.fetch_ticker = AsyncMock(return_value=101.0)
    c.place_order = AsyncMock(return_value={"orderId": "X1"})
    c.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    c.modify_stop_loss = AsyncMock(return_value={"retCode": 0})
    c.fetch_position = AsyncMock(return_value=None)
    c.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    return c


def _bot(min_r: float) -> BotIctInstance:
    return BotIctInstance(client=_client(), symbol="BTCUSDT", flip_min_r=min_r,
                          partial_tp_rr=1.5)


def _pos(entry: float = 100.0, sl: float = 98.0, tp1: float = 103.0,
         direction: Direction = Direction.LONG) -> _ActivePosition:
    """롱 entry 100 / 초기 risk 2.0 (tp1 103 = entry + 1.5R → risk 2.0)."""
    return _ActivePosition(
        direction=direction, entry=entry, stop_loss=sl, take_profit=entry + 15,
        qty=1.0, setup_ts_ms=0, tp1_price=tp1,
    )


def _target() -> HtfFvgEntry:
    return HtfFvgEntry(tf="1h", weight=4, type=FVGType.BEARISH,
                       high=105.0, low=101.0, ts_ms=0)


def test_r_from_initial_risk_not_moved_sl() -> None:
    """R 은 tp1_price 역산(진입 risk) 기준 — trail 로 SL 이 올라가도 불변."""
    bot = _bot(1.5)
    pos = _pos()                      # risk 2.0
    assert bot._flip_profit_r(pos, 101.0) == pytest.approx(0.5)   # +1.0 / 2.0
    assert bot._flip_profit_r(pos, 103.0) == pytest.approx(1.5)
    pos.stop_loss = 100.0             # BE 이동 — R 기준이 흔들리면 안 된다
    assert bot._flip_profit_r(pos, 103.0) == pytest.approx(1.5)


def test_r_short_direction() -> None:
    """숏은 가격 하락이 이익 — 부호 정규화 확인."""
    bot = _bot(1.5)
    pos = _pos(entry=100.0, sl=102.0, tp1=97.0, direction=Direction.SHORT)
    assert bot._flip_profit_r(pos, 97.0) == pytest.approx(1.5)
    assert bot._flip_profit_r(pos, 99.0) == pytest.approx(0.5)


def test_r_fallback_when_no_tp1() -> None:
    """tp1_price 없으면 현재 stop_loss 로 근사(0 나눗셈 방지)."""
    bot = _bot(1.5)
    pos = _pos(tp1=0.0)              # risk = |100-98| = 2.0
    assert bot._flip_profit_r(pos, 102.0) == pytest.approx(1.0)
    pos.stop_loss = 100.0            # risk 0 → 0.0 반환(게이트가 통과 못 시킴)
    assert bot._flip_profit_r(pos, 102.0) == 0.0


@pytest.mark.asyncio
async def test_gate_blocks_below_min_r() -> None:
    """0.5R 에서 flip 시도 → 차단. 주문 0건 + flag 미설정(다음 tick 재평가)."""
    bot = _bot(1.5)
    bot.active_position = _pos()
    await bot.handle_htf_flip(trigger_price=101.0, ts_ms=1_000, target=_target())
    bot.client.place_order.assert_not_awaited()
    assert bot.active_position is not None          # 포지션 유지
    assert bot._flip_done_for_ts != 1_000           # flag 안 세움 → 재평가 가능


@pytest.mark.asyncio
async def test_gate_allows_at_min_r() -> None:
    """1.5R 도달 시 flip 실행 — 기존 경로 회귀 없음."""
    bot = _bot(1.5)
    bot.active_position = _pos()
    await bot.handle_htf_flip(trigger_price=103.0, ts_ms=2_000, target=_target())
    bot.client.place_order.assert_awaited()         # 청산 주문 발생
    assert bot._flip_done_for_ts == 2_000


@pytest.mark.asyncio
async def test_gate_disabled_keeps_legacy() -> None:
    """flip_min_r=0 이면 이익 0.5R 이라도 기존대로 flip (하위 호환)."""
    bot = _bot(0.0)
    bot.active_position = _pos()
    await bot.handle_htf_flip(trigger_price=101.0, ts_ms=3_000, target=_target())
    bot.client.place_order.assert_awaited()
