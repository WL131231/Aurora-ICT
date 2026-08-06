"""#CLOSE-500 — Cursus 포지션을 UI 에서 청산할 때 500 이 나던 문제.

2026-08-06 파트너 제보(스크린샷): Cursus 1.0 으로 HYPE 숏을 들고 있는 상태에서
CLOSE 50% / CLOSE ALL 을 누르면 **500 Internal Server Error**.

원인: `/ict/position/close` 가 `bot._exchange_position_direction(ex_pos)` 를 직접
부르는데 이 헬퍼가 **Origo(BotIctInstance)에만** 있었다. Cursus 는 같은 판정을
인라인으로 하고 있어 메서드가 없었고, AttributeError 가 잡히지 않아 500 이 됐다.
→ 사용자가 UI 로 포지션을 정리할 수 없는 상태였다.

같이 고친 것: Cursus 의 인라인 방향 판정은 인식 실패를 **SHORT 로 단정**했다.
    direction = LONG if side in ("long","buy") else SHORT   # side="" 면 SHORT
거래소 응답에 side 가 비면 롱 포지션을 숏으로 입양해 **진입가 위에 SL** 을 걸 수
있다. 헬퍼는 인식 실패 시 None 을 주고, 복원은 입양을 보류한다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.bot.bot_trend_instance import BotTrendInstance
from aurora_ict.strategy.silver_bullet import Direction


def _client() -> AsyncMock:
    c = AsyncMock()
    c.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    c.fetch_ticker = AsyncMock(return_value=100.0)
    c.place_order = AsyncMock(return_value={"orderId": "X1"})
    c.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    c.modify_stop_loss = AsyncMock(return_value={"retCode": 0})
    c.fetch_position = AsyncMock(return_value=None)
    c.cancel_bot_orders = AsyncMock(return_value=0)
    return c


# ---- 500 의 직접 원인 ----

def test_cursus_has_direction_helper() -> None:
    """★ Cursus 에 헬퍼가 있어야 한다 — 없어서 청산이 500 이었다."""
    assert callable(getattr(BotTrendInstance, "_exchange_position_direction", None))


@pytest.mark.parametrize(
    ("side", "expect"),
    [
        ("long", Direction.LONG), ("buy", Direction.LONG),
        ("short", Direction.SHORT), ("sell", Direction.SHORT),
        ("LONG", Direction.LONG), ("Short", Direction.SHORT),
        ("", None), ("weird", None), (None, None),
    ],
)
def test_direction_parsing(side, expect) -> None:
    """방향 판정 — 인식 실패는 반드시 None(임의 방향으로 단정 금지)."""
    assert BotTrendInstance._exchange_position_direction({"side": side}) is expect


def test_both_bots_agree() -> None:
    """Origo 와 Cursus 가 같은 입력에 같은 답을 낸다(청산 경로가 공유하는 계약)."""
    for side in ("long", "buy", "short", "sell", "", "junk"):
        pos = {"side": side}
        assert (BotTrendInstance._exchange_position_direction(pos)
                is BotIctInstance._exchange_position_direction(pos))


def test_empty_dict_does_not_raise() -> None:
    """빈 응답에도 예외 없이 None — 500 의 재발 방지."""
    assert BotTrendInstance._exchange_position_direction({}) is None


# ---- 부수적으로 고친 위험 ----

@pytest.mark.asyncio
async def test_recover_skips_when_direction_unknown() -> None:
    """★ 방향을 모르면 입양하지 않는다.

    예전엔 side 가 비면 SHORT 로 단정 → 롱 포지션에 숏 기준 SL(진입가 위)을
    걸 수 있었다.
    """
    c = _client()
    c.fetch_position = AsyncMock(return_value={
        "contracts": 1.0, "side": "", "entryPrice": 100.0,
    })
    bot = BotTrendInstance(client=c)

    await bot._recover_position_from_exchange(record=False)

    assert bot.active_position is None


@pytest.mark.asyncio
async def test_recover_adopts_long_correctly() -> None:
    """정상 응답은 그대로 입양 — 롱이면 SL 이 진입가 **아래**."""
    c = _client()
    c.fetch_position = AsyncMock(return_value={
        "contracts": 1.0, "side": "long", "entryPrice": 100.0,
    })
    bot = BotTrendInstance(client=c)

    await bot._recover_position_from_exchange(record=False)

    pos = bot.active_position
    assert pos is not None
    assert pos.direction is Direction.LONG
    assert pos.stop < pos.entry
