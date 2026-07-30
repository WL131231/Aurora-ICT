"""#MIN-SIZE — 가용잔고 부족 시 극소액 진입 방지 게이트.

2026-07-30 파트너 지시. 배경: 한 포지션이 증거금 대부분을 먹은 뒤 남은 잔고로
극소액 포지션을 의미 없이 또 잡고 있었다(라이브 실측 최소 notional 0.58 USDT,
일부 유저는 진입의 1/3 이 중앙값 절반 미만).

값 선정(라이브 568건): 하한 20% → 차단 30건(5.3%), 차단분 ROI -0.37% vs 유지분
-0.35%(중립), pnl 기여 0.0%, 차단 대상 notional 중앙 4.0 USDT.
하한 30% 는 차단분 pnl +6.13(흑자)이라 부적합.

검증:
    1. 계획의 20% 미만으로 축소되면 0 반환(진입 포기).
    2. 20% 이상이면 축소된 수량으로 진입(기존 동작 유지).
    3. 가용 충분하면 원 수량 그대로 — 게이트 무영향.
    4. min_qty_ratio=0 이면 비활성(하위 호환).
    5. 판정은 **라운딩 후** 값 기준(거래소 최소단위 내림으로 더 작아질 수 있음).
    6. 가용 조회 실패(-1)면 원 수량 유지(거래소가 최종 판단).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.margin_guard import cap_qty_to_available


def _client(free: float | None, round_step: float | None = None) -> AsyncMock:
    c = AsyncMock()
    if free is None:
        c.fetch_balance = AsyncMock(side_effect=RuntimeError("net"))
    else:
        c.fetch_balance = AsyncMock(return_value={"USDT": {"free": free}})
    if round_step:
        c.round_amount = lambda sym, q: round(q / round_step) * round_step
    else:
        c.round_amount = lambda sym, q: q
    return c


@pytest.mark.asyncio
async def test_blocks_when_below_ratio() -> None:
    """가용 1 USDT — 계획 10.0 중 1.8 만 가능(18%) → 20% 하한 미달로 skip."""
    # max_notional = 1 × 20 × 0.9 = 18 → max_qty = 1.8 (price 10)
    qty = await cap_qty_to_available(_client(1.0), "BTCUSDT", qty=10.0, price=10.0,
                                     leverage=20, min_qty_ratio=0.2)
    assert qty == 0.0


@pytest.mark.asyncio
async def test_allows_at_ratio() -> None:
    """가용 1.2 USDT — 2.16 가능(21.6%) → 하한 통과, 축소 수량으로 진입."""
    qty = await cap_qty_to_available(_client(1.2), "BTCUSDT", qty=10.0, price=10.0,
                                     leverage=20, min_qty_ratio=0.2)
    assert qty == pytest.approx(2.16)


@pytest.mark.asyncio
async def test_no_effect_when_balance_sufficient() -> None:
    """가용 충분 — 원 수량 그대로. 게이트가 정상 진입을 건드리지 않는다."""
    qty = await cap_qty_to_available(_client(1000.0), "BTCUSDT", qty=10.0, price=10.0,
                                     leverage=20, min_qty_ratio=0.2)
    assert qty == 10.0


@pytest.mark.asyncio
async def test_disabled_keeps_legacy() -> None:
    """min_qty_ratio=0 이면 극소액이라도 기존대로 축소 진입(하위 호환)."""
    qty = await cap_qty_to_available(_client(1.0), "BTCUSDT", qty=10.0, price=10.0,
                                     leverage=20, min_qty_ratio=0.0)
    assert qty == pytest.approx(1.8)


@pytest.mark.asyncio
async def test_ratio_judged_after_rounding() -> None:
    """라운딩으로 더 작아지는 경우도 잡는다 — step 1.0 이면 2.16→2.0(20%) 통과,
    step 3.0 이면 2.16→3.0 이 아니라 내림 0.0 이 되어 skip."""
    ok = await cap_qty_to_available(_client(1.2, round_step=1.0), "BTCUSDT",
                                    qty=10.0, price=10.0, leverage=20,
                                    min_qty_ratio=0.2)
    assert ok == pytest.approx(2.0)
    # step 5.0 → round(2.16/5)*5 = 0.0 → 하한 미달
    zero = await cap_qty_to_available(_client(1.2, round_step=5.0), "BTCUSDT",
                                      qty=10.0, price=10.0, leverage=20,
                                      min_qty_ratio=0.2)
    assert zero == 0.0


@pytest.mark.asyncio
async def test_balance_query_failure_keeps_qty() -> None:
    """가용 조회 실패 → 원 수량 유지(거래소가 최종 판단). 게이트도 적용 안 됨."""
    qty = await cap_qty_to_available(_client(None), "BTCUSDT", qty=10.0, price=10.0,
                                     leverage=20, min_qty_ratio=0.2)
    assert qty == 10.0


@pytest.mark.asyncio
async def test_zero_balance_skips() -> None:
    """가용 0 — 기존 동작대로 skip."""
    qty = await cap_qty_to_available(_client(0.0), "BTCUSDT", qty=10.0, price=10.0,
                                     leverage=20, min_qty_ratio=0.2)
    assert qty == 0.0
