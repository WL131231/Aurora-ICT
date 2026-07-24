"""#MARGIN-GUARD 2026-07-24: 진입 전 가용잔고 체크 — 수량 축소/skip 테스트.

여러 페어 동시 운용 시 한 포지션이 증거금 대부분을 먹어 다음 진입이 '잔고부족'으로
거부되던 문제 → 진입 전 가용잔고로 수량을 캡. mock 0 — 합성 client.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.margin_guard import available_usdt, cap_qty_to_available


def _client(free: float | None) -> AsyncMock:
    c = AsyncMock()
    bal = {"USDT": {"total": (free or 0) + 100, "free": free}} if free is not None else {}
    c.fetch_balance = AsyncMock(return_value=bal)
    # round_amount: lot step 0.001 내림.
    c.round_amount = lambda sym, q: float(int(q * 1000) / 1000)
    return c


@pytest.mark.asyncio
async def test_available_usdt_parses_free() -> None:
    assert await available_usdt(_client(50.0)) == 50.0
    # 조회 실패/미상 → -1 (캡 skip 신호)
    bad = AsyncMock()
    bad.fetch_balance = AsyncMock(side_effect=RuntimeError("x"))
    assert await available_usdt(bad) == -1.0


@pytest.mark.asyncio
async def test_cap_keeps_qty_when_enough_balance() -> None:
    """가용 충분 → 원 수량 유지."""
    # 가용 100, lev 10, price 100 → max_notional=900, max_qty=9. qty 1 ≤ 9 → 유지.
    q = await cap_qty_to_available(_client(100.0), "BTC/USDT:USDT", 1.0, 100.0, 10)
    assert q == 1.0


@pytest.mark.asyncio
async def test_cap_reduces_qty_when_low_balance() -> None:
    """가용 부족 → 수량 축소(가용×lev×0.9/price)."""
    # 가용 10, lev 10, price 100 → max_notional=90, max_qty=0.9. qty 5 → 0.9 로 축소.
    q = await cap_qty_to_available(_client(10.0), "BTC/USDT:USDT", 5.0, 100.0, 10)
    assert q == pytest.approx(0.9, abs=0.001)
    assert q < 5.0


@pytest.mark.asyncio
async def test_cap_returns_zero_when_below_min() -> None:
    """가용 극소 → 축소분이 lot step 미달 → 0 (호출부 skip)."""
    # 가용 0.01, lev 10, price 100 → max_qty=0.0009 → round_amount(0.001내림) → 0.
    q = await cap_qty_to_available(_client(0.01), "BTC/USDT:USDT", 5.0, 100.0, 10)
    assert q == 0.0


@pytest.mark.asyncio
async def test_cap_skips_when_balance_fetch_fails() -> None:
    """가용 조회 실패(-1) → 원 수량 유지(거래소가 최종 판단)."""
    bad = AsyncMock()
    bad.fetch_balance = AsyncMock(side_effect=RuntimeError("x"))
    q = await cap_qty_to_available(bad, "BTC/USDT:USDT", 5.0, 100.0, 10)
    assert q == 5.0
