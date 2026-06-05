"""PairRegistry — 거래 가능 페어 화이트리스트 + TTL 캐시 (mock 0).

now 주입으로 시간 의존 없이 결정론적으로 검증. source(거래소 조회)는 메서드
인자로 주입한다.

담당: 지영민 (페어 확장 PR — 화이트리스트 동적화)
"""

from __future__ import annotations

import pytest

from aurora_ict.bot.pair_registry import MAJOR_PAIRS, PairRegistry


class _FakeSource:
    def __init__(self, pairs: list[str], *, fail: bool = False) -> None:
        self._pairs = pairs
        self.fail = fail
        self.calls = 0

    async def list_top_usdt_perps(self, limit: int = 30) -> list[str]:
        self.calls += 1
        return [] if self.fail else self._pairs[:limit]


@pytest.mark.asyncio
async def test_get_allowed_fetches_and_caches() -> None:
    src = _FakeSource(["BTC/USDT:USDT", "SOL/USDT:USDT", "ETH/USDT:USDT"])
    reg = PairRegistry(limit=30, ttl_sec=100)
    r1 = await reg.get_allowed(src, now=0.0)
    assert "SOL/USDT:USDT" in r1
    # TTL 내 재호출은 캐시 사용 — 거래소 조회 안 함.
    await reg.get_allowed(src, now=50.0)
    assert src.calls == 1
    # TTL 만료 → 재조회.
    await reg.get_allowed(src, now=200.0)
    assert src.calls == 2


@pytest.mark.asyncio
async def test_majors_always_included() -> None:
    src = _FakeSource(["SOL/USDT:USDT", "DOGE/USDT:USDT"])  # majors 미포함
    reg = PairRegistry()
    r = await reg.get_allowed(src, now=0.0)
    assert "BTC/USDT:USDT" in r
    assert "ETH/USDT:USDT" in r
    assert "SOL/USDT:USDT" in r


@pytest.mark.asyncio
async def test_fetch_failure_keeps_existing_cache() -> None:
    src = _FakeSource(["SOL/USDT:USDT"])
    reg = PairRegistry(ttl_sec=10)
    r1 = await reg.get_allowed(src, now=0.0)
    assert "SOL/USDT:USDT" in r1
    # 조회 실패 + TTL 만료 → 기존 캐시 유지.
    src.fail = True
    r2 = await reg.get_allowed(src, now=100.0)
    assert "SOL/USDT:USDT" in r2


@pytest.mark.asyncio
async def test_initial_cache_is_majors() -> None:
    src = _FakeSource([], fail=True)
    reg = PairRegistry()
    # 조회 실패해도 초기 캐시(메이저)는 보장.
    r = await reg.get_allowed(src, now=0.0)
    assert set(MAJOR_PAIRS).issubset(set(r))


@pytest.mark.asyncio
async def test_is_allowed() -> None:
    src = _FakeSource(["SOL/USDT:USDT"])
    reg = PairRegistry()
    assert await reg.is_allowed(src, "SOL/USDT:USDT", now=0.0)
    assert await reg.is_allowed(src, "BTC/USDT:USDT", now=0.0)
    assert not await reg.is_allowed(src, "NOPE/USDT:USDT", now=0.0)


@pytest.mark.asyncio
async def test_limit_passed_to_source() -> None:
    src = _FakeSource([f"C{i}/USDT:USDT" for i in range(50)])
    reg = PairRegistry(limit=10)
    r = await reg.get_allowed(src, now=0.0)
    # 상위 10 + 메이저 2 = 최대 12 (메이저가 상위 10에 없으면 추가).
    assert len([s for s in r if s.startswith("C")]) == 10
