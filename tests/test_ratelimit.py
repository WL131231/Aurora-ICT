"""RateLimiter — 경량 in-memory sliding-window rate limiter 단위 테스트 (mock 0).

now 를 명시 주입해 시간 의존 없이 결정론적으로 검증한다.

담당: 지영민 (보안 강화 PR)
"""

from __future__ import annotations

import pytest

from aurora_ict.auth.ratelimit import RateLimiter


def test_allows_under_limit() -> None:
    """한도 내 요청은 모두 허용."""
    rl = RateLimiter(max_hits=3, window_sec=60.0)
    assert rl.allow("k", now=0.0)
    assert rl.allow("k", now=1.0)
    assert rl.allow("k", now=2.0)


def test_blocks_over_limit() -> None:
    """한도 초과 시 차단(False)."""
    rl = RateLimiter(max_hits=3, window_sec=60.0)
    for i in range(3):
        assert rl.allow("k", now=float(i))
    assert not rl.allow("k", now=3.0)  # 4번째 차단
    assert not rl.allow("k", now=3.5)  # 계속 차단


def test_window_slides() -> None:
    """window 가 지나 만료된 hit 은 한도에서 빠지고 다시 허용."""
    rl = RateLimiter(max_hits=2, window_sec=10.0)
    assert rl.allow("k", now=0.0)
    assert rl.allow("k", now=1.0)
    assert not rl.allow("k", now=2.0)  # 0.0,1.0 둘 다 window 내 → 초과
    # now=11.5 → window_start=1.5, 0.0·1.0 만료(차단된 2.0 은 미적립) → 허용
    assert rl.allow("k", now=11.5)


def test_keys_isolated() -> None:
    """서로 다른 key 는 독립적으로 카운트."""
    rl = RateLimiter(max_hits=1, window_sec=60.0)
    assert rl.allow("a", now=0.0)
    assert rl.allow("b", now=0.0)  # 다른 키 — 영향 없음
    assert not rl.allow("a", now=1.0)  # a 는 한도 소진
    assert not rl.allow("b", now=1.0)


def test_blocked_request_not_counted() -> None:
    """차단된 요청은 적립되지 않아, 만료 후 정상 한도가 복구된다."""
    rl = RateLimiter(max_hits=1, window_sec=10.0)
    assert rl.allow("k", now=0.0)
    assert not rl.allow("k", now=1.0)  # 차단(미적립)
    assert not rl.allow("k", now=2.0)  # 차단(미적립)
    # now=10.5 → 0.0 만료, 차단분은 애초에 미적립 → 허용
    assert rl.allow("k", now=10.5)


def test_invalid_params() -> None:
    """잘못된 생성 인자는 ValueError."""
    with pytest.raises(ValueError):
        RateLimiter(max_hits=0, window_sec=60.0)
    with pytest.raises(ValueError):
        RateLimiter(max_hits=1, window_sec=0.0)
    with pytest.raises(ValueError):
        RateLimiter(max_hits=1, window_sec=-5.0)


def test_gc_does_not_break_active_keys() -> None:
    """lazy GC 가 돌아도 활성 key 의 카운트는 보존된다."""
    rl = RateLimiter(max_hits=2, window_sec=10.0)
    assert rl.allow("k", now=0.0)
    # window*10=100 경과 후 호출 → GC 트리거. 직전 hit(0.0)은 만료라 제거되지만
    # 이 호출 자체는 새 hit 으로 적립되어 허용.
    assert rl.allow("k", now=200.0)
    assert rl.allow("k", now=200.5)
    assert not rl.allow("k", now=201.0)  # 200.0,200.5 두 건 → 초과
