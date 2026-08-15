"""#PAIR7 2026-08-15 — 7페어 복원 + 건당 리스크 절반.

2026-07 에 알트를 뺀 근거("낙폭 85.6% 의 주범")에 두 가지 왜곡이 있었다:
  ① turtle_soup 이 알트에서 특히 나빴다(495건 -0.109R) — 알트 탓으로 오귀속.
     8/11 제거 후 알트 5페어는 1,902건 +0.066R 로 양수다.
  ② 백테가 라이브의 2배를 걸고 있었다(6.3배 고정 vs 실측 3.16배) — 낙폭 과장.
     8/10 사이징 이식으로 교정.

둘을 고치고 다시 재니 7페어가 우세하다:
    BTC+ETH x1.0  월 14건 · 자산 14.12x · 낙폭 48.3% · 파산 0.1%
    7페어  x0.5   월 46건 · 자산 10.69x · 낙폭 54.2% · 파산 0.0%

페어가 3.5배로 늘면 동시 노출도 그만큼 커지므로 **건당 리스크를 절반**으로 낮춰
위험을 현행 수준에 맞춘다. 이 파일은 그 두 가지가 실제로 적용되는지 고정한다.
"""

from __future__ import annotations

import pytest

from aurora_ict.bot.pair_registry import (
    FIXED_PAIRS,
    LEGACY_FIXED_PAIRS,
)
from aurora_ict.config.settings import IctSettings


def test_seven_fixed_pairs() -> None:
    """★ 고정 페어가 7개이고 메이저 2개를 포함한다."""
    assert len(FIXED_PAIRS) == 7
    assert "BTC/USDT:USDT" in FIXED_PAIRS
    assert "ETH/USDT:USDT" in FIXED_PAIRS


def test_no_duplicate_pairs() -> None:
    """중복 없음 — 중복이 있으면 같은 페어가 두 번 기동된다."""
    assert len(set(FIXED_PAIRS)) == len(FIXED_PAIRS)


def test_legacy_empty_after_restore() -> None:
    """복원 후 legacy 는 비어야 한다 — 안 그러면 자동 정합이 방금 켠 페어를 끈다."""
    assert not set(FIXED_PAIRS) & set(LEGACY_FIXED_PAIRS)


@pytest.mark.parametrize("tier", ["sub_30d", "sub_90d", "sub_365d"])
def test_subscription_halves_risk(tier: str) -> None:
    """★ 구독 정책에서 건당 리스크가 절반으로 강제된다."""
    s = IctSettings(license_type=tier)
    assert s.risk_per_trade_base <= 1.5
    assert s.risk_per_trade_max <= 3.0
    assert s.risk_per_trade_step <= 0.75


def test_user_tighter_risk_respected() -> None:
    """사용자가 더 낮게 잡으면 존중한다 — 위로만 막는다."""
    s = IctSettings(license_type="sub_30d", risk_per_trade_base=1.0,
                    risk_per_trade_max=2.0)
    assert s.risk_per_trade_base == 1.0
    assert s.risk_per_trade_max == 2.0


def test_referral_keeps_full_risk() -> None:
    """무료는 기존 크기 유지 — 강제는 구독 정책에만 건다."""
    s = IctSettings(license_type="referral")
    assert s.risk_per_trade_base == 3.0
    assert s.risk_per_trade_max == 6.0
