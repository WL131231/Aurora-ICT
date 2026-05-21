"""HTF FVG map 단위 테스트 — find_opposite_htf_fvg / find_supporting_htf_fvg.

변형 7 B+A 합성 검증:
- B (opposite): 반대 방향 HTF FVG 합산 가중치 > LTF 가중치 → flip target 후보
- A (supporting): 같은 방향 HTF FVG 합산 가중치 → confluence_score 보강
"""

from __future__ import annotations

from aurora_ict.indicators.fvg import FVGType
from aurora_ict.strategy.htf_fvg_map import (
    HtfFvgEntry,
    find_opposite_htf_fvg,
    find_supporting_htf_fvg,
)


def _entry(
    tf: str,
    weight: int,
    type_: FVGType,
    low: float,
    high: float,
    touch_count: int = 0,
) -> HtfFvgEntry:
    return HtfFvgEntry(
        tf=tf,
        weight=weight,
        type=type_,
        low=low,
        high=high,
        ts_ms=0,
        touch_count=touch_count,
    )


# =============================================================
# find_supporting_htf_fvg — 변형 7 B+A 의 A 부분
# =============================================================


def test_supporting_long_below_price_matches() -> None:
    """LTF LONG + bullish FVG 가격 아래 → 지지 후보."""
    htf_map = [_entry("1h", 4, FVGType.BULLISH, low=95, high=98)]
    cands = find_supporting_htf_fvg(htf_map, "buy", current_price=100.0)
    assert len(cands) == 1
    assert cands[0].tf == "1h"


def test_supporting_long_above_price_rejected() -> None:
    """LTF LONG + bullish FVG 가격 위 → 엄격 안에선 제외 (지지 역할 X)."""
    htf_map = [_entry("1h", 4, FVGType.BULLISH, low=105, high=108)]
    cands = find_supporting_htf_fvg(htf_map, "buy", current_price=100.0)
    assert cands == []


def test_supporting_long_bearish_rejected() -> None:
    """LTF LONG + bearish FVG → 같은 방향 아니라 제외."""
    htf_map = [_entry("1h", 4, FVGType.BEARISH, low=95, high=98)]
    cands = find_supporting_htf_fvg(htf_map, "buy", current_price=100.0)
    assert cands == []


def test_supporting_short_above_price_matches() -> None:
    """LTF SHORT + bearish FVG 가격 위 → 저항 후보."""
    htf_map = [_entry("4h", 10, FVGType.BEARISH, low=105, high=110)]
    cands = find_supporting_htf_fvg(htf_map, "sell", current_price=100.0)
    assert len(cands) == 1


def test_supporting_short_below_price_rejected() -> None:
    """LTF SHORT + bearish FVG 가격 아래 → 엄격 안에선 제외."""
    htf_map = [_entry("4h", 10, FVGType.BEARISH, low=92, high=95)]
    cands = find_supporting_htf_fvg(htf_map, "sell", current_price=100.0)
    assert cands == []


def test_supporting_touch_count_weakened_excluded() -> None:
    """touch 누적 임계치 도달한 FVG 는 약화로 제외."""
    htf_map = [_entry("1d", 20, FVGType.BULLISH, low=95, high=98, touch_count=3)]
    cands = find_supporting_htf_fvg(htf_map, "buy", current_price=100.0)
    assert cands == []


def test_supporting_sorted_by_distance() -> None:
    """가까운 거리 순 정렬."""
    htf_map = [
        _entry("4h", 10, FVGType.BULLISH, low=80, high=85),   # mid 82.5, dist 17.5
        _entry("1h", 4, FVGType.BULLISH, low=95, high=98),    # mid 96.5, dist 3.5
        _entry("1d", 20, FVGType.BULLISH, low=70, high=75),   # mid 72.5, dist 27.5
    ]
    cands = find_supporting_htf_fvg(htf_map, "buy", current_price=100.0)
    assert [c.tf for c in cands] == ["1h", "4h", "1d"]


# =============================================================
# find_opposite_htf_fvg — 변형 7 B+A 의 B 부분 (회귀 보강)
# =============================================================


def test_opposite_long_threshold_exceeded() -> None:
    """LTF LONG + 위쪽 bearish FVG 합산 가중치 > threshold → 반환."""
    htf_map = [_entry("4h", 10, FVGType.BEARISH, low=105, high=110)]
    cands = find_opposite_htf_fvg(
        htf_map, "buy", current_price=100.0, threshold_weight=4,
    )
    assert len(cands) == 1


def test_opposite_long_threshold_not_exceeded() -> None:
    """합산 가중치 == threshold → 빈 리스트 (strict >)."""
    htf_map = [_entry("15m", 2, FVGType.BEARISH, low=105, high=110)]
    cands = find_opposite_htf_fvg(
        htf_map, "buy", current_price=100.0, threshold_weight=2,
    )
    assert cands == []


def test_opposite_long_below_price_rejected() -> None:
    """LTF LONG + bearish FVG 가격 아래 → 위협 아님 (LTF 진행 방향 아님)."""
    htf_map = [_entry("1d", 20, FVGType.BEARISH, low=85, high=90)]
    cands = find_opposite_htf_fvg(
        htf_map, "buy", current_price=100.0, threshold_weight=4,
    )
    assert cands == []
