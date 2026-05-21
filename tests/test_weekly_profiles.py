"""Weekly Profiles 12종 단위 테스트.

CLAUDE.md mock 0 정책 — 결정론적 OHLC 입력만 (외부 호출 X).
"""
from __future__ import annotations

from aurora_ict.strategy.weekly_profiles import (
    FRI,
    MON,
    THU,
    TUE,
    WED,
    DailyBar,
    WeeklyProfileType,
    classify_weekly_profile,
)


def _bar(wd: int, o: float, h: float, lo: float, c: float) -> DailyBar:
    return DailyBar(weekday=wd, open=o, high=h, low=lo, close=c)


# ============================================================
# I / II — Classic Tuesday Low / High
# ============================================================


def test_classic_tuesday_low_bullish():
    """Tue 가 weekly low + 다른 요일 high → I."""
    bars = [
        _bar(MON, 100, 102, 98, 99),
        _bar(TUE, 99, 100, 90, 95),    # weekly low at Tue
        _bar(WED, 95, 105, 94, 103),
        _bar(THU, 103, 108, 102, 107),
        _bar(FRI, 107, 110, 105, 109),  # weekly high at Fri
    ]
    p = classify_weekly_profile(bars)
    assert p.type is WeeklyProfileType.CLASSIC_TUESDAY_LOW
    assert p.bias == "bullish"
    assert p.weekly_low_day == TUE


def test_classic_tuesday_high_bearish():
    """Tue 가 weekly high → II."""
    bars = [
        _bar(MON, 100, 102, 98, 101),
        _bar(TUE, 101, 110, 100, 105),  # weekly high at Tue
        _bar(WED, 105, 106, 95, 96),
        _bar(THU, 96, 97, 92, 93),
        _bar(FRI, 93, 95, 90, 91),     # weekly low at Fri
    ]
    p = classify_weekly_profile(bars)
    assert p.type is WeeklyProfileType.CLASSIC_TUESDAY_HIGH
    assert p.bias == "bearish"


# ============================================================
# III / IV — Wednesday Low / High
# ============================================================


def test_wednesday_low_bullish():
    """Wed 가 weekly low (Mon-Tue consol 아님) → III."""
    bars = [
        _bar(MON, 100, 105, 95, 102),   # Mon-Tue range 큰 (consol 아님)
        _bar(TUE, 102, 108, 100, 105),
        _bar(WED, 105, 106, 88, 92),    # weekly low at Wed (강한 reversal 아님 — 그냥 약세 close)
        _bar(THU, 92, 95, 90, 94),
        _bar(FRI, 94, 100, 93, 99),
    ]
    p = classify_weekly_profile(bars)
    assert p.type is WeeklyProfileType.WEDNESDAY_LOW
    assert p.bias == "bullish"


def test_wednesday_high_bearish():
    bars = [
        _bar(MON, 100, 105, 95, 102),
        _bar(TUE, 102, 105, 95, 100),
        _bar(WED, 100, 115, 99, 112),   # weekly high at Wed
        _bar(THU, 112, 113, 105, 106),
        _bar(FRI, 106, 108, 100, 102),
    ]
    p = classify_weekly_profile(bars)
    assert p.type is WeeklyProfileType.WEDNESDAY_HIGH
    assert p.bias == "bearish"


# ============================================================
# V / VI — Consolidation Thursday Reversal
# ============================================================


def test_consolidation_thursday_bullish_reversal():
    """Mon-Wed 좁은 횡보 + Thu 가 weekly low 형성 후 강한 reversal up → V."""
    bars = [
        _bar(MON, 100, 101, 99, 100.5),    # 좁은 range
        _bar(TUE, 100.5, 101.5, 99.5, 101),
        _bar(WED, 101, 102, 100, 101.5),
        _bar(THU, 101.5, 105, 95, 104.5),  # weekly low + 강한 bullish close (low 95, close 104.5)
        _bar(FRI, 104.5, 106, 103, 105),
    ]
    p = classify_weekly_profile(bars, consolidation_max_pct=0.03)
    assert p.type is WeeklyProfileType.CONSOLIDATION_THURSDAY_BULLISH
    assert p.bias == "bullish"


def test_consolidation_thursday_bearish_reversal():
    bars = [
        _bar(MON, 100, 101, 99, 100.5),
        _bar(TUE, 100.5, 101.5, 99.5, 101),
        _bar(WED, 101, 102, 100, 101.5),
        _bar(THU, 101.5, 110, 100, 100.5),  # weekly high + 강한 bearish close
        _bar(FRI, 100.5, 102, 98, 99),
    ]
    p = classify_weekly_profile(bars, consolidation_max_pct=0.03)
    assert p.type is WeeklyProfileType.CONSOLIDATION_THURSDAY_BEARISH
    assert p.bias == "bearish"


# ============================================================
# VII / VIII — Consolidation Midweek Rally / Decline
# ============================================================


def test_consolidation_midweek_rally():
    """Mon-Wed consolidation + Thu/Fri up expansion."""
    bars = [
        _bar(MON, 100, 100.5, 99.5, 100),
        _bar(TUE, 100, 100.5, 99.5, 100),
        _bar(WED, 100, 100.5, 99.5, 100),  # midweek avg ~ 100
        _bar(THU, 100, 105, 100, 104.5),    # 위로 5% 확장
        _bar(FRI, 104.5, 108, 104, 107),    # 더 위로
    ]
    p = classify_weekly_profile(bars, midweek_expansion_pct=0.03)
    # Thu/Fri 위로 확장 → midweek rally
    assert p.type is WeeklyProfileType.CONSOLIDATION_MIDWEEK_RALLY
    assert p.bias == "bullish"


def test_consolidation_midweek_decline():
    bars = [
        _bar(MON, 100, 100.5, 99.5, 100),
        _bar(TUE, 100, 100.5, 99.5, 100),
        _bar(WED, 100, 100.5, 99.5, 100),
        _bar(THU, 100, 100, 94, 95),
        _bar(FRI, 95, 96, 92, 93),
    ]
    p = classify_weekly_profile(bars, midweek_expansion_pct=0.03)
    assert p.type is WeeklyProfileType.CONSOLIDATION_MIDWEEK_DECLINE
    assert p.bias == "bearish"


# ============================================================
# IX / X — Seek & Destroy Friday
# ============================================================


def test_seek_and_destroy_bullish_friday():
    """Fri 가 weekly high AND weekly low 둘 다 + close > open → IX."""
    bars = [
        _bar(MON, 100, 101, 99, 100),
        _bar(TUE, 100, 101, 99, 100),
        _bar(WED, 100, 101, 99, 100),
        _bar(THU, 100, 101, 99, 100),
        _bar(FRI, 100, 110, 90, 105),   # Fri 가 양 extremes + bullish close
    ]
    p = classify_weekly_profile(bars)
    assert p.type is WeeklyProfileType.SEEK_AND_DESTROY_BULLISH
    assert p.bias == "bullish"
    assert p.avoid is True


def test_seek_and_destroy_bearish_friday():
    bars = [
        _bar(MON, 100, 101, 99, 100),
        _bar(TUE, 100, 101, 99, 100),
        _bar(WED, 100, 101, 99, 100),
        _bar(THU, 100, 101, 99, 100),
        _bar(FRI, 100, 110, 90, 95),    # 양 extremes + bearish close
    ]
    p = classify_weekly_profile(bars)
    assert p.type is WeeklyProfileType.SEEK_AND_DESTROY_BEARISH
    assert p.avoid is True


# ============================================================
# XI / XII — Wednesday Weekly Reversal
# ============================================================


def test_wednesday_weekly_bullish_reversal():
    """Mon-Tue 좁은 횡보 + Wed 가 weekly low + 강한 bullish reversal."""
    bars = [
        _bar(MON, 100, 100.5, 99.5, 100),
        _bar(TUE, 100, 100.5, 99.5, 100),
        _bar(WED, 100, 102, 90, 101.5),   # low 90 → close 101.5 (강한 reversal)
        _bar(THU, 101.5, 103, 100, 102),
        _bar(FRI, 102, 104, 101, 103),
    ]
    p = classify_weekly_profile(bars, consolidation_max_pct=0.03)
    assert p.type is WeeklyProfileType.WEDNESDAY_WEEKLY_BULLISH_REVERSAL
    assert p.bias == "bullish"


def test_wednesday_weekly_bearish_reversal():
    bars = [
        _bar(MON, 100, 100.5, 99.5, 100),
        _bar(TUE, 100, 100.5, 99.5, 100),
        _bar(WED, 100, 110, 99, 99.5),   # high 110 → close 99.5 (강한 bearish)
        _bar(THU, 99.5, 100, 97, 98),
        _bar(FRI, 98, 99, 96, 97),
    ]
    p = classify_weekly_profile(bars, consolidation_max_pct=0.03)
    assert p.type is WeeklyProfileType.WEDNESDAY_WEEKLY_BEARISH_REVERSAL
    assert p.bias == "bearish"


# ============================================================
# Edge cases
# ============================================================


def test_unclassified_empty_bars():
    """빈 입력 → UNCLASSIFIED."""
    p = classify_weekly_profile([])
    assert p.type is WeeklyProfileType.UNCLASSIFIED
    assert p.bias == "neutral"


def test_unclassified_no_clear_pattern():
    """Mon-Thu 모두 비슷한 range, 어느 요일도 extreme 아님 → UNCLASSIFIED.

    weekly_high_day 가 Mon, weekly_low_day 가 Thu — Tue/Wed 가 extreme 아님.
    consolidation 도 아님 (Mon-Wed range 3% 초과).
    """
    bars = [
        _bar(MON, 95, 110, 90, 100),    # weekly_high=110 at Mon
        _bar(TUE, 100, 105, 95, 102),
        _bar(WED, 102, 108, 98, 105),
        _bar(THU, 105, 107, 85, 95),    # weekly_low=85 at Thu
        # Fri 없음 (Seek & Destroy 미적용)
    ]
    p = classify_weekly_profile(bars, consolidation_max_pct=0.02)
    # Mon high (extreme Mon), Thu low (extreme Thu) — 12 종 중 어디에도 안 맞음
    assert p.type is WeeklyProfileType.UNCLASSIFIED
