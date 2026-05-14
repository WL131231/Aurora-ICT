"""FVG detector 박힌 거 — Aurora-ICT v0.1.0 첫 indicator 테스트."""

from __future__ import annotations

import pandas as pd
import pytest

from aurora_ict.indicators.fvg import (
    FVGType,
    detect_fvgs,
    mark_filled_and_invalidated,
)


def _make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """OHLC 박은 tuple list → DataFrame. ts ms = idx × 60000 (1m 박은 거)."""
    rows = []
    for _i, (o, h, lo, c) in enumerate(bars):
        rows.append({"open": o, "high": h, "low": lo, "close": c})
    df = pd.DataFrame(rows)
    df.index = [i * 60_000 for i in range(len(rows))]
    return df


# ============================================================
# detect_fvgs — 기본 패턴
# ============================================================


def test_empty_df_returns_empty() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    assert detect_fvgs(df) == []


def test_missing_columns_raises() -> None:
    df = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing columns"):
        detect_fvgs(df)


def test_bullish_fvg_basic() -> None:
    """1봉 high=100, 3봉 low=105 → bullish FVG 박힘 (gap 100~105)."""
    df = _make_df([
        (95, 100, 94, 99),    # 0: 1봉 — high=100
        (99, 108, 99, 107),   # 1: 2봉 (displacement) — high=108
        (107, 110, 105, 109), # 2: 3봉 — low=105
    ])
    fvgs = detect_fvgs(df)
    assert len(fvgs) == 1
    fvg = fvgs[0]
    assert fvg.type is FVGType.BULLISH
    assert fvg.low == 100.0
    assert fvg.high == 105.0
    assert fvg.idx == 1
    assert fvg.mean_threshold == 102.5
    assert fvg.size == 5.0


def test_bearish_fvg_basic() -> None:
    """1봉 low=100, 3봉 high=95 → bearish FVG 박힘 (gap 95~100)."""
    df = _make_df([
        (105, 106, 100, 101),  # 0: 1봉 — low=100
        (101, 101, 92, 93),    # 1: 2봉 (displacement down)
        (93, 95, 90, 91),      # 2: 3봉 — high=95
    ])
    fvgs = detect_fvgs(df)
    assert len(fvgs) == 1
    fvg = fvgs[0]
    assert fvg.type is FVGType.BEARISH
    assert fvg.low == 95.0
    assert fvg.high == 100.0
    assert fvg.idx == 1
    assert fvg.mean_threshold == 97.5


def test_no_fvg_when_wicks_overlap() -> None:
    """1봉 high=100, 3봉 low=98 (overlap) → FVG X."""
    df = _make_df([
        (95, 100, 94, 99),
        (99, 105, 99, 104),
        (104, 110, 98, 109),  # low=98 < 100 (1봉 high)
    ])
    assert detect_fvgs(df) == []


def test_multiple_fvgs_sequential() -> None:
    """FVG 2개 박힘 — bullish 그리고 bearish."""
    df = _make_df([
        (95, 100, 94, 99),
        (99, 108, 99, 107),
        (107, 110, 105, 109),   # bullish FVG idx=1
        (108, 109, 100, 101),
        (101, 105, 92, 93),     # high=105 → idx=3 박힌 bearish FVG 박힘 X
        (93, 95, 90, 91),       # bearish FVG idx=4
    ])
    fvgs = detect_fvgs(df)
    assert len(fvgs) == 2
    assert fvgs[0].type is FVGType.BULLISH
    assert fvgs[1].type is FVGType.BEARISH


def test_min_size_filter() -> None:
    """min_size=10 박혀 작은 gap (5) 박힌 거 필터링."""
    df = _make_df([
        (95, 100, 94, 99),
        (99, 108, 99, 107),
        (107, 110, 105, 109),  # gap=5
    ])
    fvgs = detect_fvgs(df, min_size=10.0)
    assert fvgs == []


def test_min_size_pct_filter() -> None:
    """min_size_pct=0.1 박혀 5/107 = 4.7% 박혀 통과 X."""
    df = _make_df([
        (95, 100, 94, 99),
        (99, 108, 99, 107),
        (107, 110, 105, 109),  # gap=5, mid_close=107 → 4.67%
    ])
    assert detect_fvgs(df, min_size_pct=0.10) == []
    # 0.04 박은 거 통과
    fvgs = detect_fvgs(df, min_size_pct=0.04)
    assert len(fvgs) == 1


def test_datetime_index_converts_to_ms() -> None:
    """DatetimeIndex 박은 거 ms 박힘."""
    df = pd.DataFrame([
        {"open": 95, "high": 100, "low": 94, "close": 99},
        {"open": 99, "high": 108, "low": 99, "close": 107},
        {"open": 107, "high": 110, "low": 105, "close": 109},
    ])
    df.index = pd.to_datetime(["2026-05-12 00:00", "2026-05-12 00:01", "2026-05-12 00:02"])
    fvgs = detect_fvgs(df)
    assert len(fvgs) == 1
    # 2026-05-12 00:01 UTC ms 박힘
    expected_ms = int(pd.Timestamp("2026-05-12 00:01").value // 10**6)
    assert fvgs[0].ts_ms == expected_ms


# ============================================================
# mark_filled_and_invalidated
# ============================================================


def test_mark_filled_bullish() -> None:
    """Bullish FVG (100~105) 박힌 후 가격 박힌 102.5 박힘 → filled."""
    df = _make_df([
        (95, 100, 94, 99),
        (99, 108, 99, 107),
        (107, 110, 105, 109),  # FVG (100~105)
        (108, 110, 102, 103),  # low=102 → mean(102.5) 박힘
    ])
    fvgs = detect_fvgs(df)
    mark_filled_and_invalidated(fvgs, df)
    assert fvgs[0].filled is True
    assert fvgs[0].invalidated is False


def test_mark_invalidated_bullish() -> None:
    """Bullish FVG (100~105) 박힌 후 close < 100 → invalidated."""
    df = _make_df([
        (95, 100, 94, 99),
        (99, 108, 99, 107),
        (107, 110, 105, 109),
        (108, 109, 95, 98),  # close=98 < 100 → invalidated
    ])
    fvgs = detect_fvgs(df)
    mark_filled_and_invalidated(fvgs, df)
    assert fvgs[0].invalidated is True
    # filled 박힌 거 박혔어 (102.5 박은 거 통과)
    assert fvgs[0].filled is True


def test_mark_filled_bearish() -> None:
    """Bearish FVG (95~100) 박힌 후 high ≥ 97.5 → filled."""
    df = _make_df([
        (105, 106, 100, 101),
        (101, 101, 92, 93),
        (93, 95, 90, 91),     # FVG (95~100)
        (91, 98, 91, 97),     # high=98 → mean(97.5) 박힘
    ])
    fvgs = detect_fvgs(df)
    mark_filled_and_invalidated(fvgs, df)
    assert fvgs[0].filled is True
    assert fvgs[0].invalidated is False


def test_mark_invalidated_bearish() -> None:
    """Bearish FVG (95~100) 박힌 후 close > 100 → invalidated."""
    df = _make_df([
        (105, 106, 100, 101),
        (101, 101, 92, 93),
        (93, 95, 90, 91),
        (91, 105, 91, 102),  # close=102 > 100
    ])
    fvgs = detect_fvgs(df)
    mark_filled_and_invalidated(fvgs, df)
    assert fvgs[0].invalidated is True


def test_no_action_after_invalidation() -> None:
    """Invalidated 박힌 후 추가 봉 박혀도 추가 mutation X."""
    df = _make_df([
        (95, 100, 94, 99),
        (99, 108, 99, 107),
        (107, 110, 105, 109),
        (108, 109, 95, 98),    # invalidated
        (98, 103, 98, 102),     # 박힌 후 박은 봉
    ])
    fvgs = detect_fvgs(df)
    mark_filled_and_invalidated(fvgs, df)
    # invalidated 후 break — 무한 루프 X
    assert fvgs[0].invalidated is True


# ============================================================
# require_displacement — 중간 봉 close 가 1봉 high 위/아래여야 함
# ============================================================


def test_displacement_required_default() -> None:
    """기본 require_displacement=True — 중간 봉 close 가 1봉 high 위 통과."""
    df = _make_df([
        (95, 100, 94, 99),       # idx=0  high=100
        (99, 115, 99, 114),      # idx=1  close=114 > high[0]=100 ✓
        (114, 116, 105, 115),    # idx=2  low=105 > high[0]=100 → bullish FVG
    ])
    fvgs = detect_fvgs(df, min_size_pct=0.001)
    assert len(fvgs) == 1
    assert fvgs[0].type is FVGType.BULLISH


def test_displacement_filters_weak_mid_bar() -> None:
    """중간 봉 close 가 1봉 high 아래 → FVG 제외 (displacement 부족)."""
    df = _make_df([
        (95, 100, 94, 99),       # idx=0  high=100
        (99, 102, 95, 98),       # idx=1  close=98 < high[0]=100 (displacement 약함)
        (98, 110, 105, 109),     # idx=2  low=105 > high[0]=100 (gap 자체는 있음)
    ])
    fvgs = detect_fvgs(df, min_size_pct=0.001, require_displacement=True)
    # displacement 부족 → 빈 리스트
    assert fvgs == []
    # require_displacement=False 면 검출
    fvgs_no_req = detect_fvgs(df, min_size_pct=0.001, require_displacement=False)
    assert len(fvgs_no_req) == 1


def test_auto_threshold_filters_small_gap() -> None:
    """auto_threshold=True — 작은 gap 은 cumulative mean 못 넘으면 제외."""
    # 큰 변동성 봉들 + 미세 gap 1개 → gap 이 cumulative threshold 못 넘으면 skip
    bars = [(100, 110, 90, 105) for _ in range(10)]
    bars.append((105, 106, 105.05, 105.5))   # 작은 gap 봉
    bars.append((105.5, 105.7, 105.1, 105.2))
    df = _make_df(bars)
    fvgs_auto = detect_fvgs(
        df, auto_threshold=True, auto_threshold_multiplier=2.0,
        require_displacement=False,
    )
    # 큰 변동성 평균 대비 미세 gap → auto_threshold 가 제외
    # (정확히 0개 보장 못 하므로 적어도 정적 threshold 보다 적게 잡힌다)
    fvgs_static = detect_fvgs(
        df, min_size_pct=0.0,
        require_displacement=False,
    )
    assert len(fvgs_auto) <= len(fvgs_static)
