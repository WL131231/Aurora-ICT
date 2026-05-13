"""Trailing Extremes — Strong/Weak High & Low (LuxAlgo SMC 패턴)."""

from __future__ import annotations

import pandas as pd
import pytest

from aurora_ict.indicators.structure import (
    StructureEvent,
    StructureType,
    TrendDirection,
    detect_structure_events,
)
from aurora_ict.indicators.swing_points import (
    SwingPoint,
    SwingType,
    detect_swing_points,
)
from aurora_ict.indicators.trailing_extremes import (
    LABEL_HIGH,
    LABEL_LOW,
    LABEL_STRONG_HIGH,
    LABEL_STRONG_LOW,
    LABEL_WEAK_HIGH,
    LABEL_WEAK_LOW,
    compute_trailing_extremes,
)


def _make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = [i * 60_000 for i in range(len(rows))]
    return df


# ============================================================
# Edge cases
# ============================================================


def test_trailing_empty_df_returns_none() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    assert compute_trailing_extremes(df, [], []) is None


def test_trailing_missing_columns_raises() -> None:
    df = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing columns"):
        compute_trailing_extremes(df, [], [])


# ============================================================
# Bias 라벨 검증
# ============================================================


def test_trailing_no_events_default_labels() -> None:
    """structure event 없음 → bias=none → 일반 High / Low 라벨."""
    df = _make_df([
        (100, 105, 99, 104),
        (104, 108, 103, 107),
        (107, 110, 106, 109),
    ])
    te = compute_trailing_extremes(df, [], [])
    assert te is not None
    assert te.top_label == LABEL_HIGH
    assert te.bottom_label == LABEL_LOW
    assert te.top == 110.0
    assert te.bottom == 99.0


def test_trailing_bullish_bias_labels() -> None:
    """BOS_BULLISH → top=Weak High, bottom=Strong Low."""
    df = _make_df([
        (100, 110, 99, 109),
        (109, 115, 108, 114),
    ])
    fake_event = StructureEvent(
        ts_ms=0,
        type=StructureType.BOS_BULLISH,
        broken_level=105.0,
        idx=0,
        broken_swing_idx=0,
        trend_before=TrendDirection.NONE,
    )
    te = compute_trailing_extremes(df, [], [fake_event])
    assert te is not None
    assert te.top_label == LABEL_WEAK_HIGH
    assert te.bottom_label == LABEL_STRONG_LOW


def test_trailing_bearish_bias_labels() -> None:
    """BOS_BEARISH → top=Strong High, bottom=Weak Low."""
    df = _make_df([
        (100, 105, 90, 92),
        (92, 95, 85, 88),
    ])
    fake_event = StructureEvent(
        ts_ms=0,
        type=StructureType.BOS_BEARISH,
        broken_level=95.0,
        idx=0,
        broken_swing_idx=0,
        trend_before=TrendDirection.NONE,
    )
    te = compute_trailing_extremes(df, [], [fake_event])
    assert te is not None
    assert te.top_label == LABEL_STRONG_HIGH
    assert te.bottom_label == LABEL_WEAK_LOW


def test_trailing_choch_bullish_labels() -> None:
    """CHOCH_BULLISH → bullish bias (BOS_BULLISH 와 동일)."""
    df = _make_df([(100, 110, 99, 109)])
    fake_event = StructureEvent(
        ts_ms=0,
        type=StructureType.CHOCH_BULLISH,
        broken_level=100.0,
        idx=0,
        broken_swing_idx=0,
        trend_before=TrendDirection.DOWN,
    )
    te = compute_trailing_extremes(df, [], [fake_event])
    assert te is not None
    assert te.top_label == LABEL_WEAK_HIGH


# ============================================================
# Top/Bottom 추적 — last swing 이후 범위
# ============================================================


def test_trailing_top_after_last_swing_low() -> None:
    """last swing low 이후의 max(high) 만 trailing.top 후보."""
    df = _make_df([
        (100, 120, 99, 119),    # idx=0, high=120 (이건 swing low 박기 전 → 제외)
        (119, 121, 80, 82),     # idx=1, swing low 박힘 (low=80)
        (82, 105, 81, 104),     # idx=2, high=105
        (104, 130, 100, 129),   # idx=3, high=130
        (129, 132, 128, 131),   # idx=4, high=132 (last swing low 이후 max)
    ])
    # 가짜 swing low @ idx=1
    sw_low = SwingPoint(ts_ms=60_000, type=SwingType.LOW, price=80.0, idx=1)
    te = compute_trailing_extremes(df, [sw_low], [])
    assert te is not None
    # max(high) over [idx=1..end] = 132
    assert te.top == 132.0


def test_trailing_bottom_after_last_swing_high() -> None:
    """last swing high 이후의 min(low) 만 trailing.bottom 후보."""
    df = _make_df([
        (100, 105, 50, 52),     # idx=0, low=50 (swing high 박기 전)
        (52, 130, 51, 129),     # idx=1, swing high (high=130)
        (129, 132, 100, 102),   # idx=2, low=100
        (102, 105, 70, 72),     # idx=3, low=70
        (72, 75, 65, 68),       # idx=4, low=65 (last swing high 이후 min)
    ])
    sw_high = SwingPoint(ts_ms=60_000, type=SwingType.HIGH, price=130.0, idx=1)
    te = compute_trailing_extremes(df, [sw_high], [])
    assert te is not None
    # min(low) over [idx=1..end] = 65
    assert te.bottom == 65.0


def test_trailing_no_swings_uses_full_df() -> None:
    """swing 없음 → df 전체 max/min."""
    df = _make_df([
        (100, 120, 99, 119),
        (119, 130, 95, 100),
    ])
    te = compute_trailing_extremes(df, [], [])
    assert te is not None
    assert te.top == 130.0
    assert te.bottom == 95.0


# ============================================================
# Integration — detect_swing_points + detect_structure_events 박은 흐름
# ============================================================


def test_trailing_pipeline_integration() -> None:
    """실제 swing + structure 검출 → trailing extremes 박은 흐름."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 103, 95, 96),
        (96, 110, 95.5, 109),
        (109, 112, 92, 93),
        (93, 100, 92.5, 99),
        (99, 105, 98, 104),
        (104, 106, 100, 101),
    ])
    swings = detect_swing_points(df)
    events = detect_structure_events(df, swings)
    te = compute_trailing_extremes(df, swings, events)
    assert te is not None
    # top / bottom 은 df 안에 있는 값이어야 함
    assert te.top in df["high"].to_numpy()
    assert te.bottom in df["low"].to_numpy()
    # 라벨은 정의된 6종 중 하나
    valid_top = {LABEL_HIGH, LABEL_STRONG_HIGH, LABEL_WEAK_HIGH}
    valid_bot = {LABEL_LOW, LABEL_STRONG_LOW, LABEL_WEAK_LOW}
    assert te.top_label in valid_top
    assert te.bottom_label in valid_bot
