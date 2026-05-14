"""Swing / Liquidity / Structure detector — Aurora-ICT v0.1.x 박힌 거."""

from __future__ import annotations

import pandas as pd
import pytest

from aurora_ict.indicators.liquidity import (
    SweepType,
    detect_equal_levels,
    detect_liquidity_sweeps,
)
from aurora_ict.indicators.structure import (
    StructureType,
    TrendDirection,
    detect_structure_events,
)
from aurora_ict.indicators.swing_points import (
    SwingType,
    detect_swing_points,
)


def _make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = [i * 60_000 for i in range(len(rows))]
    return df


# ============================================================
# Swing Points
# ============================================================


def test_swing_empty_df() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    assert detect_swing_points(df) == []


def test_swing_basic_high_low() -> None:
    """3봉 박힌 거 박힌 swing high 박힘."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 108, 100, 107),   # swing high (108)
        (107, 105, 100, 102),
        (102, 103, 95, 96),     # swing low (95)
        (96, 100, 95.5, 99),
    ])
    swings = detect_swing_points(df)
    types = [(s.type, s.price, s.idx) for s in swings]
    assert (SwingType.HIGH, 108.0, 1) in types
    assert (SwingType.LOW, 95.0, 3) in types


def test_swing_equal_high_not_pivot() -> None:
    """양옆 봉 박힌 거 박힌 high 박힌 거랑 동일 박힘 → swing X (strict >)."""
    df = _make_df([
        (100, 105, 99, 101),
        (101, 105, 100, 102),  # high 박힘 동일 (105) → swing X
        (102, 104, 99, 100),
    ])
    swings = detect_swing_points(df)
    assert swings == []


def test_swing_n_bar_window() -> None:
    """left=2, right=2 박힘 — 5봉 swing 박힘."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 110, 101, 108),  # swing high (110, 5봉 안 박힌 max)
        (108, 105, 100, 102),
        (102, 104, 99, 100),
    ])
    swings = detect_swing_points(df, left=2, right=2)
    assert len(swings) == 1
    assert swings[0].type is SwingType.HIGH
    assert swings[0].price == 110.0


def test_swing_missing_columns() -> None:
    df = pd.DataFrame({"open": [1, 2, 3]})
    with pytest.raises(ValueError, match="missing columns"):
        detect_swing_points(df)


# ============================================================
# Liquidity Sweeps
# ============================================================


def test_sweep_bearish_bsl() -> None:
    """Swing high (108) 박힌 거 박힌 박힌 wick 박힌 게 110 박힌 후 close 박은 게 105 박힘."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 108, 100, 107),  # swing high idx=1
        (107, 105, 100, 102),
        (102, 110, 101, 105),  # wick high=110 > 108, close=105 < 108 → BSL sweep
    ])
    swings = detect_swing_points(df)
    sweeps = detect_liquidity_sweeps(df, swings)
    assert len(sweeps) == 1
    assert sweeps[0].type is SweepType.BEARISH
    assert sweeps[0].swept_price == 108.0
    assert sweeps[0].wick_price == 110.0
    assert sweeps[0].idx == 3


def test_sweep_bullish_ssl() -> None:
    """Swing low (95) 박힌 거 박힌 박힌 wick 박힌 게 92 박힌 후 close 박은 게 97 박힘."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 105, 100, 102),
        (102, 103, 95, 96),    # swing low idx=2
        (96, 100, 95.5, 99),
        (99, 100, 92, 97),     # wick low=92 < 95, close=97 > 95 → SSL sweep
    ])
    swings = detect_swing_points(df)
    sweeps = detect_liquidity_sweeps(df, swings)
    # swing 박힌 거 박힌 거 박힌 거 = 박힌 거 박힌 거 박힌 거 (idx=2)
    ssl_sweeps = [s for s in sweeps if s.type is SweepType.BULLISH]
    assert len(ssl_sweeps) == 1
    assert ssl_sweeps[0].swept_price == 95.0
    assert ssl_sweeps[0].wick_price == 92.0


def test_sweep_close_breakout_not_sweep() -> None:
    """close 박은 게 swing high 박은 거보다 위 박힘 → sweep X (breakout 박힘)."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 108, 100, 107),
        (107, 105, 100, 102),
        (102, 112, 101, 111),  # close=111 > 108 → breakout, not sweep
    ])
    swings = detect_swing_points(df)
    sweeps = detect_liquidity_sweeps(df, swings)
    assert sweeps == []


def test_sweep_no_swept_swing_unchanged() -> None:
    """sweep 박힌 swing 박은 거 박힌 거 ``swept=True`` 박힘."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 108, 100, 107),
        (107, 105, 100, 102),
        (102, 110, 101, 105),
    ])
    swings = detect_swing_points(df)
    sh = next(s for s in swings if s.type is SwingType.HIGH)
    assert sh.swept is False
    detect_liquidity_sweeps(df, swings)
    assert sh.swept is True


# ============================================================
# Equal Levels (EQH / EQL)
# ============================================================


def test_equal_highs() -> None:
    """2개 swing high 박힌 거 박힌 ±0.1% 안 박힘 → EQH 박힘."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 108, 100, 107),    # swing high (108)
        (107, 105, 100, 102),
        (102, 108.05, 101, 105), # swing high (108.05) — 0.05% 박힘
        (105, 106, 100, 101),
    ])
    swings = detect_swing_points(df)
    levels = detect_equal_levels(swings, tolerance_pct=0.001)
    eqhs = [lvl for lvl in levels if lvl.type is SwingType.HIGH]
    assert len(eqhs) == 1
    assert abs(eqhs[0].price - 108.025) < 0.01
    assert len(eqhs[0].indices) == 2


def test_equal_levels_min_count() -> None:
    """min_count=3 박힘 — 박힌 2개 박힘 → 박힌 거 X."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 108, 100, 107),
        (107, 105, 100, 102),
        (102, 108.05, 101, 105),
        (105, 106, 100, 101),
    ])
    swings = detect_swing_points(df)
    levels = detect_equal_levels(swings, min_count=3)
    assert levels == []


# ============================================================
# Structure Events (BOS / CHoCH)
# ============================================================


def test_structure_choch_bullish() -> None:
    """trend None → bearish break (CHOCH_BEARISH) → bullish break (CHOCH_BULLISH)."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 103, 95, 96),     # swing low idx=1 (95)
        (96, 105, 95.5, 104),
        (104, 105, 88, 89),     # swing low idx=3 (88) — close=89 < 95 → CHOCH_BEARISH
        (89, 99, 88.5, 92),
        (92, 102, 91, 100),
        (100, 108, 99, 107),    # swing high idx=6 (108)
        (107, 107, 100, 102),   # idx=7 박힌 거 high=107 < 108 → idx=6 박힌 swing high 박힘
        (102, 110, 101, 109),   # close=109 > 108 → CHOCH_BULLISH
    ])
    swings = detect_swing_points(df)
    events = detect_structure_events(df, swings)
    types = [e.type for e in events]
    assert StructureType.CHOCH_BEARISH in types
    assert StructureType.CHOCH_BULLISH in types


def test_structure_no_event_without_break() -> None:
    """swing 박혔지만 close 박은 게 박힌 거 박힌 박힘 안 박힘 → 이벤트 X."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 108, 100, 107),  # swing high
        (107, 105, 100, 102),
        (102, 107, 101, 106),  # close=106 < 108 (break X)
    ])
    swings = detect_swing_points(df)
    events = detect_structure_events(df, swings)
    assert events == []


def test_structure_bos_bullish() -> None:
    """uptrend 박힌 박힌 박힘 swing high 박힌 거 박힘 → BOS_BULLISH."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 108, 100, 107),    # swing high (108)
        (107, 105, 100, 102),
        (102, 103, 100, 101),    # swing low (100)
        (101, 110, 100.5, 109),  # close=109 > 108 → bullish (trend None → CHOCH)
        (109, 108, 100, 102),    # swing low (100)
        (102, 109, 100.5, 103),  # swing high (109)
        (103, 105, 100, 101),    # swing low (100)
        (101, 115, 100.5, 114),  # close=114 > 109 → BOS_BULLISH (이미 UP)
    ])
    swings = detect_swing_points(df)
    events = detect_structure_events(df, swings)
    bos_bulls = [e for e in events if e.type is StructureType.BOS_BULLISH]
    assert len(bos_bulls) >= 1
    # 첫 break = CHOCH (trend None → up), 두 번째 break = BOS
    first = events[0]
    assert first.type is StructureType.CHOCH_BULLISH
    assert first.trend_before is TrendDirection.NONE


# ============================================================
# Sweep retest detection + zone coordinates
# ============================================================


def test_sweep_zone_coordinates_bearish() -> None:
    """BEARISH sweep zone = swept_price ~ wick_price (high)."""
    from aurora_ict.indicators.liquidity import (
        SweepType,
        detect_liquidity_sweeps,
    )
    df = _make_df([
        (100, 105, 99, 104),     # idx=0  high=105
        (104, 112, 103, 109),    # idx=1  swing high (112)
        (109, 110, 100, 102),    # idx=2  high=110
        (102, 118, 101, 105),    # idx=3  wick=118 > 112, close=105 < 112 → BEARISH sweep
    ])
    swings = detect_swing_points(df)
    sweeps = detect_liquidity_sweeps(df, swings)
    bearish = [s for s in sweeps if s.type is SweepType.BEARISH]
    assert len(bearish) == 1
    sw = bearish[0]
    assert sw.zone_top == 118.0   # wick_price (high)
    assert sw.zone_bottom == 112.0  # swept_price (swing high)


def test_sweep_zone_coordinates_bullish() -> None:
    """BULLISH sweep zone = wick_price (low) ~ swept_price."""
    from aurora_ict.indicators.liquidity import (
        SweepType,
        detect_liquidity_sweeps,
    )
    df = _make_df([
        (100, 105, 95, 96),      # idx=0  low=95
        (96, 100, 88, 91),       # idx=1  swing low (88), high=100 (not swing high)
        (91, 102, 90, 99),       # idx=2  low=90, high=102 > 100
        (99, 105, 80, 95),       # idx=3  wick=80 < 88, close=95 > 88 → BULLISH sweep
    ])
    swings = detect_swing_points(df)
    sweeps = detect_liquidity_sweeps(df, swings)
    bull = [s for s in sweeps if s.type is SweepType.BULLISH]
    assert len(bull) == 1
    sw = bull[0]
    assert sw.zone_top == 88.0      # swept_price (swing low)
    assert sw.zone_bottom == 80.0   # wick_price (low)


def test_sweep_retest_detected() -> None:
    """Sweep 후 후속 봉이 zone 안에 wick 으로 진입 → retested=True."""
    from aurora_ict.indicators.liquidity import detect_liquidity_sweeps
    df = _make_df([
        (100, 105, 99, 104),
        (104, 112, 103, 109),    # swing high 112
        (109, 110, 100, 102),
        (102, 118, 101, 105),    # BEARISH sweep (zone 112~118)
        (105, 115, 104, 108),    # idx=4 high=115 ∈ [112,118] → retested!
    ])
    swings = detect_swing_points(df)
    sweeps = detect_liquidity_sweeps(df, swings)
    assert any(s.retested for s in sweeps)


def test_sweep_no_retest_when_price_moves_away() -> None:
    """Sweep 후 가격이 zone 멀어지면 retested=False."""
    from aurora_ict.indicators.liquidity import detect_liquidity_sweeps
    df = _make_df([
        (100, 105, 99, 104),
        (104, 112, 103, 109),    # swing high 112
        (109, 110, 100, 102),
        (102, 118, 101, 105),    # BEARISH sweep (zone 112~118)
        (105, 108, 95, 97),       # idx=4 high=108 < 112 → zone 안 아님
        (97, 100, 90, 92),
    ])
    swings = detect_swing_points(df)
    sweeps = detect_liquidity_sweeps(df, swings)
    bearish = [s for s in sweeps if s.type.value == "bearish"]
    assert len(bearish) >= 1
    assert all(not s.retested for s in bearish)
