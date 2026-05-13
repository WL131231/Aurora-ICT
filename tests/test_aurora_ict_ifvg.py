"""IFVG (Inversion FVG) 단위 테스트."""

from __future__ import annotations

import pandas as pd

from aurora_ict.indicators.fvg import (
    FVGType,
    detect_fvgs,
    detect_ifvgs,
    mark_filled_and_invalidated,
)


def _make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = [i * 60_000 for i in range(len(rows))]
    return df


def test_ifvg_empty_when_no_fvgs() -> None:
    df = _make_df([(100, 101, 99, 100)] * 3)
    assert detect_ifvgs([], df) == []


def test_ifvg_skipped_when_not_invalidated() -> None:
    """invalidated=False 인 FVG 는 IFVG 안 됨."""
    # bullish FVG: idx=0 high=100, idx=2 low=105 → gap 100~105
    df = _make_df([
        (95, 100, 94, 99),    # idx=0  high=100
        (99, 115, 99, 114),   # idx=1  displacement
        (114, 116, 105, 115), # idx=2  low=105 → bullish FVG (100~105)
        (115, 118, 113, 117), # idx=3
        (117, 119, 115, 118), # idx=4  filled 안 됨, invalidated 안 됨
    ])
    fvgs = detect_fvgs(df, min_size_pct=0.001)
    mark_filled_and_invalidated(fvgs, df)
    ifvgs = detect_ifvgs(fvgs, df)
    assert ifvgs == []


def test_ifvg_bullish_fvg_to_bearish_ifvg() -> None:
    """bullish FVG 깨짐 → bearish IFVG."""
    df = _make_df([
        (95, 100, 94, 99),    # 0
        (99, 115, 99, 114),   # 1
        (114, 116, 105, 115), # 2  bullish FVG (100~105)
        (115, 118, 113, 117), # 3
        (117, 119, 100, 99),  # 4  close=99 < FVG low=100 → invalidated
    ])
    fvgs = detect_fvgs(df, min_size_pct=0.001)
    mark_filled_and_invalidated(fvgs, df)
    ifvgs = detect_ifvgs(fvgs, df)
    assert len(ifvgs) == 1
    ifvg = ifvgs[0]
    assert ifvg.type is FVGType.BEARISH
    assert ifvg.high == 105.0
    assert ifvg.low == 100.0
    assert ifvg.invalidated_idx == 4
    assert ifvg.origin_fvg_idx == 1


def test_ifvg_bearish_fvg_to_bullish_ifvg() -> None:
    """bearish FVG 깨짐 → bullish IFVG."""
    df = _make_df([
        (105, 106, 100, 101),  # 0  low=100
        (101, 102, 85, 86),    # 1  displacement down
        (86, 95, 85, 94),      # 2  high=95 → bearish FVG (95~100)
        (94, 96, 92, 95),
        (95, 105, 94, 101),    # 4  close=101 > 100 → invalidated
    ])
    fvgs = detect_fvgs(df, min_size_pct=0.001)
    mark_filled_and_invalidated(fvgs, df)
    ifvgs = detect_ifvgs(fvgs, df)
    assert len(ifvgs) == 1
    ifvg = ifvgs[0]
    assert ifvg.type is FVGType.BULLISH
    assert ifvg.high == 100.0
    assert ifvg.low == 95.0
    assert ifvg.mean_threshold == 97.5


def test_ifvg_size_and_mean() -> None:
    """size / mean_threshold helper 동작."""
    df = _make_df([
        (95, 100, 94, 99),
        (99, 115, 99, 114),
        (114, 116, 105, 115),
        (115, 118, 113, 117),
        (117, 119, 100, 99),
    ])
    fvgs = detect_fvgs(df, min_size_pct=0.001)
    mark_filled_and_invalidated(fvgs, df)
    ifvgs = detect_ifvgs(fvgs, df)
    assert ifvgs[0].size == 5.0
    assert ifvgs[0].mean_threshold == 102.5
