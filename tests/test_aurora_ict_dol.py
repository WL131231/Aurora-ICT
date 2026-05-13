"""DOL (Draw On Liquidity) 단위 테스트."""

from __future__ import annotations

import pandas as pd

from aurora_ict.indicators.dol import compute_dol
from aurora_ict.indicators.swing_points import SwingPoint, SwingType


def _make_df(closes: list[float]) -> pd.DataFrame:
    df = pd.DataFrame({"open": closes, "high": closes, "low": closes, "close": closes})
    df.index = [i * 60_000 for i in range(len(closes))]
    return df


def test_dol_empty_df() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    assert compute_dol(df, []) == []


def test_dol_empty_swings() -> None:
    df = _make_df([100.0])
    assert compute_dol(df, []) == []


def test_dol_returns_above_and_below() -> None:
    """현재가 100, 위 swing 110 + 아래 swing 90 → 2개 DOL."""
    df = _make_df([100.0])
    swings = [
        SwingPoint(ts_ms=0, type=SwingType.HIGH, price=110.0, idx=0, swept=False),
        SwingPoint(ts_ms=60_000, type=SwingType.LOW, price=90.0, idx=1, swept=False),
    ]
    dols = compute_dol(df, swings)
    assert len(dols) == 2
    bull = next(d for d in dols if d.type == "bullish")
    bear = next(d for d in dols if d.type == "bearish")
    assert bull.price == 110.0
    assert bear.price == 90.0
    assert bull.distance == 10.0
    assert bear.distance == 10.0


def test_dol_picks_closest_unswept() -> None:
    """위에 swing 두 개 — 가장 가까운 것 (현재가에 가까운 가격) 선택."""
    df = _make_df([100.0])
    swings = [
        SwingPoint(ts_ms=0, type=SwingType.HIGH, price=120.0, idx=0, swept=False),
        SwingPoint(ts_ms=60_000, type=SwingType.HIGH, price=105.0, idx=1, swept=False),
    ]
    dols = compute_dol(df, swings)
    assert len(dols) == 1
    assert dols[0].type == "bullish"
    assert dols[0].price == 105.0  # 120 보다 100 에 가까움


def test_dol_skips_swept_swings() -> None:
    """swept=True 인 swing 은 DOL 후보 X."""
    df = _make_df([100.0])
    swings = [
        SwingPoint(ts_ms=0, type=SwingType.HIGH, price=105.0, idx=0, swept=True),
        SwingPoint(ts_ms=60_000, type=SwingType.HIGH, price=115.0, idx=1, swept=False),
    ]
    dols = compute_dol(df, swings)
    assert len(dols) == 1
    assert dols[0].price == 115.0


def test_dol_no_above_only_below() -> None:
    """위쪽 unswept swing 없으면 아래쪽 DOL 만 반환."""
    df = _make_df([100.0])
    swings = [
        SwingPoint(ts_ms=0, type=SwingType.LOW, price=90.0, idx=0, swept=False),
    ]
    dols = compute_dol(df, swings)
    assert len(dols) == 1
    assert dols[0].type == "bearish"


def test_dol_excludes_same_side_swing() -> None:
    """현재가 100, swing high 95 (= 현재가 아래) → bullish DOL 후보 X."""
    df = _make_df([100.0])
    swings = [
        SwingPoint(ts_ms=0, type=SwingType.HIGH, price=95.0, idx=0, swept=False),
    ]
    dols = compute_dol(df, swings)
    assert dols == []
