"""Multi-TF Bias 단위 테스트."""

from __future__ import annotations

import pandas as pd

from aurora_ict.indicators.structure import TrendDirection
from aurora_ict.strategy.multi_tf_bias import (
    combine_bias,
    compute_bias_from_df,
    htf_pair,
)


def _make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = [i * 60_000 for i in range(len(rows))]
    return df


# ============================================================
# htf_pair
# ============================================================


def test_htf_pair_5m() -> None:
    assert htf_pair("5m") == ("1h", "4h")


def test_htf_pair_15m() -> None:
    assert htf_pair("15m") == ("1h", "4h")


def test_htf_pair_1h() -> None:
    assert htf_pair("1h") == ("4h", "1d")


def test_htf_pair_4h() -> None:
    assert htf_pair("4h") == ("1d", "1w")


def test_htf_pair_1d() -> None:
    assert htf_pair("1d") == ("1w", None)


def test_htf_pair_1w() -> None:
    assert htf_pair("1w") == (None, None)


def test_htf_pair_unknown() -> None:
    assert htf_pair("3m") == (None, None)


# ============================================================
# compute_bias_from_df
# ============================================================


def test_compute_bias_empty_df() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    assert compute_bias_from_df(df) is TrendDirection.NONE


def test_compute_bias_too_short() -> None:
    df = _make_df([(100, 101, 99, 100)] * 3)
    assert compute_bias_from_df(df) is TrendDirection.NONE


def test_compute_bias_bullish_event() -> None:
    """swing high → 후속 close 가 그 위로 돌파 → CHoCH_BULLISH → UP."""
    df = _make_df([
        (100, 101, 99, 100),     # 0
        (100, 104, 99, 103),     # 1 swing high (high=104)
        (103, 103, 95, 96),      # 2
        (96, 97, 94, 95),        # 3
        (95, 110, 94, 109),      # 4 close=109 > 104 → CHoCH_BULLISH
    ])
    assert compute_bias_from_df(df) is TrendDirection.UP


def test_compute_bias_bearish_event() -> None:
    """swing low → 후속 close 가 그 아래로 돌파 → CHoCH_BEARISH → DOWN."""
    df = _make_df([
        (102, 103, 99, 100),
        (100, 101, 88, 89),      # swing low (low=88)
        (89, 95, 89, 94),
        (94, 100, 90, 99),
        (99, 105, 80, 81),       # close=81 < 88 → CHoCH_BEARISH
    ])
    assert compute_bias_from_df(df) is TrendDirection.DOWN


def test_compute_bias_no_event() -> None:
    """structure event 없으면 NONE."""
    df = _make_df([(100, 101, 99, 100)] * 10)
    assert compute_bias_from_df(df) is TrendDirection.NONE


# ============================================================
# combine_bias
# ============================================================


def test_combine_both_up() -> None:
    assert (
        combine_bias(TrendDirection.UP, TrendDirection.UP) is TrendDirection.UP
    )


def test_combine_both_down() -> None:
    assert (
        combine_bias(TrendDirection.DOWN, TrendDirection.DOWN) is TrendDirection.DOWN
    )


def test_combine_one_none_uses_other() -> None:
    """한쪽 NONE 이면 다른쪽 따름."""
    assert (
        combine_bias(TrendDirection.NONE, TrendDirection.UP) is TrendDirection.UP
    )
    assert (
        combine_bias(TrendDirection.DOWN, TrendDirection.NONE) is TrendDirection.DOWN
    )


def test_combine_conflict_returns_none() -> None:
    """UP vs DOWN 충돌 → NONE."""
    assert (
        combine_bias(TrendDirection.UP, TrendDirection.DOWN) is TrendDirection.NONE
    )
    assert (
        combine_bias(TrendDirection.DOWN, TrendDirection.UP) is TrendDirection.NONE
    )


def test_combine_both_none() -> None:
    assert (
        combine_bias(TrendDirection.NONE, TrendDirection.NONE) is TrendDirection.NONE
    )
