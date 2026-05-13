"""Daily Bias 단위 테스트 — PDH/PDL/PWO + bias 판정."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from aurora_ict.indicators.daily_bias import (
    compute_daily_bias,
    compute_daily_levels,
)
from aurora_ict.indicators.structure import TrendDirection


def _make_daily_df(
    start: datetime,
    bars: list[tuple[float, float, float, float]],
) -> pd.DataFrame:
    """1d 봉 OHLC DataFrame — index 가 DatetimeIndex (UTC) 로 박힘."""
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(
        [start + pd.Timedelta(days=i) for i in range(len(rows))],
    )
    return df


# ============================================================
# compute_daily_levels
# ============================================================


def test_levels_none_when_too_short() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    assert compute_daily_levels(df) is None


def test_levels_single_bar_returns_none() -> None:
    df = _make_daily_df(
        datetime(2026, 5, 12, tzinfo=UTC),
        [(100, 110, 95, 105)],
    )
    assert compute_daily_levels(df) is None


def test_levels_extracts_previous_day_ohlc() -> None:
    """3봉 — 마지막=오늘, -2=어제, -3=그저께. 어제 OHLC 추출."""
    df = _make_daily_df(
        datetime(2026, 5, 10, tzinfo=UTC),
        [
            (100, 110, 95, 108),    # 5/10
            (108, 120, 105, 115),   # 5/11  ← 어제 (prev)
            (115, 118, 110, 112),   # 5/12  오늘
        ],
    )
    lv = compute_daily_levels(df)
    assert lv is not None
    assert lv.pdh == 120
    assert lv.pdl == 105
    assert lv.pdo == 108
    assert lv.pdc == 115


def test_levels_finds_prev_week_open_monday() -> None:
    """직전 월요일 봉 open 을 PWO 로 추출."""
    # 2026-05-04 (월) 부터 9봉 — 마지막 5/12 (화)
    df = _make_daily_df(
        datetime(2026, 5, 4, tzinfo=UTC),  # 월요일
        [(100 + i, 110 + i, 95 + i, 108 + i) for i in range(9)],
    )
    lv = compute_daily_levels(df)
    assert lv is not None
    # 마지막 = 5/12 (화), 어제 = 5/11 (월). 직전 주 월요일 = 5/4
    # 5/4 봉의 open = 100
    assert lv.pwo == 100


# ============================================================
# compute_daily_bias
# ============================================================


def test_bias_up_when_above_pdh() -> None:
    df = _make_daily_df(
        datetime(2026, 5, 10, tzinfo=UTC),
        [(100, 110, 95, 108), (108, 120, 105, 115), (115, 121, 110, 121)],
    )
    # current=121 > pdh=120 → UP
    assert compute_daily_bias(df, current_close=121) is TrendDirection.UP


def test_bias_down_when_below_pdl() -> None:
    df = _make_daily_df(
        datetime(2026, 5, 10, tzinfo=UTC),
        [(100, 110, 95, 108), (108, 120, 105, 115), (115, 116, 100, 100)],
    )
    # current=100 < pdl=105 → DOWN
    assert compute_daily_bias(df, current_close=100) is TrendDirection.DOWN


def test_bias_none_inside_range() -> None:
    df = _make_daily_df(
        datetime(2026, 5, 10, tzinfo=UTC),
        [(100, 110, 95, 108), (108, 120, 105, 115), (115, 118, 110, 112)],
    )
    # current=112 in [105, 120] → NONE
    assert compute_daily_bias(df, current_close=112) is TrendDirection.NONE


def test_bias_none_when_levels_missing() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    assert compute_daily_bias(df, current_close=100) is TrendDirection.NONE
