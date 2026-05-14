"""ICT Signal generator — Aurora-ICT v0.1.4."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from aurora_ict.indicators.structure import TrendDirection
from aurora_ict.signal.ict_signal import (
    SignalAction,
    generate_ict_signal,
)

NY = ZoneInfo("America/New_York")


def _make_df_ny(start_ny: datetime, bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    times = [start_ny + timedelta(minutes=i) for i in range(len(rows))]
    df.index = pd.DatetimeIndex(times)
    return df


def test_signal_short_df() -> None:
    """df 박힌 거 박힘 박힙 박힘 → NO_ACTION."""
    df = pd.DataFrame([
        {"open": 100, "high": 101, "low": 99, "close": 100},
    ])
    df.index = pd.DatetimeIndex([datetime(2026, 5, 12)])
    sig = generate_ict_signal(df, "BTCUSDT")
    assert sig.action is SignalAction.NO_ACTION
    assert sig.is_actionable is False
    assert "too short" in sig.reason


def test_signal_no_setup() -> None:
    """setup 박힘 X → NO_ACTION."""
    start = datetime(2026, 5, 12, 5, 0, tzinfo=NY)  # SB 박힘 X
    df = _make_df_ny(start, [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
    ])
    sig = generate_ict_signal(df, "BTCUSDT", bias=TrendDirection.UP)
    assert sig.action is SignalAction.NO_ACTION


def test_signal_enter_long() -> None:
    """valid long setup 박힘 박힌 마지막 봉 박힘 박힘 → ENTER_LONG."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 105, 99, 104),
        (104, 130, 103, 125),    # swing high 130
        (125, 124, 100, 101),
        (101, 108, 95, 96),
        (96, 110, 95.5, 109),
        (109, 112, 92, 93),
        (93, 100, 92.5, 99),
        (99, 105, 98, 104),
        (104, 106, 100, 101),
        (101, 105, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),    # 1봉
        (109, 119, 108, 118),    # 2봉 displacement (FVG 박힌 거 박힘)
        (118, 122, 115, 121),    # 3봉
    ])
    sig = generate_ict_signal(
        df,
        "BTCUSDT",
        bias=TrendDirection.UP,
        fvg_min_size_pct=0.001,
        min_rr=1.0,
    )
    assert sig.action is SignalAction.ENTER_LONG
    assert sig.is_actionable is True
    assert sig.setup is not None
    assert sig.setup.entry > 0
    assert sig.symbol == "BTCUSDT"
    assert "window=" in sig.reason


def test_signal_stale_setup_filtered() -> None:
    """setup 박힌 거 박힘 5 봉 너머 박힘 박힌 거 박힘 박힘 → NO_ACTION."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 105, 99, 104),
        (104, 130, 103, 125),
        (125, 124, 100, 101),
        (101, 108, 95, 96),
        (96, 110, 95.5, 109),
        (109, 112, 92, 93),
        (93, 100, 92.5, 99),
        (99, 105, 98, 104),
        (104, 106, 100, 101),
        (101, 105, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 119, 108, 118),    # FVG idx=12 박힘
        (118, 122, 115, 121),
        # 박힌 봉 박힘 박힘 박힘 — setup 박힌 거 박힙 박힘 박힘 박힙 박힘 박힘
        (121, 122, 119, 120),
        (120, 121, 119, 120),
        (120, 121, 119, 120),
        (120, 121, 119, 120),
        (120, 121, 119, 120),
        (120, 121, 119, 120),
    ])
    sig = generate_ict_signal(
        df,
        "BTCUSDT",
        bias=TrendDirection.UP,
        fvg_min_size_pct=0.001,
        min_rr=1.0,
    )
    # FVG idx=12 박힘 박힘, 박힌 봉 박힘 19 → bars_since = 19-12 = 7 > 5 박힘 박힙
    assert sig.action is SignalAction.NO_ACTION
    assert "stale" in sig.reason


def test_signal_symbol_propagated() -> None:
    """symbol 이 정상 전파됨."""
    df = pd.DataFrame([
        {"open": 100, "high": 101, "low": 99, "close": 100},
    ])
    df.index = pd.DatetimeIndex([datetime(2026, 5, 12)])
    sig = generate_ict_signal(df, "ETHUSDT")
    assert sig.symbol == "ETHUSDT"


# ============================================================
# require_retrace (LuxAlgo Silver Bullet "Strict" mode)
# ============================================================


def test_signal_require_retrace_skips_unretraced() -> None:
    """require_retrace=True + setup.retraced=False → NO_ACTION (awaiting retrace)."""
    # FVG 박힌 후 가격이 mean threshold 까지 retrace 안 한 케이스
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    bars = [
        (100, 105, 99, 104),
        (104, 130, 103, 125),    # swing high (TP candidate)
        (125, 124, 100, 101),
        (101, 108, 95, 96),
        (96, 110, 95.5, 109),
        (109, 112, 92, 93),
        (93, 100, 92.5, 99),
        (99, 105, 98, 104),
        (104, 106, 100, 101),
        (101, 105, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 119, 108, 118),    # FVG idx=12
        (118, 122, 115, 121),    # mean(112.5) 까지 retrace X (low=115 > 112.5)
    ]
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    times = [start + timedelta(minutes=i) for i in range(len(rows))]
    df.index = pd.DatetimeIndex(times)
    sig = generate_ict_signal(
        df, "BTCUSDT", min_rr=1.0, fvg_min_size_pct=0.001,
        require_retrace=True,
    )
    # FVG 발생했지만 retrace 미발생 → awaiting retrace
    assert sig.action is SignalAction.NO_ACTION
    assert "retrace" in sig.reason


def test_signal_require_retrace_allows_retraced() -> None:
    """require_retrace=True + setup.retraced=True → 진입 허용."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    bars = [
        (100, 105, 99, 104),
        (104, 130, 103, 125),
        (125, 124, 100, 101),
        (101, 108, 95, 96),
        (96, 110, 95.5, 109),
        (109, 112, 92, 93),
        (93, 100, 92.5, 99),
        (99, 105, 98, 104),
        (104, 106, 100, 101),
        (101, 105, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 119, 108, 118),    # FVG idx=12 (low=108, high=118 → mean=113), 1봉 high=110, 3봉 low=115
        (118, 122, 115, 121),    # idx=13 FVG anchor 의 3봉
        (121, 122, 110, 120),    # idx=14 low=110 < mean=113 → retraced!
    ]
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    times = [start + timedelta(minutes=i) for i in range(len(rows))]
    df.index = pd.DatetimeIndex(times)
    sig = generate_ict_signal(
        df, "BTCUSDT", min_rr=1.0, fvg_min_size_pct=0.001,
        require_retrace=True,
    )
    # retrace 됐으니 진입 허용
    assert sig.action is SignalAction.ENTER_LONG
