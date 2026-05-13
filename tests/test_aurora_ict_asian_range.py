"""Asian Range 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from aurora_ict.indicators.asian_range import compute_asian_range

NY = ZoneInfo("America/New_York")


def _make_df(start_ny: datetime, bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """1분 봉 DataFrame — start_ny 부터 1분 간격."""
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex(
        [start_ny + timedelta(minutes=i) for i in range(len(rows))],
    )
    return df


def test_asian_range_empty_df() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    assert compute_asian_range(df) is None


def test_asian_range_no_asian_bars() -> None:
    """NY 12:00 시작 봉만 있음 → Asian 봉 없음 → None."""
    start = datetime(2026, 5, 12, 12, 0, tzinfo=NY)
    df = _make_df(start, [(100, 101, 99, 100)] * 10)
    assert compute_asian_range(df) is None


def test_asian_range_extracts_high_low() -> None:
    """NY 01:00~01:04 봉 5개 안의 high/low 추출."""
    start = datetime(2026, 5, 12, 1, 0, tzinfo=NY)
    df = _make_df(start, [
        (100, 102, 99, 101),
        (101, 105, 100, 103),    # high=105
        (103, 104, 95, 96),      # low=95
        (96, 100, 95.5, 99),
        (99, 101, 98, 100),
    ])
    ar = compute_asian_range(df)
    assert ar is not None
    assert ar.high == 105
    assert ar.low == 95


def test_asian_range_picks_latest_session_only() -> None:
    """오늘 + 어제 Asian 둘 다 있으면 오늘 (마지막 session) 만 사용."""
    # 2026-05-11 01:00 부터 5분 + 2026-05-12 01:00 부터 5분
    bars = []
    # 어제 — high=200 (이게 잡히면 안 됨)
    for _ in range(5):
        bars.append((100, 200, 99, 100))
    # gap 봉들 (낮시간 — 채우기용)
    for _ in range(5):
        bars.append((100, 101, 99, 100))
    # 오늘 — high=110
    for _ in range(5):
        bars.append((100, 110, 99, 100))

    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    times = []
    # 어제 01:00~01:04
    yesterday_start = datetime(2026, 5, 11, 1, 0, tzinfo=NY)
    for i in range(5):
        times.append(yesterday_start + timedelta(minutes=i))
    # gap — 어제 12:00~12:04
    for i in range(5):
        times.append(datetime(2026, 5, 11, 12, 0, tzinfo=NY) + timedelta(minutes=i))
    # 오늘 01:00~01:04
    today_start = datetime(2026, 5, 12, 1, 0, tzinfo=NY)
    for i in range(5):
        times.append(today_start + timedelta(minutes=i))

    df.index = pd.DatetimeIndex(times)
    ar = compute_asian_range(df)
    assert ar is not None
    assert ar.high == 110  # 오늘만 — 어제 200 은 무시
    assert ar.low == 99
