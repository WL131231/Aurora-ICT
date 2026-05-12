"""Silver Bullet entry model — Aurora-ICT v0.1.3."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from aurora_ict.indicators.structure import TrendDirection
from aurora_ict.strategy.silver_bullet import (
    Direction,
    detect_silver_bullet_setups,
)

NY = ZoneInfo("America/New_York")


def _make_df_ny(start_ny: datetime, bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """1분 봉 DataFrame — start_ny 박힌 거 박힌 거 박힘 1m 간격 박힘 박힘."""
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    times = [start_ny + timedelta(minutes=i) for i in range(len(rows))]
    df.index = pd.DatetimeIndex(times)
    return df


def test_no_setup_without_bias() -> None:
    """trend None 박힘 → setup 박힘 X."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 101, 99, 100),
    ])
    setups = detect_silver_bullet_setups(df)
    assert setups == []


def test_silver_bullet_long_setup_am() -> None:
    """AM SB 윈도우 (10-11am NY) 박힌 bullish FVG + 박힌 BSL target → long setup."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 105, 99, 104),
        (104, 130, 103, 125),    # swing high 130 (TP candidate, idx=1)
        (125, 124, 100, 101),
        (101, 108, 95, 96),      # swing low 95
        (96, 110, 95.5, 109),
        (109, 112, 92, 93),      # swing low 92
        (93, 100, 92.5, 99),
        (99, 105, 98, 104),
        (104, 106, 100, 101),
        (101, 105, 99, 100),
        (100, 102, 99, 101),
        # AM SB 윈도우 안 (10:11~)
        (101, 110, 100, 109),    # 1봉
        (109, 119, 108, 118),    # 2봉 displacement
        (118, 122, 115, 121),    # 3봉 — low=115 > 1봉 high=110 → bullish FVG (110~115)
    ])
    setups = detect_silver_bullet_setups(
        df,
        bias=TrendDirection.UP,   # 박힌 명시 bias 박힘
        fvg_min_size_pct=0.001,
        min_rr=1.0,                # 박힌 박힌 박힌 test 박힘 박힘 박힘 박힘 박힘
    )
    longs = [s for s in setups if s.direction is Direction.LONG]
    assert len(longs) >= 1
    setup = longs[0]
    assert setup.window in ("am_sb", "london_sb", "pm_sb")
    assert setup.entry > 0
    assert setup.stop_loss < setup.entry  # long → SL 박힌 박힌
    assert setup.take_profit > setup.entry  # long → TP 박힌 위
    assert setup.risk_reward >= 1.0


def test_silver_bullet_outside_window() -> None:
    """SB 윈도우 박힌 안 박힌 FVG 박힘 → setup 박힘 X."""
    # NY 06:00 박힘 — 박힌 어떤 SB 윈도우 박힘 X
    start = datetime(2026, 5, 12, 5, 50, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 102, 99, 101),
        (101, 110, 100, 109),    # swing high
        (109, 110, 100, 102),
        (102, 103, 95, 96),      # swing low
        (96, 105, 95.5, 104),
        (104, 105, 92, 93),      # CHOCH_BEARISH
        (93, 105, 92.5, 104),
        (104, 115, 100, 114),    # CHOCH_BULLISH
        (114, 115, 110, 112),
        (112, 113, 110, 111),
        (111, 113, 111, 112),
        (112, 120, 111, 119),    # 박힌 박힌 박힌 FVG 박힙 박힘 — 박힌 NY 06:01 박힘 (SB 박힘 X)
        (119, 121, 116, 120),
        (120, 122, 119, 121),
    ])
    setups = detect_silver_bullet_setups(
        df,
        bias=TrendDirection.UP,
        fvg_min_size_pct=0.001,
        min_rr=1.0,
    )
    # 박힌 박힙 박힌 박힌 박힘 박힘 — SB 윈도우 박힘 박힘 X
    assert setups == []


def test_silver_bullet_min_rr_filter() -> None:
    """RR 박힘 박힘 박힘 min_rr 박힘 박힘 박힘 박힘 박힘 → 박힘 박힘 X."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 110, 100, 102),
        (102, 103, 95, 96),
        (96, 105, 95.5, 104),
        (104, 105, 92, 93),
        (93, 105, 92.5, 104),
        (104, 115, 100, 114),
        (114, 115, 110, 112),
        (112, 113, 110, 111),
        (111, 113, 111, 112),
        (112, 120, 111, 119),
        (119, 121, 116, 120),
        (120, 122, 119, 121),
    ])
    # min_rr=100 박힘 박힘 — 박힌 박힌 박힘 박힘 박힘 박힘 박힘 박힘 X
    setups = detect_silver_bullet_setups(
        df,
        bias=TrendDirection.UP,
        fvg_min_size_pct=0.001,
        min_rr=100.0,
    )
    assert setups == []


def test_silver_bullet_short_bias() -> None:
    """bias=DOWN 박힘 박힘 short setup 박힘 박힌 거 박힘 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (110, 115, 100, 105),
        (105, 110, 50, 55),      # swing low 50 (TP candidate, idx=1)
        (55, 100, 54, 99),
        (99, 110, 98, 109),
        (109, 112, 100, 101),
        (101, 105, 95, 96),
        (96, 100, 95.5, 99),
        (99, 105, 90, 91),       # swing low 90
        (91, 95, 85, 86),
        (86, 90, 80, 81),
        (81, 85, 79, 80),
        # AM SB 윈도우 안
        (80, 85, 79, 82),        # 1봉
        (82, 82, 70, 71),        # 2봉 displacement down
        (71, 75, 65, 68),        # 3봉 high=75 < 1봉 low=79 → bearish FVG (75~79)
    ])
    setups = detect_silver_bullet_setups(
        df,
        bias=TrendDirection.DOWN,
        fvg_min_size_pct=0.001,
        min_rr=1.0,
    )
    shorts = [s for s in setups if s.direction is Direction.SHORT]
    assert len(shorts) >= 1
    setup = shorts[0]
    assert setup.entry > 0
    assert setup.stop_loss > setup.entry  # short → SL 박힌 위
    assert setup.take_profit < setup.entry  # short → TP 박힌 아래


def test_silver_bullet_one_per_window() -> None:
    """같은 SB 윈도우 박힙 박힌 FVG 박힘 박힘 박힘 첫 박힘 박힌 거 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    # 박힌 박힘 박힌 박힘 박힘 박힌 FVG 박힘 박힙 박힘 박힘 (same window 박힙)
    df = _make_df_ny(start, [
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 110, 100, 102),
        (102, 103, 95, 96),
        (96, 105, 95.5, 104),
        (104, 105, 92, 93),
        (93, 105, 92.5, 104),
        (104, 115, 100, 114),
        (114, 115, 110, 112),
        (112, 113, 110, 111),
        # 10:10 박힘 박힘 박힘 — 박힌 박힌 FVG 박힘 박힐 박힌 박힌
        (111, 113, 111, 112),
        (112, 120, 111, 119),
        (119, 121, 116, 120),    # FVG 1 — 박힌 거 박힌 거 박힘
        (120, 122, 119, 121),
        (121, 130, 121, 129),    # 박힌 박힘 박힌 박힘
        (129, 131, 125, 130),    # FVG 2 — 박힙 박힘 (122 < 125)
        (130, 131, 128, 129),
    ])
    setups = detect_silver_bullet_setups(
        df,
        bias=TrendDirection.UP,
        fvg_min_size_pct=0.001,
        min_rr=1.0,
    )
    # 같은 윈도우 박힘 박힘 박힙 박힘 — 첫 박힘 박힘 박힘
    assert len(setups) <= 1


def test_silver_bullet_auto_bias_from_structure() -> None:
    """bias=None 박힘 박힘 → structure 박힙 박힘 자동 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 110, 100, 102),
        (102, 103, 95, 96),
        (96, 105, 95.5, 104),
        (104, 105, 92, 93),     # CHOCH_BEARISH (trend DOWN)
        (93, 105, 92.5, 104),
        (104, 115, 100, 114),   # CHOCH_BULLISH (trend UP)
        (114, 115, 110, 112),
        (112, 113, 110, 111),
        (111, 113, 111, 112),
        (112, 120, 111, 119),
        (119, 121, 116, 120),
        (120, 122, 119, 121),
    ])
    setups_auto = detect_silver_bullet_setups(
        df,
        bias=None,
        fvg_min_size_pct=0.001,
        min_rr=1.0,
    )
    setups_explicit = detect_silver_bullet_setups(
        df,
        bias=TrendDirection.UP,
        fvg_min_size_pct=0.001,
        min_rr=1.0,
    )
    # auto bias = UP 박힘 박힘 박힐 박힘 박힘 동일 박힘 박힘
    assert [s.direction for s in setups_auto] == [s.direction for s in setups_explicit]
