"""Chart markers — Aurora-ICT v0.1.8."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from aurora_ict.api.markers import ChartMarkers, to_chart_markers

NY = ZoneInfo("America/New_York")


def _make_df_ny(start_ny: datetime, bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = [{"open": o, "high": h, "low": l, "close": c} for o, h, l, c in bars]
    df = pd.DataFrame(rows)
    times = [start_ny + timedelta(minutes=i) for i in range(len(rows))]
    df.index = pd.DatetimeIndex(times)
    return df


def test_markers_empty_df() -> None:
    """짧은 df → 빈 ChartMarkers."""
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    markers = to_chart_markers(df)
    assert isinstance(markers, ChartMarkers)
    assert markers.fvgs == []
    assert markers.swings == []


def test_markers_fvg_and_swings() -> None:
    """기본 FVG / Swing 박힘 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 105, 99, 104),
        (104, 130, 103, 125),    # swing high
        (125, 124, 100, 101),
        (101, 108, 95, 96),       # swing low
        (96, 110, 95.5, 109),
        (109, 112, 92, 93),
        (93, 100, 92.5, 99),
        (99, 105, 98, 104),
        (104, 106, 100, 101),
        (101, 105, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 119, 108, 118),    # bullish FVG idx=12
        (118, 122, 115, 121),
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=0.001, min_rr=1.0)
    assert len(markers.fvgs) >= 1
    assert any(f.type == "bullish" for f in markers.fvgs)
    assert len(markers.swings) >= 2  # high + low


def test_markers_killzone_detection() -> None:
    """NY 10:00 박힌 박은 london_close (10-12pm) 박힙 박힘 AM SB 박은 안."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [(100, 101, 99, 100) for _ in range(60)])
    markers = to_chart_markers(df)
    assert len(markers.killzones) >= 1
    # 10:00 ~ 10:59 박힘 박힙 london_close 박힌 거 박힘 박힙 (ny_am 박힘 박힘 7-10am exclusive)
    lc_zones = [k for k in markers.killzones if k.name == "london_close"]
    assert len(lc_zones) >= 1
    # Silver Bullet 박힘 박힙 박힙 박힘 박힙 (10-11am 박힘 박힙 am_sb)
    assert any(k.is_silver_bullet for k in lc_zones)


def test_markers_setups_detected() -> None:
    """Silver Bullet setup 박힌 거 박힘 박힘 박힘."""
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
        (109, 119, 108, 118),
        (118, 122, 115, 121),
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=0.001, min_rr=1.0)
    assert len(markers.setups) >= 1
    s = markers.setups[0]
    assert s.direction == "long"
    assert s.entry > 0
    assert s.stop_loss < s.entry
    assert s.take_profit > s.entry


def test_markers_include_setups_false() -> None:
    """include_setups=False 박힘 → setups 박힘 X."""
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
        (109, 119, 108, 118),
        (118, 122, 115, 121),
    ])
    markers = to_chart_markers(df, include_setups=False, fvg_min_size_pct=0.001)
    assert markers.setups == []
    # FVG 박힘 박힘 박힙 박힘 박힘 박힘
    assert len(markers.fvgs) >= 1


def test_markers_to_dict_json_serializable() -> None:
    """to_dict 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘."""
    import json

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
        (109, 119, 108, 118),
        (118, 122, 115, 121),
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=0.001, min_rr=1.0)
    d = markers.to_dict()
    # JSON 박힘 박힘 박힘
    encoded = json.dumps(d)
    assert isinstance(encoded, str)
    # round-trip
    decoded = json.loads(encoded)
    assert "fvgs" in decoded
    assert "setups" in decoded
    assert "killzones" in decoded


def test_markers_swept_flag_propagated() -> None:
    """sweep 박힌 swing 박힌 거 박힌 swept=True 박힘 박힘 markers.swings 박힘 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 102, 99, 101),
        (101, 108, 100, 107),  # swing high 108
        (107, 105, 100, 102),
        (102, 110, 101, 105),  # wick high=110 > 108, close=105 < 108 → BSL sweep
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    swept = [s for s in markers.swings if s.swept]
    assert len(swept) >= 1
    assert any(s.type == "high" for s in swept)
