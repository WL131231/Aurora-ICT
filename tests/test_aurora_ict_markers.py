"""Chart markers — Aurora-ICT v0.1.8."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from aurora_ict.api.markers import ChartMarkers, to_chart_markers

NY = ZoneInfo("America/New_York")


def _make_df_ny(start_ny: datetime, bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
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
    """NY 10:00 시작 60분 → London Close (10:00-12:00) + Silver Bullet AM (10:00-11:00).

    변형 2 정통화 (NY AM = 07:00-10:00, London Close = 10:00-12:00) 적용 후,
    10:00-11:00 봉은 london_close 로 분류됨. Silver Bullet AM 윈도우 자체는
    그대로 10:00-11:00 이라 겹침 발생.
    """
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [(100, 101, 99, 100) for _ in range(60)])
    markers = to_chart_markers(df)
    assert len(markers.killzones) >= 1
    # 10:00 ~ 10:59 NY → london_close (10:00-12:00 안)
    lc_zones = [k for k in markers.killzones if k.name == "london_close"]
    assert len(lc_zones) >= 1
    # Silver Bullet AM (10-11am) 시그널 — london_close 첫 시간이 SB 와 겹침
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


def test_markers_includes_order_blocks() -> None:
    """OB indicator 결과가 markers.order_blocks 로 노출 (LuxAlgo 알고리즘 기준).

    LuxAlgo 표준 swing length = 5 — markers.py 는 internal_pivots (left=right=5) 를
    OB 검출에 사용. 따라서 fixture 는 13봉 이상이어야 swing 형성됨.
    """
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 101, 99, 100),    # 0
        (100, 102, 99, 101),    # 1
        (101, 103, 100, 102),   # 2
        (102, 104, 101, 103),   # 3
        (103, 105, 102, 104),   # 4
        (104, 110, 103, 109),   # 5
        (109, 112, 108, 111),   # 6 internal swing high (high=112)
        (111, 110, 100, 102),   # 7 bearish, range=10
        (102, 108, 95, 97),     # 8 bearish, range=13 ← OB
        (97, 105, 95, 96),      # 9 bearish, range=10
        (96, 105, 95, 96),      # 10
        (96, 105, 95, 96),      # 11
        (96, 130, 95, 128),     # 12 close=128 > 112 → CHoCH_BULLISH
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    bull = [o for o in markers.order_blocks if o.type == "bullish"]
    # ATR 필터로 가장 큰 range 봉(idx=8, range=13) 은 변동성 과대로 제외 →
    # 다음 후보 idx=9 (open=97, close=96) 가 채택됨.
    assert len(bull) >= 1
    assert bull[0].open == 97
    assert bull[0].close == 96


def test_markers_includes_macros() -> None:
    """NY 09:50–10:10 안의 봉 → am_macro_2 macro 마커."""
    # NY 09:55 시작 → 9봉 박으면 10:04 까지 = am_macro_2 (9:50-10:10) 구간 안
    start = datetime(2026, 5, 12, 9, 55, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 104, 101, 103),
        (103, 105, 102, 104),
        (104, 106, 103, 105),
        (105, 107, 104, 106),
        (106, 108, 105, 107),
        (107, 109, 106, 108),
        (108, 110, 107, 109),
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    names = [m.name for m in markers.macros]
    assert any("am_macro_2" == n for n in names)


def test_markers_to_dict_includes_new_keys() -> None:
    """to_dict 결과에 order_blocks / macros 키 포함."""
    df = _make_df_ny(datetime(2026, 5, 12, 10, 0, tzinfo=NY), [
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 104, 101, 103),
    ])
    d = to_chart_markers(df, fvg_min_size_pct=None).to_dict()
    assert "order_blocks" in d
    assert "macros" in d
    assert isinstance(d["order_blocks"], list)
    assert isinstance(d["macros"], list)


# ============================================================
# Internal vs Large scale (LuxAlgo 패턴)
# ============================================================


def test_markers_has_internal_and_large_scale_fields() -> None:
    """ChartMarkers 와 to_dict 에 두 스케일 필드가 모두 존재."""
    df = _make_df_ny(datetime(2026, 5, 12, 10, 0, tzinfo=NY), [
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 104, 101, 103),
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    # dataclass 필드
    assert hasattr(markers, "internal_swings")
    assert hasattr(markers, "internal_structure")
    assert hasattr(markers, "large_swings")
    assert hasattr(markers, "large_structure")
    # to_dict
    d = markers.to_dict()
    for key in ("internal_swings", "internal_structure", "large_swings", "large_structure"):
        assert key in d
        assert isinstance(d[key], list)


def test_markers_internal_scale_detects_small_swings() -> None:
    """left=right=5 internal scale — 11봉 박은 명확한 swing high 박은 거 검출."""
    # idx=5 가 11봉 안에서 max high → swing high (left=5, right=5)
    bars = []
    for i in range(11):
        if i == 5:
            bars.append((100, 200, 99, 199))   # 명확한 peak
        else:
            bars.append((100, 105, 99, 104))
    df = _make_df_ny(datetime(2026, 5, 12, 10, 0, tzinfo=NY), bars)
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    internal_highs = [s for s in markers.internal_swings if s.type == "high"]
    assert len(internal_highs) >= 1
    assert any(s.price == 200.0 for s in internal_highs)


def test_markers_large_scale_requires_long_df() -> None:
    """left=right=50 large scale — 짧은 df 에서는 large_swings 가 비어있음."""
    df = _make_df_ny(datetime(2026, 5, 12, 10, 0, tzinfo=NY), [
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 104, 101, 103),
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    # 3봉으로는 left+right+1 = 101봉 필요한 large scale 검출 불가
    assert markers.large_swings == []
    assert markers.large_structure == []


def test_markers_structure_has_swing_ts() -> None:
    """StructureMarker 에 swing_ts_ms 가 채워져야 segment 시각화 가능."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 102, 99, 101),
        (101, 108, 100, 107),   # swing high 108
        (107, 106, 100, 102),
        (102, 105, 95, 96),     # swing low 95
        (96, 110, 95.5, 109),
        (109, 115, 108, 115),   # close=115 > 108 → BOS bullish
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=None, min_rr=0.5)
    bull = [
        e for e in markers.structure
        if e.type in ("bos_bullish", "choch_bullish")
    ]
    assert len(bull) >= 1
    # swing_ts_ms 가 채워져야 함 (0 = 미설정)
    for ev in bull:
        assert ev.swing_ts_ms > 0, "swing_ts_ms 가 채워지지 않음 — segment 불가"
        # 시작 swing 은 돌파 봉보다 이전
        assert ev.swing_ts_ms < ev.ts_ms


def test_markers_equal_levels_swing_ts_list() -> None:
    """EqualLevelMarker.swing_ts_list 에 각 swing ts 들어있어야 함."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (100, 102, 99, 101),
        (101, 108.00, 100, 107),
        (107, 105, 100, 102),
        (102, 108.05, 101, 105),
        (105, 106, 100, 101),
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    eqh = [e for e in markers.equal_levels if e.type == "high"]
    assert len(eqh) >= 1
    # 두 swing 이상 박혀있고 ts 도 채워져 있어야 함
    assert len(eqh[0].swing_ts_list) >= 2
    # 모든 ts 가 양수
    assert all(t > 0 for t in eqh[0].swing_ts_list)


def test_markers_includes_equal_levels() -> None:
    """tolerance 안의 swing high 가 2개 박혀있으면 EQH 로 묶임."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    # 두 swing high 가 거의 같은 가격 (108 vs 108.05) 박혀있는 fixture
    df = _make_df_ny(start, [
        (100, 102, 99, 101),
        (101, 108, 100, 107),     # swing high 108 (idx=1)
        (107, 105, 100, 102),
        (102, 108.05, 101, 105),  # swing high 108.05 (idx=3) — 0.05% 차이
        (105, 106, 100, 101),
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    eqh = [e for e in markers.equal_levels if e.type == "high"]
    assert len(eqh) >= 1
    # 평균 가격은 108 근처
    assert abs(eqh[0].price - 108.025) < 0.1


def test_markers_equal_levels_in_to_dict() -> None:
    """to_dict 결과에 equal_levels 키 존재."""
    df = _make_df_ny(datetime(2026, 5, 12, 10, 0, tzinfo=NY), [
        (100, 102, 99, 101),
        (101, 103, 100, 102),
        (102, 104, 101, 103),
    ])
    d = to_chart_markers(df, fvg_min_size_pct=None).to_dict()
    assert "equal_levels" in d
    assert isinstance(d["equal_levels"], list)


def test_markers_internal_scale_is_subset_of_basic() -> None:
    """left=5 의 swing 은 left=1 보다 보통 적거나 같음 (더 strict)."""
    bars = []
    for i in range(20):
        if i % 4 == 0:
            bars.append((100, 110, 95, 105))
        else:
            bars.append((100, 102, 99, 101))
    df = _make_df_ny(datetime(2026, 5, 12, 10, 0, tzinfo=NY), bars)
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    # internal (left=5) 는 기본 (left=1) 보다 swing 수가 같거나 적음
    assert len(markers.internal_swings) <= len(markers.swings)


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


# ============================================================
# IFVG (Inversion FVG)
# ============================================================


def test_markers_exposes_ifvg() -> None:
    """bullish FVG 깨지면 bearish IFVG 가 markers.ifvgs 에 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [
        (95, 100, 94, 99),
        (99, 115, 99, 114),
        (114, 116, 105, 115),   # bullish FVG (100~105)
        (115, 118, 113, 117),
        (117, 119, 100, 99),    # close=99 < 100 → invalidated → IFVG bearish
    ])
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    assert len(markers.ifvgs) >= 1
    ifvg = markers.ifvgs[0]
    assert ifvg.type == "bearish"
    assert ifvg.low == 100.0
    assert ifvg.high == 105.0


def test_markers_ifvg_dict_serializable() -> None:
    """to_dict() 결과에 ifvgs key 포함."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [(100, 101, 99, 100)] * 5)
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    d = markers.to_dict()
    assert "ifvgs" in d
    assert isinstance(d["ifvgs"], list)


# ============================================================
# Breaker Block
# ============================================================


def test_markers_breakers_dict_serializable() -> None:
    """to_dict() 결과에 breakers key 포함."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df = _make_df_ny(start, [(100, 101, 99, 100)] * 5)
    markers = to_chart_markers(df, fvg_min_size_pct=None)
    d = markers.to_dict()
    assert "breakers" in d
    assert isinstance(d["breakers"], list)
