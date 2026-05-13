"""ICT 지표 시각 회귀 — 의도적으로 명확한 ICT 패턴 fixture 가 정확히 검출되는지 검증.

이 파일의 목적:
    실제 사용자가 차트에서 OB / FVG / Premium-Discount 영역을 봤을 때
    "이게 정말 ICT 룰대로 검출된 것인가?" 라는 의문이 들지 않도록,
    명확한 ICT 케이스를 fixture 로 정의하고 회귀 테스트한다.

각 케이스는:
    1. 의도된 ICT 패턴을 OHLC 시퀀스로 명확하게 표현
    2. 어떤 indicator 가 어디서 검출되어야 하는지 단언
    3. 실패 시 메시지에 시각 비교 힌트 (LuxAlgo 차트와 대조)

LuxAlgo SMC TradingView 지표와 시각 비교는 ``docs/VALIDATION_GUIDE.md`` 참고.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from aurora_ict.api.markers import to_chart_markers
from aurora_ict.indicators.fvg import detect_fvgs
from aurora_ict.indicators.liquidity import detect_equal_levels, detect_liquidity_sweeps
from aurora_ict.indicators.order_block import detect_order_blocks
from aurora_ict.indicators.structure import detect_structure_events
from aurora_ict.indicators.swing_points import detect_swing_points
from aurora_ict.indicators.trailing_extremes import compute_trailing_extremes

NY = ZoneInfo("America/New_York")


def _df(start: datetime, bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex([start + timedelta(hours=i) for i in range(len(rows))])
    return df


# ============================================================
# Case 1: 명확한 Bullish FVG — 중간 봉 high 가 다음 봉 low 보다 아래
# ============================================================


def test_visual_case1_clear_bullish_fvg() -> None:
    """3봉 명확한 갭: 봉1 high=102, 봉3 low=110 → 102-110 영역 = bullish FVG.

    검증 시각 힌트: LuxAlgo 의 'Fair Value Gaps' 옵션 켜면 같은 가격대 박스 표시됨.
    """
    df = _df(datetime(2026, 5, 12, 10, 0, tzinfo=NY), [
        (100, 102, 99, 101),     # 봉0: high=102
        (101, 105, 100, 104),    # 봉1: 작은 봉
        (104, 115, 110, 114),    # 봉2: low=110 → 102~110 갭
        (114, 116, 113, 115),
        (115, 117, 114, 116),
    ])
    fvgs = detect_fvgs(df, min_size_pct=None)
    bullish = [f for f in fvgs if f.type.value == "bullish"]
    assert len(bullish) >= 1, (
        "명확한 갭 (high=102 → low=110) 박혀있는데 bullish FVG 검출 실패. "
        "fvg.py 의 detect_fvgs 로직 확인 필요."
    )
    fvg = bullish[0]
    assert fvg.low <= 105, f"FVG bottom 이상함: {fvg.low}"
    assert fvg.high >= 108, f"FVG top 이상함: {fvg.high}"


# ============================================================
# Case 2: 명확한 Bullish OB — bearish 봉 후 강한 displacement
# ============================================================


def test_visual_case2_clear_bullish_ob() -> None:
    """직전 bearish 봉 (105→90) 뒤 다음 봉 close 가 그 high (105) 위로 박힘.

    검증 시각 힌트: LuxAlgo 의 'Internal Order Blocks' 옵션 켜면 같은 위치 박스.
    """
    df = _df(datetime(2026, 5, 12, 10, 0, tzinfo=NY), [
        (100, 102, 99, 101),
        (105, 105, 88, 90),      # 봉1: bearish, high=105 — Bullish OB 후보
        (90, 115, 89, 113),      # 봉2: close=113 > 105 → displacement 확인
        (113, 117, 112, 116),
        (116, 118, 115, 117),
    ])
    obs = detect_order_blocks(df, displacement_bars=3, mark_mitigation=False)
    bullish = [o for o in obs if o.type.value == "bullish" and o.idx == 1]
    assert len(bullish) >= 1, (
        "명확한 displacement (bearish high=105 → 다음 close=113) 박혀있는데 "
        "Bullish OB 검출 실패. order_block.py 확인 필요."
    )


# ============================================================
# Case 3: Liquidity Sweep — swing high 위 wick 후 close 안쪽
# ============================================================


def test_visual_case3_clear_bsl_sweep() -> None:
    """swing high (108) 위로 wick (110) 박은 후 close (105) 가 swing 안쪽 복귀.

    검증 시각 힌트: LuxAlgo 의 swing high 위 'Sweep' 텍스트 + 우리 marker 'Sweep'.
    """
    df = _df(datetime(2026, 5, 12, 10, 0, tzinfo=NY), [
        (100, 102, 99, 101),
        (101, 108, 100, 107),    # swing high = 108 (idx=1)
        (107, 106, 100, 102),
        (102, 110, 101, 105),    # wick=110 > 108, close=105 < 108 → BSL sweep
    ])
    swings = detect_swing_points(df)
    sweeps = detect_liquidity_sweeps(df, swings)
    bearish = [s for s in sweeps if s.type.value == "bearish"]
    assert len(bearish) >= 1, (
        "swing high (108) 위 wick (110) + close (105) 박혀있는데 BSL sweep 실패. "
        "liquidity.py 확인 필요."
    )


# ============================================================
# Case 4: BOS Bullish — swing high 박은 거 박은 거 close 가 돌파
# ============================================================


def test_visual_case4_bos_bullish() -> None:
    """swing high (108) 박힘 → 이후 close (115) 가 108 위로 돌파 → BOS Bullish.

    검증 시각 힌트: LuxAlgo 차트에 BOS 가로선 + 'BOS' 라벨.
    """
    df = _df(datetime(2026, 5, 12, 10, 0, tzinfo=NY), [
        (100, 102, 99, 101),
        (101, 108, 100, 107),    # swing high 108 (idx=1)
        (107, 106, 100, 102),
        (102, 105, 95, 96),      # swing low 95 (idx=3) — BOS_BULLISH 박을 trend NONE → UP
        (96, 110, 95.5, 109),    # 다음 봉
        (109, 115, 108, 115),    # close=115 > 108 → BOS Bullish
    ])
    swings = detect_swing_points(df)
    events = detect_structure_events(df, swings)
    bos_or_choch_bull = [
        e for e in events
        if e.type.value in ("bos_bullish", "choch_bullish")
    ]
    assert len(bos_or_choch_bull) >= 1, (
        "swing high (108) 박힘 + close (115) 돌파 박혀있는데 BOS Bullish 검출 실패. "
        "structure.py 확인 필요."
    )


# ============================================================
# Case 5: EQH — 두 swing high 가 같은 가격대 (0.1% 이내)
# ============================================================


def test_visual_case5_equal_highs() -> None:
    """두 swing high (108.0 vs 108.05, 0.046% 차이) → EQH 클러스터.

    검증 시각 힌트: LuxAlgo 의 'EQH' 라벨 + 두 high 잇는 dotted line.
    """
    df = _df(datetime(2026, 5, 12, 10, 0, tzinfo=NY), [
        (100, 102, 99, 101),
        (101, 108.00, 100, 107),  # swing high 1: 108.00
        (107, 105, 100, 102),
        (102, 108.05, 101, 105),  # swing high 2: 108.05 (0.046% 차이)
        (105, 106, 100, 101),
    ])
    swings = detect_swing_points(df)
    levels = detect_equal_levels(swings, tolerance_pct=0.001, min_count=2)
    eqh = [lvl for lvl in levels if lvl.type.value == "high"]
    assert len(eqh) >= 1, (
        "두 swing high (108.00 / 108.05, 0.046% 차이) 박혀있는데 EQH 검출 실패. "
        "tolerance_pct=0.001 박으면 검출되어야 함."
    )
    avg = eqh[0].price
    assert 108.0 <= avg <= 108.1, f"EQH 평균 가격 이상함: {avg}"


# ============================================================
# Case 6: Premium/Discount/Equilibrium — trailing top/bottom 박은 비율
# ============================================================


def test_visual_case6_premium_discount_zones() -> None:
    """trailing top=130, bottom=80 → equilibrium=105, premium 95-100%, discount 0-5%.

    검증 시각 힌트: LuxAlgo 의 'Premium' (위 red) / 'Equilibrium' (gray) /
    'Discount' (아래 green) 박스와 같은 위치.
    """
    df = _df(datetime(2026, 5, 12, 10, 0, tzinfo=NY), [
        (100, 130, 99, 125),    # top high = 130
        (125, 128, 80, 85),     # bottom low = 80
        (85, 110, 82, 105),
        (105, 115, 95, 108),
    ])
    swings = detect_swing_points(df)
    events = detect_structure_events(df, swings)
    te = compute_trailing_extremes(df, swings, events)
    assert te is not None
    # trailing range 안의 PD zone 검증 — 130 (top) / 80 (bottom)
    assert te.top >= 128, f"trailing.top 이상함: {te.top} (130 근처여야)"
    assert te.bottom <= 82, f"trailing.bottom 이상함: {te.bottom} (80 근처여야)"
    # equilibrium = (top + bottom) / 2 ≈ 105
    equilibrium = (te.top + te.bottom) / 2
    assert 104 <= equilibrium <= 106, f"equilibrium 이상함: {equilibrium}"


# ============================================================
# Case 7: 전체 파이프라인 — to_chart_markers 박은 게 모든 필드 채움
# ============================================================


def test_visual_case7_full_pipeline_smoke() -> None:
    """충분히 긴 fixture 한 번에 박으면 모든 marker 종류가 검출되어야 함."""
    bars = [
        (100, 102, 99, 101), (101, 108, 100, 107), (107, 106, 100, 102),
        (102, 105, 95, 96),  (96, 110, 95.5, 109), (109, 115, 108, 114),
        (114, 116, 113, 115),(115, 117, 114, 116), (116, 118, 115, 117),
        (117, 120, 116, 119),(119, 122, 118, 121), (121, 123, 120, 122),
    ]
    df = _df(datetime(2026, 5, 12, 10, 0, tzinfo=NY), bars)
    markers = to_chart_markers(df, fvg_min_size_pct=None, min_rr=0.5)
    # 최소 한 종류 이상씩 검출되어야 — 빈 fixture 회귀 방지
    assert len(markers.swings) > 0, "swings 검출 0 — swing_points.py 회귀 의심"
    assert markers.trailing is not None, "trailing extremes 박혀있어야 함"
    # FVG / OB / structure 는 fixture 따라 0 일 수 있으니 hard-assert 안 함
    # to_dict 직렬화 OK
    d = markers.to_dict()
    for key in (
        "fvgs", "sweeps", "structure", "swings", "killzones", "setups",
        "order_blocks", "macros", "trailing",
        "internal_swings", "internal_structure", "large_swings", "large_structure",
        "equal_levels",
    ):
        assert key in d, f"to_dict 에 {key} 키 누락 — markers.py 회귀"
