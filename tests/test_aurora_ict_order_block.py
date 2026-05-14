"""Order Block — 8번째 ICT 지표 (FVG 와 양대 산맥)."""

from __future__ import annotations

import pandas as pd
import pytest

from aurora_ict.indicators.order_block import (
    OrderBlockType,
    detect_order_blocks,
)


def _make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """OHLC 1분봉 df 생성 (ts_ms index)."""
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = [i * 60_000 for i in range(len(rows))]
    return df


# ============================================================
# Edge cases
# ============================================================


def test_ob_empty_df() -> None:
    df = pd.DataFrame(columns=["open", "high", "low", "close"])
    assert detect_order_blocks(df) == []


def test_ob_too_few_bars() -> None:
    """displacement_bars + 1 미만 → 빈 리스트."""
    df = _make_df([(100, 102, 99, 101)])
    assert detect_order_blocks(df, displacement_bars=3) == []


def test_ob_missing_columns_raises() -> None:
    df = pd.DataFrame({"open": [1, 2, 3, 4, 5]})
    with pytest.raises(ValueError, match="missing columns"):
        detect_order_blocks(df)


def test_ob_doji_skipped() -> None:
    """doji 봉 (open == close) → OB 아님, 다음 봉 박힘 X."""
    df = _make_df([
        (100, 102, 99, 100),     # doji (open == close)
        (100, 110, 99, 109),
        (109, 112, 108, 111),
        (111, 113, 110, 112),
    ])
    obs = detect_order_blocks(df, displacement_bars=3)
    # 첫 봉 doji 라서 skip — 다른 봉 (1,2번) 박혀있지만 그 자체 OB 후보 박혀 있으면 OK
    # 핵심은 doji 가 OB 후보가 아니라는 것
    assert all(ob.idx != 0 for ob in obs)


# ============================================================
# Bullish OB
# ============================================================


def test_ob_bullish_basic() -> None:
    """bearish 봉 (idx=1) 이후 다음 봉 close > bearish high → Bullish OB."""
    df = _make_df([
        (100, 102, 99, 101),     # idx=0  bullish
        (101, 102, 95, 96),      # idx=1  bearish (close=96 < open=101), high=102
        (96, 108, 95, 107),      # idx=2  bullish, close=107 > 102 (bearish high) ✓
        (107, 110, 106, 109),    # idx=3
        (109, 111, 108, 110),    # idx=4
    ])
    obs = detect_order_blocks(
        df, algorithm="legacy", displacement_bars=3, mark_mitigation=False,
    )
    bull = [o for o in obs if o.type is OrderBlockType.BULLISH]
    assert any(o.idx == 1 for o in bull)
    # 발견된 bullish OB 의 displacement_idx = 2
    ob = next(o for o in bull if o.idx == 1)
    assert ob.displacement_idx == 2
    assert ob.open == 101
    assert ob.close == 96


def test_ob_bullish_no_displacement_within_window() -> None:
    """bearish 봉 이후 displacement_bars 안에 high 돌파 close 없음 → OB X."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 102, 95, 96),      # bearish, high=102
        (96, 101, 95, 100),      # close=100 < 102
        (100, 101.5, 99, 101),   # close=101 < 102
        (101, 101.9, 100, 101.5),  # close=101.5 < 102
    ])
    obs = detect_order_blocks(
        df, algorithm="legacy", displacement_bars=3, mark_mitigation=False,
    )
    # idx=1 bearish 봉 이후 displacement 안에 high 돌파 close 없음 → bullish OB X
    assert not any(o.idx == 1 and o.type is OrderBlockType.BULLISH for o in obs)


# ============================================================
# Bearish OB
# ============================================================


def test_ob_bearish_basic() -> None:
    """bullish 봉 (idx=1) 이후 다음 봉 close < bullish low → Bearish OB."""
    df = _make_df([
        (100, 101, 99, 100),     # idx=0
        (100, 108, 99, 107),     # idx=1  bullish (close=107 > open=100), low=99
        (107, 108, 95, 96),      # idx=2  bearish, close=96 < 99 (bullish low) ✓
        (96, 97, 90, 92),        # idx=3
        (92, 93, 89, 90),        # idx=4
    ])
    obs = detect_order_blocks(
        df, algorithm="legacy", displacement_bars=3, mark_mitigation=False,
    )
    bear = [o for o in obs if o.type is OrderBlockType.BEARISH]
    assert any(o.idx == 1 for o in bear)
    ob = next(o for o in bear if o.idx == 1)
    assert ob.displacement_idx == 2
    assert ob.open == 100
    assert ob.close == 107


# ============================================================
# Mitigation
# ============================================================


def test_ob_mitigation_marked() -> None:
    """Bullish OB 생성 후 wick low 가 OB low 미만 → mitigated=True (LuxAlgo wick 기준)."""
    df = _make_df([
        (100, 102, 99, 101),     # idx=0
        (101, 102, 95, 96),      # idx=1  bearish OB candidate (low=95)
        (96, 108, 95, 107),      # idx=2  bullish (close=107 > 102) → displacement
        (107, 109, 106, 108),    # idx=3
        (108, 110, 94, 98),      # idx=4  wick low=94 < OB low=95 → mitigated
    ])
    obs = detect_order_blocks(
        df, algorithm="legacy", displacement_bars=3, mark_mitigation=True,
    )
    bull = next(o for o in obs if o.type is OrderBlockType.BULLISH and o.idx == 1)
    assert bull.mitigated is True


def test_ob_no_mitigation_when_disabled() -> None:
    """mark_mitigation=False → mitigated 검사 X."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 102, 95, 96),
        (96, 108, 95, 107),
        (107, 109, 106, 108),
        (108, 110, 95, 98),
    ])
    obs = detect_order_blocks(
        df, algorithm="legacy", displacement_bars=3, mark_mitigation=False,
    )
    bull = next(o for o in obs if o.type is OrderBlockType.BULLISH and o.idx == 1)
    assert bull.mitigated is False


def test_ob_no_mitigation_if_price_doesnt_retest() -> None:
    """Bullish OB 후 가격이 계속 위로 → mitigated=False."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 102, 95, 96),
        (96, 108, 95, 107),
        (107, 115, 106, 114),    # close=114, 박힌 body 위
        (114, 120, 113, 119),    # close=119, OB body 위
    ])
    obs = detect_order_blocks(
        df, algorithm="legacy", displacement_bars=3, mark_mitigation=True,
    )
    bull = next(o for o in obs if o.type is OrderBlockType.BULLISH and o.idx == 1)
    assert bull.mitigated is False


# ============================================================
# body_top / body_bottom helpers
# ============================================================


def test_ob_atr_filter_default_on() -> None:
    """atr_filter=True 기본 — 정상 봉에서는 결과 동일 (변동성 normal)."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 102, 95, 96),
        (96, 108, 95, 107),
        (107, 109, 106, 108),
        (108, 110, 107, 109),
    ])
    obs_default = detect_order_blocks(
        df, algorithm="legacy", displacement_bars=3, mark_mitigation=False,
    )
    obs_off = detect_order_blocks(
        df, algorithm="legacy", displacement_bars=3, mark_mitigation=False,
        atr_filter=False,
    )
    # 작은 변동성 봉 → ATR filter on/off 결과 (idx, type) 동일해야 함
    assert {(o.idx, o.type) for o in obs_default} == {(o.idx, o.type) for o in obs_off}


def test_ob_atr_filter_off_keeps_legacy_behavior() -> None:
    """atr_filter=False → 변동성 큰 봉 그대로 판정 (legacy)."""
    df = _make_df([
        (100, 105, 95, 100),
        (100, 200, 50, 60),    # 매우 큰 변동성 봉 (high-low=150)
        (60, 220, 55, 215),    # close 215 > 200 → bullish OB 후보 (legacy 기준)
        (215, 225, 210, 220),
        (220, 230, 215, 225),
    ])
    obs = detect_order_blocks(
        df, algorithm="legacy", displacement_bars=3, mark_mitigation=False,
        atr_filter=False,
    )
    # legacy: bar_high=200 → close 215 > 200 OK → bullish OB
    bull = [o for o in obs if o.type is OrderBlockType.BULLISH and o.idx == 1]
    assert len(bull) == 1
    assert bull[0].high == 200.0


def test_ob_atr_filter_can_be_disabled() -> None:
    """atr_filter 파라미터 noop 케이스 — 결과 list 반환."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 102, 95, 96),
        (96, 108, 95, 107),
        (107, 109, 106, 108),
        (108, 110, 107, 109),
    ])
    # 두 호출 모두 예외 없이 list 반환
    assert isinstance(
        detect_order_blocks(df, algorithm="legacy", atr_filter=True), list,
    )
    assert isinstance(
        detect_order_blocks(df, algorithm="legacy", atr_filter=False), list,
    )


def test_ob_body_top_bottom() -> None:
    """body_top = max(open, close) / body_bottom = min(open, close).

    displacement_bars=3 + 5봉 → range(2) 즉 i=0, i=1 검사.
    """
    df = _make_df([
        (100, 102, 99, 101),
        (101, 102, 95, 96),     # idx=1  bearish OB (body 96-101)
        (96, 108, 95, 107),     # idx=2  displacement (close > 102)
        (107, 109, 106, 108),
        (108, 110, 107, 109),   # 5번째 봉 추가 — range(2) 만족
    ])
    obs = detect_order_blocks(
        df, algorithm="legacy", displacement_bars=3, mark_mitigation=False,
    )
    bull = next(o for o in obs if o.type is OrderBlockType.BULLISH and o.idx == 1)
    assert bull.body_top == 101
    assert bull.body_bottom == 96


# ============================================================
# LuxAlgo 알고리즘 (BOS 트리거 + swing 구간 max range bar)
# ============================================================


def test_ob_luxalgo_bullish_basic() -> None:
    """LuxAlgo: BOS up 발생 시 swing→BOS 구간의 max range bearish 봉이 Bullish OB."""
    df = _make_df([
        (98, 100, 97, 99),       # idx=0
        (99, 112, 98, 111),      # idx=1  swing high (high=112), bullish
        (111, 110, 100, 102),    # idx=2  bearish, range=10
        (102, 108, 95, 97),      # idx=3  bearish, range=13  ← max
        (97, 105, 95, 96),       # idx=4  bearish, range=10
        (96, 115, 95, 114),      # idx=5  close=114 > 112 → BOS_BULLISH
    ])
    obs = detect_order_blocks(
        df, algorithm="luxalgo", swing_length=1, mark_mitigation=False,
        atr_filter=False,
    )
    bull = [o for o in obs if o.type is OrderBlockType.BULLISH]
    assert len(bull) == 1
    assert bull[0].idx == 3
    assert bull[0].displacement_idx == 5
    # OB zone = 봉 아래 절반 (LuxAlgo): top = hl2 = (108+95)/2 = 101.5, bottom = low = 95
    assert bull[0].high == 101.5
    assert bull[0].low == 95


def test_ob_luxalgo_bearish_basic() -> None:
    """LuxAlgo: BOS down 발생 시 swing→BOS 구간의 max range bullish 봉이 Bearish OB."""
    df = _make_df([
        (102, 103, 99, 100),     # idx=0
        (100, 101, 88, 89),      # idx=1  swing low (low=88), bearish
        (89, 95, 89, 94),        # idx=2  bullish, range=6
        (94, 100, 90, 99),       # idx=3  bullish, range=10  ← max (첫 등장)
        (99, 105, 95, 104),      # idx=4  bullish, range=10
        (104, 105, 80, 81),      # idx=5  close=81 < 88 → BOS_BEARISH
    ])
    obs = detect_order_blocks(
        df, algorithm="luxalgo", swing_length=1, mark_mitigation=False,
        atr_filter=False,
    )
    bear = [o for o in obs if o.type is OrderBlockType.BEARISH]
    assert len(bear) == 1
    assert bear[0].idx == 3
    assert bear[0].displacement_idx == 5


def test_ob_luxalgo_no_event_no_ob() -> None:
    """LuxAlgo: BOS/CHoCH 이벤트 없으면 OB 없음."""
    df = _make_df([
        (100, 101, 99, 100),
        (100, 102, 98, 101),
        (101, 103, 99, 102),
        (102, 104, 100, 103),
        (103, 105, 101, 104),
    ])
    obs = detect_order_blocks(
        df, algorithm="luxalgo", swing_length=1, atr_filter=False,
    )
    assert obs == []


def test_ob_luxalgo_too_few_bars() -> None:
    """LuxAlgo: swing_length*2 + 1 미만 → 빈 리스트."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 102, 95, 96),
    ])
    obs = detect_order_blocks(df, algorithm="luxalgo", swing_length=5)
    assert obs == []


def test_ob_luxalgo_uses_provided_swings_events() -> None:
    """swings/events 명시 주입 시 OB 검출이 그 결과를 사용 (재계산 X)."""
    df = _make_df([
        (98, 100, 97, 99),
        (99, 112, 98, 111),
        (111, 110, 100, 102),
        (102, 108, 95, 97),
        (97, 105, 95, 96),
        (96, 115, 95, 114),
    ])
    # 빈 events 주입 → OB 도 빈 리스트
    obs_empty = detect_order_blocks(
        df, algorithm="luxalgo", swings=[], events=[], mark_mitigation=False,
    )
    assert obs_empty == []


def test_ob_luxalgo_mitigation_wick_based() -> None:
    """LuxAlgo OB 도 wick mitigation (bullish OB low 미만 wick → mitigated)."""
    df = _make_df([
        (98, 100, 97, 99),
        (99, 112, 98, 111),      # swing high
        (111, 110, 100, 102),    # bearish OB 후보 (range=10)
        (102, 108, 95, 97),      # bearish OB 후보 (range=13)  ← OB
        (97, 105, 95, 96),
        (96, 115, 95, 114),      # BOS_BULLISH at idx=5
        (114, 116, 94, 100),     # idx=6 wick low=94 < OB low=95 → mitigated
    ])
    obs = detect_order_blocks(
        df, algorithm="luxalgo", swing_length=1, mark_mitigation=True,
        atr_filter=False,
    )
    bull = next(o for o in obs if o.type is OrderBlockType.BULLISH)
    assert bull.idx == 3
    assert bull.mitigated is True
