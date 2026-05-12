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
    obs = detect_order_blocks(df, displacement_bars=3, mark_mitigation=False)
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
    obs = detect_order_blocks(df, displacement_bars=3, mark_mitigation=False)
    # idx=1 박힌 bearish 봉 박힌 거 박힌 bullish OB 박힘 X
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
    obs = detect_order_blocks(df, displacement_bars=3, mark_mitigation=False)
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
    """Bullish OB 생성 후 가격이 body 안 close → mitigated=True."""
    df = _make_df([
        (100, 102, 99, 101),     # idx=0
        (101, 102, 95, 96),      # idx=1  bearish OB candidate (body 96-101)
        (96, 108, 95, 107),      # idx=2  bullish (close=107 > 102) → displacement
        (107, 109, 106, 108),    # idx=3
        (108, 110, 95, 98),      # idx=4  close=98 박힘 body (96-101) 안 ✓
    ])
    obs = detect_order_blocks(df, displacement_bars=3, mark_mitigation=True)
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
    obs = detect_order_blocks(df, displacement_bars=3, mark_mitigation=False)
    bull = next(o for o in obs if o.type is OrderBlockType.BULLISH and o.idx == 1)
    assert bull.mitigated is False


def test_ob_no_mitigation_if_price_doesnt_retest() -> None:
    """Bullish OB 후 가격이 계속 위로 → mitigated=False."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 102, 95, 96),
        (96, 108, 95, 107),
        (107, 115, 106, 114),    # close=114, 박힌 body 위
        (114, 120, 113, 119),    # close=119, 박힌 body 위
    ])
    obs = detect_order_blocks(df, displacement_bars=3, mark_mitigation=True)
    bull = next(o for o in obs if o.type is OrderBlockType.BULLISH and o.idx == 1)
    assert bull.mitigated is False


# ============================================================
# body_top / body_bottom helpers
# ============================================================


def test_ob_body_top_bottom() -> None:
    """body_top = max(open, close) / body_bottom = min(open, close)."""
    df = _make_df([
        (100, 102, 99, 101),
        (101, 102, 95, 96),
        (96, 108, 95, 107),
        (107, 109, 106, 108),
    ])
    obs = detect_order_blocks(df, displacement_bars=3, mark_mitigation=False)
    bull = next(o for o in obs if o.type is OrderBlockType.BULLISH and o.idx == 1)
    assert bull.body_top == 101
    assert bull.body_bottom == 96
