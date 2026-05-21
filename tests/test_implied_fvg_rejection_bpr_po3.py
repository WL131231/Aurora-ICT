"""그룹 2 신규 indicators 단위 테스트 — Implied FVG / Rejection / BPR / Classic Po3.

CLAUDE.md mock 0 정책 — 결정론적 OHLCV 입력만.
"""
from __future__ import annotations

import pandas as pd

from aurora_ict.indicators.bpr import detect_bpr
from aurora_ict.indicators.fvg import FVG, FVGType
from aurora_ict.indicators.implied_fvg import (
    ImpliedFVGType,
    detect_implied_fvgs,
)
from aurora_ict.indicators.rejection_block import (
    RejectionBlockType,
    detect_rejection_blocks,
)
from aurora_ict.timing.power_of_3 import (
    ClassicPo3Type,
    classify_classic_po3_day,
)

# ============================================================
# Implied FVG
# ============================================================


def test_implied_fvg_bullish():
    """Bullish: bar1 body high < bar3 body low + wick overlap."""
    # bar1: open 95 close 90 (body 90-95) wick 90-100
    # bar2: 중간 봉 (영향 X)
    # bar3: open 100 close 105 (body 100-105) wick 95-105
    # body gap: 95 < 100 (bullish implied 조건 OK)
    # wick overlap: bar1.wick_high=100 >= bar3.wick_low=95 OK
    bars = [
        {"open": 95, "high": 100, "low": 90, "close": 90},
        {"open": 95, "high": 98, "low": 92, "close": 97},
        {"open": 100, "high": 105, "low": 95, "close": 105},
    ]
    df = pd.DataFrame(bars)
    fvgs = detect_implied_fvgs(df)
    assert len(fvgs) == 1
    assert fvgs[0].type is ImpliedFVGType.BULLISH
    assert fvgs[0].body_high == 100.0  # bar3 body low
    assert fvgs[0].body_low == 95.0    # bar1 body high
    assert fvgs[0].wick_overlap_width == 5.0  # 100 - 95


def test_implied_fvg_bearish():
    """Bearish: bar1 body low > bar3 body high + wick overlap."""
    bars = [
        {"open": 100, "high": 110, "low": 95, "close": 105},
        {"open": 103, "high": 108, "low": 98, "close": 100},
        {"open": 95, "high": 105, "low": 90, "close": 90},
    ]
    df = pd.DataFrame(bars)
    fvgs = detect_implied_fvgs(df)
    assert len(fvgs) == 1
    assert fvgs[0].type is ImpliedFVGType.BEARISH


def test_implied_fvg_no_overlap_means_normal_fvg_not_implied():
    """wick 도 gap (overlap X) 이면 일반 FVG, Implied 아님 → 검출 X."""
    bars = [
        {"open": 95, "high": 100, "low": 90, "close": 100},
        {"open": 102, "high": 104, "low": 101, "close": 103},
        {"open": 110, "high": 115, "low": 105, "close": 115},  # wick low 105 > bar1 wick high 100
    ]
    df = pd.DataFrame(bars)
    assert detect_implied_fvgs(df) == []


def test_implied_fvg_empty_or_short():
    assert detect_implied_fvgs(pd.DataFrame()) == []
    short_df = pd.DataFrame([{"open": 1, "high": 1, "low": 1, "close": 1}, {"open": 2, "high": 2, "low": 2, "close": 2}])
    assert detect_implied_fvgs(short_df) == []


# ============================================================
# Rejection Block
# ============================================================


def test_rejection_block_bullish():
    """아래쪽 wick + confirm move 위로 → Bullish Rejection Block."""
    bars = [
        # idx 0: bearish 봉 (open 100, close 95), wick low 80 (아래쪽 큰 wick)
        # body_low = 95, wick_width = 95 - 80 = 15, bar_range = 105 - 80 = 25
        # wick_ratio = 15/25 = 0.6 (>= 0.5 OK)
        {"open": 100, "high": 105, "low": 80, "close": 95},
        # idx 1-3: 위로 움직임 (confirm)
        {"open": 95, "high": 110, "low": 92, "close": 108},
        {"open": 108, "high": 120, "low": 105, "close": 118},
        {"open": 118, "high": 125, "low": 115, "close": 122},
    ]
    df = pd.DataFrame(bars)
    blocks = detect_rejection_blocks(df, min_wick_ratio=0.5, min_confirm_move=1.5)
    assert len(blocks) >= 1
    b = blocks[0]
    assert b.type is RejectionBlockType.BULLISH
    assert b.idx == 0
    assert b.wick_high == 95.0   # body_low
    assert b.wick_low == 80.0


def test_rejection_block_bearish():
    """위쪽 wick + confirm move 아래로 → Bearish Rejection Block."""
    bars = [
        # idx 0: bullish 봉 (open 95, close 100), wick high 120 (위 큰 wick)
        # body_high=100, wick_width=120-100=20, bar_range=120-90=30, ratio=0.66
        {"open": 95, "high": 120, "low": 90, "close": 100},
        {"open": 100, "high": 102, "low": 85, "close": 87},
        {"open": 87, "high": 90, "low": 75, "close": 78},
        {"open": 78, "high": 82, "low": 70, "close": 72},
    ]
    df = pd.DataFrame(bars)
    blocks = detect_rejection_blocks(df, min_wick_ratio=0.5, min_confirm_move=1.5)
    assert len(blocks) >= 1
    assert blocks[0].type is RejectionBlockType.BEARISH


def test_rejection_block_low_wick_ratio_skipped():
    """wick 비율 부족하면 skip."""
    # wick 작은 일반 candle
    bars = [
        {"open": 100, "high": 102, "low": 98, "close": 95},   # wick 거의 없음
        {"open": 95, "high": 108, "low": 92, "close": 105},
    ]
    df = pd.DataFrame(bars)
    assert detect_rejection_blocks(df, min_wick_ratio=0.5) == []


def test_rejection_block_no_confirm_move_skipped():
    """confirm move 부족하면 skip."""
    bars = [
        {"open": 100, "high": 105, "low": 80, "close": 95},   # 큰 아래 wick
        {"open": 95, "high": 96, "low": 93, "close": 95},     # 위로 움직임 미미
        {"open": 95, "high": 96, "low": 94, "close": 95},
    ]
    df = pd.DataFrame(bars)
    # confirm_move = (96 - 95) / 15 = 0.067 << 1.5 → skip
    assert detect_rejection_blocks(df, min_confirm_move=1.5) == []


# ============================================================
# BPR (Balanced Price Range)
# ============================================================


def _make_fvg(idx: int, fvg_type: FVGType, high: float, low: float) -> FVG:
    """Test FVG factory."""
    return FVG(
        ts_ms=1000 * idx,
        type=fvg_type,
        high=high,
        low=low,
        idx=idx,
        filled=False,
        invalidated=False,
    )


def test_bpr_overlap_detection():
    """Bullish (90-100) + Bearish (95-105) overlap = 95-100."""
    fvgs = [
        _make_fvg(10, FVGType.BULLISH, high=100, low=90),
        _make_fvg(20, FVGType.BEARISH, high=105, low=95),
    ]
    bprs = detect_bpr(fvgs)
    assert len(bprs) == 1
    b = bprs[0]
    assert b.high == 100.0
    assert b.low == 95.0
    assert b.bullish_fvg_idx == 10
    assert b.bearish_fvg_idx == 20
    assert b.formed_at_idx == 20


def test_bpr_no_overlap_returns_empty():
    """Overlap 없으면 BPR 형성 X."""
    fvgs = [
        _make_fvg(10, FVGType.BULLISH, high=100, low=90),
        _make_fvg(20, FVGType.BEARISH, high=120, low=110),  # 110 > 100, gap
    ]
    assert detect_bpr(fvgs) == []


def test_bpr_distance_limit():
    """봉 거리 초과 시 매칭 X."""
    fvgs = [
        _make_fvg(10, FVGType.BULLISH, high=100, low=90),
        _make_fvg(100, FVGType.BEARISH, high=105, low=95),   # 거리 90봉
    ]
    bprs = detect_bpr(fvgs, max_pair_distance_bars=50)
    assert bprs == []


def test_bpr_mean_threshold():
    """mean_threshold = (high+low)/2."""
    fvgs = [
        _make_fvg(10, FVGType.BULLISH, high=100, low=90),
        _make_fvg(20, FVGType.BEARISH, high=105, low=95),
    ]
    bprs = detect_bpr(fvgs)
    assert bprs[0].mean_threshold == 97.5


# ============================================================
# Classic Po3
# ============================================================


def test_classic_po3_buy_day():
    """London 이 일일 open 아래로 sweep + 위는 안 침범 → Buy Day."""
    result = classify_classic_po3_day(
        midnight_open=100,
        london_low=98,      # 일일 open 아래 (2% sweep)
        london_high=100.5,  # 위는 거의 안 침범 (0.5%, threshold 미만)
        min_sweep_pct=0.01,
    )
    assert result.type is ClassicPo3Type.BUY_DAY
    assert result.london_extreme == 98.0


def test_classic_po3_sell_day():
    """London 이 위로 sweep + 아래는 안 침범 → Sell Day."""
    result = classify_classic_po3_day(
        midnight_open=100,
        london_low=99.5,    # 거의 안 침범
        london_high=102,    # 위로 2% sweep
        min_sweep_pct=0.01,
    )
    assert result.type is ClassicPo3Type.SELL_DAY
    assert result.london_extreme == 102.0


def test_classic_po3_neutral_both_swept():
    """양방향 sweep → 명확한 패턴 X → NEUTRAL."""
    result = classify_classic_po3_day(
        midnight_open=100,
        london_low=98,
        london_high=102,
        min_sweep_pct=0.001,
    )
    assert result.type is ClassicPo3Type.NEUTRAL
    assert result.london_extreme is None


def test_classic_po3_neutral_no_sweep():
    """양쪽 다 sweep 안 됨 → NEUTRAL."""
    result = classify_classic_po3_day(
        midnight_open=100,
        london_low=99.95,
        london_high=100.05,
        min_sweep_pct=0.001,
    )
    assert result.type is ClassicPo3Type.NEUTRAL
