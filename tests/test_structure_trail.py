"""StructureTrail 단위 테스트."""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from aurora_ict.bot.structure_trail import (
    TrailUpdate,
    compute_structure_trail,
)
from aurora_ict.strategy.silver_bullet import Direction

NY = ZoneInfo("America/New_York")


def _df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """OHLCV DataFrame — index = 1m 간격."""
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df.index = pd.DatetimeIndex(
        [start + timedelta(minutes=i) for i in range(len(rows))]
    )
    return df


# ============================================================
# 가드 조건
# ============================================================


def test_too_short_df_returns_none() -> None:
    df = _df([(100, 101, 99, 100)] * 3)
    result = compute_structure_trail(
        df, Direction.LONG, entry=100.0, current_stop_loss=95.0,
    )
    assert result is None


def test_no_swing_returns_none() -> None:
    """단조 증가/감소 — swing 없음."""
    df = _df([(100 + i, 101 + i, 99 + i, 100 + i) for i in range(20)])
    result = compute_structure_trail(
        df, Direction.LONG, entry=100.0, current_stop_loss=95.0,
    )
    assert result is None


# ============================================================
# LONG trail
# ============================================================


def test_long_trail_above_entry() -> None:
    """LONG 진입 100 → 새 swing low 105 형성 (entry 위) → SL 105 - buffer 로 이동."""
    df = _df([
        (100, 101, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),    # 상승
        (109, 115, 108, 114),
        (114, 120, 113, 119),    # swing high (no influence)
        (119, 119, 105, 106),    # swing low @ 105 (idx=5, low=105)
        (106, 112, 105.5, 111),
        (111, 117, 110, 116),
    ])
    result = compute_structure_trail(
        df, Direction.LONG, entry=100.0, current_stop_loss=95.0,
        buffer_ratio=0.001,
    )
    assert result is not None
    assert isinstance(result, TrailUpdate)
    # anchor swing low = 105, buffer = 105 * 0.001 = 0.105
    assert result.anchor_swing_price == 105.0
    assert result.new_stop_loss == 105.0 - 0.105


def test_long_swing_low_below_entry_returns_none() -> None:
    """LONG 인데 swing low 가 entry 이하 → trail 의미 없음 (break-even 미달)."""
    df = _df([
        (100, 101, 99, 100),
        (100, 102, 95, 96),      # swing low @ 95 (entry 이하)
        (96, 103, 95.5, 102),
        (102, 105, 100, 104),
        (104, 106, 101, 105),
    ])
    result = compute_structure_trail(
        df, Direction.LONG, entry=100.0, current_stop_loss=90.0,
    )
    assert result is None


def test_long_trail_no_regression() -> None:
    """새 trail SL 이 현재 SL 보다 낮으면 갱신 X (역행 금지)."""
    df = _df([
        (100, 101, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 115, 108, 114),
        (114, 120, 113, 119),
        (119, 119, 105, 106),    # swing low @ 105
        (106, 112, 105.5, 111),
    ])
    # current_stop_loss=108 (105 보다 높음) → 새 trail 105 < 108 이라 갱신 X
    result = compute_structure_trail(
        df, Direction.LONG, entry=100.0, current_stop_loss=108.0,
    )
    assert result is None


# ============================================================
# SHORT trail
# ============================================================


def test_short_trail_below_entry() -> None:
    """SHORT 진입 100 → 새 swing high 95 형성 → SL = 95 + buffer."""
    df = _df([
        (100, 101, 99, 100),
        (100, 101, 98, 99),
        (99, 100, 90, 91),       # 하락
        (91, 92, 85, 86),
        (86, 87, 80, 81),        # swing low (no influence)
        (81, 95, 81, 94),        # swing high @ 95 (idx=5, high=95)
        (94, 94, 88, 89),
        (89, 90, 83, 84),
    ])
    result = compute_structure_trail(
        df, Direction.SHORT, entry=100.0, current_stop_loss=105.0,
        buffer_ratio=0.001,
    )
    assert result is not None
    # anchor swing high = 95, buffer = 95 * 0.001 = 0.095
    assert result.anchor_swing_price == 95.0
    assert result.new_stop_loss == 95.0 + 0.095


def test_short_swing_high_above_entry_returns_none() -> None:
    """SHORT 인데 swing high 가 entry 이상 → break-even 미달."""
    df = _df([
        (100, 101, 99, 100),
        (100, 108, 99, 107),     # swing high @ 108 (entry 이상)
        (107, 109, 102, 103),
        (103, 105, 98, 99),
        (99, 100, 95, 96),
    ])
    result = compute_structure_trail(
        df, Direction.SHORT, entry=100.0, current_stop_loss=110.0,
    )
    assert result is None


def test_short_trail_no_regression() -> None:
    """SHORT 새 trail SL 이 현재 SL 보다 높으면 갱신 X."""
    df = _df([
        (100, 101, 99, 100),
        (100, 101, 98, 99),
        (99, 100, 90, 91),
        (91, 92, 85, 86),
        (86, 87, 80, 81),
        (81, 95, 81, 94),        # swing high @ 95
        (94, 94, 88, 89),
    ])
    # current_stop_loss=92 (95 보다 낮음) → 새 trail 95 > 92 이라 갱신 X
    result = compute_structure_trail(
        df, Direction.SHORT, entry=100.0, current_stop_loss=92.0,
    )
    assert result is None


# ============================================================
# TrailUpdate dataclass
# ============================================================


def test_trail_update_structure() -> None:
    update = TrailUpdate(
        new_stop_loss=104.9,
        anchor_swing_idx=5,
        anchor_swing_price=105.0,
    )
    assert update.new_stop_loss == 104.9
    assert update.anchor_swing_idx == 5
    assert update.anchor_swing_price == 105.0
