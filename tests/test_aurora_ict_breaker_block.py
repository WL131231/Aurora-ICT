"""Breaker Block 단위 테스트 — OB 가 close 로 깨졌을 때 반전 zone."""

from __future__ import annotations

import pandas as pd

from aurora_ict.indicators.order_block import (
    BreakerBlock,
    OrderBlock,
    OrderBlockType,
    detect_breaker_blocks,
)


def _make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = [i * 60_000 for i in range(len(rows))]
    return df


def _mk_ob(
    ob_type: OrderBlockType,
    *,
    idx: int = 2,
    disp_idx: int = 3,
    high: float = 105.0,
    low: float = 95.0,
) -> OrderBlock:
    return OrderBlock(
        ts_ms=idx * 60_000,
        type=ob_type,
        open=100.0,
        high=high,
        low=low,
        close=100.0,
        idx=idx,
        displacement_idx=disp_idx,
    )


def test_bb_empty_when_no_obs() -> None:
    df = _make_df([(100, 101, 99, 100)] * 5)
    assert detect_breaker_blocks([], df) == []


def test_bb_bullish_ob_close_break_becomes_bearish_breaker() -> None:
    """Bullish OB 의 low 를 close 로 깨면 Bearish Breaker."""
    df = _make_df([
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 105, 95, 100),    # idx=2 OB 후보
        (100, 110, 100, 109),   # idx=3 displacement
        (109, 110, 105, 108),
        (108, 109, 90, 93),     # idx=5 close=93 < OB.low=95 → break
    ])
    obs = [_mk_ob(OrderBlockType.BULLISH, idx=2, disp_idx=3, high=105, low=95)]
    bbs = detect_breaker_blocks(obs, df)
    assert len(bbs) == 1
    bb = bbs[0]
    assert bb.type is OrderBlockType.BEARISH
    assert bb.high == 105
    assert bb.low == 95
    assert bb.broken_idx == 5
    assert bb.origin_ob_idx == 2


def test_bb_bearish_ob_close_break_becomes_bullish_breaker() -> None:
    """Bearish OB 의 high 를 close 로 깨면 Bullish Breaker."""
    df = _make_df([
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 105, 95, 100),    # idx=2 OB
        (100, 101, 90, 91),     # idx=3 displacement down
        (91, 95, 89, 92),
        (92, 110, 91, 108),     # idx=5 close=108 > OB.high=105 → break
    ])
    obs = [_mk_ob(OrderBlockType.BEARISH, idx=2, disp_idx=3, high=105, low=95)]
    bbs = detect_breaker_blocks(obs, df)
    assert len(bbs) == 1
    assert bbs[0].type is OrderBlockType.BULLISH


def test_bb_wick_only_not_breaker() -> None:
    """wick 만 OB 영역 침범 + close 는 안에 — Breaker 아님 (strict close break 필요)."""
    df = _make_df([
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 105, 95, 100),
        (100, 110, 100, 109),
        (109, 110, 90, 100),    # wick low=90 < OB.low=95, 그러나 close=100 안 깸
        (100, 102, 99, 101),
    ])
    obs = [_mk_ob(OrderBlockType.BULLISH, idx=2, disp_idx=3, high=105, low=95)]
    bbs = detect_breaker_blocks(obs, df)
    assert bbs == []


def test_bb_no_break_returns_empty() -> None:
    """break 안 일어남 → BB 없음."""
    df = _make_df([
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 105, 95, 100),
        (100, 110, 100, 109),
        (109, 115, 105, 113),
    ])
    obs = [_mk_ob(OrderBlockType.BULLISH, idx=2, disp_idx=3, high=105, low=95)]
    bbs = detect_breaker_blocks(obs, df)
    assert bbs == []


def test_bb_mean_property() -> None:
    bb = BreakerBlock(
        ts_ms=0,
        type=OrderBlockType.BEARISH,
        high=110.0,
        low=100.0,
        broken_idx=5,
        origin_ob_idx=2,
    )
    assert bb.mean == 105.0
