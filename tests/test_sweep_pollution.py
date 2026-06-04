"""sweep 검출 오염 회귀 (#SWEEP-POLLUTION).

과거 ``detect_silver_bullet_setups`` 는 fvg 루프 안에서 매 setup 마다
``_sweep_confluence`` → ``detect_liquidity_sweeps(df, swings)`` 를 호출했다.
이게 공유 ``swings`` 를 in-place 로 ``swept=True`` 오염시켜, 뒤이은 fvg 의
``_next_liquidity_target`` 이 부적절한(더 먼) TP 를 골라 RR 필터에서 대량
탈락 → 유효 setup 이 부당하게 줄었다. 수정으로 sweep 을 루프 전 1회만 검출해
주입하도록 바꿨다. 이 회귀 테스트는 1회 호출로 고정한다.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

import aurora_ict.strategy.silver_bullet as sb
from aurora_ict.strategy.silver_bullet import detect_silver_bullet_setups

NY = ZoneInfo("America/New_York")


def _multi_fvg_df() -> pd.DataFrame:
    """여러 bullish FVG 가 나오는 합성 15봉 (LONG bias)."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    bars = [
        (100, 105, 99, 104), (104, 130, 103, 125), (125, 124, 100, 101),
        (101, 108, 95, 96), (96, 110, 95.5, 109), (109, 112, 92, 93),
        (93, 100, 92.5, 99), (99, 105, 98, 104), (104, 106, 100, 101),
        (101, 105, 99, 100), (100, 102, 99, 101), (101, 110, 100, 109),
        (109, 119, 108, 118), (118, 122, 115, 121), (121, 122, 110, 120),
    ]
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = pd.DatetimeIndex([start + timedelta(minutes=i) for i in range(len(rows))])
    return df


def test_sweep_detected_at_most_once_per_call(monkeypatch) -> None:
    """detect_silver_bullet_setups 1회 호출 → detect_liquidity_sweeps 최대 1회.

    루프 안에서 fvg 마다 재호출하면 공유 swings 가 누적 오염된다(원 버그).
    """
    calls = {"n": 0}
    orig = sb.detect_liquidity_sweeps

    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(sb, "detect_liquidity_sweeps", counting)
    detect_silver_bullet_setups(
        _multi_fvg_df(), min_rr=0.1, fvg_min_size_pct=0.001, disable_time_filter=True,
    )
    assert calls["n"] <= 1, (
        f"detect_liquidity_sweeps 가 {calls['n']}회 호출 — "
        "fvg 루프 내 반복 호출로 swings 오염 누적(회귀)"
    )


def test_setups_independent_of_fvg_count_growth() -> None:
    """같은 df 를 반복 호출해도 setup 결과가 결정적(오염 잔존 없음)."""
    df = _multi_fvg_df()
    r1 = detect_silver_bullet_setups(
        df, min_rr=0.1, fvg_min_size_pct=0.001, disable_time_filter=True)
    r2 = detect_silver_bullet_setups(
        df, min_rr=0.1, fvg_min_size_pct=0.001, disable_time_filter=True)
    assert [s.ts_ms for s in r1] == [s.ts_ms for s in r2]
    assert [s.take_profit for s in r1] == [s.take_profit for s in r2]
