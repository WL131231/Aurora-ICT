"""CISD (Change in State of Delivery) — 가격 전달 방향 전환 신호 검출 테스트.

mock 0 — 합성 OHLCV 로 결정론 검증. #CISD 2026-06-06 (누락 보강 2/4).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.indicators.cisd import CisdType, detect_cisd
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup


def _make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """OHLC df 생성 (ts_ms index). bars = [(open, high, low, close), ...]."""
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = [i * 60_000 for i in range(len(rows))]
    return df


def test_bullish_cisd_recovers_bearish_sequence_open() -> None:
    """직전 연속 하락 캔들의 첫 시초가 위로 현재 종가 마감 → BULLISH."""
    # 하락 3봉 (open>close): 100→99, 99→98, 98→97. 첫 하락봉 open=100.
    # 현재 봉이 100 위로 close → bullish CISD.
    df = _make_df([
        (100.0, 100.5, 98.5, 99.0),   # bearish (시퀀스 첫 봉, open=100)
        (99.0, 99.2, 97.5, 98.0),     # bearish
        (98.0, 98.2, 96.5, 97.0),     # bearish
        (97.0, 101.0, 96.8, 100.5),   # 현재 — close 100.5 > 100 (첫 시초가)
    ])
    assert detect_cisd(df) is CisdType.BULLISH


def test_bearish_cisd_breaks_bullish_sequence_open() -> None:
    """직전 연속 상승 캔들의 첫 시초가 아래로 현재 종가 마감 → BEARISH."""
    df = _make_df([
        (100.0, 101.5, 99.8, 101.0),  # bullish (시퀀스 첫 봉, open=100)
        (101.0, 102.5, 100.8, 102.0), # bullish
        (102.0, 103.5, 101.8, 103.0), # bullish
        (103.0, 103.2, 99.0, 99.5),   # 현재 — close 99.5 < 100 (첫 시초가)
    ])
    assert detect_cisd(df) is CisdType.BEARISH


def test_no_cisd_when_close_does_not_break_level() -> None:
    """하락 시퀀스지만 현재 종가가 첫 시초가를 못 넘으면 None."""
    df = _make_df([
        (100.0, 100.5, 98.5, 99.0),   # bearish (open=100)
        (99.0, 99.2, 97.5, 98.0),     # bearish
        (98.0, 99.5, 97.8, 99.2),     # 현재 — close 99.2 < 100 (미돌파)
    ])
    assert detect_cisd(df) is None


def test_single_bar_bearish_cisd() -> None:
    """직전 단일 하락 캔들만으로도 그 시초가 돌파 시 BULLISH (1캔들 micro)."""
    df = _make_df([
        (100.0, 100.2, 98.0, 98.5),   # bearish (open=100)
        (98.5, 101.0, 98.3, 100.5),   # 현재 — close 100.5 > 100
    ])
    assert detect_cisd(df) is CisdType.BULLISH


def test_too_short_df_returns_none() -> None:
    """봉 1개 이하면 None."""
    assert detect_cisd(_make_df([(100.0, 101.0, 99.0, 100.5)])) is None
    assert detect_cisd(_make_df([])) is None


def test_lookback_caps_sequence_scan() -> None:
    """max_lookback 으로 시퀀스 추적 범위 제한 — 한도 안의 첫 시초가만 level."""
    # 하락 5봉. max_lookback=2 면 직전 2봉만 봐서 level = bars[-3].open.
    df = _make_df([
        (105.0, 105.2, 104.0, 104.5),  # bearish
        (104.5, 104.7, 103.0, 103.5),  # bearish
        (103.5, 103.7, 102.0, 102.5),  # bearish — lookback=2 시 level 후보 open=103.5
        (102.5, 102.7, 101.0, 101.5),  # bearish (직전, open=102.5)
        (101.5, 104.0, 101.3, 103.8),  # 현재 — close 103.8
    ])
    # lookback=2: 직전(102.5) + 그 전(103.5) → level=103.5, close 103.8>103.5 → BULLISH
    assert detect_cisd(df, max_lookback=2) is CisdType.BULLISH
    # lookback=10(기본): level=105.0(첫 봉), close 103.8<105.0 → None
    assert detect_cisd(df) is None


# ============================================================
# bot 통합 — _apply_cisd_boost (방향 일치 시 confluence +1)
# ============================================================


def _bull_cisd_df() -> pd.DataFrame:
    """직전 하락 시퀀스 첫 시초가 상향 돌파 = bullish CISD."""
    return _make_df([
        (100.0, 100.5, 98.5, 99.0),
        (99.0, 99.2, 97.5, 98.0),
        (98.0, 98.2, 96.5, 97.0),
        (97.0, 101.0, 96.8, 100.5),
    ])


def _setup(direction: Direction) -> SilverBulletSetup:
    return SilverBulletSetup(
        ts_ms=0, direction=direction, window="am_sb",
        entry=100.0, stop_loss=99.0, take_profit=103.0, risk_reward=3.0,
    )


def test_apply_cisd_boost_adds_score_on_direction_match() -> None:
    """bullish CISD + LONG setup → confluence +1."""
    bot = BotIctInstance(client=AsyncMock(), symbol="BTCUSDT")
    setup = _setup(Direction.LONG)
    bot._apply_cisd_boost(setup, _bull_cisd_df())
    assert setup.confluence_score == 1
    assert any("cisd=bullish" in c for c in setup.confluences)


def test_apply_cisd_boost_no_score_on_direction_mismatch() -> None:
    """bullish CISD + SHORT setup → 가점 없음 (방향 불일치)."""
    bot = BotIctInstance(client=AsyncMock(), symbol="BTCUSDT")
    setup = _setup(Direction.SHORT)
    bot._apply_cisd_boost(setup, _bull_cisd_df())
    assert setup.confluence_score == 0
    assert setup.confluences == []
