"""SMT Divergence — 7번째 ICT 지표 (상관 자산 다이버전스)."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.indicators.smt import (
    SmtType,
    detect_smt_divergence,
)
from aurora_ict.indicators.swing_points import (
    SwingPoint,
    SwingType,
    detect_swing_points,
)
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup


def _make_df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """1분봉 OHLC df 생성 (ts_ms index)."""
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    df.index = [i * 60_000 for i in range(len(rows))]
    return df


# ============================================================
# Edge cases
# ============================================================


def test_smt_empty_swings() -> None:
    df = _make_df([(100, 102, 99, 101)])
    assert detect_smt_divergence([], df) == []


def test_smt_single_swing_not_enough() -> None:
    """swing 1개 — pair 불가, 빈 리스트."""
    sw = [SwingPoint(ts_ms=0, type=SwingType.HIGH, price=100.0, idx=0)]
    df = _make_df([(100, 102, 99, 101)])
    assert detect_smt_divergence(sw, df) == []


def test_smt_empty_corr_df() -> None:
    sw = [
        SwingPoint(ts_ms=0, type=SwingType.HIGH, price=100.0, idx=0),
        SwingPoint(ts_ms=60_000, type=SwingType.HIGH, price=102.0, idx=1),
    ]
    empty = pd.DataFrame(columns=["high", "low"])
    assert detect_smt_divergence(sw, empty) == []


def test_smt_missing_columns_raises() -> None:
    sw = [
        SwingPoint(ts_ms=0, type=SwingType.HIGH, price=100.0, idx=0),
        SwingPoint(ts_ms=60_000, type=SwingType.HIGH, price=102.0, idx=1),
    ]
    df = pd.DataFrame({"open": [1, 2]})
    with pytest.raises(ValueError, match="missing columns"):
        detect_smt_divergence(sw, df)


# ============================================================
# Bearish SMT (swing high pair)
# ============================================================


def test_smt_bearish_main_hh_corr_lh() -> None:
    """main HH(↑) + corr LH(↓) → bearish SMT — corr 가 못 따라옴."""
    # main: 2개 swing high (105 → 108, HH)
    main_swings = [
        SwingPoint(ts_ms=0, type=SwingType.HIGH, price=105.0, idx=1),
        SwingPoint(ts_ms=300_000, type=SwingType.HIGH, price=108.0, idx=5),
    ]
    # corr: 같은 시점 high (200 → 198, LH = divergence)
    corr_df = _make_df([
        (200, 200, 198, 199),    # idx 0  ts=0       high=200
        (199, 199, 197, 198),    # idx 1  ts=60_000
        (198, 199, 197, 198),    # idx 2  ts=120_000
        (198, 198, 196, 197),    # idx 3  ts=180_000
        (197, 198, 196, 197),    # idx 4  ts=240_000
        (197, 198, 196, 197),    # idx 5  ts=300_000  high=198 < 200
    ])
    events = detect_smt_divergence(main_swings, corr_df)
    assert len(events) == 1
    assert events[0].type is SmtType.BEARISH
    assert events[0].main_price == 108.0
    assert events[0].swing_type is SwingType.HIGH


def test_smt_bearish_main_lh_corr_hh() -> None:
    """main LH(↓) + corr HH(↑) → bearish SMT — main 이 못 따라옴 (대칭 case)."""
    main_swings = [
        SwingPoint(ts_ms=0, type=SwingType.HIGH, price=108.0, idx=1),
        SwingPoint(ts_ms=300_000, type=SwingType.HIGH, price=105.0, idx=5),  # LH
    ]
    corr_df = _make_df([
        (200, 200, 198, 199),
        (199, 199, 197, 198),
        (198, 199, 197, 198),
        (198, 198, 196, 197),
        (197, 203, 196, 202),
        (202, 205, 201, 204),  # idx 5: high=205 > 200 (corr HH)
    ])
    events = detect_smt_divergence(main_swings, corr_df)
    assert len(events) == 1
    assert events[0].type is SmtType.BEARISH


def test_smt_no_event_both_hh() -> None:
    """main HH + corr HH → 둘 다 강세 follow-through, divergence 없음."""
    main_swings = [
        SwingPoint(ts_ms=0, type=SwingType.HIGH, price=105.0, idx=1),
        SwingPoint(ts_ms=300_000, type=SwingType.HIGH, price=108.0, idx=5),
    ]
    corr_df = _make_df([
        (200, 200, 198, 199),
        (199, 199, 197, 198),
        (198, 199, 197, 198),
        (198, 198, 196, 197),
        (197, 203, 196, 202),
        (202, 205, 201, 204),  # corr 도 HH
    ])
    assert detect_smt_divergence(main_swings, corr_df) == []


# ============================================================
# Bullish SMT (swing low pair)
# ============================================================


def test_smt_bullish_main_ll_corr_hl() -> None:
    """main LL(↓) + corr HL(↑) → bullish SMT — corr 가 더 강함."""
    main_swings = [
        SwingPoint(ts_ms=0, type=SwingType.LOW, price=95.0, idx=1),
        SwingPoint(ts_ms=300_000, type=SwingType.LOW, price=92.0, idx=5),  # LL
    ]
    corr_df = _make_df([
        (200, 201, 195, 198),    # idx 0  low=195
        (198, 199, 197, 198),
        (198, 199, 197, 198),
        (198, 198, 196, 197),
        (197, 198, 196, 197),
        (197, 198, 196, 197),    # idx 5  low=196 > 195 (HL)
    ])
    events = detect_smt_divergence(main_swings, corr_df)
    assert len(events) == 1
    assert events[0].type is SmtType.BULLISH
    assert events[0].main_price == 92.0
    assert events[0].swing_type is SwingType.LOW


def test_smt_no_event_both_ll() -> None:
    """main LL + corr LL → 둘 다 약세 follow-through, divergence 없음."""
    main_swings = [
        SwingPoint(ts_ms=0, type=SwingType.LOW, price=95.0, idx=1),
        SwingPoint(ts_ms=300_000, type=SwingType.LOW, price=92.0, idx=5),
    ]
    corr_df = _make_df([
        (200, 201, 195, 198),
        (198, 199, 197, 198),
        (198, 199, 197, 198),
        (198, 198, 196, 197),
        (197, 198, 192, 195),
        (195, 198, 190, 193),  # corr 도 LL (low=190 < 195)
    ])
    assert detect_smt_divergence(main_swings, corr_df) == []


# ============================================================
# 통합 — detect_swing_points → detect_smt_divergence
# ============================================================


def test_smt_pipeline_integration() -> None:
    """실제 OHLC → swing detect → SMT divergence 전체 흐름."""
    # main: swing high 박은 2개 (105 → 108 HH)
    main_df = _make_df([
        (100, 102, 99, 101),
        (101, 105, 100, 103),    # swing high (105) idx=1
        (103, 104, 100, 101),
        (101, 103, 100, 102),
        (102, 106, 101, 105),
        (105, 108, 104, 107),    # swing high (108) idx=5 — HH
        (107, 107, 105, 106),
    ])
    swings = detect_swing_points(main_df)
    # 적어도 swing high 2개 박혀있어야 함
    sh = [s for s in swings if s.type is SwingType.HIGH]
    assert len(sh) >= 2

    # corr: 같은 시간대에 high 박힌 게 LH (200 → 198)
    corr_df = _make_df([
        (200, 201, 199, 200),
        (200, 203, 199, 201),    # idx=1: high=203 (corr swing high 1)
        (201, 202, 199, 200),
        (200, 201, 199, 200),
        (200, 202, 199, 201),
        (201, 201, 199, 200),    # idx=5: high=201 < 203 (corr LH)
        (200, 200, 198, 199),
    ])
    events = detect_smt_divergence(swings, corr_df)
    bearish = [e for e in events if e.type is SmtType.BEARISH]
    assert len(bearish) >= 1


# ============================================================
# bot 통합 — _apply_smt_boost (#SMT 2026-06-06, confluence 가점 배선)
# ============================================================


def _main_df_ll() -> pd.DataFrame:
    """main(BTC) swing low pair = LL (95 → 92). swing high 는 idx2 단일(pair 없음)."""
    return _make_df([
        (100, 100, 99, 100),
        (98, 98, 95, 96),     # swing low 1 = 95
        (100, 100, 99, 100),  # swing high (단일)
        (98, 98, 92, 93),     # swing low 2 = 92 (LL)
        (100, 100, 99, 100),
        (100, 100, 99, 100),
        (100, 100, 99, 100),
    ])


def _corr_rows_hl() -> list[list[Any]]:
    """corr(ETH) — 같은 시점 low 가 HL (195 → 198): main LL 과 divergence → bullish."""
    lows = [199, 195, 199, 198, 199, 199, 199]
    return [[i * 60_000, 200.0, 201.0, lo, 200.0, 100.0] for i, lo in enumerate(lows)]


def _setup(direction: Direction) -> SilverBulletSetup:
    return SilverBulletSetup(
        ts_ms=0, direction=direction, window="am_sb",
        entry=100.0, stop_loss=99.0, take_profit=103.0, risk_reward=3.0,
    )


@pytest.mark.asyncio
async def test_apply_smt_boost_adds_score_on_match() -> None:
    """bullish SMT(main LL + corr HL) + LONG setup → confluence +1."""
    client = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=_corr_rows_hl())
    bot = BotIctInstance(client=client, symbol="BTCUSDT", smt_enabled=True)
    setup = _setup(Direction.LONG)
    await bot._apply_smt_boost(setup, _main_df_ll())
    assert setup.confluence_score == 1
    assert any(c.startswith("smt=bullish") for c in setup.confluences)
    # corr 심볼(ETHUSDT)로 fetch 했는지 확인.
    assert client.fetch_ohlcv.await_args_list[0].args[0] == "ETHUSDT"


@pytest.mark.asyncio
async def test_apply_smt_boost_no_score_on_mismatch() -> None:
    """bullish SMT + SHORT setup → 가점 없음 (방향 불일치)."""
    client = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=_corr_rows_hl())
    bot = BotIctInstance(client=client, symbol="BTCUSDT", smt_enabled=True)
    setup = _setup(Direction.SHORT)
    await bot._apply_smt_boost(setup, _main_df_ll())
    assert setup.confluence_score == 0


@pytest.mark.asyncio
async def test_apply_smt_boost_skips_when_disabled() -> None:
    """smt_enabled=False → corr fetch 자체를 안 하고 가점 없음."""
    client = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=_corr_rows_hl())
    bot = BotIctInstance(client=client, symbol="BTCUSDT", smt_enabled=False)
    setup = _setup(Direction.LONG)
    await bot._apply_smt_boost(setup, _main_df_ll())
    assert setup.confluence_score == 0
    client.fetch_ohlcv.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_smt_boost_skips_unpaired_symbol() -> None:
    """상관 짝 없는 알트 심볼 → SMT 적용 안 함 (가점 0, fetch 0)."""
    client = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=_corr_rows_hl())
    bot = BotIctInstance(client=client, symbol="SOLUSDT", smt_enabled=True)
    setup = _setup(Direction.LONG)
    await bot._apply_smt_boost(setup, _main_df_ll())
    assert setup.confluence_score == 0
    client.fetch_ohlcv.assert_not_awaited()
