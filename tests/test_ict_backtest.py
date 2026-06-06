"""ICT 백테스트 하니스 — 순수 시뮬 함수 결정론 검증 + 실데이터 smoke.

mock 0 — 합성 입력 + 실제 parquet 일부. #BACKTEST 2026-06-06.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aurora_ict.backtest.replay import (
    BacktestConfig,
    _aggregate,
    _gate_pass,
    _simulate_exit,
    load_ohlcv_parquet,
    run_backtest,
)
from aurora_ict.strategy.silver_bullet import Direction

_PARQUET = Path(__file__).resolve().parents[1] / "data" / "ETHUSDT_1m_sample.parquet"


# ============================================================
# _gate_pass — confluence 게이트 (bot step() 재현)
# ============================================================


def test_gate_pass_meets_min_confluence() -> None:
    cfg = BacktestConfig(min_confluence=2, high_rr_bypass_min_rr=2.3)
    assert _gate_pass(2, 1.5, cfg) is True   # score 충족
    assert _gate_pass(3, 1.5, cfg) is True


def test_gate_pass_high_rr_bypass() -> None:
    cfg = BacktestConfig(min_confluence=2, high_rr_bypass_min_rr=2.3)
    # score 1 미달이지만 rr 2.5 >= 2.3 + score>=1 → 통과
    assert _gate_pass(1, 2.5, cfg) is True
    # score 0 이면 bypass 불가
    assert _gate_pass(0, 9.0, cfg) is False
    # rr 미달 + score 미달 → 차단
    assert _gate_pass(1, 2.0, cfg) is False


def test_gate_pass_bypass_disabled() -> None:
    cfg = BacktestConfig(min_confluence=2, high_rr_bypass_min_rr=0.0)
    assert _gate_pass(1, 9.0, cfg) is False  # bypass 0 = 비활성


# ============================================================
# _simulate_exit — SL/TP 도달 판정
# ============================================================


def _arr(vals: list[float]) -> np.ndarray:
    return np.array(vals, dtype=float)


def test_simulate_exit_long_hits_tp() -> None:
    cfg = BacktestConfig()
    # entry_idx=0, long, sl=99 tp=105. idx1 high 106 >= tp → tp.
    highs = _arr([100, 106, 107])
    lows = _arr([100, 100, 100])
    closes = _arr([100, 104, 106])
    idx, px, outcome = _simulate_exit(highs, lows, closes, 0, Direction.LONG, 99, 105, cfg)
    assert (idx, px, outcome) == (1, 105, "tp")


def test_simulate_exit_long_hits_sl() -> None:
    cfg = BacktestConfig()
    # idx1 low 98 <= sl 99 → sl.
    highs = _arr([100, 101, 102])
    lows = _arr([100, 98, 97])
    closes = _arr([100, 99, 98])
    idx, px, outcome = _simulate_exit(highs, lows, closes, 0, Direction.LONG, 99, 110, cfg)
    assert (idx, px, outcome) == (1, 99, "sl")


def test_simulate_exit_sl_priority_on_same_bar() -> None:
    cfg = BacktestConfig(sl_priority=True)
    # idx1 에서 high 110(tp) + low 98(sl) 동시 → SL 우선(보수).
    highs = _arr([100, 110, 110])
    lows = _arr([100, 98, 98])
    closes = _arr([100, 105, 105])
    idx, px, outcome = _simulate_exit(highs, lows, closes, 0, Direction.LONG, 99, 108, cfg)
    assert outcome == "sl"


def test_simulate_exit_short_hits_tp() -> None:
    cfg = BacktestConfig()
    # short: tp 아래. idx1 low 94 <= tp 95 → tp.
    highs = _arr([100, 101, 102])
    lows = _arr([100, 94, 93])
    closes = _arr([100, 96, 95])
    idx, px, outcome = _simulate_exit(highs, lows, closes, 0, Direction.SHORT, 105, 95, cfg)
    assert (idx, px, outcome) == (1, 95, "tp")


def test_simulate_exit_eod_when_no_touch() -> None:
    cfg = BacktestConfig()
    # SL/TP 둘 다 안 닿음 → 마지막 봉 close 로 강제 청산.
    highs = _arr([100, 101, 102])
    lows = _arr([100, 99.5, 99.6])
    closes = _arr([100, 100.5, 101.0])
    idx, px, outcome = _simulate_exit(highs, lows, closes, 0, Direction.LONG, 90, 110, cfg)
    assert (idx, outcome) == (2, "eod")
    assert px == 101.0


# ============================================================
# _aggregate
# ============================================================


def test_aggregate_empty() -> None:
    res = _aggregate(BacktestConfig(), [])
    assert res.n_trades == 0
    assert res.win_rate == 0.0
    assert res.total_net_pnl_pct == 0.0


# ============================================================
# load_ohlcv_parquet + run_backtest smoke (실데이터 일부)
# ============================================================


@pytest.mark.skipif(not _PARQUET.exists(), reason="sample parquet 없음")
def test_load_parquet_shape() -> None:
    df = load_ohlcv_parquet(_PARQUET)
    assert {"open", "high", "low", "close"}.issubset(df.columns)
    assert len(df) > 0
    # DatetimeIndex (ts 기반) — detect 의 macro confluence 가 ts 필요.
    assert str(df.index.dtype).startswith("datetime64")


@pytest.mark.skipif(not _PARQUET.exists(), reason="sample parquet 없음")
def test_run_backtest_smoke() -> None:
    """실데이터 일부로 run_backtest 가 크래시 없이 집계 산출 + 일관성."""
    df = load_ohlcv_parquet(_PARQUET).iloc[:1500]
    cfg = BacktestConfig(window=200)
    res = run_backtest(df, cfg)
    assert res.n_trades >= 0
    assert res.long_count + res.short_count == res.n_trades
    assert res.n_wins <= res.n_trades
    # 집계 일관성 — total = trade net 합.
    assert res.total_net_pnl_pct == pytest.approx(
        sum(t.net_pnl_pct for t in res.trades), rel=1e-9, abs=1e-9,
    )
