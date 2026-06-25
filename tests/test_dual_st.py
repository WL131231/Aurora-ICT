"""Dual SuperTrend 추세형 전략 결정론 테스트 — mock 0, 합성 OHLCV.

백테(dst_trend_bt.py)와 동일 신호 로직을 쓰는 dual_st 모듈의 진입 신호·트레일
스탑 방향을 합성 추세 데이터로 검증. 외부 네트워크/거래소 호출 없음.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aurora_ict.strategy.dual_st import (
    DualSTConfig,
    compute_signals,
    latest_trail_stop,
)


def _synth(prices: list[float]) -> pd.DataFrame:
    """종가 리스트 → OHLCV (±0.5 wick 합성)."""
    arr = np.asarray(prices, dtype=float)
    return pd.DataFrame(
        {"open": arr, "high": arr + 0.5, "low": arr - 0.5, "close": arr, "volume": 1.0},
    )


def test_signal_columns_present() -> None:
    s = compute_signals(_synth(list(range(100, 160))), DualSTConfig())
    for col in ("st1", "st2", "trail", "buy_sig", "sell_sig"):
        assert col in s.columns


def test_uptrend_aligns_long_and_signals() -> None:
    # 단조 상승 → 마지막 봉이 롱 정렬(close > st1 & st2) + 롱 진입 신호 발생.
    s = compute_signals(_synth(list(range(100, 200))), DualSTConfig())
    last = s.iloc[-1]
    assert float(last["close"]) > float(last["st1"])
    assert float(last["close"]) > float(last["st2"])
    assert bool(s["buy_sig"].any())


def test_downtrend_aligns_short_and_signals() -> None:
    s = compute_signals(_synth(list(range(200, 100, -1))), DualSTConfig())
    last = s.iloc[-1]
    assert float(last["close"]) < float(last["st1"])
    assert float(last["close"]) < float(last["st2"])
    assert bool(s["sell_sig"].any())


def test_trail_stop_below_price_in_uptrend() -> None:
    # 상승 추세에선 트레일 ST 가 가격 아래(롱 스탑이 가격을 따라 올라옴).
    df = _synth(list(range(100, 200)))
    assert latest_trail_stop(df, DualSTConfig()) < float(df["close"].iloc[-1])


def test_trail_stop_above_price_in_downtrend() -> None:
    df = _synth(list(range(200, 100, -1)))
    assert latest_trail_stop(df, DualSTConfig()) > float(df["close"].iloc[-1])


def test_wider_trail_is_farther_from_price() -> None:
    # trail_mult 가 클수록 스탑이 가격에서 더 멀다(상승 추세 기준 더 아래).
    df = _synth(list(range(100, 220)))
    near = latest_trail_stop(df, DualSTConfig(trail_mult=4.0))
    far = latest_trail_stop(df, DualSTConfig(trail_mult=8.0))
    assert far < near < float(df["close"].iloc[-1])
