"""TF 단위 3-state 추세 판정기 — up / sideways / down.

세 시그널을 1/-1/0 으로 평가하고 가중합(=합산)으로 분류한다:

1. 마지막 캔들 body 방향 (close > open → +1, < → -1, 동일 → 0).
2. 마지막 close vs EMA20 (위 → +1, 아래 → -1, 동일 → 0).
3. 최근 swing direction — Higher-High / Higher-Low → +1,
   Lower-High / Lower-Low → -1, 그 외 (혼합/부족) → 0.

합산 score ≥ 2 → "up", ≤ -2 → "down", 그 외 → "sideways".

HTF FVG 가중치 시스템의 보조 정보로 쓰인다. 단독으로 진입 차단하지는 않는다.
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

TrendState = Literal["up", "sideways", "down"]


def _ema_last(closes: list[float], period: int) -> float | None:
    if len(closes) < period + 1:
        return None
    k = 2.0 / (period + 1)
    ema = sum(closes[:period]) / period  # SMA seed
    for px in closes[period:]:
        ema = px * k + ema * (1.0 - k)
    return ema


def _recent_swing_direction(df: pd.DataFrame, lookback: int = 20) -> int:
    """최근 swing 방향: HH/HL → +1, LH/LL → -1, 모호 → 0."""
    if len(df) < lookback + 2:
        return 0
    sub = df.tail(lookback)
    highs = sub["high"].astype(float).to_numpy()
    lows = sub["low"].astype(float).to_numpy()
    # 단순화: 절반 구간 vs 후반 구간의 max high / min low 비교.
    half = len(sub) // 2
    if half < 2:
        return 0
    h1, h2 = float(highs[:half].max()), float(highs[half:].max())
    l1, l2 = float(lows[:half].min()), float(lows[half:].min())
    hh = h2 > h1
    hl = l2 > l1
    lh = h2 < h1
    ll = l2 < l1
    if hh and hl:
        return 1
    if lh and ll:
        return -1
    return 0


def evaluate_trend(ohlcv: pd.DataFrame, ema_period: int = 20) -> TrendState:
    """OHLCV DataFrame 을 받아 3-state 추세 라벨 반환.

    Args:
        ohlcv: open/high/low/close 컬럼을 가진 DataFrame.
        ema_period: 가격 vs EMA 비교용 EMA 기간 (기본 20).

    Returns:
        "up" / "sideways" / "down".
    """
    if ohlcv is None or len(ohlcv) < ema_period + 2:
        return "sideways"

    last_open = float(ohlcv["open"].iloc[-1])
    last_close = float(ohlcv["close"].iloc[-1])

    s_body = 0
    if last_close > last_open:
        s_body = 1
    elif last_close < last_open:
        s_body = -1

    closes = ohlcv["close"].astype(float).tolist()
    ema = _ema_last(closes, ema_period)
    if ema is None:
        s_ema = 0
    elif last_close > ema:
        s_ema = 1
    elif last_close < ema:
        s_ema = -1
    else:
        s_ema = 0

    s_swing = _recent_swing_direction(ohlcv)

    score = s_body + s_ema + s_swing
    if score >= 2:
        return "up"
    if score <= -2:
        return "down"
    return "sideways"


__all__ = ["TrendState", "evaluate_trend"]
