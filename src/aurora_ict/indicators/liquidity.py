"""Liquidity Sweep + BSL/SSL detector — ICT 핵심 entry trigger.

ICT 박은 정의:
- **BSL** (Buy Side Liquidity) = 옛 swing high 박은 위 (short stop 박혀있음)
- **SSL** (Sell Side Liquidity) = 옛 swing low 박은 아래 (long stop 박혀있음)
- **Liquidity Sweep** = wick 박은 게 옛 swing high/low 박은 거 박은 후 close 박은 게
  박힌 안 박은 봉 (1봉 박힘). 다음 봉이 sweep candle close 박지 않으면 valid.

ICT 박힌 가정: smart money 박은 박힌 게 옛 swing point 박은 위/아래 박힌 stop 박힌 거
청산 박힘 후 반대 방향 박힘. 그래서 sweep 박힌 후 entry 박는 게 본질.

종류:
- **Bullish Sweep** (= SSL sweep 박힘) → 박힌 거 박은 후 long entry candidate
- **Bearish Sweep** (= BSL sweep 박힘) → 박힌 거 박은 후 short entry candidate

또 **Equal Highs (EQH) / Equal Lows (EQL)** 박힌 거 = 가장 박힌 liquidity 박힘 (개미 박힌
"강한 저항" 박은 거 박은 게 smart money 박힌 사냥감). 같은 swing 박힌 거 박힌 거 박힌
연속 박힌 거 박힘.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from aurora_ict.indicators.swing_points import SwingPoint, SwingType


class SweepType(StrEnum):
    """Sweep 방향."""

    BULLISH = "bullish"  # SSL sweep (옛 swing low 박힘 → 박은 후 long 박힘)
    BEARISH = "bearish"  # BSL sweep (옛 swing high 박힘 → 박은 후 short 박힘)


@dataclass(slots=True)
class LiquiditySweep:
    """Liquidity Sweep 1개.

    Attributes:
        ts_ms: sweep 봉 open time (ms).
        type: BULLISH / BEARISH.
        swept_price: 옛 swing 박힌 가격 (BSL = swing high, SSL = swing low).
        wick_price: sweep candle wick 박힌 거 (BSL = high, SSL = low).
        idx: sweep candle index.
        swing_idx: 박힌 옛 swing point 박힌 index.
    """

    ts_ms: int
    type: SweepType
    swept_price: float
    wick_price: float
    idx: int
    swing_idx: int


@dataclass(slots=True)
class EqualLevel:
    """Equal Highs / Equal Lows — 가장 박힌 liquidity 자리.

    Attributes:
        type: HIGH (EQH) / LOW (EQL).
        price: 평균 가격.
        indices: 박힌 swing 박힌 거 박힌 indices (≥ 2).
        tolerance_pct: 박힌 거 박힌 거 박힌 % tolerance 박힌 거.
    """

    type: SwingType
    price: float
    indices: list[int]
    tolerance_pct: float


def detect_liquidity_sweeps(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    require_close_back: bool = True,
) -> list[LiquiditySweep]:
    """Liquidity Sweep 박힌 거 박힘.

    각 swing point 박힌 거 박힌 거 박은 후 봉 박힌 게 박힌 거의 wick 박힌 게 swing
    가격 박힌 거 박힌 후 close 박은 게 박힌 안 박은 봉 박는 거.

    Args:
        df: OHLCV DataFrame — ``open / high / low / close`` 박힘.
        swings: 박힌 swing point list (``detect_swing_points()`` 박은 결과).
        require_close_back: True 박힘 (표준) — sweep candle close 박은 게 swing 박은
            거 박힌 안 박혀야 valid. False 박힘 — wick 박힌 거만 박힘.

    Returns:
        LiquiditySweep list — 시간순. 또 sweep 박힌 swing 박힌 거 박힌 ``swept=True``
        박힘 (in-place mutation 박힘).
    """
    if not swings or len(df) < 2:
        return []

    if isinstance(df.index, pd.DatetimeIndex):
        ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
    else:
        ts_arr = df.index.to_numpy()

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    sweeps: list[LiquiditySweep] = []

    for swing in swings:
        if swing.swept:
            continue
        # swing 박힌 거 박힌 다음 봉부터 박은 후 박힌 거 박힌 거 박은
        for j in range(swing.idx + 1, len(df)):
            if swing.type is SwingType.HIGH:
                # BSL sweep — wick 박힌 게 swing high 박은 거보다 위, close 박은 게 박힌 안
                if highs[j] > swing.price:
                    if require_close_back and closes[j] >= swing.price:
                        # close 박은 게 박힌 위 박힘 → sweep X (breakout 박힘)
                        swing.swept = True
                        break
                    sweeps.append(LiquiditySweep(
                        ts_ms=int(ts_arr[j]),
                        type=SweepType.BEARISH,
                        swept_price=swing.price,
                        wick_price=float(highs[j]),
                        idx=j,
                        swing_idx=swing.idx,
                    ))
                    swing.swept = True
                    break
            else:  # LOW
                # SSL sweep — wick 박힌 게 swing low 박은 거보다 아래, close 박은 게 박힌 안
                if lows[j] < swing.price:
                    if require_close_back and closes[j] <= swing.price:
                        # close 박은 게 박힌 아래 → sweep X (breakout 박힘)
                        swing.swept = True
                        break
                    sweeps.append(LiquiditySweep(
                        ts_ms=int(ts_arr[j]),
                        type=SweepType.BULLISH,
                        swept_price=swing.price,
                        wick_price=float(lows[j]),
                        idx=j,
                        swing_idx=swing.idx,
                    ))
                    swing.swept = True
                    break

    return sweeps


def detect_equal_levels(
    swings: list[SwingPoint],
    tolerance_pct: float = 0.001,
    min_count: int = 2,
) -> list[EqualLevel]:
    """Equal Highs / Lows 박힌 거 박힘.

    같은 type (HIGH/LOW) 박힌 swing 박힌 거 박힌 가격이 ``tolerance_pct`` 안 박힌 거
    여러 개 박힌 거 박힘 → EQH / EQL 박힘.

    Args:
        swings: swing point list.
        tolerance_pct: 박힌 거 박힌 거 박힌 % tolerance (예: 0.001 = 0.1%).
        min_count: 최소 박힌 swing 박힌 거 (표준 = 2 박힘).

    Returns:
        EqualLevel list — 평균 가격 박힌 거 박힘.
    """
    if len(swings) < min_count:
        return []

    levels: list[EqualLevel] = []
    used: set[int] = set()

    for stype in (SwingType.HIGH, SwingType.LOW):
        typed_swings = [s for s in swings if s.type is stype]
        for i, anchor in enumerate(typed_swings):
            if anchor.idx in used:
                continue
            cluster_indices = [anchor.idx]
            cluster_prices = [anchor.price]
            for other in typed_swings[i + 1:]:
                if other.idx in used:
                    continue
                if abs(other.price - anchor.price) / anchor.price <= tolerance_pct:
                    cluster_indices.append(other.idx)
                    cluster_prices.append(other.price)
            if len(cluster_indices) >= min_count:
                avg = sum(cluster_prices) / len(cluster_prices)
                levels.append(EqualLevel(
                    type=stype,
                    price=avg,
                    indices=cluster_indices,
                    tolerance_pct=tolerance_pct,
                ))
                used.update(cluster_indices)

    return levels


__all__ = [
    "EqualLevel",
    "LiquiditySweep",
    "SweepType",
    "detect_equal_levels",
    "detect_liquidity_sweeps",
]
