"""Liquidity Sweep + BSL/SSL detector — ICT 핵심 entry trigger.

ICT 정의:
- **BSL** (Buy Side Liquidity) = 직전 swing high 위 (short stop이 몰려 있는 자리)
- **SSL** (Sell Side Liquidity) = 직전 swing low 아래 (long stop이 몰려 있는 자리)
- **Liquidity Sweep** = wick은 직전 swing high/low를 찍었지만 close는 그 안으로
  돌아온 봉 (보통 1봉). 다음 봉이 sweep candle close를 깨지 않으면 valid.

ICT 가설: smart money가 직전 swing point 위/아래의 stop을 청산시킨 뒤 반대 방향으로
가격을 밀어 올린다. 따라서 sweep 직후 entry를 잡는 게 본질.

종류:
- **Bullish Sweep** (SSL sweep) → 이후 long entry 후보
- **Bearish Sweep** (BSL sweep) → 이후 short entry 후보

**Equal Highs (EQH) / Equal Lows (EQL)** = 가장 매력적인 liquidity 풀 (개미가
"강한 저항"으로 보는 자리 = smart money의 사냥감). 같은 가격대의 swing이
2개 이상 연속될 때 발생.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from aurora_ict.indicators.swing_points import SwingPoint, SwingType


class SweepType(StrEnum):
    """Sweep 방향."""

    BULLISH = "bullish"  # SSL sweep (직전 swing low 찍고 회복 → 이후 long 후보)
    BEARISH = "bearish"  # BSL sweep (직전 swing high 찍고 회복 → 이후 short 후보)


@dataclass(slots=True)
class LiquiditySweep:
    """Liquidity Sweep 1개.

    Attributes:
        ts_ms: sweep 봉 open time (ms).
        type: BULLISH / BEARISH.
        swept_price: 청산 대상 swing 가격 (BSL = swing high, SSL = swing low).
        wick_price: sweep candle wick 가격 (BSL = high, SSL = low).
        idx: sweep candle index.
        swing_idx: sweep된 swing point index.
    """

    ts_ms: int
    type: SweepType
    swept_price: float
    wick_price: float
    idx: int
    swing_idx: int


@dataclass(slots=True)
class EqualLevel:
    """Equal Highs / Equal Lows — 가장 매력적인 liquidity 자리.

    Attributes:
        type: HIGH (EQH) / LOW (EQL).
        price: 평균 가격.
        indices: 클러스터에 포함된 swing index 리스트 (≥ 2).
        tolerance_pct: 클러스터링에 사용된 % tolerance.
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
    """Liquidity Sweep 검출.

    각 swing point 이후 봉을 순회하며, wick은 swing 가격을 넘어섰지만 close는
    swing 안쪽으로 돌아온 봉을 찾는다.

    Args:
        df: OHLCV DataFrame — ``open / high / low / close`` 필수.
        swings: ``detect_swing_points()`` 결과.
        require_close_back: True (표준) — sweep candle close가 swing 안쪽으로
            돌아와야 valid. False — wick 조건만으로 인정.

    Returns:
        LiquiditySweep list — 시간순. sweep된 swing은 ``swept=True``로
        in-place 갱신됨.
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
        # swing 봉 다음부터 sweep 여부 검사
        for j in range(swing.idx + 1, len(df)):
            if swing.type is SwingType.HIGH:
                # BSL sweep — wick은 swing high 위, close는 swing 안으로 복귀
                if highs[j] > swing.price:
                    if require_close_back and closes[j] >= swing.price:
                        # close가 swing 위에 머무름 → sweep 아님 (breakout)
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
                # SSL sweep — wick은 swing low 아래, close는 swing 안으로 복귀
                if lows[j] < swing.price:
                    if require_close_back and closes[j] <= swing.price:
                        # close가 swing 아래에 머무름 → sweep 아님 (breakout)
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
    """Equal Highs / Lows 검출.

    같은 type (HIGH/LOW) swing 중 가격이 ``tolerance_pct`` 안에 모인 것들을
    하나의 클러스터로 묶어 EQH / EQL을 만든다.

    Args:
        swings: swing point list.
        tolerance_pct: 클러스터링 % tolerance (예: 0.001 = 0.1%).
        min_count: 클러스터 최소 멤버 수 (표준 = 2).

    Returns:
        EqualLevel list — 클러스터별 평균 가격 포함.
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
