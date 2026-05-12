"""FVG (Fair Value Gap) detector — Aurora-ICT v0.1.0 첫 indicator.

ICT 핵심 PD-Array 박은 거. 3봉 패턴 박힌 imbalance:
- **Bullish FVG (BISI — Buyside Imbalance, Sellside Inefficiency)**: 1봉 high < 3봉 low
- **Bearish FVG (SIBI — Sellside Imbalance, Buyside Inefficiency)**: 1봉 low > 3봉 high

중간 (2번째) 봉은 보통 큰 displacement candle 박힘 (장대봉). 이 봉 박힌 거 박힌 wick
overlap 박힌 안 박혀야 valid FVG. 즉 1봉 wick와 3봉 wick 사이에 박힌 안 박힌 gap.

가격 측 FVG 박힌 거 다시 박을 가능성 높음 (mean reversion 박는 거 = IPDA의 fair value
re-balance). 그래서 ICT 진입 박는 거 = FVG retest 박힌 시점.

Inverse FVG (IFVG): FVG 박힌 거 깨졌을 때 (close 박은 게 박힘 너머) 박힘. 첫 momentum
shift 신호.

Key threshold:
- **High** = FVG top edge (bullish 측 3봉 low / bearish 측 1봉 low)
- **Low** = FVG bottom edge (bullish 측 1봉 high / bearish 측 3봉 high)
- **Mean Threshold (50%)** = (high + low) / 2 — Consequent Encroachment (C.E)

진입 박힐 때 보통 mean threshold 박힌 거까지 박은 retest 박힘 후 reaction 박힘.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class FVGType(StrEnum):
    """FVG 방향."""

    BULLISH = "bullish"  # BISI
    BEARISH = "bearish"  # SIBI


@dataclass(slots=True)
class FVG:
    """FVG 1개 박힌 거.

    Attributes:
        ts_ms: 중간 (displacement) 봉 open time (ms).
        type: bullish (BISI) / bearish (SIBI).
        high: FVG 상단 가격 (gap 박힌 거의 위 edge).
        low: FVG 하단 가격 (gap 박힌 거의 아래 edge).
        idx: DataFrame 박힌 거의 중간 봉 index.
        filled: 추후 가격이 박힌 거 채웠는지 (mean threshold 박힌 거 박혔는지).
        invalidated: close 박은 게 박힌 거 너머 박혔는지 (IFVG 박힘).
    """

    ts_ms: int
    type: FVGType
    high: float
    low: float
    idx: int
    filled: bool = False
    invalidated: bool = False

    @property
    def mean_threshold(self) -> float:
        """Consequent Encroachment (50% 박힘) — 박힌 진입 박을 때 가장 sensitive level."""
        return (self.high + self.low) / 2.0

    @property
    def size(self) -> float:
        """FVG 박힌 크기 (가격 단위)."""
        return self.high - self.low


def detect_fvgs(
    df: pd.DataFrame,
    min_size: float | None = None,
    min_size_pct: float | None = None,
) -> list[FVG]:
    """3봉 패턴 FVG 검출.

    Args:
        df: OHLCV DataFrame — index = timestamp (ms 또는 datetime), columns =
            ``open / high / low / close`` (volume optional). 최소 3 row 박혀야.
        min_size: 절대 가격 단위 최소 gap 박는 거 (None = 필터 X).
        min_size_pct: 중간 봉 close 박은 거 대비 % 최소 gap (예: 0.001 = 0.1%).
            ``min_size``와 함께 박으면 둘 다 만족해야 박힘.

    Returns:
        FVG list — 시간순. 빈 list = 박힌 거 없음.

    Notes:
        - df.index 측 DatetimeIndex 박힘 → ``ts_ms = index.astype(int) // 10**6``.
        - df.index 측 int (ms) 박힘 → 그대로.
        - 중간 봉이 1봉/3봉 wick 박힌 거 박는지 검증 X — 단순히 1봉 high < 3봉 low (bullish)
          만족하면 박음. ICT 박힘 "wick overlap 박히면 implied FVG 박힘" — 별도 박힘.
    """
    if len(df) < 3:
        return []

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    # ts_ms 박힌 거 추출
    if isinstance(df.index, pd.DatetimeIndex):
        ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
    else:
        ts_arr = df.index.to_numpy()

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    fvgs: list[FVG] = []
    for i in range(1, len(df) - 1):
        # Bullish FVG: 1봉 high < 3봉 low → gap 박은 위
        if highs[i - 1] < lows[i + 1]:
            gap_low = float(highs[i - 1])
            gap_high = float(lows[i + 1])
            gap_size = gap_high - gap_low
            if _passes_size(gap_size, closes[i], min_size, min_size_pct):
                fvgs.append(FVG(
                    ts_ms=int(ts_arr[i]),
                    type=FVGType.BULLISH,
                    high=gap_high,
                    low=gap_low,
                    idx=i,
                ))
        # Bearish FVG: 1봉 low > 3봉 high → gap 박은 아래
        elif lows[i - 1] > highs[i + 1]:
            gap_high = float(lows[i - 1])
            gap_low = float(highs[i + 1])
            gap_size = gap_high - gap_low
            if _passes_size(gap_size, closes[i], min_size, min_size_pct):
                fvgs.append(FVG(
                    ts_ms=int(ts_arr[i]),
                    type=FVGType.BEARISH,
                    high=gap_high,
                    low=gap_low,
                    idx=i,
                ))

    return fvgs


def _passes_size(
    gap_size: float,
    mid_close: float,
    min_size: float | None,
    min_size_pct: float | None,
) -> bool:
    """절대/% 최소 size 필터 통과 검사."""
    if min_size is not None and gap_size < min_size:
        return False
    if min_size_pct is not None:
        if mid_close <= 0 or gap_size / mid_close < min_size_pct:
            return False
    return True


def mark_filled_and_invalidated(
    fvgs: list[FVG],
    df: pd.DataFrame,
) -> None:
    """FVG list 박힌 거 박힌 후 가격 동작 박힘 mark.

    각 FVG 박힌 거 박힌 이후 봉 박힘 검사:
    - **filled**: 가격이 mean_threshold 박힌 거 박힘 (50% retest 박힘).
      - bullish: low 박은 게 mean_threshold ≤ 박힘
      - bearish: high 박은 게 mean_threshold ≥ 박힘
    - **invalidated**: close 박은 게 FVG 박힌 거 너머 박힘.
      - bullish: close < low (FVG 박힌 거 깨짐)
      - bearish: close > high

    in-place mutation 박음.
    """
    if not fvgs:
        return

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    for fvg in fvgs:
        # 중간 봉 다음 봉부터 검사
        for j in range(fvg.idx + 2, len(df)):
            if fvg.type is FVGType.BULLISH:
                if not fvg.filled and lows[j] <= fvg.mean_threshold:
                    fvg.filled = True
                if closes[j] < fvg.low:
                    fvg.invalidated = True
                    break
            else:  # BEARISH
                if not fvg.filled and highs[j] >= fvg.mean_threshold:
                    fvg.filled = True
                if closes[j] > fvg.high:
                    fvg.invalidated = True
                    break


__all__ = [
    "FVG",
    "FVGType",
    "detect_fvgs",
    "mark_filled_and_invalidated",
]
