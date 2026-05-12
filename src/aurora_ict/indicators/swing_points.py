"""Swing High/Low pivot detector — ICT 3-bar swing (Ali Khan Bible 정의 기반).

ICT swing 정의:
- **Swing High** = 중간 봉 high가 양옆 봉 high보다 strictly 높음
- **Swing Low** = 중간 봉 low가 양옆 봉 low보다 strictly 낮음

표준 3봉 패턴 — left/right window는 일반화 가능 (n=1 = 표준).

ICT의 swing level 계층:
- **STH/STL** (Short Term High/Low) = 기본 3봉 swing
- **ITH/ITL** (Intermediate Term) = 양옆에 더 낮은 STH가 있는 STH
- **LTH/LTL** (Long Term) = 양옆에 더 낮은 ITH가 있는 ITH

이 파일은 STH/STL만 다룬다. ITH/LTH는 structure.py 등 다른 layer에서 처리 가능.

Aurora의 harmonic pivot 검출과 유사하지만 ICT 정의(양옆 = 직전/직후 봉)와 미세 차이가
있어 별도로 구현한다. harmonic 쪽은 N봉 윈도우 기반.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class SwingType(StrEnum):
    """Swing 방향."""

    HIGH = "high"
    LOW = "low"


@dataclass(slots=True)
class SwingPoint:
    """Swing point 1개.

    Attributes:
        ts_ms: pivot 봉 open time (ms).
        type: HIGH (swing high) / LOW (swing low).
        price: pivot 가격 (HIGH = high, LOW = low).
        idx: DataFrame 상의 pivot 봉 index.
        swept: 추후 가격이 이 swing을 sweep했는지 (liquidity 추적에 사용).
    """

    ts_ms: int
    type: SwingType
    price: float
    idx: int
    swept: bool = False


def detect_swing_points(
    df: pd.DataFrame,
    left: int = 1,
    right: int = 1,
) -> list[SwingPoint]:
    """Swing High/Low pivot 검출.

    Args:
        df: OHLCV DataFrame — index = timestamp (ms 또는 datetime), columns =
            ``open / high / low / close``. 최소 ``left + right + 1`` row 필요.
        left: 왼쪽 비교 봉 수. 표준 = 1.
        right: 오른쪽 비교 봉 수. 표준 = 1.

    Returns:
        SwingPoint list — 시간순.

    Notes:
        - ``left=2, right=2``로 주면 5봉 swing (더 강한 pivot).
        - Swing high: ``high[i] > high[i-left:i] and high[i] > high[i+1:i+right+1]``
        - Swing low: 동일하지만 low 기준.
        - 양옆 봉 high와 같은 값이면 swing 아님 (strict >). ICT 표준.
    """
    if len(df) < left + right + 1:
        return []

    required = {"high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    if isinstance(df.index, pd.DatetimeIndex):
        ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
    else:
        ts_arr = df.index.to_numpy()

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    swings: list[SwingPoint] = []
    for i in range(left, len(df) - right):
        hi = highs[i]
        lo = lows[i]

        # Swing high — 양옆 봉 high 보다 strictly 높음
        is_swing_high = (
            all(hi > highs[i - k] for k in range(1, left + 1))
            and all(hi > highs[i + k] for k in range(1, right + 1))
        )
        if is_swing_high:
            swings.append(SwingPoint(
                ts_ms=int(ts_arr[i]),
                type=SwingType.HIGH,
                price=float(hi),
                idx=i,
            ))
            continue  # 같은 봉이 swing high + low 동시 성립 X

        # Swing low — 양옆 봉 low 보다 strictly 낮음
        is_swing_low = (
            all(lo < lows[i - k] for k in range(1, left + 1))
            and all(lo < lows[i + k] for k in range(1, right + 1))
        )
        if is_swing_low:
            swings.append(SwingPoint(
                ts_ms=int(ts_arr[i]),
                type=SwingType.LOW,
                price=float(lo),
                idx=i,
            ))

    return swings


__all__ = ["SwingPoint", "SwingType", "detect_swing_points"]
