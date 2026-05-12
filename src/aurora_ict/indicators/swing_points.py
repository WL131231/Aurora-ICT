"""Swing High/Low pivot detector — ICT 3-bar swing 박힘 (Ali Khan Bible 박은 정의).

ICT 박은 swing 정의:
- **Swing High** = 중간 봉 high 박은 게 양옆 봉 high 박은 거보다 ↑
- **Swing Low** = 중간 봉 low 박은 게 양옆 봉 low 박은 거보다 ↓

3봉 패턴 박힘 — n-bar 박은 박은 게 박힌 박는 거 (n=1 박은 거 = 표준).

ICT 박힌 swing level 박은 3가지:
- **STH/STL** (Short Term High/Low) = 기본 3봉 박힘
- **ITH/ITL** (Intermediate Term) = STH/STL 양옆에 더 낮은 STH 박힌 거
- **LTH/LTL** (Long Term) = ITH/ITL 양옆에 더 낮은 ITH 박힌 거

여기는 STH/STL만 박음. ITH/LTH 박힘 별도 박힐 수 있음 (structure.py 박힌 거 박힘).

Aurora 박은 harmonic 박은 거의 pivot 박은 거 비슷 박혀서 그 코드 박은 거 박을지 박지만,
ICT 박은 거 박은 정의 미세 차이 박힘 (양옆 = 직전/직후 봉 박는 거 박혀, harmonic 박힌 거는
N봉 박힘 박힘). 그래서 별도 박음.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd


class SwingType(str, Enum):
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
        idx: DataFrame 박힌 거의 pivot 봉 index.
        swept: 추후 가격이 박힌 거 박은지 (liquidity sweep 박힘 박는 데 박힘).
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
    """Swing High/Low pivot 박힌 거 박힘.

    Args:
        df: OHLCV DataFrame — index = timestamp (ms 또는 datetime), columns =
            ``open / high / low / close``. 최소 ``left + right + 1`` row 박혀야.
        left: 박힌 양옆 봉 수 (왼쪽). 표준 = 1.
        right: 박힌 양옆 봉 수 (오른쪽). 표준 = 1.

    Returns:
        SwingPoint list — 시간순.

    Notes:
        - ``left=2, right=2`` 박으면 5봉 swing 박힘 (더 강한 pivot).
        - Swing high 박힘: ``high[i] > high[i-left:i] and high[i] > high[i+1:i+right+1]``
        - Swing low 박힘: 동일하지만 low 박힘 박힘.
        - 양옆 봉 박힌 거 박힌 high 박힌 거랑 같으면 swing X (strict >). ICT 박힘 표준.
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
        h = highs[i]
        l = lows[i]

        # Swing high — 양옆 봉 high 박은 거보다 strictly 높음
        is_swing_high = (
            all(h > highs[i - k] for k in range(1, left + 1))
            and all(h > highs[i + k] for k in range(1, right + 1))
        )
        if is_swing_high:
            swings.append(SwingPoint(
                ts_ms=int(ts_arr[i]),
                type=SwingType.HIGH,
                price=float(h),
                idx=i,
            ))
            continue  # 같은 봉이 swing high + low 동시 박힘 X

        # Swing low — 양옆 봉 low 박은 거보다 strictly 낮음
        is_swing_low = (
            all(l < lows[i - k] for k in range(1, left + 1))
            and all(l < lows[i + k] for k in range(1, right + 1))
        )
        if is_swing_low:
            swings.append(SwingPoint(
                ts_ms=int(ts_arr[i]),
                type=SwingType.LOW,
                price=float(l),
                idx=i,
            ))

    return swings


__all__ = ["SwingPoint", "SwingType", "detect_swing_points"]
