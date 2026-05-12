"""Market Structure Shift (MSS) / BOS / CHoCH detector — ICT 핵심 추세 박는 거.

ICT 박은 정의:
- **BOS** (Break of Structure) = 박힌 추세 방향 박힘. swing high 박은 거 박힘 (bull) /
  swing low 박은 거 박힘 (bear). 추세 continuation 박힘.
- **CHoCH** (Change of Character) = 박힌 추세 반대 방향 박힘. uptrend 박힌 거 swing
  low 박은 거 박힘 (bearish CHoCH) / downtrend 박힌 swing high 박은 거 박힘 (bullish
  CHoCH). 추세 reversal 박힘.
- **MSS** (Market Structure Shift) = CHoCH 박힌 거의 다른 이름 (2024 멘토십 박힘).
  일부 박힌 박은 BOS도 박힘 — 여기는 CHoCH 박힌 거만 MSS 박힘.

박힌 거 박힘 박힘:
1. swing point list 박은 후
2. 박은 추세 박힘 (bullish = HH+HL 박힘 / bearish = LH+LL 박힘)
3. 박은 추세 박힌 박힌 박힌 추세 방향 박힌 swing 박힌 거 박힌 거 박힘 → BOS
4. 박은 추세 박힌 박힌 반대 방향 박힌 swing 박힌 거 박힌 거 박힘 → CHoCH (= MSS)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from aurora_ict.indicators.swing_points import SwingPoint, SwingType


class StructureType(StrEnum):
    """구조 이벤트 박힌 거 종류."""

    BOS_BULLISH = "bos_bullish"      # uptrend 박힘 박힘 swing high 박힘 박힘
    BOS_BEARISH = "bos_bearish"      # downtrend 박힘 박힘 swing low 박힘 박힘
    CHOCH_BULLISH = "choch_bullish"  # downtrend → uptrend 박힘 (MSS up)
    CHOCH_BEARISH = "choch_bearish"  # uptrend → downtrend 박힘 (MSS down)


class TrendDirection(StrEnum):
    """추세 방향."""

    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass(slots=True)
class StructureEvent:
    """구조 이벤트 1개.

    Attributes:
        ts_ms: 박힌 봉 (close 박은 게 박힌 거 박힘) open time (ms).
        type: BOS_* / CHOCH_*.
        broken_level: 박힌 swing 박힌 가격.
        idx: 박힌 봉 index (close 박은 게 박은 봉).
        broken_swing_idx: 박힌 swing 박은 거 박힌 index.
        trend_before: 박힘 박은 거 박힌 추세 방향.
    """

    ts_ms: int
    type: StructureType
    broken_level: float
    idx: int
    broken_swing_idx: int
    trend_before: TrendDirection


def detect_structure_events(
    df: pd.DataFrame,
    swings: list[SwingPoint],
) -> list[StructureEvent]:
    """BOS / CHoCH 이벤트 박힌 거 박힘.

    추세 박힘 박힌 거:
    - 시작 추세 박힘 None.
    - 첫 박은 swing 박은 거 박힌 거 박힘 박은 거 박힌 거 박힌 거 박힘 (HH/HL 박힘 up,
      LH/LL 박힘 down).
    - 박은 추세 박힌 박힌 박은 swing 박힌 거 박힌 거 박힘 → BOS
    - 반대 방향 박힌 swing 박힌 거 박힌 거 박힘 → CHoCH

    Args:
        df: OHLCV DataFrame — ``close`` 박힘.
        swings: swing point list.

    Returns:
        StructureEvent list — 시간순.

    Notes:
        - Close 박힌 거 박힌 거 박힘 (close > swing high 박힘 = break). wick 박힌 거 X.
        - swing 박힌 거 박힌 거 박힌 박힌 추세 박힌 거 박힌 거 박힘 박힌 X (e.g. swing
          high 박힌 거 박힌 박힌 swing high 박힌 거 박힌 거 박힘 박힌 거 박힘).
    """
    if not swings or len(df) < 2:
        return []

    if isinstance(df.index, pd.DatetimeIndex):
        ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
    else:
        ts_arr = df.index.to_numpy()

    closes = df["close"].to_numpy()

    events: list[StructureEvent] = []
    trend = TrendDirection.NONE
    # 박힌 박은 swing high/low 박은 거 (가장 최근)
    last_high: SwingPoint | None = None
    last_low: SwingPoint | None = None
    # 박힌 거 박힌 거 박힌 swing 박힌 거 박힌 index — break 검사 박은 cursor
    cursor_idx = 0

    for swing in swings:
        # 박힌 swing 박힌 거 박은 거 cursor_idx ~ swing.idx 박힘 박힌 거 박힌 break 박힘
        # 박힘 검사
        if swing.type is SwingType.HIGH:
            # swing high 박힌 거 박힌 박힌 swing high 박힌 거 박힌 거 박힘 검사
            if last_high is not None:
                # cursor_idx ~ swing.idx 박힘 박힌 close 박은 게 last_high.price 박힌 거 박힘
                for j in range(cursor_idx, swing.idx + 1):
                    if closes[j] > last_high.price:
                        # break 박힘
                        event_type = (
                            StructureType.BOS_BULLISH if trend is TrendDirection.UP
                            else StructureType.CHOCH_BULLISH
                        )
                        events.append(StructureEvent(
                            ts_ms=int(ts_arr[j]),
                            type=event_type,
                            broken_level=last_high.price,
                            idx=j,
                            broken_swing_idx=last_high.idx,
                            trend_before=trend,
                        ))
                        trend = TrendDirection.UP
                        break
            last_high = swing
        else:  # LOW
            if last_low is not None:
                for j in range(cursor_idx, swing.idx + 1):
                    if closes[j] < last_low.price:
                        event_type = (
                            StructureType.BOS_BEARISH if trend is TrendDirection.DOWN
                            else StructureType.CHOCH_BEARISH
                        )
                        events.append(StructureEvent(
                            ts_ms=int(ts_arr[j]),
                            type=event_type,
                            broken_level=last_low.price,
                            idx=j,
                            broken_swing_idx=last_low.idx,
                            trend_before=trend,
                        ))
                        trend = TrendDirection.DOWN
                        break
            last_low = swing
        cursor_idx = swing.idx + 1

    # 마지막 swing 박힌 거 박힌 거 박힌 거 박힌 박힌 봉 박힘 박힘 검사
    if last_high is not None:
        for j in range(cursor_idx, len(df)):
            if closes[j] > last_high.price:
                event_type = (
                    StructureType.BOS_BULLISH if trend is TrendDirection.UP
                    else StructureType.CHOCH_BULLISH
                )
                events.append(StructureEvent(
                    ts_ms=int(ts_arr[j]),
                    type=event_type,
                    broken_level=last_high.price,
                    idx=j,
                    broken_swing_idx=last_high.idx,
                    trend_before=trend,
                ))
                break
    if last_low is not None:
        for j in range(cursor_idx, len(df)):
            if closes[j] < last_low.price:
                event_type = (
                    StructureType.BOS_BEARISH if trend is TrendDirection.DOWN
                    else StructureType.CHOCH_BEARISH
                )
                events.append(StructureEvent(
                    ts_ms=int(ts_arr[j]),
                    type=event_type,
                    broken_level=last_low.price,
                    idx=j,
                    broken_swing_idx=last_low.idx,
                    trend_before=trend,
                ))
                break

    # ts_ms 박힘 sort 박힘 (chronological 박힘)
    events.sort(key=lambda e: e.ts_ms)
    return events


__all__ = [
    "StructureEvent",
    "StructureType",
    "TrendDirection",
    "detect_structure_events",
]
