"""Market Structure Shift (MSS) / BOS / CHoCH detector — ICT 추세 판독기.

ICT 정의:
- **BOS** (Break of Structure) = 기존 추세 방향으로의 돌파. uptrend에서 swing high
  돌파 (bull BOS) / downtrend에서 swing low 돌파 (bear BOS). 추세 continuation.
- **CHoCH** (Change of Character) = 추세 반대 방향으로의 돌파. uptrend 중 swing low
  하향 돌파 (bearish CHoCH) / downtrend 중 swing high 상향 돌파 (bullish CHoCH).
  추세 reversal.
- **MSS** (Market Structure Shift) = CHoCH의 다른 이름 (2024 멘토십 용어). 일부
  자료는 BOS까지 묶기도 하지만 여기서는 CHoCH만 MSS로 본다.

처리 흐름:
1. swing point list를 받는다
2. 초기 추세 판단 (bullish = HH+HL 패턴 / bearish = LH+LL 패턴)
3. 현재 추세 방향과 같은 방향의 swing 돌파 → BOS
4. 현재 추세 반대 방향의 swing 돌파 → CHoCH (= MSS)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from aurora_ict.indicators.swing_points import SwingPoint, SwingType


class StructureType(StrEnum):
    """구조 이벤트 종류."""

    BOS_BULLISH = "bos_bullish"      # uptrend 지속 — swing high 상향 돌파
    BOS_BEARISH = "bos_bearish"      # downtrend 지속 — swing low 하향 돌파
    CHOCH_BULLISH = "choch_bullish"  # downtrend → uptrend 전환 (MSS up)
    CHOCH_BEARISH = "choch_bearish"  # uptrend → downtrend 전환 (MSS down)


class TrendDirection(StrEnum):
    """추세 방향."""

    UP = "up"
    DOWN = "down"
    NONE = "none"


@dataclass(slots=True)
class StructureEvent:
    """구조 이벤트 1개.

    Attributes:
        ts_ms: 돌파가 확정된 봉 (close가 swing을 넘어선 봉) open time (ms).
        type: BOS_* / CHOCH_*.
        broken_level: 돌파된 swing 가격.
        idx: 돌파 확정 봉 index.
        broken_swing_idx: 돌파된 swing의 index.
        trend_before: 이벤트 직전 추세 방향.
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
    """BOS / CHoCH 이벤트 검출.

    추세 추적 규칙:
    - 시작 추세는 NONE.
    - 첫 돌파 발생 시 그 방향으로 추세를 설정 (상향 돌파 → up, 하향 돌파 → down).
    - 현재 추세와 같은 방향 swing 돌파 → BOS.
    - 현재 추세 반대 방향 swing 돌파 → CHoCH.

    Args:
        df: OHLCV DataFrame — ``close`` 필수.
        swings: swing point list.

    Returns:
        StructureEvent list — 시간순.

    Notes:
        - Close 기준 (close > swing high = break). wick만으로는 판정 X.
        - 같은 type swing이 연속 갱신되어도 추세는 별도로 옮겨지지 않음 (즉 swing
          high가 새로 만들어졌다고 자동 BOS는 아님).
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
    # 가장 최근 swing high / swing low
    last_high: SwingPoint | None = None
    last_low: SwingPoint | None = None
    # 직전 swing.idx + 1 — 다음 break 검사 시작점 cursor
    cursor_idx = 0

    for swing in swings:
        # 직전 swing 이후 ~ 현재 swing.idx 구간을 순회하며 close 돌파를 검사
        if swing.type is SwingType.HIGH:
            # 새 swing high 갱신 전, 이전 swing high가 돌파됐는지 확인
            if last_high is not None:
                # cursor_idx ~ swing.idx 구간에서 close > last_high.price 인지 검사
                for j in range(cursor_idx, swing.idx + 1):
                    if closes[j] > last_high.price:
                        # 돌파 확정
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

    # swing list 소진 후 남은 봉(cursor_idx ~ len(df))도 마지막 swing 돌파 여부 검사
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

    # 시간순 정렬 (high/low 처리 순서가 섞일 수 있으므로 ts_ms 기준 재정렬)
    events.sort(key=lambda e: e.ts_ms)
    return events


__all__ = [
    "StructureEvent",
    "StructureType",
    "TrendDirection",
    "detect_structure_events",
]
