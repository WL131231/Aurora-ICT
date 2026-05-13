"""Trailing Extremes — Strong/Weak High & Low (LuxAlgo SMC 패턴 포팅).

ICT 해석:
    - 현재 봉까지의 최고점 (running max high) 과 최저점 (running min low) 을
      ``trailing.top`` / ``trailing.bottom`` 으로 추적한다.
    - 현재 swing trend bias 에 따라 라벨이 결정된다:
        * **BEARISH bias** → ``trailing.top`` = "Strong High" (반전 후보 — 매도 측 유동성),
          ``trailing.bottom`` = "Weak Low"
        * **BULLISH bias** → ``trailing.top`` = "Weak High",
          ``trailing.bottom`` = "Strong Low" (반전 후보 — 매수 측 유동성)
        * **NONE bias** → 그냥 "High" / "Low"

LuxAlgo 의 ``updateTrailingExtremes`` 는 streaming 으로 매 봉마다 갱신하지만,
본 모듈은 df 전체로부터 한 번에 계산하는 batch 형태다.

구간 정의:
    - top    = max(high)  over [last_swing_low_ts .. df_end] (없으면 df 전체)
    - bottom = min(low)   over [last_swing_high_ts .. df_end] (없으면 df 전체)

이렇게 하면 가장 최근 swing 이후로 새로 형성된 trailing extreme 만 잡힘.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aurora_ict.indicators.structure import StructureEvent, StructureType
from aurora_ict.indicators.swing_points import SwingPoint, SwingType

# 라벨 문자열 — UI 직접 사용
LABEL_STRONG_HIGH = "Strong High"
LABEL_WEAK_HIGH = "Weak High"
LABEL_STRONG_LOW = "Strong Low"
LABEL_WEAK_LOW = "Weak Low"
LABEL_HIGH = "High"
LABEL_LOW = "Low"


@dataclass(slots=True)
class TrailingExtremes:
    """현재 시점의 trailing high/low + LuxAlgo 식 라벨.

    Attributes:
        top: 현재까지 최고점 (last swing low 이후의 max high).
        bottom: 현재까지 최저점 (last swing high 이후의 min low).
        top_ts_ms: top 이 형성된 봉의 ts (ms).
        bottom_ts_ms: bottom 이 형성된 봉의 ts (ms).
        top_label: "Strong High" / "Weak High" / "High".
        bottom_label: "Strong Low" / "Weak Low" / "Low".
    """

    top: float
    bottom: float
    top_ts_ms: int
    bottom_ts_ms: int
    top_label: str
    bottom_label: str


def _bias_from_events(events: list[StructureEvent]) -> str:
    """가장 최근 structure event 의 방향을 bias 로 반환 ("up"/"down"/"none")."""
    if not events:
        return "none"
    last = events[-1]
    if last.type in (StructureType.BOS_BULLISH, StructureType.CHOCH_BULLISH):
        return "up"
    return "down"


def _ts_array(df: pd.DataFrame) -> list[int]:
    if isinstance(df.index, pd.DatetimeIndex):
        return (df.index.astype("int64") // 10**6).tolist()
    return [int(v) for v in df.index.tolist()]


def compute_trailing_extremes(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    structure_events: list[StructureEvent],
) -> TrailingExtremes | None:
    """현재 시점의 trailing high/low + 라벨 계산.

    Args:
        df: OHLCV DataFrame.
        swings: 박혀있는 swing points (detect_swing_points 결과).
        structure_events: 박혀있는 BOS/CHoCH 이벤트.

    Returns:
        ``TrailingExtremes`` 객체. df 가 비어있으면 ``None``.

    Raises:
        ValueError: df 에 high/low 컬럼 빠진 경우.
    """
    if len(df) == 0:
        return None
    missing = {"high", "low"} - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    ts_arr = _ts_array(df)
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    # 가장 최근 swing high / low ts 추출
    last_swing_high_ts: int | None = None
    last_swing_low_ts: int | None = None
    for sw in reversed(swings):
        if sw.type is SwingType.HIGH and last_swing_high_ts is None:
            last_swing_high_ts = sw.ts_ms
        elif sw.type is SwingType.LOW and last_swing_low_ts is None:
            last_swing_low_ts = sw.ts_ms
        if last_swing_high_ts is not None and last_swing_low_ts is not None:
            break

    # top 후보 범위 — last swing low 이후의 봉 (없으면 전체)
    top_start_idx = 0
    if last_swing_low_ts is not None:
        for i, t in enumerate(ts_arr):
            if t >= last_swing_low_ts:
                top_start_idx = i
                break

    # bottom 후보 범위 — last swing high 이후
    bot_start_idx = 0
    if last_swing_high_ts is not None:
        for i, t in enumerate(ts_arr):
            if t >= last_swing_high_ts:
                bot_start_idx = i
                break

    top_slice = highs[top_start_idx:]
    bot_slice = lows[bot_start_idx:]
    top_local_idx = int(top_slice.argmax())
    bot_local_idx = int(bot_slice.argmin())
    top_idx = top_start_idx + top_local_idx
    bot_idx = bot_start_idx + bot_local_idx

    bias = _bias_from_events(structure_events)
    if bias == "down":
        top_label = LABEL_STRONG_HIGH
        bottom_label = LABEL_WEAK_LOW
    elif bias == "up":
        top_label = LABEL_WEAK_HIGH
        bottom_label = LABEL_STRONG_LOW
    else:
        top_label = LABEL_HIGH
        bottom_label = LABEL_LOW

    return TrailingExtremes(
        top=float(highs[top_idx]),
        bottom=float(lows[bot_idx]),
        top_ts_ms=int(ts_arr[top_idx]),
        bottom_ts_ms=int(ts_arr[bot_idx]),
        top_label=top_label,
        bottom_label=bottom_label,
    )


__all__ = [
    "LABEL_HIGH",
    "LABEL_LOW",
    "LABEL_STRONG_HIGH",
    "LABEL_STRONG_LOW",
    "LABEL_WEAK_HIGH",
    "LABEL_WEAK_LOW",
    "TrailingExtremes",
    "compute_trailing_extremes",
]
