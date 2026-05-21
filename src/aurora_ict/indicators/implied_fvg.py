"""Implied FVG (Hidden FVG) — ICT 정통 숨은 빈 공간.

ICT 정의 (Practical p.39):
    일반 FVG = 3봉 패턴에서 ``bar[i].high < bar[i+2].low`` (bullish) 으로 wick 까지
    포함한 명시적 gap. 차트에 즉시 보임.

    **Implied FVG** = wick 으로는 채워져 있어 명시적 gap 안 보이지만, **body 만**
    보면 gap 있는 자리. 차트에서 "숨어있는" FVG.

    - Bullish Implied: ``bar[i].body_high < bar[i+2].body_low``
      AND ``bar[i].wick_high >= bar[i+2].wick_low`` (wick overlap)
    - Bearish Implied: 반대

용도:
    - 일반 FVG 가 안 보이는 깔끔한 추세 구간에서 진입 후보
    - mean threshold = 두 body 사이 중간 (CE Fibonacci 50%)
    - 일반 FVG 보다 약한 신호 — confluence 추가 시 진입

vs Inverse FVG (IFVG):
    ⚠️ 이름 비슷한데 다른 개념.
    - **IFVG** (`fvg.py`): 이미 형성된 FVG 가 invalidated (가격이 통과) 된 후
      반대 방향 zone 으로 작동
    - **Implied FVG** (본 모듈): body 만의 hidden gap, 처음부터 명시적 X
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class ImpliedFVGType(StrEnum):
    """방향."""

    BULLISH = "bullish"   # 가격이 위로 갈 때 형성, 아래쪽이 zone (long retest 자리)
    BEARISH = "bearish"


@dataclass(slots=True, frozen=True)
class ImpliedFVG:
    """Implied FVG 1개.

    Attributes:
        type: BULLISH / BEARISH.
        idx: 3봉 패턴의 마지막 봉 (bar[i+2]) 인덱스.
        body_high: zone 상단 (body 기준).
        body_low: zone 하단.
        wick_overlap_width: 두 wick 의 겹치는 폭 (overlap 클수록 더 hidden).
    """

    type: ImpliedFVGType
    idx: int
    body_high: float
    body_low: float
    wick_overlap_width: float

    @property
    def mean_threshold(self) -> float:
        """CE (Consequent Encroachment) = body 사이 50% 자리."""
        return (self.body_high + self.body_low) / 2.0


def detect_implied_fvgs(df: pd.DataFrame) -> list[ImpliedFVG]:
    """3봉 패턴에서 Implied FVG 검출.

    각 ``i >= 2`` 봉에 대해 ``bar[i-2], bar[i-1], bar[i]`` 검사.

    Bullish 조건:
        - bar[i-2].body_high < bar[i].body_low  (body gap 위쪽)
        - bar[i-2].wick_high >= bar[i].wick_low  (wick overlap — 명시적 FVG 아님)

    Bearish 조건: 반대.

    Args:
        df: ``open``, ``high``, ``low``, ``close`` 컬럼.

    Returns:
        ``ImpliedFVG`` 리스트 — 인덱스순.
    """
    if df.empty or len(df) < 3:
        return []

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    fvgs: list[ImpliedFVG] = []
    for i in range(2, len(df)):
        bar1_body_high = max(float(opens[i - 2]), float(closes[i - 2]))
        bar1_body_low = min(float(opens[i - 2]), float(closes[i - 2]))
        bar1_wick_high = float(highs[i - 2])
        bar1_wick_low = float(lows[i - 2])

        bar3_body_high = max(float(opens[i]), float(closes[i]))
        bar3_body_low = min(float(opens[i]), float(closes[i]))
        bar3_wick_high = float(highs[i])
        bar3_wick_low = float(lows[i])

        # Bullish Implied: body gap 위쪽 + wick overlap
        if (
            bar1_body_high < bar3_body_low
            and bar1_wick_high >= bar3_wick_low
        ):
            overlap = bar1_wick_high - bar3_wick_low
            fvgs.append(ImpliedFVG(
                type=ImpliedFVGType.BULLISH,
                idx=i,
                body_high=bar3_body_low,    # zone 상단 = bar3 body low
                body_low=bar1_body_high,    # zone 하단 = bar1 body high
                wick_overlap_width=max(0.0, overlap),
            ))
            continue

        # Bearish Implied
        if (
            bar1_body_low > bar3_body_high
            and bar1_wick_low <= bar3_wick_high
        ):
            overlap = bar3_wick_high - bar1_wick_low
            fvgs.append(ImpliedFVG(
                type=ImpliedFVGType.BEARISH,
                idx=i,
                body_high=bar1_body_low,    # zone 상단 = bar1 body low
                body_low=bar3_body_high,    # zone 하단 = bar3 body high
                wick_overlap_width=max(0.0, overlap),
            ))

    return fvgs


__all__ = [
    "ImpliedFVG",
    "ImpliedFVGType",
    "detect_implied_fvgs",
]
