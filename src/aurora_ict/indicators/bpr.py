"""BPR (Balanced Price Range) — 두 FVG 겹친 강한 zone.

ICT 정의:
    - 한 방향 FVG (bullish) + 그 위/아래에서 형성된 반대 방향 FVG (bearish) 가
      **같은 가격대에서 overlap** 하면 매우 강한 zone.
    - 두 FVG 가 서로 "균형" 을 이루는 가격 범위 = BPR.
    - 어느 방향이든 지지/저항으로 강력하게 작동.

물리적 의미:
    - Bullish FVG 만 있는 자리: 매수 흐름의 흔적 → 가격 돌아오면 매수 자리
    - Bearish FVG 만 있는 자리: 매도 흐름의 흔적 → 가격 돌아오면 매도 자리
    - **BPR (둘 다 있는 자리)**: 매수·매도 흐름이 모두 강했던 격전지 →
      가격 돌아오면 **양쪽 다 가능, 단 어느 한쪽으로 강한 reversal** 발생

검출:
    1. 모든 bullish FVG 와 bearish FVG 를 비교
    2. 가격대가 overlap 하면 (둘의 high/low 가 교차하면) BPR 형성
    3. overlap 영역의 high, low = BPR zone

진입 운용:
    - HTF bias 와 일치하는 방향으로 진입 (BPR 자체는 양방향 신호)
    - mean threshold (BPR mid) 도달 시 진입 후보
    - Confluence 강도 ↑ — 일반 FVG 보다 2배 강한 신호로 간주
"""
from __future__ import annotations

from dataclasses import dataclass

from aurora_ict.indicators.fvg import FVG, FVGType


@dataclass(slots=True, frozen=True)
class BalancedPriceRange:
    """BPR 1개 — 두 반대 방향 FVG 의 overlap 영역.

    Attributes:
        high: BPR zone 상단 (두 FVG 의 max(low) 또는 min(high) 중 작은 쪽 = overlap top).
        low: BPR zone 하단 (overlap bottom).
        bullish_fvg_idx: 참여한 bullish FVG 의 봉 인덱스.
        bearish_fvg_idx: 참여한 bearish FVG 의 봉 인덱스.
        formed_at_idx: 두 FVG 중 나중에 형성된 것의 인덱스 (BPR 활성화 시점).
    """

    high: float
    low: float
    bullish_fvg_idx: int
    bearish_fvg_idx: int
    formed_at_idx: int

    @property
    def mean_threshold(self) -> float:
        """BPR 중심 — 진입 mean."""
        return (self.high + self.low) / 2.0

    @property
    def width(self) -> float:
        return self.high - self.low


def detect_bpr(
    fvgs: list[FVG],
    *,
    max_pair_distance_bars: int = 50,
) -> list[BalancedPriceRange]:
    """FVG 리스트에서 BPR (두 반대 방향 FVG overlap) 검출.

    Args:
        fvgs: ``detect_fvgs`` 결과 (bullish + bearish 섞임).
        max_pair_distance_bars: 두 FVG 의 봉 인덱스 차이 한도. 너무 멀리 떨어진
            FVG 끼리 매칭 안 함. default 50봉.

    Returns:
        ``BalancedPriceRange`` 리스트 — 형성 시점순.
    """
    if not fvgs:
        return []

    bullish = [f for f in fvgs if f.type is FVGType.BULLISH]
    bearish = [f for f in fvgs if f.type is FVGType.BEARISH]

    bprs: list[BalancedPriceRange] = []
    for bull in bullish:
        for bear in bearish:
            # 봉 거리 한도
            if abs(bull.idx - bear.idx) > max_pair_distance_bars:
                continue

            # overlap 영역 계산
            overlap_high = min(bull.high, bear.high)
            overlap_low = max(bull.low, bear.low)
            if overlap_high <= overlap_low:
                continue   # overlap 없음

            bprs.append(BalancedPriceRange(
                high=overlap_high,
                low=overlap_low,
                bullish_fvg_idx=bull.idx,
                bearish_fvg_idx=bear.idx,
                formed_at_idx=max(bull.idx, bear.idx),
            ))

    # 형성 시점순 정렬
    bprs.sort(key=lambda b: b.formed_at_idx)
    return bprs


__all__ = [
    "BalancedPriceRange",
    "detect_bpr",
]
