"""Inducement (IDM) — ICT 의 retail trap 개념.

ICT 정의:
    진짜 swing high/low 가 sweep 되기 전, 그 직전에 만들어지는 "가짜 작은
    swing". retail trader 가 이 작은 swing 을 진짜 reversal point 로 오해해
    진입하면, smart money 는 이 작은 swing 을 sweep 한 후 진짜 swing 까지 가져감.

식별:
    - 큰 swing high (예: 50봉 pivot) 직전 N봉 안에 작은 swing high (3봉 pivot)
    - 작은 high < 큰 high → inducement 후보 (큰 high 의 "유인용")
    - 동일하게 swing low 도 적용

활용:
    - 작은 swing 이 sweep 되면 retail 진입 트랩 발생 가능성 ↑
    - 그 후 가격이 큰 swing 까지 가는 시나리오 = 진입 회피 또는 reverse trade
"""

from __future__ import annotations

from dataclasses import dataclass

from aurora_ict.indicators.swing_points import SwingPoint, SwingType


@dataclass(slots=True)
class Inducement:
    """Inducement 한 건 — 작은 swing(유인용) + 큰 swing(진짜 target) 페어.

    Attributes:
        type: "high" (양쪽 모두 swing high) / "low" (양쪽 모두 swing low).
        target_idx: 큰 swing 의 df index (진짜 sweep target).
        target_price: 큰 swing 가격.
        target_ts_ms: 큰 swing 의 ts.
        idm_idx: 작은 swing 의 df index (inducement).
        idm_price: 작은 swing 가격.
        idm_ts_ms: 작은 swing 의 ts.
    """

    type: str
    target_idx: int
    target_price: float
    target_ts_ms: int
    idm_idx: int
    idm_price: float
    idm_ts_ms: int


def detect_inducements(
    big_swings: list[SwingPoint],
    small_swings: list[SwingPoint],
    lookback: int = 20,
) -> list[Inducement]:
    """큰 swing 직전 lookback 봉 안의 작은 swing (같은 방향) 페어링.

    조건:
    - 같은 방향 swing 만 페어 (high↔high, low↔low)
    - 큰 swing 의 idx - lookback ≤ 작은 swing.idx < 큰 swing.idx
    - High 페어: 작은 swing.price < 큰 swing.price (즉 작은 high 가 진짜 high 보다 낮음)
    - Low 페어: 작은 swing.price > 큰 swing.price (즉 작은 low 가 진짜 low 보다 높음)
    - 페어 후보 여럿이면 큰 swing 에 가장 가까운 (가장 최근) 작은 swing 선택

    Args:
        big_swings: 큰 swing pivot 리스트 (예: detect_swing_points(df, left=50)).
        small_swings: 작은 swing pivot 리스트 (예: left=1).
        lookback: 큰 swing 직전 몇 봉까지 작은 swing 찾을지.

    Returns:
        Inducement list — 큰 swing 시간순.
    """
    if not big_swings or not small_swings:
        return []

    inducements: list[Inducement] = []
    for big in big_swings:
        # 같은 방향 + lookback 범위 + price 조건 만족하는 후보들
        candidates: list[SwingPoint] = []
        for small in small_swings:
            if small.type is not big.type:
                continue
            if not (big.idx - lookback <= small.idx < big.idx):
                continue
            if big.type is SwingType.HIGH:
                if small.price < big.price:
                    candidates.append(small)
            elif small.price > big.price:
                candidates.append(small)
        if not candidates:
            continue
        # 큰 swing 에 가장 가까운 (idx 차이 최소) 후보 선택
        idm = max(candidates, key=lambda s: s.idx)
        inducements.append(Inducement(
            type=big.type.value,
            target_idx=big.idx,
            target_price=big.price,
            target_ts_ms=big.ts_ms,
            idm_idx=idm.idx,
            idm_price=idm.price,
            idm_ts_ms=idm.ts_ms,
        ))

    return inducements


__all__ = ["Inducement", "detect_inducements"]
