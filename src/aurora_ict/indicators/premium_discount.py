"""Premium/Discount + Dealing Range — ICT 50% equilibrium.

ICT 정의:
- **Dealing Range** = 최근 swing high → swing low로 만든 range.
- **50% equilibrium** = (high + low) / 2 — 핵심 reference line.
- **Premium** = equilibrium 위 (sell zone).
- **Discount** = equilibrium 아래 (buy zone).

ICT 원칙: **buy at discount, sell at premium**.

range 계산은 가장 최근 swing high와 가장 최근 swing low를 사용한다. swing이 갱신되면
dealing range도 함께 갱신된다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from aurora_ict.indicators.swing_points import SwingPoint, SwingType


class PDZone(StrEnum):
    """Premium / Discount zone enum."""

    PREMIUM = "premium"
    DISCOUNT = "discount"
    EQUILIBRIUM = "equilibrium"  # ±0.1% 안의 50% 자리


@dataclass(slots=True)
class DealingRange:
    """Dealing range 1개.

    Attributes:
        high: swing high 가격.
        low: swing low 가격.
        high_idx: swing high index.
        low_idx: swing low index.
    """

    high: float
    low: float
    high_idx: int
    low_idx: int

    @property
    def equilibrium(self) -> float:
        """50% 라인."""
        return (self.high + self.low) / 2.0

    @property
    def size(self) -> float:
        """range 폭."""
        return self.high - self.low

    def classify(self, price: float, tolerance_pct: float = 0.001) -> PDZone:
        """price가 어느 zone에 속하는지 분류.

        Args:
            price: 분류할 가격.
            tolerance_pct: equilibrium 주변 ±% 허용 폭 (표준 0.1%).

        Returns:
            ``PDZone.PREMIUM`` / ``PDZone.DISCOUNT`` / ``PDZone.EQUILIBRIUM``.
        """
        eq = self.equilibrium
        eq_tol = eq * tolerance_pct
        if abs(price - eq) <= eq_tol:
            return PDZone.EQUILIBRIUM
        return PDZone.PREMIUM if price > eq else PDZone.DISCOUNT

    def fib_level(self, ratio: float) -> float:
        """Fibonacci level 계산.

        - ratio=0 → low (discount 끝)
        - ratio=0.5 → equilibrium
        - ratio=0.618 → OTE sweet spot lower
        - ratio=0.786 → OTE sweet spot upper
        - ratio=1 → high (premium 끝)
        """
        return self.low + (self.high - self.low) * ratio


def latest_dealing_range(swings: list[SwingPoint]) -> DealingRange | None:
    """가장 최근 swing high + 가장 최근 swing low로 dealing range 구성.

    두 swing의 index가 시간순서대로 정렬돼 있을 필요는 없다 (high가 더 최근일 수도,
    low가 더 최근일 수도 있다).

    Args:
        swings: swing point list.

    Returns:
        ``DealingRange``. high/low 중 하나라도 없으면 ``None``.
    """
    last_high: SwingPoint | None = None
    last_low: SwingPoint | None = None
    for swing in reversed(swings):
        if last_high is None and swing.type is SwingType.HIGH:
            last_high = swing
        elif last_low is None and swing.type is SwingType.LOW:
            last_low = swing
        if last_high is not None and last_low is not None:
            break
    if last_high is None or last_low is None:
        return None
    return DealingRange(
        high=last_high.price,
        low=last_low.price,
        high_idx=last_high.idx,
        low_idx=last_low.idx,
    )


def is_ote_zone(price: float, dealing_range: DealingRange, bias: str = "long") -> bool:
    """Optimal Trade Entry (OTE) zone 여부 판정.

    ICT OTE = Fibonacci 0.618 ~ 0.786 retrace 구간 sweet spot.

    Args:
        price: 판정할 가격.
        dealing_range: 기준 dealing range.
        bias: ``"long"``이면 discount 쪽 OTE — low=0 기준으로 low에 가까운 retrace
            영역(fib 0.214~0.382)을 사용.
            ``"short"``이면 premium 쪽 OTE — fib 0.618~0.786 구간.

    Notes:
        ICT 본문은 swing 시작점을 1, 끝점을 0으로 잡지만 여기서는 low=0, high=1로
        놓고 계산한다. 그래서 long bias의 retrace 구간이 fib 0.214~0.382로 표현된다.
    """
    if bias == "long":
        return dealing_range.fib_level(0.214) <= price <= dealing_range.fib_level(0.382)
    elif bias == "short":
        return dealing_range.fib_level(0.618) <= price <= dealing_range.fib_level(0.786)
    raise ValueError(f"invalid bias: {bias}")


__all__ = [
    "DealingRange",
    "PDZone",
    "is_ote_zone",
    "latest_dealing_range",
]
