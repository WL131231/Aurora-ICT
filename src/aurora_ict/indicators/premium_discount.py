"""Premium/Discount + Dealing Range — ICT 50% equilibrium.

ICT 박힌 정의:
- **Dealing Range** = 박힌 swing high → swing low 박힌 거 박은 range.
- **50% equilibrium** = (high + low) / 2 — 박힘 박힌 핵심 reference.
- **Premium** = 박힌 50% 박힌 위 (sell zone).
- **Discount** = 박힌 50% 박힌 아래 (buy zone).

ICT 박힌 원칙: **buy at discount, sell at premium**.

박힌 거 박힘 박힌 박힌 swing high/low 박힌 거 박힌 거 박힌 거 박은 거 (가장 최근 swing 박힌
거 박은 거). 박힌 거 박힘 박힌 거 박힌 dealing range 박힘 갱신 박힘 박힘 박힘.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from aurora_ict.indicators.swing_points import SwingPoint, SwingType


class PDZone(str, Enum):
    """Premium / Discount 박힌 zone."""

    PREMIUM = "premium"
    DISCOUNT = "discount"
    EQUILIBRIUM = "equilibrium"  # ±0.1% 박힌 거 박힌 50% 자리


@dataclass(slots=True)
class DealingRange:
    """Dealing range 1개.

    Attributes:
        high: swing high 박힌 가격.
        low: swing low 박힌 가격.
        high_idx: swing high index.
        low_idx: swing low index.
    """

    high: float
    low: float
    high_idx: int
    low_idx: int

    @property
    def equilibrium(self) -> float:
        """50% 박힌 자리."""
        return (self.high + self.low) / 2.0

    @property
    def size(self) -> float:
        """range 박힌 크기."""
        return self.high - self.low

    def classify(self, price: float, tolerance_pct: float = 0.001) -> PDZone:
        """price 박힌 거 박힌 zone 박힘.

        Args:
            price: 박힌 가격.
            tolerance_pct: equilibrium 박힌 거 박힌 ±% 박힌 거 (표준 0.1%).

        Returns:
            ``PDZone.PREMIUM`` / ``PDZone.DISCOUNT`` / ``PDZone.EQUILIBRIUM``.
        """
        eq = self.equilibrium
        eq_tol = eq * tolerance_pct
        if abs(price - eq) <= eq_tol:
            return PDZone.EQUILIBRIUM
        return PDZone.PREMIUM if price > eq else PDZone.DISCOUNT

    def fib_level(self, ratio: float) -> float:
        """Fibonacci level 박힌 거 박힘.

        - ratio=0 → low (discount 박힌 끝)
        - ratio=0.5 → equilibrium
        - ratio=0.618 → OTE sweet spot lower
        - ratio=0.786 → OTE sweet spot upper
        - ratio=1 → high (premium 박힌 끝)
        """
        return self.low + (self.high - self.low) * ratio


def latest_dealing_range(swings: list[SwingPoint]) -> DealingRange | None:
    """가장 최근 박힌 swing high + swing low 박힌 거 박힘 → dealing range 박힘.

    박힌 거 박힌 거 박힌 거 = 가장 최근 high 박힌 거 + 가장 최근 low 박힌 거 박힘 박힘.
    박힌 두 박힌 거 박힌 index 박힌 거 박힌 거 박힌 거 박힘 (chronological 박힘 X).

    Args:
        swings: swing point list.

    Returns:
        ``DealingRange`` 박힘, 박힌 swing 박힌 거 박힘 박힘 ``None``.
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
    """Optimal Trade Entry (OTE) zone 박힌 거 박힘.

    ICT OTE = Fibonacci 0.618 ~ 0.786 박힌 거 박힘 박은 박힘 sweet spot.

    Args:
        price: 박힌 가격.
        dealing_range: 박힌 dealing range.
        bias: ``"long"`` 박힘 박힘 박은 discount 박힌 OTE (0.618~0.786 박힘 박힌 low 박힌
            쪽 박힘 박은 거 — fib level 0.214~0.382 박힘 박힌 거 박힌 거 박힘).
            ``"short"`` 박힘 박힘 박은 premium 박힌 OTE (fib 0.618~0.786 박힘 박힌 거).

    Notes:
        ICT 박힌 fib 박힌 거 박힘 dealing range 박힌 거 박힘 박힌 거 박힌 거 박힘 ↔ swing
        박힌 거 박힙 박힘 박힌 거 박힘. 여기는 low=0, high=1 박힘 → long bias 박힘 박힘
        retrace 박힌 거 박힌 거 박힌 거 박힘 박은 박힘 박힘 박힌 거 = 0.214~0.382 박힘
        박힌 거 박힘.
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
