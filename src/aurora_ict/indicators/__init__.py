"""Aurora-ICT indicators — ICT PD-Array / Liquidity / Structure detectors."""

from aurora_ict.indicators.fvg import (
    FVG,
    FVGType,
    detect_fvgs,
    mark_filled_and_invalidated,
)
from aurora_ict.indicators.liquidity import (
    EqualLevel,
    LiquiditySweep,
    SweepType,
    detect_equal_levels,
    detect_liquidity_sweeps,
)
from aurora_ict.indicators.premium_discount import (
    DealingRange,
    PDZone,
    is_ote_zone,
    latest_dealing_range,
)
from aurora_ict.indicators.structure import (
    StructureEvent,
    StructureType,
    TrendDirection,
    detect_structure_events,
)
from aurora_ict.indicators.swing_points import (
    SwingPoint,
    SwingType,
    detect_swing_points,
)

__all__ = [
    # FVG
    "FVG",
    "FVGType",
    "detect_fvgs",
    "mark_filled_and_invalidated",
    # Liquidity
    "EqualLevel",
    "LiquiditySweep",
    "SweepType",
    "detect_equal_levels",
    "detect_liquidity_sweeps",
    # Premium/Discount
    "DealingRange",
    "PDZone",
    "is_ote_zone",
    "latest_dealing_range",
    # Structure
    "StructureEvent",
    "StructureType",
    "TrendDirection",
    "detect_structure_events",
    # Swing
    "SwingPoint",
    "SwingType",
    "detect_swing_points",
]
