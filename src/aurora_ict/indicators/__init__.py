"""Aurora-ICT indicators — ICT PD-Array / Liquidity / Structure detectors."""

from aurora_ict.indicators.cbdr import (
    CBDRBiasState,
    CBDRBox,
    classify_price_vs_cbdr,
    detect_cbdr_boxes,
    is_within_acceptable_range,
)
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
from aurora_ict.indicators.mitigation_block import (
    MitigationBlock,
    detect_mitigation_blocks,
    filter_retested,
)
from aurora_ict.indicators.order_block import (
    OrderBlock,
    OrderBlockType,
    detect_order_blocks,
)
from aurora_ict.indicators.premium_discount import (
    DealingRange,
    PDZone,
    is_ote_zone,
    latest_dealing_range,
)
from aurora_ict.indicators.smt import (
    SmtEvent,
    SmtType,
    detect_smt_divergence,
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
from aurora_ict.indicators.trailing_extremes import (
    TrailingExtremes,
    compute_trailing_extremes,
)
from aurora_ict.indicators.turtle_soup import (
    TurtleSoupDirection,
    TurtleSoupSetup,
    detect_turtle_soup_setups,
)

__all__ = [
    # CBDR
    "CBDRBiasState",
    "CBDRBox",
    "classify_price_vs_cbdr",
    "detect_cbdr_boxes",
    "is_within_acceptable_range",
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
    # Mitigation Block
    "MitigationBlock",
    "detect_mitigation_blocks",
    "filter_retested",
    # Order Block
    "OrderBlock",
    "OrderBlockType",
    "detect_order_blocks",
    # Premium/Discount
    "DealingRange",
    "PDZone",
    "is_ote_zone",
    "latest_dealing_range",
    # SMT
    "SmtEvent",
    "SmtType",
    "detect_smt_divergence",
    # Structure
    "StructureEvent",
    "StructureType",
    "TrendDirection",
    "detect_structure_events",
    # Swing
    "SwingPoint",
    "SwingType",
    "detect_swing_points",
    # Trailing Extremes
    "TrailingExtremes",
    "compute_trailing_extremes",
    # Turtle Soup
    "TurtleSoupDirection",
    "TurtleSoupSetup",
    "detect_turtle_soup_setups",
]
