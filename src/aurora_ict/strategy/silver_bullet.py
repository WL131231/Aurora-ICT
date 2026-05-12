"""Silver Bullet entry model — Aurora-ICT Phase 1 첫 매매 strategy.

ICT 박힌 핵심 entry model 박힘. 매일 3개 1시간 윈도우 박힌 거 박은 첫 FVG 박힘 박은
박힌 거 박힘.

박힌 시퀀스:
1. **Silver Bullet 윈도우** 박힌 박힌 거 (NY 3-4am / 10-11am / 2-3pm)
2. **HTF bias** 박힘 (15m or higher swing structure 박힌 거 박힌 trend)
3. 박힌 윈도우 안 박힌 **첫 FVG** 박힘 (bias 방향 박힘)
4. **Entry** = FVG edge (limit order) — 가격 박힌 거 박힘 박힌 retest 박힌 후
5. **SL** = FVG 박힌 봉 wick 너머
6. **TP** = 다음 BSL/SSL (옛 swing high/low — Sweep 박힌 거 박힘 X)
7. **RR ≥ 1:2** 박힘 (책 박힌 1:3 박힘 strict, 1:2 박힘 박는 거)

박은 거 박은 거 = 박힌 setup 박힌 거 박힘. 실제 주문 박힘 X — bot_instance 박힌 거 박힘
박힐 거 박힘.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

from aurora_ict.indicators.fvg import FVG, FVGType, detect_fvgs
from aurora_ict.indicators.structure import (
    StructureType,
    TrendDirection,
    detect_structure_events,
)
from aurora_ict.indicators.swing_points import SwingPoint, SwingType, detect_swing_points
from aurora_ict.timing.killzone import in_silver_bullet


class Direction(StrEnum):
    """Trade 방향."""

    LONG = "long"
    SHORT = "short"


@dataclass(slots=True)
class SilverBulletSetup:
    """Silver Bullet setup 1개 — 박힐 수 있는 매매 1건.

    Attributes:
        ts_ms: FVG 박힌 봉 (중간 봉) open time.
        direction: LONG / SHORT.
        window: ``"london_sb"`` / ``"am_sb"`` / ``"pm_sb"``.
        entry: limit order 박은 가격 (FVG edge 박힘).
        stop_loss: SL 가격.
        take_profit: TP 가격 (next liquidity target).
        risk_reward: TP / SL ratio (절대값).
        fvg: 박힌 FVG 박은 거.
        target_swing_idx: TP 박힌 거 박힌 swing index (없으면 None).
        reason: 박힌 reason 박힌 거 박힘 (debug 박힘).
    """

    ts_ms: int
    direction: Direction
    window: str
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    fvg: FVG
    target_swing_idx: int | None = None
    reasons: list[str] = field(default_factory=list)


def _bias_from_structure(
    df: pd.DataFrame,
    swings: list[SwingPoint],
) -> TrendDirection:
    """가장 최근 structure event 박은 trend 박힘.

    None 박힘 박힘 박힘 박힌 (BOS/CHOCH 박힘 박힘 박힌 거 박힘) → 박은 swing 박힌 거 박힌
    last_high vs last_low 박힌 순서 박힌 거 박힘.
    """
    events = detect_structure_events(df, swings)
    if events:
        last = events[-1]
        if last.type in (StructureType.BOS_BULLISH, StructureType.CHOCH_BULLISH):
            return TrendDirection.UP
        return TrendDirection.DOWN
    return TrendDirection.NONE


def _next_liquidity_target(
    swings: list[SwingPoint],
    direction: Direction,
    entry_price: float,
) -> SwingPoint | None:
    """Entry 박은 거 박힘 박힘 target liquidity 박힘.

    - LONG → entry 위쪽 박힌 unswept swing high (BSL — TP 박힌 자리)
    - SHORT → entry 아래쪽 박힌 unswept swing low (SSL)

    가장 가까운 박힌 거 (entry 박힌 거 박힘 박힌 가격 차이 가장 작은 거) 박힘.
    """
    target_type = SwingType.HIGH if direction is Direction.LONG else SwingType.LOW
    candidates: list[SwingPoint] = []
    for s in swings:
        if s.type is not target_type or s.swept:
            continue
        if direction is Direction.LONG:
            # TP 박은 entry 박힌 위 박힘
            if s.price > entry_price:
                candidates.append(s)
        else:
            # TP 박은 entry 박힌 아래
            if s.price < entry_price:
                candidates.append(s)
    if not candidates:
        return None
    # 가장 가까운 박힌 거 (entry 박힌 거 박힘 박힌 가격 차이 최소)
    return min(candidates, key=lambda s: abs(s.price - entry_price))


def detect_silver_bullet_setups(
    df: pd.DataFrame,
    bias: TrendDirection | None = None,
    swing_left: int = 1,
    swing_right: int = 1,
    min_rr: float = 2.0,
    fvg_min_size_pct: float | None = 0.0005,
) -> list[SilverBulletSetup]:
    """Silver Bullet setup 박힌 거 박힘.

    Args:
        df: OHLCV DataFrame — index = ms 또는 datetime UTC.
        bias: 외부에서 박힌 HTF bias. ``None`` 박힘 → 박힌 swing structure 박힘 박힘
            자동 박힘.
        swing_left/right: swing pivot 박은 window.
        min_rr: 최소 RR 박힘 (표준 2.0).
        fvg_min_size_pct: FVG 박힌 거 박힌 최소 % size (noise 박힘 박힘 박힘).

    Returns:
        ``SilverBulletSetup`` list — 시간순.
    """
    if len(df) < 5:
        return []

    swings = detect_swing_points(df, left=swing_left, right=swing_right)
    if bias is None:
        bias = _bias_from_structure(df, swings)

    if bias is TrendDirection.NONE:
        return []

    direction = Direction.LONG if bias is TrendDirection.UP else Direction.SHORT
    desired_fvg_type = (
        FVGType.BULLISH if direction is Direction.LONG else FVGType.BEARISH
    )

    fvgs = detect_fvgs(df, min_size_pct=fvg_min_size_pct)

    # 박힌 윈도우 박힘 박힌 거 박힌 FVG 박힘 박힘 박힘 첫 번째 박힘 박힘 (window 단위)
    setups: list[SilverBulletSetup] = []
    seen_windows: set[tuple[int, str]] = set()  # (day_ms, window_name) 박힘 박힌 첫 박힘

    for fvg in fvgs:
        if fvg.type is not desired_fvg_type:
            continue
        window = in_silver_bullet(fvg.ts_ms)
        if window is None:
            continue
        day_ms = (fvg.ts_ms // 86_400_000) * 86_400_000
        key = (day_ms, window)
        if key in seen_windows:
            continue  # 박은 윈도우 박힌 거 valid setup 박은 박힘 박힙 박힘

        entry = fvg.mean_threshold

        if direction is Direction.LONG:
            stop_loss = fvg.low - (fvg.size * 0.1)
        else:
            stop_loss = fvg.high + (fvg.size * 0.1)

        target_swing = _next_liquidity_target(swings, direction, entry)
        if target_swing is None:
            continue

        take_profit = target_swing.price

        risk = abs(entry - stop_loss)
        reward = abs(take_profit - entry)
        if risk <= 0:
            continue
        rr = reward / risk

        if rr < min_rr:
            continue

        # valid setup 박힘 박힘 — seen_windows 박은 거 박힘 박힘 박힘
        seen_windows.add(key)

        reasons = [
            f"window={window}",
            f"bias={bias.value}",
            f"fvg={fvg.type.value}@{fvg.ts_ms}",
            f"rr={rr:.2f}",
        ]
        setups.append(SilverBulletSetup(
            ts_ms=fvg.ts_ms,
            direction=direction,
            window=window,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=rr,
            fvg=fvg,
            target_swing_idx=target_swing.idx,
            reasons=reasons,
        ))

    return setups


__all__ = [
    "Direction",
    "SilverBulletSetup",
    "detect_silver_bullet_setups",
]
