"""Silver Bullet entry model — Aurora-ICT Phase 1 첫 매매 strategy.

ICT의 대표 entry model. 매일 3개의 1시간 Silver Bullet 윈도우에서 첫 FVG를 잡아
진입하는 단순한 형태.

처리 시퀀스:
1. **Silver Bullet 윈도우** 진입 (NY 3-4am / 10-11am / 2-3pm)
2. **HTF bias** 결정 (15m 이상 swing structure 기준 trend)
3. 윈도우 안의 **첫 FVG** (bias 방향과 일치하는 것)
4. **Entry** = FVG mean threshold (limit order) — retest 노림
5. **SL** = FVG 봉 wick 너머
6. **TP** = 다음 BSL/SSL (아직 sweep되지 않은 swing high/low)
7. **RR ≥ 1:2** (책의 1:3은 strict, 1:2 정도면 채택)

이 모듈은 setup 후보만 생성한다. 실제 주문 집행은 bot_instance가 담당.
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
    """Silver Bullet setup 한 건 — 진입 후보 매매 1건.

    Attributes:
        ts_ms: FVG 봉 (중간 봉) open time.
        direction: LONG / SHORT.
        window: ``"london_sb"`` / ``"am_sb"`` / ``"pm_sb"``.
        entry: limit order 가격 (FVG mean threshold).
        stop_loss: SL 가격.
        take_profit: TP 가격 (next liquidity target).
        risk_reward: TP / SL ratio (절대값).
        fvg: 트리거 FVG 객체.
        target_swing_idx: TP로 잡은 swing index (없으면 None).
        reasons: 디버그용 사유 문자열 리스트.
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
    """가장 최근 structure event로부터 trend를 추출.

    이벤트가 없으면 (BOS/CHoCH 아직 미발생) → 마지막 high vs 마지막 low의 시간
    순서로 fallback 결정.
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
    """Entry 가격을 기준으로 다음 target liquidity 탐색.

    - LONG → entry 위쪽의 unswept swing high (BSL — TP 자리)
    - SHORT → entry 아래쪽의 unswept swing low (SSL)

    가장 가까운 것 (entry와 가격 차이가 최소) 반환.
    """
    target_type = SwingType.HIGH if direction is Direction.LONG else SwingType.LOW
    candidates: list[SwingPoint] = []
    for s in swings:
        if s.type is not target_type or s.swept:
            continue
        if direction is Direction.LONG:
            # TP는 entry 위쪽이어야 함
            if s.price > entry_price:
                candidates.append(s)
        else:
            # TP는 entry 아래쪽이어야 함
            if s.price < entry_price:
                candidates.append(s)
    if not candidates:
        return None
    # entry와 가격 차이가 가장 작은 후보
    return min(candidates, key=lambda s: abs(s.price - entry_price))


def detect_silver_bullet_setups(
    df: pd.DataFrame,
    bias: TrendDirection | None = None,
    swing_left: int = 1,
    swing_right: int = 1,
    min_rr: float = 2.0,
    fvg_min_size_pct: float | None = 0.0005,
) -> list[SilverBulletSetup]:
    """Silver Bullet setup 후보 검출.

    Args:
        df: OHLCV DataFrame — index = ms 또는 datetime UTC.
        bias: 외부에서 주입하는 HTF bias. ``None``이면 swing structure로 자동 추정.
        swing_left/right: swing pivot 윈도우.
        min_rr: 최소 RR (표준 2.0).
        fvg_min_size_pct: FVG 최소 % size (noise 필터).

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

    # 각 (day, window) 조합에 대해 첫 valid FVG 한 건만 setup으로 채택
    setups: list[SilverBulletSetup] = []
    seen_windows: set[tuple[int, str]] = set()  # (day_ms, window_name) — 일자별 1회 제한

    for fvg in fvgs:
        if fvg.type is not desired_fvg_type:
            continue
        window = in_silver_bullet(fvg.ts_ms)
        if window is None:
            continue
        day_ms = (fvg.ts_ms // 86_400_000) * 86_400_000
        key = (day_ms, window)
        if key in seen_windows:
            continue  # 같은 (day, window)에서 이미 valid setup 채택됨

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

        # valid setup 확정 — seen_windows에 추가해 같은 (day, window) 중복 차단
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
