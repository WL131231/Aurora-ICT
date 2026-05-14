"""LtfEntryConfirmer — HTF setup 활성 시 LTF 에서 진입 확정 신호 검증.

ICT 정통 multi-TF entry framework 2단계:
1. (PR #53) HtfSetupTracker — HTF setup 추적
2. (이 모듈) LtfEntryConfirmer — LTF 에서 진입 trigger 확정
3. (PR #55) MultiTfBotIctInstance — 통합 + 실행

확정 시퀀스:
1. 현재 가격이 HTF FVG zone 안에 들어왔는지 (이미 retrace 시작)
2. LTF 최근 lookback 봉 안에 HTF 와 같은 방향의 structure shift (BOS/CHoCH) 박혔는지
3. structure shift 이후 LTF FVG (같은 방향) 박혔는지 — entry trigger
4. 확정 시 entry = LTF FVG mean, SL = LTF FVG wick 너머, TP = HTF setup TP

이로써 HTF 그림에 LTF 미세 진입을 결합 — 정통 ICT.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aurora_ict.indicators.fvg import FVG, FVGType, detect_fvgs
from aurora_ict.indicators.structure import (
    StructureType,
    detect_structure_events,
)
from aurora_ict.indicators.swing_points import detect_swing_points
from aurora_ict.strategy.htf_setup_tracker import HtfActiveSetup
from aurora_ict.strategy.silver_bullet import Direction


@dataclass(slots=True)
class ConfirmedEntry:
    """LTF 에서 확정된 진입 신호.

    Attributes:
        direction: LONG / SHORT (HTF setup 방향과 일치).
        entry: 진입 가격 (LTF FVG mean).
        stop_loss: SL (LTF FVG wick 너머 + buffer).
        take_profit: TP (HTF setup 의 TP — final liquidity).
        ltf_fvg: LTF entry trigger FVG 객체.
        htf_tf: 원천 HTF.
        htf_setup_ts_ms: HTF setup ts_ms (이중 진입 방지용).
    """

    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    ltf_fvg: FVG
    htf_tf: str
    htf_setup_ts_ms: int


def _last_structure_event_in_lookback(
    ltf_df: pd.DataFrame,
    lookback_bars: int,
    desired_direction: Direction,
) -> int | None:
    """LTF lookback 안의 마지막 structure event (같은 방향) 의 봉 index.

    Args:
        ltf_df: LTF OHLCV DataFrame.
        lookback_bars: 최근 N 봉 안.
        desired_direction: 매칭할 방향 (LONG → BOS/CHoCH bullish).

    Returns:
        매칭 event 의 봉 index, 없으면 None.
    """
    if len(ltf_df) < 5:
        return None
    swings = detect_swing_points(ltf_df)
    if not swings:
        return None
    events = detect_structure_events(ltf_df, swings)
    if not events:
        return None
    bullish_types = (StructureType.BOS_BULLISH, StructureType.CHOCH_BULLISH)
    bearish_types = (StructureType.BOS_BEARISH, StructureType.CHOCH_BEARISH)
    target_types = bullish_types if desired_direction is Direction.LONG else bearish_types
    last_idx = len(ltf_df) - 1
    earliest_allowed_idx = max(0, last_idx - lookback_bars)
    for ev in reversed(events):
        if ev.idx < earliest_allowed_idx:
            return None
        if ev.type in target_types:
            return ev.idx
    return None


def _first_fvg_after_idx(
    ltf_df: pd.DataFrame,
    after_idx: int,
    desired_direction: Direction,
    min_size_pct: float | None,
) -> FVG | None:
    """주어진 idx 이후의 첫 LTF FVG (같은 방향).

    Args:
        ltf_df: LTF OHLCV DataFrame.
        after_idx: 이 봉 이후 FVG 만 채택.
        desired_direction: 매칭할 방향.
        min_size_pct: FVG 최소 % size.

    Returns:
        조건 맞는 FVG, 없으면 None.
    """
    fvgs = detect_fvgs(ltf_df, min_size_pct=min_size_pct)
    target = FVGType.BULLISH if desired_direction is Direction.LONG else FVGType.BEARISH
    for fvg in fvgs:
        if fvg.idx <= after_idx:
            continue
        if fvg.type is target:
            return fvg
    return None


def confirm_ltf_entry(
    htf_active: HtfActiveSetup,
    ltf_df: pd.DataFrame,
    *,
    lookback_bars: int = 30,
    fvg_min_size_pct: float | None = 0.0005,
    sl_buffer_ratio: float = 0.1,
) -> ConfirmedEntry | None:
    """HTF setup 활성 시 LTF 에서 진입 trigger 검증.

    검증 흐름:
    1. 현재 가격 (LTF 마지막 close) 이 HTF FVG zone 안에 있는지
    2. LTF lookback 봉 안에 같은 방향 structure shift (BOS/CHoCH) 있는지
    3. structure shift 이후 같은 방향 LTF FVG (entry trigger) 있는지

    Args:
        htf_active: 추적 중인 HTF active setup.
        ltf_df: LTF OHLCV DataFrame.
        lookback_bars: LTF structure shift 검색 lookback.
        fvg_min_size_pct: LTF FVG 최소 % size.
        sl_buffer_ratio: SL = FVG wick + size * ratio.

    Returns:
        확정된 진입 신호, 없으면 None.
    """
    if len(ltf_df) < 5:
        return None
    current_price = float(ltf_df["close"].iloc[-1])

    if not htf_active.contains_price(current_price):
        return None

    direction = htf_active.setup.direction
    shift_idx = _last_structure_event_in_lookback(
        ltf_df, lookback_bars, direction,
    )
    if shift_idx is None:
        return None

    fvg = _first_fvg_after_idx(
        ltf_df, shift_idx, direction, fvg_min_size_pct,
    )
    if fvg is None:
        return None

    entry = fvg.mean_threshold
    if direction is Direction.LONG:
        stop_loss = fvg.low - (fvg.size * sl_buffer_ratio)
    else:
        stop_loss = fvg.high + (fvg.size * sl_buffer_ratio)

    return ConfirmedEntry(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        take_profit=htf_active.setup.take_profit,
        ltf_fvg=fvg,
        htf_tf=htf_active.htf_tf,
        htf_setup_ts_ms=htf_active.setup.ts_ms,
    )


__all__ = [
    "ConfirmedEntry",
    "confirm_ltf_entry",
]
