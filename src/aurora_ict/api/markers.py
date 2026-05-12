"""UI 차트 marker DTO — frontend (lightweight-charts)에 그릴 마커 일체.

지원 marker 6종:
1. **FVG box** — bullish/bearish, low/high price + duration
2. **Sweep wick** — bullish/bearish 방향의 liquidity sweep 표시
3. **Structure event** — BOS / CHoCH line marker
4. **Swing pivot** — HIGH/LOW marker
5. **Killzone window** — vertical band (시작/끝 ts) + name
6. **Silver Bullet setup** — 진입 후보 setup의 entry/SL/TP marker

``to_chart_markers(df)``가 ``ChartMarkers`` bundle을 생성하고, ``.to_dict()``로
JSON-직렬화 가능한 dict를 만들어 REST에 실어 보낸다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from aurora_ict.indicators.fvg import detect_fvgs
from aurora_ict.indicators.liquidity import detect_liquidity_sweeps
from aurora_ict.indicators.order_block import detect_order_blocks
from aurora_ict.indicators.structure import detect_structure_events
from aurora_ict.indicators.swing_points import detect_swing_points
from aurora_ict.strategy.silver_bullet import detect_silver_bullet_setups
from aurora_ict.timing.killzone import (
    classify_killzone,
    in_macro,
    in_silver_bullet,
)


@dataclass(slots=True)
class FVGMarker:
    """FVG box marker — frontend에 그릴 박스 정보."""

    ts_ms: int       # 중간 봉 open time (FVG의 anchor)
    type: str        # "bullish" / "bearish"
    low: float
    high: float
    mean: float
    filled: bool
    invalidated: bool


@dataclass(slots=True)
class SweepMarker:
    """Liquidity Sweep marker — sweep wick 정보."""

    ts_ms: int
    type: str        # "bullish" (SSL) / "bearish" (BSL)
    swept_price: float
    wick_price: float


@dataclass(slots=True)
class StructureMarker:
    """BOS / CHoCH line marker."""

    ts_ms: int
    type: str        # "bos_bullish" / "bos_bearish" / "choch_bullish" / "choch_bearish"
    broken_level: float


@dataclass(slots=True)
class SwingMarker:
    """Swing high/low pivot marker."""

    ts_ms: int
    type: str        # "high" / "low"
    price: float
    swept: bool


@dataclass(slots=True)
class KillzoneMarker:
    """Killzone window — start/end UTC ms."""

    start_ms: int
    end_ms: int
    name: str        # "london" / "ny_am" / "london_close" / "pm" / "asian"
    is_silver_bullet: bool = False


@dataclass(slots=True)
class SetupMarker:
    """Silver Bullet setup marker — entry/SL/TP."""

    ts_ms: int
    direction: str   # "long" / "short"
    window: str      # "london_sb" / "am_sb" / "pm_sb"
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float


@dataclass(slots=True)
class OrderBlockMarker:
    """Order Block marker — body 영역 (open/close) + 방향 + mitigation 상태."""

    ts_ms: int
    type: str        # "bullish" / "bearish"
    open: float
    high: float
    low: float
    close: float
    mitigated: bool


@dataclass(slots=True)
class MacroMarker:
    """Macro window marker — Silver Bullet 안의 정밀 sub-window (20~30분)."""

    start_ms: int
    end_ms: int
    name: str        # "london_macro_1" / "am_macro_2" / "lunch_macro" 등


@dataclass(slots=True)
class ChartMarkers:
    """전체 marker bundle — UI 한 번 호출로 받아갈 결과 묶음."""

    fvgs: list[FVGMarker] = field(default_factory=list)
    sweeps: list[SweepMarker] = field(default_factory=list)
    structure: list[StructureMarker] = field(default_factory=list)
    swings: list[SwingMarker] = field(default_factory=list)
    killzones: list[KillzoneMarker] = field(default_factory=list)
    setups: list[SetupMarker] = field(default_factory=list)
    order_blocks: list[OrderBlockMarker] = field(default_factory=list)
    macros: list[MacroMarker] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict 변환."""
        return {
            "fvgs": [asdict(f) for f in self.fvgs],
            "sweeps": [asdict(s) for s in self.sweeps],
            "structure": [asdict(s) for s in self.structure],
            "swings": [asdict(s) for s in self.swings],
            "killzones": [asdict(k) for k in self.killzones],
            "setups": [asdict(s) for s in self.setups],
            "order_blocks": [asdict(o) for o in self.order_blocks],
            "macros": [asdict(m) for m in self.macros],
        }


def _ts_at_idx(df: pd.DataFrame, idx: int) -> int:
    """df의 idx 위치 row를 ts_ms (UTC ms)로 변환."""
    if isinstance(df.index, pd.DatetimeIndex):
        return int(df.index[idx].value // 10**6)
    return int(df.index[idx])


def to_chart_markers(
    df: pd.DataFrame,
    include_setups: bool = True,
    fvg_min_size_pct: float | None = 0.0005,
    min_rr: float = 2.0,
) -> ChartMarkers:
    """DataFrame 한 건으로부터 모든 ICT marker를 한꺼번에 계산.

    Args:
        df: OHLCV DataFrame.
        include_setups: ``True``일 때 Silver Bullet setup marker도 포함.
        fvg_min_size_pct: FVG 최소 % size.
        min_rr: setup 최소 RR.

    Returns:
        ``ChartMarkers`` bundle.
    """
    if len(df) < 3:
        return ChartMarkers()

    markers = ChartMarkers()

    # 1. FVG
    fvgs = detect_fvgs(df, min_size_pct=fvg_min_size_pct)
    for fvg in fvgs:
        markers.fvgs.append(FVGMarker(
            ts_ms=fvg.ts_ms,
            type=fvg.type.value,
            low=fvg.low,
            high=fvg.high,
            mean=fvg.mean_threshold,
            filled=fvg.filled,
            invalidated=fvg.invalidated,
        ))

    # 2. Swings
    swings = detect_swing_points(df)
    for sw in swings:
        markers.swings.append(SwingMarker(
            ts_ms=sw.ts_ms,
            type=sw.type.value,
            price=sw.price,
            swept=sw.swept,
        ))

    # 3. Sweeps (detect_liquidity_sweeps가 swing.swept 플래그를 in-place로 갱신함)
    sweeps = detect_liquidity_sweeps(df, swings)
    for sw in sweeps:
        markers.sweeps.append(SweepMarker(
            ts_ms=sw.ts_ms,
            type=sw.type.value,
            swept_price=sw.swept_price,
            wick_price=sw.wick_price,
        ))
    # detect_liquidity_sweeps가 swing.swept를 갱신했으므로 markers.swings를 최신
    # swept 플래그로 재생성한다 (in-place mutation 결과 반영).
    markers.swings = [
        SwingMarker(
            ts_ms=sw.ts_ms,
            type=sw.type.value,
            price=sw.price,
            swept=sw.swept,
        )
        for sw in swings
    ]

    # 4. Structure events
    events = detect_structure_events(df, swings)
    for ev in events:
        markers.structure.append(StructureMarker(
            ts_ms=ev.ts_ms,
            type=ev.type.value,
            broken_level=ev.broken_level,
        ))

    # 4-b. Order Blocks
    obs = detect_order_blocks(df, displacement_bars=3, mark_mitigation=True)
    for ob in obs:
        markers.order_blocks.append(OrderBlockMarker(
            ts_ms=ob.ts_ms,
            type=ob.type.value,
            open=ob.open,
            high=ob.high,
            low=ob.low,
            close=ob.close,
            mitigated=ob.mitigated,
        ))

    # 5. Killzones — 봉 단위로 killzone 변경 시점을 모아 (start_ms, end_ms) 구간으로
    # 압축한다.
    if len(df) > 0:
        prev_kz: str | None = None
        zone_start_ms: int | None = None
        for i in range(len(df)):
            ts_ms = _ts_at_idx(df, i)
            kz = classify_killzone(ts_ms)
            kz_name = kz.value if kz is not None else None
            if kz_name != prev_kz:
                if prev_kz is not None and zone_start_ms is not None:
                    markers.killzones.append(KillzoneMarker(
                        start_ms=zone_start_ms,
                        end_ms=ts_ms,
                        name=prev_kz,
                        is_silver_bullet=in_silver_bullet(zone_start_ms) is not None,
                    ))
                prev_kz = kz_name
                zone_start_ms = ts_ms if kz_name is not None else None
        # 마지막 진행 중인 zone flush
        if prev_kz is not None and zone_start_ms is not None:
            last_ts = _ts_at_idx(df, len(df) - 1)
            markers.killzones.append(KillzoneMarker(
                start_ms=zone_start_ms,
                end_ms=last_ts,
                name=prev_kz,
                is_silver_bullet=in_silver_bullet(zone_start_ms) is not None,
            ))

    # 5-b. Macros — Silver Bullet 안의 정밀 sub-window (봉 단위 grouping)
    if len(df) > 0:
        prev_macro: str | None = None
        macro_start_ms: int | None = None
        for i in range(len(df)):
            ts_ms = _ts_at_idx(df, i)
            m_name = in_macro(ts_ms)
            if m_name != prev_macro:
                if prev_macro is not None and macro_start_ms is not None:
                    markers.macros.append(MacroMarker(
                        start_ms=macro_start_ms,
                        end_ms=ts_ms,
                        name=prev_macro,
                    ))
                prev_macro = m_name
                macro_start_ms = ts_ms if m_name is not None else None
        if prev_macro is not None and macro_start_ms is not None:
            last_ts = _ts_at_idx(df, len(df) - 1)
            markers.macros.append(MacroMarker(
                start_ms=macro_start_ms,
                end_ms=last_ts,
                name=prev_macro,
            ))

    # 6. Silver Bullet setups
    if include_setups:
        setups = detect_silver_bullet_setups(
            df,
            min_rr=min_rr,
            fvg_min_size_pct=fvg_min_size_pct,
        )
        for s in setups:
            markers.setups.append(SetupMarker(
                ts_ms=s.ts_ms,
                direction=s.direction.value,
                window=s.window,
                entry=s.entry,
                stop_loss=s.stop_loss,
                take_profit=s.take_profit,
                risk_reward=s.risk_reward,
            ))

    return markers


__all__ = [
    "ChartMarkers",
    "FVGMarker",
    "KillzoneMarker",
    "MacroMarker",
    "OrderBlockMarker",
    "SetupMarker",
    "StructureMarker",
    "SweepMarker",
    "SwingMarker",
    "to_chart_markers",
]
