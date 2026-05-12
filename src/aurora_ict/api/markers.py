"""UI 차트 marker DTO — Aurora-ICT 박힌 거 박힘 박힘 박힘.

frontend 박힌 거 박힘 lightweight-charts 박힌 거 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘
박힘 박힘 박힘 박힘 박힘 marker 박힘 박힘 박힘 박힘.

박힌 거 박힘 5 종 marker:
1. **FVG box** — bullish/bearish 박힘, low/high price + duration
2. **Sweep wick** — bullish/bearish 박힘 박힘 박은 sweep 박힘 박힘
3. **Structure event** — BOS / CHoCH 박힘 박힘 line marker
4. **Swing pivot** — HIGH/LOW marker
5. **Killzone window** — vertical band (시작/끝 ts) + name
6. **Silver Bullet setup** — 박힌 setup 박은 entry/SL/TP marker

박힌 거 박힘 박힘 ``to_chart_markers(df)`` 박힌 거 박힘 박힘 박힘 ``ChartMarkers`` 박힘
박힘 박힘 박힘 박힘 박힘 박힘 박힘 ``.to_dict()`` 박힘 박힘 박힘 JSON 박힘 박힘 박힘 박힘
박힘 박힘 박힘 박힘 박힘.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

from aurora_ict.indicators.fvg import detect_fvgs
from aurora_ict.indicators.liquidity import detect_liquidity_sweeps
from aurora_ict.indicators.structure import detect_structure_events
from aurora_ict.indicators.swing_points import detect_swing_points
from aurora_ict.strategy.silver_bullet import detect_silver_bullet_setups
from aurora_ict.timing.killzone import classify_killzone, in_silver_bullet


@dataclass(slots=True)
class FVGMarker:
    """FVG box marker — 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은 박은."""

    ts_ms: int       # 박힌 박힌 박힌 박힘 박힌 거 (FVG 박힌 거 박힘 박힘 박힙 박힘)
    type: str        # "bullish" / "bearish"
    low: float
    high: float
    mean: float
    filled: bool
    invalidated: bool


@dataclass(slots=True)
class SweepMarker:
    """Liquidity Sweep marker — wick 박힌 거 박힘."""

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
class ChartMarkers:
    """전체 marker bundle — UI 박힌 거 박힘 박힘 박힘."""

    fvgs: list[FVGMarker] = field(default_factory=list)
    sweeps: list[SweepMarker] = field(default_factory=list)
    structure: list[StructureMarker] = field(default_factory=list)
    swings: list[SwingMarker] = field(default_factory=list)
    killzones: list[KillzoneMarker] = field(default_factory=list)
    setups: list[SetupMarker] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict 박힘 박힘 박힘."""
        return {
            "fvgs": [asdict(f) for f in self.fvgs],
            "sweeps": [asdict(s) for s in self.sweeps],
            "structure": [asdict(s) for s in self.structure],
            "swings": [asdict(s) for s in self.swings],
            "killzones": [asdict(k) for k in self.killzones],
            "setups": [asdict(s) for s in self.setups],
        }


def _ts_at_idx(df: pd.DataFrame, idx: int) -> int:
    """df 박힌 거 박힘 idx 박힌 거 박힌 거 박힘 ts_ms 박힘 박힘."""
    if isinstance(df.index, pd.DatetimeIndex):
        return int(df.index[idx].value // 10**6)
    return int(df.index[idx])


def to_chart_markers(
    df: pd.DataFrame,
    include_setups: bool = True,
    fvg_min_size_pct: float | None = 0.0005,
    min_rr: float = 2.0,
) -> ChartMarkers:
    """DataFrame 박힌 거 박힘 박힘 박힘 모든 ICT marker 박힘 박힘.

    Args:
        df: OHLCV DataFrame.
        include_setups: ``True`` 박힘 박힘 Silver Bullet setup 박힘 박힘 박힘 박힘.
        fvg_min_size_pct: FVG 박힌 최소 % size.
        min_rr: setup 박힌 최소 RR.

    Returns:
        ``ChartMarkers`` 박힘 박힘.
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

    # 3. Sweeps (swept flag 박힘 박힘 박힙 박힘 박힘 detect_liquidity_sweeps 박힘 박힘)
    sweeps = detect_liquidity_sweeps(df, swings)
    for sw in sweeps:
        markers.sweeps.append(SweepMarker(
            ts_ms=sw.ts_ms,
            type=sw.type.value,
            swept_price=sw.swept_price,
            wick_price=sw.wick_price,
        ))
    # swings 박힌 거 박힌 거 박힙 박힘 박힌 sweeps 박힌 거 박힘 박힘 swept flag 박힙 박힘 박힘
    # 박힘 박힘 markers.swings 박은 박은 박힌 거 박힘 박힘 다시 박힘 — 박힌 거 박힘 박힘 박힘
    # mutation 박힌 거 박힙 박힘. 박은 거 박힙 박힘 다시 박힘 박힘 박힙 박힘.
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

    # 5. Killzones — 박힌 박힌 박힘 박힘 봉 박힘 박힘 박힙 박힘 박힙 박힙 박힘 박힙 박힘 박힙
    # 박힘 박힘 — 시작/끝 박힌 거 박힘 박힘 박힘 박힙 박힘.
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
        # 마지막 박힘 박힘 박힘 박힘
        if prev_kz is not None and zone_start_ms is not None:
            last_ts = _ts_at_idx(df, len(df) - 1)
            markers.killzones.append(KillzoneMarker(
                start_ms=zone_start_ms,
                end_ms=last_ts,
                name=prev_kz,
                is_silver_bullet=in_silver_bullet(zone_start_ms) is not None,
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
    "SetupMarker",
    "StructureMarker",
    "SweepMarker",
    "SwingMarker",
    "to_chart_markers",
]
