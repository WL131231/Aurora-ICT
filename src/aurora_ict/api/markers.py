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

from aurora_ict.indicators.dol import compute_dol
from aurora_ict.indicators.fvg import (
    detect_fvgs,
    detect_ifvgs,
    mark_filled_and_invalidated,
)
from aurora_ict.indicators.liquidity import detect_equal_levels, detect_liquidity_sweeps
from aurora_ict.indicators.order_block import (
    detect_breaker_blocks,
    detect_order_blocks,
)
from aurora_ict.indicators.structure import detect_structure_events
from aurora_ict.indicators.swing_points import detect_swing_points
from aurora_ict.indicators.trailing_extremes import compute_trailing_extremes
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
class IFVGMarker:
    """Inversion FVG box marker — invalidated FVG 의 반전 entry zone."""

    ts_ms: int       # invalidation 봉 open time
    type: str        # 반전된 새 방향 "bullish" / "bearish"
    low: float
    high: float
    mean: float
    origin_fvg_ts_ms: int  # 원래 FVG anchor 봉 ts


@dataclass(slots=True)
class BreakerBlockMarker:
    """Breaker Block box marker — OB 가 close 로 깨진 후 반전 entry zone."""

    ts_ms: int       # break 봉 open time
    type: str        # 반전된 새 방향 "bullish" / "bearish"
    low: float
    high: float
    mean: float
    origin_ob_idx: int


@dataclass(slots=True)
class DOLMarker:
    """Draw On Liquidity — 다음 sweep target 가로선."""

    type: str        # "bullish" (위쪽 BSL) / "bearish" (아래쪽 SSL)
    price: float
    ts_ms: int       # 그 swing 의 ts (segment 시작점)
    distance: float  # 현재가에서 DOL 까지 가격 거리


@dataclass(slots=True)
class SweepMarker:
    """Liquidity Sweep marker — sweep wick + zone box + retest 정보."""

    ts_ms: int
    type: str        # "bullish" (SSL) / "bearish" (BSL)
    swept_price: float
    wick_price: float
    retested: bool = False
    # Zone box 좌표 (UI sweep zone 박스 렌더용)
    zone_top: float = 0.0
    zone_bottom: float = 0.0


@dataclass(slots=True)
class StructureMarker:
    """BOS / CHoCH segment marker — swing 시작점 → 돌파 봉.

    Attributes:
        ts_ms: 돌파 봉 (close 가 swing 가격을 넘은 봉) open time.
        type: "bos_bullish" / "bos_bearish" / "choch_bullish" / "choch_bearish".
        broken_level: 돌파된 swing 가격 (segment 의 가격 레벨).
        swing_ts_ms: 돌파된 swing 봉 open time (segment 시작점).
    """

    ts_ms: int
    type: str
    broken_level: float
    swing_ts_ms: int = 0  # 0 = 미설정 (호환)


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
    """Silver Bullet setup marker — entry/SL/TP + confluence."""

    ts_ms: int
    direction: str   # "long" / "short"
    window: str      # "london_sb" / "am_sb" / "pm_sb"
    entry: float
    stop_loss: float
    take_profit: float
    risk_reward: float
    confluence_score: int = 0
    confluences: list[str] = field(default_factory=list)


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
class TrailingExtremeMarker:
    """Strong/Weak High & Low — 차트 오른쪽 끝까지 그릴 가로선 한 쌍."""

    top_price: float
    bottom_price: float
    top_ts_ms: int
    bottom_ts_ms: int
    top_label: str         # "Strong High" / "Weak High" / "High"
    bottom_label: str      # "Strong Low" / "Weak Low" / "Low"


@dataclass(slots=True)
class EqualLevelMarker:
    """EQH (Equal Highs) / EQL (Equal Lows) — 같은 가격대 swing 클러스터.

    swing_ts_list 가 채워져 있으면 첫 swing → 마지막 swing 잇는 짧은 segment
    형태로 차트에 표시할 수 있다.
    """

    type: str                       # "high" (EQH) / "low" (EQL)
    price: float                    # 클러스터 평균 가격
    indices: list[int]              # 클러스터 swing 봉 index (df 안)
    swing_ts_list: list[int] = field(default_factory=list)  # 각 swing ts (ms)


@dataclass(slots=True)
class ChartMarkers:
    """전체 marker bundle — UI 한 번 호출로 받아갈 결과 묶음."""

    fvgs: list[FVGMarker] = field(default_factory=list)
    ifvgs: list[IFVGMarker] = field(default_factory=list)
    breakers: list[BreakerBlockMarker] = field(default_factory=list)
    dols: list[DOLMarker] = field(default_factory=list)
    sweeps: list[SweepMarker] = field(default_factory=list)
    structure: list[StructureMarker] = field(default_factory=list)
    swings: list[SwingMarker] = field(default_factory=list)
    killzones: list[KillzoneMarker] = field(default_factory=list)
    setups: list[SetupMarker] = field(default_factory=list)
    order_blocks: list[OrderBlockMarker] = field(default_factory=list)
    macros: list[MacroMarker] = field(default_factory=list)
    trailing: TrailingExtremeMarker | None = None
    # Internal scale (left=right=5) — 작은 swing / 짧은 구조 변화
    internal_swings: list[SwingMarker] = field(default_factory=list)
    internal_structure: list[StructureMarker] = field(default_factory=list)
    # Large scale (left=right=50) — 큰 swing / 장기 구조 변화
    large_swings: list[SwingMarker] = field(default_factory=list)
    large_structure: list[StructureMarker] = field(default_factory=list)
    large_sweeps: list[SweepMarker] = field(default_factory=list)
    # Equal Highs / Equal Lows (같은 가격대 swing 클러스터)
    equal_levels: list[EqualLevelMarker] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict 변환."""
        return {
            "fvgs": [asdict(f) for f in self.fvgs],
            "ifvgs": [asdict(f) for f in self.ifvgs],
            "breakers": [asdict(b) for b in self.breakers],
            "dols": [asdict(d) for d in self.dols],
            "sweeps": [asdict(s) for s in self.sweeps],
            "structure": [asdict(s) for s in self.structure],
            "swings": [asdict(s) for s in self.swings],
            "killzones": [asdict(k) for k in self.killzones],
            "setups": [asdict(s) for s in self.setups],
            "order_blocks": [asdict(o) for o in self.order_blocks],
            "macros": [asdict(m) for m in self.macros],
            "trailing": asdict(self.trailing) if self.trailing else None,
            "internal_swings": [asdict(s) for s in self.internal_swings],
            "internal_structure": [asdict(s) for s in self.internal_structure],
            "large_swings": [asdict(s) for s in self.large_swings],
            "large_structure": [asdict(s) for s in self.large_structure],
            "large_sweeps": [asdict(s) for s in self.large_sweeps],
            "equal_levels": [asdict(e) for e in self.equal_levels],
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

    # 1. FVG + IFVG (Inversion FVG — invalidated FVG 반전 zone)
    fvgs = detect_fvgs(df, min_size_pct=fvg_min_size_pct)
    mark_filled_and_invalidated(fvgs, df)
    ifvgs = detect_ifvgs(fvgs, df)
    for ifvg in ifvgs:
        # IFVG anchor origin FVG ts 룩업
        origin_ts = next(
            (f.ts_ms for f in fvgs if f.idx == ifvg.origin_fvg_idx),
            ifvg.ts_ms,
        )
        markers.ifvgs.append(IFVGMarker(
            ts_ms=ifvg.ts_ms,
            type=ifvg.type.value,
            low=ifvg.low,
            high=ifvg.high,
            mean=ifvg.mean_threshold,
            origin_fvg_ts_ms=origin_ts,
        ))
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
            retested=sw.retested,
            zone_top=sw.zone_top,
            zone_bottom=sw.zone_bottom,
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

    # 4. Structure events — swing 시작점 ts 함께 채움 (segment 시각화용)
    events = detect_structure_events(df, swings)
    # swing.idx → swing.ts_ms 룩업 dict (broken_swing_idx 에서 ts 박을 거)
    swing_ts_by_idx = {sw.idx: sw.ts_ms for sw in swings}
    for ev in events:
        markers.structure.append(StructureMarker(
            ts_ms=ev.ts_ms,
            type=ev.type.value,
            broken_level=ev.broken_level,
            swing_ts_ms=swing_ts_by_idx.get(ev.broken_swing_idx, ev.ts_ms),
        ))

    # 4-b. Internal scale (left=right=5) — 작은 swing / 짧은 구조 변화
    internal_pivots = detect_swing_points(df, left=5, right=5)
    for sw in internal_pivots:
        markers.internal_swings.append(SwingMarker(
            ts_ms=sw.ts_ms,
            type=sw.type.value,
            price=sw.price,
            swept=sw.swept,
        ))
    internal_events = detect_structure_events(df, internal_pivots)
    internal_ts_by_idx = {sw.idx: sw.ts_ms for sw in internal_pivots}
    for ev in internal_events:
        markers.internal_structure.append(StructureMarker(
            ts_ms=ev.ts_ms,
            type=ev.type.value,
            broken_level=ev.broken_level,
            swing_ts_ms=internal_ts_by_idx.get(ev.broken_swing_idx, ev.ts_ms),
        ))

    # 4-c. Large scale (left=right=50) — 큰 swing / 장기 구조 변화
    large_pivots = detect_swing_points(df, left=50, right=50)
    for sw in large_pivots:
        markers.large_swings.append(SwingMarker(
            ts_ms=sw.ts_ms,
            type=sw.type.value,
            price=sw.price,
            swept=sw.swept,
        ))
    large_events = detect_structure_events(df, large_pivots)
    large_ts_by_idx = {sw.idx: sw.ts_ms for sw in large_pivots}
    for ev in large_events:
        markers.large_structure.append(StructureMarker(
            ts_ms=ev.ts_ms,
            type=ev.type.value,
            broken_level=ev.broken_level,
            swing_ts_ms=large_ts_by_idx.get(ev.broken_swing_idx, ev.ts_ms),
        ))
    # 4-c2. Large sweeps — 50봉 swing 의 wick sweep (작은 swing 무관)
    large_sweeps = detect_liquidity_sweeps(df, large_pivots)
    for sw in large_sweeps:
        markers.large_sweeps.append(SweepMarker(
            ts_ms=sw.ts_ms,
            type=sw.type.value,
            swept_price=sw.swept_price,
            wick_price=sw.wick_price,
            retested=sw.retested,
            zone_top=sw.zone_top,
            zone_bottom=sw.zone_bottom,
        ))

    # 4-d. EQH / EQL — 같은 가격대 swing 클러스터 (liquidity)
    eql_levels = detect_equal_levels(swings, tolerance_pct=0.001, min_count=2)
    for lvl in eql_levels:
        ts_list = [
            swing_ts_by_idx[idx] for idx in lvl.indices if idx in swing_ts_by_idx
        ]
        markers.equal_levels.append(EqualLevelMarker(
            type=lvl.type.value,
            price=lvl.price,
            indices=list(lvl.indices),
            swing_ts_list=ts_list,
        ))

    # 4-a. Trailing extremes (Strong/Weak High & Low) — 단일 객체
    te = compute_trailing_extremes(df, swings, events)
    if te is not None:
        markers.trailing = TrailingExtremeMarker(
            top_price=te.top,
            bottom_price=te.bottom,
            top_ts_ms=te.top_ts_ms,
            bottom_ts_ms=te.bottom_ts_ms,
            top_label=te.top_label,
            bottom_label=te.bottom_label,
        )

    # 4-b. Order Blocks — LuxAlgo SMC 알고리즘 (BOS 트리거 + swing 구간 max range)
    # LuxAlgo 표준 swing length = 5 → internal_pivots (left=right=5) 사용.
    # 작은 swings(left=1) 으로 OB 잡으면 BOS event 너무 많아 OB 도배됨.
    obs = detect_order_blocks(
        df,
        algorithm="luxalgo",
        swings=internal_pivots,
        events=internal_events,
        mark_mitigation=True,
    )
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

    # 4-c-2. DOL — 현재가 기준 위/아래 가장 가까운 unswept swing
    for dol in compute_dol(df, swings):
        markers.dols.append(DOLMarker(
            type=dol.type,
            price=dol.price,
            ts_ms=dol.ts_ms,
            distance=dol.distance,
        ))

    # 4-c. Breaker Blocks — OB 가 close 로 깨졌을 때 반전 zone
    breakers = detect_breaker_blocks(obs, df)
    for bb in breakers:
        markers.breakers.append(BreakerBlockMarker(
            ts_ms=bb.ts_ms,
            type=bb.type.value,
            low=bb.low,
            high=bb.high,
            mean=bb.mean,
            origin_ob_idx=bb.origin_ob_idx,
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
                confluence_score=s.confluence_score,
                confluences=list(s.confluences),
            ))

    return markers


__all__ = [
    "BreakerBlockMarker",
    "ChartMarkers",
    "DOLMarker",
    "EqualLevelMarker",
    "FVGMarker",
    "IFVGMarker",
    "KillzoneMarker",
    "MacroMarker",
    "OrderBlockMarker",
    "SetupMarker",
    "StructureMarker",
    "SweepMarker",
    "SwingMarker",
    "TrailingExtremeMarker",
    "to_chart_markers",
]
