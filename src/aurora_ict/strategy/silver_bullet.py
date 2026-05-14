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
from aurora_ict.indicators.liquidity import detect_liquidity_sweeps
from aurora_ict.indicators.order_block import (
    OrderBlock,
    OrderBlockType,
    detect_order_blocks,
)
from aurora_ict.indicators.structure import (
    StructureType,
    TrendDirection,
    detect_structure_events,
)
from aurora_ict.indicators.swing_points import SwingPoint, SwingType, detect_swing_points
from aurora_ict.timing.killzone import (
    classify_killzone,
    in_macro,
    in_silver_bullet,
    macro_priority,
)


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
        take_profit: TP final 가격 (next liquidity target — 가변).
        risk_reward: TP final / SL ratio (절대값).
        tp1: 1R partial TP (entry ± 1R) — ICT 정통 첫 청산점.
        tp2: 2R partial TP.
        tp3: 3R partial TP.
        fvg: 트리거 FVG 객체.
        target_swing_idx: TP로 잡은 swing index (없으면 None).
        confluence_score: 같은 시점/방향 보강 지표 수 (0~3). OB/Macro/Sweep 각 +1.
        confluences: 어떤 confluence 가 발견됐는지 (디버그 / UI 표시용).
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
    tp1: float = 0.0
    tp2: float = 0.0
    tp3: float = 0.0
    target_swing_idx: int | None = None
    confluence_score: int = 0
    confluences: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # LuxAlgo Silver Bullet "Strict" mode — FVG 가 mean threshold 까지 retrace 됐는지.
    # entry limit 가 FVG mean 인데, 가격이 그 raw 까지 실제로 닿았는지 확인.
    # True → active setup (entry 가능). False → retrace 미발생 (대기 또는 skip).
    retraced: bool = False

    def __post_init__(self) -> None:
        """tp1/tp2/tp3 자동 계산 — 명시 안 됐을 때 1R/2R/3R 으로 채움."""
        r = abs(self.entry - self.stop_loss)
        sign = 1.0 if self.direction is Direction.LONG else -1.0
        if self.tp1 == 0.0:
            self.tp1 = self.entry + sign * r
        if self.tp2 == 0.0:
            self.tp2 = self.entry + sign * 2 * r
        if self.tp3 == 0.0:
            self.tp3 = self.entry + sign * 3 * r


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


def _ob_confluence(
    obs: list[OrderBlock],
    direction: Direction,
    setup_idx: int,
    lookback: int = 12,
) -> OrderBlock | None:
    """Setup 직전 ``lookback`` 봉 안의 같은 방향 OB 후보 탐색.

    같은 방향 = LONG → bullish OB / SHORT → bearish OB.
    아직 mitigated 되지 않은 OB 만 confluence 로 인정.
    """
    target = OrderBlockType.BULLISH if direction is Direction.LONG else OrderBlockType.BEARISH
    for ob in reversed(obs):
        if ob.type is not target:
            continue
        if ob.mitigated:
            continue
        if not (setup_idx - lookback <= ob.idx <= setup_idx):
            continue
        return ob
    return None


def _sweep_confluence(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    direction: Direction,
    setup_ts_ms: int,
    lookback_bars: int = 12,
) -> bool:
    """Setup 직전 ``lookback_bars`` 안에 같은 방향 sweep 이 있었는지.

    LONG → bullish sweep (SSL, 저점 sweep 후 반등)
    SHORT → bearish sweep (BSL, 고점 sweep 후 하락)

    detect_liquidity_sweeps 는 in-place 로 swing.swept 도 갱신하지만 본 함수는
    검출 결과 list 만 사용한다.
    """
    sweeps = detect_liquidity_sweeps(df, swings)
    from aurora_ict.indicators.liquidity import SweepType

    target = SweepType.BULLISH if direction is Direction.LONG else SweepType.BEARISH
    # 1m 봉 가정 시 lookback_bars × 60_000 ms 안의 sweep
    # 그러나 timeframe 가 다르면 부정확 — 가장 안전한 건 봉 개수 기반.
    # df 의 idx 가 sweep.idx 와 일치하므로 idx 차이로 계산.
    if isinstance(df.index, pd.DatetimeIndex):
        ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
    else:
        ts_arr = df.index.to_numpy()
    # setup_ts_ms 의 df index 찾기
    setup_idx = None
    for i, t in enumerate(ts_arr):
        if int(t) == setup_ts_ms:
            setup_idx = i
            break
    if setup_idx is None:
        return False
    for sw in sweeps:
        if sw.type is not target:
            continue
        if setup_idx - lookback_bars <= sw.idx <= setup_idx:
            return True
    return False


def detect_silver_bullet_setups(
    df: pd.DataFrame,
    bias: TrendDirection | None = None,
    swing_left: int = 1,
    swing_right: int = 1,
    min_rr: float = 2.0,
    fvg_min_size_pct: float | None = 0.0005,
    min_confluence: int = 0,
    expand_to_killzone: bool = False,
    disable_time_filter: bool = False,
) -> list[SilverBulletSetup]:
    """Silver Bullet setup 후보 검출.

    Args:
        df: OHLCV DataFrame — index = ms 또는 datetime UTC.
        bias: 외부에서 주입하는 HTF bias. ``None``이면 swing structure로 자동 추정.
        swing_left/right: swing pivot 윈도우.
        min_rr: 최소 RR (표준 2.0).
        fvg_min_size_pct: FVG 최소 % size (noise 필터).
        expand_to_killzone: ``True``면 Silver Bullet 1시간 윈도우 대신 Killzone
            전체 (Asian/London/NY_AM/Close/PM) 안 FVG 도 setup 으로 채택.
            진입 빈도 ↑ (5세션 ≈ 12시간/일 vs 기존 SB 3시간/일).
        disable_time_filter: ``True``면 SB / Killzone 시간 윈도우 검사 자체를 skip
            → 24시간 진입 허용 (window 라벨 = ``"any"``). expand_to_killzone 보다
            우선 적용.

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
    # mark_filled_and_invalidated 적용 — fvg.filled (retrace) / invalidated 갱신.
    # LuxAlgo Silver Bullet "Strict" mode 에서 retrace 검증에 사용.
    from aurora_ict.indicators.fvg import mark_filled_and_invalidated
    mark_filled_and_invalidated(fvgs, df)

    # OB 는 setup 검출 후 confluence 평가에서 사용 — 1회만 계산.
    obs = detect_order_blocks(df, displacement_bars=3, mark_mitigation=True)

    # 각 (day, window) 조합에 대해 첫 valid FVG 한 건만 setup으로 채택
    setups: list[SilverBulletSetup] = []
    seen_windows: set[tuple[int, str]] = set()  # (day_ms, window_name) — 일자별 1회 제한

    for fvg in fvgs:
        if fvg.type is not desired_fvg_type:
            continue
        # 시간 윈도우 검사 — disable_time_filter 면 skip (24h 매매).
        # 아니면 SB 윈도우 우선, 없으면 Killzone (expand 모드 시).
        if disable_time_filter:
            sb_win = in_silver_bullet(fvg.ts_ms)
            if sb_win is not None:
                window = sb_win
            else:
                kz = classify_killzone(fvg.ts_ms)
                window = kz.value if kz is not None else "any"
        else:
            window = in_silver_bullet(fvg.ts_ms)
            if window is None:
                if not expand_to_killzone:
                    continue
                kz = classify_killzone(fvg.ts_ms)
                if kz is None:
                    continue
                window = kz.value
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

        # Confluence 평가 — OB / Macro / Sweep 각 +1
        confluences: list[str] = []
        score = 0

        ob_match = _ob_confluence(obs, direction, fvg.idx)
        if ob_match is not None:
            score += 1
            confluences.append(f"ob={ob_match.type.value}@{ob_match.ts_ms}")

        macro_name = in_macro(fvg.ts_ms)
        if macro_name is not None:
            # 기본 +1, high priority macro (예: am_macro_2 9:50-10:10) 는 +2.
            priority = macro_priority(macro_name)
            if priority == "high":
                score += 2
                confluences.append(f"macro_high={macro_name}")
            elif priority == "low":
                # low priority 는 confluence 만 기록, score 가산 없음.
                confluences.append(f"macro_low={macro_name}")
            else:
                score += 1
                confluences.append(f"macro={macro_name}")

        if _sweep_confluence(df, swings, direction, fvg.ts_ms):
            score += 1
            confluences.append("sweep")

        # min_confluence 미달은 제외 (기본 0 이라 호환성 유지)
        if score < min_confluence:
            continue

        # valid setup 확정 — seen_windows에 추가해 같은 (day, window) 중복 차단
        seen_windows.add(key)

        reasons = [
            f"window={window}",
            f"bias={bias.value}",
            f"fvg={fvg.type.value}@{fvg.ts_ms}",
            f"rr={rr:.2f}",
            f"confluence={score}",
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
            confluence_score=score,
            confluences=confluences,
            reasons=reasons,
            retraced=fvg.filled,  # mean threshold 까지 닿았는지
        ))

    return setups


__all__ = [
    "Direction",
    "SilverBulletSetup",
    "detect_silver_bullet_setups",
]
