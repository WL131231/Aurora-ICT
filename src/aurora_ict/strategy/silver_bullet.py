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
from aurora_ict.indicators.implied_fvg import ImpliedFVG, ImpliedFVGType, detect_implied_fvgs
from aurora_ict.indicators.liquidity import LiquiditySweep, detect_liquidity_sweeps
from aurora_ict.indicators.mitigation_block import MitigationBlock, detect_mitigation_blocks
from aurora_ict.indicators.order_block import (
    OrderBlock,
    OrderBlockType,
    _atr,
    detect_order_blocks,
)
from aurora_ict.indicators.rejection_block import (
    RejectionBlock,
    RejectionBlockType,
    detect_rejection_blocks,
)
from aurora_ict.indicators.structure import (
    StructureType,
    TrendDirection,
    detect_structure_events,
)
from aurora_ict.indicators.swing_points import SwingPoint, SwingType, detect_swing_points
from aurora_ict.indicators.turtle_soup import (
    TurtleSoupDirection,
    TurtleSoupSetup,
    detect_turtle_soup_setups,
)
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


class SetupSource(StrEnum):
    """Setup 의 생성 출처 (v0.4.71+ Phase B 통합).

    봇 진입 흐름은 모든 source 의 setup 을 동일 ``SilverBulletSetup`` dataclass
    로 받지만, source 에 따라 SL/TP 위치 / confluence / fvg 의존성이 다름.
    """

    FVG = "fvg"                     # 기존 — Silver Bullet FVG retest
    TURTLE_SOUP = "turtle_soup"     # N-bar swing wick sweep + close 안쪽
    MITIGATION_BLOCK = "mitigation_block"   # mitigated OB 의 2차 retest
    IMPLIED_FVG = "implied_fvg"     # body gap (wick overlap)
    REJECTION_BLOCK = "rejection_block"     # wick 정밀 진입 zone
    MMBM = "mmbm"                   # #FST7 마켓메이커 매수/매도 모델 — HTF정합 반전 셋업
    # [08-10 연구] 정통 PD-array 중 미구현이던 것들. 프로덕션 미배포 —
    # 홀드아웃 4관문 통과가 배포 조건이다.
    INVERSE_FVG = "ifvg"            # 지지/저항 실패로 뒤집힌 FVG (모멘텀 첫 전환)
    BREAKER = "breaker"             # 추세 반대 OB 가 깨진 것 (failed OB)
    UNICORN = "unicorn"             # Breaker ∩ 같은 방향 FVG 겹침
    BPR_ZONE = "bpr"                # 반대 방향 FVG 두 개의 overlap


@dataclass(slots=True)
class SilverBulletSetup:
    """Silver Bullet setup 한 건 — 진입 후보 매매 1건.

    Attributes:
        ts_ms: FVG 봉 (중간 봉) open time.
        direction: LONG / SHORT.
        window: ``"london_sb"`` / ``"am_sb"`` / ``"pm_sb"``.
        entry: limit order 가격 (FVG mean threshold).
        stop_loss: SL 가격.
        take_profit: TP 가격 (next liquidity target). ICT 정통 단일 TP.
        risk_reward: TP / SL ratio (절대값). min_rr 2.0 이상 강제.
        fvg: 트리거 FVG 객체.
        target_swing_idx: TP로 잡은 swing index (없으면 None).
        confluence_score: 같은 시점/방향 보강 지표 합산 점수. 점수 매핑:
            - OB confluence (mitigated 안 된 같은 방향 OB lookback 12봉): +1
            - Macro priority high (예: am_macro_2 9:50-10:10): +2
            - Macro priority normal: +1 / low: +0 (기록만)
            - Sweep confluence (같은 방향 sweep lookback 12봉): +1
            - HTF supporting (변형 7 B+A, 외부 ``_apply_htf_supporting_boost`` 가 가산):
              합산 weight 4-9 → +1, 10-19 → +2, 20+ → +3
            range 는 0~10+ 까지 가능. 호출처 ``_calc_qty`` 가 position_pct 단계 결정.
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
    fvg: FVG | None = None
    target_swing_idx: int | None = None
    confluence_score: int = 0
    confluences: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    # LuxAlgo Silver Bullet "Strict" mode — FVG 가 mean threshold 까지 retrace 됐는지.
    # entry limit 가 FVG mean 인데, 가격이 그 raw 까지 실제로 닿았는지 확인.
    # True → active setup (entry 가능). False → retrace 미발생 (대기 또는 skip).
    retraced: bool = False
    # v0.4.71+ Phase B: 모든 source 공통 zone 필드 (fvg 가 None 인 새 source 용).
    # source="fvg" 면 자동 fvg.high/low/idx 로 채워짐 (post_init 또는 property).
    source: SetupSource = SetupSource.FVG
    _zone_high: float | None = None
    _zone_low: float | None = None
    _anchor_idx: int | None = None

    @property
    def zone_high(self) -> float:
        """모든 source 공통 zone 상단 — sb 진입 박스 + UI markers 등에서 사용.

        source=FVG 면 fvg.high 자동 반환. 다른 source 는 builder 에서 박은
        _zone_high 반환.
        """
        if self._zone_high is not None:
            return self._zone_high
        if self.fvg is not None:
            return self.fvg.high
        raise ValueError(f"setup ({self.source}) has no zone_high")

    @property
    def zone_low(self) -> float:
        if self._zone_low is not None:
            return self._zone_low
        if self.fvg is not None:
            return self.fvg.low
        raise ValueError(f"setup ({self.source}) has no zone_low")

    @property
    def anchor_idx(self) -> int:
        """setup 의 시간 기준 봉 index — stale 봉 계산용.

        FVG: fvg.idx. Turtle: sweep_idx. Mitigation: retest_idx. Implied: idx.
        Rejection: idx.
        """
        if self._anchor_idx is not None:
            return self._anchor_idx
        if self.fvg is not None:
            return self.fvg.idx
        raise ValueError(f"setup ({self.source}) has no anchor_idx")


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
    sweeps: list[LiquiditySweep],
    df: pd.DataFrame,
    direction: Direction,
    setup_ts_ms: int,
    lookback_bars: int = 12,
) -> bool:
    """Setup 직전 ``lookback_bars`` 안에 같은 방향 sweep 이 있었는지.

    LONG → bullish sweep (SSL, 저점 sweep 후 반등)
    SHORT → bearish sweep (BSL, 고점 sweep 후 하락)

    Args:
        sweeps: 호출처에서 루프 전 1회 검출해 주입한 sweep list.

    Why:
        과거엔 본 함수가 매 호출마다 ``detect_liquidity_sweeps(df, swings)`` 를
        다시 돌렸는데, 그게 공유 ``swings`` 를 in-place 로 ``swept=True`` 오염시켜
        뒤쪽 fvg 의 TP 후보(``_next_liquidity_target`` 는 unswept 만 채택)가
        사라지는 fvg 루프 순서 의존을 만들었다. sweep 을 루프 전 1회만 검출해
        주입받도록 바꿔 모든 fvg 가 동일 스냅샷을 보게 한다.
    """
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
    min_rr: float = 1.5,
    fvg_min_size_pct: float | None = 0.0005,
    min_confluence: int = 0,
    expand_to_killzone: bool = False,
    disable_time_filter: bool = False,
    min_sl_distance_pct: float = 0.0,
    ote_level: float = 0.5,
    nyse_gate: bool = True,
    window_once: bool = True,
    max_per_fvg: int = 0,
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
    atr_val = _atr_last(df)  # #LIVE-7: SL 최소 폭 (1.5×ATR)
    if bias is None:
        bias = _bias_from_structure(df, swings)
    # 2026-06-06 #BIAS-DIRECTION: bias 가 더 이상 진입 방향을 강제하지 않는다.
    # 양방향(BULLISH→롱, BEARISH→숏) setup 을 모두 생성하고, 추세 방향 최종 필터는
    # 진입 단계 HTF EMA bias 게이트(_passes_htf_ema_bias)가 담당한다. bias 는
    # confluence 보너스(+1)로만 쓴다.
    # (간밤 19연속 숏 사고: bias=DOWN 이 숏 setup 만 만들어 상승장에서 풀히트)
    bias_dir: Direction | None = None
    if bias is TrendDirection.UP:
        bias_dir = Direction.LONG
    elif bias is TrendDirection.DOWN:
        bias_dir = Direction.SHORT

    fvgs = detect_fvgs(df, min_size_pct=fvg_min_size_pct)
    # mark_filled_and_invalidated 적용 — fvg.filled (retrace) / invalidated 갱신.
    # LuxAlgo Silver Bullet "Strict" mode 에서 retrace 검증에 사용.
    from aurora_ict.indicators.fvg import mark_filled_and_invalidated
    mark_filled_and_invalidated(fvgs, df)

    # OB 는 setup 검출 후 confluence 평가에서 사용 — 1회만 계산.
    obs = detect_order_blocks(df, displacement_bars=3, mark_mitigation=True)

    # sweep 을 fvg 루프 전 1회만 검출 — swing.swept 상태를 확정해 모든 fvg 가
    # 동일 스냅샷을 본다. (매 fvg 마다 재검출하면 공유 swings 가 in-place 로
    # 오염돼 뒤쪽 fvg 의 TP 후보가 사라지는 순서 의존 발생 — #SWEEP-POLLUTION)
    all_sweeps = detect_liquidity_sweeps(df, swings)

    # 각 (day, window) 조합에 대해 첫 valid FVG 한 건만 setup으로 채택
    setups: list[SilverBulletSetup] = []
    seen_windows: set[tuple[int, str]] = set()  # (day_ms, window_name) — 일자별 1회 제한
    # [08-10] #FVG-REUSE 파트너 결정: 창당 1회 제한은 2026-05-12 첫 커밋부터
    # 있었고 근거 기록이 없다(정통 SB '하루 세 발' 개념을 그대로 옮긴 것으로
    # 보인다). 킬존·매크로까지 창을 넓힌 지금은 전제가 맞지 않고, 빈도가
    # 최대 약점인데 상한을 걸고 있었다. 대신 **같은 FVG 재사용을 제한**한다 —
    # FVG 는 채워질수록 힘이 빠지므로 2회까지가 합리적이라는 판단.
    fvg_use: dict[int, int] = {}

    for fvg in fvgs:
        # 양방향 — FVG 타입으로 진입 방향 결정 (bias 강제 제거).
        fvg_direction = (
            Direction.LONG if fvg.type is FVGType.BULLISH else Direction.SHORT
        )
        # 시간 윈도우 검사 — disable_time_filter 면 skip (24h 매매, referral).
        # 아니면 (sub_*): 2026-05-28 파트너 결정 — "미장 전체 X, 미장 안의
        # Killzone/Macro/Silver Bullet 만". in_trade_window_sub 가 게이트.
        if disable_time_filter:
            sb_win = in_silver_bullet(fvg.ts_ms)
            if sb_win is not None:
                window = sb_win
            else:
                kz = classify_killzone(fvg.ts_ms)
                window = kz.value if kz is not None else "any"
        else:
            # #KZ-WIDE 2026-08-09 (연구 스위치): nyse_gate=False 면 미장 제약을
            # 빼고 킬존/매크로/SB 이면 전부 허용한다. 2026-05-28 에 거래 #6
            # (뉴욕 03:02 런던 킬존, -283 USDT) 한 건을 보고 미장 밖을 막았는데
            # 그 결정은 이후 백테로 검증된 적이 없다.
            from aurora_ict.timing.killzone import (
                classify_killzone,
                in_macro,
                in_silver_bullet,
                in_trade_window_sub,
            )
            if nyse_gate:
                if not in_trade_window_sub(fvg.ts_ms):
                    continue
            elif not (classify_killzone(fvg.ts_ms) is not None
                      or in_silver_bullet(fvg.ts_ms) is not None
                      or in_macro(fvg.ts_ms) is not None):
                continue
            # window label — SB → Killzone → Macro 우선순위
            sb_win = in_silver_bullet(fvg.ts_ms)
            if sb_win is not None:
                window = sb_win
            else:
                kz = classify_killzone(fvg.ts_ms)
                if kz is not None:
                    window = kz.value
                else:
                    mc = in_macro(fvg.ts_ms)
                    window = mc if mc is not None else "any"
        day_ms = (fvg.ts_ms // 86_400_000) * 86_400_000
        key = (day_ms, window)
        if window_once and key in seen_windows:
            continue  # 같은 (day, window)에서 이미 valid setup 채택됨
        if max_per_fvg > 0 and fvg_use.get(int(fvg.idx), 0) >= max_per_fvg:
            continue  # 같은 FVG 를 이미 max_per_fvg 회 썼다

        entry = fvg.ote_threshold(ote_level)

        # 정통 ICT: SL = FVG 영역 가장자리 (단순). 추가 버퍼 없음.
        # Why: Silver Bullet PDF / Practical 모두 wick 너머 단순 정의.
        # FVG 폭의 10% 버퍼 + sl_buffer_ratio 는 정통 이탈이라 제거.
        if fvg_direction is Direction.LONG:
            stop_loss = fvg.low
        else:
            stop_loss = fvg.high
        stop_loss = _widen_sl_to_atr(entry, stop_loss, fvg_direction, atr_val)  # #LIVE-7

        target_swing = _next_liquidity_target(swings, fvg_direction, entry)
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

        # SL 거리 너무 작은 setup skip — noise-trap 회피.
        # SL 거리 / entry < min_sl_distance_pct 면 1분 안 SL hit 가능성 ↑.
        if min_sl_distance_pct > 0 and entry > 0:
            if (risk / entry) < min_sl_distance_pct:
                continue

        # Confluence 평가 — OB / Macro / Sweep 각 +1
        confluences: list[str] = []
        score = 0

        ob_match = _ob_confluence(obs, fvg_direction, fvg.idx)
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

        if _sweep_confluence(all_sweeps, df, fvg_direction, fvg.ts_ms):
            score += 1
            confluences.append("sweep")

        # #BIAS-DIRECTION: HTF/daily 추세(bias)와 같은 방향이면 confluence +1.
        # bias 가 방향을 강제하지 않는 대신 "추세 순응" 셋업에 가점을 준다.
        if bias_dir is not None and fvg_direction is bias_dir:
            score += 1
            confluences.append(f"bias={bias.value}")

        # min_confluence 미달은 제외 (기본 0 이라 호환성 유지)
        if score < min_confluence:
            continue

        # valid setup 확정 — seen_windows에 추가해 같은 (day, window) 중복 차단
        seen_windows.add(key)
        fvg_use[int(fvg.idx)] = fvg_use.get(int(fvg.idx), 0) + 1

        reasons = [
            f"window={window}",
            f"dir={fvg_direction.value}",
            f"bias={bias.value}",
            f"fvg={fvg.type.value}@{fvg.ts_ms}",
            f"rr={rr:.2f}",
            f"confluence={score}",
        ]
        setups.append(SilverBulletSetup(
            ts_ms=fvg.ts_ms,
            direction=fvg_direction,
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
    "SetupSource",
    "SilverBulletSetup",
    "build_extra_source_setups",
    "detect_silver_bullet_setups",
]


# ============================================================
# v0.4.71+ Phase B: 새 source 별 setup builder
# ============================================================
#
# Turtle Soup / Mitigation Block / Implied FVG / Rejection Block 의 detect
# 결과를 ``SilverBulletSetup`` 으로 변환. ICT 정통 SL/TP 위치를 각 source 별
# 정확히 박는다.
#
# SL/TP 정통 ICT 위치 (Practical + Bible):
#     - Turtle Soup: SL = sweep wick 너머 (+ buffer) / TP = opposite_target (N-bar 반대편 swing)
#     - Mitigation Block: SL = zone 반대편 (bullish 면 zone.low 아래) / TP = 다음 BSL/SSL
#     - Implied FVG: SL = body gap 반대편 wick 너머 / TP = 다음 BSL/SSL
#     - Rejection Block: SL = wick_extreme 너머 (+ buffer) / TP = 다음 BSL/SSL
#
# Entry 위치:
#     - 모두 zone mean (50% — CE Fibo) 박음. retest 진입 — 정통 OTE 접근.
#
# Confluence score:
#     - 새 source 자체 +1 (정통 setup 으로 인정)
#     - 같은 zone 안에 OB / sweep / macro 있으면 추가 가산 (기존 score 와 합산)

_SL_BUFFER_RATIO = 0.0005  # 0.05% — SL 살짝 여유 (정확한 wick 노이즈 회피)


def _find_target_swing(
    swings: list[SwingPoint],
    *,
    direction: Direction,
    after_idx: int,
    current_price: float,
) -> SwingPoint | None:
    """LONG: ``after_idx`` 이후 unswept swing high. SHORT: swing low.

    가장 가까운 (currently 가격 기준) 미체결 swing 만 채택 — TP target.
    """
    target_type = SwingType.HIGH if direction is Direction.LONG else SwingType.LOW
    candidates = [
        s for s in swings
        if s.type is target_type and s.idx > after_idx and not s.swept
    ]
    if not candidates:
        return None
    if direction is Direction.LONG:
        # 가장 가까운 (현재 가격 위쪽) swing high
        valid = [s for s in candidates if s.price > current_price]
        if not valid:
            return None
        return min(valid, key=lambda s: s.price - current_price)
    # SHORT — 가장 가까운 (가격 아래) swing low
    valid = [s for s in candidates if s.price < current_price]
    if not valid:
        return None
    return min(valid, key=lambda s: current_price - s.price)


def _calc_rr(
    direction: Direction,
    entry: float,
    sl: float,
    tp: float,
) -> float:
    """Risk-Reward 비율 (절대값)."""
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    reward = abs(tp - entry)
    return reward / risk


# #LIVE-7 (2026-05-23 사용자 결정): SL 폭 최소 = ATR_SL_MULT × ATR.
# 구조적 SL (FVG 가장자리 / wick) 이 5m 봉 변동의 절반(~0.05%) 밖에 안 돼 진입 직후
# 노이즈에 SL 털리던 문제 → ATR 기반 최소 폭 보장 (방향 맞아도 못 버티던 것 해소).
_ATR_SL_MULT = 1.5
_ATR_SL_PERIOD = 14


def _atr_last(df: pd.DataFrame, period: int = _ATR_SL_PERIOD) -> float:
    """df 마지막 봉의 ATR (Wilder) — SL 최소 폭 계산용. 실패 시 0.0."""
    if df is None or len(df) == 0:
        return 0.0
    series = _atr(
        df["high"].to_numpy(),
        df["low"].to_numpy(),
        df["close"].to_numpy(),
        period,
    )
    if len(series) == 0:
        return 0.0
    val = float(series.iloc[-1])
    return 0.0 if pd.isna(val) else val


def _widen_sl_to_atr(
    entry: float,
    sl: float,
    direction: Direction,
    atr_val: float,
) -> float:
    """SL 폭이 ``_ATR_SL_MULT × ATR`` 미만이면 ATR 기준으로 넓힘 (#LIVE-7).

    구조적 SL 이 더 넓으면 그대로 (ICT 구조 의미 존중). atr_val<=0 이면 원본 유지.
    """
    if atr_val <= 0:
        return sl
    min_dist = _ATR_SL_MULT * atr_val
    if abs(entry - sl) >= min_dist:
        return sl
    return entry - min_dist if direction is Direction.LONG else entry + min_dist


def _ts_ms_at_idx(df: pd.DataFrame, idx: int) -> int:
    """봉 idx 의 ts(ms) 추출 — timestamp 컬럼 우선, 없으면 DatetimeIndex/int index.

    #LIVE-2 fix: 새 source build 가 ``df["timestamp"]`` 컬럼만 보고 없으면 0 을 박던
    버그 — 봇 실제 df 는 timestamp 컬럼이 없고 DatetimeIndex 라 ts_ms=0 → step 의
    중복 진입 차단 (setup.ts_ms == _last_setup_ts_ms(0)) 에 걸려 새 source 진입 0건.
    """
    if "timestamp" in df.columns:
        return int(df["timestamp"].iloc[idx])
    if isinstance(df.index, pd.DatetimeIndex):
        return int(df.index[idx].value // 10**6)
    try:
        return int(df.index[idx])
    except (ValueError, TypeError):
        return 0


def _build_turtle_setup(
    ts: TurtleSoupSetup,
    df: pd.DataFrame,
    swings: list[SwingPoint],
    *,
    min_rr: float,
    atr_val: float = 0.0,
) -> SilverBulletSetup | None:
    """Turtle Soup → SilverBulletSetup. None 반환 시 setup 무효 (RR/swing 부족)."""
    direction = (Direction.SHORT if ts.direction is TurtleSoupDirection.SHORT
                 else Direction.LONG)

    # Entry = sweep_price 자체 (sweep 봉의 close 안쪽 자리)
    entry = ts.sweep_price
    # SL = wick_extreme 너머 (방향별 buffer)
    if direction is Direction.SHORT:
        sl = ts.wick_extreme * (1 + _SL_BUFFER_RATIO)
    else:
        sl = ts.wick_extreme * (1 - _SL_BUFFER_RATIO)
    sl = _widen_sl_to_atr(entry, sl, direction, atr_val)  # #LIVE-7

    # TP = opposite_target (N-bar 반대편). swing 보다 그게 더 명확 (Turtle 정통).
    tp = ts.opposite_target
    rr = _calc_rr(direction, entry, sl, tp)
    if rr < min_rr:
        return None

    return SilverBulletSetup(
        ts_ms=_ts_ms_at_idx(df, ts.sweep_idx),
        direction=direction,
        window="turtle",   # source 식별 — killzone window 아님
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_reward=rr,
        fvg=None,
        target_swing_idx=None,
        confluence_score=1,        # Turtle 자체 +1 (정통 setup)
        confluences=["turtle_soup"],
        reasons=[f"turtle soup {direction.value} sweep at {ts.sweep_price:.2f}"],
        retraced=True,             # sweep 봉 자체가 zone retest
        source=SetupSource.TURTLE_SOUP,
        _zone_high=max(ts.sweep_price, ts.wick_extreme),
        _zone_low=min(ts.sweep_price, ts.wick_extreme),
        _anchor_idx=ts.sweep_idx,
    )


def _build_mitigation_setup(
    mb: MitigationBlock,
    df: pd.DataFrame,
    swings: list[SwingPoint],
    *,
    min_rr: float,
    atr_val: float = 0.0,
) -> SilverBulletSetup | None:
    """Mitigation Block (retest 완료된) → SilverBulletSetup."""
    if mb.retest_idx is None:
        return None  # 아직 retest 안 됨

    direction = (Direction.LONG if mb.type is OrderBlockType.BULLISH
                 else Direction.SHORT)

    # Entry = zone 중간 (50% retest)
    entry = (mb.high + mb.low) / 2.0
    # SL = zone 반대편 너머 (FVG 와 동일 패턴)
    if direction is Direction.LONG:
        sl = mb.low * (1 - _SL_BUFFER_RATIO)
    else:
        sl = mb.high * (1 + _SL_BUFFER_RATIO)
    sl = _widen_sl_to_atr(entry, sl, direction, atr_val)  # #LIVE-7

    target_swing = _find_target_swing(
        swings, direction=direction, after_idx=mb.retest_idx,
        current_price=entry,
    )
    if target_swing is None:
        return None
    tp = target_swing.price
    rr = _calc_rr(direction, entry, sl, tp)
    if rr < min_rr:
        return None

    return SilverBulletSetup(
        ts_ms=_ts_ms_at_idx(df, mb.retest_idx),
        direction=direction,
        window="mitigation",
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_reward=rr,
        fvg=None,
        target_swing_idx=target_swing.idx,
        confluence_score=1,
        confluences=["mitigation_block"],
        reasons=[f"mitigation block {direction.value} 2nd retest"],
        retraced=True,
        source=SetupSource.MITIGATION_BLOCK,
        _zone_high=mb.high,
        _zone_low=mb.low,
        _anchor_idx=mb.retest_idx,
    )


def _build_zone_setup(
    z,
    df: pd.DataFrame,
    swings: list[SwingPoint],
    *,
    min_rr: float,
    atr_val: float = 0.0,
):
    """#SRC-ALL 2026-08-10: 연구용 Zone → SilverBulletSetup.

    `indicators/research_sources.Zone`(ifvg/breaker/unicorn/bpr)을 기존 소스들과
    **같은 규약**으로 셋업화한다 — 진입가 = 구간 중앙, SL = 구간 반대편 + 버퍼
    (ATR 바닥 적용), TP = 다음 미스윕 스윙, min_rr 미달이면 버린다.
    규약을 맞추는 게 중요한 이유: 소스마다 SL/TP 방식이 다르면 성적 차이가
    소스 때문인지 청산 규칙 때문인지 갈리지 않는다.
    """
    direction = Direction.LONG if z.bullish else Direction.SHORT
    entry = z.mean
    if entry <= 0:
        return None
    sl = (z.low * (1 - _SL_BUFFER_RATIO) if direction is Direction.LONG
          else z.high * (1 + _SL_BUFFER_RATIO))
    sl = _widen_sl_to_atr(entry, sl, direction, atr_val)

    target_swing = _find_target_swing(
        swings, direction=direction, after_idx=z.idx, current_price=entry,
    )
    if target_swing is None:
        return None
    tp = target_swing.price
    rr = _calc_rr(direction, entry, sl, tp)
    if rr < min_rr:
        return None

    src_map = {
        "ifvg": SetupSource.INVERSE_FVG,
        "breaker": SetupSource.BREAKER,
        "unicorn": SetupSource.UNICORN,
        "bpr": SetupSource.BPR_ZONE,
    }
    return SilverBulletSetup(
        ts_ms=z.ts_ms,
        direction=direction,
        window=z.kind,
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_reward=rr,
        fvg=None,
        target_swing_idx=target_swing.idx,
        confluence_score=1,
        confluences=[z.kind],
        reasons=[f"{z.kind} {direction.value}"],
        retraced=True,
        source=src_map.get(z.kind, SetupSource.FVG),
        _zone_high=z.high,
        _zone_low=z.low,
        _anchor_idx=z.idx,
    )


def _build_implied_fvg_setup(
    ifvg: ImpliedFVG,
    df: pd.DataFrame,
    swings: list[SwingPoint],
    *,
    min_rr: float,
    atr_val: float = 0.0,
) -> SilverBulletSetup | None:
    """Implied FVG → SilverBulletSetup. zone = body_high ~ body_low."""
    direction = (Direction.LONG if ifvg.type is ImpliedFVGType.BULLISH
                 else Direction.SHORT)

    entry = ifvg.mean_threshold
    if direction is Direction.LONG:
        sl = ifvg.body_low * (1 - _SL_BUFFER_RATIO)
    else:
        sl = ifvg.body_high * (1 + _SL_BUFFER_RATIO)
    sl = _widen_sl_to_atr(entry, sl, direction, atr_val)  # #LIVE-7

    target_swing = _find_target_swing(
        swings, direction=direction, after_idx=ifvg.idx, current_price=entry,
    )
    if target_swing is None:
        return None
    tp = target_swing.price
    rr = _calc_rr(direction, entry, sl, tp)
    if rr < min_rr:
        return None

    return SilverBulletSetup(
        ts_ms=_ts_ms_at_idx(df, ifvg.idx),
        direction=direction,
        window="implied_fvg",
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_reward=rr,
        fvg=None,
        target_swing_idx=target_swing.idx,
        confluence_score=1,
        confluences=["implied_fvg"],
        reasons=[f"implied fvg {direction.value} body gap"],
        retraced=True,
        source=SetupSource.IMPLIED_FVG,
        _zone_high=ifvg.body_high,
        _zone_low=ifvg.body_low,
        _anchor_idx=ifvg.idx,
    )


def _build_rejection_setup(
    rb: RejectionBlock,
    df: pd.DataFrame,
    swings: list[SwingPoint],
    *,
    min_rr: float,
    atr_val: float = 0.0,
) -> SilverBulletSetup | None:
    """Rejection Block → SilverBulletSetup. zone = wick_high ~ wick_low."""
    direction = (Direction.LONG if rb.type is RejectionBlockType.BULLISH
                 else Direction.SHORT)

    # Entry = wick 중간 (정밀 진입)
    entry = (rb.wick_high + rb.wick_low) / 2.0
    if direction is Direction.LONG:
        sl = rb.wick_low * (1 - _SL_BUFFER_RATIO)
    else:
        sl = rb.wick_high * (1 + _SL_BUFFER_RATIO)
    sl = _widen_sl_to_atr(entry, sl, direction, atr_val)  # #LIVE-7

    target_swing = _find_target_swing(
        swings, direction=direction, after_idx=rb.idx, current_price=entry,
    )
    if target_swing is None:
        return None
    tp = target_swing.price
    rr = _calc_rr(direction, entry, sl, tp)
    if rr < min_rr:
        return None

    return SilverBulletSetup(
        ts_ms=_ts_ms_at_idx(df, rb.idx),
        direction=direction,
        window="rejection",
        entry=entry,
        stop_loss=sl,
        take_profit=tp,
        risk_reward=rr,
        fvg=None,
        target_swing_idx=target_swing.idx,
        confluence_score=1,
        confluences=["rejection_block"],
        reasons=[f"rejection block {direction.value} wick"],
        retraced=True,
        source=SetupSource.REJECTION_BLOCK,
        _zone_high=rb.wick_high,
        _zone_low=rb.wick_low,
        _anchor_idx=rb.idx,
    )


def _phase_b_in_time_window(
    ts_ms: int, disable_time_filter: bool, nyse_gate: bool = True,
) -> bool:
    """Phase B 소스 시간 필터.

    2026-05-28 변경 — sub_* 정책 (disable_time_filter=False) 일 때 Phase A 와
    정합하게 in_trade_window_sub 사용 (미장 안의 Killzone/Macro/SB 만).

    disable_time_filter=True 면 무조건 통과, 아니면 SB 또는
    Killzone 안일 때만 통과. (Phase B 소스는 ``window`` 필드를 source 이름으로 쓰므로
    별도 시간 검사 필요. 사용자 결정 2026-05-27: 시간 필터를 Phase B 에도 적용해 손실 컷.)
    """
    if disable_time_filter:
        return True
    # 2026-05-28: sub_* 정책 — 미장 안의 Killzone/Macro/SB 만. Phase A 와 정합.
    from aurora_ict.timing.killzone import (
        classify_killzone,
        in_macro,
        in_silver_bullet,
        in_trade_window_sub,
    )
    if nyse_gate:
        return in_trade_window_sub(ts_ms)
    # #KZ-WIDE: 미장 제약 없이 킬존/매크로/SB 전체 (Phase A 와 동일 규칙)
    return (classify_killzone(ts_ms) is not None
            or in_silver_bullet(ts_ms) is not None
            or in_macro(ts_ms) is not None)


def build_extra_source_setups(
    df: pd.DataFrame,
    *,
    min_rr: float = 1.5,
    bias: TrendDirection | None = None,
    disable_time_filter: bool = True,
    nyse_gate: bool = True,
    research_sources: tuple[str, ...] = (),
    enable_turtle_soup: bool = True,
) -> list[SilverBulletSetup]:
    """v0.4.71+ Phase B: 새 4 source 의 detect 호출 + setup 변환.

    Args:
        df: OHLCV DataFrame.
        min_rr: 최소 RR (RR 미달 setup skip).
        bias: HTF bias. 박혀있으면 반대 방향 setup 제외 (bias 일치만).
        disable_time_filter: True 면 시간 필터 skip (기존 동작 / UI 마커 호환).
            False 면 ``ts_ms`` 가 SB 또는 Killzone 안 일 때만 통과 (2026-05-27 fix —
            Phase B 가 ``window`` 를 source 식별자로 써서 silver_bullet 의 시간 필터를
            우회하던 사각지대 메꿈).

    Returns:
        ``SilverBulletSetup`` 리스트 — 각 source 의 빈도 균등하게.
    """
    from aurora_ict.indicators.liquidity import detect_liquidity_sweeps
    from aurora_ict.indicators.order_block import detect_order_blocks
    from aurora_ict.indicators.swing_points import detect_swing_points

    if len(df) < 25:
        return []

    swings = detect_swing_points(df)
    atr_val = _atr_last(df)  # #LIVE-7: SL 최소 폭 (1.5×ATR) 계산용
    setups: list[SilverBulletSetup] = []

    # #BIAS-DIRECTION 2026-06-06: Phase B 도 bias 방향 강제 제거(양방향).
    # 추세 방향 최종 필터는 진입 단계 HTF EMA bias 게이트가 담당한다.
    def _accept(built: SilverBulletSetup | None) -> None:
        if built is None:
            return
        if not _phase_b_in_time_window(built.ts_ms, disable_time_filter, nyse_gate):
            return
        setups.append(built)

    # 1) Turtle Soup
    # #DROP-TURTLE 2026-08-11 (프로덕션 #421 과 동기화) — 기본 비활성.
    # 5년·7심볼·3,478건에서 유일한 적자 확정 소스(-0.109~-0.248R). 제거가 3단
    # 검증(홀드아웃 p=0.0034 · 플라시보 p=0.0000 · 다중비교 · 워크포워드)을 통과했다.
    if enable_turtle_soup:
        ts_setups = detect_turtle_soup_setups(df, lookback=20)
        for ts in ts_setups[-3:]:   # 최근 3개만 (오래된 거 stale)
            _accept(_build_turtle_setup(ts, df, swings, min_rr=min_rr, atr_val=atr_val))

    # 2) Mitigation Block
    detect_liquidity_sweeps(df, swings)   # mitigated 마킹
    obs = detect_order_blocks(df, swings=swings)
    mbs = detect_mitigation_blocks(obs, df)
    for mb in mbs[-3:]:
        _accept(_build_mitigation_setup(mb, df, swings, min_rr=min_rr, atr_val=atr_val))

    # 3) Implied FVG
    ifvgs = detect_implied_fvgs(df)
    for ifvg in ifvgs[-3:]:
        _accept(_build_implied_fvg_setup(
            ifvg, df, swings, min_rr=min_rr, atr_val=atr_val,
        ))

    # 4) Rejection Block
    rbs = detect_rejection_blocks(df)
    for rb in rbs[-3:]:
        _accept(_build_rejection_setup(rb, df, swings, min_rr=min_rr, atr_val=atr_val))

    # [08-10 연구] 정통 PD-array 중 미구현이던 소스들 — 기본 off.
    # 호출부가 명시적으로 켠 것만 돈다. 소스마다 검출 비용이 있어 전부 켜면 느리다.
    if research_sources:
        from aurora_ict.indicators.research_sources import ALL_DETECTORS
        for kind in research_sources:
            fn = ALL_DETECTORS.get(kind)
            if fn is None:
                continue
            for z in fn(df, swings)[-3:]:      # 다른 소스와 같은 규약(최근 3개)
                _accept(_build_zone_setup(z, df, swings,
                                          min_rr=min_rr, atr_val=atr_val))

    return setups
