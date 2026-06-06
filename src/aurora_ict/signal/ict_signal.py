"""ICT Signal generator — strategy 결과를 bot이 소비하는 signal로 변환.

bot_ict_instance 호출 흐름:
1. 매 분 OHLCV (BTCUSDT 1m) fetch
2. ``generate_ict_signal(df)`` 호출
3. ``action`` 기반 분기:
   - ``NO_ACTION`` — 진입할 대상 없음
   - ``ENTER_LONG`` / ``ENTER_SHORT`` — limit order 등록
   - ``CANCEL`` — 기존 미체결 limit order 취소 (현 구현엔 enum 미사용)

returned signal은 가장 최근 setup 1건만 반영 — bot은 시점당 1개 의사결정을 하는
실시간 루프 구조.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from aurora_ict.indicators.structure import TrendDirection
from aurora_ict.strategy.silver_bullet import (
    Direction,
    SilverBulletSetup,
    detect_silver_bullet_setups,
)


class SignalAction(StrEnum):
    """bot이 수행할 action."""

    NO_ACTION = "no_action"
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"


@dataclass(slots=True)
class ICTSignal:
    """ICT signal 1개.

    Attributes:
        action: 수행할 action.
        setup: 대상 SilverBulletSetup (``NO_ACTION``이면 ``None``).
        symbol: 대상 symbol (e.g. "BTCUSDT").
        ts_ms: signal 발생 ts (= 마지막 봉 시점).
        reason: 진단/로깅용 사유 문자열.
    """

    action: SignalAction
    setup: SilverBulletSetup | None
    symbol: str
    ts_ms: int
    reason: str = ""

    @property
    def is_actionable(self) -> bool:
        """진입성(action) signal 여부."""
        return self.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT)


def generate_ict_signal(
    df: pd.DataFrame,
    symbol: str,
    bias: TrendDirection | None = None,
    min_rr: float = 1.5,
    fvg_min_size_pct: float | None = 0.0005,
    stale_bars: int = 5,
    require_retrace: bool = False,
    expand_to_killzone: bool = False,
    disable_time_filter: bool = False,
    min_sl_distance_pct: float = 0.0,
    prefer_direction: Direction | None = None,
) -> ICTSignal:
    """OHLCV DataFrame으로부터 ICT signal을 한 건 생성.

    Args:
        df: OHLCV DataFrame — 마지막 봉이 현재 시점.
        symbol: 대상 symbol (e.g. "BTCUSDT").
        bias: HTF bias. ``None``이면 swing structure로 자동 추정.
        min_rr: 최소 RR.
        fvg_min_size_pct: FVG 최소 % size.

    Returns:
        ``ICTSignal`` — actionable이면 entry 정보 포함.
    """
    if isinstance(df.index, pd.DatetimeIndex):
        last_ts_ms = int(df.index[-1].value // 10**6)
    elif len(df) > 0:
        last_ts_ms = int(df.index[-1])
    else:
        last_ts_ms = 0

    if len(df) < 5:
        return ICTSignal(
            action=SignalAction.NO_ACTION,
            setup=None,
            symbol=symbol,
            ts_ms=last_ts_ms,
            reason="df too short",
        )

    setups = detect_silver_bullet_setups(
        df,
        bias=bias,
        min_rr=min_rr,
        fvg_min_size_pct=fvg_min_size_pct,
        expand_to_killzone=expand_to_killzone,
        disable_time_filter=disable_time_filter,
        min_sl_distance_pct=min_sl_distance_pct,
    )

    # v0.4.71+ Phase B: 새 4 source (Turtle / Mitigation / Implied / Rejection) setup
    # 추가 — 기존 FVG setup 과 같은 후보 리스트로 통합.
    from aurora_ict.strategy.silver_bullet import build_extra_source_setups
    extra_setups = build_extra_source_setups(
        df, min_rr=min_rr, bias=bias,
        disable_time_filter=disable_time_filter,
    )
    if extra_setups:
        setups = list(setups) + extra_setups
        # anchor_idx 순 정렬 — 가장 최근 setup 이 마지막에 오게
        setups.sort(key=lambda s: s.anchor_idx)

    # 2026-05-29 #MIN-SL-EXTRA fix: build_extra_source_setups (turtle/mitigation/
    # implied/rejection) 가 min_sl_distance_pct 인자를 받지 않아 SL 거리 가드
    # 우회됐던 버그. 새벽~오후 5건 매매 모두 source = turtle_soup / mitigation_block /
    # implied_fvg → SL 거리 검사 자체를 받지 않아 0.13~0.18% 타이트 SL setup
    # 통과 → 풀히트 100%. 합친 후 한 곳에서 SL 거리 가드 필터링으로 두 source
    # 경로 모두 일관 적용.
    if min_sl_distance_pct > 0 and setups:
        before_n = len(setups)
        kept = []
        for s in setups:
            entry_v = float(getattr(s, "entry", 0.0) or 0.0)
            sl_v = float(getattr(s, "stop_loss", 0.0) or 0.0)
            if entry_v <= 0 or sl_v <= 0:
                continue
            if (abs(entry_v - sl_v) / entry_v) >= min_sl_distance_pct:
                kept.append(s)
        setups = kept
        if len(setups) != before_n:
            # 운영 로그 — 어떤 source 가 필터링됐는지 다음 단계 grep 가능.
            import logging as _log
            _log.getLogger(__name__).info(
                "min_sl_distance_pct=%.4f 가드로 %d→%d setup 필터링",
                min_sl_distance_pct, before_n, len(setups),
            )

    if not setups:
        return ICTSignal(
            action=SignalAction.NO_ACTION,
            setup=None,
            symbol=symbol,
            ts_ms=last_ts_ms,
            reason="no setup",
        )

    # #BIAS-DIRECTION 2026-06-06: prefer_direction(HTF EMA 추세) 방향 setup 만
    # 후보로 선택 → 상승장 숏/하락장 롱을 원천 차단(간밤 19연속 숏 사고 방지).
    # 양방향 setup 중 추세 방향만 진입. 추세 방향 setup 이 없으면 진입 안 함.
    # prefer_direction=None 이면 기존 동작(최근 setup, 방향 무관).
    if prefer_direction is not None:
        dir_setups = [s for s in setups if s.direction is prefer_direction]
        if not dir_setups:
            return ICTSignal(
                action=SignalAction.NO_ACTION,
                setup=None,
                symbol=symbol,
                ts_ms=last_ts_ms,
                reason=f"no setup matching trend {prefer_direction.value}",
            )
        setups = dir_setups
    # 가장 최근 setup만 채택 (마지막 봉 기준) — bot은 시점당 1개 의사결정.
    setup = setups[-1]

    # 마지막 봉에서 setup 봉이 너무 멀면 stale 처리 (stale_bars 안에서만 신뢰).
    bars_since = len(df) - 1 - setup.anchor_idx
    if bars_since > stale_bars:
        return ICTSignal(
            action=SignalAction.NO_ACTION,
            setup=None,
            symbol=symbol,
            ts_ms=last_ts_ms,
            reason=f"setup stale ({bars_since} bars > {stale_bars})",
        )

    # LuxAlgo Silver Bullet "Strict" mode — FVG 가 mean threshold 까지 retrace 되어야 진입.
    if require_retrace and not setup.retraced:
        return ICTSignal(
            action=SignalAction.NO_ACTION,
            setup=None,
            symbol=symbol,
            ts_ms=last_ts_ms,
            reason="setup awaiting retrace",
        )

    action = (
        SignalAction.ENTER_LONG
        if setup.direction is Direction.LONG
        else SignalAction.ENTER_SHORT
    )

    return ICTSignal(
        action=action,
        setup=setup,
        symbol=symbol,
        ts_ms=last_ts_ms,
        reason="; ".join(setup.reasons),
    )


__all__ = ["ICTSignal", "SignalAction", "generate_ict_signal"]
