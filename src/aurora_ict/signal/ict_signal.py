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
    min_rr: float = 2.0,
    fvg_min_size_pct: float | None = 0.0005,
    stale_bars: int = 5,
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
    )

    if not setups:
        return ICTSignal(
            action=SignalAction.NO_ACTION,
            setup=None,
            symbol=symbol,
            ts_ms=last_ts_ms,
            reason="no setup",
        )

    # 가장 최근 setup만 채택 (마지막 봉 기준) — bot은 시점당 1개 의사결정.
    setup = setups[-1]

    # 마지막 봉에서 setup 봉이 너무 멀면 stale 처리 (stale_bars 안에서만 신뢰).
    bars_since = len(df) - 1 - setup.fvg.idx
    if bars_since > stale_bars:
        return ICTSignal(
            action=SignalAction.NO_ACTION,
            setup=None,
            symbol=symbol,
            ts_ms=last_ts_ms,
            reason=f"setup stale ({bars_since} bars > {stale_bars})",
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
