"""ICT Signal generator — strategy 박힌 거 박힌 bot 박힐 박는 signal.

bot_ict_instance 박힌 거 박힌 거 박은 거:
1. 매 분 OHLCV (BTCUSDT 1m) fetch 박음
2. ``generate_ict_signal(df)`` 박은 거 박힌 거 박힘
3. ``action`` 박힌 거 박힌 거 박힘 박힌 거 박힘:
   - ``NO_ACTION`` — 진입 박힌 거 없음
   - ``ENTER_LONG`` / ``ENTER_SHORT`` — limit order 박은 거 박힘
   - ``CANCEL`` — 박힌 박힌 박힘 limit order 박은 거 박은 거 박힘

박힌 signal 박힌 거 박힘 = 박힌 setup 박힌 가장 가까운 거 박힘 박힘 박힘 — bot 박힌 거
박은 거 박힐 거 박힘 1개 시점 박힌 거 박힘 (실시간 박힘).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from aurora_ict.indicators.structure import TrendDirection
from aurora_ict.strategy.silver_bullet import (
    Direction,
    SilverBulletSetup,
    detect_silver_bullet_setups,
)


class SignalAction(str, Enum):
    """bot 박힐 박은 action 박힌 거."""

    NO_ACTION = "no_action"
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"


@dataclass(slots=True)
class ICTSignal:
    """ICT signal 1개.

    Attributes:
        action: 박힌 action 박힌 거.
        setup: 박힌 SilverBulletSetup 박힌 거 (``NO_ACTION`` 박힘 박힘 ``None``).
        symbol: 박힌 symbol (e.g. "BTCUSDT").
        ts_ms: signal 박은 ts (= 마지막 봉 박힘 박힘).
        reason: 박힌 reason 박힌 거 박힙 박힘.
    """

    action: SignalAction
    setup: SilverBulletSetup | None
    symbol: str
    ts_ms: int
    reason: str = ""

    @property
    def is_actionable(self) -> bool:
        """진입 박은 action 박힌 거 박힘."""
        return self.action in (SignalAction.ENTER_LONG, SignalAction.ENTER_SHORT)


def generate_ict_signal(
    df: pd.DataFrame,
    symbol: str,
    bias: TrendDirection | None = None,
    min_rr: float = 2.0,
    fvg_min_size_pct: float | None = 0.0005,
) -> ICTSignal:
    """OHLCV 박힌 거 박힘 박힌 ICT signal 박힘.

    Args:
        df: OHLCV DataFrame — 마지막 봉 박은 현재 시점.
        symbol: 박힌 symbol (e.g. "BTCUSDT").
        bias: HTF bias 박힘. ``None`` 박힘 자동 박힘.
        min_rr: 최소 RR.
        fvg_min_size_pct: FVG 박힌 최소 % size.

    Returns:
        ``ICTSignal`` — actionable 박힘 박힘 박은 entry 박은 거 박힘.
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

    # 가장 최근 setup 박은 거 박힘 (마지막 봉 박힌 거 박힘 박힘 박힌 setup)
    # — bot 박힌 거 박힘 박힘 박힐 박힘 박힌 거 박힘 1개 박힌 거 박힘
    setup = setups[-1]

    # 박힌 setup 박힌 거 박힘 마지막 봉 박힌 거 박힘 박힌 거 박힘 박힘 — stale setup 박힘 X
    # (박힌 거 박힌 거 박힌 거 박힘 박힌 거 박힘 박힌 거 박힘 박힘 박힘 박힘 박힘)
    # 박힌 거 박힌 거 박힘 = 박힌 봉 박힌 거 박힘 박힘 박힌 거 박힙 박힐 박힌 거 박힘 박힘
    # (within 5 bars 박힌 거 박힘 박힘 박힘 박힘)
    bars_since = len(df) - 1 - setup.fvg.idx
    if bars_since > 5:
        return ICTSignal(
            action=SignalAction.NO_ACTION,
            setup=None,
            symbol=symbol,
            ts_ms=last_ts_ms,
            reason=f"setup stale ({bars_since} bars)",
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
