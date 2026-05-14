"""Multi-Timeframe Bias — ICT 정통 HTF → MTF → LTF 계층 구조.

ICT 정통 흐름:
    1. HTF (Daily/4h) 에서 bias (long/short) 확인
    2. MTF (1h/15m) 에서 POI (OB/FVG) 식별
    3. LTF (5m/1m) 에서 refined entry trigger

이 모듈은 (1) 의 HTF bias 만 담당. Trade TF 에 대응하는 HTF1, HTF2 두 단계를
훑어 두 bias 가 일치할 때만 진입을 허용.

Bias 결합 규칙:
    - HTF1 UP + HTF2 UP → UP (양쪽 합의 → 진입 OK)
    - HTF1 DOWN + HTF2 DOWN → DOWN
    - 한쪽이 NONE → 다른쪽 따름 (한쪽이 아직 미정이라도 다른쪽 추세 사용)
    - 두 HTF 충돌 (UP vs DOWN) → NONE (진입 금지, 추세 불확실)
"""

from __future__ import annotations

import pandas as pd

from aurora_ict.indicators.structure import (
    StructureType,
    TrendDirection,
    detect_structure_events,
)
from aurora_ict.indicators.swing_points import detect_swing_points

# Trade TF → (HTF1, HTF2) 자동 매핑.
# HTF1 = trade TF 직상 단계, HTF2 = bias 확정 단계 (HTF1 위).
# 1d / 1w 같은 큰 TF 는 HTF2 가 없거나 둘 다 없음 (이미 충분히 큼).
_HTF_MAP: dict[str, tuple[str | None, str | None]] = {
    "5m":  ("1h", "4h"),
    "15m": ("1h", "4h"),
    "30m": ("4h", "1d"),
    "1h":  ("4h", "1d"),
    "2h":  ("4h", "1d"),
    "4h":  ("1d", "1w"),
    "1d":  ("1w", None),
    "1w":  (None, None),
}


def htf_pair(trade_tf: str) -> tuple[str | None, str | None]:
    """Trade TF 에 대응하는 (HTF1, HTF2) 반환.

    Args:
        trade_tf: 사용자 선택 Trade TF (5m/15m/30m/1h/2h/4h/1d/1w).

    Returns:
        (HTF1, HTF2) 튜플. 매핑 없는 TF 는 (None, None).
    """
    return _HTF_MAP.get(trade_tf, (None, None))


def compute_bias_from_df(df: pd.DataFrame) -> TrendDirection:
    """HTF DataFrame 에서 trend direction 추출.

    마지막 structure event 기반:
    - BOS_BULLISH / CHOCH_BULLISH → UP
    - BOS_BEARISH / CHOCH_BEARISH → DOWN
    - event 없음 → NONE

    Args:
        df: HTF OHLCV DataFrame (open/high/low/close). 5봉 미만이면 NONE.

    Returns:
        TrendDirection.
    """
    if df is None or len(df) < 5:
        return TrendDirection.NONE
    swings = detect_swing_points(df)
    if not swings:
        return TrendDirection.NONE
    events = detect_structure_events(df, swings)
    if not events:
        return TrendDirection.NONE
    last = events[-1]
    if last.type in (StructureType.BOS_BULLISH, StructureType.CHOCH_BULLISH):
        return TrendDirection.UP
    return TrendDirection.DOWN


def combine_bias(
    htf1: TrendDirection,
    htf2: TrendDirection,
) -> TrendDirection:
    """두 HTF bias 결합.

    - 둘 다 같은 방향 → 그 방향
    - 한쪽 NONE → 다른쪽 따름
    - 충돌 (UP vs DOWN) → **HTF1 우선** (직속 상위 TF 가 가장 적절)

    충돌 시 NONE 차단보다 HTF1 우선이 진입 빈도 ↑. ICT 정통은 HTF1 (Trade TF
    직속) 의 신호도 충분히 강한 entry trigger 로 인정.
    """
    if htf1 is TrendDirection.NONE:
        return htf2
    if htf2 is TrendDirection.NONE:
        return htf1
    if htf1 == htf2:
        return htf1
    # 충돌 — HTF1 우선 (직속 상위 TF 의 단기 추세 따름)
    return htf1


__all__ = [
    "combine_bias",
    "compute_bias_from_df",
    "htf_pair",
]
