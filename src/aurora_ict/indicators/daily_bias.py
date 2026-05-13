"""Daily Bias — ICT 정통의 고정 reference point 기반 방향성.

ICT 정통 Daily Bias:
    - PDH (Previous Day High) = 전일 (UTC 일봉) high
    - PDL (Previous Day Low) = 전일 low
    - PDC (Previous Day Close) = 전일 close
    - PWO (Previous Week Open) = 지난 주 open (월요일 00:00 UTC)

판정 규칙 (단순화):
    - 현재 close > PDH → UP (전일 high 돌파 → 매수 우위)
    - 현재 close < PDL → DOWN (전일 low 이탈 → 매도 우위)
    - PDL ≤ current ≤ PDH → NONE (range 안, 방향 불확실)

이는 ICT 의 "Daily Wrap" 개념 — 전일 range 가 새로 작용하는 매물대.
multi_tf_bias (HTF1/HTF2 structure) 와 결합해 더 견고한 진입 필터로 사용.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aurora_ict.indicators.structure import TrendDirection


@dataclass(slots=True)
class DailyLevels:
    """Daily 고정 reference 가격 묶음.

    Attributes:
        pdh: Previous Day High.
        pdl: Previous Day Low.
        pdo: Previous Day Open.
        pdc: Previous Day Close.
        pwo: Previous Week Open. 일봉 5개 미만이면 None.
        pdh_ts_ms / pdl_ts_ms / pdo_ts_ms / pdc_ts_ms / pwo_ts_ms: 각 봉 ts.
    """

    pdh: float
    pdl: float
    pdo: float
    pdc: float
    pdh_ts_ms: int
    pdl_ts_ms: int
    pdo_ts_ms: int
    pdc_ts_ms: int
    pwo: float | None = None
    pwo_ts_ms: int | None = None


def compute_daily_levels(daily_df: pd.DataFrame) -> DailyLevels | None:
    """1d 봉 DataFrame 에서 전일 H/L/O/C + 전주 open 추출.

    Args:
        daily_df: 1d timeframe OHLCV DataFrame. 마지막 봉 = 오늘 (진행 중일 수
            있음), 그 직전 봉 = 어제.

    Returns:
        DailyLevels — 일봉 2개 미만이면 None.
    """
    if daily_df is None or len(daily_df) < 2:
        return None

    # 마지막 봉 = 오늘(진행 중), -2 번째 = 어제
    prev = daily_df.iloc[-2]
    prev_ts = _ts_of(daily_df, -2)

    pwo: float | None = None
    pwo_ts: int | None = None
    # 지난 주 월요일 = 오늘 - (오늘 weekday + 7) 일.
    # 예: 오늘 화 (weekday=1) → 8일 전 / 오늘 일 (weekday=6) → 13일 전.
    if isinstance(daily_df.index, pd.DatetimeIndex):
        today_dow = daily_df.index[-1].weekday()  # 0=월
        target_offset = today_dow + 7
        target_idx = len(daily_df) - 1 - target_offset
        if target_idx >= 0:
            pwo = float(daily_df["open"].iloc[target_idx])
            pwo_ts = _ts_of(daily_df, target_idx)

    return DailyLevels(
        pdh=float(prev["high"]),
        pdl=float(prev["low"]),
        pdo=float(prev["open"]),
        pdc=float(prev["close"]),
        pdh_ts_ms=prev_ts,
        pdl_ts_ms=prev_ts,
        pdo_ts_ms=prev_ts,
        pdc_ts_ms=prev_ts,
        pwo=pwo,
        pwo_ts_ms=pwo_ts,
    )


def compute_daily_bias(
    daily_df: pd.DataFrame,
    current_close: float,
) -> TrendDirection:
    """전일 H/L 과 현재가 비교로 daily bias 산출.

    Args:
        daily_df: 1d OHLCV DataFrame (마지막=오늘, -2=어제).
        current_close: 현재 LTF 마지막 봉 close.

    Returns:
        - current > PDH → UP
        - current < PDL → DOWN
        - 범위 안 → NONE
    """
    lv = compute_daily_levels(daily_df)
    if lv is None:
        return TrendDirection.NONE
    if current_close > lv.pdh:
        return TrendDirection.UP
    if current_close < lv.pdl:
        return TrendDirection.DOWN
    return TrendDirection.NONE


def _ts_of(df: pd.DataFrame, idx: int) -> int:
    """df.index[idx] 를 ms 정수로 변환."""
    if isinstance(df.index, pd.DatetimeIndex):
        return int(df.index[idx].value // 10**6)
    return int(df.index[idx])


__all__ = [
    "DailyLevels",
    "compute_daily_bias",
    "compute_daily_levels",
]
