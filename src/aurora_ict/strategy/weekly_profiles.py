"""Weekly Profiles 12종 — ICT 정통 주간 패턴 분류.

[[ict-canon-models]] 의 Weekly Profile 12 종 자동 분류. 한 주의 일별 OHLC 입력
→ 가장 가까운 profile 반환.

12 종 (정통 Practical 출처):

    I.    Classic Tuesday Low of Week (Bullish)         — Tue 가 weekly low
    II.   Classic Tuesday High of Week (Bearish)        — Tue 가 weekly high
    III.  Wednesday Low of Week (Bullish)               — Wed 가 weekly low
    IV.   Wednesday High of Week (Bearish)              — Wed 가 weekly high
    V.    Consolidation Thursday Bullish Reversal        — Mon-Wed 횡보 + Thu sweep low 후 반전
    VI.   Consolidation Thursday Bearish Reversal        — 미러
    VII.  Consolidation Midweek Rally (Bullish)         — Mon-Wed 횡보 + 위로 확장
    VIII. Consolidation Midweek Decline (Bearish)       — 미러
    IX.   Seek & Destroy Bullish Friday                 — Fri 가 weekly extremes 둘 다, ⚠ AVOID
    X.    Seek & Destroy Bearish Friday                 — 미러, ⚠ AVOID
    XI.   Wednesday Weekly Bullish Reversal             — Wed 가 weekly low + 강한 반전 + Mon-Tue 횡보
    XII.  Wednesday Weekly Bearish Reversal             — 미러

분류 우선순위 (높은 → 낮은):
    1. Seek & Destroy (Friday range max)
    2. Wednesday Weekly Reversal (XI/XII) — 강한 reversal + Mon-Tue 좁은 range
    3. Consolidation Thursday Reversal (V/VI)
    4. Consolidation Midweek Rally/Decline (VII/VIII)
    5. Wednesday Low/High (III/IV)
    6. Tuesday Low/High (I/II)
    7. UNCLASSIFIED

사용:
    - 주 시작 시 (Mon-Tue 시점) 미분류 → setup 진입 자유
    - 주 중반 (Wed) classification 확정 → 그 주의 큰 방향 잡고 매매
    - bot 의 weekly_bias 자료 보강 (Daily Bias 와 보완)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class WeeklyProfileType(StrEnum):
    """12 weekly profile + UNCLASSIFIED."""

    CLASSIC_TUESDAY_LOW = "classic_tuesday_low"
    CLASSIC_TUESDAY_HIGH = "classic_tuesday_high"
    WEDNESDAY_LOW = "wednesday_low"
    WEDNESDAY_HIGH = "wednesday_high"
    CONSOLIDATION_THURSDAY_BULLISH = "consolidation_thursday_bullish"
    CONSOLIDATION_THURSDAY_BEARISH = "consolidation_thursday_bearish"
    CONSOLIDATION_MIDWEEK_RALLY = "consolidation_midweek_rally"
    CONSOLIDATION_MIDWEEK_DECLINE = "consolidation_midweek_decline"
    SEEK_AND_DESTROY_BULLISH = "seek_and_destroy_bullish"
    SEEK_AND_DESTROY_BEARISH = "seek_and_destroy_bearish"
    WEDNESDAY_WEEKLY_BULLISH_REVERSAL = "wednesday_weekly_bullish_reversal"
    WEDNESDAY_WEEKLY_BEARISH_REVERSAL = "wednesday_weekly_bearish_reversal"
    UNCLASSIFIED = "unclassified"


# 요일 인덱스 (Python weekday: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6)
MON, TUE, WED, THU, FRI = 0, 1, 2, 3, 4


@dataclass(slots=True, frozen=True)
class DailyBar:
    """주간 분류용 일별 봉.

    Attributes:
        weekday: 0=Mon, 1=Tue, ..., 4=Fri.
        open / high / low / close: OHLC.
    """

    weekday: int
    open: float
    high: float
    low: float
    close: float


@dataclass(slots=True, frozen=True)
class WeeklyProfile:
    """분류된 주간 profile.

    Attributes:
        type: 12종 + UNCLASSIFIED.
        weekly_high: 한 주 최고가.
        weekly_low: 한 주 최저가.
        weekly_high_day: 최고가 형성 요일 (0-4).
        weekly_low_day: 최저가 형성 요일.
        bias: "bullish" / "bearish" / "neutral" — 그 주의 큰 방향.
        avoid: True 면 매매 자제 권장 (Seek & Destroy).
    """

    type: WeeklyProfileType
    weekly_high: float
    weekly_low: float
    weekly_high_day: int
    weekly_low_day: int
    bias: str
    avoid: bool = False


def _is_consolidation(
    bars: list[DailyBar],
    *,
    days: list[int],
    max_range_pct: float,
) -> bool:
    """``days`` 봉들의 high-low 폭이 평균 close 대비 ``max_range_pct`` 이하인지."""
    subset = [b for b in bars if b.weekday in days]
    if not subset:
        return False
    consol_high = max(b.high for b in subset)
    consol_low = min(b.low for b in subset)
    avg_close = sum(b.close for b in subset) / len(subset)
    if avg_close <= 0:
        return False
    return (consol_high - consol_low) / avg_close <= max_range_pct


def _strong_reversal(bar: DailyBar, *, direction: str) -> bool:
    """봉이 강한 reversal — long wick + close 반대편.

    direction='bullish': low 찍고 close 가 봉 상반부.
    direction='bearish': high 찍고 close 가 봉 하반부.
    """
    bar_range = bar.high - bar.low
    if bar_range <= 0:
        return False
    body_mid = (bar.open + bar.close) / 2.0
    upper_third = bar.low + bar_range * 0.66
    lower_third = bar.low + bar_range * 0.34
    if direction == "bullish":
        return bar.close > body_mid and bar.close >= upper_third
    return bar.close < body_mid and bar.close <= lower_third


def classify_weekly_profile(
    bars: list[DailyBar],
    *,
    consolidation_max_pct: float = 0.02,
    midweek_expansion_pct: float = 0.03,
) -> WeeklyProfile:
    """일주일 OHLC 봉들 → Weekly Profile 분류.

    Args:
        bars: ``DailyBar`` 리스트 (Mon-Fri, 5개 권장. 최소 3개).
        consolidation_max_pct: Mon-Wed 횡보 판정 폭 (평균 close 대비).
            default 0.02 = 2%.
        midweek_expansion_pct: midweek 이후 확장 판정 폭. default 0.03 = 3%.

    Returns:
        ``WeeklyProfile`` — 분류 + bias + avoid.
    """
    if not bars:
        return WeeklyProfile(
            type=WeeklyProfileType.UNCLASSIFIED,
            weekly_high=0.0, weekly_low=0.0,
            weekly_high_day=-1, weekly_low_day=-1,
            bias="neutral",
        )

    weekly_high = max(b.high for b in bars)
    weekly_low = min(b.low for b in bars)
    weekly_high_day = next(b.weekday for b in bars if b.high == weekly_high)
    weekly_low_day = next(b.weekday for b in bars if b.low == weekly_low)

    by_day: dict[int, DailyBar] = {b.weekday: b for b in bars}
    fri = by_day.get(FRI)
    thu = by_day.get(THU)
    wed = by_day.get(WED)

    # ---------- 1. Seek & Destroy (Friday) ----------
    if fri is not None:
        fri_makes_high = fri.high == weekly_high
        fri_makes_low = fri.low == weekly_low
        if fri_makes_high and fri_makes_low:
            # Fri 가 weekly extremes 둘 다 — 양방향 큰 sweep
            bias = "bullish" if fri.close > fri.open else "bearish"
            ptype = (WeeklyProfileType.SEEK_AND_DESTROY_BULLISH
                     if bias == "bullish"
                     else WeeklyProfileType.SEEK_AND_DESTROY_BEARISH)
            return WeeklyProfile(
                type=ptype,
                weekly_high=weekly_high, weekly_low=weekly_low,
                weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
                bias=bias,
                avoid=True,
            )

    # ---------- 2. Wednesday Weekly Reversal (XI/XII) ----------
    if wed is not None:
        mon_tue_consol = _is_consolidation(
            bars, days=[MON, TUE], max_range_pct=consolidation_max_pct,
        )
        if mon_tue_consol:
            if wed.low == weekly_low and _strong_reversal(wed, direction="bullish"):
                return WeeklyProfile(
                    type=WeeklyProfileType.WEDNESDAY_WEEKLY_BULLISH_REVERSAL,
                    weekly_high=weekly_high, weekly_low=weekly_low,
                    weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
                    bias="bullish",
                )
            if wed.high == weekly_high and _strong_reversal(wed, direction="bearish"):
                return WeeklyProfile(
                    type=WeeklyProfileType.WEDNESDAY_WEEKLY_BEARISH_REVERSAL,
                    weekly_high=weekly_high, weekly_low=weekly_low,
                    weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
                    bias="bearish",
                )

    # ---------- 3. Consolidation Thursday Reversal (V/VI) ----------
    if thu is not None:
        mon_wed_consol = _is_consolidation(
            bars, days=[MON, TUE, WED], max_range_pct=consolidation_max_pct,
        )
        if mon_wed_consol:
            if thu.low == weekly_low and _strong_reversal(thu, direction="bullish"):
                return WeeklyProfile(
                    type=WeeklyProfileType.CONSOLIDATION_THURSDAY_BULLISH,
                    weekly_high=weekly_high, weekly_low=weekly_low,
                    weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
                    bias="bullish",
                )
            if thu.high == weekly_high and _strong_reversal(thu, direction="bearish"):
                return WeeklyProfile(
                    type=WeeklyProfileType.CONSOLIDATION_THURSDAY_BEARISH,
                    weekly_high=weekly_high, weekly_low=weekly_low,
                    weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
                    bias="bearish",
                )

    # ---------- 4. Consolidation Midweek Rally/Decline (VII/VIII) ----------
    if fri is not None or thu is not None:
        mon_wed_consol = _is_consolidation(
            bars, days=[MON, TUE, WED], max_range_pct=consolidation_max_pct,
        )
        if mon_wed_consol:
            # midweek (Mon-Wed) 평균 close 대비 Thu/Fri high or low 의 확장폭
            mw_subset = [b for b in bars if b.weekday in (MON, TUE, WED)]
            mw_avg = sum(b.close for b in mw_subset) / len(mw_subset)
            for late_bar in (fri, thu):
                if late_bar is None:
                    continue
                up_move = (late_bar.high - mw_avg) / mw_avg if mw_avg > 0 else 0
                down_move = (mw_avg - late_bar.low) / mw_avg if mw_avg > 0 else 0
                if up_move >= midweek_expansion_pct and up_move > down_move:
                    return WeeklyProfile(
                        type=WeeklyProfileType.CONSOLIDATION_MIDWEEK_RALLY,
                        weekly_high=weekly_high, weekly_low=weekly_low,
                        weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
                        bias="bullish",
                    )
                if down_move >= midweek_expansion_pct and down_move > up_move:
                    return WeeklyProfile(
                        type=WeeklyProfileType.CONSOLIDATION_MIDWEEK_DECLINE,
                        weekly_high=weekly_high, weekly_low=weekly_low,
                        weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
                        bias="bearish",
                    )

    # ---------- 5. Wednesday Low/High (III/IV) ----------
    if weekly_low_day == WED:
        return WeeklyProfile(
            type=WeeklyProfileType.WEDNESDAY_LOW,
            weekly_high=weekly_high, weekly_low=weekly_low,
            weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
            bias="bullish",
        )
    if weekly_high_day == WED:
        return WeeklyProfile(
            type=WeeklyProfileType.WEDNESDAY_HIGH,
            weekly_high=weekly_high, weekly_low=weekly_low,
            weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
            bias="bearish",
        )

    # ---------- 6. Classic Tuesday Low/High (I/II) ----------
    if weekly_low_day == TUE:
        return WeeklyProfile(
            type=WeeklyProfileType.CLASSIC_TUESDAY_LOW,
            weekly_high=weekly_high, weekly_low=weekly_low,
            weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
            bias="bullish",
        )
    if weekly_high_day == TUE:
        return WeeklyProfile(
            type=WeeklyProfileType.CLASSIC_TUESDAY_HIGH,
            weekly_high=weekly_high, weekly_low=weekly_low,
            weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
            bias="bearish",
        )

    # ---------- 7. UNCLASSIFIED ----------
    return WeeklyProfile(
        type=WeeklyProfileType.UNCLASSIFIED,
        weekly_high=weekly_high, weekly_low=weekly_low,
        weekly_high_day=weekly_high_day, weekly_low_day=weekly_low_day,
        bias="neutral",
    )


def daily_bars_from_df(df: pd.DataFrame) -> list[DailyBar]:
    """OHLCV DataFrame → DailyBar 리스트 (timestamp 기준 weekday 계산).

    Args:
        df: ``timestamp`` (UTC ms), ``open``, ``high``, ``low``, ``close`` 컬럼.

    Returns:
        ``DailyBar`` 리스트. timestamp 의 NY local weekday 기준.
    """
    if df.empty:
        return []
    # NY local weekday 사용 (정통 ICT 시간대 기준)
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    ny = ts.dt.tz_convert("America/New_York")
    weekdays = ny.dt.weekday.to_numpy()
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    bars = []
    for i in range(len(df)):
        wd = int(weekdays[i])
        if wd > FRI:
            continue   # 주말 skip (정통 기준)
        bars.append(DailyBar(
            weekday=wd,
            open=float(opens[i]),
            high=float(highs[i]),
            low=float(lows[i]),
            close=float(closes[i]),
        ))
    return bars


__all__ = [
    "DailyBar",
    "WeeklyProfile",
    "WeeklyProfileType",
    "classify_weekly_profile",
    "daily_bars_from_df",
]
