"""Order Block — ICT 진입의 핵심 spot (FVG 와 양대 산맥).

ICT 정의:
    - **Bullish OB** = 강한 상승 직전 마지막 bearish 봉 (open > close).
      이후 displacement 봉 안에서 가격이 그 봉의 high 를 break + close 함.
      가격이 retest 하러 돌아오면 매수 자리.
    - **Bearish OB** = 강한 하락 직전 마지막 bullish 봉 (open < close).
      이후 displacement 봉 안에서 가격이 그 봉의 low 를 break + close 함.
      retest 시 매도 자리.

알고리즘 모드 (``algorithm`` 파라미터):
    - ``"luxalgo"`` (기본): LuxAlgo SMC 표준 — swing pivot 기반 BOS 트리거.
      BOS/CHoCH 가 발생할 때마다, 돌파된 swing 봉과 BOS 봉 사이 구간에서
      ATR 필터링된 최대 range 의 반대 방향 봉 1개를 OB 로 채택.
      장점: false positive 적음, 차트 깔끔.
    - ``"legacy"``: 봉 단위 displacement 검사. 각 bearish/bullish 봉마다
      이후 N 봉 안에 high/low break close 가 있는지 본다. 더 촘촘하게 많이
      잡지만 noise 도 많다.

ATR 필터:
    - luxalgo: max-range bar 후보의 (high-low) 가 ATR×multiplier 초과면 변동성
      과대 봉으로 보고 OB 제외.
    - legacy: (high-low) 가 ATR×multiplier 이상이면 parsed high/low 를 뒤집어
      평가 (좁은 body 가 OB 의 origin).

Mitigation:
    LuxAlgo 'High/Low' mode — 후속 봉 wick 이 OB low/high 를 침범하면 즉시
    mitigated=True. Mitigated OB 는 setup 후보에서 제외하는 것이 일반적.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

import pandas as pd

from aurora_ict.indicators.structure import (
    StructureEvent,
    StructureType,
    detect_structure_events,
)
from aurora_ict.indicators.swing_points import SwingPoint, detect_swing_points


class OrderBlockType(StrEnum):
    """OB 방향."""

    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass(slots=True)
class OrderBlock:
    """Order Block 1개.

    Attributes:
        ts_ms: OB 봉 open time (ms).
        type: BULLISH / BEARISH.
        open / high / low / close: OB 봉 OHLC.
        idx: DataFrame 의 OB 봉 index.
        displacement_idx: BOS/displacement 봉 (high/low 돌파 close) 의 index.
        mitigated: 이후 가격이 OB 영역을 wick 으로라도 침범했는지.
    """

    ts_ms: int
    type: OrderBlockType
    open: float
    high: float
    low: float
    close: float
    idx: int
    displacement_idx: int
    mitigated: bool = False

    @property
    def body_top(self) -> float:
        return max(self.open, self.close)

    @property
    def body_bottom(self) -> float:
        return min(self.open, self.close)


def _atr(highs, lows, closes, period: int) -> pd.Series:  # noqa: ANN001
    """Wilder ATR (period). 짧은 df 에선 simple TR mean fallback."""
    import numpy as np
    n = len(highs)
    if n == 0:
        return pd.Series([], dtype=float)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
    # Wilder smoothing — 첫 period 평균 + 이후 (prev*(p-1)+cur)/p
    atr = np.empty(n)
    if n < period:
        # fallback: cumulative mean
        atr[0] = tr[0]
        for i in range(1, n):
            atr[i] = (atr[i - 1] * i + tr[i]) / (i + 1)
        return pd.Series(atr)
    atr[: period - 1] = float("nan")
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(atr)


def detect_order_blocks(
    df: pd.DataFrame,
    *,
    algorithm: Literal["luxalgo", "legacy"] = "luxalgo",
    swings: list[SwingPoint] | None = None,
    events: list[StructureEvent] | None = None,
    swing_length: int = 5,
    displacement_bars: int = 3,
    mark_mitigation: bool = True,
    atr_filter: bool = True,
    atr_period: int = 200,
    atr_multiplier: float = 2.0,
) -> list[OrderBlock]:
    """Order Block 검출.

    Args:
        df: OHLCV DataFrame — index = timestamp (ms 또는 datetime), columns =
            ``open / high / low / close``.
        algorithm: ``"luxalgo"`` (기본, BOS 트리거 + swing 구간 max range) 또는
            ``"legacy"`` (봉별 displacement 검사).
        swings: ``algorithm="luxalgo"`` 시 사용. None 이면 내부에서
            ``detect_swing_points(df, left=swing_length, right=swing_length)``
            로 계산.
        events: ``algorithm="luxalgo"`` 시 사용. None 이면 내부에서
            ``detect_structure_events(df, swings)`` 로 계산.
        swing_length: luxalgo swing pivot left/right window (기본 5).
        displacement_bars: legacy 모드 전용 — OB 봉 이후 몇 봉 안에 돌파
            close 일어나야 OB 로 인정.
        mark_mitigation: True 면 OB 후속 봉을 스캔해 wick mitigation 마킹.
        atr_filter: ATR 변동성 필터. luxalgo 는 OB 후보의 (high-low) 가
            ATR×mult 초과면 제외, legacy 는 parsed high/low 뒤집기.
        atr_period: ATR 윈도우 (기본 200).
        atr_multiplier: 변동성 임계 배수 (기본 2.0).

    Returns:
        OB list — 시간순 (OB 봉의 ts_ms 기준).

    Raises:
        ValueError: df 에 OHLC 컬럼 빠진 경우.
    """
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    if algorithm == "legacy":
        if len(df) < displacement_bars + 1:
            return []
    elif swings is None and len(df) < 2 * swing_length + 1:
        # luxalgo: swing 내부 계산 시 최소 봉 조건. 주입된 swings 가 있으면 패스.
        return []

    if isinstance(df.index, pd.DatetimeIndex):
        ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
    else:
        ts_arr = df.index.to_numpy()

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    atr_series = None
    if atr_filter:
        atr_arr = _atr(highs, lows, closes, atr_period).to_numpy()
        atr_series = pd.Series(atr_arr).fillna(0.0).to_numpy()

    if algorithm == "luxalgo":
        obs = _detect_obs_luxalgo(
            df=df,
            ts_arr=ts_arr,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            swings=swings,
            events=events,
            swing_length=swing_length,
            atr_filter=atr_filter,
            atr_series=atr_series,
            atr_multiplier=atr_multiplier,
        )
    else:
        obs = _detect_obs_legacy(
            ts_arr=ts_arr,
            opens=opens,
            highs=highs,
            lows=lows,
            closes=closes,
            n=len(df),
            displacement_bars=displacement_bars,
            atr_filter=atr_filter,
            atr_series=atr_series,
            atr_multiplier=atr_multiplier,
        )

    if mark_mitigation:
        _mark_mitigation(obs, highs, lows)

    return obs


def _detect_obs_luxalgo(  # noqa: PLR0913
    df: pd.DataFrame,
    ts_arr,  # noqa: ANN001
    opens,   # noqa: ANN001
    highs,   # noqa: ANN001
    lows,    # noqa: ANN001
    closes,  # noqa: ANN001
    swings: list[SwingPoint] | None,
    events: list[StructureEvent] | None,
    swing_length: int,
    atr_filter: bool,
    atr_series,  # noqa: ANN001
    atr_multiplier: float,
) -> list[OrderBlock]:
    """LuxAlgo SMC 표준: BOS/CHoCH 트리거 + swing 구간 ATR 필터 max range bar.

    각 BOS/CHoCH event 마다:
      - Bullish event (swing high break): [broken_swing_idx, ev.idx) 구간의
        bearish 봉 중 (high-low) 가 최대인 봉이 Bullish OB.
      - Bearish event (swing low break): 동일 구간의 bullish 봉 중 max range
        가 Bearish OB.
      - ATR 필터: 후보 봉의 (high-low) 가 ATR×mult 초과면 변동성 과대로 제외.
    """
    if swings is None:
        swings = detect_swing_points(df, left=swing_length, right=swing_length)
    if events is None:
        events = detect_structure_events(df, swings)

    if not events:
        return []

    obs: list[OrderBlock] = []
    for ev in events:
        lo_idx = ev.broken_swing_idx
        hi_idx = ev.idx
        if hi_idx <= lo_idx:
            continue

        is_bull_event = ev.type in (
            StructureType.BOS_BULLISH,
            StructureType.CHOCH_BULLISH,
        )

        best_idx = -1
        best_range = -1.0
        for k in range(lo_idx, hi_idx):
            # 방향 필터: bullish event → bearish 봉 후보 / bearish event → bullish 봉
            if is_bull_event:
                if closes[k] >= opens[k]:
                    continue
            else:  # bearish event
                if closes[k] <= opens[k]:
                    continue

            rng = float(highs[k] - lows[k])
            if rng <= 0:
                continue

            # ATR 필터: 변동성 과대 봉 제외
            if (
                atr_filter
                and atr_series is not None
                and atr_series[k] > 0
                and rng > atr_multiplier * atr_series[k]
            ):
                continue

            if rng > best_range:
                best_range = rng
                best_idx = k

        if best_idx < 0:
            continue

        # OB zone 은 봉 절반 (LuxAlgo OB Detector 표준).
        # Bullish OB → 봉 아래 절반 (hl2 ~ low) = demand zone
        # Bearish OB → 봉 위 절반 (high ~ hl2) = supply zone
        bar_high = float(highs[best_idx])
        bar_low = float(lows[best_idx])
        hl2 = (bar_high + bar_low) / 2.0
        if is_bull_event:
            ob_top = hl2
            ob_bottom = bar_low
        else:
            ob_top = bar_high
            ob_bottom = hl2

        obs.append(OrderBlock(
            ts_ms=int(ts_arr[best_idx]),
            type=OrderBlockType.BULLISH if is_bull_event else OrderBlockType.BEARISH,
            open=float(opens[best_idx]),
            high=ob_top,
            low=ob_bottom,
            close=float(closes[best_idx]),
            idx=best_idx,
            displacement_idx=hi_idx,
        ))

    return obs


def _detect_obs_legacy(  # noqa: PLR0913
    ts_arr,   # noqa: ANN001
    opens,    # noqa: ANN001
    highs,    # noqa: ANN001
    lows,     # noqa: ANN001
    closes,   # noqa: ANN001
    n: int,
    displacement_bars: int,
    atr_filter: bool,
    atr_series,  # noqa: ANN001
    atr_multiplier: float,
) -> list[OrderBlock]:
    """Legacy: 봉별 displacement 검사 (구버전 알고리즘, 회귀 호환용).

    각 bearish/bullish 봉 i 에 대해 i+1 ~ i+displacement_bars 안에서
    high/low 돌파 close 발생하면 OB 채택. ATR 필터는 parsed high/low 뒤집기.
    """
    if atr_filter and atr_series is not None:
        high_vol = (highs - lows) >= (atr_multiplier * atr_series)
        parsed_highs = pd.Series(highs).where(~high_vol, pd.Series(lows)).to_numpy()
        parsed_lows = pd.Series(lows).where(~high_vol, pd.Series(highs)).to_numpy()
    else:
        parsed_highs = highs
        parsed_lows = lows

    obs: list[OrderBlock] = []
    for i in range(n - displacement_bars):
        bar_open = opens[i]
        bar_close = closes[i]
        cmp_high = parsed_highs[i]
        cmp_low = parsed_lows[i]
        raw_high = float(highs[i])
        raw_low = float(lows[i])

        is_bearish_bar = bar_close < bar_open
        is_bullish_bar = bar_close > bar_open
        if not (is_bearish_bar or is_bullish_bar):
            continue

        if is_bearish_bar:
            for j in range(i + 1, min(i + 1 + displacement_bars, n)):
                if closes[j] > cmp_high:
                    obs.append(OrderBlock(
                        ts_ms=int(ts_arr[i]),
                        type=OrderBlockType.BULLISH,
                        open=float(bar_open),
                        high=raw_high,
                        low=raw_low,
                        close=float(bar_close),
                        idx=i,
                        displacement_idx=j,
                    ))
                    break
        elif is_bullish_bar:
            for j in range(i + 1, min(i + 1 + displacement_bars, n)):
                if closes[j] < cmp_low:
                    obs.append(OrderBlock(
                        ts_ms=int(ts_arr[i]),
                        type=OrderBlockType.BEARISH,
                        open=float(bar_open),
                        high=raw_high,
                        low=raw_low,
                        close=float(bar_close),
                        idx=i,
                        displacement_idx=j,
                    ))
                    break

    return obs


def _mark_mitigation(
    obs: list[OrderBlock],
    highs,  # numpy array  # noqa: ANN001
    lows,   # numpy array  # noqa: ANN001
) -> None:
    """OB 생성 이후 가격이 OB 영역을 wick 으로라도 침범하면 mitigated=True (LuxAlgo 방식).

    - Bullish OB: wick low 가 OB low 미만 → mitigated
    - Bearish OB: wick high 가 OB high 초과 → mitigated

    LuxAlgo 의 'High/Low' mitigation mode 와 동일.
    """
    for ob in obs:
        for k in range(ob.displacement_idx + 1, len(highs)):
            if ob.type is OrderBlockType.BULLISH:
                if lows[k] < ob.low:
                    ob.mitigated = True
                    break
            else:  # BEARISH
                if highs[k] > ob.high:
                    ob.mitigated = True
                    break


@dataclass(slots=True)
class BreakerBlock:
    """Breaker Block — OB 가 close 로 깨졌을 때의 반전 zone.

    ICT 정통: OB 가 단순 wick 침범이 아닌 close 로 영역을 이탈하면 그 OB 는
    Breaker 로 전환. 방향 반전 — Bullish OB → Bearish Breaker, vice versa.
    가격이 retest 하러 돌아오면 새 반전 진입 zone.

    IFVG 와 유사한 개념이지만 OB 기반이라 보통 더 큰 zone + 더 강한 신호.

    Attributes:
        ts_ms: break 봉 (close 가 OB 너머로 간 봉) open time.
        type: 반전된 새 방향 (origin OB 의 반대).
        high: 원래 OB 의 high.
        low: 원래 OB 의 low.
        broken_idx: break 봉 idx.
        origin_ob_idx: 원래 OB 봉 idx.
    """

    ts_ms: int
    type: OrderBlockType
    high: float
    low: float
    broken_idx: int
    origin_ob_idx: int

    @property
    def mean(self) -> float:
        return (self.high + self.low) / 2.0


def detect_breaker_blocks(
    obs: list[OrderBlock],
    df: pd.DataFrame,
) -> list[BreakerBlock]:
    """OB 가 close 로 깨진 경우 BreakerBlock 으로 변환.

    ICT 정통 strict break — wick 침범은 mitigation 일 뿐이고, close 가 OB
    영역 너머로 가야 진짜 Breaker.

    Args:
        obs: ``detect_order_blocks`` 결과.
        df: 동일 OHLCV DataFrame.

    Returns:
        BreakerBlock list — 시간순 (break 봉 ts 기준).
    """
    if not obs or len(df) < 2:
        return []

    if isinstance(df.index, pd.DatetimeIndex):
        ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
    else:
        ts_arr = df.index.to_numpy()

    closes = df["close"].to_numpy()
    bbs: list[BreakerBlock] = []

    for ob in obs:
        broken_idx = -1
        for k in range(ob.displacement_idx + 1, len(closes)):
            if ob.type is OrderBlockType.BULLISH:
                if closes[k] < ob.low:
                    broken_idx = k
                    break
            elif closes[k] > ob.high:
                broken_idx = k
                break
        if broken_idx < 0:
            continue
        inverted = (
            OrderBlockType.BEARISH
            if ob.type is OrderBlockType.BULLISH
            else OrderBlockType.BULLISH
        )
        bbs.append(BreakerBlock(
            ts_ms=int(ts_arr[broken_idx]),
            type=inverted,
            high=ob.high,
            low=ob.low,
            broken_idx=broken_idx,
            origin_ob_idx=ob.idx,
        ))

    bbs.sort(key=lambda b: b.ts_ms)
    return bbs


__all__ = [
    "BreakerBlock",
    "OrderBlock",
    "OrderBlockType",
    "detect_breaker_blocks",
    "detect_order_blocks",
]
