"""Order Block — ICT 진입의 핵심 spot (FVG 와 양대 산맥).

ICT 정의:
    - **Bullish OB** = 강한 상승 직전 마지막 bearish 봉 (open > close).
      이후 displacement 봉 안에서 가격이 그 봉의 high 를 break + close 함.
      가격이 retest 하러 돌아오면 매수 자리.
    - **Bearish OB** = 강한 하락 직전 마지막 bullish 봉 (open < close).
      이후 displacement 봉 안에서 가격이 그 봉의 low 를 break + close 함.
      retest 시 매도 자리.

검출 로직:
    각 봉 i 에 대해:
      1. i 가 반대 방향 봉인지 확인 (bullish OB → i 는 bearish, vice versa)
      2. i+1 ~ i+displacement 봉 중 하나가 i 봉의 high (또는 low) 를
         돌파하며 close 했는지 검증
      3. 모두 만족 → OB 1개 생성

ATR 필터 (LuxAlgo 패턴):
    봉의 (high - low) 가 ATR(N) 의 2배 이상이면 변동성 과대 봉으로 보고
    OB 검출에서 high/low 를 뒤집어 평가한다 (좁은 body 가 OB 의 실제 origin).
    false positive 감소.

Mitigation:
    OB 생성 후 가격이 OB 봉 body 안으로 close 한 경우 → mitigated=True.
    Mitigated OB 는 setup 후보에서 제외하는 것이 일반적 (1회용 zone).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


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
        displacement_idx: displacement 봉 (high/low 돌파 close) 의 index.
        mitigated: 이후 가격이 OB body 안 close 했는지 (1회 retest 완료).
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
    atr[: period - 1] = np.nan
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return pd.Series(atr)


def detect_order_blocks(
    df: pd.DataFrame,
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
        displacement_bars: OB 봉 이후 몇 봉 안에 돌파 close 일어나야 OB 로 인정.
            기본 3. 너무 작으면 noise, 너무 크면 false positive.
        mark_mitigation: True 면 각 OB 후속 봉을 스캔해 mitigation 마킹.
        atr_filter: LuxAlgo 패턴 — high-low 가 ATR×multiplier 이상인 봉은
            변동성 과대로 분류해 parsed high/low 를 뒤집어 OB 검출에 사용.
            기본 True (false positive 감소).
        atr_period: ATR 윈도우 (기본 200).
        atr_multiplier: 변동성 임계 배수 (기본 2.0).

    Returns:
        OB list — 시간순 (OB 봉의 ts_ms 기준).

    Raises:
        ValueError: df 에 OHLC 컬럼 빠진 경우.
    """
    if len(df) < displacement_bars + 1:
        return []

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    if isinstance(df.index, pd.DatetimeIndex):
        ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
    else:
        ts_arr = df.index.to_numpy()

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    # ATR 필터 — 변동성 과대 봉(high-low ≥ ATR×mult)이면 parsed high/low 뒤집기
    # (LuxAlgo: highVolatilityBar ? low : high / highVolatilityBar ? high : low)
    if atr_filter:
        atr_series = _atr(highs, lows, closes, atr_period).to_numpy()
        # nan 은 0 으로 — 초기 봉은 필터 미적용
        atr_series = pd.Series(atr_series).fillna(0.0).to_numpy()
        high_vol = (highs - lows) >= (atr_multiplier * atr_series)
        # parsed* 는 LuxAlgo 의 parsedHigh / parsedLow 와 동일 로직
        parsed_highs = pd.Series(highs).where(~high_vol, pd.Series(lows)).to_numpy()
        parsed_lows = pd.Series(lows).where(~high_vol, pd.Series(highs)).to_numpy()
    else:
        parsed_highs = highs
        parsed_lows = lows

    obs: list[OrderBlock] = []

    for i in range(len(df) - displacement_bars):
        bar_open = opens[i]
        bar_close = closes[i]
        # 검출 비교는 parsed 값 사용 (ATR 필터 효과). 저장은 원본 high/low.
        cmp_high = parsed_highs[i]
        cmp_low = parsed_lows[i]
        raw_high = float(highs[i])
        raw_low = float(lows[i])

        is_bearish_bar = bar_close < bar_open
        is_bullish_bar = bar_close > bar_open
        if not (is_bearish_bar or is_bullish_bar):
            continue  # doji — skip

        # Bullish OB 후보: bearish 봉 + 이후 displacement 안에서 high 돌파 close
        if is_bearish_bar:
            for j in range(i + 1, min(i + 1 + displacement_bars, len(df))):
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
                    break  # 첫 displacement 만 인정 (중복 방지)

        # Bearish OB 후보: bullish 봉 + 이후 displacement 안에서 low 돌파 close
        elif is_bullish_bar:
            for j in range(i + 1, min(i + 1 + displacement_bars, len(df))):
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

    if mark_mitigation:
        _mark_mitigation(obs, closes)

    return obs


def _mark_mitigation(
    obs: list[OrderBlock],
    closes,  # numpy array  # noqa: ANN001
) -> None:
    """OB 생성 이후 가격이 body 안 close 한 경우 mitigated=True."""
    for ob in obs:
        for k in range(ob.displacement_idx + 1, len(closes)):
            close_k = closes[k]
            if ob.body_bottom <= close_k <= ob.body_top:
                ob.mitigated = True
                break


__all__ = ["OrderBlock", "OrderBlockType", "detect_order_blocks"]
