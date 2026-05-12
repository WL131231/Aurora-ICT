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


def detect_order_blocks(
    df: pd.DataFrame,
    displacement_bars: int = 3,
    mark_mitigation: bool = True,
) -> list[OrderBlock]:
    """Order Block 검출.

    Args:
        df: OHLCV DataFrame — index = timestamp (ms 또는 datetime), columns =
            ``open / high / low / close``.
        displacement_bars: OB 봉 이후 몇 봉 안에 돌파 close 일어나야 OB 로 인정.
            기본 3. 너무 작으면 noise, 너무 크면 false positive.
        mark_mitigation: True 면 각 OB 후속 봉을 스캔해 mitigation 마킹.

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

    obs: list[OrderBlock] = []

    for i in range(len(df) - displacement_bars):
        bar_open = opens[i]
        bar_close = closes[i]
        bar_high = highs[i]
        bar_low = lows[i]

        is_bearish_bar = bar_close < bar_open
        is_bullish_bar = bar_close > bar_open
        if not (is_bearish_bar or is_bullish_bar):
            continue  # doji — skip

        # Bullish OB 후보: bearish 봉 + 이후 displacement 안에서 high 돌파 close
        if is_bearish_bar:
            for j in range(i + 1, min(i + 1 + displacement_bars, len(df))):
                if closes[j] > bar_high:
                    obs.append(OrderBlock(
                        ts_ms=int(ts_arr[i]),
                        type=OrderBlockType.BULLISH,
                        open=float(bar_open),
                        high=float(bar_high),
                        low=float(bar_low),
                        close=float(bar_close),
                        idx=i,
                        displacement_idx=j,
                    ))
                    break  # 첫 displacement 만 인정 (중복 방지)

        # Bearish OB 후보: bullish 봉 + 이후 displacement 안에서 low 돌파 close
        elif is_bullish_bar:
            for j in range(i + 1, min(i + 1 + displacement_bars, len(df))):
                if closes[j] < bar_low:
                    obs.append(OrderBlock(
                        ts_ms=int(ts_arr[i]),
                        type=OrderBlockType.BEARISH,
                        open=float(bar_open),
                        high=float(bar_high),
                        low=float(bar_low),
                        close=float(bar_close),
                        idx=i,
                        displacement_idx=j,
                    ))
                    break

    if mark_mitigation:
        _mark_mitigation(obs, opens, closes)

    return obs


def _mark_mitigation(
    obs: list[OrderBlock],
    opens,  # numpy array  # noqa: ANN001
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
