"""CISD (Change in State of Delivery) — 가격 전달 방향 전환 신호.

ICT 정의 (Practical / ICT Essentials):
    "가격 전달(delivery) 방향의 변화" — MSS(구조 전환)의 1캔들 micro 버전.
    swing 을 깨기 전에 더 빨리 잡히는 전환 신호.

    - **Bullish CISD**: 직전 연속 하락(bearish) 캔들들의 *첫(가장 이른) 캔들 시초가
      (open)* 위로 현재 캔들이 close → 매수세 전환 시작.
    - **Bearish CISD**: 직전 연속 상승(bullish) 캔들들의 첫 캔들 시초가 아래로 close.

    정통은 "bearish candle 의 opening price 위로 close" (1캔들) 로도 정의하나,
    연속 시퀀스의 시초가 라인을 기준으로 보는 게 노이즈에 강하다 — 마지막
    delivery leg 전체의 시작점을 회복해야 진짜 전환으로 인정.

담당: 지영민 (#CISD 2026-06-06 — 누락 보강 2/4)
"""

from __future__ import annotations

from enum import StrEnum

import pandas as pd


class CisdType(StrEnum):
    """CISD 방향."""

    BULLISH = "bullish"
    BEARISH = "bearish"


def detect_cisd(df: pd.DataFrame, *, max_lookback: int = 10) -> CisdType | None:
    """가장 최근(마지막) 봉 기준 CISD 발생 방향 반환.

    직전 봉의 방향으로 연속 같은 색 시퀀스를 뒤로 추적해 그 시퀀스의 첫 캔들
    시초가(open)를 CISD level 로 삼고, 현재 봉 close 가 그 level 을 반대로
    돌파했는지 본다.

    Args:
        df: OHLCV DataFrame (columns: open/close 필수). 마지막 행이 '현재' 봉.
        max_lookback: 연속 시퀀스를 뒤로 추적할 최대 봉 수 (과도한 과거 무시).

    Returns:
        BULLISH / BEARISH, 또는 CISD 미발생 시 None.
    """
    if df is None or len(df) < 2:
        return None
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    last = len(df) - 1
    prev = last - 1
    cur_close = float(closes[last])

    prev_bearish = closes[prev] < opens[prev]
    prev_bullish = closes[prev] > opens[prev]

    if prev_bearish:
        # 직전 연속 하락 시퀀스의 첫(가장 이른) 캔들 시초가 = bullish CISD level.
        level = float(opens[prev])
        j = prev - 1
        cnt = 1
        while j >= 0 and cnt < max_lookback and closes[j] < opens[j]:
            level = float(opens[j])
            j -= 1
            cnt += 1
        if cur_close > level:
            return CisdType.BULLISH
    elif prev_bullish:
        # 직전 연속 상승 시퀀스의 첫 캔들 시초가 = bearish CISD level.
        level = float(opens[prev])
        j = prev - 1
        cnt = 1
        while j >= 0 and cnt < max_lookback and closes[j] > opens[j]:
            level = float(opens[j])
            j -= 1
            cnt += 1
        if cur_close < level:
            return CisdType.BEARISH
    return None


__all__ = ["CisdType", "detect_cisd"]
