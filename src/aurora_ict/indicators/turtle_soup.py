"""Turtle Soup — ICT 정통 false breakout 진입 setup.

ICT 정의:
    - 원본 "Turtle" 시스템 (Richard Dennis, 1980s) = N-day breakout 진입 전략.
    - "Turtle Soup" (Linda Raschke) = 그 breakout 이 *실패* 했을 때 반대 진입.
    - ICT 는 이걸 차용 + 정밀화:
        1. 직전 N 봉 (정통: 4-day = 4 daily bars) 의 high/low 가 확정 swing
        2. 다음 봉이 그 swing 을 wick 으로만 갱신 (close 는 안쪽 복귀)
        3. 후속 봉 close 가 swing 안쪽 (sweep 인정) → 반대 방향 진입 setup

Sweep 과의 차이:
    - 일반 Sweep (`liquidity.py`): swing 1개 sweep, 어떤 swing 이든 OK
    - Turtle Soup: **N-day swing** (다일 / 다시간 큰 단위) 의 sweep — 더 빡빡, 더 강한 signal
    - Turtle Soup 진입은 직접적 reversal entry, sweep 는 setup 자료

크기 단위:
    - 정통: 4-day (daily TF). Aurora 는 자유롭게 N-bar (default 20봉) 적용.
    - 5m TF: 20봉 = 100분 안에 형성된 swing
    - 1h TF: 20봉 = 약 20시간 (1일 미만)
    - 1d TF: 20봉 = 정통 4-day 보다 길게 (보수적)

Entry / SL / TP:
    - Entry: sweep 봉 close 직후 또는 다음 봉 open
    - SL: sweep 봉의 wick 너머 (swing 갱신했던 그 wick)
    - TP: 직전 N-bar 범위의 반대편 swing (또는 더 가까운 EQH/EQL)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class TurtleSoupDirection(StrEnum):
    """Turtle Soup 진입 방향."""

    LONG = "long"      # 하방 sweep 후 long (4-bar low 깬 후 안쪽 복귀)
    SHORT = "short"    # 상방 sweep 후 short (4-bar high 깬 후 안쪽 복귀)


@dataclass(slots=True, frozen=True)
class TurtleSoupSetup:
    """Turtle Soup 진입 setup.

    Attributes:
        direction: long / short.
        sweep_idx: sweep 발생 봉 인덱스 (이 봉의 wick 이 N-bar swing 갱신).
        sweep_price: sweep 된 swing 가격 (4-bar high or low).
        wick_extreme: sweep 봉의 wick 가장 멀리 (long 이면 wick low, short 이면 wick high).
                      SL 위치 자료.
        opposite_target: TP 후보 — 직전 N-bar 의 반대편 swing 가격.
        lookback: 사용된 N-bar 값 (정통 4-day → Aurora 는 가변).
    """

    direction: TurtleSoupDirection
    sweep_idx: int
    sweep_price: float
    wick_extreme: float
    opposite_target: float
    lookback: int


def detect_turtle_soup_setups(
    df: pd.DataFrame,
    *,
    lookback: int = 20,
    require_close_inside: bool = True,
) -> list[TurtleSoupSetup]:
    """OHLCV 에서 Turtle Soup setup 검출.

    각 봉 ``i`` 에 대해 직전 ``lookback`` 봉 (i-lookback ~ i-1) 의 high/low 를
    측정. 봉 ``i`` 가:
        - wick high > 직전 N-bar high  AND  close < 직전 N-bar high   → short setup
        - wick low  < 직전 N-bar low   AND  close > 직전 N-bar low    → long setup

    Args:
        df: ``open``, ``high``, ``low``, ``close`` 컬럼 보유 (정렬: 시간순).
        lookback: swing 측정 봉 수. 정통은 4 (daily TF), 본 함수 default 20봉
            (LTF 적용 시 더 안정적 — 4봉은 LTF 에서 너무 잦음).
        require_close_inside: True 면 close 가 swing 안쪽이어야 sweep 인정 (정통).
            False 면 wick 갱신만으로 OK (더 느슨).

    Returns:
        ``TurtleSoupSetup`` 리스트 — 시간순.
    """
    if df.empty or len(df) <= lookback:
        return []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    setups: list[TurtleSoupSetup] = []
    for i in range(lookback, len(df)):
        window_high = float(highs[i - lookback:i].max())
        window_low = float(lows[i - lookback:i].min())

        bar_high = float(highs[i])
        bar_low = float(lows[i])
        bar_close = float(closes[i])

        # SHORT setup — wick 으로 N-bar high 갱신 + close 안쪽
        if bar_high > window_high:
            if not require_close_inside or bar_close < window_high:
                setups.append(TurtleSoupSetup(
                    direction=TurtleSoupDirection.SHORT,
                    sweep_idx=i,
                    sweep_price=window_high,
                    wick_extreme=bar_high,
                    opposite_target=window_low,
                    lookback=lookback,
                ))
                continue   # 같은 봉이 동시에 long sweep 되는 케이스 차단

        # LONG setup — wick 으로 N-bar low 갱신 + close 안쪽
        if bar_low < window_low:
            if not require_close_inside or bar_close > window_low:
                setups.append(TurtleSoupSetup(
                    direction=TurtleSoupDirection.LONG,
                    sweep_idx=i,
                    sweep_price=window_low,
                    wick_extreme=bar_low,
                    opposite_target=window_high,
                    lookback=lookback,
                ))

    return setups


__all__ = [
    "TurtleSoupDirection",
    "TurtleSoupSetup",
    "detect_turtle_soup_setups",
]
