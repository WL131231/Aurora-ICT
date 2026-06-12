"""리플레이 엔진 — 1분봉을 한 봉씩 흘려보내며 다중 TF OHLCV 집계.

실시간 봇과 동일한 시그널 로직을 시뮬에서도 재현하기 위함.

기존 trading_bot/core/replay_engine.py 차용 (AI 부분 제거).

담당: 팀원 C
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

# 분 단위 TF 매핑
TF_MINUTES = {
    "1m": 1, "3m": 3, "5m": 5, "15m": 15, "30m": 30,
    "1H": 60, "2H": 120, "4H": 240, "1D": 1440, "1W": 10_080,
}


# 월요일 정렬용 epoch — 1970-01-05 (월요일) 00:00.
# pandas ``.floor("7D")`` 는 1970-01-01(목) 기준이라 거래소 관행(월요일 주 시작)과 어긋남.
# 모든 TF 에 대해 이 epoch + tf_minutes 정수 나눗셈으로 통일하여 bucket open_time 산출.
_BUCKET_EPOCH = pd.Timestamp("1970-01-05")


@dataclass(slots=True)
class AggregatedBar:
    """집계 결과 한 봉.

    Attributes:
        open_time: 봉 시작 시각 (TF bucket 의 시작점).
        open: 봉 시작 가격.
        high: 봉 기간 최고가.
        low: 봉 기간 최저가.
        close: 봉 종료 가격 (마지막 1 분봉의 close).
        volume: 봉 기간 누적 거래량.
        close_ts: 봉 마감 시각 (= open_time + TF 길이). 라이브 환경에서 "봉이 닫혔는가"
            판정용. 백테스트에서는 look-ahead 검증 용도 (예: signal 산출 시점에 대해
            ``assert bar.close_ts <= signal_time`` 형태로 사용).
    """

    open_time: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    volume: float
    close_ts: pd.Timestamp


def _bucket_open_time(ts: pd.Timestamp, tf_minutes: int) -> pd.Timestamp:
    """주어진 시각이 속한 TF bucket 의 open_time 을 반환.

    ``_BUCKET_EPOCH`` (1970-01-05 월) 기준으로 ``tf_minutes`` 단위 floor 한 결과.
    1W bucket 이 월요일 00:00 에 정렬되도록 하기 위해 명시적 epoch 사용.
    하위 TF (1H/4H 등) 는 어차피 epoch 이 자정이라 동일하게 정렬됨.

    tz-aware ``ts`` 가 들어오면 동일 tz 의 epoch 와 비교 (해당 tz 의 자정 기준 정렬).
    거래소 데이터는 보통 UTC tz-naive 이므로 그대로 동작.

    Args:
        ts: 1 분봉의 open_time.
        tf_minutes: TF 길이(분). 예: ``60`` (1H), ``240`` (4H), ``10080`` (1W).

    Returns:
        ``ts`` 가 속한 bucket 의 open_time (Timestamp).
    """
    epoch = _BUCKET_EPOCH if ts.tz is None else pd.Timestamp("1970-01-05", tz=ts.tz)
    # 분 단위 정수 나눗셈 — 부동소수 오차 회피
    delta_minutes = int((ts - epoch).total_seconds() // 60)
    bucket_offset = (delta_minutes // tf_minutes) * tf_minutes
    return epoch + pd.Timedelta(minutes=bucket_offset)


def _new_bar(
    open_time: pd.Timestamp,
    minute_bar: pd.Series,
    tf_minutes: int,
) -> AggregatedBar:
    """1 분봉 한 개로 새 TF 봉을 시작.

    OHLCV 다섯 필드는 모두 1 분봉의 값을 그대로 사용 (이 시점에 봉 안에는
    이 1 분봉 하나만 들어있으므로). ``close_ts`` 는 ``open_time + tf_minutes``.

    Args:
        open_time: 새 TF 봉의 시작 시각 (``_bucket_open_time`` 으로 구한 값).
        minute_bar: open/high/low/close/volume 필드를 가진 pandas Series.
        tf_minutes: TF 길이(분).

    Returns:
        새 ``AggregatedBar`` 인스턴스.
    """
    return AggregatedBar(
        open_time=open_time,
        open=float(minute_bar["open"]),
        high=float(minute_bar["high"]),
        low=float(minute_bar["low"]),
        close=float(minute_bar["close"]),
        volume=float(minute_bar["volume"]),
        close_ts=open_time + pd.Timedelta(minutes=tf_minutes),
    )


def _update_in_place(bar: AggregatedBar, minute_bar: pd.Series) -> None:
    """진행 중인 TF 봉을 새 1 분봉으로 갱신 (in-place mutation).

    같은 bucket 안에서 여러 1 분봉을 누적할 때 사용.
    high 는 더 큰 값으로, low 는 더 작은 값으로 갱신, close 는 항상 최신 1 분봉의
    close 로 덮어쓰기, volume 은 누적 합산. ``open_time`` / ``open`` / ``close_ts`` 는
    봉이 시작될 때 확정된 값이라 변경하지 않음.

    Args:
        bar: 진행 중인 ``AggregatedBar`` (mutated in place).
        minute_bar: 새로 들어온 1 분봉.
    """
    high = float(minute_bar["high"])
    low = float(minute_bar["low"])
    if high > bar.high:
        bar.high = high
    if low < bar.low:
        bar.low = low
    bar.close = float(minute_bar["close"])
    bar.volume += float(minute_bar["volume"])


class MultiTfAggregator:
    """1 분봉 입력 → 사용자가 지정한 TF 들로 자동 집계.

    봉 마감 의미론: 다음 bucket 의 1 분봉이 도착하는 순간 직전 봉을 닫음
    (라이브 환경과 동일 — 봉이 닫힌 시점에는 미래 정보가 없음).

    데이터 갭은 채우지 않음. 갭이 들어간 봉은 부분봉(partial bar)으로 그대로
    emit 하고, 갭이 완전히 포함된 상위 TF bucket 의 봉은 emit 하지 않음
    (24/7 운영 시장 가정).
    """

    def __init__(self, timeframes: list[str], buffer_size: int = 3000) -> None:
        """집계기 초기화.

        Args:
            timeframes: 집계할 TF 리스트. ``TF_MINUTES`` 의 키만 허용
                (예: ``["5m", "1H", "4H"]``).
            buffer_size: TF 별 deque ``maxlen``. 기본 3000 — EMA 480 의 안정 warmup
                (3~5×period) 을 충분히 커버. 테스트 등에서 작은 값으로 override 가능.

        Raises:
            ValueError: ``timeframes`` 에 ``TF_MINUTES`` 에 없는 키가 있거나, 중복된
                값이 있을 때.
        """
        unknown = [tf for tf in timeframes if tf not in TF_MINUTES]
        if unknown:
            raise ValueError(
                f"지원하지 않는 timeframe: {unknown}. "
                f"허용 값: {list(TF_MINUTES.keys())}"
            )
        # 중복 timeframe 은 step() 내부에서 같은 1 분봉을 두 번 처리해 volume 이중 집계
        # 등 silent 버그를 유발하므로 fail-fast.
        if len(set(timeframes)) != len(timeframes):
            raise ValueError(f"timeframes 에 중복된 값이 있습니다: {timeframes}")

        self.timeframes = timeframes
        self.buffers: dict[str, deque[AggregatedBar]] = {
            tf: deque(maxlen=buffer_size) for tf in timeframes
        }
        # 현재 형성 중인 봉 (TF 별, 아직 buffers 에 들어가지 않은 미마감 봉)
        self._current: dict[str, AggregatedBar | None] = {tf: None for tf in timeframes}

    def step(self, minute_bar: pd.Series) -> dict[str, AggregatedBar | None]:
        """1 분봉 한 개 입력 → 닫힌 TF 별 새 봉 dict 반환.

        ``minute_bar.name`` 이 1 분봉의 open_time (Timestamp). pandas DataFrame 을
        ``iterrows`` 로 순회할 때 자연스럽게 들어오는 형태.

        각 TF 에 대해 다음 3 분기:

        1. ``_current[tf] is None`` (cold start) → 새 봉 시작, 결과는 None.
        2. 같은 bucket 안 → 진행 중 봉을 in-place 갱신, 결과는 None.
        3. 다른 bucket → 진행 중 봉을 ``buffers`` 에 push 하고 결과로 반환,
           새 봉 시작.

        Args:
            minute_bar: open/high/low/close/volume 필드 + ``name`` 이 open_time 인 Series.

        Returns:
            ``{tf: AggregatedBar | None}`` — ``None`` 이면 그 TF 에서 봉이 닫히지 않음.
        """
        ts = minute_bar.name
        closed: dict[str, AggregatedBar | None] = {tf: None for tf in self.timeframes}

        for tf in self.timeframes:
            tf_minutes = TF_MINUTES[tf]
            bucket_open = _bucket_open_time(ts, tf_minutes)
            current = self._current[tf]

            if current is None:
                # cold start — 첫 1 분봉, 새 봉 시작
                self._current[tf] = _new_bar(bucket_open, minute_bar, tf_minutes)
            elif current.open_time == bucket_open:
                # 같은 bucket — 진행 중 봉 갱신
                _update_in_place(current, minute_bar)
            else:
                # 다른 bucket — 직전 봉 마감 후 새 봉 시작
                self.buffers[tf].append(current)
                closed[tf] = current
                self._current[tf] = _new_bar(bucket_open, minute_bar, tf_minutes)

        return closed

    def get_df(self, timeframe: str) -> pd.DataFrame:
        """누적된 닫힌 봉들을 DataFrame 으로 변환.

        진행 중인 ``_current[timeframe]`` 봉은 의도적으로 제외 — 호출자가 이
        DataFrame 으로 지표를 계산해도 미마감 봉의 미완성 데이터가 섞이지 않음
        (look-ahead 방지).

        Args:
            timeframe: ``__init__`` 에서 등록한 TF 중 하나.

        Returns:
            인덱스가 ``open_time``, 컬럼이 ``[open, high, low, close, volume, close_ts]``.
            아직 닫힌 봉이 없으면 동일 컬럼 구조의 빈 DataFrame.

        Raises:
            ValueError: ``timeframe`` 이 등록되지 않았을 때.
        """
        if timeframe not in self.buffers:
            raise ValueError(
                f"등록되지 않은 timeframe: {timeframe!r}. "
                f"등록된 값: {self.timeframes}"
            )

        bars = self.buffers[timeframe]
        columns = ["open", "high", "low", "close", "volume", "close_ts"]
        if not bars:
            empty = pd.DataFrame(columns=columns)
            # 정상 케이스(set_index("open_time")) 와 index.name 일관성 유지.
            empty.index.name = "open_time"
            return empty

        df = pd.DataFrame(
            [
                {
                    "open_time": b.open_time,
                    "open": b.open,
                    "high": b.high,
                    "low": b.low,
                    "close": b.close,
                    "volume": b.volume,
                    "close_ts": b.close_ts,
                }
                for b in bars
            ]
        )
        return df.set_index("open_time")
