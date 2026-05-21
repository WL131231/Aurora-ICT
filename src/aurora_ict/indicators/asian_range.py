"""Asian Range — ICT 의 일중 reference 가이드 라인.

ICT 정의:
    Asian session (NY 00:00 ~ 05:00 또는 19:00 전일 ~ 05:00 — 시점 모델마다 다름)
    의 high/low 가 그날 London/NY 세션의 reference 로 작용.

활용:
    - London/NY 의 첫 sweep target = Asian high 또는 Asian low
    - Asian range 위/아래 close → 그날 방향성 단서 (Power of 3 의 Distribution 방향)
    - 매매 진입 시 Asian range 안 = noise (Accumulation phase)

본 모듈은 봇 내 다른 timing 모듈 (``timing/killzone.py``, ``timing/power_of_3.py``) 과
**Asian session 시간 정의 통일** — Lumi Traders ICT Killzones 차트 기준 19:00 ~ 23:59
NY local. 이전엔 00:00-05:00 으로 박혀 #BUG-4 (모듈 간 정의 불일치) 발생 → 2026-05-21 통일.

session 이 자정을 가로지르지 않으므로 ``crosses_midnight`` 처리 불필요.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from zoneinfo import ZoneInfo

import pandas as pd

NY_TZ = ZoneInfo("America/New_York")

# Asian session 정의 (NY local time) — killzone.py 의 ASIAN 과 동일 (#BUG-4 해소).
_ASIAN_START = time(19, 0)
_ASIAN_END = time(23, 59, 59)


@dataclass(slots=True)
class AsianRange:
    """Asian session 의 high/low + 봉 ts.

    Attributes:
        high: Asian session 안의 최고 high.
        low: 최저 low.
        high_ts_ms: high 봉 ts.
        low_ts_ms: low 봉 ts.
        session_start_ms: Asian session 시작 봉 ts (정렬용).
        session_end_ms: Asian session 마지막 봉 ts.
    """

    high: float
    low: float
    high_ts_ms: int
    low_ts_ms: int
    session_start_ms: int
    session_end_ms: int


def _to_ny(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).astimezone(NY_TZ)


def compute_asian_range(df: pd.DataFrame) -> AsianRange | None:
    """df 의 마지막 Asian session (NY 00:00~05:00) high/low 추출.

    Args:
        df: OHLCV DataFrame — index DatetimeIndex (UTC) 또는 ms.

    Returns:
        ``AsianRange`` 또는 ``None`` (Asian 봉 없으면).
    """
    if df is None or len(df) == 0:
        return None

    if isinstance(df.index, pd.DatetimeIndex):
        ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
    else:
        ts_arr = df.index.to_numpy()

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    # NY date 별로 group — 가장 최근 Asian session 추출
    # 정렬: 봉을 ts 역순으로 훑고, Asian time 안의 봉만 모음.
    # 첫 만난 Asian 봉이 속한 NY date 의 봉들만 사용.
    target_date = None
    asian_indices: list[int] = []
    for i in range(len(df) - 1, -1, -1):
        ny_dt = _to_ny(int(ts_arr[i]))
        if not (_ASIAN_START <= ny_dt.time() < _ASIAN_END):
            if target_date is not None:
                break  # Asian session 끝났음 (이전 봉들은 다른 날 또는 different session)
            continue
        if target_date is None:
            target_date = ny_dt.date()
            asian_indices.append(i)
        elif ny_dt.date() == target_date:
            asian_indices.append(i)
        else:
            break

    if not asian_indices:
        return None

    asian_indices.sort()
    sub_highs = highs[asian_indices]
    sub_lows = lows[asian_indices]
    high_local = int(sub_highs.argmax())
    low_local = int(sub_lows.argmin())
    high_idx = asian_indices[high_local]
    low_idx = asian_indices[low_local]

    return AsianRange(
        high=float(highs[high_idx]),
        low=float(lows[low_idx]),
        high_ts_ms=int(ts_arr[high_idx]),
        low_ts_ms=int(ts_arr[low_idx]),
        session_start_ms=int(ts_arr[asian_indices[0]]),
        session_end_ms=int(ts_arr[asian_indices[-1]]),
    )


__all__ = ["AsianRange", "compute_asian_range"]
