"""SMT Divergence — Smart Money Technique (상관 자산 간 다이버전스).

ICT 정의:
    두 상관 자산(BTC ↔ ETH, ES ↔ NQ 등)이 같은 시점에 swing 을 형성할 때,
    한 쪽은 새 swing high/low 를 박았는데 다른 쪽은 박지 못한 경우 — divergence.
    "기관 흐름의 누설" 로 해석되며, 반대 방향 setup 신호로 사용.

검출 규칙 (main 자산 기준):
    - **Bearish SMT** (swing high pair):
        main HH(↑) + corr LH(↓) → corr 가 못 따라옴 → main 도 곧 하락 가능
        또는 main LH + corr HH → main 이 못 따라옴 → 같은 의미
        → 둘 다 ``BEARISH`` 로 표시 (main 약세 신호)
    - **Bullish SMT** (swing low pair):
        main LL(↓) + corr HL(↑) → corr 가 더 강함 → main 도 곧 상승
        또는 main HL + corr LL → main 이 더 강함
        → 둘 다 ``BULLISH`` 로 표시 (main 강세 신호)

corr 가격 매칭:
    main swing 의 ``ts_ms`` 시점에 가장 가까운 corr 봉의 high(swing high 비교)
    또는 low(swing low 비교) 사용. 1시간 이상 차이 나면 skip (TF 불일치 보호).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from aurora_ict.indicators.swing_points import SwingPoint, SwingType


class SmtType(StrEnum):
    """SMT 방향."""

    BULLISH = "bullish"
    BEARISH = "bearish"


@dataclass(slots=True)
class SmtEvent:
    """SMT divergence 이벤트 1개.

    Attributes:
        ts_ms: divergence 가 확인된 swing 봉의 open time (ms).
        type: BULLISH / BEARISH.
        main_price: main 자산의 swing 가격.
        corr_price: 같은 시점 corr 자산의 high(bearish) 또는 low(bullish).
        swing_type: 비교한 swing 종류 (HIGH = bearish 후보, LOW = bullish 후보).
        prev_swing_ts_ms: 비교 대상 직전 swing 의 ts.
        idx: main DataFrame 의 swing 봉 index.
    """

    ts_ms: int
    type: SmtType
    main_price: float
    corr_price: float
    swing_type: SwingType
    prev_swing_ts_ms: int
    idx: int


_MAX_TS_DRIFT_MS = 3_600_000  # corr 봉과 매칭할 때 허용하는 최대 ts 차이 = 1H


def _corr_ts_array(corr_df: pd.DataFrame) -> list[int]:
    if isinstance(corr_df.index, pd.DatetimeIndex):
        return (corr_df.index.astype("int64") // 10**6).tolist()
    return [int(v) for v in corr_df.index.tolist()]


def detect_smt_divergence(
    main_swings: list[SwingPoint],
    corr_df: pd.DataFrame,
) -> list[SmtEvent]:
    """SMT divergence 검출 — main 자산 swing pair 박고 corr 자산 가격 비교.

    Args:
        main_swings: main 자산 swing point (``detect_swing_points`` 결과).
        corr_df: correlated 자산 OHLCV — index = ts_ms 또는 DatetimeIndex,
            columns 에 ``high`` / ``low`` 필요.

    Returns:
        SmtEvent list — 시간순.

    Raises:
        ValueError: corr_df 에 high/low 컬럼 빠진 경우.
    """
    if len(main_swings) < 2 or len(corr_df) == 0:
        return []

    missing = {"high", "low"} - set(corr_df.columns)
    if missing:
        raise ValueError(f"corr_df missing columns: {missing}")

    corr_ts = _corr_ts_array(corr_df)
    corr_highs = corr_df["high"].to_numpy()
    corr_lows = corr_df["low"].to_numpy()

    # 2026-08-09 #SMT-PERF: 선형 최근접 탐색이 O(스윙수 × 봉수) 였다. 5년 5분봉
    # (52만 봉) × 스윙 수천 쌍 = 수십억 비교로 백테가 4시간 넘게 걸렸다(py-spy 로
    # 이 함수 특정). OHLCV 인덱스는 시간 오름차순이므로 이진 탐색이면 19회면 된다.
    # 정렬이 깨진 입력(비정상 df)에서는 기존 선형 경로로 폴백해 결과를 보존한다.
    _ts = np.asarray(corr_ts, dtype=np.int64)
    _sorted = bool(_ts.size < 2 or np.all(_ts[1:] >= _ts[:-1]))

    def _nearest_idx(ts_ms: int) -> int:
        if not _sorted:
            return int(np.argmin(np.abs(_ts - ts_ms)))
        # searchsorted 는 삽입 위치 → 좌/우 이웃 둘만 비교하면 최근접이다.
        j = int(np.searchsorted(_ts, ts_ms))
        if j <= 0:
            return 0
        if j >= _ts.size:
            return _ts.size - 1
        return j if (_ts[j] - ts_ms) < (ts_ms - _ts[j - 1]) else j - 1

    def _corr_price_at(ts_ms: int, *, is_high: bool) -> float | None:
        """주어진 ts_ms 에 가장 가까운 corr 봉의 high/low. drift 초과 시 None."""
        nearest_i = _nearest_idx(ts_ms)
        if abs(int(_ts[nearest_i]) - ts_ms) > _MAX_TS_DRIFT_MS:
            return None
        return float(corr_highs[nearest_i] if is_high else corr_lows[nearest_i])

    # swing high / swing low 쌍 분리 (직전 같은 type swing 과 pair)
    sh_pairs: list[tuple[SwingPoint, SwingPoint]] = []
    sl_pairs: list[tuple[SwingPoint, SwingPoint]] = []
    prev_high: SwingPoint | None = None
    prev_low: SwingPoint | None = None
    for sw in main_swings:
        if sw.type is SwingType.HIGH:
            if prev_high is not None:
                sh_pairs.append((prev_high, sw))
            prev_high = sw
        else:
            if prev_low is not None:
                sl_pairs.append((prev_low, sw))
            prev_low = sw

    events: list[SmtEvent] = []

    # Bearish SMT — swing high pair 박고 main vs corr 방향 비교
    for prev, curr in sh_pairs:
        corr_prev = _corr_price_at(prev.ts_ms, is_high=True)
        corr_curr = _corr_price_at(curr.ts_ms, is_high=True)
        if corr_prev is None or corr_curr is None:
            continue
        main_higher = curr.price > prev.price
        corr_higher = corr_curr > corr_prev
        if main_higher != corr_higher:  # 한쪽만 HH = divergence
            events.append(SmtEvent(
                ts_ms=curr.ts_ms,
                type=SmtType.BEARISH,
                main_price=curr.price,
                corr_price=corr_curr,
                swing_type=SwingType.HIGH,
                prev_swing_ts_ms=prev.ts_ms,
                idx=curr.idx,
            ))

    # Bullish SMT — swing low pair
    for prev, curr in sl_pairs:
        corr_prev = _corr_price_at(prev.ts_ms, is_high=False)
        corr_curr = _corr_price_at(curr.ts_ms, is_high=False)
        if corr_prev is None or corr_curr is None:
            continue
        main_lower = curr.price < prev.price
        corr_lower = corr_curr < corr_prev
        if main_lower != corr_lower:
            events.append(SmtEvent(
                ts_ms=curr.ts_ms,
                type=SmtType.BULLISH,
                main_price=curr.price,
                corr_price=corr_curr,
                swing_type=SwingType.LOW,
                prev_swing_ts_ms=prev.ts_ms,
                idx=curr.idx,
            ))

    events.sort(key=lambda e: e.ts_ms)
    return events


__all__ = ["SmtEvent", "SmtType", "detect_smt_divergence"]
