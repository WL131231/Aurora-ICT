"""Rejection Block — ICT 정통 wick 기반 정밀 진입 zone.

ICT 정의:
    OB 는 봉 전체 (open ~ close) zone 인데, **Rejection Block** 은 그 봉의 wick
    부분만 zone 으로 봄. wick 이 깊게 들어왔다가 빠진 자리 = 큰손이 일부러 찍고
    빠진 정밀 자리.

    - **Bullish Rejection Block**: bearish 봉 (open > close) 의 *아래쪽 wick*
      (= low ~ body_low). 가격이 거기 도달했다가 즉시 빠진 자리 → bullish 진입 후보.
    - **Bearish Rejection Block**: bullish 봉 (open < close) 의 *위쪽 wick*
      (= body_high ~ high). 가격이 거기 도달했다가 즉시 빠진 자리 → bearish 진입 후보.

OB 와의 차이:
    | 항목 | OB | Rejection Block |
    |---|---|---|
    | zone | 봉 전체 (open ~ close) | wick 부분만 |
    | 검출 트리거 | BOS/CHoCH displacement | 봉 자체 (wick 비율 큼) |
    | 정밀도 | 넓은 zone, 진입 기회 ↑ | 좁은 zone, 정밀도 ↑ |

검출 조건:
    - wick 비율: 본 모듈은 ``min_wick_ratio=0.5`` (wick > body) 박힌 봉만 후보
    - displacement: 그 봉 후 가격이 zone 반대 방향으로 충분히 움직여야 (= wick
      rejection 의 확인) — ``min_move_ratio`` (default 1.5)

용도:
    - 정밀 진입 — SL 좁게, 손익비 ↑
    - OB confluence 보강 (같은 봉이 OB + Rejection Block 두 자료)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class RejectionBlockType(StrEnum):
    """방향."""

    BULLISH = "bullish"   # 아래쪽 wick — long 진입 후보
    BEARISH = "bearish"


@dataclass(slots=True, frozen=True)
class RejectionBlock:
    """Rejection Block 1개.

    Attributes:
        type: BULLISH / BEARISH.
        idx: 봉 인덱스.
        wick_high: wick zone 상단.
        wick_low: wick zone 하단.
        wick_ratio: wick 폭 / 봉 전체 폭 (1.0 = 봉 전체가 wick).
        confirm_move: 봉 이후 가격이 zone 반대 방향으로 얼만큼 움직였나 (zone 폭 대비).
    """

    type: RejectionBlockType
    idx: int
    wick_high: float
    wick_low: float
    wick_ratio: float
    confirm_move: float


def detect_rejection_blocks(
    df: pd.DataFrame,
    *,
    min_wick_ratio: float = 0.5,
    min_confirm_move: float = 1.5,
    confirm_lookforward: int = 5,
) -> list[RejectionBlock]:
    """OHLCV 에서 Rejection Block 검출.

    각 봉 ``i`` 에 대해:
        1. bearish 봉이면 아래쪽 wick (low ~ min(open, close)) 의 폭이
           봉 전체 폭의 ``min_wick_ratio`` 이상인지
        2. 이후 ``confirm_lookforward`` 봉 안에 가격이 wick zone 폭의
           ``min_confirm_move`` 배 이상 위로 움직였는지 (rejection 확인)
        3. 둘 다 만족 → Bullish Rejection Block

    Args:
        df: OHLC 컬럼.
        min_wick_ratio: wick / 봉 전체 폭 최소 비율. 0.5 = wick 이 절반 이상.
        min_confirm_move: rejection 확인 move 배수 (zone 폭 대비).
        confirm_lookforward: 확인용 후속 봉 수.

    Returns:
        ``RejectionBlock`` 리스트 — 인덱스순.
    """
    if df.empty or len(df) < 2:
        return []

    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)

    blocks: list[RejectionBlock] = []
    for i in range(n - 1):
        o = float(opens[i])
        h = float(highs[i])
        lo = float(lows[i])
        c = float(closes[i])

        bar_range = h - lo
        if bar_range <= 0:
            continue

        body_low = min(o, c)
        body_high = max(o, c)

        # 아래쪽 wick — bearish 봉 (c < o) 의 low ~ body_low
        if c < o:
            wick_width = body_low - lo
            wick_ratio = wick_width / bar_range
            if wick_ratio >= min_wick_ratio:
                # 후속 confirm move — 위로 움직였나
                end_idx = min(i + 1 + confirm_lookforward, n)
                future_high = float(highs[i + 1:end_idx].max()) if end_idx > i + 1 else h
                move = (future_high - body_low) / max(wick_width, 1e-12)
                if move >= min_confirm_move:
                    blocks.append(RejectionBlock(
                        type=RejectionBlockType.BULLISH,
                        idx=i,
                        wick_high=body_low,
                        wick_low=lo,
                        wick_ratio=wick_ratio,
                        confirm_move=move,
                    ))
                    continue

        # 위쪽 wick — bullish 봉 (c > o) 의 body_high ~ high
        if c > o:
            wick_width = h - body_high
            wick_ratio = wick_width / bar_range
            if wick_ratio >= min_wick_ratio:
                end_idx = min(i + 1 + confirm_lookforward, n)
                future_low = float(lows[i + 1:end_idx].min()) if end_idx > i + 1 else lo
                move = (body_high - future_low) / max(wick_width, 1e-12)
                if move >= min_confirm_move:
                    blocks.append(RejectionBlock(
                        type=RejectionBlockType.BEARISH,
                        idx=i,
                        wick_high=h,
                        wick_low=body_high,
                        wick_ratio=wick_ratio,
                        confirm_move=move,
                    ))

    return blocks


__all__ = [
    "RejectionBlock",
    "RejectionBlockType",
    "detect_rejection_blocks",
]
