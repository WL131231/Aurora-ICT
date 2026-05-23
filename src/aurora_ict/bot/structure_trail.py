"""Structure-based trailing stop — ICT 정통 trail.

진입 후 가격이 내 방향으로 가면서 새 swing low/high 가 형성될 때마다 SL 을
그 자리로 이동시킨다.

- LONG: 새 swing low 형성 → SL = swing low - buffer
- SHORT: 새 swing high 형성 → SL = swing high + buffer

ICT 철학:
- "구조(structure)가 깨지면 추세 끝났다" → 새 swing 가 추세 살아있다는 증거
- 그 swing 깨질 때까지 포지션 유지 → 멀리까지 trail
- LuxAlgo SMC 인디케이터가 쓰는 방식

이 모듈은 trail 계산만 담당. 실제 거래소 SL 수정은 호출 측 (BotIctInstance) 책임.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aurora_ict.indicators.swing_points import SwingType, detect_swing_points
from aurora_ict.strategy.silver_bullet import Direction


@dataclass(slots=True)
class TrailUpdate:
    """Trail SL 이동 제안.

    Attributes:
        new_stop_loss: 이동된 SL 가격.
        anchor_swing_idx: trail 기준이 된 swing 봉 index (디버그용).
        anchor_swing_price: 기준 swing 가격 (high/low).
    """

    new_stop_loss: float
    anchor_swing_idx: int
    anchor_swing_price: float


def compute_structure_trail(
    df: pd.DataFrame,
    direction: Direction,
    entry: float,
    current_stop_loss: float,
    *,
    swing_left: int = 1,
    swing_right: int = 1,
    buffer_ratio: float = 0.001,
) -> TrailUpdate | None:
    """Structure 기반 trail SL 계산.

    Args:
        df: LTF OHLCV DataFrame (진입 후 가격 움직임 반영).
        direction: 포지션 방향 (LONG / SHORT).
        entry: 진입 가격 (LONG → swing low > entry 만 유효 — break-even 이상).
        current_stop_loss: 현재 SL (이보다 유리한 자리로만 이동).
        swing_left/right: swing detector window (default 1=ICT 표준).
        buffer_ratio: SL 거리 buffer = swing 가격 × buffer_ratio.

    Returns:
        새 SL 이동 제안 (TrailUpdate). 갱신할 필요 없으면 None:
        - swing 없음
        - 가장 최근 swing 이 entry 보다 불리한 쪽 (LONG 인데 swing low < entry 등)
        - 새 SL 이 현재 SL 보다 불리한 방향 (역행 이동 금지)
    """
    if len(df) < 5:
        return None

    # 현재가 (마지막 종가). 트레일 SL 이 이 값을 넘어가면(SHORT: 현재가 이하 /
    # LONG: 현재가 이상) 거래소가 거부(10001)하고 내부 SL 만 잘못 갱신돼 조기청산을
    # 유발하므로, 아래 보호선 검사에 사용.
    current_price = float(df["close"].iloc[-1])

    swings = detect_swing_points(df, left=swing_left, right=swing_right)
    if not swings:
        return None

    if direction is Direction.LONG:
        target_type = SwingType.LOW
    else:
        target_type = SwingType.HIGH

    # 가장 최근 (시간순) 같은 방향 swing 찾기.
    anchor = None
    for s in reversed(swings):
        if s.type is target_type:
            anchor = s
            break
    if anchor is None:
        return None

    # break-even 이상 조건 — LONG 은 swing low 가 entry 이상, SHORT 은 그 반대.
    # buffer 적용 후 new_sl 도 entry 보다 유리한 쪽에 있어야 진짜 break-even.
    # (자기 SL 자체보다 불리하면 의미 없음)
    if direction is Direction.LONG:
        if anchor.price <= entry:
            return None
        buffer = anchor.price * buffer_ratio
        new_sl = anchor.price - buffer
        # buffer 적용 후 new_sl 이 entry 아래로 떨어지면 손실 SL → skip.
        if new_sl <= entry:
            return None
        # 역행 금지 — 새 SL 이 현재 SL 보다 낮으면 갱신 X.
        if new_sl <= current_stop_loss:
            return None
        # 현재가 보호선 — LONG SL 은 현재가 아래여야 유효. 가격이 swing 위로 못
        # 버티고 new_sl 이 현재가 이상이면 trail 금지 (거래소 거부 + 조기청산 방지).
        if new_sl >= current_price:
            return None
    else:
        if anchor.price >= entry:
            return None
        buffer = anchor.price * buffer_ratio
        new_sl = anchor.price + buffer
        # buffer 적용 후 new_sl 이 entry 위로 올라가면 손실 SL → skip.
        if new_sl >= entry:
            return None
        if new_sl >= current_stop_loss:
            return None
        # 현재가 보호선 — SHORT SL 은 현재가 위여야 유효. 가격이 swing 위로 되돌아와
        # new_sl 이 현재가 이하면 trail 금지 (거래소 거부 + 조기청산 방지).
        if new_sl <= current_price:
            return None

    return TrailUpdate(
        new_stop_loss=new_sl,
        anchor_swing_idx=anchor.idx,
        anchor_swing_price=anchor.price,
    )


__all__ = [
    "TrailUpdate",
    "compute_structure_trail",
]
