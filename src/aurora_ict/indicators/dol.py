"""DOL (Draw On Liquidity) — 다음 가격 target 으로서의 liquidity pool 식별.

ICT 핵심 concept: 가격은 항상 liquidity (= unswept swing high/low) 를 향해
끌린다. 그래서 "현재 가격에서 가장 가까운 unswept swing" 이 다음 sweep
target = DOL.

분류:
    - **Bullish DOL** = 현재가 위쪽의 가장 가까운 unswept swing high (BSL pool).
      가격이 매수세에 의해 그곳을 sweep 하러 갈 가능성.
    - **Bearish DOL** = 현재가 아래쪽의 가장 가까운 unswept swing low (SSL).
      매도세가 끌고 갈 target.

활용:
    - DOL = 현재 시점의 매수/매도 magnet
    - LONG 시 가까운 Bullish DOL = TP 자리 (silver_bullet 의 next_liquidity_target
      과 동일 아이디어, 단 setup 시점이 아닌 매 봉 갱신)
    - 두 DOL (위/아래) 의 거리 비율 = 다음 방향성 단서 (가까운 쪽이 magnet)
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aurora_ict.indicators.swing_points import SwingPoint, SwingType


@dataclass(slots=True)
class DOL:
    """Draw On Liquidity — 가격이 다음에 sweep 할 target liquidity.

    Attributes:
        type: "bullish" (위쪽 BSL — swing high) / "bearish" (아래쪽 SSL — swing low).
        price: target swing 가격.
        ts_ms: 그 swing 의 ts.
        swing_idx: swing index (df 안).
        distance: 현재가에서 DOL 까지 절대 가격 거리.
    """

    type: str
    price: float
    ts_ms: int
    swing_idx: int
    distance: float


def compute_dol(
    df: pd.DataFrame,
    swings: list[SwingPoint],
) -> list[DOL]:
    """현재가 (df 마지막 close) 기준 위/아래 가장 가까운 unswept swing 을 DOL 로.

    Args:
        df: OHLCV DataFrame. 마지막 close 를 현재가로 사용.
        swings: ``detect_swing_points`` + 선택적 sweep 마킹 (swept 갱신) 결과.

    Returns:
        list[DOL]: 최대 2건 (Bullish 1 + Bearish 1).
            unswept swing 이 없는 방향은 결과에서 생략.
    """
    if df is None or len(df) == 0 or not swings:
        return []

    current_price = float(df["close"].iloc[-1])

    # 위쪽 가까운 unswept swing high
    above: SwingPoint | None = None
    for s in swings:
        if s.type is not SwingType.HIGH or s.swept or s.price <= current_price:
            continue
        if above is None or s.price < above.price:
            above = s

    # 아래쪽 가까운 unswept swing low
    below: SwingPoint | None = None
    for s in swings:
        if s.type is not SwingType.LOW or s.swept or s.price >= current_price:
            continue
        if below is None or s.price > below.price:
            below = s

    dols: list[DOL] = []
    if above is not None:
        dols.append(DOL(
            type="bullish",
            price=above.price,
            ts_ms=above.ts_ms,
            swing_idx=above.idx,
            distance=above.price - current_price,
        ))
    if below is not None:
        dols.append(DOL(
            type="bearish",
            price=below.price,
            ts_ms=below.ts_ms,
            swing_idx=below.idx,
            distance=current_price - below.price,
        ))
    return dols


__all__ = ["DOL", "compute_dol"]
