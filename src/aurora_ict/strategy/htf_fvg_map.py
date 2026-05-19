"""HTF FVG 가중치 맵 — 변경 3 핵심 모듈.

여러 HTF (15m, 1h, 2h, 4h, 1d, 1w) 의 미체결 FVG 를 모아 가중치를 매긴다.

가중치 정책 (대형 TF 일수록 의미 ↑):
    5m=1, 15m=2, 1h=4, 2h=6, 4h=10, 1d=20, 1w=40

용도:
- 진입 직전: LTF setup 의 반대 방향 HTF FVG 가중치 합이 LTF setup 가중치를 넘으면
  방향 차단 또는 flip target 으로 사용 (override).
- 봉 close 시점: 가격이 flip target FVG zone 안으로 들어오면 flip 발동.

ccxt fetch 비용을 줄이려고 caller 측에서 TF 별 마지막 fetch ts 캐시를 둔다.
이 모듈 자체는 stateless — 입력으로 받은 DataFrame 만 사용한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from aurora_ict.indicators.fvg import FVGType, detect_fvgs, mark_filled_and_invalidated

# TF → 가중치 매핑. settings 에서 dict 박기 까다로워 상수로 둔다.
TF_WEIGHT: dict[str, int] = {
    "5m": 1,
    "15m": 2,
    "1h": 4,
    "2h": 6,
    "4h": 10,
    "1d": 20,
    "1w": 40,
}


@dataclass(slots=True)
class HtfFvgEntry:
    """HTF FVG 한 건 + 메타.

    Attributes:
        tf: 소속 TF ("4h" 등).
        weight: TF 가중치 (TF_WEIGHT 참조).
        type: bullish / bearish.
        high: zone 상단.
        low: zone 하단.
        ts_ms: FVG 중간봉 ts.
    """

    tf: str
    weight: int
    type: FVGType
    high: float
    low: float
    ts_ms: int
    # 변경 7: touch 누적. 가격이 "밖 → 안" 전환 시마다 +1.
    # 임계치 (settings.htf_fvg_max_touch_count) 도달 시 mitigation 후보에서 제외 — FVG 약화.
    touch_count: int = 0

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    def contains(self, price: float) -> bool:
        return self.low <= price <= self.high

    def key(self) -> tuple[str, str, int, float, float]:
        """동일 FVG 식별용 키 — 캐시 재빌드 후에도 touch_count 이전 박아둔 거 매핑."""
        return (self.tf, self.type.value, self.ts_ms, self.high, self.low)


async def build_htf_fvg_map(
    fetch_ohlcv_tf: Any,  # async callable(tf, limit) -> DataFrame
    tfs: tuple[str, ...],
    fvg_min_size_pct: float,
    limit: int = 200,
) -> list[HtfFvgEntry]:
    """각 TF OHLCV fetch → FVG 검출 → 미체결 FVG entry 리스트 반환.

    Args:
        fetch_ohlcv_tf: ``async (tf, limit) -> pd.DataFrame`` (bot instance 가 제공).
        tfs: 대상 TF tuple.
        fvg_min_size_pct: FVG 최소 % size (noise 필터).
        limit: TF 당 fetch 봉 수.

    Returns:
        미체결 (filled=False, invalidated=False) FVG 만 가중치 부여해 모은 리스트.
    """
    out: list[HtfFvgEntry] = []
    for tf in tfs:
        weight = TF_WEIGHT.get(tf, 1)
        try:
            df: pd.DataFrame = await fetch_ohlcv_tf(tf, limit)
        except Exception:
            # 네트워크 / TF 미지원 — 안전하게 그 TF 만 skip.
            continue
        if df is None or len(df) < 5:
            continue
        fvgs = detect_fvgs(df, min_size_pct=fvg_min_size_pct)
        mark_filled_and_invalidated(fvgs, df)
        for fv in fvgs:
            if fv.filled or fv.invalidated:
                continue
            out.append(
                HtfFvgEntry(
                    tf=tf,
                    weight=weight,
                    type=fv.type,
                    high=float(fv.high),
                    low=float(fv.low),
                    ts_ms=int(fv.ts_ms),
                ),
            )
    return out


def find_opposite_htf_fvg(
    htf_map: list[HtfFvgEntry],
    ltf_direction: str,  # "buy"/"sell" 또는 "long"/"short"
    current_price: float,
    threshold_weight: int,
    max_touch_count: int = 3,
) -> list[HtfFvgEntry]:
    """LTF setup 의 반대 방향 HTF FVG 중 합산 가중치가 threshold 초과인 것들 반환.

    조건:
    - LTF 가 buy → 반대 = bearish FVG, 위치는 current_price 보다 **위쪽** (LTF 진행 방향).
    - LTF 가 sell → 반대 = bullish FVG, 위치는 current_price 보다 **아래쪽**.

    정렬: 거리 (mid - current_price 절대값) 가까운 것부터.

    Args:
        htf_map: build_htf_fvg_map 결과.
        ltf_direction: LTF setup 의 방향. "buy"/"long" 또는 "sell"/"short".
        current_price: 현재 가격.
        threshold_weight: LTF setup 의 가중치 (이 합산 가중치를 넘어야 의미 있음).

    Returns:
        조건을 만족할 때만 가까운 순으로 정렬된 후보 리스트. 합산 가중치가 threshold
        이하면 빈 리스트 반환.
    """
    is_buy = ltf_direction.lower() in ("buy", "long")
    opposite_type = FVGType.BEARISH if is_buy else FVGType.BULLISH
    cands: list[HtfFvgEntry] = []
    for e in htf_map:
        if e.type is not opposite_type:
            continue
        # 변경 7: touch 누적이 임계치 도달한 FVG 는 약화로 보고 후보에서 제외.
        if e.touch_count >= max_touch_count:
            continue
        if is_buy and e.low <= current_price:
            # 가격보다 위쪽 zone 만 (LTF 가 올라가다가 만날 자리).
            continue
        if (not is_buy) and e.high >= current_price:
            continue
        cands.append(e)
    total_weight = sum(e.weight for e in cands)
    if total_weight <= threshold_weight:
        return []
    cands.sort(key=lambda e: abs(e.mid - current_price))
    return cands


__all__ = [
    "TF_WEIGHT",
    "HtfFvgEntry",
    "build_htf_fvg_map",
    "find_opposite_htf_fvg",
]
