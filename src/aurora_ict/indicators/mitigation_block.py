"""Mitigation Block — ICT 정통 OB 의 2차 진입 zone.

ICT 정의:
    - **OB** = 1차 진입 자리 (BOS/CHoCH 직전 봉)
    - **Breaker Block** = OB 가 close 로 깨졌을 때 반전된 zone
    - **Mitigation Block** = OB 가 wick 으로 한 번 침범 (mitigated) 됐다가,
      가격이 잠시 떠난 후 다시 그 zone 으로 돌아온 자리 — **2차 진입 후보**

OB / Breaker / Mitigation 의 차이:
    | 항목 | 정의 | 진입 시점 |
    |---|---|---|
    | OB | 신선한 (touched X) zone | 첫 retest |
    | Breaker | close 로 깨진 OB → 반대 zone | 깨진 후 재진입 |
    | Mitigation | wick 침범 (mitigated) 후 떠났다 돌아온 OB | 2차 retest |

ICT 통상 운용:
    - OB 첫 retest 가 가장 강함 (fresh)
    - 2차 retest (= Mitigation Block) 는 약해진 zone — confluence 추가 시에만 진입
    - 3차 retest 부터는 잘 안 잡힘 (zone 약화)

Aurora-ICT 통합:
    - 기존 ``OrderBlock.mitigated`` 필드만으론 mitigation_idx (언제 침범됐는지)
      모름 → 본 모듈이 그 정보 + retest 시점 별도 추적
    - 진입 시 confluence_score 보강 자료로 사용
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from aurora_ict.indicators.order_block import OrderBlock, OrderBlockType


@dataclass(slots=True, frozen=True)
class MitigationBlock:
    """OB 가 mitigated 된 후 2차 retest 가능한 zone.

    Attributes:
        type: BULLISH / BEARISH (원본 OB 와 동일).
        origin_ob_idx: 원본 OB 봉 index.
        mitigation_idx: 가격이 처음 OB zone 을 wick 으로 침범한 봉 index.
        departure_idx: mitigation 후 가격이 OB zone 을 벗어난 봉 index
            (이 시점 이후 retest 가능).
        retest_idx: 가격이 다시 OB zone 안으로 돌아온 봉 index
            (= Mitigation Block 형성 시점). None 이면 아직 retest 안 됨.
        high / low: zone 상하단 (원본 OB 의 high / low 와 동일).

    Note:
        retest_idx 가 None 이면 "잠재 후보" — 아직 진입 자료로 못 씀.
        retest_idx 가 박힌 후에 setup 후보로 활용.
    """

    type: OrderBlockType
    origin_ob_idx: int
    mitigation_idx: int
    departure_idx: int
    retest_idx: int | None
    high: float
    low: float


def detect_mitigation_blocks(
    obs: list[OrderBlock],
    df: pd.DataFrame,
    *,
    departure_buffer_pct: float = 0.001,
) -> list[MitigationBlock]:
    """mitigated OB 중 2차 retest 후보 / 완성된 Mitigation Block 검출.

    각 mitigated OB 에 대해:
        1. ``mitigation_idx`` = OB zone 을 처음 wick 으로 침범한 봉
        2. ``departure_idx`` = mitigation 후 가격이 OB zone 을 벗어난 봉
           (close 가 zone + buffer 밖)
        3. ``retest_idx`` = departure 후 다시 OB zone 안으로 wick 진입한 봉

    Args:
        obs: ``detect_order_blocks`` 결과 (``mitigated`` 필드 박혀있어야).
        df: OHLCV — high / low / close 컬럼.
        departure_buffer_pct: departure 판정의 여유 (zone 경계에서 살짝 떨어진
            가격을 "벗어났다" 로 인정하지 않게). 0.001 = 0.1%.

    Returns:
        ``MitigationBlock`` 리스트 — origin_ob_idx 순.
    """
    if df.empty or not obs:
        return []

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)

    blocks: list[MitigationBlock] = []
    for ob in obs:
        if not ob.mitigated:
            continue

        zone_high = ob.high
        zone_low = ob.low
        zone_mid = (zone_high + zone_low) / 2.0
        buffer = zone_mid * departure_buffer_pct

        # 1) mitigation_idx 찾기 — OB 봉 다음부터 처음 wick 침범
        mitigation_idx: int | None = None
        for k in range(ob.idx + 1, n):
            if ob.type is OrderBlockType.BULLISH:
                # bullish OB: wick low 가 zone 안 (low >= zone_low) 인지
                if lows[k] <= zone_high and highs[k] >= zone_low:
                    mitigation_idx = k
                    break
            else:
                # bearish OB
                if highs[k] >= zone_low and lows[k] <= zone_high:
                    mitigation_idx = k
                    break
        if mitigation_idx is None:
            continue

        # 2) departure_idx — mitigation 후 가격이 zone 밖으로 close
        departure_idx: int | None = None
        for k in range(mitigation_idx + 1, n):
            close = float(closes[k])
            if ob.type is OrderBlockType.BULLISH:
                # bullish: zone 위쪽으로 close (zone_high + buffer 이상)
                if close > zone_high + buffer:
                    departure_idx = k
                    break
            else:
                # bearish: zone 아래쪽으로 close
                if close < zone_low - buffer:
                    departure_idx = k
                    break
        if departure_idx is None:
            # 아직 zone 안에 머무름 — Mitigation Block 후보 아님
            continue

        # 3) retest_idx — departure 후 다시 zone 진입 (wick)
        retest_idx: int | None = None
        for k in range(departure_idx + 1, n):
            if lows[k] <= zone_high and highs[k] >= zone_low:
                retest_idx = k
                break

        blocks.append(MitigationBlock(
            type=ob.type,
            origin_ob_idx=ob.idx,
            mitigation_idx=mitigation_idx,
            departure_idx=departure_idx,
            retest_idx=retest_idx,
            high=zone_high,
            low=zone_low,
        ))

    return blocks


def filter_retested(blocks: list[MitigationBlock]) -> list[MitigationBlock]:
    """retest 가 완료된 Mitigation Block 만 — 즉시 진입 후보."""
    return [b for b in blocks if b.retest_idx is not None]


__all__ = [
    "MitigationBlock",
    "detect_mitigation_blocks",
    "filter_retested",
]
