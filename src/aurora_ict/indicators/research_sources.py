"""#SRC-ALL 2026-08-10: 미구현 ICT PD-array 검출기 모음 — 연구 전용.

파트너 지시("다른 ICT 매매법들도 다 테스트해봐. 전부"). 지금 진입 소스로 쓰는 건
FVG · turtle soup · implied FVG · rejection block 넷뿐이고, 정통 PD-array 9종 중
**Breaker · Inverse FVG · Vacuum Block** 과 조합 모델 **Unicorn** 이 아예 없다.

여기 있는 것들은 **연구 사본 전용**이다. 프로덕션에 넣기 전에 홀드아웃 4관문을
통과해야 한다.

정의 출처: memory/ict_canon_pdarrays.md (Bible + Practical 정리본).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aurora_ict.indicators.fvg import FVG, FVGType, detect_fvgs
from aurora_ict.indicators.order_block import OrderBlock, OrderBlockType, detect_order_blocks
from aurora_ict.indicators.swing_points import SwingPoint


@dataclass(slots=True)
class Zone:
    """진입 후보 구간 하나 — 소스가 달라도 형태는 같다.

    Attributes:
        ts_ms: 형성 시각 (구간이 확정된 봉).
        idx: 형성 봉 인덱스.
        high/low: 구간 상·하단.
        bullish: True 면 매수 구간(지지), False 면 매도 구간(저항).
        kind: 소스 이름.
    """

    ts_ms: int
    idx: int
    high: float
    low: float
    bullish: bool
    kind: str

    @property
    def mean(self) -> float:
        """구간 중앙 — 진입 지정가로 쓴다(FVG mean threshold 와 같은 규약)."""
        return (self.high + self.low) / 2.0


def _ts(df: pd.DataFrame, i: int) -> int:
    v = df.index[i]
    return int(v.value // 10**6) if hasattr(v, "value") else int(v)


# --------------------------------------------------------------- Inverse FVG
def detect_inverse_fvgs(
    df: pd.DataFrame,
    *,
    min_size_pct: float = 0.0006,
    max_age: int = 200,
) -> list[Zone]:
    """IFVG — FVG 가 지지/저항에 실패해 **반대 방향으로 뚫린** 구간.

    정통: "FVG 가 가격을 지지/저항 못 하고 깨질 때 형성. 모멘텀 첫 전환 신호.
    깨진 FVG 영역이 이제 반대 방향 supply/demand 로 작동."

    판정 — bullish FVG 는 종가가 **low 아래로** 마감하면 뒤집힌다(그 반대도 동일).
    꼬리만 스친 건 제외한다(종가 기준) — 정통의 "지지 실패"가 마감 기준이기 때문.

    Args:
        df: OHLCV.
        min_size_pct: FVG 최소 크기 (가격 대비).
        max_age: FVG 형성 후 이 봉 수 안에 뒤집혀야 인정(오래된 건 무의미).

    Returns:
        뒤집힌 시점 기준 Zone 목록. bullish 는 **뒤집힌 뒤의 역할**을 뜻한다.
    """
    fvgs = detect_fvgs(df, min_size_pct=min_size_pct)
    if not fvgs:
        return []
    c = df["close"].to_numpy(float)
    n = len(c)
    out: list[Zone] = []
    for f in fvgs:
        start = int(f.idx) + 1
        end = min(start + max_age, n)
        for j in range(start, end):
            if f.type is FVGType.BULLISH and c[j] < f.low:
                # 지지 실패 → 이제 저항(매도 구간)
                out.append(Zone(_ts(df, j), j, f.high, f.low, False, "ifvg"))
                break
            if f.type is FVGType.BEARISH and c[j] > f.high:
                out.append(Zone(_ts(df, j), j, f.high, f.low, True, "ifvg"))
                break
    out.sort(key=lambda z: z.idx)
    return out


# --------------------------------------------------------------- Breaker
def detect_breakers(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    *,
    max_age: int = 200,
) -> list[Zone]:
    """Breaker — 추세와 반대 방향 OB 가 깨진 것(failed OB).

    정통 Bullish Breaker 시퀀스: 유효한 bearish OB → 가격이 그 **high 위로 종가
    마감** → 그 OB 구간이 이제 지지로 작동. bearish 는 미러.

    Args:
        df: OHLCV.
        swings: 스윙 포인트 (OB 검출에 필요).
        max_age: OB 형성 후 이 봉 수 안에 깨져야 인정.

    Returns:
        깨진 시점 기준 Zone 목록.
    """
    obs = detect_order_blocks(df, swings=swings)
    if not obs:
        return []
    c = df["close"].to_numpy(float)
    n = len(c)
    out: list[Zone] = []
    for ob in obs:
        i0 = int(getattr(ob, "idx", 0))
        hi = float(getattr(ob, "high", 0.0))
        lo = float(getattr(ob, "low", 0.0))
        if hi <= 0 or lo <= 0 or hi <= lo:
            continue
        end = min(i0 + 1 + max_age, n)
        for j in range(i0 + 1, end):
            if ob.type is OrderBlockType.BEARISH and c[j] > hi:
                out.append(Zone(_ts(df, j), j, hi, lo, True, "breaker"))
                break
            if ob.type is OrderBlockType.BULLISH and c[j] < lo:
                out.append(Zone(_ts(df, j), j, hi, lo, False, "breaker"))
                break
    out.sort(key=lambda z: z.idx)
    return out


# --------------------------------------------------------------- Unicorn
def detect_unicorns(
    df: pd.DataFrame,
    swings: list[SwingPoint],
    *,
    min_size_pct: float = 0.0006,
    max_gap_bars: int = 20,
) -> list[Zone]:
    """Unicorn — Breaker 와 같은 방향 FVG 가 **겹치는** 구간.

    ICT 커뮤니티에서 고확률 조합으로 다뤄진다. 두 PD-array 가 같은 자리를 가리킬 때만
    진입하므로 빈도는 낮고 선별력이 높다는 주장. 겹침 구간만 Zone 으로 낸다.

    Args:
        max_gap_bars: breaker 와 FVG 형성 시점이 이 봉 수 안이어야 한 쌍으로 본다.
    """
    brks = detect_breakers(df, swings)
    if not brks:
        return []
    fvgs = detect_fvgs(df, min_size_pct=min_size_pct)
    out: list[Zone] = []
    for b in brks:
        for f in fvgs:
            if abs(int(f.idx) - b.idx) > max_gap_bars:
                continue
            same_dir = (f.type is FVGType.BULLISH) == b.bullish
            if not same_dir:
                continue
            hi = min(b.high, float(f.high))
            lo = max(b.low, float(f.low))
            if hi <= lo:
                continue                      # 겹치지 않음
            j = max(b.idx, int(f.idx))
            out.append(Zone(_ts(df, j), j, hi, lo, b.bullish, "unicorn"))
            break
    out.sort(key=lambda z: z.idx)
    return out


# --------------------------------------------------------------- Vacuum Block
def detect_vacuum_blocks(
    df: pd.DataFrame,
    *,
    min_gap_pct: float = 0.0015,
) -> list[Zone]:
    """Vacuum Block — 봉 사이 **가격 공백**(직전 종가와 다음 시가 갭).

    정통: "현재가 위에서 open → 채우려 내려옴 → 다시 상승"(bullish VB).
    크립토는 24시간 시장이라 갭이 드물지만, 급변 구간에서 실제로 발생한다.
    빈도가 0 에 가까우면 그것 자체가 결론이다(주식/선물 전용 개념).

    Args:
        min_gap_pct: 이 비율 이상 벌어져야 갭으로 인정.
    """
    o = df["open"].to_numpy(float)
    c = df["close"].to_numpy(float)
    out: list[Zone] = []
    for i in range(1, len(o)):
        prev_c, cur_o = c[i - 1], o[i]
        if prev_c <= 0:
            continue
        gap = (cur_o - prev_c) / prev_c
        if gap >= min_gap_pct:               # 위로 갭 → 아래 공백이 지지
            out.append(Zone(_ts(df, i), i, cur_o, prev_c, True, "vacuum"))
        elif gap <= -min_gap_pct:
            out.append(Zone(_ts(df, i), i, prev_c, cur_o, False, "vacuum"))
    return out


# --------------------------------------------------------------- BPR (기존 모듈 래핑)
def detect_bpr_zones(
    df: pd.DataFrame,
    *,
    min_size_pct: float = 0.0006,
) -> list[Zone]:
    """BPR — 반대 방향 FVG 두 개가 겹치는 구간. 기존 `indicators/bpr.py` 를 쓴다.

    방향은 **뒤에 형성된 FVG 쪽**을 따른다(나중 것이 현재 모멘텀).
    """
    from aurora_ict.indicators.bpr import detect_bpr

    fvgs = detect_fvgs(df, min_size_pct=min_size_pct)
    out: list[Zone] = []
    for b in detect_bpr(fvgs):
        later_bull = int(b.bullish_fvg_idx) > int(b.bearish_fvg_idx)
        j = int(b.formed_at_idx)
        if j >= len(df):
            continue
        out.append(Zone(_ts(df, j), j, float(b.high), float(b.low), later_bull, "bpr"))
    return out


ALL_DETECTORS = {
    "ifvg": lambda df, sw: detect_inverse_fvgs(df),
    "breaker": lambda df, sw: detect_breakers(df, sw),
    "unicorn": lambda df, sw: detect_unicorns(df, sw),
    "vacuum": lambda df, sw: detect_vacuum_blocks(df),
    "bpr": lambda df, sw: detect_bpr_zones(df),
}
