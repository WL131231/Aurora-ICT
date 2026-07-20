"""MMBM/MMSM (Market Maker Buy/Sell Model) 셋업 감지 — Origo 2번째 진입 모델.

담당: 지영민 (FST#7 2026-07-20). 정통 ICT 마켓메이커 모델의 핵심 = HTF 추세정합
방향으로 딜링레인지의 discount(싼 자리)/premium(비싼 자리)에서 구조전환(CHoCH,
= smart money reversal) 후 반전 진입. Silver Bullet(FVG 되돌림, 단일셋업)과 다른
자리를 잡아 빈도·분산 보강.

검증(백테 5년 7페어, maker 지정가 진입 + 고정 2R):
    - discount/premium + HTF정합이 엣지 필수조건 (둘 중 하나 빠지면 적자).
    - maker 체결(FVG 지정가) 가정 시 5/6년 흑자(+205~236%), 단 횡보장(2025) 약세.
    - Silver Bullet 과 dedup 0% (완전히 다른 거래).
    - ⚠️ taker 체결이면 비용에 엣지 먹힘 → 반드시 지정가(maker) 진입 전제.

라이브 사용: 매 step 마다 최신 봉 기준 detect. 갓 형성된 CHoCH 가 조건 충족 시
Silver Bullet 과 동일한 SilverBulletSetup(source=MMBM) 반환 → 봇이 지정가 대기 진입.
"""
from __future__ import annotations

import pandas as pd

from aurora_ict.indicators.fvg import FVGType, detect_fvgs
from aurora_ict.indicators.structure import StructureType, detect_structure_events
from aurora_ict.indicators.swing_points import SwingType, detect_swing_points
from aurora_ict.strategy.silver_bullet import (
    Direction,
    SetupSource,
    SilverBulletSetup,
)

_RANGE_N = 288       # 딜링레인지 lookback (24h @ 5m)
_SL_LB = 24          # 구조 SL lookback (최근 저/고점)
_FVG_WIN = 10        # CHoCH 전후 FVG 탐색 범위
_FRESH = 2           # CHoCH 가 최근 이 봉 안에 형성돼야 (라이브 신선도)
_SWING_LR = 3        # 굵은 스윙 (노이즈 CHoCH 억제)


def detect_mmbm_setup(
    df: pd.DataFrame,
    htf_bias_sign: float,
    min_rr: float = 2.0,
    fvg_min_size_pct: float = 0.001,
) -> SilverBulletSetup | None:
    """최신 봉 기준 MMBM/MMSM 셋업 1개 감지 (없으면 None).

    조건 (buy=MMBM):
        1) 갓(≤_FRESH봉) 형성된 bullish CHoCH(MSS up)
        2) 가격이 딜링레인지 discount(하위 50%)
        3) HTF 추세 정합 (htf_bias_sign >= 0)
        4) CHoCH 전후 bullish FVG 존재 → 지정가 진입(maker) 대상
    SL = min(최근 저점, FVG 하단). TP = 다음 미스윕 swing high(최소 min_rr R).
    sell=MMSM 은 premium 미러.

    Args:
        df: 5m OHLCV (index=DatetimeIndex, columns=open/high/low/close/volume).
        htf_bias_sign: 상위TF(1h) 추세 부호 (+상승/-하락/0중립). 봇이 계산해 전달.
        min_rr: 최소 손익비 (TP 가 이 R 미만이면 min_rr R 로 보정).
        fvg_min_size_pct: FVG 최소 크기 필터.

    Returns:
        ``SilverBulletSetup``(source=MMBM) 또는 조건 미충족 시 ``None``.
    """
    n = len(df)
    if n < _RANGE_N + 5:
        return None
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    last = n - 1

    swings = detect_swing_points(df, left=_SWING_LR, right=_SWING_LR)
    events = detect_structure_events(df, swings)
    if not events:
        return None
    ev = events[-1]
    # 신선도: 갓 형성된 CHoCH 만 (라이브 = 이번 봉 근처)
    if ev.idx < last - _FRESH:
        return None
    i = ev.idx
    if i < _RANGE_N:
        return None

    rhi = h[i - _RANGE_N:i].max()
    rlo = lo[i - _RANGE_N:i].min()
    if rhi <= rlo:
        return None
    pos = (c[i] - rlo) / (rhi - rlo)

    fvgs = detect_fvgs(df, min_size_pct=fvg_min_size_pct)
    swing_highs = sorted((s.idx, s.price) for s in swings if s.type is SwingType.HIGH)
    swing_lows = sorted((s.idx, s.price) for s in swings if s.type is SwingType.LOW)

    def recent_fvg(ftype: FVGType):
        cands = [f for f in fvgs if f.type is ftype and i - _FVG_WIN <= f.idx <= i]
        return cands[-1] if cands else None

    if ev.type is StructureType.CHOCH_BULLISH and pos < 0.5 and htf_bias_sign >= 0:
        fvg = recent_fvg(FVGType.BULLISH)
        if fvg is None:
            return None
        entry = fvg.mean_threshold
        sl = min(float(lo[max(0, i - _SL_LB):i + 1].min()), fvg.low)
        if entry - sl <= 0:
            return None
        risk = entry - sl
        liq = [p for (si, p) in swing_highs if si <= i and p > entry]
        tp = max(min(liq) if liq else entry + min_rr * risk, entry + min_rr * risk)
        direction = Direction.LONG
    elif ev.type is StructureType.CHOCH_BEARISH and pos > 0.5 and htf_bias_sign <= 0:
        fvg = recent_fvg(FVGType.BEARISH)
        if fvg is None:
            return None
        entry = fvg.mean_threshold
        sl = max(float(h[max(0, i - _SL_LB):i + 1].max()), fvg.high)
        if sl - entry <= 0:
            return None
        risk = sl - entry
        liq = [p for (si, p) in swing_lows if si <= i and p < entry]
        tp = min(max(liq) if liq else entry - min_rr * risk, entry - min_rr * risk)
        direction = Direction.SHORT
    else:
        return None

    rr = abs(tp - entry) / abs(entry - sl)
    ts_ms = int(df.index[i].value // 10**6)
    return SilverBulletSetup(
        ts_ms=ts_ms, direction=direction, window="mmbm",
        entry=float(entry), stop_loss=float(sl), take_profit=float(tp),
        risk_reward=float(rr), fvg=fvg, source=SetupSource.MMBM,
    )
