"""#AUTONOMOUS 2026-07-20: MMBM/MMSM 백테 프로토타입 — 2번째 진입모델 엣지 검증.

파트너: 정통 ICT 모델 중 우리 없는 것 4개 중 MMBM 먼저. MMBM(마켓메이커 매수모델)
핵심 = HTF discount PD-array 에서 smart money reversal(MSS) 후 매수. MMSM = premium
미러. 우리 엔진은 Silver Bullet 단일셋업 — MMBM 은 다른 자리 잡는 별개 모델이라
빈도↑·분산 여지. 정통 완벽구현 아닌 핵심 시퀀스 근사로 엣지 스크리닝.

시퀀스(MMBM 롱):
  1. 가격이 딜링레인지 discount(최근 N봉 range 하위)에 위치 (= 싸게 사는 자리)
  2. 5m bullish CHoCH(MSS up) 발생 (smart money reversal 확인)
  3. 진입 = CHoCH close. SL = 직전 swing low 아래. TP = 2R.
MMSM 숏 = premium 미러. 비용 왕복 0.06% 반영.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402

from aurora_ict.indicators.fvg import FVGType, detect_fvgs  # noqa: E402
from aurora_ict.indicators.structure import StructureType, detect_structure_events  # noqa: E402
from aurora_ict.indicators.swing_points import SwingType, detect_swing_points  # noqa: E402

FEE = 0.0006
RANGE_N = 288      # 딜링레인지 lookback (24h @ 5m)
DISC_THR = 0.5     # discount/premium 경계
RETRACE_TTL = 12   # MSS 후 FVG 되돌림 대기 봉 수
HTF_LB = 20        # 1h 바이어스 lookback (20h)


def backtest(sym: str, rr: float = 2.0, use_htf: bool = True):
    df = _resample(_load_full(sym))
    c = df["close"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    n = len(c)
    # 1h 바이어스 (5m→1h close, 20h 변화 부호)
    c1h = df["close"].resample("1h").last().ffill()
    bias1h = np.sign(c1h - c1h.shift(HTF_LB)).reindex(df.index, method="ffill").fillna(0).to_numpy()
    # 굵은 스윙(3,3) → 구조적 CHoCH (노이즈↓)
    swings = detect_swing_points(df, left=3, right=3)
    events = detect_structure_events(df, swings)
    fvgs = detect_fvgs(df, min_size_pct=0.001)
    bull_fvg = sorted([f for f in fvgs if f.type is FVGType.BULLISH], key=lambda f: f.idx)
    bear_fvg = sorted([f for f in fvgs if f.type is FVGType.BEARISH], key=lambda f: f.idx)
    swing_lows = [(s.idx, s.price) for s in swings if s.type is SwingType.LOW]
    swing_highs = [(s.idx, s.price) for s in swings if s.type is SwingType.HIGH]

    def sl_below(idx, price):
        cands = [p for (si, p) in swing_lows if si <= idx and p < price]
        return max(cands) if cands else None

    def sl_above(idx, price):
        cands = [p for (si, p) in swing_highs if si <= idx and p > price]
        return min(cands) if cands else None

    def recent_fvg(fvg_list, idx):
        # idx 이전(±5) 가장 가까운 FVG (변위로 만든 것)
        cands = [f for f in fvg_list if idx - 10 <= f.idx <= idx + 2]
        return cands[-1] if cands else None

    trades = []
    for ev in events:
        i = ev.idx
        if i < RANGE_N or i >= n - 1:
            continue
        rhi = h[i - RANGE_N:i].max()
        rlo = lo[i - RANGE_N:i].min()
        if rhi <= rlo:
            continue
        pos = (c[i] - rlo) / (rhi - rlo)
        if ev.type is StructureType.CHOCH_BULLISH and pos < DISC_THR:
            if use_htf and bias1h[i] < 0:  # 1h 하락바이어스면 롱 자제
                continue
            fvg = recent_fvg(bull_fvg, i)
            if fvg is None:
                continue
            entry = fvg.mean_threshold  # 되돌림 진입가 (FVG CE)
            sl = sl_below(i, entry) or fvg.low
            if entry - sl <= 0:
                continue
            tp = entry + rr * (entry - sl)
            direction = 1
        elif ev.type is StructureType.CHOCH_BEARISH and pos > (1 - DISC_THR):
            if use_htf and bias1h[i] > 0:
                continue
            fvg = recent_fvg(bear_fvg, i)
            if fvg is None:
                continue
            entry = fvg.mean_threshold
            sl = sl_above(i, entry) or fvg.high
            if sl - entry <= 0:
                continue
            tp = entry - rr * (sl - entry)
            direction = -1
        else:
            continue
        # 되돌림 진입 대기: RETRACE_TTL 안에 entry 가격 터치해야 체결
        fill = None
        for j in range(i + 1, min(i + 1 + RETRACE_TTL, n)):
            if lo[j] <= entry <= h[j]:
                fill = j
                break
        if fill is None:
            continue
        # 체결 후 SL/TP 시뮬 (최대 288봉)
        outcome = 0.0
        for j in range(fill, min(fill + 289, n)):
            if direction == 1:
                if lo[j] <= sl:
                    outcome = (sl - entry) / entry; break
                if h[j] >= tp:
                    outcome = (tp - entry) / entry; break
            else:
                if h[j] >= sl:
                    outcome = (entry - sl) / entry; break
                if lo[j] <= tp:
                    outcome = (entry - tp) / entry; break
        trades.append((df.index[fill].value, outcome - FEE, direction))
    return df, trades


def main() -> int:
    pairs = sys.argv[1:] or ["BTCUSDT"]
    for sym in pairs:
        df, trades = backtest(sym)
        if not trades:
            print(f"{sym}: 거래 0")
            continue
        nets = np.array([t[1] for t in trades]) * 100
        span = max((df.index[-1] - df.index[0]).days, 1)
        w = (nets > 0).sum()
        wins = nets[nets > 0]
        los = nets[nets < 0]
        rr = (wins.mean() / abs(los.mean())) if len(wins) and len(los) else 0
        print(f"{sym:<9} n={len(nets):4d} net={nets.sum():+7.1f}% 승률={100*w/len(nets):3.0f}%"
              f" RR={rr:.2f} 빈도={len(nets)/span:.3f}/일")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
