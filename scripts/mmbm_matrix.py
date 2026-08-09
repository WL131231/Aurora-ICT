"""#AUTONOMOUS 2026-07-24: MMBM 정통 전체구현 vs 라이브 근사 — 요소분해 매트릭스.

파트너: "전체 mmbm 구현해서 백테 한 번 돌려봐". 배경: 라이브 MMBM(Origo 2.2 MMBM)
청산 24건 전패(-43.97, 횡보 롱편향) — 백테 경고했던 횡보 약세 실현. 7/20 완전구현은
taker 0.12% 비용으로 기각됐으나 라이브에서 maker 체결이 실증됨 → maker 비용으로
정통 요소를 분해 재평가한다.

라이브(V0) = CHoCH(fresh)+discount/premium+1h bias+FVG 지정가, SL=min(24봉저점,fvg),
TP=max(다음 unswept BSL, 2R) — 유동성TP 는 이미 라이브에 있음.
정통 대비 라이브에 없는 3요소를 켜며 비교:
  V1 = V0 + 스윕요구(진입 전 24봉 내 SSL sweep — 유동성 엔지니어링 단계 확인)
  V2 = V0 + SMT 필터(BTC/ETH corr 다이버전스 동반 시만)
  V3 = V0 + 트레일(2R/1.5R)+BE@1R 청산(고정 TP 대신)
  V4 = V0 + 스윕 + 트레일/BE (풀 정통 근사)
비용: maker 진입 + taker 청산 왕복 0.08% (라이브 실증 시나리오).
구간: 전체 / ~2023 / 2024 / 2025 / 2026 / 최근90일 — 횡보 구간 성적이 핵심.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402

from aurora_ict.indicators.fvg import FVGType, detect_fvgs  # noqa: E402
from aurora_ict.indicators.liquidity import SweepType, detect_liquidity_sweeps  # noqa: E402
from aurora_ict.indicators.structure import StructureType, detect_structure_events  # noqa: E402
from aurora_ict.indicators.swing_points import SwingType, detect_swing_points  # noqa: E402

RTCOST = 0.0008     # maker 진입 0.02% + taker 청산 0.055% ≈ 왕복 0.08% (슬립 포함 근사)
RANGE_N = 288
FRESH = 2           # 라이브 신선도 (CHoCH 후 2봉 내)
SL_LB = 24
FVG_WIN = 10
RETRACE_TTL = 12
HTF_LB = 20
SWEEP_LB = 24
SMT_LB = 12
TRAIL_TRIG, TRAIL_DIST, BE_AT = 2.0, 1.5, 1.0
CORR = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT"}


def collect_setups(sym: str):
    """CHoCH 후보 전수 수집 — 각 후보에 컨텍스트 플래그(sweep/smt) 부착."""
    df = _resample(_load_full(sym))
    c = df["close"].to_numpy(); h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    n = len(c)
    c1h = df["close"].resample("1h").last().ffill()
    bias1h = np.sign(c1h - c1h.shift(HTF_LB)).reindex(df.index, method="ffill").fillna(0).to_numpy()
    swings = detect_swing_points(df, left=3, right=3)
    events = detect_structure_events(df, swings)
    fvgs = detect_fvgs(df, min_size_pct=0.001)
    bull_fvg = sorted([f for f in fvgs if f.type is FVGType.BULLISH], key=lambda f: f.idx)
    bear_fvg = sorted([f for f in fvgs if f.type is FVGType.BEARISH], key=lambda f: f.idx)
    swing_highs = sorted((s.idx, s.price) for s in swings if s.type is SwingType.HIGH)
    swing_lows = sorted((s.idx, s.price) for s in swings if s.type is SwingType.LOW)
    sweeps = detect_liquidity_sweeps(df, swings)
    ssl_idx = sorted(s.idx for s in sweeps if s.type is SweepType.BULLISH)
    bsl_idx = sorted(s.idx for s in sweeps if s.type is SweepType.BEARISH)
    # SMT — 전수 감지는 O(스윙^2)로 실용 불가 → 셋업 지점당 O(1) 검사로 대체.
    # 정의 충실: 직전 두 스윙 저점에서 본자산 LL(낮아짐) vs corr HL(안 낮아짐) = bullish SMT.
    corr = CORR.get(sym)
    clo = chi = None
    if corr:
        try:
            cdf = _resample(_load_full(corr)).reindex(df.index, method="ffill")
            clo = cdf["low"].to_numpy(); chi = cdf["high"].to_numpy()
        except Exception as e:
            print(f"  (corr 로드 실패 {corr}: {e})", flush=True)

    def had(lst, i, lb):
        import bisect
        j = bisect.bisect_right(lst, i)
        return j > 0 and lst[j - 1] >= i - lb

    import bisect
    sl_idx = [si for (si, _) in swing_lows]
    sh_idx = [si for (si, _) in swing_highs]

    def smt_at(i, bull):
        # CHoCH i 직전 두 스윙(저점/고점)로 본자산 vs corr 다이버전스 판정.
        if clo is None:
            return False
        if bull:
            j = bisect.bisect_left(sl_idx, i)
            if j < 2:
                return False
            i1, p1 = swing_lows[j - 2]; i2, p2 = swing_lows[j - 1]
            return p2 < p1 and clo[i2] > clo[i1]  # 본 LL, corr HL
        j = bisect.bisect_left(sh_idx, i)
        if j < 2:
            return False
        i1, p1 = swing_highs[j - 2]; i2, p2 = swing_highs[j - 1]
        return p2 > p1 and chi[i2] < chi[i1]      # 본 HH, corr LH

    def recent_fvg(fl, i):
        cands = [f for f in fl if i - FVG_WIN <= f.idx <= i]
        return cands[-1] if cands else None

    setups = []
    for ev in events:
        i = ev.idx
        if i < RANGE_N or i >= n - 1:
            continue
        rhi = h[i - RANGE_N:i].max(); rlo = lo[i - RANGE_N:i].min()
        if rhi <= rlo:
            continue
        pos = (c[i] - rlo) / (rhi - rlo)
        if ev.type is StructureType.CHOCH_BULLISH and pos < 0.5 and bias1h[i] >= 0:
            fvg = recent_fvg(bull_fvg, i)
            if fvg is None:
                continue
            entry = fvg.mean_threshold
            sl = min(float(lo[max(0, i - SL_LB):i + 1].min()), fvg.low)
            if entry - sl <= 0:
                continue
            risk = entry - sl
            liq = [p for (si, p) in swing_highs if si <= i and p > entry]
            tp = max(min(liq) if liq else entry + 2 * risk, entry + 2 * risk)
            _sw = [p for (si, p) in swing_lows if si <= i and p < c[i]]
            setups.append(dict(i=i, d=1, entry=entry, sl=sl, tp=tp,
                               centry=float(c[i]), swsl=(max(_sw) if _sw else fvg.low),
                               sweep=had(ssl_idx, i, SWEEP_LB), smt=smt_at(i, True)))
        elif ev.type is StructureType.CHOCH_BEARISH and pos > 0.5 and bias1h[i] <= 0:
            fvg = recent_fvg(bear_fvg, i)
            if fvg is None:
                continue
            entry = fvg.mean_threshold
            sl = max(float(h[max(0, i - SL_LB):i + 1].max()), fvg.high)
            if sl - entry <= 0:
                continue
            risk = sl - entry
            liq = [p for (si, p) in swing_lows if si <= i and p < entry]
            tp = min(max(liq) if liq else entry - 2 * risk, entry - 2 * risk)
            _sw = [p for (si, p) in swing_highs if si <= i and p > c[i]]
            setups.append(dict(i=i, d=-1, entry=entry, sl=sl, tp=tp,
                               centry=float(c[i]), swsl=(min(_sw) if _sw else fvg.high),
                               sweep=had(bsl_idx, i, SWEEP_LB), smt=smt_at(i, False)))
    return df, h, lo, n, setups


def sim_fixed(h, lo, n, fill, entry, sl, tp, d):
    for j in range(fill, min(fill + 289, n)):
        if d == 1:
            if lo[j] <= sl:
                return (sl - entry) / entry
            if h[j] >= tp:
                return (tp - entry) / entry
        else:
            if h[j] >= sl:
                return (entry - sl) / entry
            if lo[j] <= tp:
                return (entry - tp) / entry
    return 0.0


def sim_trail(h, lo, n, fill, entry, sl, tp, d):
    risk = abs(entry - sl)
    if risk <= 0:
        return 0.0
    cur_sl = sl; be = False; on = False
    peak = entry
    for j in range(fill, min(fill + 289, n)):
        if d == 1:
            peak = max(peak, h[j])
            prof = (peak - entry) / risk
            if not be and prof >= BE_AT:
                cur_sl = max(cur_sl, entry); be = True
            if not on and prof >= TRAIL_TRIG:
                on = True
            if on:
                cur_sl = max(cur_sl, peak - TRAIL_DIST * risk)
            if lo[j] <= cur_sl:
                return (cur_sl - entry) / entry
            if h[j] >= tp:
                return (tp - entry) / entry
        else:
            peak = min(peak, lo[j])
            prof = (entry - peak) / risk
            if not be and prof >= BE_AT:
                cur_sl = min(cur_sl, entry); be = True
            if not on and prof >= TRAIL_TRIG:
                on = True
            if on:
                cur_sl = min(cur_sl, peak + TRAIL_DIST * risk)
            if h[j] >= cur_sl:
                return (entry - cur_sl) / entry
            if lo[j] <= tp:
                return (entry - tp) / entry
    return 0.0


def run_variant(df, h, lo, n, setups, need_sweep, need_smt, trail,
                close_entry=False, cost=RTCOST):
    trades = []
    for s in setups:
        if need_sweep and not s["sweep"]:
            continue
        if need_smt and not s["smt"]:
            continue
        i = s["i"]
        if close_entry:
            # 7/20 검증구성 재현: CHoCH 종가 즉시진입 + 스윙 SL + 고정 2R.
            entry = s["centry"]; slv = s["swsl"]
            risk = (entry - slv) if s["d"] == 1 else (slv - entry)
            if risk <= 0:
                continue
            tpv = entry + 2 * risk * s["d"]
            fill = i + 1
            if fill >= n:
                continue
            net = sim_fixed(h, lo, n, fill, entry, slv, tpv, s["d"]) - cost
            trades.append((df.index[fill], net * 100))
            continue
        entry = s["entry"]
        fill = None
        for j in range(i + 1, min(i + 1 + RETRACE_TTL, n)):
            if lo[j] <= entry <= h[j]:
                fill = j
                break
        if fill is None:
            continue
        sim = sim_trail if trail else sim_fixed
        net = sim(h, lo, n, fill, entry, s["sl"], s["tp"], s["d"]) - cost
        trades.append((df.index[fill], net * 100))
    return trades


def seg_stats(trades, lo_ts, hi_ts):
    xs = [p for (t, p) in trades if lo_ts <= t < hi_ts]
    if not xs:
        return "n=0"
    a = np.array(xs)
    w = (a > 0).sum()
    return f"n={len(a):4d} net={a.sum():+8.1f}% 승률={100 * w / len(a):3.0f}%"


VARIANTS = [
    # (이름, 스윕요구, SMT, 트레일, 종가진입, 비용)
    ("V0 라이브구성", False, False, False, False, RTCOST),
    ("V1 +스윕요구", True, False, False, False, RTCOST),
    ("V2 +SMT", False, True, False, False, RTCOST),
    ("V3 +트레일/BE", False, False, True, False, RTCOST),
    ("V4 +스윕+트레일(풀)", True, False, True, False, RTCOST),
    ("V5 종가진입2R@0.08", False, False, False, True, 0.0008),
    ("V6 종가진입2R@0.04", False, False, False, True, 0.0004),
    ("V7 종가+스윕@0.08", True, False, False, True, 0.0008),
]

SEGS = [
    ("전체", "2000-01-01", "2100-01-01"),
    ("~2023", "2000-01-01", "2024-01-01"),
    ("2024", "2024-01-01", "2025-01-01"),
    ("2025", "2025-01-01", "2026-01-01"),
    ("2026", "2026-01-01", "2100-01-01"),
]


def main() -> int:
    pairs = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT"]
    agg: dict[str, list] = {v[0]: [] for v in VARIANTS}
    for sym in pairs:
        print(f"\n===== {sym} =====", flush=True)
        df, h, lo, n, setups = collect_setups(sym)
        print(f"후보 셋업 {len(setups)}건 (스윕동반 {sum(1 for s in setups if s['sweep'])}, "
              f"SMT동반 {sum(1 for s in setups if s['smt'])})", flush=True)
        for name, sw, sm, tr, ce, cost in VARIANTS:
            trades = run_variant(df, h, lo, n, setups, sw, sm, tr, ce, cost)
            agg[name].extend(trades)
            segs = "  ".join(
                f"{sn}[{seg_stats(trades, pd.Timestamp(a, tz='UTC'), pd.Timestamp(b, tz='UTC'))}]"
                for sn, a, b in SEGS)
            print(f"{name:22s} {segs}", flush=True)
    print("\n===== 종합 (전 페어 합산) =====", flush=True)
    for name, *_ in VARIANTS:
        trades = agg[name]
        segs = "  ".join(
            f"{sn}[{seg_stats(trades, pd.Timestamp(a, tz='UTC'), pd.Timestamp(b, tz='UTC'))}]"
            for sn, a, b in SEGS)
        print(f"{name:22s} {segs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
