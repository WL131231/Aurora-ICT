"""#AUTONOMOUS 2026-07-20: MMBM/MMSM 정통 4단계 완전구현 백테.

파트너: 근사 아닌 정통 4단계 + 짚은것들(유동성TP·트레일·SMT·dedup·현실비용) 전부.

정통 MMBM 4단계 (buy):
  1) Consolidation — 딜링레인지 (양 한계 사이)
  2) Engineering Liquidity — sell-side 스윕(SSL sweep, 하단 stop 청산)
  3) Smart Money Reversal — HTF bullish PD-array(discount) 도달 + MSS(CHoCH up) + SMT
  4) Liquidity Hunt — 반대쪽 유동성(다음 unswept BSL=swing high) 향해 확장 = TP 타깃
진입: MSS 후 FVG 되돌림. SL=스윕된 저점 아래. TP=다음 BSL(최소2R). 트레일2R/1.5R+BE@1R.
MMSM = premium 미러. 비용=taker 0.04%×2 + 슬리피지.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402
from aurora_ict.indicators.fvg import FVGType, detect_fvgs  # noqa: E402
from aurora_ict.indicators.liquidity import SweepType, detect_liquidity_sweeps  # noqa: E402
from aurora_ict.indicators.smt import SmtType, detect_smt_divergence  # noqa: E402
from aurora_ict.indicators.structure import StructureType, detect_structure_events  # noqa: E402
from aurora_ict.indicators.swing_points import SwingType, detect_swing_points  # noqa: E402

FEE = 0.0004        # taker 편도
SLIP = 0.0002       # 슬리피지 편도 근사
RTCOST = 2 * (FEE + SLIP)  # 왕복 0.12%
RANGE_N = 288
SWEEP_LB = 24       # 스윕이 CHoCH 이전 이 봉 안에 있어야 (engineering→reversal 연결)
RETRACE_TTL = 12
HTF_LB = 20
TRAIL_TRIG, TRAIL_DIST, BE_AT = 2.0, 1.5, 1.0

CORR = {"BTCUSDT": "ETHUSDT", "ETHUSDT": "BTCUSDT"}


def simulate_exit(c_h, c_l, fill, entry, sl, tp, direction, n, detail=False):
    """유동성TP + 트레일(2R/1.5R) + BE@1R 청산 시뮬.

    Args:
        detail: True 면 (gross, 청산봉 인덱스) 반환. False 면 gross 만(기존 호환).
    """
    def _ret(g, j):
        return (g, j) if detail else g
    risk = abs(entry - sl)
    if risk <= 0:
        return _ret(0.0, fill)
    be_done = trail_on = False
    cur_sl = sl
    peak = entry
    for j in range(fill, min(fill + 289, n)):
        hi, lodw = c_h[j], c_l[j]
        if direction == 1:
            peak = max(peak, hi)
            prof = (peak - entry) / risk
            if not be_done and prof >= BE_AT:
                cur_sl = max(cur_sl, entry); be_done = True
            if not trail_on and prof >= TRAIL_TRIG:
                trail_on = True
            if trail_on:
                cur_sl = max(cur_sl, peak - TRAIL_DIST * risk)
            if lodw <= cur_sl:
                return _ret((cur_sl - entry) / entry, j)
            if hi >= tp:
                return _ret((tp - entry) / entry, j)
        else:
            peak = min(peak, lodw)
            prof = (entry - peak) / risk
            if not be_done and prof >= BE_AT:
                cur_sl = min(cur_sl, entry); be_done = True
            if not trail_on and prof >= TRAIL_TRIG:
                trail_on = True
            if trail_on:
                cur_sl = min(cur_sl, peak + TRAIL_DIST * risk)
            if hi >= cur_sl:
                return _ret((entry - cur_sl) / entry, j)
            if lodw <= tp:
                return _ret((entry - tp) / entry, j)
    # 미청산 — 마지막 종가
    _j = min(fill + 288, n - 1)
    last = c_h[_j]
    g = (last - entry) / entry if direction == 1 else (entry - last) / entry
    return _ret(g, _j)


def backtest(sym, use_smt=True, require_sweep=True, detail=False):
    df = _resample(_load_full(sym))
    c = df["close"].to_numpy(); h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    n = len(c)
    ts_ms = (df.index.astype("int64") // 10**6).to_numpy()
    c1h = df["close"].resample("1h").last().ffill()
    bias1h = np.sign(c1h - c1h.shift(HTF_LB)).reindex(df.index, method="ffill").fillna(0).to_numpy()
    swings = detect_swing_points(df, left=3, right=3)
    events = detect_structure_events(df, swings)
    fvgs = detect_fvgs(df, min_size_pct=0.001)
    sweeps = detect_liquidity_sweeps(df, swings)
    bull_fvg = sorted([f for f in fvgs if f.type is FVGType.BULLISH], key=lambda f: f.idx)
    bear_fvg = sorted([f for f in fvgs if f.type is FVGType.BEARISH], key=lambda f: f.idx)
    ssl_sweep = sorted([s.idx for s in sweeps if s.type is SweepType.BULLISH])  # 하단 스윕
    bsl_sweep = sorted([s.idx for s in sweeps if s.type is SweepType.BEARISH])
    swing_highs = sorted([(s.idx, s.price) for s in swings if s.type is SwingType.HIGH])
    swing_lows = sorted([(s.idx, s.price) for s in swings if s.type is SwingType.LOW])

    # SMT (corr 자산)
    smt_bull = set(); smt_bear = set()
    if use_smt:
        corr_sym = CORR.get(sym, "BTCUSDT")
        if corr_sym != sym:
            try:
                cdf = _resample(_load_full(corr_sym))
                for sm in detect_smt_divergence(swings, cdf):
                    (smt_bull if sm.type is SmtType.BULLISH else smt_bear).add(sm.idx)
            except Exception as _e:  # noqa: BLE001
                print(f'  ⚠️ SMT 비활성({sym}): {_e}', flush=True)
                use_smt = False

    def had(idxlist, i, lb):
        return any(i - lb <= x <= i for x in idxlist)

    def recent_fvg(fl, i):
        cd = [f for f in fl if i - 10 <= f.idx <= i + 2]
        return cd[-1] if cd else None

    def next_bsl(i, entry):  # 다음 unswept swing high (위)
        cands = [p for (si, p) in swing_highs if si <= i and p > entry]
        return min(cands) if cands else None

    def next_ssl(i, entry):
        cands = [p for (si, p) in swing_lows if si <= i and p < entry]
        return max(cands) if cands else None

    def sl_low(i):
        return lo[max(0, i - SWEEP_LB):i + 1].min()

    def sl_high(i):
        return h[max(0, i - SWEEP_LB):i + 1].max()

    trades = []
    for ev in events:
        i = ev.idx
        if i < RANGE_N or i >= n - 1:
            continue
        rhi = h[i - RANGE_N:i].max(); rlo = lo[i - RANGE_N:i].min()
        if rhi <= rlo:
            continue
        pos = (c[i] - rlo) / (rhi - rlo)
        bull = ev.type is StructureType.CHOCH_BULLISH
        bear = ev.type is StructureType.CHOCH_BEARISH
        if bull and pos < 0.5 and bias1h[i] >= 0:
            if require_sweep and not had(ssl_sweep, i, SWEEP_LB):
                continue
            if use_smt and i not in smt_bull and not had(sorted(smt_bull), i, SWEEP_LB):
                continue
            fvg = recent_fvg(bull_fvg, i)
            if fvg is None:
                continue
            entry = fvg.mean_threshold
            sl = min(sl_low(i), fvg.low)
            if entry - sl <= 0:
                continue
            risk = entry - sl
            liq = next_bsl(i, entry)
            tp = max(liq if liq else entry + 2 * risk, entry + 2 * risk)
            direction = 1
        elif bear and pos > 0.5 and bias1h[i] <= 0:
            if require_sweep and not had(bsl_sweep, i, SWEEP_LB):
                continue
            if use_smt and i not in smt_bear and not had(sorted(smt_bear), i, SWEEP_LB):
                continue
            fvg = recent_fvg(bear_fvg, i)
            if fvg is None:
                continue
            entry = fvg.mean_threshold
            sl = max(sl_high(i), fvg.high)
            if sl - entry <= 0:
                continue
            risk = sl - entry
            liq = next_ssl(i, entry)
            tp = min(liq if liq else entry - 2 * risk, entry - 2 * risk)
            direction = -1
        else:
            continue
        # 되돌림 체결
        fill = None
        for j in range(i + 1, min(i + 1 + RETRACE_TTL, n)):
            if lo[j] <= entry <= h[j]:
                fill = j; break
        if fill is None:
            continue
        if detail:
            gross, ex_j = simulate_exit(h, lo, fill, entry, sl, tp, direction, n,
                                        detail=True)
            r_mult = gross * entry / risk if risk > 0 else 0.0
            trades.append((ts_ms[fill], (gross - RTCOST) * 100, direction,
                           ts_ms[ex_j], r_mult, gross))
        else:
            gross = simulate_exit(h, lo, fill, entry, sl, tp, direction, n)
            trades.append((ts_ms[fill], (gross - RTCOST) * 100, direction))
    return df, trades


def sb_entry_times(sym):
    """Silver Bullet(Origo 2.0) 진입 시각 — dedup 비교용."""
    base = dict(htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
                min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3, apply_cisd=True,
                apply_po3=True, disable_time_filter=False, size_pct=0.9, ote_level=0.707,
                min_rr=2.0, entry_ttl_bars=6, trail_trigger=2.0, trail_dist=1.5,
                partial_tp_rr=1.5, partial_be=True)
    df = _resample(_load_full(sym))
    cfg = BacktestConfig(**base)
    tl = cached_setup_timeline(df, cfg, sym)
    bt = run_backtest_from_timeline(df, tl, cfg)
    ts = (df.index.astype("int64") // 10**6).to_numpy()
    return set(ts[t.entry_idx] for t in bt.trades)


def mdd(cum):
    peak = -1e9; md = 0.0
    for v in cum:
        peak = max(peak, v); md = min(md, v - peak)
    return md


def main() -> int:
    PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
    for label, kw in [("정통4단계(sweep+SMT)", dict(use_smt=True, require_sweep=True)),
                      ("sweep만(SMT제외)", dict(use_smt=False, require_sweep=True)),
                      ("반전만(sweep·SMT제외=구프로토)", dict(use_smt=False, require_sweep=False))]:
        allt = []
        overlap = 0; total = 0
        for s in PAIRS:
            df, tr = backtest(s, **kw)
            if kw.get("require_sweep") and kw.get("use_smt"):  # dedup 은 정통판만
                sbt = sb_entry_times(s)
                for (t, _, _) in tr:
                    total += 1
                    if any(abs(t - x) <= 12 * 5 * 60 * 1000 for x in sbt):
                        overlap += 1
            allt += [(t, net) for (t, net, _) in tr]
        allt.sort()
        a = np.array([x[1] for x in allt]); tss = np.array([x[0] for x in allt])
        m = np.median(tss) if len(tss) else 0
        wr = 100 * (a > 0).mean() if len(a) else 0
        r = a.sum() / abs(mdd(np.cumsum(a))) if len(a) else 0
        dd = f" dedup중복={100*overlap/total:.0f}%" if total else ""
        print(f"{label:<30} n={len(a):5d} net={a.sum():+7.1f}% 승률={wr:3.0f}% "
              f"net/MDD={r:.2f} H1={a[tss<m].sum():+.0f} H2={a[tss>=m].sum():+.0f}{dd}")
    print("\n(왕복비용 0.12% 반영, 유동성TP+트레일2R/1.5R+BE@1R)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
