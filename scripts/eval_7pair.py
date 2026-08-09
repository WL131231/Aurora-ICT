"""안정형 7페어 검증 — 0.707/tp1/ttl6(라이브 적용안) vs 현행, 시드 1000 USDT.

BTC 단독서 압승(DD18·승56%·양수월57%)한 안정형을 7페어로 일반화 검증. 라이브
배포 전 안전 확인. timeline 캐시(BTC 재사용 + 6페어 첫빌드). 페어별·합산 USDT/
최대DD/승률/양수월/방향/빈도. cisd+po3, sl x3, conf4, 킬존, P/D OFF.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/eval_7pair.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)
# (라벨, ote, min_rr, tp_rr, ttl)
CANDS = [("현행(0.5/swing/6)", 0.5, 2.5, 0.0, 6), ("안정(0.707/tp1/6)", 0.707, 2.0, 1.0, 6)]


def _metrics(trades, days):
    ts = sorted(trades, key=lambda t: t.exit_idx)
    if not ts:
        return None
    cum = peak = mdd = 0.0
    for t in ts:
        cum += t.net_pnl_pct
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    n = len(ts)
    nwin = sum(1 for t in ts if t.net_pnl_pct > 0)
    nl = sum(1 for t in ts if t.direction == "long")
    bucket = 288 * 30
    bk = {}
    for t in ts:
        bk[t.exit_idx // bucket] = bk.get(t.exit_idx // bucket, 0.0) + t.net_pnl_pct
    mon = (sum(1 for v in bk.values() if v > 0) / len(bk) * 100) if bk else 0.0
    return dict(usdt=cum * SEED / 100, mdd=mdd * SEED / 100, wr=nwin / n * 100,
                n=n, lp=nl / n * 100, mon=mon, freq=n / days)


def main() -> int:
    per = {c[0]: [] for c in CANDS}  # 라벨 -> per-pair metrics
    agg = {c[0]: {"usdt": 0.0, "mdd": 0.0, "n": 0, "nwin": 0, "nl": 0, "days": 0.0} for c in CANDS}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        days = len(df5) / 288.0
        ttl_btc = 12 if sym == "BTCUSDT" else 6
        for label, ote, mr, tp, ttl in CANDS:
            detect_cfg = {**BASE, "ote_level": ote, "min_rr": mr, "entry_ttl_bars": ttl_btc}
            tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), sym)
            cfg = {**BASE, "ote_level": ote, "min_rr": mr,
                   "entry_ttl_bars": (ttl_btc if ttl == 6 else ttl), "tp_rr_override": tp}
            bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
            m = _metrics(bt.trades, days)
            if m:
                m["sym"] = sym
                per[label].append(m)
                a = agg[label]
                a["usdt"] += m["usdt"]; a["mdd"] += m["mdd"]; a["n"] += m["n"]
                a["nwin"] += round(m["wr"] / 100 * m["n"]); a["nl"] += round(m["lp"] / 100 * m["n"])
                a["days"] += days
        print(f"  {sym} done", flush=True)

    lines = ["===== 안정형 7페어 검증 (현행 vs 0.707/tp1/6, 시드 1000) ====="]
    for label in (c[0] for c in CANDS):
        lines.append(f"\n[{label}]")
        lines.append(f"  {'페어':<9} {'USDT':>7} {'최대DD':>8} {'승률':>6} {'양수월':>7} {'빈도':>7}")
        for m in per[label]:
            lines.append(f"  {m['sym']:<9} {m['usdt']:+7.0f} {m['mdd']:7.0f}↓ {m['wr']:5.0f}% {m['mon']:5.0f}% {m['freq']:6.2f}회")
        a = agg[label]
        wr = (a["nwin"] / a["n"] * 100) if a["n"] else 0.0
        lp = (a["nl"] / a["n"] * 100) if a["n"] else 0.0
        lines.append(f"  {'합계':<9} {a['usdt']:+7.0f} {a['mdd']:7.0f}↓ {wr:5.0f}% (롱{lp:.0f}%) {a['n'] / 7:.2f}회/일(평균)")
    lines.append("\n※ 0.707/tp1 이 현행보다 합산 USDT↑·최대DD↓·승률↑·양수월↑ 면 라이브 배포 GO.")

    txt = "\n".join(lines)
    with open("eval_7pair_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 eval_7pair_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
