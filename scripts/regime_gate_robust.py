"""국면 게이트 배포 전 강건성 — 상승차단 + 횡보NY_PM 제외 (2026-07-10).

배포 후보 패키지:
    G1 = 상승(z>0.75) 국면 진입 차단 (구제 불가 판정 후 유일 대안)
    G2 = 횡보 국면 NY_PM(02-05 KST) 진입 제외
검증: 각 게이트가 제거하는 버킷의 페어별 일관성(음(-)인 페어 수) + 전/후반
분할에서 양쪽 다 이득인지. 패키지(G1+G2) 합계·walk-forward 포함.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/regime_gate_robust.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_edge_lab import classify_days  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl})
    tl = cached_setup_timeline(df5, cfg, sym)
    days_idx, labels = classify_days(df5)
    day_of = days_idx.searchsorted(df5.index.normalize(), side="right") - 1
    hours = (df5.index + timedelta(hours=9)).hour
    npm = (hours >= 2) & (hours < 5)
    half = len(df5) // 2

    def lab(i):
        di = day_of[i]
        return labels[di] if di >= 0 else "횡보"

    bt = run_backtest_from_timeline(df5, tl, cfg)
    up_bucket = [0.0, 0.0]      # [전반, 후반] 상승 버킷 net
    npm_bucket = [0.0, 0.0]     # 횡보&NY_PM 버킷 net
    base_halves = [0.0, 0.0]
    n_up = n_npm = 0
    for t in bt.trades:
        h = 0 if t.entry_idx < half else 1
        base_halves[h] += t.net_pnl_pct
        if lab(t.entry_idx) == "상승":
            up_bucket[h] += t.net_pnl_pct
            n_up += 1
        elif lab(t.entry_idx) == "횡보" and npm[t.entry_idx]:
            npm_bucket[h] += t.net_pnl_pct
            n_npm += 1
    print(f"  {sym} done", flush=True)
    return sym, dict(up=up_bucket, npmb=npm_bucket, base=base_halves,
                     n_up=n_up, n_npm=n_npm)


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    lines = ["===== 국면 게이트 강건성 (7페어 5년, 시드% 단리) =====",
             "제거 대상 버킷이 음(-)이어야 게이트가 이득.",
             "",
             f"{'페어':<10}{'상승버킷':>10}{'(n)':>5}{'횡보NYPM':>10}{'(n)':>5}"]
    up_neg = npm_neg = 0
    tot_up = [0.0, 0.0]
    tot_npm = [0.0, 0.0]
    tot_base = [0.0, 0.0]
    for sym, m in results:
        u = sum(m["up"])
        v = sum(m["npmb"])
        up_neg += u < 0
        npm_neg += v < 0
        for h in (0, 1):
            tot_up[h] += m["up"][h]
            tot_npm[h] += m["npmb"][h]
            tot_base[h] += m["base"][h]
        lines.append(f"{sym:<10}{u * 100:>+9.0f}%{m['n_up']:>5d}{v * 100:>+9.0f}%{m['n_npm']:>5d}")
    lines.append("")
    lines.append(f"상승버킷 음(-) 페어: {up_neg}/7 · 전/후반 {tot_up[0] * 100:+.0f}%/{tot_up[1] * 100:+.0f}%")
    lines.append(f"횡보NYPM 음(-) 페어: {npm_neg}/7 · 전/후반 {tot_npm[0] * 100:+.0f}%/{tot_npm[1] * 100:+.0f}%")
    g1 = sum(tot_base) - sum(tot_up)
    g12 = g1 - sum(tot_npm)
    lines.append("")
    lines.append(f"기준 합계 {sum(tot_base) * 100:+.0f}% (전/후반 {tot_base[0] * 100:+.0f}/{tot_base[1] * 100:+.0f})")
    lines.append(f"G1(상승차단) 채택 시   {g1 * 100:+.0f}% "
                 f"(전/후반 {(tot_base[0] - tot_up[0]) * 100:+.0f}/{(tot_base[1] - tot_up[1]) * 100:+.0f})")
    lines.append(f"G1+G2(패키지) 채택 시 {g12 * 100:+.0f}% "
                 f"(전/후반 {(tot_base[0] - tot_up[0] - tot_npm[0]) * 100:+.0f}"
                 f"/{(tot_base[1] - tot_up[1] - tot_npm[1]) * 100:+.0f})")
    txt = "\n".join(lines)
    with open("regime_gate_robust_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
