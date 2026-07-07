"""Origo 1.5 페어 확장 검증 — 빈도 회복 축 (FST #4 자율연구, 2026-07-07).

conf5 게이트로 빈도 0.24/일(고정7 합) — 게이트를 풀지 않고 빈도를 회복하는
정석은 페어 확장(#EDGE-V2 로드맵). 6/13 알트 탐색 추천순위(NEAR>ENA>FIL>ARB,
+BCH 추천후보) + 와일드카드 AVAX 에 Origo 1.5 풀구성(conf5/SLx4/rr2.0 +
trail 2.0/1.5 + BE@1R)을 적용해 5년 net·전/후반 robust 를 본다.

채택 기준(고정 리스트 편입 제안): net > 0 AND 전/후반 모두 > 0 (walk-forward).
사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/origo15_pair_expand.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CANDIDATES = ["NEARUSDT", "ENAUSDT", "FILUSDT", "ARBUSDT", "BCHUSDT", "AVAXUSDT"]
SEED = 1000.0
CFG15 = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=5.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
    entry_ttl_bars=6,
)


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    cfg = BacktestConfig(**CFG15)
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    ts = sorted(bt.trades, key=lambda t: t.exit_idx)
    half = len(df5) // 2
    cum = peak = mdd = 0.0
    for t in ts:
        cum += t.net_pnl_pct
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    wins = [t.net_pnl_pct for t in ts if t.net_pnl_pct > 0]
    h1 = sum(t.net_pnl_pct for t in ts if t.exit_idx < half)
    h2 = sum(t.net_pnl_pct for t in ts if t.exit_idx >= half)
    days = len(df5) / 288.0
    print(f"  {sym} done n={len(ts)}", flush=True)
    return sym, dict(usdt=cum * SEED / 100, mdd=mdd * SEED / 100,
                     wr=len(wins) / len(ts) * 100 if ts else 0, n=len(ts),
                     h1=h1 * SEED / 100, h2=h2 * SEED / 100, freq=len(ts) / days)


def main() -> int:
    with Pool(3) as p:
        results = p.map(_pair_worker, CANDIDATES)
    lines = ["===== Origo 1.5 페어 확장 후보 (5년, 시드1000, conf5/SLx4/rr2.0+trail+BE) =====",
             f"{'페어':<10}{'USDT':>8}{'DD':>7}{'승률':>6}{'거래':>6}{'빈도':>7}"
             f"{'전반':>8}{'후반':>8}  판정"]
    for sym, m in results:
        robust = m["usdt"] > 0 and m["h1"] > 0 and m["h2"] > 0
        verdict = "✅ 편입 후보" if robust else ("△ net+" if m["usdt"] > 0 else "❌")
        lines.append(f"{sym:<10}{m['usdt']:>+8.0f}{m['mdd']:>7.0f}{m['wr']:>5.0f}%"
                     f"{m['n']:>6d}{m['freq']:>6.2f}회{m['h1']:>+8.0f}{m['h2']:>+8.0f}  {verdict}")
    txt = "\n".join(lines)
    with open("origo15_pair_expand_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
