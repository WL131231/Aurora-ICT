"""Origo 1.3 정합 스윕 보충 — rr2.0 계열 청산 변형 (FST #2, 2026-07-02).

1차 정합 스윕 발견: 라이브 강제 min_rr 2.5 가 +124(rr2.0 검증치)를 +15 로 침식.
rr2.5+trail(+138) 이 최선이었으나, rr2.0 으로 되돌린 위에 trail/partial 을 얹으면
더 나은지 확인 (rr2.0 타임라인은 캐시 → 재생만, 수 분).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/origo13_align2.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
BASE13 = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=1.0,
)
VARIANTS = [
    ("rr2.0 고정tp (기준)", {}),
    ("rr2.0 + partial1.5R+BE", dict(partial_tp_rr=1.5, partial_be=True)),
    ("rr2.0 + trail2.0/1.5", dict(tp_rr_override=5.0, trail_trigger=2.0, trail_dist=1.5)),
    ("rr2.0 + trail1.5/1.0", dict(tp_rr_override=5.0, trail_trigger=1.5, trail_dist=1.0)),
    ("rr2.0 + partial1.5+trail2/1.5",
     dict(tp_rr_override=5.0, partial_tp_rr=1.5, partial_be=True,
          trail_trigger=2.0, trail_dist=1.5)),
]


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    out = {}
    for label, ov in VARIANTS:
        cfg = {**BASE13, "entry_ttl_bars": ttl, **ov}
        tl = cached_setup_timeline(df5, BacktestConfig(**cfg), sym)
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
        ts = list(bt.trades)
        wins = [t.net_pnl_pct for t in ts if t.net_pnl_pct > 0]
        cum = peak = mdd = 0.0
        for t in sorted(ts, key=lambda t: t.exit_idx):
            cum += t.net_pnl_pct
            peak = max(peak, cum)
            mdd = max(mdd, peak - cum)
        out[label] = dict(cum=cum, mdd=mdd, n=len(ts), nwin=len(wins),
                          gw=sum(wins), gl=sum(x for x in
                                               (t.net_pnl_pct for t in ts) if x < 0),
                          days=len(df5) / 288.0)
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    lines = ["===== Origo 1.3 정합 보충 (rr2.0 청산 변형, 7페어 5년) =====",
             f"{'변형':<32}{'USDT':>8}{'DD':>7}{'승률':>6}{'RR':>6}{'거래':>7}{'빈도':>7}"]
    for label, _ in VARIANTS:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0, "nwin": 0, "gw": 0.0, "gl": 0.0, "days": 0.0}
        for _sym, out in results:
            for k in tot:
                tot[k] += out[label][k]
        n = tot["n"]
        wr = tot["nwin"] / n * 100 if n else 0.0
        aw = tot["gw"] / tot["nwin"] if tot["nwin"] else 0.0
        al = tot["gl"] / (n - tot["nwin"]) if (n - tot["nwin"]) else 0.0
        rr = aw / -al if al < 0 else 0.0
        freq = n / (tot["days"] / len(PAIRS)) if tot["days"] else 0.0
        lines.append(f"{label:<30}{tot['cum'] * SEED / 100:>+8.0f}{tot['mdd'] * SEED / 100:>7.0f}"
                     f"{wr:>5.0f}%{rr:>6.2f}{n:>7d}{freq:>6.2f}회")
    txt = "\n".join(lines)
    with open("origo13_align2_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
