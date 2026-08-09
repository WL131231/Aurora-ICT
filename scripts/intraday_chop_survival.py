"""#FST6 2026-07-17 자율연구: 장중 극단 톱질 구간 생존 — 현재장 정조준.

최근 2주 손실은 일별국면(횡보 +10.4 흑자)이 아니라 '장중 극단 톱질'(효율비
ER 1~7%). 각 진입 직전 로컬 ER(Kaufman efficiency ratio, 48봉=4h)을 계산해
trade 를 ER 분위로 나눔. 최저 ER(현재장 아날로그)에서 어떤 필터가 살아남는지.

ER = |close[i]-close[i-n]| / Σ|close[j]-close[j-1]|. 낮을수록 톱질(무추세).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=True, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
)
ER_N = 48  # 4시간(5m×48) 로컬 효율비


def er_at(closes: np.ndarray, idx: int, n: int = ER_N) -> float:
    if idx < n:
        return 1.0
    seg = closes[idx - n:idx + 1]
    net = abs(seg[-1] - seg[0])
    path = np.abs(np.diff(seg)).sum()
    return net / path if path > 0 else 0.0


def collect():
    recs = []  # (ER, |trend|, aligned, net)
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        closes = df5["close"].to_numpy()
        cfg = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        for t in bt.trades:
            h = df5.index[t.entry_idx].hour
            if 17 <= h < 21:
                continue
            er = er_at(closes, t.entry_idx)
            sign = 1.0 if t.direction == "long" else -1.0
            recs.append((er, abs(t.entry_trend_pct), t.entry_trend_pct * sign > 0,
                         t.net_pnl_pct))
    return recs


def stat(g):
    n = len(g)
    if not n:
        return 0.0, 0.0, 0
    net = sum(r[3] for r in g)
    w = 100 * sum(1 for r in g if r[3] > 0) / n
    return net, w, n


def main() -> int:
    recs = collect()
    ers = sorted(r[0] for r in recs)
    q20 = np.percentile(ers, 20)
    q40 = np.percentile(ers, 40)
    print(f"ER 분포: 최저20%<{q20:.2f}  최저40%<{q40:.2f}  (총 {len(recs)}건)")
    print("(현재장 ER 1~7%=0.01~0.07 → 최저 버킷이 현재장 아날로그)\n")

    # ER 버킷별 base 성과
    print("=== ER 버킷별 base 성과 ===")
    buckets = [("극톱질 ER<q20", lambda r: r[0] < q20),
               ("중 q20~q40", lambda r: q20 <= r[0] < q40),
               ("추세 ER>=q40", lambda r: r[0] >= q40)]
    for bname, bfn in buckets:
        net, w, n = stat([r for r in recs if bfn(r)])
        print(f"  {bname:16s} net={net:+7.1f} 승률={w:.0f}% n={n}")

    # 극톱질 버킷에서 필터별 성과 (현재장 생존 핵심)
    mags = sorted(r[1] for r in recs)
    q70 = np.percentile(mags, 70)
    print("\n=== 극톱질(ER<q20) 구간 — 필터별 생존 ===")
    chop = [r for r in recs if r[0] < q20]
    filters = {
        "base(무필터)": lambda r: True,
        "align(정합)": lambda r: r[2],
        "cond_align(q70)": lambda r: r[1] >= q70 or r[2],
        "역추세만(반전)": lambda r: not r[2],
    }
    for fname, fn in filters.items():
        net, w, n = stat([r for r in chop if fn(r)])
        print(f"  {fname:16s} net={net:+7.1f} 승률={w:.0f}% n={n}")
    print("\n→ 극톱질서 +net 유지하는 필터 = 현재장 생존책")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
