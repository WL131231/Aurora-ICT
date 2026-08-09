"""#AUTONOMOUS 2026-07-20: 시장 전체(BTC) 국면 오버레이 — 장 의존도 낮추기 2차.

거래-순서 clustering 은 멀티유저 착시(백테 자기상관~0). 진짜 국면 지속성은 캘린더
시간에 있음(일별 승률편차 30%p). 크립토는 전부 BTC 베타 → BTC 시장국면(효율비/추세)
을 전 페어 진입에 오버레이. BTC가 극톱질/무추세면 전 페어 skip(시장 전체가 나쁨).

per-pair regime_filter(자기 추세)와 직교: 이건 '시장 전체' 신호. BTC 효율비(ER,
48봉=4h) 또는 |BTC 20봉추세| 기준. 진입 시각의 BTC 상태로 게이팅.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

FIXED = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
    trail_trigger=2.0, trail_dist=1.5, partial_tp_rr=1.5, partial_be=True,
)


def gated(bt, df5):
    mags = [abs(t.entry_trend_pct) for t in bt.trades
            if not (17 <= df5.index[t.entry_idx].hour < 21)]
    q70 = np.percentile(mags, 70) if mags else 0.0
    out = []
    for t in bt.trades:
        h = df5.index[t.entry_idx].hour
        if 17 <= h < 21:
            continue
        sign = 1.0 if t.direction == "long" else -1.0
        if abs(t.entry_trend_pct) < q70 and t.entry_trend_pct * sign < 0:
            continue
        out.append((df5.index[t.entry_idx].value, t.net_pnl_pct))
    return out


def mdd(cum):
    peak = -1e9
    md = 0.0
    for v in cum:
        peak = max(peak, v)
        md = min(md, v - peak)
    return md


def main() -> int:
    # BTC 시장국면 시계열 (5m): ER(48봉) + |20봉추세|
    btc5 = _resample(_load_full("BTCUSDT"))
    bc = btc5["close"].to_numpy()
    bt_ns = btc5.index.values.astype("int64")
    ER_N = 48

    def btc_er_at_ns(ns):
        i = np.searchsorted(bt_ns, ns)
        if i < ER_N or i >= len(bc):
            return 1.0
        seg = bc[i - ER_N:i + 1]
        path = np.abs(np.diff(seg)).sum()
        return abs(seg[-1] - seg[0]) / path if path > 0 else 0.0

    # 전 페어 trade 수집 (시각 + net)
    alltr = []
    for sym in FIXED:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        for ns, net in gated(bt, df5):
            alltr.append((ns, net, btc_er_at_ns(ns)))
    alltr.sort()
    ers = sorted(t[2] for t in alltr)

    def evalf(er_min):
        kept = [(ns, net) for ns, net, er in alltr if er >= er_min]
        net = sum(x[1] for x in kept)
        n = len(kept)
        w = sum(1 for x in kept if x[1] > 0)
        cum = np.cumsum([x[1] for x in kept])
        return net, n, (100 * w / n if n else 0), mdd(cum)

    print(f"전체 {len(alltr)}거래 (BTC 시장 ER 오버레이)\n")
    print(f"{'BTC ER 문턱':<16}{'net%':>8}{'거래':>6}{'승률':>7}{'MDD%':>8}{'net/MDD':>9}")
    bn, bnn, bw, bmd = evalf(0.0)
    print(f"{'base(오버레이 X)':<16}{bn:>+8.1f}{bnn:>6}{bw:>6.0f}%{bmd:>+8.1f}{bn/abs(bmd) if bmd else 0:>9.2f}")
    for q in (10, 20, 30, 40):
        thr = np.percentile(ers, q)
        net, n, wr, md = evalf(thr)
        print(f"{'BTC촙제거<q'+str(q):<16}{net:>+8.1f}{n:>6}{wr:>6.0f}%{md:>+8.1f}"
              f"{net/abs(md) if md else 0:>9.2f}")
    print("\n→ net 유지하며 MDD↓(net/MDD↑) = BTC 시장국면으로 나쁜장 회피 성공")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
