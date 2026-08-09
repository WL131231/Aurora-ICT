"""#AUTONOMOUS 2026-07-20: LuxAlgo 신규 지표 대입 스캔 — 무기 발굴.

파트너: LuxAlgo 지표들 중 우리 무기 될 것 탐색. ICT 계열은 이미 완비 → 진짜 신규
(볼륨/RSI/SuperTrend/Nadaraya-Watson)를 개념 재구현해 기존 2.0 trade 에 진입 필터로
대입. 각 지표가 net 개선(잘못된 진입 걸러냄)하는지 측정. Pine 직접이식 대신 개념 근사.

필터(진입 시점 지표값으로 keep/skip):
  - vol: 진입봉 볼륨 > 최근평균×k (볼륨 확인)
  - rsi: RSI 방향정합 (롱=RSI>50 등 / 과매수숏·과매도롱)
  - st: SuperTrend 방향정합
  - nw: 커널회귀(가우시안) 중심선 대비 위치 정합
각 필터 keep 후 net·거래·승률 vs base. 개선하면 confluence 무기 후보.
"""
from __future__ import annotations

import os
import sys

import numpy as np

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


def rsi(c, n=14):
    d = np.diff(c)
    up = np.where(d > 0, d, 0.0)
    dn = np.where(d < 0, -d, 0.0)
    ru = np.convolve(up, np.ones(n) / n, "full")[:len(up)]
    rd = np.convolve(dn, np.ones(n) / n, "full")[:len(dn)]
    rs = ru / (rd + 1e-9)
    return np.concatenate([[50.0], 100 - 100 / (1 + rs)])


def supertrend_dir(h, l, c, period=10, mult=3.0):
    # ATR
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    atr = np.concatenate([[tr[0]], tr])
    atr = np.convolve(atr, np.ones(period) / period, "full")[:len(atr)]
    hl2 = (h + l) / 2
    up = hl2 - mult * atr
    dn = hl2 + mult * atr
    d = np.ones(len(c))
    for i in range(1, len(c)):
        d[i] = 1 if c[i] > dn[i - 1] else (-1 if c[i] < up[i - 1] else d[i - 1])
    return d  # +1 상승추세, -1 하락추세


def nw_center(c, bw=8.0, win=50):
    # 가우시안 커널 회귀 중심선 근사 (인과적: 과거만)
    out = np.copy(c).astype(float)
    for i in range(len(c)):
        lo = max(0, i - win)
        idx = np.arange(lo, i + 1)
        w = np.exp(-((i - idx) ** 2) / (2 * bw ** 2))
        out[i] = np.sum(c[lo:i + 1] * w) / np.sum(w)
    return out


def collect(sym):
    df5 = _resample(_load_full(sym))
    cfg = BacktestConfig(**BASE)
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    c = df5["close"].to_numpy()
    h = df5["high"].to_numpy()
    lo = df5["low"].to_numpy()
    v = df5["volume"].to_numpy()
    r = rsi(c)
    st = supertrend_dir(h, lo, c)
    nw = nw_center(c)
    volma = np.convolve(v, np.ones(20) / 20, "full")[:len(v)]
    mags = [abs(t.entry_trend_pct) for t in bt.trades
            if not (17 <= df5.index[t.entry_idx].hour < 21)]
    q70 = np.percentile(mags, 70) if mags else 0.0
    out = []
    for t in bt.trades:
        hh = df5.index[t.entry_idx].hour
        if 17 <= hh < 21:
            continue
        sgn = 1.0 if t.direction == "long" else -1.0
        if abs(t.entry_trend_pct) < q70 and t.entry_trend_pct * sgn < 0:
            continue
        i = t.entry_idx
        out.append(dict(
            net=t.net_pnl_pct, long=(t.direction == "long"),
            volr=v[i] / (volma[i] + 1e-9),
            rsi=r[i], st=st[i],
            nwpos=(c[i] - nw[i]) / nw[i] * 100,  # 중심선 대비 % (+위 -아래)
        ))
    return out


def stat(g):
    n = len(g)
    if not n:
        return "n=0"
    net = sum(x["net"] for x in g)
    w = sum(1 for x in g if x["net"] > 0)
    return f"n={n:3d} net={net:+7.1f} 승률={100*w/n:3.0f}% net/거래={net/n:+.3f}"


def main() -> int:
    allt = []
    for sym in FIXED:
        allt += collect(sym)
    print(f"base 전체: {stat(allt)}\n")
    print("=== LuxAlgo 신규지표 진입필터 대입 (keep 조건별 net) ===")
    filters = {
        "볼륨>평균1.0x": lambda x: x["volr"] >= 1.0,
        "볼륨>평균1.3x": lambda x: x["volr"] >= 1.3,
        "볼륨<평균0.8x(역)": lambda x: x["volr"] < 0.8,
        "RSI방향정합(롱>50/숏<50)": lambda x: (x["rsi"] > 50) == x["long"],
        "RSI역(롱<50/숏>50)": lambda x: (x["rsi"] > 50) != x["long"],
        "RSI과매수숏/과매도롱": lambda x: (x["rsi"] < 40 and x["long"]) or (x["rsi"] > 60 and not x["long"]),
        "SuperTrend방향정합": lambda x: (x["st"] > 0) == x["long"],
        "SuperTrend역": lambda x: (x["st"] > 0) != x["long"],
        "NW중심선정합(롱>중심)": lambda x: (x["nwpos"] > 0) == x["long"],
        "NW역(롱<중심=되돌림)": lambda x: (x["nwpos"] > 0) != x["long"],
    }
    for name, fn in filters.items():
        kept = [x for x in allt if fn(x)]
        print(f"  {name:<26} {stat(kept)}")
    print("\n→ base 대비 net 유지·상승 & net/거래 개선 = 무기 후보 (품질 향상)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
