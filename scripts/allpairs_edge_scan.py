"""#AUTONOMOUS 2026-07-20 Phase1: 전 페어 Origo 2.0 엣지 스캔 (직렬, 증분기록).

파트너 밤샘 자율위임 "모든 경우의 수, 봇 살리기". 페어확장이 최대 레버 —
빈도부족(만성문제) + 분산(약점)을 흑자 저상관 페어 추가로 동시 해결.
데이터 있는 31페어에 라이브 2.0 config(NY_PM 제외 + cond_align post-filter) 5년
백테 → 페어별 net/승률/빈도/RR + walk-forward 전/후반. 직렬(병렬 Pool Windows 좀비
회피), 페어마다 즉시 flush.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

ALL = [
    "AAVEUSDT", "ADAUSDT", "APTUSDT", "ARBUSDT", "ATOMUSDT", "AVAXUSDT", "BCHUSDT",
    "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ENAUSDT", "ETCUSDT", "ETHUSDT",
    "FILUSDT", "HYPEUSDT", "INJUSDT", "LINKUSDT", "LTCUSDT", "NEARUSDT", "OPUSDT",
    "SEIUSDT", "SOLUSDT", "SUIUSDT", "TIAUSDT", "TONUSDT", "TRXUSDT", "UNIUSDT",
    "WIFUSDT", "WLDUSDT", "XRPUSDT",
]
FIXED = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"}
EXCL = {"APTUSDT", "BNBUSDT", "SUIUSDT", "TONUSDT", "UNIUSDT"}
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=True, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
    trail_trigger=2.0, trail_dist=1.5, partial_tp_rr=1.5, partial_be=True,
)
OUT = "allpairs_edge_result.txt"


def gate_kept(bt, df5):
    """2.0 게이트 post-filter (NY_PM 제외 + cond_align)."""
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
        out.append(t)
    return out


def stats(trades):
    n = len(trades)
    if not n:
        return dict(n=0, net=0, wr=0, rr=0)
    net = sum(t.net_pnl_pct for t in trades)
    w = sum(1 for t in trades if t.net_pnl_pct > 0)
    wins = [t.net_pnl_pct for t in trades if t.net_pnl_pct > 0]
    los = [t.net_pnl_pct for t in trades if t.net_pnl_pct < 0]
    rr = (sum(wins) / len(wins)) / abs(sum(los) / len(los)) if wins and los else 0
    return dict(n=n, net=net, wr=100 * w / n, rr=rr)


def main() -> int:
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("#Phase1 전 페어 Origo 2.0 엣지 (5년, NY_PM제외+cond_align)\n")
        f.write(f"{'페어':<10}{'구분':<7}{'거래':>6}{'net%':>9}{'승률':>7}{'RR':>6}"
                f"{'빈도/일':>8}{'H1net':>8}{'H2net':>8}\n")
        f.flush()
        for sym in ALL:
            t0 = time.time()
            try:
                df5 = _resample(_load_full(sym))
                cfg = BacktestConfig(**BASE)
                tl = cached_setup_timeline(df5, cfg, sym)
                bt = run_backtest_from_timeline(df5, tl, cfg)
                kept = gate_kept(bt, df5)
                s = stats(kept)
                span = max((df5.index[-1] - df5.index[0]).days, 1)
                # walk-forward 전/후반
                mid = df5.index[len(df5) // 2].value
                h1 = [t for t in kept if df5.index[t.entry_idx].value < mid]
                h2 = [t for t in kept if df5.index[t.entry_idx].value >= mid]
                tag = "고정" if sym in FIXED else ("제외" if sym in EXCL else "후보")
                line = (f"{sym:<10}{tag:<7}{s['n']:>6}{s['net']:>+9.1f}{s['wr']:>6.0f}%"
                        f"{s['rr']:>6.2f}{s['n']/span:>8.3f}"
                        f"{sum(t.net_pnl_pct for t in h1):>+8.1f}"
                        f"{sum(t.net_pnl_pct for t in h2):>+8.1f}")
            except Exception as e:  # noqa: BLE001
                line = f"{sym:<10} ERROR: {str(e)[:50]}"
            with open(OUT, "a", encoding="utf-8") as g:
                g.write(line + f"   [{time.time()-t0:.0f}s]\n")
            print(line, flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
