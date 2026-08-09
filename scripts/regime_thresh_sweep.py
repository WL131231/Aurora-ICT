"""횡보회피 임계 스윕 — swing TP 고정, 횡보 컷 분위 q0~q40 훑어 DD↓ vs 빈도↓ 최적.

절충안 = 0.707/swing + 횡보회피(net흑자+DD↓). 횡보 컷을 강하게(q40)할수록 DD↓지만
빈도↓·net↓. 약하게(q20)면 빈도↑ net↑ DD↑. 7페어 합산으로 net흑자 유지선에서
DD/빈도 균형점 찾기. 시드 1000.

  ote0.707, swing × 횡보컷 분위 [0(전체)/0.20/0.25/0.33/0.40] × 7페어.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/regime_thresh_sweep.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
OTE = 0.707
MIN_RR = 2.0
CUTS = [0.0, 0.20, 0.25, 0.33, 0.40]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3, entry_ttl_bars=6,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def _metrics(trades):
    ts = sorted(trades, key=lambda t: t.exit_idx)
    if not ts:
        return (0.0, 0.0, 0, 0)
    cum = peak = mdd = 0.0
    for t in ts:
        cum += t.net_pnl_pct
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    nwin = sum(1 for t in ts if t.net_pnl_pct > 0)
    return (cum, mdd, len(ts), nwin)


def main() -> int:
    agg = {c: [0.0, 0.0, 0, 0, 0.0] for c in CUTS}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        days = len(df5) / 288.0
        detect_cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR}
        tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), sym)
        cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR, "tp_rr_override": 0.0}
        trades = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg)).trades
        if len(trades) < 9:
            continue
        absten = sorted(abs(t.entry_trend_pct) for t in trades)
        for c in CUTS:
            thr = absten[int(len(absten) * c)] if c > 0 else -1.0
            sub = [t for t in trades if abs(t.entry_trend_pct) >= thr]
            net, mdd, n, nwin = _metrics(sub)
            a = agg[c]
            a[0] += net; a[1] += mdd; a[2] += n; a[3] += nwin; a[4] += days
        print(f"  {sym} done", flush=True)

    lines = ["===== 횡보회피 임계 스윕 (ote0.707/swing, 7페어, 시드1000) =====",
             f"  {'횡보컷':<8} {'USDT':>8} {'최대DD합':>9} {'승률':>6} {'1일빈도':>8} {'거래':>6}"]
    base_dd = agg[0.0][1] * SEED / 100
    for c in CUTS:
        net, mdd, n, nwin, days = agg[c]
        wr = (nwin / n * 100) if n else 0.0
        freq = n / (days / 7) if days else 0.0
        ddv = mdd * SEED / 100
        label = "없음(전체)" if c == 0 else f"하위{int(c*100)}%"
        ddpct = (ddv - base_dd) / base_dd * 100 if base_dd else 0
        lines.append(f"  {label:<8} {net * SEED / 100:+8.0f} {ddv:8.0f}↓ {wr:5.0f}% {freq:7.2f}회 {n:6d}  (DD{ddpct:+.0f}%)")
    lines.append("\n※ net흑자 유지선에서 DD 최저 + 빈도 덜 깎이는 컷이 라이브 임계.")

    txt = "\n".join(lines)
    with open("regime_thresh_sweep_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 regime_thresh_sweep_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
