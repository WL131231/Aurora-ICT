"""TP 재조정 — 안정형 진입(0.707) 고정, TP만 그리드. 7페어 net 흑자+승률 균형점.

7페어 검증: 0.707/tp1 은 net 적자(-139, 알트 SOL/DOGE/LINK 가 주범 — 작은 TP가
대박 못먹어 net 깎음). swing(현행)은 net 흑자지만 승률 27%. → tp_rr 을 1.0~swing
사이에서 훑어 7페어 합산 net 흑자 유지하며 승률 최대인 균형점 탐색. 0.707 timeline
캐시 재사용(즉시). 시드 1000.

  ote 0.707 고정 × tp_rr: swing(0)/1.0/1.5/2.0/2.5/3.0 × 7페어. 5분봉 확정.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/tp_retune.py
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
TP_RRS = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]
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
    n = len(ts)
    nwin = sum(1 for t in ts if t.net_pnl_pct > 0)
    return (cum, mdd, n, nwin)


def main() -> int:
    agg = {tp: [0.0, 0.0, 0, 0, 0.0] for tp in TP_RRS}  # net, mdd합, n, nwin, days
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        days = len(df5) / 288.0
        ttl = 12 if sym == "BTCUSDT" else 6
        detect_cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR, "entry_ttl_bars": ttl}
        tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), sym)
        for tp in TP_RRS:
            cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR, "entry_ttl_bars": ttl, "tp_rr_override": tp}
            bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
            net, mdd, n, nwin = _metrics(bt.trades)
            a = agg[tp]
            a[0] += net; a[1] += mdd; a[2] += n; a[3] += nwin; a[4] += days
        print(f"  {sym} done", flush=True)

    lines = ["===== TP 재조정 (ote 0.707 고정, 7페어 합산, 시드 1000) =====",
             f"{'tp_rr':<10} {'USDT':>8} {'최대DD합':>9} {'승률':>6} {'1일빈도':>8} {'거래':>6}"]
    for tp in TP_RRS:
        net, mdd, n, nwin, days = agg[tp]
        wr = (nwin / n * 100) if n else 0.0
        freq = n / (days / 7) if days else 0.0  # 7페어 합 1일 (days 는 7페어 합산일)
        label = "swing" if tp == 0 else f"x{tp}"
        lines.append(f"{label:<10} {net * SEED / 100:+8.0f} {mdd * SEED / 100:8.0f}↓ {wr:5.0f}% {n / (days / 7):7.2f}회 {n:6d}")
    lines.append("\n※ USDT>0(흑자) 유지하며 승률 최대 + 최대DD↓ 인 tp 가 7페어 균형점.")
    # 흑자 중 승률 최대
    pos = [(tp, *agg[tp]) for tp in TP_RRS if agg[tp][0] > 0]
    if pos:
        best = max(pos, key=lambda x: (x[4] / x[3] if x[3] else 0))  # 승률
        wr = best[4] / best[3] * 100 if best[3] else 0
        lines.append(f"[흑자 중 승률 최대] tp{best[0] or 'swing'}: USDT{best[1] * SEED / 100:+.0f} 승률{wr:.0f}%")
    else:
        lines.append("[흑자 tp 없음 — 0.707 진입 자체 재검토 필요]")

    txt = "\n".join(lines)
    with open("tp_retune_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 tp_retune_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
