"""P/D 필터 BTC 검증 — 안정형(깊은진입+tp1R)에 프리미엄/디스카운트 게이트 ON/OFF.

ICT P/D 필터(롱=디스카운트/숏=프리미엄)가 ① 방향 균형(숏 고착 해결) ② 승률↑
③ 최대DD↓ 를 내는지. P/D 는 재생 게이트라 timeline 캐시 재사용(즉시). 시드 1000.

  안정형: 0.886/tp1/ttl6, 0.707/tp1/ttl6  × P/D[OFF, ON]
  각 net·USDT·최대DD·승률·롱숏비율·양수월·빈도.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/pd_verify.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

SYM = "BTCUSDT"
SEED = 1000.0
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)
CANDS = [("0.886/tp1/6", 0.886, 2.0, 1.0, 6), ("0.707/tp1/6", 0.707, 2.0, 1.0, 6)]


def _metrics(trades):
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
    return dict(net=cum, usdt=cum * SEED / 100, mdd_usdt=mdd * SEED / 100,
               wr=nwin / n * 100, n=n, lp=nl / n * 100, mon=mon)


def main() -> int:
    df5 = _resample(_load_full(SYM))
    days = len(df5) / 288.0
    print(f"BTC {len(df5)} ({days:.0f}일)", flush=True)
    lines = ["===== P/D 필터 BTC 검증 (안정형 × P/D ON/OFF, 시드 1000) =====",
             f"{'조합':<20} {'USDT':>7} {'최대DD':>8} {'승률':>6} {'롱비율':>7} {'양수월':>7} {'빈도':>7} {'거래':>5}"]
    for label, ote, mr, tp, ttl in CANDS:
        detect_cfg = {**BASE, "ote_level": ote, "min_rr": mr, "entry_ttl_bars": 6}
        tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), SYM)
        for pd_on in (False, True):
            cfg = {**BASE, "ote_level": ote, "min_rr": mr, "entry_ttl_bars": ttl,
                   "tp_rr_override": tp, "apply_pd_filter": pd_on}
            bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
            m = _metrics(bt.trades)
            if not m:
                continue
            tag = f"{label} +P/D" if pd_on else f"{label}"
            lines.append(
                f"{tag:<20} {m['usdt']:+7.0f} {m['mdd_usdt']:7.0f}↓ {m['wr']:5.0f}% "
                f"{m['lp']:5.0f}%/숏{100 - m['lp']:.0f} {m['mon']:5.0f}% {m['n'] / days:6.2f}회 {m['n']:5d}"
            )
        print(f"  {label} done", flush=True)
    lines.append("\n※ P/D ON 시: 롱/숏 비율 50:50 근접(숏 고착 해소) + 승률↑·최대DD↓ 면 채택. 빈도는 절반↓(한쪽 존만).")

    txt = "\n".join(lines)
    with open("pd_verify_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 pd_verify_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
