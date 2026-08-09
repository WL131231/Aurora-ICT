"""5분봉 vs 15분봉 비교 — BTC·ETH, 현행 vs 안정형(0.707/tp1). 시드 1000 USDT.

파트너(6/23): 5분(현행 Origo) vs 15분 TF 비교. 15분은 더 큰 구조(setup 적음·
빈도↓·노이즈↓) — 안정형에 유리한지. TF별 USDT/최대DD/승률/양수월/빈도. ttl 은
봉수 6 동일(5분=30분/15분=90분, TF 특성 반영). timeline 캐시(TF별 df 다름→별도키).

  TF: 5min(현행) / 15min   ×   현행(0.5/swing) / 안정(0.707/tp1)   ×   BTC·ETH
cisd+po3, sl x3, conf4, 킬존, P/D OFF.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/tf_compare.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT"]
TFS = [("5min", 5), ("15min", 15)]
SEED = 1000.0
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3, entry_ttl_bars=6,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)
CANDS = [("현행0.5/swing", 0.5, 2.5, 0.0), ("안정0.707/tp1", 0.707, 2.0, 1.0)]


def _metrics(trades, days):
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
    bucket_bars = int(288 * 30 * 5 / 1)  # 30일 = 분단위 환산은 exit_idx(봉) 기반 아래서
    bk = {}
    # exit_idx 는 봉 인덱스 — 30일치 봉수로 버킷
    return dict(usdt=cum * SEED / 100, mdd=mdd * SEED / 100, wr=nwin / n * 100, n=n,
                freq=n / days, cum=cum, ts=ts)


def _posmonth(ts, bars_per_30d):
    bk = {}
    for t in ts:
        bk[t.exit_idx // bars_per_30d] = bk.get(t.exit_idx // bars_per_30d, 0.0) + t.net_pnl_pct
    return (sum(1 for v in bk.values() if v > 0) / len(bk) * 100) if bk else 0.0


def main() -> int:
    lines = ["===== 5분 vs 15분 비교 (BTC·ETH, 시드 1000) ====="]
    for sym in PAIRS:
        load = _load_full(sym)
        lines.append(f"\n### {sym} ###")
        lines.append(f"  {'TF/조합':<22} {'USDT':>7} {'최대DD':>8} {'승률':>6} {'양수월':>7} {'빈도':>8} {'거래':>5}")
        for tf_rule, tf_min in TFS:
            df = _resample(load, tf_rule)
            days = len(df) * tf_min / 1440.0
            bars_30d = int(30 * 1440 / tf_min)  # 30일치 봉수
            for label, ote, mr, tp in CANDS:
                detect_cfg = {**BASE, "ote_level": ote, "min_rr": mr}
                tl = cached_setup_timeline(df, BacktestConfig(**detect_cfg), f"{sym}_{tf_rule}")
                cfg = {**BASE, "ote_level": ote, "min_rr": mr, "tp_rr_override": tp}
                bt = run_backtest_from_timeline(df, tl, BacktestConfig(**cfg))
                m = _metrics(bt.trades, days)
                if not m:
                    lines.append(f"  {tf_rule} {label:<14} 거래 0")
                    continue
                mon = _posmonth(m["ts"], bars_30d)
                lines.append(
                    f"  {tf_rule} {label:<14} {m['usdt']:+7.0f} {m['mdd']:7.0f}↓ {m['wr']:5.0f}% "
                    f"{mon:5.0f}% {m['freq']:7.2f}회 {m['n']:5d}"
                )
            print(f"  {sym} {tf_rule} done", flush=True)
    lines.append("\n※ 15분이 5분보다 USDT↑·최대DD↓·승률↑·양수월↑(빈도는↓ 예상) 면 안정형엔 15분 유리.")

    txt = "\n".join(lines)
    with open("tf_compare_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 tf_compare_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
