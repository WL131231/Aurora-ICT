"""횡보회피 robust 검증 — 라이브 후보(0.707/swing/횡보컷33%)가 전·후반 일관 흑자인지.

라이브 후보 확정 전 마지막 관문: 횡보회피33%가 특정 시기 과적합 아닌지. 7페어 swing
거래를 시간순(exit_idx) 전반/후반 반토막 → 각 구간 net·DD (전체 vs 횡보컷33%).
양 구간 모두 횡보컷이 net흑자 유지 + DD↓ 면 robust. 캐시 재사용(재빌드 없음 —
trade 후처리만). 시드 1000.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/regime_robust.py
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
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3, entry_ttl_bars=6,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def _metrics(trades):
    ts = sorted(trades, key=lambda t: t.exit_idx)
    if not ts:
        return (0.0, 0.0, 0)
    cum = peak = mdd = 0.0
    for t in ts:
        cum += t.net_pnl_pct
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return (cum, mdd, len(ts))


def main() -> int:
    # half(0/1) × filtered(F/T) -> [net, mdd, n]
    agg = {(h, f): [0.0, 0.0, 0] for h in (0, 1) for f in (False, True)}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        detect_cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR}
        tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), sym)
        cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR, "tp_rr_override": 0.0}
        trades = sorted(run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg)).trades,
                        key=lambda t: t.exit_idx)
        if len(trades) < 18:
            continue
        thr = sorted(abs(t.entry_trend_pct) for t in trades)[len(trades) // 3]
        mid = len(trades) // 2
        for h, part in ((0, trades[:mid]), (1, trades[mid:])):
            for filtered in (False, True):
                sub = [t for t in part if (not filtered) or abs(t.entry_trend_pct) >= thr]
                net, mdd, n = _metrics(sub)
                a = agg[(h, filtered)]
                a[0] += net; a[1] += mdd; a[2] += n
        print(f"  {sym} done", flush=True)

    lines = ["===== 횡보회피33% robust (전후반, ote0.707/swing, 7페어, 시드1000) ====="]
    for h, name in ((0, "전반"), (1, "후반")):
        lines.append(f"\n[{name}]")
        lines.append(f"  {'필터':<10} {'USDT':>8} {'최대DD합':>9} {'거래':>6}")
        for filtered in (False, True):
            net, mdd, n = agg[(h, filtered)]
            label = "횡보컷33%" if filtered else "전체"
            lines.append(f"  {label:<10} {net * SEED / 100:+8.0f} {mdd * SEED / 100:8.0f}↓ {n:6d}")
    lines.append("\n※ 전·후반 양쪽서 횡보컷이 net흑자 + DD↓ 유지면 robust → 라이브 GO.")

    txt = "\n".join(lines)
    with open("regime_robust_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 regime_robust_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
