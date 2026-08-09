"""횡보장 회피 게이트 — 7페어 검증. 횡보 국면(|entry_trend| 하위33%) 진입차단 효과.

횡보장 연구 1차(BTC): 횡보 국면은 모든 TP 적자(tp1 -3.1 최악). 중간/추세는 tp1 흑자.
→ 횡보 국면 진입을 빼면 net·승률·DD 개선되는지 7페어로 검증. 거래 후처리로
각 페어 |entry_trend_pct| 하위33%(횡보) 제외한 metric 재계산(진입차단 근사 — 거래
독립이라 후속영향 무시 가능). swing/tp1 둘 다. 시드 1000.

  ote0.707 × [swing, tp1] × [전체 / 횡보(q33)제외] × 7페어.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/regime_filter.py
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
TPS = {"swing": 0.0, "tp1": 1.0}


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
    # agg[(tpname, filtered)] = [net, mdd, n, nwin, days]
    agg = {(name, f): [0.0, 0.0, 0, 0, 0.0] for name in TPS for f in (False, True)}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        days = len(df5) / 288.0
        detect_cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR}
        tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), sym)
        for name, tp in TPS.items():
            cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR, "tp_rr_override": tp}
            trades = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg)).trades
            if len(trades) < 9:
                continue
            # 횡보 경계: 이 페어 |entry_trend| 하위33%
            q33 = sorted(abs(t.entry_trend_pct) for t in trades)[len(trades) // 3]
            for filtered in (False, True):
                sub = [t for t in trades if (not filtered) or abs(t.entry_trend_pct) >= q33]
                net, mdd, n, nwin = _metrics(sub)
                a = agg[(name, filtered)]
                a[0] += net; a[1] += mdd; a[2] += n; a[3] += nwin; a[4] += days
        print(f"  {sym} done", flush=True)

    lines = ["===== 횡보 회피 게이트 7페어 검증 (ote0.707, 시드1000) ====="]
    for name in TPS:
        lines.append(f"\n[{name}]")
        lines.append(f"  {'필터':<10} {'USDT':>8} {'최대DD합':>9} {'승률':>6} {'1일빈도':>8} {'거래':>6}")
        for filtered in (False, True):
            net, mdd, n, nwin, days = agg[(name, filtered)]
            wr = (nwin / n * 100) if n else 0.0
            freq = n / (days / 7) if days else 0.0
            label = "횡보제외" if filtered else "전체"
            lines.append(f"  {label:<10} {net * SEED / 100:+8.0f} {mdd * SEED / 100:8.0f}↓ {wr:5.0f}% {freq:7.2f}회 {n:6d}")

    lines.append("\n※ 횡보제외가 전체 대비 USDT↑·최대DD↓·승률↑(빈도↓는 감수) 면 횡보회피 게이트 채택.")

    txt = "\n".join(lines)
    with open("regime_filter_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 regime_filter_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
