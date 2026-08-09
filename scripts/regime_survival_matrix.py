"""#FST6 2026-07-17 자율연구: 국면별 생존 매트릭스 — 횡보장서 이기는 조합 탐색.

파트너 위임: "이런 장(극단 횡보)도 살아남거나 이기는 조합/매매방식".
1.9 base(NY_PM제외) trade 를 classify_days 국면(상승/하락/횡보/전이)으로 라벨,
후처리 필터 조합별로 국면별 net/승률/빈도 평가. 특히 '횡보' 열이 핵심.

필터 축(후처리, timeline 재빌드 불요):
  - align       : 진입방향이 20봉 추세와 정합(signed>0)
  - cond_align  : 강추세(|trend|>=q70)면 반전허용, 아니면 정합강제
  - magq{N}     : |trend| 하위 N% floor (횡보 회피)
  - with_trend  : |trend|>=q50 강제(추세있을때만) — 극단 횡보 회피
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from regime_edge_lab import classify_days  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=True, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
)
REGIMES = ["상승", "하락", "횡보", "전이↑", "전이↓"]


def collect():
    recs = []  # (regime, |trend|, aligned, net)
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        days_idx, labels = classify_days(df5)
        lab_series = pd.Series(labels, index=days_idx)
        for t in bt.trades:
            h = df5.index[t.entry_idx].hour
            if 17 <= h < 21:  # NY_PM 제외 (1.9 base)
                continue
            entry_day = df5.index[t.entry_idx].normalize()
            pos = lab_series.index.searchsorted(entry_day, side="right") - 1
            reg = lab_series.iloc[pos] if 0 <= pos < len(lab_series) else "횡보"
            sign = 1.0 if t.direction == "long" else -1.0
            recs.append((reg, abs(t.entry_trend_pct), t.entry_trend_pct * sign > 0,
                         t.net_pnl_pct))
    return recs


def main() -> int:
    recs = collect()
    mags = sorted(r[1] for r in recs)
    q50 = np.percentile(mags, 50)
    q70 = np.percentile(mags, 70)
    q40 = np.percentile(mags, 40)

    filters = {
        "base": lambda r: True,
        "align": lambda r: r[2],
        "cond_align(q70)": lambda r: r[1] >= q70 or r[2],
        "magq40": lambda r: r[1] >= q40,
        "with_trend(q50)": lambda r: r[1] >= q50,
        "cond+trend": lambda r: (r[1] >= q70 or r[2]) and r[1] >= q40,
    }

    # 국면별 총량 (base)
    print("=== 국면별 거래 분포 (base) ===")
    for reg in REGIMES:
        sub = [r for r in recs if r[0] == reg]
        if sub:
            net = sum(r[3] for r in sub)
            w = 100 * sum(1 for r in sub if r[3] > 0) / len(sub)
            print(f"  {reg:5s} n={len(sub):3d} net={net:+7.1f} 승률={w:.0f}%")

    print("\n=== 필터 × 국면 net (핵심=횡보 열) ===")
    hdr = f"{'필터':<16}" + "".join(f"{reg:>9}" for reg in REGIMES) + f"{'총net':>9}{'거래':>7}"
    print(hdr)
    for fname, fn in filters.items():
        kept = [r for r in recs if fn(r)]
        row = f"{fname:<16}"
        for reg in REGIMES:
            s = sum(r[3] for r in kept if r[0] == reg)
            row += f"{s:>+9.1f}"
        row += f"{sum(r[3] for r in kept):>+9.1f}{len(kept):>7}"
        print(row)
    print("\n→ '횡보' 열이 +거나 base보다 덜 음수인 필터 = 이런 장 생존")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
