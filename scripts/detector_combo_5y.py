"""미활용 detector 조합 5년 — 단독에서 base 이긴 dailybias/cisd/po3/cbdr/dol 을
조합하면 더 오르나. 정합 BASE(t6/s3) BTC·ETH. timeline 1회 → 조합 재생.

파트너(2026-06-16): 미활용 detector 가 정합BASE에서 base 이김 → 조합 탐색.

사용: PYTHONPATH=src python scripts/detector_combo_5y.py
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import (
    BacktestConfig,
    build_setup_timeline,
    run_backtest_from_timeline,
)

PAIRS = ["BTCUSDT", "ETHUSDT"]

BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, sl_liq_cap=True,
    entry_ttl_bars=6, setup_stale_bars=3,
)

VARIANTS = {
    "base":            {},
    "db":              dict(apply_dailybias=True),
    "cisd":            dict(apply_cisd=True),
    "db+cisd":         dict(apply_dailybias=True, apply_cisd=True),
    "db+cisd+po3":     dict(apply_dailybias=True, apply_cisd=True, apply_po3=True),
    "db+cisd+po3+cbdr": dict(
        apply_dailybias=True, apply_cisd=True, apply_po3=True, apply_cbdr=True,
    ),
    "all5":            dict(
        apply_dailybias=True, apply_cisd=True, apply_po3=True,
        apply_cbdr=True, apply_dol=True,
    ),
    "db+po3":          dict(apply_dailybias=True, apply_po3=True),
    "cisd+po3":        dict(apply_cisd=True, apply_po3=True),
}


def _resample(d: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    o = d["open"].resample(rule).first()
    h = d["high"].resample(rule).max()
    lo = d["low"].resample(rule).min()
    c = d["close"].resample(rule).last()
    v = d["volume"].resample(rule).sum()
    return pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "volume": v},
    ).dropna()


def _load_full(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]]


def main() -> int:
    totals = {v: [0, 0.0, 0] for v in VARIANTS}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        if len(df5) < 700:
            continue
        tl = build_setup_timeline(df5, BacktestConfig(**BASE))
        for vname, vcfg in VARIANTS.items():
            bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**{**BASE, **vcfg}))
            t = totals[vname]
            t[0] += bt.n_trades
            t[1] += bt.total_net_pnl_pct
            t[2] += bt.n_wins
        print(f"{sym} done", flush=True)

    base_net = totals["base"][1]
    rows = []
    for vname, t in totals.items():
        n, net, w = t
        wr = w / n * 100 if n else 0.0
        rows.append((vname, n, wr, net, net - base_net))
    rows.sort(key=lambda r: -r[3])
    lines = [f"===== 미활용 detector 조합 5년 (정합 t6/s3, BTC·ETH) | base={base_net:+.2f}% ====="]
    for vname, n, wr, net, d in rows:
        mark = " <<TOP" if net > base_net and vname != "base" else ""
        lines.append(
            f"  {vname:18s} n={n:4d} w={wr:4.1f}% net={net:+7.2f}% (vs base {d:+5.2f}){mark}"
        )
    txt = "\n".join(lines)
    with open("detector_combo_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 detector_combo_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
