"""best 조합(cisd+po3) 7페어 robust 재확인 5년 — BTC·ETH 에서 base 이긴 게 알트
포함 전체에서도 유지되는지. 정합 BASE(t6/s3). timeline 1회 → 변형 재생.

파트너(2026-06-16): 유망하면 7페어 재확인. cisd+po3 가 BTC·ETH 에서 +1.94%(best).

사용: PYTHONPATH=src python scripts/best_7pair_5y.py
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import (
    BacktestConfig,
    build_setup_timeline,
    run_backtest_from_timeline,
)

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]

BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, sl_liq_cap=True,
    entry_ttl_bars=6, setup_stale_bars=3,
)

VARIANTS = {
    "base":      {},
    "cisd":      dict(apply_cisd=True),
    "cisd+po3":  dict(apply_cisd=True, apply_po3=True),
    "db":        dict(apply_dailybias=True),
    "db+cisd":   dict(apply_dailybias=True, apply_cisd=True),
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
    per_sym = {v: {} for v in VARIANTS}
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
            per_sym[vname][sym] = bt.total_net_pnl_pct
        print(f"{sym} done", flush=True)

    base_net = totals["base"][1]
    lines = [f"===== best 조합 7페어 5년 (정합 t6/s3) | base={base_net:+.2f}% ====="]
    for vname, t in totals.items():
        n, net, w = t
        wr = w / n * 100 if n else 0.0
        mark = " <<" if net > base_net and vname != "base" else ""
        lines.append(
            f"  {vname:12s} n={n:5d} w={wr:4.1f}% net={net:+7.2f}% (vs base {net-base_net:+5.2f}){mark}"
        )
    lines.append("----- 페어별 net% -----")
    for sym in PAIRS:
        row = " ".join(f"{vn[:8]}={per_sym[vn].get(sym,0):+5.1f}" for vn in VARIANTS)
        lines.append(f"  {sym:9s} {row}")
    txt = "\n".join(lines)
    with open("best_7pair_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 best_7pair_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
