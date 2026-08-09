"""정합 BASE(t6/s3) 위에서 게이트·boost 변형 전부 5년 재검증 — MSS·EMA·regime·
미활용 detector 를 한 timeline 으로 일괄 재생. 6주 샘플에서 다 실패였는데 5년 정합
BASE 에서도 그런지. BTC·ETH 2페어(속도).

파트너(2026-06-16): 시도한 모든 연구 5년치. OTE·min_rr 은 detect 변형이라 별도.

사용: PYTHONPATH=src python scripts/all_variants_5y.py
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
    "base":               {},
    "mss_flip":           dict(mss_flip=True),
    "EMA_off(B)":         dict(htf_ema_bias="off"),
    "mss_gate(D)":        dict(htf_ema_bias="off", mss_bias_gate=True),
    "align+gate+flip(E)": dict(mss_bias_gate=True, mss_flip=True),
    "align_mss_fill(F)":  dict(align_mss_fill=True, mss_flip=True),
    "regime_0.05":        dict(regime_adaptive=True, regime_spread_thr=0.05),
    "+cbdr":              dict(apply_cbdr=True),
    "+dol":               dict(apply_dol=True),
    "+po3":               dict(apply_po3=True),
    "+dailybias":         dict(apply_dailybias=True),
    "+cisd":              dict(apply_cisd=True),
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
    lines = [f"===== 게이트·boost 변형 5년 (정합 t6/s3, BTC·ETH) | base={base_net:+.2f}% ====="]
    for vname, t in totals.items():
        n, net, w = t
        wr = w / n * 100 if n else 0.0
        delta = net - base_net
        flag = "흑자" if net > 0 else "적자"
        mark = " <<BEST" if net > base_net and vname != "base" else ""
        lines.append(
            f"  {vname:18s} n={n:4d} w={wr:4.1f}% net={net:+7.2f}% (vs base {delta:+5.2f}) {flag}{mark}"
        )
    txt = "\n".join(lines)
    with open("all_variants_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 all_variants_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
