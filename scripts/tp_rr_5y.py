"""TP/RR sweep 5년 — TP 를 낮춰 승률을 손익분기(28.6%) 위로 올려 흑자 전환하나.
정합 BASE = 최타이트 t6/s3(ttl 30분, stale 15분; ttl_stale_5y 에서 -0.76% 거의 본전,
승률 25.8%). 여기에 tp_rr_override 로 TP 를 risk 배수로 강제 → 승률↑·RR↓ 순효과.

파트너(2026-06-16): TP 낮춰 승률 높이기. timeline 1회 캐시 → tp_rr 재생.

사용: PYTHONPATH=src python scripts/tp_rr_5y.py
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import (
    BacktestConfig,
    build_setup_timeline,
    run_backtest_from_timeline,
)

FIXED7 = ["BTCUSDT", "ETHUSDT"]  # 속도 위해 2페어 (파트너 2026-06-16)

# 정합 BASE = 최타이트 (ttl_stale_5y 최선).
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, sl_liq_cap=True,
    entry_ttl_bars=6, setup_stale_bars=3,
)
BUILD = dict(**BASE)

VARIANTS = {
    "tp_target(base)": {},          # target swing TP (현행)
    "tp_rr_1.0":       dict(tp_rr_override=1.0),
    "tp_rr_1.3":       dict(tp_rr_override=1.3),
    "tp_rr_1.5":       dict(tp_rr_override=1.5),
    "tp_rr_2.0":       dict(tp_rr_override=2.0),
    "tp_rr_2.5":       dict(tp_rr_override=2.5),
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
    for sym in FIXED7:
        df5 = _resample(_load_full(sym))
        if len(df5) < 700:
            continue
        tl = build_setup_timeline(df5, BacktestConfig(**BUILD))
        for vname, vcfg in VARIANTS.items():
            bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**BASE, **vcfg))
            t = totals[vname]
            t[0] += bt.n_trades
            t[1] += bt.total_net_pnl_pct
            t[2] += bt.n_wins
        print(f"{sym} done", flush=True)

    lines = ["===== TP/RR sweep 5년 (정합 t6/s3, 고정7) ====="]
    for vname, t in totals.items():
        n, net, w = t
        wr = w / n * 100 if n else 0.0
        # tp_rr_override 의 손익분기 승률 (수수료 전): 1/(1+rr)
        rr = float(vname.split("_")[-1]) if vname.startswith("tp_rr") else 2.5
        be = 100 / (1 + rr)
        flag = "흑자" if net > 0 else ("본전권" if net > -1 else "적자")
        lines.append(
            f"  {vname:16s} n={n:5d} w={wr:4.1f}%(BE~{be:.1f}) net={net:+8.2f}% {flag}"
        )
    txt = "\n".join(lines)
    with open("tp_rr_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 tp_rr_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
