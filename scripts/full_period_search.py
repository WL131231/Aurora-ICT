"""전체 기간(2021-07~2026-06, 약 5년) 연속 백테스트 — 6주 샘플이 아닌 전 구간.
상승·하락·횡보 전부 포함한 종합 성적으로 국면 적응형 게이트(regime)를 검증.

detect 캐시(build_setup_timeline) 1회 빌드 → base + regime 임계 sweep 재생(빠름).
regime 은 게이트라 detect 파라미터 불변 → timeline 1개 공유.

파트너(2026-06-15): 전부 기간대로 쭉 돌려서 상승·하락·변동성 다 측정 = 정확한 데이터.

사용: PYTHONPATH=src python scripts/full_period_search.py
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import (
    BacktestConfig,
    build_setup_timeline,
    run_backtest_from_timeline,
)

FIXED7 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]

# detect 파라미터(timeline 빌드용) + 게이트 공통.
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, entry_ttl_bars=120, setup_stale_bars=30, sl_liq_cap=True,
)

# 게이트만 다른 변형(detect 동일 → timeline 공유).
VARIANTS = {
    "base(현행)":   {},
    "regime_0.030": dict(regime_adaptive=True, regime_spread_thr=0.030),
    "regime_0.040": dict(regime_adaptive=True, regime_spread_thr=0.040),
    "regime_0.050": dict(regime_adaptive=True, regime_spread_thr=0.050),
    "regime_0.060": dict(regime_adaptive=True, regime_spread_thr=0.060),
    "regime_0.070": dict(regime_adaptive=True, regime_spread_thr=0.070),
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
    for sym in FIXED7:
        df5 = _resample(_load_full(sym))
        if len(df5) < 700:
            print(f"{sym}: 데이터 부족 skip", flush=True)
            continue
        tl = build_setup_timeline(df5, BacktestConfig(**BASE))  # detect 1회
        for vname, vcfg in VARIANTS.items():
            bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**BASE, **vcfg))
            t = totals[vname]
            t[0] += bt.n_trades
            t[1] += bt.total_net_pnl_pct
            t[2] += bt.n_wins
            per_sym[vname][sym] = bt.total_net_pnl_pct
        print(f"{sym} done (5m봉 {len(df5)})", flush=True)

    lines = ["===== 전체 기간(약 5년) 종합 - 고정7 합산 ====="]
    for vname, t in totals.items():
        n, net, w = t
        wr = w / n * 100 if n else 0.0
        lines.append(f"  {vname:14s} n={n:5d} w={wr:4.1f}% net={net:+8.2f}%")
    lines.append("----- 페어별 net% (base vs regime) -----")
    for sym in FIXED7:
        row = " ".join(
            f"{vn.split('_')[-1][:5]}={per_sym[vn].get(sym, 0):+6.1f}" for vn in VARIANTS
        )
        lines.append(f"  {sym:9s} {row}")
    txt = "\n".join(lines)
    with open("full_period_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 full_period_result.txt 에 저장됨)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
