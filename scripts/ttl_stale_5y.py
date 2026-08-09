"""라이브 정합 규명 — entry_ttl_bars / setup_stale_bars 가 5년 base 성적에 미치는
영향. 백테스트 BASE(ttl120/stale30)가 라이브(ttl 7200s=24봉 / stale 30min=6봉)와
불일치 → -8% 적자의 주범인지 검증. timeline 1회 캐시 → ttl/stale 조합 재생(빠름).

파트너(2026-06-16): 시도한 모든 연구 5년치. 먼저 라이브 정합부터(가장 임팩트 큼).

사용: PYTHONPATH=src python scripts/ttl_stale_5y.py
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

# ttl/stale 외 공통(현행 라이브 게이트).
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, sl_liq_cap=True,
)
# timeline 빌드용(detect 파라미터 — stale 은 넉넉히 큰 값으로 빌드, 재생 때 변형).
BUILD = dict(**BASE, entry_ttl_bars=120, setup_stale_bars=500)

VARIANTS = {
    "현행_t120_s30": dict(entry_ttl_bars=120, setup_stale_bars=30),
    "정합_t24_s6":   dict(entry_ttl_bars=24, setup_stale_bars=6),
    "t24_s30":       dict(entry_ttl_bars=24, setup_stale_bars=30),
    "t120_s6":       dict(entry_ttl_bars=120, setup_stale_bars=6),
    "t48_s12":       dict(entry_ttl_bars=48, setup_stale_bars=12),
    "t12_s6":        dict(entry_ttl_bars=12, setup_stale_bars=6),
    "t6_s3":         dict(entry_ttl_bars=6, setup_stale_bars=3),
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

    lines = ["===== ttl/stale 5년 민감도 (고정7 합산) ====="]
    for vname, t in totals.items():
        n, net, w = t
        wr = w / n * 100 if n else 0.0
        be = 100 / (1 + 2.5)  # min_rr 2.5 손익분기 승률 28.6%
        flag = "흑자" if net > 0 else "적자"
        lines.append(
            f"  {vname:16s} n={n:5d} w={wr:4.1f}%(BE {be:.1f}) net={net:+8.2f}% {flag}"
        )
    txt = "\n".join(lines)
    with open("ttl_stale_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 ttl_stale_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
