"""국면별 EMA align 게이트 백테스트 — off vs strict vs align 비교.

2026-06-10: 5년 다국면 데이터(_full.parquet)를 시장 국면으로 잘라
파트너 EMA 가중치(align) 게이트가 상승/반등 국면에서 숏 고착을 막는지 검증.
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest


def load_slice(sym: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


# (라벨, 시작, 끝) — 국면 대표 구간(각 ~6주). 반등/상승 = align 가치 검증 핵심.
PHASES = [
    ("2021-11 하락추세", "2021-11-10", "2021-12-22"),
    ("2023-01 베어바닥반등", "2023-01-05", "2023-02-16"),
    ("2024-02 강상승", "2024-01-25", "2024-03-08"),
    ("2024-08 급락반등", "2024-08-01", "2024-09-12"),
    ("2026 반등=실거래구간", "2026-04-28", "2026-06-10"),
]
VARIANTS = [
    ("off", dict(htf_ema_bias="off")),
    ("strict(현행)", dict(htf_ema_bias="strict")),
    ("align T1", dict(htf_ema_bias="align", htf_align_threshold=1)),
    ("align T2", dict(htf_ema_bias="align", htf_align_threshold=2)),
    ("align T3", dict(htf_ema_bias="align", htf_align_threshold=3)),
]


def main() -> int:
    for sym in ["BTCUSDT", "ETHUSDT"]:
        for pname, s, e in PHASES:
            df = load_slice(sym, s, e)
            print(f"== {sym} | {pname} ({len(df):,}봉) ==", flush=True)
            for vname, kw in VARIANTS:
                cfg = BacktestConfig(min_confluence=2, min_rr=2.0, **kw)
                r = run_backtest(df, cfg)
                print(
                    f"  {vname:13s} n={r.n_trades:3d} win={r.win_rate * 100:4.1f}% "
                    f"net={r.total_net_pnl_pct:+7.2f}% "
                    f"L/S={r.long_count}/{r.short_count}",
                    flush=True,
                )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
