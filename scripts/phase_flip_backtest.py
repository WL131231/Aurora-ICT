"""동적 flip 검증 — 보유 중 EMA 점수 강반전 시 즉시 청산(조윤 동적 전환).

2026-06-10: 정적 align 게이트 대비, 동적 flip(보유 중 추세 반전 시 SL 안 기다리고
청산)이 반등장 손실을 줄이는지 + whipsaw(잦은 flip) 부작용은 없는지 국면별 검증.
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest


def load_slice(sym: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


PHASES = [
    ("2021-11 하락추세", "2021-11-10", "2021-12-22"),
    ("2023-01 베어바닥반등", "2023-01-05", "2023-02-16"),
    ("2024-02 강상승", "2024-01-25", "2024-03-08"),
    ("2024-08 급락반등", "2024-08-01", "2024-09-12"),
    ("2026 반등=실거래구간", "2026-04-28", "2026-06-10"),
]
# baseline(정적 align T2) vs 동적 flip 추가 vs flip 단독.
VARIANTS = [
    ("align T2 (정적)", dict(htf_ema_bias="align", htf_align_threshold=2)),
    ("align T2 + flip T3", dict(
        htf_ema_bias="align", htf_align_threshold=2,
        htf_align_flip=True, htf_align_flip_threshold=3,
    )),
    ("off + flip T3", dict(
        htf_ema_bias="off", htf_align_flip=True, htf_align_flip_threshold=3,
    )),
]


def main() -> int:
    for sym in ["BTCUSDT", "ETHUSDT"]:
        for pname, s, e in PHASES:
            df = load_slice(sym, s, e)
            print(f"== {sym} | {pname} ({len(df):,}봉) ==", flush=True)
            for vname, kw in VARIANTS:
                cfg = BacktestConfig(min_confluence=2, min_rr=2.0, **kw)
                r = run_backtest(df, cfg)
                n_flip = sum(1 for t in r.trades if t.outcome == "flip")
                print(
                    f"  {vname:20s} n={r.n_trades:3d} win={r.win_rate * 100:4.1f}% "
                    f"net={r.total_net_pnl_pct:+7.2f}% "
                    f"L/S={r.long_count}/{r.short_count} flip={n_flip}",
                    flush=True,
                )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
