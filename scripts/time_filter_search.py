"""흑자 엣지 탐색 2단계 — 시간필터(진입 시점) 효과.

24h(상시) vs 킬존만 vs 실버불릿 윈도우만. align T2, conf2/rr2.0 고정.
진입 시점 제한이 손익에 도움 되는지 (노이즈 시간대 회피).
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest

SYMBOLS = sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]


def load_slice(sym: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


PHASES = [
    ("2021-11 하락", "2021-11-10", "2021-12-22"),
    ("2023-01 반등", "2023-01-05", "2023-02-16"),
    ("2024-02 강상승", "2024-01-25", "2024-03-08"),
    ("2024-08 급락반등", "2024-08-01", "2024-09-12"),
    ("2026 실거래", "2026-04-28", "2026-06-10"),
]
TIME_MODES = [
    ("24h 상시", dict(disable_time_filter=True)),
    ("킬존만", dict(disable_time_filter=False, expand_to_killzone=True)),
    ("실버불릿만", dict(disable_time_filter=False, expand_to_killzone=False)),
]


def main() -> int:
    for sym in SYMBOLS:
        agg: dict[str, list] = {t[0]: [0, 0.0, 0, 0] for t in TIME_MODES}
        for pname, s, e in PHASES:
            df = load_slice(sym, s, e)
            print(f"== {sym} | {pname} ({len(df):,}봉) ==", flush=True)
            for tname, kw in TIME_MODES:
                cfg = BacktestConfig(
                    htf_ema_bias="align", htf_align_threshold=2,
                    min_confluence=2, min_rr=2.0, **kw,
                )
                bt = run_backtest(df, cfg)
                a = agg[tname]
                a[0] += bt.n_trades
                a[1] += bt.total_net_pnl_pct
                a[2] += bt.n_wins
                a[3] += bt.n_trades - bt.n_wins
                print(
                    f"  {tname:10s}: n={bt.n_trades:3d} "
                    f"win={bt.win_rate * 100:4.1f}% net={bt.total_net_pnl_pct:+6.2f}%",
                    flush=True,
                )
        print(f"\n##### {sym} 5국면 합산 #####", flush=True)
        for tname, (n, net, w, lo) in sorted(agg.items(), key=lambda x: -x[1][1]):
            wr = w / (w + lo) * 100 if (w + lo) else 0
            print(f"  {tname:10s}: n={n:3d} win={wr:4.1f}% net합={net:+6.2f}%", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
