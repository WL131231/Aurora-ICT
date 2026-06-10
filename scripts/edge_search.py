"""흑자 엣지 탐색 — 파라미터 그리드 sweep (align T2 기준).

2026-06-10 파트너 요청: SL/타점 등 뭐든 바꿔 흑자 조합 탐색.
1단계: 진입품질(min_confluence) × 손익비(min_rr) 그리드. 시간필터/SL 은 다음 단계.
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest

SYMBOLS = sys.argv[1:] or ["BTCUSDT"]


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
GRID = [(c, r) for c in (2, 3, 4) for r in (2.0, 2.5, 3.0, 3.5)]


def main() -> int:
    for sym in SYMBOLS:
        # 국면 합산 누적 — 흑자는 전 국면 합으로 봐야 의미.
        agg: dict[tuple, list] = {g: [0, 0.0, 0, 0] for g in GRID}  # n, net%합, w, l
        for pname, s, e in PHASES:
            df = load_slice(sym, s, e)
            print(f"== {sym} | {pname} ({len(df):,}봉) ==", flush=True)
            for c, r in GRID:
                cfg = BacktestConfig(
                    htf_ema_bias="align", htf_align_threshold=2,
                    min_confluence=c, min_rr=r,
                )
                bt = run_backtest(df, cfg)
                a = agg[(c, r)]
                a[0] += bt.n_trades
                a[1] += bt.total_net_pnl_pct
                a[2] += bt.n_wins
                a[3] += bt.n_trades - bt.n_wins
                print(
                    f"  conf{c} rr{r}: n={bt.n_trades:3d} "
                    f"win={bt.win_rate * 100:4.1f}% net={bt.total_net_pnl_pct:+6.2f}% "
                    f"L/S={bt.long_count}/{bt.short_count}",
                    flush=True,
                )
        print(f"\n##### {sym} 5국면 합산 #####", flush=True)
        for (c, r), (n, net, w, lo) in sorted(agg.items(), key=lambda x: -x[1][1]):
            wr = w / (w + lo) * 100 if (w + lo) else 0
            print(
                f"  conf{c} rr{r}: n={n:3d} win={wr:4.1f}% net합={net:+6.2f}%",
                flush=True,
            )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
