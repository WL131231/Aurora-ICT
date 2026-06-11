"""멀티TF × SL거리 조합 — 흑자 후보 정밀 검증 (IN/OUT).

발견: 멀티TF(높은 TF 우선+TF별 ttl)가 단일TF 를 전 사분면에서 능가(SLx1.0,
BTC OUT +0.38%). SL 넓히기(단조 개선 확인)와 결합해 전 사분면 흑자 탐색.
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_multitf

SYMBOLS = sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]

IN_PHASES = [
    ("2021-11 하락", "2021-11-10", "2021-12-22"),
    ("2023-01 반등", "2023-01-05", "2023-02-16"),
    ("2024-02 강상승", "2024-01-25", "2024-03-08"),
]
OUT_PHASES = [
    ("2024-08 급락반등", "2024-08-01", "2024-09-12"),
    ("2026 실거래", "2026-04-28", "2026-06-10"),
]
SLS = [1.0, 1.5, 2.0, 2.5, 3.0]


def load_slice(sym: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def _agg(sym: str, phases: list, sl: float) -> tuple[int, float, int]:
    n = w = 0
    net = 0.0
    for _pn, s, e in phases:
        df = load_slice(sym, s, e)
        bt = run_backtest_multitf(df, BacktestConfig(
            htf_ema_bias="align", htf_align_threshold=2,
            min_confluence=3, min_rr=2.0, disable_time_filter=False,
            sl_dist_mult=sl,
        ))
        n += bt.n_trades
        net += bt.total_net_pnl_pct
        w += bt.n_wins
    return n, net, w


def main() -> int:
    for sym in SYMBOLS:
        print(f"##### {sym} — 멀티TF × SL (IN vs OUT) #####", flush=True)
        for sl in SLS:
            ni, neti, wi = _agg(sym, IN_PHASES, sl)
            no, neto, wo = _agg(sym, OUT_PHASES, sl)
            wri = wi / ni * 100 if ni else 0
            wro = wo / no * 100 if no else 0
            tag = " <4분면+>" if (neti > 0 and neto > 0) else ""
            print(
                f"  SLx{sl:<4} IN n={ni:3d} w={wri:4.1f}% net={neti:+6.2f}%  "
                f"| OUT n={no:3d} w={wro:4.1f}% net={neto:+6.2f}%{tag}",
                flush=True,
            )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
