"""흑자 엣지 탐색 3단계 — SL 거리 변형 (in-sample / out-of-sample 분리).

과적합 방지: 후보는 IN(과거 3국면)에서 보고 OUT(최근 2국면)으로 검증.
둘 다 좋아야 진짜. align T2, conf2/rr2.0 고정 + SL 거리 mult sweep.
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest

SYMBOLS = sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]

# 과적합 방지: 시간순 분리 (IN=찾기용 과거, OUT=검증용 최근).
IN_PHASES = [
    ("2021-11 하락", "2021-11-10", "2021-12-22"),
    ("2023-01 반등", "2023-01-05", "2023-02-16"),
    ("2024-02 강상승", "2024-01-25", "2024-03-08"),
]
OUT_PHASES = [
    ("2024-08 급락반등", "2024-08-01", "2024-09-12"),
    ("2026 실거래", "2026-04-28", "2026-06-10"),
]
SL_MULTS = [0.7, 1.0, 1.5, 2.0, 3.0]


def load_slice(sym: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def _run_set(sym: str, phases: list, mult: float) -> tuple[int, float, int, int]:
    n = w = lo = 0
    net = 0.0
    for _pname, s, e in phases:
        df = load_slice(sym, s, e)
        cfg = BacktestConfig(
            htf_ema_bias="align", htf_align_threshold=2,
            min_confluence=2, min_rr=2.0, sl_dist_mult=mult,
        )
        bt = run_backtest(df, cfg)
        n += bt.n_trades
        net += bt.total_net_pnl_pct
        w += bt.n_wins
        lo += bt.n_trades - bt.n_wins
    return n, net, w, lo


def main() -> int:
    for sym in SYMBOLS:
        print(f"##### {sym} — SL 거리 sweep (IN vs OUT) #####", flush=True)
        print(f"  {'mult':6s} | {'IN n/win/net':24s} | {'OUT n/win/net':24s}", flush=True)
        for m in SL_MULTS:
            ni, neti, wi, loi = _run_set(sym, IN_PHASES, m)
            no, neto, wo, loo = _run_set(sym, OUT_PHASES, m)
            wri = wi / (wi + loi) * 100 if (wi + loi) else 0
            wro = wo / (wo + loo) * 100 if (wo + loo) else 0
            print(
                f"  x{m:<5.1f}| n={ni:3d} w={wri:4.1f}% net={neti:+6.2f}%   "
                f"| n={no:3d} w={wro:4.1f}% net={neto:+6.2f}%",
                flush=True,
            )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
