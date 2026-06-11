"""멀티TF + TF플립 검증 — flip off vs on, 흑자조합 기준, IN/OUT 분리.

파트너 의도: 멀티TF(높은 TF 우선 진입) + 보유 중 HTF 신호 뜨면 플립(전환).
흑자조합(conf3 + 시간필터 + align T2) 위에서 플립이 도움 되는지 국면별 검증.
"""
from __future__ import annotations

import collections
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


def load_slice(sym: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def _cfg(flip: bool) -> BacktestConfig:
    return BacktestConfig(
        htf_ema_bias="align", htf_align_threshold=2,
        min_confluence=3, min_rr=2.0, disable_time_filter=False,
        tf_flip=flip,
    )


def _agg(sym: str, phases: list, flip: bool) -> tuple[int, float, int, int, int]:
    n = w = lo = nflip = 0
    net = 0.0
    for _pn, s, e in phases:
        df = load_slice(sym, s, e)
        bt = run_backtest_multitf(df, _cfg(flip))
        n += bt.n_trades
        net += bt.total_net_pnl_pct
        w += bt.n_wins
        lo += bt.n_trades - bt.n_wins
        nflip += sum(1 for t in bt.trades if t.outcome == "tf_flip")
    return n, net, w, lo, nflip


def main() -> int:
    for sym in SYMBOLS:
        print(f"##### {sym} — 멀티TF flip OFF vs ON (IN/OUT) #####", flush=True)
        for label, flip in [("flip OFF", False), ("flip ON ", True)]:
            ni, neti, wi, loi, fi = _agg(sym, IN_PHASES, flip)
            no, neto, wo, loo, fo = _agg(sym, OUT_PHASES, flip)
            wri = wi / (wi + loi) * 100 if (wi + loi) else 0
            wro = wo / (wo + loo) * 100 if (wo + loo) else 0
            print(
                f"  {label} IN n={ni:3d} w={wri:4.1f}% net={neti:+6.2f}%  "
                f"| OUT n={no:3d} w={wro:4.1f}% net={neto:+6.2f}%"
                f"  flip건수={fi + fo}",
                flush=True,
            )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
