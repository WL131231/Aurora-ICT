"""멀티 TF 검증 — 단일 TF vs 멀티 TF(5m·15m·1h 우선+TF별 ttl), IN/OUT 분리.

흑자 조합(conf3 + 시간필터 + align T2) 위에 멀티 TF 를 얹어 순효과 측정.
파트너 아이디어: 높은 TF setup 우선 + 그 TF 에 맞춘 대기시간(ttl).
"""
from __future__ import annotations

import collections
import sys

import pandas as pd

from aurora_ict.backtest.replay import (
    BacktestConfig,
    run_backtest,
    run_backtest_multitf,
)

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


def _cfg() -> BacktestConfig:
    # 흑자 조합: conf3 + 시간필터 + align T2 (SL 은 1.0 — SL 정밀은 별도 sweep).
    return BacktestConfig(
        htf_ema_bias="align", htf_align_threshold=2,
        min_confluence=3, min_rr=2.0, disable_time_filter=False,
    )


def _agg(fn, sym: str, phases: list) -> tuple[int, float, int, int, dict]:
    n = w = lo = 0
    net = 0.0
    tfc: collections.Counter = collections.Counter()
    for _pn, s, e in phases:
        df = load_slice(sym, s, e)
        bt = fn(df, _cfg())
        n += bt.n_trades
        net += bt.total_net_pnl_pct
        w += bt.n_wins
        lo += bt.n_trades - bt.n_wins
        tfc.update(getattr(t, "source_tf", None) for t in bt.trades)
    return n, net, w, lo, dict(tfc)


def main() -> int:
    for sym in SYMBOLS:
        print(f"##### {sym} — 단일 vs 멀티 TF (IN vs OUT) #####", flush=True)
        for label, fn in [("단일TF", run_backtest), ("멀티TF", run_backtest_multitf)]:
            ni, neti, wi, loi, tfi = _agg(fn, sym, IN_PHASES)
            no, neto, wo, loo, tfo = _agg(fn, sym, OUT_PHASES)
            wri = wi / (wi + loi) * 100 if (wi + loi) else 0
            wro = wo / (wo + loo) * 100 if (wo + loo) else 0
            tf = {k: v for k, v in {**tfi, **tfo}.items() if k}
            print(
                f"  {label:7s} IN n={ni:3d} w={wri:4.1f}% net={neti:+6.2f}%  "
                f"| OUT n={no:3d} w={wro:4.1f}% net={neto:+6.2f}%"
                + (f"  TF={tf}" if tf else ""),
                flush=True,
            )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
