"""등급(confluence) 스윕 — 최종설정에서 min_confluence 만 바꿔 빈도 vs 손익.

파트너 질문(2026-06-12): 현행 등급4 → 등급3이면 매매 빈도 대비 손익이 어떤가.
다른 모든 설정(RR2.5+SLx3+캡+신선도30+킬존+alignT2)은 최종설정 고정.

사용: python scripts/conf_sweep.py <SYM> <CONF>   (페어×등급당 프로세스 — 병렬)
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest

SYM = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"
CONF = int(sys.argv[2]) if len(sys.argv) > 2 else 3

IN_PHASES = [
    ("2021-11", "2021-11-10", "2021-12-22"),
    ("2023-01", "2023-01-05", "2023-02-16"),
    ("2024-02", "2024-01-25", "2024-03-08"),
]
OUT_PHASES = [
    ("2024-08", "2024-08-01", "2024-09-12"),
    ("2026", "2026-04-28", "2026-06-10"),
]

CFG = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, entry_ttl_bars=120, setup_stale_bars=30, sl_liq_cap=True,
)


def load_slice(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{SYM}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def main() -> int:
    tot = {"in": [0, 0.0, 0], "out": [0, 0.0, 0]}
    days = 0.0
    for grp, phases in (("in", IN_PHASES), ("out", OUT_PHASES)):
        for _pname, s, e in phases:
            df = load_slice(s, e)
            if len(df) < 5000:
                continue
            days += len(df) / 1440.0
            bt = run_backtest(df, BacktestConfig(min_confluence=CONF, **CFG))
            tot[grp][0] += bt.n_trades
            tot[grp][1] += bt.total_net_pnl_pct
            tot[grp][2] += bt.n_wins
    ni, neti, wi = tot["in"]
    no, neto, wo = tot["out"]
    wri = wi / ni * 100 if ni else 0.0
    wro = wo / no * 100 if no else 0.0
    freq = (ni + no) / days if days else 0.0
    print(
        f"{SYM} conf{CONF}: IN n={ni:3d} w={wri:4.1f}% net={neti:+6.2f}% | "
        f"OUT n={no:3d} w={wro:4.1f}% net={neto:+6.2f}% | {freq:.2f}회/일",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
