"""흑자 엣지 탐색 4단계 — robust 레버 조합 (IN/OUT 분리 검증).

conf3 + 시간필터(킬존+SB) + SL넓히기 를 조합. 각 레버 기여 분리도 포함.
과적합 방지: IN(과거 3국면)에서 보고 OUT(최근 2국면)으로 확증. align T2 고정.
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest

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

# (라벨, 변형) — 현행 → 레버 누적 → 분리. align T2/rr2.0 공통.
COMBOS = [
    ("현행 conf2/24h/SLx1", dict(min_confluence=2, disable_time_filter=True, sl_dist_mult=1.0)),
    ("conf3 단독", dict(min_confluence=3, disable_time_filter=True, sl_dist_mult=1.0)),
    ("conf3+시간필터", dict(min_confluence=3, disable_time_filter=False, sl_dist_mult=1.0)),
    ("conf3+시간+SLx2", dict(min_confluence=3, disable_time_filter=False, sl_dist_mult=2.0)),
    ("conf3+시간+SLx3", dict(min_confluence=3, disable_time_filter=False, sl_dist_mult=3.0)),
    ("conf2+시간+SLx3", dict(min_confluence=2, disable_time_filter=False, sl_dist_mult=3.0)),
    ("conf3+시간+SLx3+rr2.5", dict(min_confluence=3, disable_time_filter=False, sl_dist_mult=3.0, min_rr=2.5)),
]


def load_slice(sym: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def _run_set(sym: str, phases: list, kw: dict) -> tuple[int, float, int, int]:
    n = w = lo = 0
    net = 0.0
    base = dict(htf_ema_bias="align", htf_align_threshold=2, min_rr=2.0)
    base.update(kw)
    for _pn, s, e in phases:
        df = load_slice(sym, s, e)
        bt = run_backtest(df, BacktestConfig(**base))
        n += bt.n_trades
        net += bt.total_net_pnl_pct
        w += bt.n_wins
        lo += bt.n_trades - bt.n_wins
    return n, net, w, lo


def main() -> int:
    for sym in SYMBOLS:
        print(f"##### {sym} — 조합 (IN vs OUT) #####", flush=True)
        for label, kw in COMBOS:
            ni, neti, wi, loi = _run_set(sym, IN_PHASES, kw)
            no, neto, wo, loo = _run_set(sym, OUT_PHASES, kw)
            wri = wi / (wi + loi) * 100 if (wi + loi) else 0
            wro = wo / (wo + loo) * 100 if (wo + loo) else 0
            print(
                f"  {label:24s} IN n={ni:3d} w={wri:4.1f}% net={neti:+6.2f}%  "
                f"| OUT n={no:3d} w={wro:4.1f}% net={neto:+6.2f}%",
                flush=True,
            )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
