"""선택 풀 2차 — BCH·ARB·SUI·NEAR 표본 보강 (검증 구간 5→9개).

1차(portfolio_sim_final, 5국면)에서 BCH·ARB 가 양면 흑자였으나 표본 n<25 로
고정 승격을 보류함. 구간을 4개 추가(24-11 랠리 / 25-02 / 25-06 / 25-09)해
표본을 채우고 승격/기각을 결정한다. 시간 순서 분리 유지: 과거 5 = IN,
최근 4 = OUT.

사용: python scripts/choice_pool_round2.py BCHUSDT  (페어당 프로세스 — 병렬)
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest

SYM = sys.argv[1] if len(sys.argv) > 1 else "BCHUSDT"

# IN = 과거 5국면 (상장 전 구간은 자동 skip), OUT = 최근 4국면.
IN_PHASES = [
    ("2021-11 하락", "2021-11-10", "2021-12-22"),
    ("2023-01 반등", "2023-01-05", "2023-02-16"),
    ("2024-02 강상승", "2024-01-25", "2024-03-08"),
    ("2024-08 급락반등", "2024-08-01", "2024-09-12"),
    ("2024-11 대선랠리", "2024-11-01", "2024-12-13"),
]
OUT_PHASES = [
    ("2025-02 조정", "2025-02-01", "2025-03-15"),
    ("2025-06 구간", "2025-06-01", "2025-07-15"),
    ("2025-09 구간", "2025-09-01", "2025-10-15"),
    ("2026 실거래", "2026-04-28", "2026-06-10"),
]

CFG = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, entry_ttl_bars=120, setup_stale_bars=30, sl_liq_cap=True,
)


def load_slice(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{SYM}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def main() -> int:
    print(f"##### {SYM} — 선택 풀 2차 (9국면 표본 보강) #####", flush=True)
    tot = {"in": [0, 0.0, 0], "out": [0, 0.0, 0]}
    for grp, phases in (("in", IN_PHASES), ("out", OUT_PHASES)):
        for pname, s, e in phases:
            df = load_slice(s, e)
            if len(df) < 5000:
                print(f"  [{grp}] {pname}: 데이터 부족 ({len(df)}봉) — skip", flush=True)
                continue
            bt = run_backtest(df, BacktestConfig(**CFG))
            tot[grp][0] += bt.n_trades
            tot[grp][1] += bt.total_net_pnl_pct
            tot[grp][2] += bt.n_wins
            print(
                f"  [{grp}] {pname}: n={bt.n_trades:3d} win={bt.win_rate * 100:4.1f}% "
                f"net={bt.total_net_pnl_pct:+6.2f}%",
                flush=True,
            )
    ni, neti, wi = tot["in"]
    no, neto, wo = tot["out"]
    wri = wi / ni * 100 if ni else 0
    wro = wo / no * 100 if no else 0
    print(
        f"\n{SYM} 합산: IN n={ni} w={wri:.1f}% net={neti:+.2f}% | "
        f"OUT n={no} w={wro:.1f}% net={neto:+.2f}%",
        flush=True,
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
