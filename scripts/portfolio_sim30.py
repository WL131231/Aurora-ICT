"""페어 확장 포트폴리오 시뮬 — 최종 설정(#EDGE-V2)을 페어별 IN/OUT 검증.

파트너 질문: "페어 확장 구현 시 어떻게 되는지" — 5페어(BTC/ETH/SOL/XRP/DOGE)
각각에 확정 설정(등급4+RR2.5+SLx3+ttl120+킬존+align T2)을 돌려 페어별
엣지 유무와 포트폴리오 합산(총 빈도·총 손익)을 본다.

사용: python scripts/portfolio_sim.py SOLUSDT  (페어당 프로세스 — 병렬)
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest

SYM = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"

IN_PHASES = [
    ("2021-11 하락", "2021-11-10", "2021-12-22"),
    ("2023-01 반등", "2023-01-05", "2023-02-16"),
    ("2024-02 강상승", "2024-01-25", "2024-03-08"),
]
OUT_PHASES = [
    ("2024-08 급락반등", "2024-08-01", "2024-09-12"),
    ("2026 실거래", "2026-04-28", "2026-06-10"),
]

CFG = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, entry_ttl_bars=120, setup_stale_bars=30,
)


def load_slice(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{SYM}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def main() -> int:
    print(f"##### {SYM} — #EDGE-V2 페어 검증 #####", flush=True)
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
        f"\n{SYM} 합산: IN n={ni} w={wri:.1f}% net={neti:+.2f}% "
        f"| OUT n={no} w={wro:.1f}% net={neto:+.2f}% "
        f"| {(ni + no) / 211:.2f}회/일",
        flush=True,
    )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
