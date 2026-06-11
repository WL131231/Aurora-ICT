"""빈도 특화 흑자 탐색 — 하루 2~4회 목표 (파트너 요구) × IN/OUT 흑자 유지.

레버: 시간대(킬존 vs 24h) × conf(3,4) × rr(2.5 — 검증된 영역) × SL(2.5,3.0)
× ttl(30,120,240). 지표에 거래/일 추가 — 빈도와 엣지의 파레토를 본다.
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import (
    BacktestConfig,
    build_setup_timeline,
    run_backtest_from_timeline,
)

SYM = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"

IN_PHASES = [
    ("2021-11", "2021-11-10", "2021-12-22"),
    ("2023-01", "2023-01-05", "2023-02-16"),
    ("2024-02", "2024-01-25", "2024-03-08"),
]
OUT_PHASES = [
    ("2024-08", "2024-08-01", "2024-09-12"),
    ("2026", "2026-04-28", "2026-06-10"),
]
TOTAL_DAYS_IN = 42 * 3
TOTAL_DAYS_OUT = 42 + 43

CONFS = [3, 4]
SLS = [2.5, 3.0]
TTLS = [30, 120, 240]
TIME_MODES = [("킬존", False), ("24h", True)]  # disable_time_filter


def load_slice(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{SYM}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def main() -> int:
    # 시간대는 detect 인자 → (slice × time_mode) 타임라인. rr2.5 고정(검증 영역).
    frames: dict[tuple, tuple] = {}
    for pname, s, e in IN_PHASES + OUT_PHASES:
        df = load_slice(s, e)
        for tname, dis in TIME_MODES:
            print(f"[build] {pname} {tname}...", flush=True)
            cfg = BacktestConfig(min_rr=2.5, disable_time_filter=dis)
            frames[(pname, tname)] = (df, build_setup_timeline(df, cfg))
    print("[build] 완료", flush=True)

    rows = []
    for tname, dis in TIME_MODES:
        for c in CONFS:
            for sl in SLS:
                for ttl in TTLS:
                    cfg = BacktestConfig(
                        htf_ema_bias="align", htf_align_threshold=2,
                        min_confluence=c, min_rr=2.5,
                        disable_time_filter=dis,
                        sl_dist_mult=sl, entry_ttl_bars=ttl,
                    )
                    agg = {"in": [0, 0.0], "out": [0, 0.0]}
                    for grp, phases in (("in", IN_PHASES), ("out", OUT_PHASES)):
                        for pname, _s, _e in phases:
                            df, tl = frames[(pname, tname)]
                            bt = run_backtest_from_timeline(df, tl, cfg)
                            agg[grp][0] += bt.n_trades
                            agg[grp][1] += bt.total_net_pnl_pct
                    ni, neti = agg["in"]
                    no, neto = agg["out"]
                    per_day = (ni + no) / (TOTAL_DAYS_IN + TOTAL_DAYS_OUT)
                    ok = neti > 0 and neto > 0
                    rows.append((tname, c, sl, ttl, ni, neti, no, neto, per_day, ok))
                    print(
                        f"  {tname} c{c} sl{sl} ttl{ttl:3d}: "
                        f"IN n={ni:3d} {neti:+6.2f}% | OUT n={no:3d} {neto:+6.2f}%"
                        f" | {per_day:.2f}회/일{' <흑자>' if ok else ''}",
                        flush=True,
                    )
    print(f"\n##### {SYM} 빈도-엣지 파레토 (흑자만, 거래/일 내림차순) #####", flush=True)
    for r in sorted([r for r in rows if r[9]], key=lambda x: -x[8]):
        print(
            f"  {r[0]} c{r[1]} sl{r[2]} ttl{r[3]}: {r[8]:.2f}회/일 "
            f"IN {r[5]:+.2f}% OUT {r[7]:+.2f}%",
            flush=True,
        )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
