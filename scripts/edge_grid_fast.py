"""자율 흑자 탐색 — 고속 그리드 (타임라인 캐시, 수백 조합).

detect(비용 99%)를 (slice × min_rr)당 1회만 빌드하고, conf×SL×ttl×alignT 는
run_backtest_from_timeline 재생(0.01s)으로 평가. IN/OUT 분리 + robust 추출.

사용: python scripts/edge_grid_fast.py BTCUSDT  (심볼당 프로세스 — 병렬 실행용)
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
    ("2021-11 하락", "2021-11-10", "2021-12-22"),
    ("2023-01 반등", "2023-01-05", "2023-02-16"),
    ("2024-02 강상승", "2024-01-25", "2024-03-08"),
]
OUT_PHASES = [
    ("2024-08 급락반등", "2024-08-01", "2024-09-12"),
    ("2026 실거래", "2026-04-28", "2026-06-10"),
]

RRS = [2.0, 2.5]            # detect 인자 — rr 별 타임라인 분리
CONFS = [2, 3, 4]
SLS = [1.0, 1.5, 2.0, 2.25, 2.5, 3.0]
TTLS = [5, 15, 30, 60, 120]
ALIGN_TS = [1, 2]
MIN_TRADES = 25             # 합산 거래수 미만이면 통계 불신(과적합) → robust 제외


def load_slice(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{SYM}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def main() -> int:
    # 1) slice × rr 타임라인 빌드 (전체 비용의 대부분 — 1회).
    frames: dict[tuple, tuple] = {}
    for pname, s, e in IN_PHASES + OUT_PHASES:
        df = load_slice(s, e)
        for rr in RRS:
            cfg = BacktestConfig(min_rr=rr, disable_time_filter=False)
            print(f"[build] {pname} rr{rr} ({len(df):,}봉)...", flush=True)
            tl = build_setup_timeline(df, cfg)
            frames[(pname, rr)] = (df, tl)
    print("[build] 완료 — 재생 시작", flush=True)

    # 2) 조합 재생 — IN/OUT 합산.
    rows = []
    for rr in RRS:
        for c in CONFS:
            for sl in SLS:
                for ttl in TTLS:
                    for at in ALIGN_TS:
                        cfg = BacktestConfig(
                            htf_ema_bias="align", htf_align_threshold=at,
                            min_confluence=c, min_rr=rr,
                            disable_time_filter=False,
                            sl_dist_mult=sl, entry_ttl_bars=ttl,
                        )
                        agg = {"in": [0, 0.0, 0], "out": [0, 0.0, 0]}
                        for grp, phases in (("in", IN_PHASES), ("out", OUT_PHASES)):
                            for pname, _s, _e in phases:
                                df, tl = frames[(pname, rr)]
                                bt = run_backtest_from_timeline(df, tl, cfg)
                                agg[grp][0] += bt.n_trades
                                agg[grp][1] += bt.total_net_pnl_pct
                                agg[grp][2] += bt.n_wins
                        ni, neti, wi = agg["in"]
                        no, neto, wo = agg["out"]
                        ntot = ni + no
                        robust = neti > 0 and neto > 0 and ntot >= MIN_TRADES
                        rows.append((rr, c, sl, ttl, at, ni, neti, no, neto, robust))
                        if robust:
                            print(
                                f"  <ROBUST> rr{rr} c{c} sl{sl} ttl{ttl} T{at}: "
                                f"IN n={ni} {neti:+.2f}% | OUT n={no} {neto:+.2f}%",
                                flush=True,
                            )
    # 3) 요약 — OUT 기준 상위 12.
    rows.sort(key=lambda x: -(x[8]))
    print(f"\n##### {SYM} TOP12 (OUT net 기준) #####", flush=True)
    for rr, c, sl, ttl, at, ni, neti, no, neto, rb in rows[:12]:
        tag = " <ROBUST>" if rb else ""
        print(
            f"  rr{rr} c{c} sl{sl} ttl{ttl:3d} T{at}: "
            f"IN n={ni:3d} {neti:+6.2f}% | OUT n={no:3d} {neto:+6.2f}%{tag}",
            flush=True,
        )
    nrb = sum(1 for r in rows if r[9])
    print(f"\nROBUST(4중 양수+n>={MIN_TRADES}) = {nrb}/{len(rows)}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
