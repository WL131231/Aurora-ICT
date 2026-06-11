"""자율 연구 3라운드 — 미탐색 수치: CISD 가점 × align 주기 × stale (IN/OUT).

EDGE-V2(등급4+RR2.5+SLx3+ttl120+킬존) 위에 변형을 얹어 개선 여부 확인.
타임라인 캐시 재사용(detect 인자 동일) — 변형은 전부 재생 단계라 저렴.

사용: python scripts/research_round3.py BTCUSDT
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

BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, entry_ttl_bars=120,
)

VARIANTS = [
    ("stale 30분", dict(setup_stale_bars=30)),
    ("stale 90분", dict(setup_stale_bars=90)),
    ("기준(EDGE-V2)", {}),
    ("CISD 가점 ON", dict(apply_cisd=True)),
    ("align 짧게(20~480)", dict(htf_align_periods=(20, 60, 120, 200, 350, 480))),
    ("align 3개(60~200)", dict(htf_align_periods=(60, 120, 200))),
    ("align 길게(120~620)", dict(htf_align_periods=(120, 200, 350, 480, 620))),
    ("stale 60분", dict(setup_stale_bars=60)),
    ("stale 240분", dict(setup_stale_bars=240)),
]


def load_slice(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{SYM}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def main() -> int:
    frames = {}
    build_cfg = BacktestConfig(min_rr=2.5, disable_time_filter=False)
    for pname, s, e in IN_PHASES + OUT_PHASES:
        df = load_slice(s, e)
        print(f"[build] {pname}...", flush=True)
        frames[pname] = (df, build_setup_timeline(df, build_cfg))
    print("[build] 완료", flush=True)

    print(f"##### {SYM} — 3라운드 변형 (IN vs OUT) #####", flush=True)
    for label, kw in VARIANTS:
        cfg = BacktestConfig(**{**BASE, **kw})
        agg = {"in": [0, 0.0], "out": [0, 0.0]}
        for grp, phases in (("in", IN_PHASES), ("out", OUT_PHASES)):
            for pname, _s, _e in phases:
                df, tl = frames[pname]
                bt = run_backtest_from_timeline(df, tl, cfg)
                agg[grp][0] += bt.n_trades
                agg[grp][1] += bt.total_net_pnl_pct
        ni, neti = agg["in"]
        no, neto = agg["out"]
        ok = " <개선>" if (neti > 0 and neto > 0) else ""
        print(
            f"  {label:18s} IN n={ni:3d} {neti:+6.2f}% | OUT n={no:3d} {neto:+6.2f}%{ok}",
            flush=True,
        )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
