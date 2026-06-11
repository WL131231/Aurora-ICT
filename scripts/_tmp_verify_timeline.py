"""타임라인 캐시 백테스트 정합성 검증 (임시 — 검증 후 삭제).

BTCUSDT 2024-02-01~02-20 슬라이스에서 4개 cfg 각각
run_backtest vs build_setup_timeline+run_backtest_from_timeline 의
Trade 리스트 필드 단위 완전 일치 확인 + 빌드/재생/원본 시간 측정.
"""
from __future__ import annotations

import dataclasses
import sys
import time

import pandas as pd

from aurora_ict.backtest.replay import (
    BacktestConfig,
    build_setup_timeline,
    run_backtest,
    run_backtest_from_timeline,
)


def load_slice(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet("data/BTCUSDT_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=3, disable_time_filter=False,
)
CFGS = [
    ("cfg1 base", BacktestConfig(**BASE)),
    ("cfg2 sl_dist_mult=2.0", BacktestConfig(**BASE, sl_dist_mult=2.0)),
    ("cfg3 entry_ttl_bars=60", BacktestConfig(**BASE, entry_ttl_bars=60)),
    ("cfg4 conf=2 thr=1", BacktestConfig(**{
        **BASE, "min_confluence": 2, "htf_align_threshold": 1,
    })),
]


def main() -> int:
    df = load_slice("2024-02-01", "2024-02-20")
    print(f"slice bars={len(df):,} ({df.index[0]} ~ {df.index[-1]})", flush=True)

    # 타임라인 1회 빌드 (4개 cfg 모두 detect 파라미터 동일 → 공유)
    t0 = time.perf_counter()
    timeline = build_setup_timeline(df, CFGS[0][1])
    t_build = time.perf_counter() - t0
    n_hits = sum(1 for x in timeline if x is not None)
    print(f"timeline build: {t_build:.1f}s (setup 있는 봉 {n_hits:,}/{len(timeline):,})",
          flush=True)

    all_ok = True
    for name, cfg in CFGS:
        t0 = time.perf_counter()
        ref = run_backtest(df, cfg)
        t_ref = time.perf_counter() - t0
        t0 = time.perf_counter()
        fast = run_backtest_from_timeline(df, timeline, cfg)
        t_fast = time.perf_counter() - t0

        ok = len(ref.trades) == len(fast.trades)
        if ok:
            for k, (a, b) in enumerate(zip(ref.trades, fast.trades, strict=True)):
                for f in dataclasses.fields(a):
                    va, vb = getattr(a, f.name), getattr(b, f.name)
                    if va != vb:
                        ok = False
                        print(f"  MISMATCH trade[{k}].{f.name}: {va!r} != {vb!r}",
                              flush=True)
        else:
            print(f"  MISMATCH n_trades: {len(ref.trades)} != {len(fast.trades)}",
                  flush=True)
        all_ok = all_ok and ok
        status = "MATCH" if ok else "FAIL"
        print(
            f"{name:24s} {status}  trades={ref.n_trades:3d} "
            f"net={ref.total_net_pnl_pct:+.4f} | run_backtest={t_ref:.1f}s "
            f"replay={t_fast:.2f}s (x{t_ref / max(t_fast, 1e-9):.0f})",
            flush=True,
        )
    print("ALL MATCH" if all_ok else "VERIFY FAILED", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
