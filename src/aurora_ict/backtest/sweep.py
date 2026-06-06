"""ICT 백테스트 파라미터 sweep — 최적 설정 탐색.

실행:
    python -m aurora_ict.backtest.sweep [parquet_path]

grid 의 모든 파라미터 조합을 백테스트해 net PnL 내림차순으로 출력. 절대 수치는
낙관 편향(즉시 체결 가정 등)이 있으므로 **조합 간 상대 비교**용으로 해석할 것.

담당: 지영민 (#BACKTEST 2026-06-06)
"""

from __future__ import annotations

import sys
from itertools import product

import pandas as pd

from aurora_ict.backtest.replay import (
    BacktestConfig,
    BacktestResult,
    load_ohlcv_parquet,
    run_backtest,
)

DEFAULT_PARQUET = "data/ETHUSDT_1m_sample.parquet"


def run_sweep(
    df: pd.DataFrame, grid: dict[str, list],
) -> list[tuple[dict, BacktestResult]]:
    """grid(파라미터→값 리스트)의 모든 조합 백테스트 → net PnL 내림차순 정렬.

    Args:
        df: load_ohlcv_parquet 산출 OHLCV.
        grid: ``{"min_rr": [1.5, 2.0], "min_confluence": [1, 2]}`` 형태.

    Returns:
        ``[(params, result), ...]`` — total_net_pnl_pct 내림차순.
    """
    keys = list(grid.keys())
    rows: list[tuple[dict, BacktestResult]] = []
    for combo in product(*grid.values()):
        params = dict(zip(keys, combo, strict=True))
        rows.append((params, run_backtest(df, BacktestConfig(**params))))
    rows.sort(key=lambda x: x[1].total_net_pnl_pct, reverse=True)
    return rows


def main(argv: list[str] | None = None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    path = argv[0] if argv else DEFAULT_PARQUET
    df = load_ohlcv_parquet(path)
    print(f"loaded {len(df)} bars from {path}\n")
    grid = {
        "min_rr": [1.5, 2.0, 2.5, 3.0],
        "min_confluence": [1, 2, 3],
    }
    rows = run_sweep(df, grid)
    for params, res in rows:
        ps = " ".join(f"{k}={v}" for k, v in params.items())
        print(f"{ps:32} | {res.summary()}")


if __name__ == "__main__":
    main()
