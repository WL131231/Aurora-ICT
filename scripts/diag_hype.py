"""improvements_verify 행(hang) 진단 — HYPE timeline + variant별 소요시간/거래수.

어느 variant(BASE/A_CT/B_X/B_Y/AB)에서 멈추는지 flush 출력으로 실시간 추적.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from bt_par import _load_full, _resample  # noqa: E402
from improvements_verify import BASE, VARIANTS  # noqa: E402

from aurora_ict.backtest.replay import (  # noqa: E402
    BacktestConfig,
    build_setup_timeline,
    run_backtest_from_timeline,
)

df5 = _resample(_load_full("HYPEUSDT"))
print(f"len(df5)={len(df5)}", flush=True)
base_cfg = {**BASE, "entry_ttl_bars": 6}
t = time.time()
tl = build_setup_timeline(df5, BacktestConfig(**base_cfg))
print(f"timeline build {time.time() - t:.1f}s, setups={sum(1 for v in tl if v)}", flush=True)
for v, extra in VARIANTS.items():
    t = time.time()
    bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**{**base_cfg, **extra}))
    print(f"  {v}: {time.time() - t:.1f}s  trades={len(bt.trades)}", flush=True)
print("DONE", flush=True)
