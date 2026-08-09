"""breakeven trigger×lock 그리드 5년 — be_lock(본전+α 잠금)이 파트너 철학(높은 승률
+작은 확실 수익)에 best. sweet spot 탐색 후 7페어 robust. cisd+po3 위, 정합 t6/s3.

파트너(2026-06-17): be_0.5_lock0.2 가 승률 68.7%+net+2.69 → 세분화.

사용: PYTHONPATH=src python scripts/be_lock_grid_5y.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import fmt, run_parallel  # noqa: E402

# 7페어 robust 같이 (병렬 6코어라 감당). 표본·알트 확인.
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]

BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, sl_liq_cap=True,
    entry_ttl_bars=6, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True,
)

VARIANTS = {
    "base(be_off)":     {},
    "t0.5_lock0.1":     dict(be_trigger=0.5, be_lock=0.1),
    "t0.5_lock0.2":     dict(be_trigger=0.5, be_lock=0.2),
    "t0.5_lock0.3":     dict(be_trigger=0.5, be_lock=0.3),
    "t0.7_lock0.2":     dict(be_trigger=0.7, be_lock=0.2),
    "t0.7_lock0.3":     dict(be_trigger=0.7, be_lock=0.3),
    "t1.0_lock0.3":     dict(be_trigger=1.0, be_lock=0.3),
    "t1.0_lock0.5":     dict(be_trigger=1.0, be_lock=0.5),
}


def main() -> int:
    totals, per_sym = run_parallel(PAIRS, BASE, VARIANTS, nproc=6)
    txt = fmt("breakeven trigger×lock 7페어 5년 (cisd+po3 위)", totals, per_sym, PAIRS,
              base_key="base(be_off)")
    with open("be_lock_grid_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 be_lock_grid_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
