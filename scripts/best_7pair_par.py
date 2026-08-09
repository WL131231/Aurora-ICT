"""best 조합(cisd+po3 등) 7페어 robust 재확인 — 병렬판(bt_par). 정합 BASE(t6/s3).

사용: PYTHONPATH=src python scripts/best_7pair_par.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import fmt, run_parallel  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]

BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, sl_liq_cap=True,
    entry_ttl_bars=6, setup_stale_bars=3,
)

VARIANTS = {
    "base":     {},
    "cisd":     dict(apply_cisd=True),
    "cisd+po3": dict(apply_cisd=True, apply_po3=True),
    "db":       dict(apply_dailybias=True),
    "db+cisd":  dict(apply_dailybias=True, apply_cisd=True),
}


def main() -> int:
    totals, per_sym = run_parallel(PAIRS, BASE, VARIANTS, nproc=6)
    txt = fmt("best 조합 7페어 5년 (정합 t6/s3)", totals, per_sym, PAIRS)
    with open("best_7pair_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 best_7pair_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
