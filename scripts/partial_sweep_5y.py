"""부분익절 sweep 7페어 5년 — "절반 일찍 챙기고 나머지 추세 끝까지"가 breakeven(전량
본전)보다 net 보존+승률↑ 절충인가. cisd+po3 위, 정합 t6/s3. 파트너 철학 "10% 반익".

사용: PYTHONPATH=src python scripts/partial_sweep_5y.py
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
    apply_cisd=True, apply_po3=True,
)

VARIANTS = {
    "base(off)":  {},
    "p0.7_be":    dict(partial_tp_rr=0.7, partial_be=True),
    "p1.0":       dict(partial_tp_rr=1.0),
    "p1.0_be":    dict(partial_tp_rr=1.0, partial_be=True),
    "p1.5":       dict(partial_tp_rr=1.5),
    "p1.5_be":    dict(partial_tp_rr=1.5, partial_be=True),
    "p2.0_be":    dict(partial_tp_rr=2.0, partial_be=True),
}


def main() -> int:
    totals, per_sym = run_parallel(PAIRS, BASE, VARIANTS, nproc=6)
    txt = fmt("부분익절 sweep 7페어 5년 (cisd+po3 위)", totals, per_sym, PAIRS,
              base_key="base(off)")
    with open("partial_sweep_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 partial_sweep_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
