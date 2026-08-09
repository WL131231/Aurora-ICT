"""OTE / min_rr sweep 5년 — cisd+po3(확정 흑자) 위에서 진입깊이·RR필터가 더
끌어올리나. detect 변형이라 변형마다 timeline (페어 병렬). 정합 BASE(t6/s3).

파트너(2026-06-16): 추가 흑자축 찾기 + 병렬.

사용: PYTHONPATH=src python scripts/ote_minrr_par.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import fmt, run_parallel_detect  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT"]  # detect 변형 무거워 2페어 먼저, 유망축은 7페어 재확인

# 확정 흑자 토대: 정합 t6/s3 + cisd+po3.
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, sl_liq_cap=True,
    entry_ttl_bars=6, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True,
)

VARIANTS = {
    "base(cisd+po3)": {**BASE},
    "ote_0.62":       {**BASE, "ote_level": 0.62},
    "ote_0.705":      {**BASE, "ote_level": 0.705},
    "ote_0.786":      {**BASE, "ote_level": 0.786},
    "min_rr_1.5":     {**BASE, "min_rr": 1.5},
    "min_rr_2.0":     {**BASE, "min_rr": 2.0},
    "min_rr_3.0":     {**BASE, "min_rr": 3.0},
    "min_rr_2.0+conf3": {**BASE, "min_rr": 2.0, "min_confluence": 3},
}


def main() -> int:
    totals, per_sym = run_parallel_detect(PAIRS, VARIANTS, nproc=6)
    txt = fmt("OTE/min_rr 7페어 5년 (cisd+po3 위)", totals, per_sym, PAIRS,
              base_key="base(cisd+po3)")
    with open("ote_minrr_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 ote_minrr_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
