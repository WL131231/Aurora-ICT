"""트레일링 SL sweep 7페어 5년 — breakeven(본전 고정, 대박 못먹음)의 업그레이드.
trail_trigger(이익 R 도달 후 시작) × trail_dist(최고가에서 SL 거리 R). 승률 60%+ 중
net 최대 절충 탐색 (파트너: 승률 60%+ 유지하며 net도). cisd+po3 위, 정합 t6/s3.

사용: PYTHONPATH=src python scripts/trail_sweep_5y.py
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
    size_pct=0.9,  # 운영 정합: score 4+ → sizing 90% (min(40+15*score,90))
)

VARIANTS = {
    "base(off)":   {},
    "t0.5_d1.0":   dict(trail_trigger=0.5, trail_dist=1.0),
    "t1.0_d0.5":   dict(trail_trigger=1.0, trail_dist=0.5),
    "t1.0_d1.0":   dict(trail_trigger=1.0, trail_dist=1.0),
    "t1.0_d1.5":   dict(trail_trigger=1.0, trail_dist=1.5),
    "t1.0_d2.0":   dict(trail_trigger=1.0, trail_dist=2.0),
    "t1.5_d1.0":   dict(trail_trigger=1.5, trail_dist=1.0),
    "t1.5_d1.5":   dict(trail_trigger=1.5, trail_dist=1.5),
}


def main() -> int:
    totals, per_sym = run_parallel(PAIRS, BASE, VARIANTS, nproc=6)
    txt = fmt("트레일링 SL sweep 7페어 5년 (cisd+po3 위)", totals, per_sym, PAIRS,
              base_key="base(off)")
    with open("trail_sweep_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 trail_sweep_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
