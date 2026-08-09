"""ttl 1h vs 30분/90분 7페어 확정 — cisd+po3, trail 없음 (기각).

BTC ttl sweep 에서 ttl 1h 가 net 최고(+2.52%) 확인 → 7페어 robust 확정.
빈도는 ttl 로 안 풀림(고정7 한계 1일 0.5회) → 별도 페어확장 과제.
배포 ttl 결정용: ttl 6(30분)/12(1h)/18(90분) 7페어 net 비교.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/ttl_7pair.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import fmt, run_parallel  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]

BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0,
    setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False,
    size_pct=0.9,
)

VARIANTS = {
    "ttl6_30m":  dict(entry_ttl_bars=6),
    "ttl12_1h":  dict(entry_ttl_bars=12),
    "ttl18_90m": dict(entry_ttl_bars=18),
}


def main() -> int:
    totals, per_sym = run_parallel(PAIRS, BASE, VARIANTS, nproc=6)
    txt = fmt("ttl 7페어 확정 (cisd+po3, trail 없음)", totals, per_sym, PAIRS,
              base_key="ttl6_30m")
    with open("ttl_7pair_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 ttl_7pair_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
