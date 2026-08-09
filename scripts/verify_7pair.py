"""1차 그리드(BTC) 상위 후보 7페어 robust 검증 (파트너: 검증은 7페어).

후보 5개: 현행(base) vs net최고 vs 빈도형 vs 승률형(mc5) vs 중간.
특히 승률형(mc5)을 7페어로 → 승률 60% 유지하며 거래 합산↑(빈도 보충) 되는지 확인.
detect 인자(min_rr·dol)가 후보마다 달라 run_parallel_detect(페어별 timeline) 사용.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/verify_7pair.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import fmt, run_parallel_detect  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]

COMMON = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    entry_ttl_bars=6, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False,
    size_pct=0.9,
)

CAND = {
    # 현행 cisd+po3 (트레일 off) — 비교 기준
    "base_mc4": {**COMMON, "min_confluence": 4, "min_rr": 2.5, "sl_dist_mult": 3.0},
    # BTC net 1위 — 진입완화 + 느슨 트레일
    "net1_mc3": {**COMMON, "min_confluence": 3, "min_rr": 2.0, "sl_dist_mult": 3.0,
                 "apply_dol": False, "trail_trigger": 1.5, "trail_dist": 1.5},
    # 빈도+net — +유동성신호
    "freq_mc3dol": {**COMMON, "min_confluence": 3, "min_rr": 2.0, "sl_dist_mult": 3.0,
                    "apply_dol": True, "trail_trigger": 1.5, "trail_dist": 1.5},
    # 승률형 — 극엄격 mc5 + 타이트 트레일 (7페어로 빈도 보충 기대)
    "win_mc5": {**COMMON, "min_confluence": 5, "min_rr": 2.0, "sl_dist_mult": 3.0,
                "apply_dol": False, "trail_trigger": 1.0, "trail_dist": 0.5},
    # 중간 절충 — 완화+유동성 + 중간 트레일 (BTC 승률 49.7%, 빈도 296)
    "mid_mc3dol": {**COMMON, "min_confluence": 3, "min_rr": 2.0, "sl_dist_mult": 3.5,
                   "apply_dol": True, "trail_trigger": 1.0, "trail_dist": 1.0},
}


def main() -> int:
    totals, per_sym = run_parallel_detect(PAIRS, CAND, nproc=6)
    txt = fmt("1차 후보 7페어 robust 검증", totals, per_sym, PAIRS, base_key="base_mc4")
    with open("verify_7pair_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 verify_7pair_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
