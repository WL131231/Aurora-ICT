"""#FST6 2026-07-17 자율연구: 진입 파라미터 그리드 — 촙 생존 조합 탐색 (병렬).

파트너 위임 "경우의 수 다 조합". 촙 생존 가설별 진입파라미터 변형을 5년 7페어
병렬 백테. 총net + 페어robust 로 후보 선별 → 유망하면 국면분할 정밀검증.

가설:
  - 넓은 SL(sl_dist_mult 5/6): 톱질 whipsaw 생존 (bb_breakout 은 trail x6 흑자였음)
  - 엄격 진입(conf6, align_thr 3/4): 빈도↓ 질↑
  - 깊은 OTE(0.786): 체결·되돌림 우위
  - 인내(ttl 12): 성급한 진입 회피
  - 조합: 엄격+넓SL 등
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import fmt, run_parallel  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=True, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
)

VARIANTS = {
    "base": {},
    # 넓은 SL — 톱질 생존
    "sl5": {"sl_dist_mult": 5.0},
    "sl6": {"sl_dist_mult": 6.0},
    # 엄격 진입
    "conf6": {"min_confluence": 6},
    "align3": {"htf_align_threshold": 3},
    "align4": {"htf_align_threshold": 4},
    # 깊은 OTE
    "ote786": {"ote_level": 0.786},
    # 인내
    "ttl12": {"entry_ttl_bars": 12},
    # 높은 RR 문턱
    "rr25": {"min_rr": 2.5},
    # 조합
    "conf6+sl5": {"min_confluence": 6, "sl_dist_mult": 5.0},
    "align3+sl5": {"htf_align_threshold": 3, "sl_dist_mult": 5.0},
    "align3+ote786": {"htf_align_threshold": 3, "ote_level": 0.786},
    "conf6+align3": {"min_confluence": 6, "htf_align_threshold": 3},
    "strict_all": {"min_confluence": 6, "htf_align_threshold": 3, "sl_dist_mult": 5.0},
}


def main() -> int:
    totals, per_sym = run_parallel(PAIRS, BASE, VARIANTS, nproc=6)
    print(fmt("진입 파라미터 그리드 (5년 7페어, 촙 생존 탐색)",
              totals, per_sym, PAIRS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
