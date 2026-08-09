"""브레이크이븐 sweep 5년 — cisd+po3(확정 흑자) 위에서 be_trigger 가 "이익 반납
손절"(방향 맞는데 TP 전 되돌림→SL)을 본전 청산으로 바꿔 net 을 올리나. 파트너 실거래
관찰(2026-06-17). 정합 BASE(t6/s3) BTC·ETH. be 는 청산 로직이라 timeline 1개 재생.

사용: PYTHONPATH=src python scripts/be_sweep_5y.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import fmt, run_parallel  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT"]

BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, sl_liq_cap=True,
    entry_ttl_bars=6, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True,
)

VARIANTS = {
    "base(be_off)":   {},
    "be_0.3":         dict(be_trigger=0.3),
    "be_0.5":         dict(be_trigger=0.5),
    "be_0.7":         dict(be_trigger=0.7),
    "be_1.0":         dict(be_trigger=1.0),
    "be_1.5":         dict(be_trigger=1.5),
    "be_0.5_lock0.2": dict(be_trigger=0.5, be_lock=0.2),
    "be_1.0_lock0.5": dict(be_trigger=1.0, be_lock=0.5),
}


def main() -> int:
    totals, per_sym = run_parallel(PAIRS, BASE, VARIANTS, nproc=6)
    txt = fmt("브레이크이븐 sweep 5년 (cisd+po3 위, BTC·ETH)", totals, per_sym, PAIRS,
              base_key="base(be_off)")
    with open("be_sweep_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 be_sweep_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
