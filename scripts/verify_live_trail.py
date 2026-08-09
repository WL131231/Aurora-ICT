"""라이브 구독제 정합(mc4) + cisd+po3 + trail 7페어 — 승률·흑자 전환 검증.

라이브 실측(2026-06-17 trades_all_users.csv 18.9일): 빈도는 충족(활발 사용자 1일 2.7회),
승률 30.4% / 누적 -970 USDT 손실. 손익비는 우수(TP건당+72 vs SL-18, 4:1)나
TP 도달 12%. → trail 로 SL행을 중간익절 전환 시 승률↑+흑자 가능한지가 핵심.

라이브 구독제 정합: mc4 + rr2.5 + sl3.0 + 진입대기 2h(ttl 24봉) + 신선도 30분(stale 6봉).
trail off(base, =현행) 대비 trail 8종 비교. timeline 1개/페어(=가벼움) + 재생.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/verify_live_trail.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import fmt, run_parallel  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]

# 라이브 구독제 정합 (#EDGE-V2 + #FRESH-30)
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0,
    # trail(청산 방식) 효과는 ttl 과 독립 → 정합 BASE(30분/15분)로 고속 재생.
    # ttl 24봉(라이브 2h)은 백테스트 재생을 폭증시켜 47분+ 소요(2026-06-17 확인).
    entry_ttl_bars=6, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False,
    size_pct=0.9,
)

VARIANTS = {
    "base(현행)":  {},
    "t0.5_d0.5":   dict(trail_trigger=0.5, trail_dist=0.5),
    "t0.5_d1.0":   dict(trail_trigger=0.5, trail_dist=1.0),
    "t1.0_d0.5":   dict(trail_trigger=1.0, trail_dist=0.5),
    "t1.0_d1.0":   dict(trail_trigger=1.0, trail_dist=1.0),
    "t1.0_d1.5":   dict(trail_trigger=1.0, trail_dist=1.5),
    "t1.5_d0.5":   dict(trail_trigger=1.5, trail_dist=0.5),
    "t1.5_d1.0":   dict(trail_trigger=1.5, trail_dist=1.0),
    "t1.5_d1.5":   dict(trail_trigger=1.5, trail_dist=1.5),
}


def main() -> int:
    totals, per_sym = run_parallel(PAIRS, BASE, VARIANTS, nproc=6)
    txt = fmt("라이브 정합(mc4)+cisd+po3 +trail 7페어 — 승률·흑자", totals, per_sym,
              PAIRS, base_key="base(현행)")
    with open("verify_live_trail_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 verify_live_trail_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
