"""①번 정합 비교 — 백테 "진입된 거래"의 entry_trend q33 (라이브 역산과 같은 기준).

라이브 역산(진입거래 q33)이 백테 하드코딩(후보전체 q33)보다 컸는데, 모집단이 달라
(진입거래 vs 후보전체) 직접 비교 부적절. 백테에서도 "진입된 거래"의 entry_trend q33
을 뽑아 라이브와 사과 대 사과 비교 → 라이브 변동성이 정말 백테보다 큰지(=하드코딩
상향보정 필요) 판정. 라이브 Origo 1.1 설정(0.5 CE) 기준.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "HYPEUSDT", "XRPUSDT", "SOLUSDT", "LINKUSDT"]
# 라이브 역산 q33 (진입거래 기준, reverse_entry_trend.py 결과)
LIVE_Q33 = {"BTCUSDT": 0.463, "ETHUSDT": 0.903, "DOGEUSDT": 0.412, "HYPEUSDT": 1.749,
            "XRPUSDT": 0.202, "SOLUSDT": 1.433, "LINKUSDT": 0.588}
HARD = {"BTCUSDT": 0.230, "ETHUSDT": 0.268, "SOLUSDT": 0.396, "XRPUSDT": 0.271,
        "DOGEUSDT": 0.275, "LINKUSDT": 0.315, "HYPEUSDT": 0.527}
# 라이브 Origo 1.1 = 0.5 CE 진입 (CT-SL+OTE+cisd+po3, align)
# ote_level=0.707 (오늘 만든 캐시 재사용). entry_trend 는 진입 *시점* 추세라
# 0.5 vs 0.707 진입 깊이 차이와 거의 무관 → 라이브(0.5) 비교에 충분히 근사.
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True, ote_level=0.707,
    min_confluence=4, min_rr=2.0, sl_dist_mult=3.0, setup_stale_bars=3, entry_ttl_bars=6,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def main() -> int:
    lines = ["===== ①번 정합 비교: 백테 진입거래 entry_trend q33 vs 라이브 =====",
             f"{'페어':<10} {'백테표본':>7} {'백테q33':>8} {'라이브q33':>9} {'라이브/백테':>10} {'하드코딩':>8}"]
    ratios = []
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        print(f"  {sym} timeline 로드...", flush=True)
        tl = cached_setup_timeline(df5, BacktestConfig(**BASE), sym)
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**BASE))
        print(f"  {sym} done ({len(bt.trades)} trades)", flush=True)
        trends = sorted(abs(t.entry_trend_pct) for t in bt.trades)
        if len(trends) < 9:
            lines.append(f"{sym:<10} {len(trends):7d}  (표본부족)")
            continue
        bt_q33 = trends[len(trends) // 3]
        live_q33 = LIVE_Q33[sym]
        ratio = live_q33 / bt_q33 if bt_q33 > 0 else 0
        ratios.append(ratio)
        lines.append(f"{sym:<10} {len(trends):7d} {bt_q33:8.3f} {live_q33:9.3f} {ratio:9.2f}x {HARD[sym]:8.3f}")

    if ratios:
        avg = sum(ratios) / len(ratios)
        med = sorted(ratios)[len(ratios) // 2]
        lines.append(f"\n라이브/백테 배율: 평균 {avg:.2f}x  중앙값 {med:.2f}x")
        lines.append(f"※ 배율>1 이면 라이브 진입 변동성이 백테보다 큼 → 하드코딩 floor 를 ×{med:.2f} 보정 후보.")
        lines.append(f"  배율≈1 이면 라이브 역산 차이는 편향(진입거래 모집단) → 롤링 자연학습으로 충분.")

    txt = "\n".join(lines)
    with open("backtest_entry_trend_q33_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print(txt + "\nDONE")
    except UnicodeEncodeError:
        print("(결과는 backtest_entry_trend_q33_result.txt)\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
