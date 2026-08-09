"""BTC ttl 30분 vs 1시간 과적합 체크 — 전반/후반 기간 분할 robust 검증 (병렬판).

파트너 제안(2026-06-17): BTC만 ttl 1h + 나머지 30분 = +12.0%(전체 30분 +9.54 대비 ↑).
BUT BTC 1h 가 5년 전체 우연인지(trail·ttl1h 처럼 BTC 단독 함정) 확인 필요.
전반/후반 둘 다 1h 가 30분 이기면 robust → 페어별 ttl 채택. 한쪽만이면 우연.

6 백테스트(3구간 × 2 ttl)를 6코어 병렬 — 단일코어 순차(31분+)를 몇 분으로.
사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/btc_ttl_robust.py
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402

from aurora_ict.backtest.replay import (  # noqa: E402
    BacktestConfig,
    build_setup_timeline,
    run_backtest_from_timeline,
)

BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0,
    setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False,
    size_pct=0.9,
)


def _worker(payload):
    """한 (구간, ttl) 백테스트. (multiprocessing top-level)"""
    seg_label, ttl = payload
    df5 = _resample(_load_full("BTCUSDT"))
    n = len(df5)
    half = n // 2
    seg = {"전체": df5, "전반": df5.iloc[:half], "후반": df5.iloc[half:]}[seg_label]
    cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl})
    tl = build_setup_timeline(seg, cfg)
    bt = run_backtest_from_timeline(seg, tl, cfg)
    return (seg_label, ttl, bt.n_trades, bt.total_net_pnl_pct)


def main() -> int:
    # "전체"(5년 1h) 재생이 병목이라 제외 — robust 판정은 전반/후반만으로 충분.
    payloads = [(s, t) for s in ("전반", "후반") for t in (6, 12)]
    with Pool(6) as p:
        results = p.map(_worker, payloads)

    by_seg: dict[str, dict[int, tuple[int, float]]] = {}
    for sl, ttl, n, net in results:
        by_seg.setdefault(sl, {})[ttl] = (n, net)

    lines = ["===== BTC ttl 30분 vs 1h 과적합 체크 (기간 분할) ====="]
    lines.append("  구간   ttl30m              ttl1h               승자")
    robust_count = 0
    for sl in ("전반", "후반"):
        n30, net30 = by_seg[sl][6]
        n60, net60 = by_seg[sl][12]
        winner = "1h" if net60 > net30 else "30m"
        if sl in ("전반", "후반") and winner == "1h":
            robust_count += 1
        lines.append(
            f"  {sl:5s} n={n30:4d} {net30:+7.2f}%   n={n60:4d} {net60:+7.2f}%   {winner}"
        )
    verdict = ("✅ robust (전·후반 둘 다 1h) → BTC 페어별 1h 채택"
               if robust_count == 2 else
               "❌ 우연 (한쪽만 1h) → BTC 도 전체 30분 안전")
    lines.append(f"\n판정: {verdict}")

    txt = "\n".join(lines)
    with open("btc_ttl_robust_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 btc_ttl_robust_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
