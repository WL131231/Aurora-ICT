"""#FST5 2026-07-16: NY_PM 킬존 게이트 검증 — 5년·페어별 robust.

라이브 진단(진입시각 기준): Origo 1.8 NY_PM(02-05KST) 승률 10%/-29(손실 81%).
백테·6/24 연구도 NY_PM 최악 삼중 일치. 여기선 5년 백테 trade 를 진입시각
킬존으로 분류해 NY_PM 제외가 페어별로 net 개선하는지(robust) 확인한다.

통과: 총 net 개선 + 페어 과반 개선(한두 페어 요행 아님) → Origo 1.9 NY_PM 게이트.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import (  # noqa: E402
    BacktestConfig,
    run_backtest_from_timeline,
)

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
# 라이브 정합 — referral 24h(disable_time_filter=True) 로 전 킬존 진입 재현 후 분류.
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=True, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
)


def is_nypm(h: int) -> bool:
    # NY_PM = 02-05 KST = 17-20 UTC.
    return 17 <= h < 21


def main() -> int:
    print(f"{'페어':<10}{'전체net':>9}{'NY_PM':>9}{'제외net':>9}{'개선':>8}{'NYPM건':>8}")
    tot_all = tot_ex = tot_nypm = 0.0
    improved = 0
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        allnet = nypm = 0.0
        nypm_n = 0
        for t in bt.trades:
            h = df5.index[t.entry_idx].hour
            allnet += t.net_pnl_pct
            if is_nypm(h):
                nypm += t.net_pnl_pct
                nypm_n += 1
        ex = allnet - nypm
        d = ex - allnet
        if d > 0:
            improved += 1
        print(f"{sym:<10}{allnet:>+9.1f}{nypm:>+9.1f}{ex:>+9.1f}{d:>+8.1f}{nypm_n:>8}")
        tot_all += allnet; tot_ex += ex; tot_nypm += nypm
    print("-" * 54)
    print(f"{'합계':<10}{tot_all:>+9.1f}{tot_nypm:>+9.1f}{tot_ex:>+9.1f}{tot_ex-tot_all:>+8.1f}")
    print(f"\n페어 개선: {improved}/{len(PAIRS)}  (과반+총net개선이면 robust)")
    print(f"판정: NY_PM 제외 net {tot_all:+.1f}% → {tot_ex:+.1f}% "
          f"({'개선✓' if tot_ex > tot_all else '악화✗'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
