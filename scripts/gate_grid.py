"""진입 게이트 그리드 — min_confluence × min_rr → net·승률·빈도(1일 거래수).

라이브 문제: 승률 30%, RR 4:1 이라 흑자지만 승률 올리면 견고. 게이트를
조이면 승률·질↑·빈도↓, 풀면 빈도↑·승률↓. 구독제 목표 1일 2~4회를 만족하는
선에서 승률·net 최적 게이트 탐색. min_confluence/min_rr 은 재생 단계 게이트라
timeline 재사용(replay 872). cisd+po3, ttl6, sl x3, size 0.9, 7페어 5년.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/gate_grid.py
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

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
CONFS = [3, 4, 5]
RRS = [2.0, 2.5, 3.0]
# timeline 은 가장 느슨한 게이트(min_confluence=3, min_rr=2.0)로 1회 빌드 후
# 재생 때 더 빡센 게이트로 거른다. (게이트는 detect 가 아닌 재생 단계 필터)
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    sl_dist_mult=3.0, setup_stale_bars=3, entry_ttl_bars=6,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def _pair(sym):
    """한 페어: min_rr 별 timeline(detect 인자) → conf 재생 → (conf,rr,net,n_win,n,days).

    min_rr 은 detect 단계 인자라 rr 마다 timeline 재빌드. min_confluence 는
    재생 단계 게이트라 한 timeline 에서 conf 만 바꿔 재사용.
    """
    df5 = _resample(_load_full(sym))
    if len(df5) < 1400:
        return []
    days = len(df5) / 288.0  # 288 봉/일(5분봉)
    out = []
    for rr in RRS:
        tl = build_setup_timeline(df5, BacktestConfig(**{**BASE, "min_confluence": 3, "min_rr": rr}))
        for conf in CONFS:
            bt = run_backtest_from_timeline(
                df5, tl, BacktestConfig(**{**BASE, "min_confluence": conf, "min_rr": rr})
            )
            net = sum(t.net_pnl_pct for t in bt.trades)
            nwin = sum(1 for t in bt.trades if t.net_pnl_pct > 0)
            out.append((conf, rr, net, nwin, len(bt.trades), days))
    return out


def main() -> int:
    with Pool(min(6, len(PAIRS))) as p:
        results = p.map(_pair, PAIRS)
    rows = [r for sub in results for r in sub]
    if not rows:
        print("거래 없음")
        return 1

    # (conf,rr) 별 7페어 합산
    agg = {}
    total_days = 0.0
    seen_days = set()
    for conf, rr, net, nwin, n, days in rows:
        key = (conf, rr)
        agg.setdefault(key, [0.0, 0, 0])
        agg[key][0] += net
        agg[key][1] += nwin
        agg[key][2] += n
    # 페어별 days 는 동일 조합마다 반복 → 한 조합 기준 7페어 days 합
    first = (CONFS[0], RRS[0])
    total_days = sum(days for conf, rr, _, _, _, days in rows if (conf, rr) == first)

    lines = ["===== 진입 게이트 그리드 (min_confluence × min_rr, 7페어 5년 cisd+po3) ====="]
    lines.append(f"기간 합산 {total_days:.0f}일(7페어). 빈도 = 7페어 합 1일 거래수.")
    lines.append("")
    lines.append("        " + "  ".join(f"   rr{rr}      " for rr in RRS))
    for conf in CONFS:
        cells = []
        for rr in RRS:
            net, nwin, n = agg.get((conf, rr), [0.0, 0, 0])
            wr = (nwin / n * 100) if n else 0.0
            freq = n / total_days if total_days else 0.0
            cells.append(f"{net:+6.1f}/{wr:4.1f}%/{freq:.1f}회")
        lines.append(f"conf{conf}  " + "  ".join(cells))

    lines.append("\n각 셀: net합% / 승률 / 1일빈도(7페어합).  구독제 목표 1일 2~4회.")
    # 빈도 2~4 만족하며 net·승률 최적 추천
    cand = []
    for conf in CONFS:
        for rr in RRS:
            net, nwin, n = agg[(conf, rr)]
            wr = (nwin / n * 100) if n else 0.0
            freq = n / total_days if total_days else 0.0
            cand.append((conf, rr, net, wr, freq))
    in_band = [c for c in cand if 2.0 <= c[4] <= 4.0]
    pool = in_band if in_band else cand
    best_net = max(pool, key=lambda c: c[2])
    best_wr = max(pool, key=lambda c: c[3])
    lines.append(f"\n빈도 2~4회 만족 조합: {len(in_band)}개")
    lines.append(f"  net 최대: conf{best_net[0]}/rr{best_net[1]} → {best_net[2]:+.1f}% 승률{best_net[3]:.1f}% {best_net[4]:.1f}회")
    lines.append(f"  승률 최대: conf{best_wr[0]}/rr{best_wr[1]} → {best_wr[2]:+.1f}% 승률{best_wr[3]:.1f}% {best_wr[4]:.1f}회")

    txt = "\n".join(lines)
    with open("gate_grid_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 gate_grid_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
