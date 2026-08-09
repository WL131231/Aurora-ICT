"""유동적 ttl 연구 1단계 — 진입 변동성(ATR) 분위별 최적 ttl 탐색.

가설: 변동성 클 때 짧은 ttl(빨리 포기) vs 작을 때 긴 ttl(되돌림 대기) — 어느 쪽이
유리한가? 각 거래의 진입 ATR%(Trade.entry_atr_pct)를 low/mid/high 3분위로 나눠
ttl 6(30분)/12(1h)/18(90분)별 net 을 집계 → 변동성 구간별 최적 ttl.

robust 하면(전후반 일관) 동적 ttl(진입 시 변동성 보고 ttl 선택) 적용 가치.
cisd+po3, size 0.9. 7페어 (절대 ATR% 분위 — 페어 혼합).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/atr_ttl_search.py
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
TTLS = [6, 12, 18]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def _pair_trades(sym):
    """한 페어: timeline 1회 + ttl 별 재생 → (ttl, entry_atr, net) 리스트."""
    df5 = _resample(_load_full(sym))
    if len(df5) < 700:
        return []
    tl = build_setup_timeline(df5, BacktestConfig(**BASE))
    out = []
    for ttl in TTLS:
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**{**BASE, "entry_ttl_bars": ttl}))
        for t in bt.trades:
            if t.entry_atr_pct > 0:
                out.append((ttl, t.entry_atr_pct, t.net_pnl_pct))
    return out


def main() -> int:
    with Pool(min(6, len(PAIRS))) as p:
        results = p.map(_pair_trades, PAIRS)
    rows = [r for sub in results for r in sub]
    if not rows:
        print("거래 없음")
        return 1

    # 변동성 분위 (전체 entry_atr 33/66%)
    atrs = sorted(a for _, a, _ in rows)
    q33 = atrs[len(atrs) // 3]
    q66 = atrs[2 * len(atrs) // 3]

    def bucket(a):
        return "low " if a < q33 else ("high" if a >= q66 else "mid ")

    agg: dict[tuple, list] = {}
    for ttl, a, net in rows:
        key = (bucket(a), ttl)
        agg.setdefault(key, [0.0, 0])
        agg[key][0] += net
        agg[key][1] += 1

    lines = ["===== 진입 변동성(ATR%) 분위 × ttl net (cisd+po3 7페어) ====="]
    lines.append(f"분위 경계: low<{q33:.3f}% / mid / high>={q66:.3f}%  (총 {len(rows)}거래)")
    lines.append("  분위  ttl6_30m         ttl12_1h         ttl18_90m       최적")
    for b in ("low ", "mid ", "high"):
        cells = []
        best_ttl, best_net = None, -1e9
        for ttl in TTLS:
            net, n = agg.get((b, ttl), [0.0, 0])
            cells.append(f"n={n:4d} {net:+7.2f}%")
            if net > best_net:
                best_net, best_ttl = net, ttl
        lines.append(f"  {b}  {'  '.join(cells)}   →ttl{best_ttl}")
    lines.append("\n※ 분위마다 최적 ttl 이 다르면(예 high=ttl6, low=ttl18) → 유동적 ttl 근거.")

    txt = "\n".join(lines)
    with open("atr_ttl_search_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 atr_ttl_search_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
