"""유동적 ttl robust 검증 — 변동성 분위 × ttl 패턴이 전·후반 일관인지.

atr_ttl_search 발견(high vol→ttl6, low vol→ttl12)이 5년 우연인지 확인.
각 페어 전반/후반 분리 → 구간별 변동성 분위 × ttl net. 두 구간 모두 같은
패턴(high vol 에서 ttl6 우세 등)이면 robust → 동적 ttl 구현 가치.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/atr_ttl_robust.py
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
    """한 페어: 전반/후반 각각 timeline+ttl재생 → (seg, ttl, atr, net) 리스트."""
    df5 = _resample(_load_full(sym))
    if len(df5) < 1400:
        return []
    half = len(df5) // 2
    out = []
    for seg_label, seg in (("전반", df5.iloc[:half]), ("후반", df5.iloc[half:])):
        tl = build_setup_timeline(seg, BacktestConfig(**BASE))
        for ttl in TTLS:
            bt = run_backtest_from_timeline(seg, tl, BacktestConfig(**{**BASE, "entry_ttl_bars": ttl}))
            for t in bt.trades:
                if t.entry_atr_pct > 0:
                    out.append((seg_label, ttl, t.entry_atr_pct, t.net_pnl_pct))
    return out


def main() -> int:
    with Pool(min(6, len(PAIRS))) as p:
        results = p.map(_pair_trades, PAIRS)
    rows = [r for sub in results for r in sub]
    if not rows:
        print("거래 없음")
        return 1

    lines = ["===== 유동적 ttl robust (전·후반 각 변동성 분위 × ttl) ====="]
    verdict_best = {}
    for seg_label in ("전반", "후반"):
        seg_rows = [(t, a, n) for s, t, a, n in rows if s == seg_label]
        atrs = sorted(a for _, a, _ in seg_rows)
        if len(atrs) < 9:
            continue
        q33 = atrs[len(atrs) // 3]
        q66 = atrs[2 * len(atrs) // 3]

        def bucket(a, q33=q33, q66=q66):
            return "low " if a < q33 else ("high" if a >= q66 else "mid ")

        agg: dict[tuple, list] = {}
        for ttl, a, net in seg_rows:
            key = (bucket(a), ttl)
            agg.setdefault(key, [0.0, 0])
            agg[key][0] += net
            agg[key][1] += 1
        lines.append(f"\n--- {seg_label} (경계 low<{q33:.3f}/high>={q66:.3f}, {len(seg_rows)}거래) ---")
        for b in ("low ", "mid ", "high"):
            cells = []
            best_ttl, best_net = None, -1e9
            for ttl in TTLS:
                net, n = agg.get((b, ttl), [0.0, 0])
                cells.append(f"ttl{ttl}:{net:+6.2f}(n{n})")
                if net > best_net:
                    best_net, best_ttl = net, ttl
            verdict_best[(seg_label, b.strip())] = best_ttl
            lines.append(f"  {b}  {'  '.join(cells)}  →ttl{best_ttl}")

    # robust 판정 — 각 분위 최적 ttl 이 전후반 동일?
    lines.append("\n===== robust 판정 (분위별 전·후반 최적 ttl 일치?) =====")
    robust_all = True
    for b in ("low", "mid", "high"):
        f = verdict_best.get(("전반", b))
        s = verdict_best.get(("후반", b))
        ok = (f == s)
        robust_all = robust_all and ok
        lines.append(f"  {b:4s}: 전반 ttl{f} / 후반 ttl{s}  {'✅일치' if ok else '❌불일치'}")
    lines.append(f"\n종합: {'✅ robust — 동적 ttl 구현 가치' if robust_all else '⚠ 일부 불일치 — 신중'}")

    txt = "\n".join(lines)
    with open("atr_ttl_robust_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 atr_ttl_robust_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
