"""국면 기반 ttl robust — 진입 방향 정합 추세 분위 × ttl, 전·후반 일관성.

ATR robust 가 부분 통과(high vol→ttl6만)였고 전후반 손익이 통째로 뒤집힌 게
국면(추세) 영향 신호였음. 국면 = 진입 방향 정합 추세:
    signed_trend = entry_trend_pct × (+1 LONG / -1 SHORT)
    순방향(+, 추세 순응) / 역방향(-, 눌림목·역추세) / 횡보(~0)
각 페어 전반/후반 분리 → 국면 분위 × ttl net. 전후반 일치하면 robust →
국면 기반 동적 ttl 구현 가치. cisd+po3, size 0.9.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/trend_ttl_robust.py
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
    """한 페어: 전·후반 각 timeline+ttl재생 → (seg, ttl, signed_trend, net)."""
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
                sgn = 1.0 if t.direction == "long" else -1.0
                signed = t.entry_trend_pct * sgn
                out.append((seg_label, ttl, signed, t.net_pnl_pct))
    return out


def main() -> int:
    with Pool(min(6, len(PAIRS))) as p:
        results = p.map(_pair_trades, PAIRS)
    rows = [r for sub in results for r in sub]
    if not rows:
        print("거래 없음")
        return 1

    lines = ["===== 국면 기반 ttl robust (방향정합 추세 분위 × ttl, 전·후반) ====="]
    best = {}
    for seg_label in ("전반", "후반"):
        seg_rows = [(t, s, n) for sl, t, s, n in rows if sl == seg_label]
        signs = sorted(s for _, s, _ in seg_rows)
        if len(signs) < 9:
            continue
        q33 = signs[len(signs) // 3]
        q66 = signs[2 * len(signs) // 3]

        def bucket(s, q33=q33, q66=q66):
            return "역추세" if s < q33 else ("순추세" if s >= q66 else "횡보 ")

        agg = {}
        for ttl, s, net in seg_rows:
            key = (bucket(s), ttl)
            agg.setdefault(key, [0.0, 0])
            agg[key][0] += net
            agg[key][1] += 1
        lines.append(f"\n--- {seg_label} (경계 역<{q33:+.2f}/순>={q66:+.2f}, {len(seg_rows)}거래) ---")
        for b in ("역추세", "횡보 ", "순추세"):
            cells = []
            bt_ttl, bt_net = None, -1e9
            for ttl in TTLS:
                net, n = agg.get((b, ttl), [0.0, 0])
                cells.append(f"ttl{ttl}:{net:+6.2f}(n{n})")
                if net > bt_net:
                    bt_net, bt_ttl = net, ttl
            best[(seg_label, b.strip())] = bt_ttl
            lines.append(f"  {b}  {'  '.join(cells)}  →ttl{bt_ttl}")

    lines.append("\n===== robust 판정 (국면별 전·후반 최적 ttl 일치?) =====")
    all_ok = True
    for b in ("역추세", "횡보", "순추세"):
        f = best.get(("전반", b))
        s = best.get(("후반", b))
        ok = (f == s)
        all_ok = all_ok and ok
        lines.append(f"  {b}: 전반 ttl{f} / 후반 ttl{s}  {'✅' if ok else '❌'}")
    lines.append(f"\n종합: {'✅ robust — 국면 기반 동적 ttl 가치' if all_ok else '⚠ 일부 불일치'}")

    txt = "\n".join(lines)
    with open("trend_ttl_robust_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 trend_ttl_robust_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
