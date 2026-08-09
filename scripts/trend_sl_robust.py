"""국면 기반 SL/TP 거리 — 방향정합 추세 분위 × sl_dist_mult, net + 승률.

HYPE 인사이트(6/17): 신고가 순추세 진입인데 sl_dist_mult 3.0 고정이라 TP 가
16.7% 로 멀었음. 순추세장은 먼 TP 가 닿지만 횡보장은 안 닿아 ttl 만료/SL →
승률 깎임. 국면별로 TP 거리(=sl_dist_mult) 동적화하면 승률↑ 가설 검증.

sl_dist_mult 는 timeline 재생 단계 적용(replay 971)이라 timeline 재사용 가능.
각 페어 전·후반 분리 → signed_trend(=trend×방향) 분위(역/횡/순) × mult →
(net, 승률). 전후반 최적 mult 일치하면 robust → 국면별 동적 SL/TP 가치.
cisd+po3, ttl6(7페어 best), size 0.9.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/trend_sl_robust.py
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
MULTS = [2.0, 3.0, 4.0]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, setup_stale_bars=3, entry_ttl_bars=6,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def _pair_trades(sym):
    """한 페어: 전·후반 timeline 1회 → mult별 재생 → (seg, mult, signed, net, win)."""
    df5 = _resample(_load_full(sym))
    if len(df5) < 1400:
        return []
    half = len(df5) // 2
    out = []
    for seg_label, seg in (("전반", df5.iloc[:half]), ("후반", df5.iloc[half:])):
        tl = build_setup_timeline(seg, BacktestConfig(**BASE))
        for mult in MULTS:
            bt = run_backtest_from_timeline(seg, tl, BacktestConfig(**{**BASE, "sl_dist_mult": mult}))
            for t in bt.trades:
                sgn = 1.0 if t.direction == "long" else -1.0
                out.append((seg_label, mult, t.entry_trend_pct * sgn, t.net_pnl_pct, 1 if t.net_pnl_pct > 0 else 0))
    return out


def main() -> int:
    with Pool(min(6, len(PAIRS))) as p:
        results = p.map(_pair_trades, PAIRS)
    rows = [r for sub in results for r in sub]
    if not rows:
        print("거래 없음")
        return 1

    lines = ["===== 국면 기반 SL/TP (방향정합 추세 분위 × sl_dist_mult, net+승률) ====="]
    best = {}
    for seg_label in ("전반", "후반"):
        seg_rows = [(m, s, n, w) for sl, m, s, n, w in rows if sl == seg_label]
        signs = sorted(s for _, s, _, _ in seg_rows)
        if len(signs) < 9:
            continue
        q33 = signs[len(signs) // 3]
        q66 = signs[2 * len(signs) // 3]

        def bucket(s, q33=q33, q66=q66):
            return "역추세" if s < q33 else ("순추세" if s >= q66 else "횡보 ")

        agg = {}
        for mult, s, net, win in seg_rows:
            key = (bucket(s), mult)
            agg.setdefault(key, [0.0, 0, 0])
            agg[key][0] += net
            agg[key][1] += 1
            agg[key][2] += win
        lines.append(f"\n--- {seg_label} (경계 역<{q33:+.2f}/순>={q66:+.2f}, {len(seg_rows)//len(MULTS)}거래/mult) ---")
        for b in ("역추세", "횡보 ", "순추세"):
            cells = []
            bt_mult, bt_net = None, -1e9
            for mult in MULTS:
                net, n, win = agg.get((b, mult), [0.0, 0, 0])
                wr = (win / n * 100) if n else 0.0
                cells.append(f"x{mult}:{net:+6.2f}/{wr:4.1f}%(n{n})")
                if net > bt_net:
                    bt_net, bt_mult = net, mult
            best[(seg_label, b.strip())] = bt_mult
            lines.append(f"  {b}  {'  '.join(cells)}  →x{bt_mult}")

    lines.append("\n===== robust 판정 (국면별 전·후반 최적 mult 일치?) =====")
    all_ok = True
    for b in ("역추세", "횡보", "순추세"):
        f = best.get(("전반", b))
        s = best.get(("후반", b))
        ok = (f == s)
        all_ok = all_ok and ok
        lines.append(f"  {b}: 전반 x{f} / 후반 x{s}  {'✅' if ok else '❌'}")
    lines.append(f"\n종합: {'✅ robust — 국면별 동적 SL/TP 가치' if all_ok else '⚠ 일부 불일치'}")

    txt = "\n".join(lines)
    with open("trend_sl_robust_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 trend_sl_robust_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
