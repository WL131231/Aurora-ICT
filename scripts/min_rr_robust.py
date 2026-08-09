"""min_rr 완화 robust 검증 — 진입 RR 기준이 너무 엄격(2.5)한지.

gate_grid 1회 결과: conf4에서 rr2.0 net+19.8 vs rr2.5(현행)+9.5 — 2배 신호.
파트너 직관 "진입 기준이 너무 많/엄격해서 좋은 setup 버림". RR 2.0~2.5 구간
setup 이 흑자인데 min_rr 2.5 가 버리는지 전후반 robust 검증. min_rr 은 detect
인자라 timeline 재빌드. cisd+po3·conf4·ttl6(BTC12)·sl x3·킬존·size0.9.

  min_rr: 2.0 / 2.25 / 2.5(현행) / 3.0  × 전·후반
전후반 모두 rr2.0 net 우세면 → min_rr 완화 배포 가치. 빈도·승률도 같이.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/min_rr_robust.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402

from aurora_ict.backtest.replay import (  # noqa: E402
    BacktestConfig,
    build_setup_timeline,
    run_backtest_from_timeline,
)

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
RRS = [2.0, 2.25, 2.5, 3.0]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def _pair(sym):
    df5 = _resample(_load_full(sym))
    if len(df5) < 1400:
        return []
    half = len(df5) // 2
    ttl = 12 if sym == "BTCUSDT" else 6
    out = []
    for seg_label, seg in (("전반", df5.iloc[:half]), ("후반", df5.iloc[half:])):
        for rr in RRS:
            # min_rr 은 detect 인자 → rr 마다 timeline 빌드.
            tl = build_setup_timeline(seg, BacktestConfig(**{**BASE, "min_rr": rr, "entry_ttl_bars": ttl}))
            bt = run_backtest_from_timeline(seg, tl, BacktestConfig(**{**BASE, "min_rr": rr, "entry_ttl_bars": ttl}))
            net = sum(t.net_pnl_pct for t in bt.trades)
            nwin = sum(1 for t in bt.trades if t.net_pnl_pct > 0)
            out.append((seg_label, rr, net, nwin, len(bt.trades)))
    return out


def main() -> int:
    rows = []
    for sym in PAIRS:
        rows.append((sym, _pair(sym)))
        print(f"  [{len(rows)}/{len(PAIRS)}] {sym} done", flush=True)
    flat = [r for _, sub in rows for r in sub]

    lines = ["===== min_rr 완화 robust (전·후반 × min_rr, 7페어) ====="]
    best = {}
    for seg in ("전반", "후반"):
        agg = {}
        for sl, rr, net, nwin, n in flat:
            if sl != seg:
                continue
            a = agg.setdefault(rr, [0.0, 0, 0])
            a[0] += net; a[1] += nwin; a[2] += n
        lines.append(f"\n--- {seg} ---")
        bn, brr = -1e9, None
        for rr in RRS:
            net, nwin, n = agg.get(rr, [0.0, 0, 0])
            wr = (nwin / n * 100) if n else 0.0
            mark = " ←현행" if rr == 2.5 else ""
            lines.append(f"  rr{rr}: net{net:+7.1f} 승{wr:4.1f}% 거래{n:4d}{mark}")
            if net > bn:
                bn, brr = net, rr
        best[seg] = brr
        lines.append(f"  → {seg} 최적 rr{brr}")

    lines.append("\n===== robust 판정 =====")
    ok = best.get("전반") == best.get("후반")
    lines.append(f"  전반 rr{best.get('전반')} / 후반 rr{best.get('후반')}  {'✅ 일치' if ok else '⚠ 불일치'}")
    if ok and best.get("전반") != 2.5:
        lines.append(f"  → min_rr {best['전반']} 가 전후반 robust 우세. 현행 2.5 완화 배포 가치!")
    elif ok:
        lines.append("  → 현행 2.5 가 robust 최적. 유지.")
    else:
        lines.append("  → 불일치 — 신중.")

    txt = "\n".join(lines)
    with open("min_rr_robust_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 min_rr_robust_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
