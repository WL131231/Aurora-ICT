"""짜잘 수익 로직 연구 — TP 거리(tp_rr_override)별 net·승률·평균손익 트레이드오프.

현재 Origo 1.1 = swing TP(먼 TP, RR 4:1, 승률 낮음, 가끔 대박). 라이브 진단:
대박이 드물어(빈도 0.4회/일) 입장료(작은 SL 다발) 못 메움. 반대 가설 = TP 를
가깝게(risk 의 1.0~2.5배) 강제해 승률↑·작은 수익 자주 → 변동성↓·안정적 흑자?
tp_rr_override 는 재생 단계(timeline 재사용). 7페어 5년 Origo 1.1 게이트.

  tp_rr=0(현행 swing) / 1.0 / 1.5 / 2.0 / 2.5 / 3.0
각 net·승률·평균손익·최대단일이익(대박 의존도)·빈도. 승률↑면서 net 유지하면
'짜잘 수익형'(Origo 2 후보) 가치. cisd+po3·conf4·ttl6·sl x3·size0.9.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/scalp_tp_research.py
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
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)
TP_RRS = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]  # 0 = 현행 swing TP


def _pair(sym):
    df5 = _resample(_load_full(sym))
    if len(df5) < 1400:
        return (sym, {}, 1.0)
    days = len(df5) / 288.0
    ttl = 12 if sym == "BTCUSDT" else 6
    base_cfg = {**BASE, "entry_ttl_bars": ttl}
    tl = build_setup_timeline(df5, BacktestConfig(**base_cfg))
    out = {}
    for tp in TP_RRS:
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**{**base_cfg, "tp_rr_override": tp}))
        nets = [t.net_pnl_pct for t in bt.trades]
        n = len(nets)
        net = sum(nets)
        nwin = sum(1 for x in nets if x > 0)
        mx = max(nets) if nets else 0.0
        out[tp] = (net, nwin, n, mx)
    return (sym, out, days)


def main() -> int:
    rows = []
    for sym in PAIRS:
        rows.append(_pair(sym))
        print(f"  [{len(rows)}/{len(PAIRS)}] {sym} done", flush=True)
    total_days = sum(d for _, o, d in rows if o)
    agg = {tp: [0.0, 0, 0, 0.0] for tp in TP_RRS}
    for _, out, _ in rows:
        for tp, (net, nwin, n, mx) in out.items():
            a = agg[tp]
            a[0] += net; a[1] += nwin; a[2] += n; a[3] = max(a[3], mx)

    lines = ["===== 짜잘 수익 연구 (TP 거리별, 7페어 5년 Origo 1.1) ====="]
    lines.append(f"{'tp_rr':<10} {'net%':>8} {'승률':>6} {'거래':>6} {'평균손익':>8} {'최대단일':>8} {'1일빈도':>8}")
    for tp in TP_RRS:
        net, nwin, n, mx = agg[tp]
        wr = (nwin / n * 100) if n else 0.0
        avg = net / n if n else 0.0
        freq = n / total_days * 7 if total_days else 0.0  # 7페어 합/일
        label = "swing(현행)" if tp == 0 else f"x{tp}"
        lines.append(f"{label:<10} {net:+8.1f} {wr:5.1f}% {n:5d} {avg:+8.3f} {mx:+8.1f} {freq:7.2f}회")

    lines.append("\n※ tp_rr 낮을수록 TP 가까워 승률↑·평균손익↓·최대단일↓(대박 의존↓).")
    lines.append("  net 유지하며 승률 크게↑ + 변동성(최대단일)↓ 면 '짜잘 수익형'(Origo 2 후보) 가치.")
    # 현행 대비 net 보존율 + 승률 개선
    base_net = agg[0.0][0]
    base_wr = (agg[0.0][1] / agg[0.0][2] * 100) if agg[0.0][2] else 0.0
    lines.append(f"\n[현행(swing) 대비]")
    for tp in TP_RRS:
        if tp == 0.0:
            continue
        net, nwin, n, mx = agg[tp]
        wr = (nwin / n * 100) if n else 0.0
        lines.append(f"  x{tp}: net {net - base_net:+.1f}%p, 승률 {wr - base_wr:+.1f}%p, 최대단일 {mx:.0f}(현행 {agg[0.0][3]:.0f})")

    txt = "\n".join(lines)
    with open("scalp_tp_research_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 scalp_tp_research_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
