"""빈도 연구 — 라이브 0.4회/일을 목표 2~4회로. net 유지하며 빈도 끌어올릴 레버 탐색.

가장 큰 빈도 병목은 킬존(Silver Bullet 시간대만 진입)으로 추정. 24h 로 풀면 진입
기회 급증하나 net 희생 가능 → 그 트레이드오프가 핵심. 부 레버로 setup_stale(신선도)
완화·min_confluence 완화도 같이. cisd+po3·ttl6(BTC12)·sl x3·rr2.5·size0.9 고정.

  시간필터: KZ(킬존만) vs 24h(disable_time_filter)  ← detect 인자, timeline 2개
  stale: 3(신선) vs 6(완화)   conf: 4(현행) vs 3(완화)   ← 둘 다 재생 단계

각 조합 7페어 합산 net·승률·1일빈도. 목표 2~4회 달성하며 net 최대 조합 추천.
빌드 느려 순차(메모리 안전) + 진행로그. timeline 은 가장 느슨(conf3·stale6)으로
빌드 후 재생서 조임(gate_grid 와 동일 기법).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/freq_research.py
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
    min_rr=2.5, sl_dist_mult=3.0, apply_cisd=True, apply_po3=True, size_pct=0.9,
)
TIMES = [("KZ", False), ("24h", True)]  # disable_time_filter
STALES = [3, 6]
CONFS = [4, 3]


def _pair(sym):
    """한 페어: KZ/24h 각 timeline(느슨) 1회 → stale×conf 재생 → {(t,stale,conf):(net,nwin,n)}, days."""
    df5 = _resample(_load_full(sym))
    if len(df5) < 1400:
        return (sym, {}, 1.0)
    days = len(df5) / 288.0
    ttl = 12 if sym == "BTCUSDT" else 6
    base_cfg = {**BASE, "entry_ttl_bars": ttl}
    out = {}
    for tlabel, dtf in TIMES:
        # 느슨(conf3·stale6) timeline 1회 → 재생서 conf/stale 조임.
        tl = build_setup_timeline(df5, BacktestConfig(**{
            **base_cfg, "disable_time_filter": dtf, "min_confluence": 3, "setup_stale_bars": 6,
        }))
        for stale in STALES:
            for conf in CONFS:
                bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**{
                    **base_cfg, "disable_time_filter": dtf,
                    "setup_stale_bars": stale, "min_confluence": conf,
                }))
                net = sum(t.net_pnl_pct for t in bt.trades)
                nwin = sum(1 for t in bt.trades if t.net_pnl_pct > 0)
                out[(tlabel, stale, conf)] = (net, nwin, len(bt.trades))
    return (sym, out, days)


def main() -> int:
    rows = []
    for sym in PAIRS:
        rows.append(_pair(sym))
        print(f"  [{len(rows)}/{len(PAIRS)}] {sym} done", flush=True)

    total_days = sum(d for _, o, d in rows if o)
    agg = {}
    for _, out, _ in rows:
        for key, (net, nwin, n) in out.items():
            a = agg.setdefault(key, [0.0, 0, 0])
            a[0] += net; a[1] += nwin; a[2] += n

    lines = ["===== 빈도 연구 (시간필터 × stale × conf, 7페어 5년 cisd+po3) ====="]
    lines.append(f"기간 7페어 합 {total_days:.0f}일. 빈도 = 7페어 합 1일 거래수. 목표 2~4회.")
    lines.append(f"\n{'조합':<18} {'net%':>8} {'승률':>6} {'1일빈도':>8} {'거래':>6}")
    band = []
    for tlabel, _ in TIMES:
        for stale in STALES:
            for conf in CONFS:
                key = (tlabel, stale, conf)
                net, nwin, n = agg[key]
                wr = (nwin / n * 100) if n else 0.0
                freq = n / total_days if total_days else 0.0
                tag = " ✅2~4" if 2.0 <= freq <= 4.0 else ("  ~2" if freq < 2.0 else " >4")
                lines.append(f"{tlabel}/stale{stale}/conf{conf:<4} {net:+8.1f} {wr:5.1f}% {freq:7.2f}회 {n:5d}{tag}")
                band.append((key, net, wr, freq, n))

    lines.append("\n[빈도 2~4회 달성 조합 중 net 최대]")
    in_band = [b for b in band if 2.0 <= b[3] <= 4.0]
    if in_band:
        best = max(in_band, key=lambda b: b[1])
        lines.append(f"  {best[0]} → net{best[1]:+.1f} 승률{best[2]:.1f}% {best[3]:.2f}회")
    else:
        lines.append("  (2~4회 달성 조합 없음 — 가장 빈도 높은 조합:)")
        top = max(band, key=lambda b: b[3])
        lines.append(f"  {top[0]} → net{top[1]:+.1f} 승률{top[2]:.1f}% {top[3]:.2f}회")
    lines.append("\n※ KZ(킬존) net 유지하며 24h 빈도↑면 시간필터 완화 검토. net 급락이면 킬존 유지.")

    txt = "\n".join(lines)
    with open("freq_research_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 freq_research_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
