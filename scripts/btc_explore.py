"""BTC 1페어 종합 탐색 — 진입깊이(ote) × 짧은단타 TP(tp_rr) × ttl × sl, 캐시 활용.

파트너 방식(6/22): BTC 로 빠르게 탐색 → 좋은 것만 7페어 검증. + 짧은 단타 로직
(작은 TP·짧은 ttl, 기존 tp_rr_override) 연구. timeline 캐시로 ote별 1회 빌드 후
재생 변형(tp_rr/ttl/sl) 즉시. 각 net·승률·손절률·평균손익·최대단일(변동성).

  ote(진입깊이): 0.5(현행) / 0.886(깊은, 타점개선 확인됨)
  tp_rr(TP):     0(swing 현행) / 1.5 / 2.0 / 2.5  ← 작을수록 짧은단타(승률↑·변동성↓)
  ttl:           6(현행 30분) / 3(15분 짧은) / 12(1h)
  sl_dist:       3.0(현행)

ote 는 detect 인자(timeline 캐시), tp_rr/ttl/sl 은 재생(즉시). cisd+po3·conf4·rr2.5.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/btc_explore.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

SYM = "BTCUSDT"
OTES = [0.5, 0.886]
TP_RRS = [0.0, 1.5, 2.0, 2.5]
TTLS = [6, 3, 12]
SL = 3.0
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=SL, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def main() -> int:
    df5 = _resample(_load_full(SYM))
    days = len(df5) / 288.0
    print(f"BTC df5 {len(df5)} ({days:.0f}일)", flush=True)
    results = []  # (ote, tp_rr, ttl, net, wr, slr, avg, mx, n)
    for ote in OTES:
        detect_cfg = {**BASE, "ote_level": ote, "entry_ttl_bars": 6}
        tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), SYM)
        print(f"  ote{ote} timeline ready (setups {sum(1 for v in tl if v)})", flush=True)
        for tp_rr in TP_RRS:
            for ttl in TTLS:
                cfg = {**BASE, "ote_level": ote, "entry_ttl_bars": ttl, "tp_rr_override": tp_rr}
                bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
                nets = [t.net_pnl_pct for t in bt.trades]
                n = len(nets)
                net = sum(nets)
                nw = sum(1 for x in nets if x > 0)
                wr = (nw / n * 100) if n else 0.0
                slr = ((n - nw) / n * 100) if n else 0.0
                avg = net / n if n else 0.0
                mx = max(nets) if nets else 0.0
                results.append((ote, tp_rr, ttl, net, wr, slr, avg, mx, n))

    results.sort(key=lambda r: r[3], reverse=True)  # net 내림
    lines = ["===== BTC 종합 탐색 (ote × tp_rr × ttl, net 순) ====="]
    lines.append(f"{'ote':>5} {'tp_rr':>6} {'ttl':>4} {'net%':>8} {'승률':>6} {'손절':>6} {'평균':>7} {'최대단일':>8} {'거래':>5}")
    for ote, tp_rr, ttl, net, wr, slr, avg, mx, n in results:
        tp_l = "swing" if tp_rr == 0 else f"{tp_rr}"
        lines.append(f"{ote:>5} {tp_l:>6} {ttl:>4} {net:+8.1f} {wr:5.1f}% {slr:5.1f}% {avg:+7.3f} {mx:+8.1f} {n:5d}")

    lines.append("\n[Top 5 net]")
    for r in results[:5]:
        lines.append(f"  ote{r[0]}/tp{r[1] or 'swing'}/ttl{r[2]}: net{r[3]:+.1f} 승{r[4]:.0f}% 손절{r[5]:.0f}% 최대단일{r[7]:.0f}")
    lines.append("\n[승률 Top 5 (안정형 후보)]")
    for r in sorted(results, key=lambda x: x[4], reverse=True)[:5]:
        lines.append(f"  ote{r[0]}/tp{r[1] or 'swing'}/ttl{r[2]}: 승{r[4]:.0f}% net{r[3]:+.1f} 손절{r[5]:.0f}% 최대단일{r[7]:.0f}")
    lines.append("\n※ 7페어 검증 후보: net 상위 + 승률 높고 최대단일(변동성)↓ 조합 선별.")

    txt = "\n".join(lines)
    with open("btc_explore_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 btc_explore_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
