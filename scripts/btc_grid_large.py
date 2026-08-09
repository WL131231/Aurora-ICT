"""BTC 대규모 자율 탐색 — 피보 진입비율 × 단타TP × ttl × RR게이트 수백 조합.

파트너(6/22): 7페어 보류, BTC 1페어로 수십수백 조합 자율 연구(단타 로직·피보 비율
뭐가 좋은지). timeline 캐시로 detect 변형(ote×min_rr)만 빌드하고 재생 변형
(tp_rr×ttl) 즉시. b1tgc33o3 가 만든 ote0.5/0.886 캐시 재사용.

  ote(피보 진입깊이): 0.382/0.5/0.618/0.707/0.786/0.886/1.0  ← detect
  min_rr(RR게이트):   2.0/2.5                                  ← detect
  tp_rr(단타 TP):     swing(0)/1.0/1.5/2.0/2.5/3.0             ← 재생
  ttl(진입대기):       3/6/12/18                                ← 재생
  → detect 14 timeline(캐시) × 재생 24 = 336 조합. sl x3·conf4·cisd+po3 고정.

각 net·승률·손절률·평균손익·최대단일(변동성). Top(net/승률/안정) 선별 보고.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/btc_grid_large.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

SYM = "BTCUSDT"
OTES = [0.382, 0.5, 0.618, 0.707, 0.786, 0.886, 1.0]
MIN_RRS = [2.0, 2.5]
TP_RRS = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0]
TTLS = [3, 6, 12, 18]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def main() -> int:
    df5 = _resample(_load_full(SYM))
    print(f"BTC df5 {len(df5)} ({len(df5) / 288:.0f}일)", flush=True)
    results = []
    nbuilt = 0
    for ote in OTES:
        for min_rr in MIN_RRS:
            detect_cfg = {**BASE, "ote_level": ote, "min_rr": min_rr, "entry_ttl_bars": 6}
            tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), SYM)
            nbuilt += 1
            print(f"  [{nbuilt}/14] ote{ote} rr{min_rr} timeline ready (setups {sum(1 for v in tl if v)})", flush=True)
            for tp_rr in TP_RRS:
                for ttl in TTLS:
                    cfg = {**BASE, "ote_level": ote, "min_rr": min_rr,
                           "entry_ttl_bars": ttl, "tp_rr_override": tp_rr}
                    bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
                    nets = [t.net_pnl_pct for t in bt.trades]
                    n = len(nets)
                    net = sum(nets)
                    nw = sum(1 for x in nets if x > 0)
                    wr = (nw / n * 100) if n else 0.0
                    slr = ((n - nw) / n * 100) if n else 0.0
                    mx = max(nets) if nets else 0.0
                    results.append((ote, min_rr, tp_rr, ttl, net, wr, slr, mx, n))

    def fmt(r):
        ote, mr, tp, ttl, net, wr, slr, mx, n = r
        tpl = "swing" if tp == 0 else str(tp)
        return f"ote{ote}/rr{mr}/tp{tpl}/ttl{ttl}: net{net:+.1f} 승{wr:.0f}% 손절{slr:.0f}% 최대{mx:.0f} n{n}"

    lines = ["===== BTC 대규모 탐색 (피보×RR×단타TP×ttl, 336조합) ====="]
    lines.append(f"\n[net Top 12]")
    for r in sorted(results, key=lambda x: x[4], reverse=True)[:12]:
        lines.append("  " + fmt(r))
    lines.append(f"\n[승률 Top 12 (안정형 후보 — 변동성↓)]")
    for r in sorted(results, key=lambda x: x[5], reverse=True)[:12]:
        lines.append("  " + fmt(r))
    # net>0 이면서 승률 높은 (흑자+안정)
    pos = [r for r in results if r[4] > 0]
    lines.append(f"\n[흑자({len(pos)}개) 중 승률 Top 12 (흑자+안정 핵심)]")
    for r in sorted(pos, key=lambda x: x[5], reverse=True)[:12]:
        lines.append("  " + fmt(r))
    # 피보 비율별 평균 net
    lines.append("\n[피보 진입비율(ote)별 평균 net]")
    for ote in OTES:
        sub = [r for r in results if r[0] == ote]
        an = sum(r[4] for r in sub) / len(sub)
        aw = sum(r[5] for r in sub) / len(sub)
        lines.append(f"  ote{ote}: 평균net{an:+.1f} 평균승률{aw:.0f}%")

    txt = "\n".join(lines)
    with open("btc_grid_large_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 btc_grid_large_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
