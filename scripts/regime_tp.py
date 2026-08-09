"""횡보장 대처 연구 1차 — BTC 거래를 국면(|추세|)별로 나눠 TP별 net·승률.

파트너(6/23): 횡보장 대처. 전체 net흑자+승률은 트레이드오프지만, 국면별로 보면
횡보장(|entry_trend_pct| 작음)에선 작은 TP(mean reversion)가, 추세장에선 swing이
나을 수 있음 → 국면 적응형 TP. BTC 0.707 timeline 캐시 재사용. 거래의
entry_trend_pct(진입직전 20봉 변화율) abs 분위로 횡보/중간/추세 분리.

  ote 0.707 × tp[swing/1.0/1.5/2.0] → 국면별(횡보<q33 / 중간 / 추세>q66) net·승률.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/regime_tp.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

SYM = "BTCUSDT"
OTE = 0.707
MIN_RR = 2.0
TP_RRS = [0.0, 1.0, 1.5, 2.0]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3, entry_ttl_bars=12,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def main() -> int:
    df5 = _resample(_load_full(SYM))
    detect_cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR}
    tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), SYM)
    print("timeline ready", flush=True)
    # tp별 trade 수집 (entry_trend_pct 포함)
    by_tp = {}
    for tp in TP_RRS:
        cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR, "tp_rr_override": tp}
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
        by_tp[tp] = bt.trades
        print(f"  tp{tp} done ({len(bt.trades)})", flush=True)

    # 국면 경계: swing(tp0) 거래의 |entry_trend_pct| 분위 (진입 동일하므로 기준 통일)
    ref = by_tp[0.0]
    absten = sorted(abs(t.entry_trend_pct) for t in ref)
    if len(absten) < 9:
        print("거래 부족")
        return 1
    q33 = absten[len(absten) // 3]
    q66 = absten[2 * len(absten) // 3]

    def regime(t):
        a = abs(t.entry_trend_pct)
        return "횡보" if a < q33 else ("추세" if a >= q66 else "중간")

    lines = [f"===== 횡보장 연구 (BTC, ote0.707, 국면×TP) =====",
             f"국면 경계: 횡보<{q33:.2f} / 추세>={q66:.2f} (|진입추세%|)"]
    for reg in ("횡보", "중간", "추세"):
        lines.append(f"\n--- {reg} 국면 ---")
        lines.append(f"  {'tp':<8} {'net%':>8} {'승률':>6} {'거래':>5}")
        bn, btp = -1e9, None
        for tp in TP_RRS:
            sub = [t for t in by_tp[tp] if regime(t) == reg]
            if not sub:
                continue
            net = sum(t.net_pnl_pct for t in sub)
            nw = sum(1 for t in sub if t.net_pnl_pct > 0)
            wr = nw / len(sub) * 100
            label = "swing" if tp == 0 else f"x{tp}"
            lines.append(f"  {label:<8} {net:+8.1f} {wr:5.0f}% {len(sub):5d}")
            if net > bn:
                bn, btp = net, tp
        lines.append(f"  → {reg} 최적 TP: {'swing' if btp == 0 else f'x{btp}'} (net{bn:+.1f})")

    lines.append("\n※ 횡보 국면서 작은 TP(x1)가 net·승률 우위면 → 국면적응형 TP(횡보=작은TP/추세=swing) 가치.")

    txt = "\n".join(lines)
    with open("regime_tp_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 regime_tp_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
