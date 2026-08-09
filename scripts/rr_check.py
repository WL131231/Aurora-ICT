"""분할익절 RR 점검 — 고래 RR사수 교훈(②번 RR4.97) 대비 우리 avgWin/avgLoss 변화.

고래 데이터: 흑자 고레버 = RR 비대칭 사수가 생존1순위. 우리 분할익절(1R 50%+be)이
RR(avgWin/avgLoss)을 얼마나 양보하는지 점검 — 과하게 깎으면 고레버 생존공식 이탈.
7페어 0.707 캐시 × partial[off/1.0/1.5] → avgWin/avgLoss/RR/net. 캐시 재생.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
OTE = 0.707
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True, ote_level=OTE,
    min_confluence=4, min_rr=2.0, sl_dist_mult=3.0, setup_stale_bars=3, entry_ttl_bars=6,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)
VARIANTS = [("off(swing)", 0.0, False), ("분할1.0/be", 1.0, True), ("분할1.5/be", 1.5, True)]


def main() -> int:
    # 라벨 -> (win합, win수, loss합, loss수, net)
    agg = {v[0]: [0.0, 0, 0.0, 0, 0.0] for v in VARIANTS}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        tl = cached_setup_timeline(df5, BacktestConfig(**{**BASE, "tp_rr_override": 0.0}), sym)
        for label, ptp, pbe in VARIANTS:
            cfg = {**BASE, "tp_rr_override": 0.0, "partial_tp_rr": ptp, "partial_be": pbe}
            trades = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg)).trades
            a = agg[label]
            for t in trades:
                p = t.net_pnl_pct
                a[4] += p
                if p > 0:
                    a[0] += p; a[1] += 1
                elif p < 0:
                    a[2] += abs(p); a[3] += 1
        print(f"  {sym} done", flush=True)

    lines = ["===== 분할익절 RR 점검 (7페어, 0.707, net_pnl_pct 기준) =====",
             f"{'변형':<12} {'avgWin':>8} {'avgLoss':>8} {'RR':>6} {'net%':>8} {'승률':>6}"]
    for label, _, _ in VARIANTS:
        win_s, win_n, loss_s, loss_n, net = agg[label]
        aw = win_s / win_n if win_n else 0.0
        al = loss_s / loss_n if loss_n else 0.0
        rr = aw / al if al else 0.0
        wr = win_n / (win_n + loss_n) * 100 if (win_n + loss_n) else 0.0
        lines.append(f"{label:<12} {aw:8.3f} {al:8.3f} {rr:6.2f} {net:+8.1f} {wr:5.0f}%")
    lines.append("\n※ 고래 ② RR 4.97(avgWin$1628/avgLoss$327). 분할익절이 RR 너무 깎으면(off 대비"
                 " 큰폭↓) 고레버 생존공식 이탈 → 분할비율 재고. 단 우리는 상용 체감승률 절충.")

    txt = "\n".join(lines)
    with open("rr_check_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE")
    except UnicodeEncodeError:
        print("(결과는 rr_check_result.txt)\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
