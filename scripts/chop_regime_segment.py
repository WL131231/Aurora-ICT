"""#FST6 2026-07-16: align 게이트 효과의 국면 집중도 — 횡보 vs 추세 구간 분해.

align(방향정합)이 5년 net 은 거의 유지하며 robust↑. 결정 질문: 이 이득이
'횡보 구간 손실 절감'에서 오는가(=지금 파트너 문제 정조준)? 진입 시점
|entry_trend_pct| 삼분위(약추세=횡보 / 중 / 강추세)로 나눠 align 유무 net·승률 비교.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=True, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
)


def main() -> int:
    recs = []  # (|trend|, signed_ok, net)
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        for t in bt.trades:
            h = df5.index[t.entry_idx].hour
            if 17 <= h < 21:  # NY_PM 제외 (1.9 base)
                continue
            sign = 1.0 if t.direction == "long" else -1.0
            recs.append((abs(t.entry_trend_pct), t.entry_trend_pct * sign > 0,
                         t.net_pnl_pct))

    mags = sorted(r[0] for r in recs)
    t1 = np.percentile(mags, 33)
    t2 = np.percentile(mags, 66)

    def bucket(m):
        return "약추세(횡보)" if m < t1 else ("중추세" if m < t2 else "강추세")

    print(f"|trend| 삼분위 경계: q33={t1:.3f}%  q66={t2:.3f}%  (총 {len(recs)}건)\n")
    print(f"{'국면':<14}{'전체net':>9}{'승률':>7}{'건':>6} | "
          f"{'align후net':>10}{'승률':>7}{'건':>6}{'역추세제거net':>13}")
    for b in ["약추세(횡보)", "중추세", "강추세"]:
        sub = [r for r in recs if bucket(r[0]) == b]
        al = [r for r in sub if r[1]]           # align 통과(정합)
        ct = [r for r in sub if not r[1]]       # 역추세(제거 대상)
        def stat(g):
            n = len(g); net = sum(x[2] for x in g)
            w = 100 * sum(1 for x in g if x[2] > 0) / n if n else 0
            return net, w, n
        n0, w0, c0 = stat(sub)
        n1, w1, c1 = stat(al)
        nct, _, _ = stat(ct)
        print(f"{b:<14}{n0:>+9.1f}{w0:>6.0f}%{c0:>6} | "
              f"{n1:>+10.1f}{w1:>6.0f}%{c1:>6}{nct:>+13.1f}")
    print("\n→ 약추세(횡보) 구간의 역추세제거net 이 크게 음수면 = align 이 횡보 손실을 정조준")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
