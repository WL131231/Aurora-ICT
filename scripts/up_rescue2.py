"""상승 정복 2차 — OTE 0.786 전역/조건부 + 0.886 + 결합 (2026-07-10).

1차: ote0.786 이 상승 버킷 유일 흑자전환(+94%) & 전/후반 동시 개선. 후속:
    A. 전역 ote0.786 — 전 국면 분해+합계+반기 (전역이면 배포가 설정 한 줄)
    B. 전역 ote0.886 (더 깊게 — 단조 개선인지 봉우리인지; 타임라인 신규 빌드)
    C. 조건부(상승만 0.786) 혼합 타임라인 — 전역이 타국면을 깎을 경우 대비
    D. 전역 0.786 + BE1.5 (1차 후반 +808 후보 결합)
판정: 합계·반기·타국면 모두 기준 이상이어야 전역 채택.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/up_rescue2.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_edge_lab import REGIMES, classify_days  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    min_rr=2.0, tp_rr_override=0.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    days_idx, labels = classify_days(df5)
    day_of = days_idx.searchsorted(df5.index.normalize(), side="right") - 1
    half = len(df5) // 2

    def lab(i):
        di = day_of[i]
        return labels[di] if di >= 0 else "횡보"

    def buckets(bt):
        out = {r: [0.0, 0] for r in REGIMES}
        halves = [0.0, 0.0]
        for t in bt.trades:
            out[lab(t.entry_idx)][0] += t.net_pnl_pct
            out[lab(t.entry_idx)][1] += 1
            halves[0 if t.entry_idx < half else 1] += t.net_pnl_pct
        return out, halves

    def run_cfg(ote, extra=None):
        cfg = BacktestConfig(**{**BASE, "ote_level": ote, "entry_ttl_bars": ttl,
                                **(extra or {})})
        tl = cached_setup_timeline(df5, cfg, sym)
        return buckets(run_backtest_from_timeline(df5, tl, cfg)), tl, cfg

    out = {}
    (out["기준 0.707"], tl7, cfg7) = run_cfg(0.707)
    (out["전역 0.786"], tl786, _c) = run_cfg(0.786)
    (out["전역 0.886"], _t, _c2) = run_cfg(0.886)
    (out["전역 0.786+BE1.5"], _t2, _c3) = run_cfg(0.786, dict(be_trigger=1.5))
    # 조건부 — 상승일만 0.786, 그 외 0.707 (혼합 타임라인, 재생은 cfg7)
    from aurora_ict.backtest.replay import run_backtest_from_timeline as _run
    mtl = [tl786[i] if lab(i) == "상승" else tl7[i] for i in range(len(tl7))]
    out["조건부(상승만 0.786)"] = buckets(_run(df5, mtl, cfg7))
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    labels = ["기준 0.707", "전역 0.786", "전역 0.886", "전역 0.786+BE1.5",
              "조건부(상승만 0.786)"]
    lines = ["===== OTE 심화 (7페어 5년, 시드% 단리) =====",
             f"{'변형':<22}" + "".join(f"{r:>12}" for r in REGIMES)
             + f"{'합계':>10}{'전/후반':>16}"]
    for label in labels:
        tot = {r: [0.0, 0] for r in REGIMES}
        hh = [0.0, 0.0]
        for _s, out in results:
            b, h = out[label]
            for r in REGIMES:
                tot[r][0] += b[r][0]
                tot[r][1] += b[r][1]
            hh[0] += h[0]
            hh[1] += h[1]
        g = sum(v[0] for v in tot.values())
        seg = "".join(f"{tot[r][0] * 100:>+9.0f}({tot[r][1]:>3})" for r in REGIMES)
        lines.append(f"{label:<22}{seg}{g * 100:>+9.0f}%"
                     f"{hh[0] * 100:>+8.0f}/{hh[1] * 100:+.0f}%")
    txt = "\n".join(lines)
    with open("up_rescue2_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
