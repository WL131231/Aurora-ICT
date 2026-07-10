"""상승 정복 3차 — 국면별 OTE/BE 조합의 스플라이스 강건성 (2026-07-10).

2차: 조건부(상승만 0.786) +3006 확정. 추가 후보(버킷 스플라이스 추정):
    - 상승 = 0.786 + BE1.5 (전역 0.786+BE1.5 의 상승 버킷 +326)
    - 횡보 = 0.886 (전역 0.886 의 횡보 버킷 +1402)
스플라이스 근사 유효성 판정 위해 4개 전역 구성의 **국면×반기×페어** 분해를 뽑아
각 후보 버킷의 반기 일관성·페어 일관성을 본다.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/up_rescue3.py
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
CONFIGS = [
    ("0.707", dict(ote_level=0.707)),
    ("0.786", dict(ote_level=0.786)),
    ("0.886", dict(ote_level=0.886)),
    ("0.786+BE1.5", dict(ote_level=0.786, be_trigger=1.5)),
]


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    days_idx, labels = classify_days(df5)
    day_of = days_idx.searchsorted(df5.index.normalize(), side="right") - 1
    half = len(df5) // 2
    out = {}
    for label, ov in CONFIGS:
        cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl, **ov})
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        # 국면 → [전반, 후반]
        b = {r: [0.0, 0.0] for r in REGIMES}
        for t in bt.trades:
            di = day_of[t.entry_idx]
            lab = labels[di] if di >= 0 else "횡보"
            b[lab][0 if t.entry_idx < half else 1] += t.net_pnl_pct
        out[label] = b
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    lines = ["===== 3차 — 국면×반기×페어 분해 (시드% 단리) =====", ""]
    # 후보 버킷 강건성 표
    for cfg_label, regime in [("0.786", "상승"), ("0.786+BE1.5", "상승"),
                              ("0.886", "횡보"), ("0.707", "상승"), ("0.707", "횡보")]:
        h1 = h2 = 0.0
        neg = 0
        rows = []
        for sym, out in results:
            v = out[cfg_label][regime]
            h1 += v[0]
            h2 += v[1]
            tot = v[0] + v[1]
            neg += tot < 0
            rows.append(f"{sym.replace('USDT',''):<6}{tot * 100:+.0f}%")
        lines.append(f"[{cfg_label} × {regime}] 전/후반 {h1 * 100:+.0f}/{h2 * 100:+.0f}% "
                     f"· 음(-) 페어 {neg}/7 · " + " ".join(rows))
    # 스플라이스 조합 합계 (기준 타국면 + 후보 버킷)
    def bucket_total(cfg_label, regime):
        return sum(sum(out[cfg_label][regime]) for _s, out in results)

    base_total = sum(bucket_total("0.707", r) for r in REGIMES)
    combo1 = base_total - bucket_total("0.707", "상승") + bucket_total("0.786", "상승")
    combo2 = base_total - bucket_total("0.707", "상승") + bucket_total("0.786+BE1.5", "상승")
    combo3 = combo2 - bucket_total("0.707", "횡보") + bucket_total("0.886", "횡보")
    lines.append("")
    lines.append(f"기준(0.707 전역)                    {base_total * 100:+.0f}%")
    lines.append(f"C1 상승만 0.786                    {combo1 * 100:+.0f}%")
    lines.append(f"C2 상승만 0.786+BE1.5              {combo2 * 100:+.0f}%")
    lines.append(f"C3 = C2 + 횡보만 0.886             {combo3 * 100:+.0f}%")
    txt = "\n".join(lines)
    with open("up_rescue3_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
