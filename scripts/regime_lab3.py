"""국면 랩 3차 — 상승 국면 롱 구제 실험 (2026-07-10).

2차: 상승 국면 롱 자체가 적자(-57%, 90건/29%). 가설 — 강추세에서 OTE 깊은
되돌림 체결 = 추세 꺾임 역선택 + 되돌림 바닥 스탑헌트. 구제 후보:
    SLx5 / SLx6 (스탑헌트 생존), stale 6봉(신선도 완화 — 체결 타이밍 다양화),
    ttl 12봉(대기 연장). 각 full-run 후 '상승 버킷만' 비교 (국면 적응 도입 시
    해당 버킷만 바뀐다는 근사 — 경계 겹침 오차 소폭).
비교 지표: 상승 버킷 net + "조건부 채택 시 합계" (기준 타국면 + 변형 상승).
참고선: 상승 진입 전면 차단 = 기준 합계에서 상승 버킷 제거.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/regime_lab3.py
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
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)
VARIANTS = [
    ("기준(SLx4)", {}),
    ("SLx5", dict(sl_dist_mult=5.0)),
    ("SLx6", dict(sl_dist_mult=6.0)),
    ("stale6", dict(setup_stale_bars=6)),
    ("ttl12", dict(entry_ttl_bars=12)),
]


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    days_idx, labels = classify_days(df5)
    day_of = days_idx.searchsorted(df5.index.normalize(), side="right") - 1

    def buckets(bt):
        out = {r: [0.0, 0, 0] for r in REGIMES}
        for t in bt.trades:
            di = day_of[t.entry_idx]
            if di < 0:
                continue
            b = out[labels[di]]
            b[0] += t.net_pnl_pct
            b[1] += 1
            b[2] += 1 if t.net_pnl_pct > 0 else 0
        return out

    out = {}
    for label, ov in VARIANTS:
        cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl, **ov})
        tl = cached_setup_timeline(df5, cfg, sym)
        out[label] = buckets(run_backtest_from_timeline(df5, tl, cfg))
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)

    def agg(label):
        tot = {r: [0.0, 0, 0] for r in REGIMES}
        for _s, out in results:
            for r in REGIMES:
                b = out[label][r]
                tot[r][0] += b[0]
                tot[r][1] += b[1]
                tot[r][2] += b[2]
        return tot

    base = agg("기준(SLx4)")
    base_others = sum(v[0] for r, v in base.items() if r != "상승")
    lines = ["===== 상승 국면 롱 구제 (7페어 5년, 시드% 단리) =====",
             f"기준 상승 버킷: {base['상승'][0] * 100:+.0f}% ({base['상승'][1]}건)",
             f"참고: 상승 전면 차단 시 합계 = {base_others * 100:+.0f}%",
             "",
             f"{'변형':<14}{'상승버킷':>10}{'건수':>6}{'승률':>6}{'조건부 채택시 합계':>18}"]
    for label, _ in VARIANTS[1:]:
        t = agg(label)["상승"]
        wr = t[2] / t[1] * 100 if t[1] else 0
        lines.append(f"{label:<14}{t[0] * 100:>+9.0f}%{t[1]:>6d}{wr:>5.0f}%"
                     f"{(base_others + t[0]) * 100:>+17.0f}%")
    txt = "\n".join(lines)
    with open("regime_lab3_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
