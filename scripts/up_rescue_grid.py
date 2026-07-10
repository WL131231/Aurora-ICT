"""상승 국면 정복 그리드 — 전면 탐색 (파트너 전권 위임, 2026-07-10).

강건성 반전: 상승 적자(-96%)는 전반부(2021~23 폭주 알트런, SOL -406/LINK -175)
집중, 후반부는 +470% 흑자 → '차단'은 오답. 목표 = 전반부형 폭주 상승에서 죽지
않으면서 후반부 수익을 지키는 구성.

Phase 1 — 기준 상승 92건 전차원 슬라이스 (윈도우/시간/요일/점수/소스/페어/반기).
Phase 2 — 변형 ~20종 full-run → 상승 버킷만 (반기 분해 포함):
    진입: ote 0.618/0.786(캐시) · conf6 · stale2 · ema_bias off(양방향) ·
          sweep성분 필수 · killzone별 허용(슬라이스 후속)
    청산: tp_rr 0.5/0.75/1.0/1.5 (빠른 익절 — 되돌림 초기 반등 수확 가설) ·
          trail 1.0/0.75 · 1.5/1.0 · 2.5/2.0 · partial1.0+BE · BE 0.5/1.5
판정 기준: 상승 버킷 전반부 개선 AND 후반부 +470% 크게 안 깎임 AND 타국면 무영향
(조건부 적용 가정).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/up_rescue_grid.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_edge_lab import classify_days  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)
# (라벨, override) — ote 변형은 캐시 타임라인 사용(키 포함), 나머지 재생만.
VARIANTS = [
    ("기준(1.6)", {}),
    ("ote0.618", dict(ote_level=0.618)),
    ("ote0.786", dict(ote_level=0.786)),
    ("conf6", dict(min_confluence=6)),
    ("stale2", dict(setup_stale_bars=2)),
    ("ema off(양방향)", dict(htf_ema_bias="off")),
    ("tp 0.5R", dict(tp_rr_override=0.5, trail_trigger=0.0, trail_dist=0.0,
                     be_trigger=0.0)),
    ("tp 0.75R", dict(tp_rr_override=0.75, trail_trigger=0.0, trail_dist=0.0,
                      be_trigger=0.0)),
    ("tp 1.0R", dict(tp_rr_override=1.0, trail_trigger=0.0, trail_dist=0.0,
                     be_trigger=0.0)),
    ("tp 1.5R", dict(tp_rr_override=1.5, trail_trigger=0.0, trail_dist=0.0,
                     be_trigger=0.0)),
    ("trail 1.0/0.75", dict(trail_trigger=1.0, trail_dist=0.75)),
    ("trail 1.5/1.0", dict(trail_trigger=1.5, trail_dist=1.0)),
    ("trail 2.5/2.0", dict(trail_trigger=2.5, trail_dist=2.0)),
    ("partial1.0+BE", dict(partial_tp_rr=1.0, partial_be=True)),
    ("BE 0.5", dict(be_trigger=0.5)),
    ("BE 1.5", dict(be_trigger=1.5)),
]


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    days_idx, labels = classify_days(df5)
    day_of = days_idx.searchsorted(df5.index.normalize(), side="right") - 1
    hours = (df5.index + timedelta(hours=9)).hour
    dows = (df5.index + timedelta(hours=9)).dayofweek
    half = len(df5) // 2

    def lab(i):
        di = day_of[i]
        return labels[di] if di >= 0 else "횡보"

    out = {"slices": {}}
    for label, ov in VARIANTS:
        cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl, **ov})
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        up = [0.0, 0.0, 0, 0]  # 전반net, 후반net, n, wins
        for t in bt.trades:
            if lab(t.entry_idx) != "상승":
                continue
            h = 0 if t.entry_idx < half else 1
            up[h] += t.net_pnl_pct
            up[2] += 1
            up[3] += 1 if t.net_pnl_pct > 0 else 0
        out[label] = up
        # Phase 1 슬라이스 — 기준만
        if label == "기준(1.6)":
            sl = out["slices"]
            for t in bt.trades:
                if lab(t.entry_idx) != "상승":
                    continue
                keys = [
                    ("window", getattr(t, "window", None) or "?"),
                    ("hourKST", hours[t.entry_idx] // 4 * 4),
                    ("dow", int(dows[t.entry_idx])),
                    ("score", t.confluence_score),
                    ("pair", sym),
                ]
                for dim, val in keys:
                    b = sl.setdefault((dim, str(val)), [0.0, 0])
                    b[0] += t.net_pnl_pct
                    b[1] += 1
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    lines = ["===== 상승 국면 정복 그리드 (7페어 5년, 상승 버킷만, 시드% 단리) =====",
             f"{'변형':<18}{'전반':>8}{'후반':>8}{'합':>8}{'n':>5}{'승률':>6}"]
    for label, _ in VARIANTS:
        h1 = h2 = 0.0
        n = w = 0
        for _s, out in results:
            u = out[label]
            h1 += u[0]
            h2 += u[1]
            n += u[2]
            w += u[3]
        wr = w / n * 100 if n else 0
        lines.append(f"{label:<18}{h1 * 100:>+7.0f}%{h2 * 100:>+7.0f}%"
                     f"{(h1 + h2) * 100:>+7.0f}%{n:>5d}{wr:>5.0f}%")
    # Phase 1 슬라이스 집계 (n>=10 만)
    agg = {}
    for _s, out in results:
        for k, v in out["slices"].items():
            b = agg.setdefault(k, [0.0, 0])
            b[0] += v[0]
            b[1] += v[1]
    lines.append("")
    lines.append("-- 기준 상승 버킷 슬라이스 (n>=10, net순) --")
    for (dim, val), (net, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        if n >= 10:
            lines.append(f"  {dim}={val:<12} net {net * 100:+.0f}%  n={n}")
    txt = "\n".join(lines)
    with open("up_rescue_grid_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
