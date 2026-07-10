"""국면 랩 2차 — 상승장 적자 해부 + 조건부 게이트 조합 (2026-07-10).

1차 발견: 상승 국면만 적자(-96%, 92건/28%). 후속:
    A. 상승 버킷 방향 분해 — 롱이 지는가 숏이 지는가 (원인 특정).
    B. 조건부 조합 검증 (필터 간 상호작용 포함 실측):
       - 횡보 국면만 NY_PM 진입 제외 (1차: 횡보 +1135→+1359 시사)
       - 상승 국면만 OTE 0.5 (얕은 되돌림 — 강추세에서 0.707 은 미체결/역선택 가설).
         혼합 타임라인: 상승일=ote0.5 타임라인, 그 외=ote0.707.
       - 위 둘 결합.
표기 = 시드% (단리). 사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/regime_lab2.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta
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
    from aurora_ict.strategy.silver_bullet import Direction
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    cfg7 = BacktestConfig(**{**BASE, "ote_level": 0.707, "entry_ttl_bars": ttl})
    cfg5 = BacktestConfig(**{**BASE, "ote_level": 0.5, "entry_ttl_bars": ttl})
    tl7 = cached_setup_timeline(df5, cfg7, sym)
    tl5 = cached_setup_timeline(df5, cfg5, sym)
    days_idx, labels = classify_days(df5)
    day_of = days_idx.searchsorted(df5.index.normalize(), side="right") - 1
    hours = (df5.index + timedelta(hours=9)).hour
    npm = (hours >= 2) & (hours < 5)

    def lab(i):
        di = day_of[i]
        return labels[di] if di >= 0 else "횡보"

    def buckets_of(bt, split_dir=False):
        if split_dir:
            out = {}
            for t in bt.trades:
                key = (lab(t.entry_idx), t.direction)
                b = out.setdefault(key, [0.0, 0, 0])
                b[0] += t.net_pnl_pct
                b[1] += 1
                b[2] += 1 if t.net_pnl_pct > 0 else 0
            return out
        out = {r: [0.0, 0, 0] for r in REGIMES}
        for t in bt.trades:
            b = out[lab(t.entry_idx)]
            b[0] += t.net_pnl_pct
            b[1] += 1
            b[2] += 1 if t.net_pnl_pct > 0 else 0
        return out

    res = {}
    # A. 기준 방향 분해
    bt0 = run_backtest_from_timeline(df5, tl7, cfg7)
    res["dir_split"] = buckets_of(bt0, split_dir=True)
    res["기준"] = buckets_of(bt0)

    # B1. 횡보만 NY_PM 제외
    ftl = list(tl7)
    for i, item in enumerate(ftl):
        if item is not None and lab(i) == "횡보" and npm[i]:
            ftl[i] = None
    res["횡보만 NY_PM 제외"] = buckets_of(run_backtest_from_timeline(df5, ftl, cfg7))

    # B2. 상승만 OTE 0.5 (혼합 타임라인) — 상승일은 tl5, 그 외 tl7
    mtl = [tl5[i] if lab(i) == "상승" else tl7[i] for i in range(len(tl7))]
    res["상승만 OTE0.5"] = buckets_of(run_backtest_from_timeline(df5, mtl, cfg7))

    # B3. 결합
    ctl = [tl5[i] if lab(i) == "상승" else tl7[i] for i in range(len(tl7))]
    for i, item in enumerate(ctl):
        if item is not None and lab(i) == "횡보" and npm[i]:
            ctl[i] = None
    res["결합(B1+B2)"] = buckets_of(run_backtest_from_timeline(df5, ctl, cfg7))
    print(f"  {sym} done", flush=True)
    return sym, res


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    lines = ["===== 국면 랩 2차 (7페어 5년, 시드% 단리) =====", "",
             "-- A. 기준 상승 국면 방향 분해 --"]
    from aurora_ict.strategy.silver_bullet import Direction
    for d in (Direction.LONG, Direction.SHORT):
        net = n = w = 0
        for _s, res in results:
            b = res["dir_split"].get(("상승", d), [0.0, 0, 0])
            net += b[0]
            n += b[1]
            w += b[2]
        wr = w / n * 100 if n else 0
        lines.append(f"  상승 {d.value:<6} net {net * 100:+.0f}%  n={n}  승률 {wr:.0f}%")
    lines.append("")
    for label in ["기준", "횡보만 NY_PM 제외", "상승만 OTE0.5", "결합(B1+B2)"]:
        tot = {r: [0.0, 0, 0] for r in REGIMES}
        for _s, res in results:
            for r in REGIMES:
                b = res[label][r]
                tot[r][0] += b[0]
                tot[r][1] += b[1]
                tot[r][2] += b[2]
        g = sum(v[0] for v in tot.values())
        gn = sum(v[1] for v in tot.values())
        seg = "  ".join(f"{r} {tot[r][0] * 100:+.0f}%({tot[r][1]})" for r in REGIMES)
        lines.append(f"{label:<18} {seg} | 합계 {g * 100:+.0f}% ({gn}건)")
    txt = "\n".join(lines)
    with open("regime_lab2_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
