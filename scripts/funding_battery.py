"""#AUTONOMOUS 2026-07-27: 펀딩 오버레이 검증배터리 — 고펀딩 구간 Origo 열세 가설.

발견(소표본): 진입 시점 펀딩 롤링90일 분위 상위25% 구간의 Origo 거래 avg -0.054
(41%) vs 하위 +0.300 (61%). 원신호는 고펀딩=24h 상승 드리프트(t=6.5, 중첩 과대).
가설 후보: 고펀딩(과열 상승)에서 Origo 숏이 드리프트에 역행.

배터리(수축부스트 탈락 기준과 동일 잣대):
  [1] 방향 분해 — 열세가 숏에 국한인지
  [2] 연도별 / [3] 페어별 일관성
  [4] 파라미터 이웃 — 분위 컷(70/75/80%) × 롤링(60/90/120d)
  [5] 셔플 검정 10,000회
  [6] 게이트 시뮬 — '고펀딩 숏 skip' 적용 시 net/MDD (연도별)
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from chop_gate_bakeoff import BASE, FIXED  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402


def collect(sym: str) -> list[dict]:
    """라이브게이트 거래 + 방향 + 펀딩 분위(롤링 60/90/120d, 인과 shift)."""
    df5 = _resample(_load_full(sym))
    cfg = BacktestConfig(**BASE)
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    f = pd.read_parquet(f"data/{sym}_funding.parquet")["rate"].astype(float).sort_index()
    qs = {}
    for days in (60, 90, 120):
        qs[days] = f.rolling(days * 3, min_periods=days).rank(pct=True).shift(1)
    mags = [abs(t.entry_trend_pct) for t in bt.trades
            if not (17 <= df5.index[t.entry_idx].hour < 21)]
    q70 = np.percentile(mags, 70) if mags else 0.0
    out = []
    for t in bt.trades:
        hh = df5.index[t.entry_idx].hour
        if 17 <= hh < 21:
            continue
        sgn = 1.0 if t.direction == "long" else -1.0
        if abs(t.entry_trend_pct) < q70 and t.entry_trend_pct * sgn < 0:
            continue
        ts = df5.index[t.entry_idx]
        rec = dict(ts=ts, net=t.net_pnl_pct, sym=sym,
                   long=(t.direction == "long"), hour=hh)
        ok = True
        for days, q in qs.items():
            k = q.index.asof(ts)
            v = q.loc[k] if not pd.isna(k) else np.nan
            if pd.isna(v):
                ok = False
                break
            rec[f"fq{days}"] = float(v)
        if ok:
            out.append(rec)
    return out


def s(g: list[dict]) -> str:
    n = len(g)
    if not n:
        return "n=  0"
    net = sum(x["net"] for x in g)
    w = sum(1 for x in g if x["net"] > 0)
    return f"n={n:3d} avg={net / n:+.3f} 승률={100 * w / n:.0f}%"


def main() -> int:
    allt: list[dict] = []
    for sym in FIXED:
        try:
            rows = collect(sym)
        except Exception as e:  # noqa: BLE001
            print(f"{sym}: 스킵 {e}", flush=True)
            continue
        allt += rows
    allt.sort(key=lambda x: x["ts"])
    hi = [x for x in allt if x["fq90"] > 0.75]
    rest = [x for x in allt if x["fq90"] <= 0.75]
    print(f"기준: 고펀딩(90d q>75%) {s(hi)} | 나머지 {s(rest)}", flush=True)

    print("\n[1] 방향 분해 (고펀딩 안 / 밖)", flush=True)
    for name, fn in (("고펀딩 롱", lambda x: x["fq90"] > 0.75 and x["long"]),
                     ("고펀딩 숏", lambda x: x["fq90"] > 0.75 and not x["long"]),
                     ("저·중 롱", lambda x: x["fq90"] <= 0.75 and x["long"]),
                     ("저·중 숏", lambda x: x["fq90"] <= 0.75 and not x["long"])):
        print(f"  {name:<8} {s([x for x in allt if fn(x)])}", flush=True)

    print("\n[2] 연도별 (고펀딩avg | 나머지avg | 고펀딩n)", flush=True)
    for y in sorted({x["ts"].year for x in allt}):
        a = [x for x in hi if x["ts"].year == y]
        b = [x for x in rest if x["ts"].year == y]
        av = sum(x['net'] for x in a) / len(a) if a else float('nan')
        bv = sum(x['net'] for x in b) / len(b) if b else float('nan')
        print(f"  {y}: {av:+.3f} | {bv:+.3f} | n={len(a)}", flush=True)

    print("\n[3] 페어별 (고펀딩avg | n)", flush=True)
    for sym in FIXED:
        a = [x for x in hi if x["sym"] == sym]
        if a:
            print(f"  {sym:<9}: {sum(x['net'] for x in a) / len(a):+.3f} | n={len(a)}", flush=True)

    print("\n[4] 파라미터 이웃 (컷×롤링 — 고펀딩avg-나머지avg 격차, -면 가설방향)", flush=True)
    for days in (60, 90, 120):
        for cut in (0.70, 0.75, 0.80):
            a = [x for x in allt if x[f"fq{days}"] > cut]
            b = [x for x in allt if x[f"fq{days}"] <= cut]
            if len(a) >= 12:
                ga = sum(x["net"] for x in a) / len(a)
                gb = sum(x["net"] for x in b) / len(b)
                print(f"  d{days} cut{int(cut * 100)}: gap={ga - gb:+.3f} (n={len(a)} avg={ga:+.3f})",
                      flush=True)

    print("\n[5] 셔플 검정", flush=True)
    rng = np.random.default_rng(7)
    nets = np.array([x["net"] for x in allt])
    k = len(hi)
    obs = (sum(x["net"] for x in hi) / max(k, 1)) - (sum(x["net"] for x in rest) / max(len(rest), 1))
    cnt = 0
    for _ in range(10000):
        idx = rng.permutation(len(nets))
        d = nets[idx[:k]].mean() - nets[idx[k:]].mean()
        if d <= obs:  # 가설 방향 = 고펀딩이 더 나쁨(음의 격차)
            cnt += 1
    print(f"  관측 격차 {obs:+.3f} → p={cnt / 10000:.4f}", flush=True)

    print("\n[6] 게이트 시뮬 — '고펀딩(90d>75%) 숏만 skip'", flush=True)
    kept = [x for x in allt if not (x["fq90"] > 0.75 and not x["long"])]
    cut_ = [x for x in allt if x["fq90"] > 0.75 and not x["long"]]
    print(f"  skip 대상 {s(cut_)}", flush=True)
    print(f"  적용 후   {s(kept)}  (base {s(allt)})", flush=True)
    ys: dict[int, float] = {}
    for x in kept:
        ys[x["ts"].year] = ys.get(x["ts"].year, 0.0) + x["net"]
    print("  적용 후 연도별:", " ".join(f"{y}:{v:+.1f}" for y, v in sorted(ys.items())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
