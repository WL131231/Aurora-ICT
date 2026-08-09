"""#AUTONOMOUS 2026-07-27: 수축 부스트 검증 배터리 — 배포 전 다중 검증(파트너 지시).

대상 가설: BBW(1h,20,2σ) < 롤링90일 q33(수축) 구간의 Origo 거래가 우월 → 사이즈 1.3x.
smart_size 교훈(1차 발견이 look-ahead 로 부풀던 것) 재발 방지 — 전 지표 인과 계산
(직전 완결 1h봉 + 롤링 분위) 위에서:
  1) 연도별 분해 — 수축 vs 비수축 avg 가 연도 일관인지
  2) 페어별 분해 — 1~2 페어 몰빵인지
  3) 파라미터 이웃 — BBW 창(14/20/40)·TF(1h/4h)·분위(q25/q33/q40)·롤링(60/90/120d)
  4) 교차 지표 — ATR%(1h) 수축으로도 같은 효과인지(메커니즘 확인)
  5) 셔플 검정 — 수축 라벨 무작위 재배치 10,000회, avg 차이 p-value
  6) 부스트 장부 — 1.0x vs 1.3x vs 1.5x net/MDD(가중 누적곡선)
  7) 방향(롱/숏)·킬존 분해 — 특정 조건 몰빵인지
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from chop_gate_bakeoff import BASE, FIXED, bbw20, roll_q  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402


def atrp(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    """ATR% — 절대 변동성(가격 대비 %). 낮으면 수축."""
    tr = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    a = pd.Series(tr).rolling(n).mean().to_numpy()
    return a / (c + 1e-12) * 100


def collect(sym: str) -> list[dict]:
    """라이브게이트 거래 + 다중 수축 지표(전부 직전 완결봉·롤링 분위 — 인과)."""
    df5 = _resample(_load_full(sym))
    cfg = BacktestConfig(**BASE)
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    frames = {}
    for tf in ("1h", "4h"):
        d = df5.resample(tf).agg({"high": "max", "low": "min", "close": "last"}).dropna()
        h, lo_, c = (d[k].to_numpy() for k in ("high", "low", "close"))
        cols = {
            f"bbw14_{tf}": bbw20(c, 14), f"bbw20_{tf}": bbw20(c, 20),
            f"bbw40_{tf}": bbw20(c, 40), f"atrp_{tf}": atrp(h, lo_, c),
        }
        bars_day = 24 if tf == "1h" else 6
        ind = pd.DataFrame(cols, index=d.index)
        for base_col in list(cols):
            x = ind[base_col].to_numpy()
            for days in (60, 90, 120):
                for q in (0.25, 0.33, 0.40):
                    ind[f"{base_col}_d{days}q{int(q * 100)}"] = roll_q(
                        x, bars_day * days, q)
        frames[tf] = ind.shift(1).reindex(df5.index, method="ffill")
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
        rec = dict(ts=df5.index[t.entry_idx], net=t.net_pnl_pct, sym=sym,
                   long=(t.direction == "long"), hour=hh)
        ok = True
        for tf, ind in frames.items():
            row = ind.iloc[t.entry_idx]
            if row.isna().any():
                ok = False
                break
            for k, v in row.items():
                rec[k] = v
        if ok:
            out.append(rec)
    return out


def avg(g: list[dict]) -> float:
    return sum(x["net"] for x in g) / len(g) if g else float("nan")


def wr(g: list[dict]) -> float:
    return 100 * sum(1 for x in g if x["net"] > 0) / len(g) if g else float("nan")


def split(allt: list[dict], col: str, thr_col: str):
    inb = [x for x in allt if x[col] < x[thr_col]]
    outb = [x for x in allt if not (x[col] < x[thr_col])]
    return inb, outb


def main() -> int:
    allt: list[dict] = []
    for sym in FIXED:
        rows = collect(sym)
        allt += rows
        print(f"{sym}: {len(rows)}건", flush=True)
    allt.sort(key=lambda x: x["ts"])
    main_col, main_thr = "bbw20_1h", "bbw20_1h_d90q33"
    inb, outb = split(allt, main_col, main_thr)
    print(f"\n기준 가설(BBW20 1h < 90d q33): 수축 n={len(inb)} avg={avg(inb):+.3f} "
          f"승률={wr(inb):.0f}% | 비수축 n={len(outb)} avg={avg(outb):+.3f} 승률={wr(outb):.0f}%",
          flush=True)

    print("\n[1] 연도별 (수축avg | 비수축avg | 수축n)", flush=True)
    for y in sorted({x["ts"].year for x in allt}):
        yi = [x for x in inb if x["ts"].year == y]
        yo = [x for x in outb if x["ts"].year == y]
        print(f"  {y}: {avg(yi):+.3f} | {avg(yo):+.3f} | n={len(yi)}", flush=True)

    print("\n[2] 페어별 (수축avg | 비수축avg | 수축n)", flush=True)
    for s in FIXED:
        si = [x for x in inb if x["sym"] == s]
        so = [x for x in outb if x["sym"] == s]
        print(f"  {s:<9}: {avg(si):+.3f} | {avg(so):+.3f} | n={len(si)}", flush=True)

    print("\n[3] 파라미터 이웃 (수축avg-비수축avg 격차, +면 가설방향)", flush=True)
    for tf in ("1h", "4h"):
        for w in (14, 20, 40):
            for days in (60, 90, 120):
                for q in (25, 33, 40):
                    col = f"bbw{w}_{tf}"
                    thr = f"{col}_d{days}q{q}"
                    a, b = split(allt, col, thr)
                    if len(a) >= 15:
                        print(f"  {col} d{days} q{q}: gap={avg(a) - avg(b):+.3f} "
                              f"(수축n={len(a)} avg={avg(a):+.3f})", flush=True)

    print("\n[4] 교차 지표 ATR% 수축", flush=True)
    for tf in ("1h", "4h"):
        col = f"atrp_{tf}"
        a, b = split(allt, col, f"{col}_d90q33")
        print(f"  {col}<90d q33: 수축 n={len(a)} avg={avg(a):+.3f} 승률={wr(a):.0f}% "
              f"| 비수축 avg={avg(b):+.3f}", flush=True)

    print("\n[5] 셔플 검정 (라벨 무작위 10,000회)", flush=True)
    rng = np.random.default_rng(42)
    nets = np.array([x["net"] for x in allt])
    k = len(inb)
    obs = avg(inb) - avg(outb)
    cnt = 0
    for _ in range(10000):
        idx = rng.permutation(len(nets))
        d = nets[idx[:k]].mean() - nets[idx[k:]].mean()
        if d >= obs:
            cnt += 1
    print(f"  관측 격차 {obs:+.3f} → p={cnt / 10000:.4f}", flush=True)

    print("\n[6] 부스트 장부 (net / MDD, 시간순 누적)", flush=True)
    for k_ in (1.0, 1.3, 1.5):
        eq = 0.0
        peak = 0.0
        mdd = 0.0
        tot = 0.0
        for x in allt:
            wgt = k_ if x[main_col] < x[main_thr] else 1.0
            tot += x["net"] * wgt
            eq += x["net"] * wgt
            peak = max(peak, eq)
            mdd = max(mdd, peak - eq)
        print(f"  x{k_}: net={tot:+.1f} MDD={mdd:.1f} net/MDD={tot / max(mdd, 1e-9):.2f}",
              flush=True)

    print("\n[7] 방향·세션 분해 (수축 안에서)", flush=True)
    for name, fn in (("롱", lambda x: x["long"]), ("숏", lambda x: not x["long"]),
                     ("아시아(0-8 UTC)", lambda x: 0 <= x["hour"] < 8),
                     ("런던(8-13)", lambda x: 8 <= x["hour"] < 13),
                     ("NY(13-17)", lambda x: 13 <= x["hour"] < 17),
                     ("기타(21-24)", lambda x: x["hour"] >= 21)):
        g = [x for x in inb if fn(x)]
        print(f"  {name:<14}: n={len(g):3d} avg={avg(g):+.3f} 승률={wr(g):.0f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
