"""#AUTONOMOUS 2026-07-27: B트랙 — 펀딩비 신호 연구 (미탐색 축).

가설: 펀딩 극단 = 군중 쏠림 → 역방향 스퀴즈 성향. 검사 2단:
  [1] 원신호 진단 — 펀딩 롤링90일 분위 상/하위 10%·25% 구간의 이후 24h 수익률이
      중립 대비 다른가 (페어별+합산, 인과: 직전 확정 펀딩만).
  [2] Origo 오버레이 — 라이브게이트 거래의 진입 시점 펀딩 분위 × 방향 정합
      (역군중 롱 = 펀딩 하위에서 롱 등) 별 avg net 차이.
원신호가 죽으면 [2]는 참고만 — 배포 후보 아님으로 즉시 종결(과최적화 방지).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402
from chop_gate_bakeoff import FIXED, collect as collect_trades  # noqa: E402

ROLL = 90 * 3  # 8h 펀딩 × 3/일 × 90일


def load_funding(sym: str) -> pd.Series:
    f = pd.read_parquet(f"data/{sym}_funding.parquet")["rate"].astype(float)
    f.index = pd.DatetimeIndex(f.index, tz="UTC") if f.index.tz is None else f.index
    return f.sort_index()


def main() -> int:
    print("[1] 원신호 진단 — 펀딩 분위별 이후 24h 수익률(%)", flush=True)
    agg: dict[str, list[float]] = {"하위10%": [], "하위25%": [], "중립": [], "상위25%": [], "상위10%": []}
    for sym in FIXED:
        try:
            f = load_funding(sym)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym}: 펀딩 없음 {e}", flush=True)
            continue
        px = _resample(_load_full(sym))["close"]
        p1h = px.resample("1h").last().dropna()
        q = f.rolling(ROLL, min_periods=ROLL // 3).rank(pct=True).shift(1)  # 인과 분위
        rows = {k: [] for k in agg}
        for ts, rk in q.dropna().items():
            t0 = p1h.index.asof(ts)
            if pd.isna(t0):
                continue
            i0 = p1h.index.get_loc(t0)
            if i0 + 24 >= len(p1h):
                continue
            ret = (p1h.iloc[i0 + 24] / p1h.iloc[i0] - 1) * 100
            k = ("하위10%" if rk < 0.10 else "하위25%" if rk < 0.25 else
                 "상위10%" if rk > 0.90 else "상위25%" if rk > 0.75 else "중립")
            rows[k].append(ret)
        line = "  ".join(f"{k}:{np.mean(v):+.3f}(n={len(v)})" for k, v in rows.items() if v)
        print(f"  {sym:<9} {line}", flush=True)
        for k, v in rows.items():
            agg[k] += v
    print("  --- 합산 ---", flush=True)
    for k, v in agg.items():
        if v:
            t = np.mean(v) / (np.std(v) / np.sqrt(len(v)) + 1e-12)
            print(f"  {k:<7} mean={np.mean(v):+.4f}% n={len(v):5d} t={t:+.2f}", flush=True)

    print("\n[2] Origo 오버레이 — 진입 시점 펀딩 분위 × 방향", flush=True)
    allt = []
    for sym in FIXED:
        try:
            f = load_funding(sym)
        except Exception:  # noqa: BLE001
            continue
        q = f.rolling(ROLL, min_periods=ROLL // 3).rank(pct=True).shift(1)
        for tr in collect_trades(sym):
            ts = tr["ts"]
            k = q.index.asof(ts)
            if pd.isna(k) or pd.isna(q.loc[k]):
                continue
            tr["fq"] = float(q.loc[k])
            allt.append(tr)

    def s(g):
        n = len(g)
        if not n:
            return "n=  0"
        net = sum(x["net"] for x in g)
        w = sum(1 for x in g if x["net"] > 0)
        return f"n={n:3d} avg={net / n:+.3f} 승률={100 * w / n:.0f}%"

    longs = [x for x in allt if x.get("adx") is not None]
    print(f"  전체 {s(allt)}", flush=True)
    buckets = {
        "역군중 (롱×펀딩<25% / 숏×펀딩>75%)":
            lambda x: (x["fq"] < 0.25) if x["net"] is not None and x_is_long(x) else (x["fq"] > 0.75),
    }
    # 단순 명확하게 4분할
    def x_is_long(x):
        return x.get("long", None)
    # collect_trades 는 long 필드가 없음 — bakeoff collect 확인 필요시 스킵
    has_dir = all("long" in x for x in allt[:3]) if allt else False
    if not has_dir:
        # 방향 무관 분위 4분할만
        for name, lo_, hi in (("펀딩 하위25%", 0.0, 0.25), ("중간", 0.25, 0.75),
                              ("상위25%", 0.75, 1.01)):
            g = [x for x in allt if lo_ <= x["fq"] < hi]
            print(f"  {name:<10} {s(g)}", flush=True)
    else:
        for name, fn in (
            ("역군중(롱저펀딩/숏고펀딩)", lambda x: (x["long"] and x["fq"] < 0.25) or (not x["long"] and x["fq"] > 0.75)),
            ("순군중(롱고펀딩/숏저펀딩)", lambda x: (x["long"] and x["fq"] > 0.75) or (not x["long"] and x["fq"] < 0.25)),
            ("중립", lambda x: 0.25 <= x["fq"] <= 0.75),
        ):
            g = [x for x in allt if fn(x)]
            print(f"  {name:<16} {s(g)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
