"""#AUTONOMOUS 2026-07-27: MMBM 국면 조건부 분석 — 2026 양수(+15%)가 추세장 효과인가.

매트릭스(mmbm_matrix)서 V0/V1 이 5년 적자인데 2026 만 양수 — 국면(추세/횡보) 게이트로
살릴 수 있는지 판정. 진입 시점의 1d Kaufman ER(14일)로 국면 3분위 분류 후
분위별 net/승률 + 월별 분해. robust 하면 "추세장 전용 MMBM" 재활성 후보,
아니면 2026 양수는 소표본 노이즈로 종결.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from mmbm_matrix import RTCOST, collect_setups, run_variant  # noqa: E402


def er_series(df: pd.DataFrame, days: int = 14) -> pd.Series:
    """1d 종가 Kaufman ER(효율비) — |net변화| / 절대변화합. 높음=추세, 낮음=횡보."""
    d1 = df["close"].resample("1d").last().dropna()
    change = (d1 - d1.shift(days)).abs()
    vol = d1.diff().abs().rolling(days).sum()
    er = (change / vol).clip(0, 1)
    return er


def main() -> int:
    pairs = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT"]
    rows = []  # (ts, net%, er, sym, variant)
    for sym in pairs:
        print(f"--- {sym}", flush=True)
        df, h, lo, n, setups = collect_setups(sym)
        er = er_series(df)
        for vname, sw in (("V0", False), ("V1", True)):
            trades = run_variant(df, h, lo, n, setups, sw, False, False, False, RTCOST)
            for ts, net in trades:
                e = er.asof(ts)
                if pd.notna(e):
                    rows.append((ts, net, float(e), sym, vname))
    t = pd.DataFrame(rows, columns=["ts", "net", "er", "sym", "v"])
    for vname in ("V0", "V1"):
        sub = t[t.v == vname].copy()
        q1, q2 = sub.er.quantile([1 / 3, 2 / 3])
        sub["reg"] = np.where(sub.er < q1, "횡보(ER하위)", np.where(sub.er < q2, "중간", "추세(ER상위)"))
        print(f"\n===== {vname} 국면 3분위 (ER 경계 {q1:.3f}/{q2:.3f}) =====", flush=True)
        for g, gg in sub.groupby("reg"):
            w = (gg.net > 0).mean() * 100
            print(f"{g:12s} n={len(gg):5d} net={gg.net.sum():+8.1f}% 승률={w:4.1f}%", flush=True)
        # 추세 상위분위만 연도별 — robust 확인
        top = sub[sub.er >= q2]
        top["yr"] = top.ts.dt.year
        print("  [추세 상위분위 연도별]", flush=True)
        for y, gg in top.groupby("yr"):
            print(f"   {y}: n={len(gg):4d} net={gg.net.sum():+7.1f}% 승률={(gg.net > 0).mean() * 100:4.1f}%", flush=True)
        # 2026 월별 (양수가 몇 달에 몰렸나)
        s26 = sub[sub.ts >= pd.Timestamp("2026-01-01", tz="UTC")]
        print("  [2026 월별]", flush=True)
        for m, gg in s26.groupby(s26.ts.dt.month):
            print(f"   {m}월: n={len(gg):3d} net={gg.net.sum():+7.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
