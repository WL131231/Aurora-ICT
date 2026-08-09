"""#AUTONOMOUS 2026-07-27: 페어 확장 재스캔 — 현행(2.2) 설정으로 전 알트 재검증.

무한 연구루프 1타. 6~7월 페어 스캔은 옛 설정 기준 — 이후 NY_PM 게이트·cond_align·
min_rr/sl 강제 등이 붙어 현행 구독 설정으로 전 후보를 다시 돌린다. NEAR(구 후보,
미배포) 포함. 판정: 양반기(H1/H2) 흑자 + 연도별 다수 흑자 + net/MDD — 고정7 평균과
비교해 추가 가치 있는 페어만 후보 승격(이후 검증배터리 풀버전).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from chop_gate_bakeoff import BASE  # noqa: E402 — 현행 설정 재사용

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

ALTS = [
    "NEARUSDT", "ENAUSDT", "FILUSDT", "ARBUSDT", "AAVEUSDT", "ADAUSDT",
    "ATOMUSDT", "AVAXUSDT", "BCHUSDT", "BNBUSDT", "DOTUSDT", "ETCUSDT",
    "INJUSDT", "LTCUSDT", "OPUSDT", "SEIUSDT", "SUIUSDT", "TIAUSDT",
    "TONUSDT", "TRXUSDT", "UNIUSDT", "WIFUSDT", "WLDUSDT",
]


def scan(sym: str) -> str:
    try:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
    except Exception as e:  # noqa: BLE001
        return f"{sym:<9} 실패: {e}"
    # 라이브게이트 재현(NY_PM 제외 + cond_align)
    trs = [t for t in bt.trades if not (17 <= df5.index[t.entry_idx].hour < 21)]
    mags = [abs(t.entry_trend_pct) for t in trs]
    q70 = np.percentile(mags, 70) if mags else 0.0
    kept = []
    for t in trs:
        sgn = 1.0 if t.direction == "long" else -1.0
        if abs(t.entry_trend_pct) < q70 and t.entry_trend_pct * sgn < 0:
            continue
        kept.append((df5.index[t.entry_idx], t.net_pnl_pct))
    if len(kept) < 6:
        return f"{sym:<9} n={len(kept)} (표본부족)"
    kept.sort()
    nets = [p for (_, p) in kept]
    net = sum(nets)
    w = sum(1 for p in nets if p > 0)
    half = len(kept) // 2
    h1 = sum(p for (_, p) in kept[:half])
    h2 = sum(p for (_, p) in kept[half:])
    # 연도별
    ys: dict[int, float] = {}
    for ts, p in kept:
        ys[ts.year] = ys.get(ts.year, 0.0) + p
    ypos = sum(1 for v in ys.values() if v > 0)
    # MDD
    eq = pk = mdd = 0.0
    for _, p in kept:
        eq += p
        pk = max(pk, eq)
        mdd = max(mdd, pk - eq)
    nm = net / max(mdd, 1e-9)
    yearly = " ".join(f"{y}:{v:+.1f}" for y, v in sorted(ys.items()))
    flag = "★" if (h1 > 0 and h2 > 0 and ypos >= max(1, len(ys) - 1) and net > 0) else " "
    return (f"{flag}{sym:<9} n={len(kept):3d} net={net:+7.1f} 승률={100 * w / len(kept):3.0f}% "
            f"H1={h1:+.1f} H2={h2:+.1f} net/MDD={nm:4.1f} 연도[{yearly}]")


def main() -> int:
    print("현행(2.2) 설정 알트 재스캔 — ★=양반기+연도 robust 후보", flush=True)
    for sym in ALTS:
        print(scan(sym), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
