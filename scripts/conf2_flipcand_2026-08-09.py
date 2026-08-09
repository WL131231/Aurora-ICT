"""#PARITY-CONF 2026-08-09 — 후보의 엣지가 **flip 버전에 의존하는지** 검사.

## 왜

conf2_flipver 로 확인: 백테 BASE(flip_min_r=1.5, Origo 2.3)와 라이브 기준선
(Origo 2.2, flip_min_r 없음)은 다른 버전이고, flip_min_r=0 으로 되돌리면
승률 25→38%(라이브 46%) · RR 2.50→1.38(라이브 0.94) · 승리 중 1R미만
15→54%(라이브 86%) 로 라이브 쪽으로 이동한다. **건당R 은 −0.067→−0.069 로 불변.**

BASE 는 EV 가 불변이지만 후보(macro_high AND bias)는 사정이 다를 수 있다.
후보의 이익은 TP(+2.6R) 같은 **큰 승자**에 얹혀 있는데, flip 은 그 승자를
0.6R 대에서 자르는 장치다. 후보 엣지가 "2.3 에서만 성립"하면 라이브 2.2 기준선
위에서 고른 항목 데이터와 논리가 어긋난다.

## 변형

    MH_BIAS5      후보 (문턱5 + macro_high + bias), flip_min_r=1.5  (현행 2.3)
    MH_BIAS5_F0   같은 후보, flip_min_r=0.0                          (구 2.2)
    BASE_F0       대조군 (이미 flipver 에서 측정, 재확인용)
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT"]
OUT = "data/conf2/flipcand.json"
REQ = ("macro_high", "bias")
VARIANTS = {
    "MH_BIAS5": {"require_items": REQ},
    "MH_BIAS5_F0": {"require_items": REQ, "flip_min_r": 0.0},
}


def _run(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from live_parity import live_cfg

    from aurora_ict.backtest.replay import run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    tl = cached_setup_timeline(df5, live_cfg(sym), sym)
    out = {}
    for vn, ex in VARIANTS.items():
        bt = run_backtest_from_timeline(df5, tl, live_cfg(sym, ex))
        rows = []
        for t in bt.trades:
            risk = abs(t.entry - t.entry_sl) / t.entry if t.entry > 0 else 0.0
            rows.append({"sym": sym, "ts": df5.index[t.entry_idx].isoformat(),
                         "r": (float(t.raw_pnl_pct) / risk) if risk > 0 else float("nan"),
                         "raw": float(t.raw_pnl_pct), "outcome": t.outcome})
        out[vn] = rows
    return sym, out


def main() -> int:
    agg: dict[str, list] = {vn: [] for vn in VARIANTS}
    with Pool(len(PAIRS), maxtasksperchild=1) as p:
        for sym, res in p.imap_unordered(_run, PAIRS):
            for vn in VARIANTS:
                agg[vn] += res[vn]
            print(f"[{sym}] " + " ".join(f"{vn}={len(res[vn])}" for vn in VARIANTS),
                  flush=True)
    out = {}
    print(f"\n  {'변형':<14}{'n':>5}{'승률':>7}{'RR':>7}{'건당R':>9}{'ΣR':>8}"
          f"{'승리중1R미만':>13}", flush=True)
    for vn, rows in agg.items():
        r = np.array([x["r"] for x in rows], float)
        raw = np.array([x["raw"] for x in rows], float)
        w, l = raw[raw > 0], raw[raw < 0]
        wr = 100.0 * len(w) / len(raw) if len(raw) else float("nan")
        rr = (w.mean() / abs(l.mean())) if len(w) and len(l) else float("nan")
        rw = r[raw > 0]
        oc: dict[str, int] = {}
        for x in rows:
            oc[x["outcome"]] = oc.get(x["outcome"], 0) + 1
        out[vn] = {"n": len(rows), "wr": wr, "rr": rr,
                   "r_mean": float(np.nanmean(r)), "r_sum": float(np.nansum(r)),
                   "win_lt1r_pct": float(100 * np.mean(rw < 1)) if len(rw) else None,
                   "outcomes": oc}
        print(f"  {vn:<14}{len(rows):>5}{wr:>6.0f}%{rr:>7.2f}"
              f"{np.nanmean(r):>+9.3f}{np.nansum(r):>+8.1f}"
              f"{(100 * np.mean(rw < 1)) if len(rw) else float('nan'):>12.0f}%",
              flush=True)
        print(f"                청산: " + " ".join(f"{k}={v}" for k, v in oc.items()),
              flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"저장 → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
