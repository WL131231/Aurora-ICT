"""#PARITY 2026-08-09 — 승률·RR 이 라이브와 2배씩 어긋나는 이유가 **봇 버전 차이**인지 검증.

## 문제

정합 백테 BASE 는 승률 25% / RR 2.50 인데 라이브 실측은 승률 46% / RR 0.94 다.
방향이 정확히 반대다. 항목 출현율·빈도는 맞는데(진입은 정합) 청산 분포만 다르다.

## 가설

라이브 기준선 100건은 **2026-05-28~07-30 Origo 2.2** = #FLIP-MIN-R **배포 전**이다.
그때 flip(HTF FVG 반대 신호)은 이익 크기와 무관하게 즉시 익절해 평균 0.61R 에서
승자를 잘랐다(2026-07-30 규명: 승리의 86% 가 1R 미만). 2026-07-30 Origo 2.3 이
`flip_min_r=1.5` 를 넣어 1.5R 미만이면 flip 을 무시하고 홀드하게 고쳤고,
live_parity.LIVE_BASE 는 그 **고친 버전**을 담고 있다.

즉 백테(2.3)와 라이브 기준선(2.2)은 다른 버전일 수 있다. 그렇다면 승률·RR 차이는
이식 오류가 아니라 **버전 차이**이고, flip_min_r 만 0 으로 되돌리면 라이브 값이
재현돼야 한다.

## 검증

    FLIP0   BASE + flip_min_r=0.0   (Origo 2.2 재현)  → 승률·RR 이 46%/0.94 로 가는가
    BASE    현행 (flip_min_r=1.5)   (Origo 2.3)

재현되면: 진입·청산 이식 모두 정합. 라이브 기준선만 구버전이라 대조표를 2.2/2.3
둘로 나눠 봐야 한다는 뜻.
재현 안 되면: 청산 경로에 별도 이식 오류가 있다는 뜻 → 아래 모든 성적 숫자 보류.
"""

from __future__ import annotations

import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT"]
OUT = "data/conf2/flipver.json"
VARIANTS = {"BASE": {}, "FLIP0": {"flip_min_r": 0.0}}


def _run(sym: str):
    """한 페어: 두 변형 재생 → (승률, RR, 건당R, 승리 중 1R미만 비율)."""
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
            rows.append({"r": (float(t.raw_pnl_pct) / risk) if risk > 0 else float("nan"),
                         "raw": float(t.raw_pnl_pct), "outcome": t.outcome})
        out[vn] = rows
    return sym, out


def summarize(rows: list[dict]) -> dict:
    r = np.array([x["r"] for x in rows], float)
    raw = np.array([x["raw"] for x in rows], float)
    w, l = raw[raw > 0], raw[raw < 0]
    wr = 100.0 * len(w) / len(raw) if len(raw) else float("nan")
    rr = (w.mean() / abs(l.mean())) if len(w) and len(l) else float("nan")
    rw = r[raw > 0]
    oc: dict[str, int] = {}
    for x in rows:
        oc[x["outcome"]] = oc.get(x["outcome"], 0) + 1
    return {"n": len(rows), "wr": wr, "rr": rr, "r_mean": float(np.nanmean(r)),
            "r_sum": float(np.nansum(r)),
            "win_lt1r_pct": float(100 * np.mean(rw < 1)) if len(rw) else float("nan"),
            "outcomes": oc}


def main() -> int:
    agg: dict[str, list] = {vn: [] for vn in VARIANTS}
    with Pool(len(PAIRS), maxtasksperchild=1) as p:
        for sym, res in p.imap_unordered(_run, PAIRS):
            for vn in VARIANTS:
                agg[vn] += res[vn]
            print(f"[{sym}] " + " ".join(f"{vn}={len(res[vn])}" for vn in VARIANTS),
                  flush=True)
    out = {vn: summarize(rows) for vn, rows in agg.items()}
    print(f"\n  {'변형':<8}{'n':>6}{'승률':>7}{'RR':>7}{'건당R':>8}"
          f"{'승리중 1R미만':>13}", flush=True)
    for vn, s in out.items():
        print(f"  {vn:<8}{s['n']:>6}{s['wr']:>6.0f}%{s['rr']:>7.2f}"
              f"{s['r_mean']:>+8.3f}{s['win_lt1r_pct']:>12.0f}%", flush=True)
        print(f"           청산: " + " ".join(f"{k}={v}" for k, v in s["outcomes"].items()),
              flush=True)
    print("\n  라이브 실측(Origo 2.2, 청산 100건): 승률 46% · RR 0.94 · "
          "승리 중 1R미만 86%", flush=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"저장 → {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
