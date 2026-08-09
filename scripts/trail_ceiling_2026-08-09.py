"""#AUTONOMOUS 2026-08-09: 트레일링 축 — SB 청산 마지막 후보. 상한부터.

## 왜 여기가 남았나
SB 손익분기 부족분 0.37R 을 메울 후보 중 진입선별·소스제거·TP목표는 전부 닫혔다.
트레일링만 메커니즘이 달라(최고점 추종 후 되돌림 청산) 미탐색으로 남았다.

## 현행 설정에서 발견한 것
`trail_trigger=2.0R` · `trail_dist=1.5R` 인데, MFE 실측에서 **2R 도달률이 21%** 다.
즉 **79% 의 거래에서 트레일이 무장조차 되지 않는다.** 게다가 무장돼도 거리가 1.5R 이라
최고점에서 1.5R 을 되돌려줘야 청산된다(2R 에서 무장 → 0.5R 에서 청산).

## 방법 — 봉별 경로가 필요하다
MFE 하나로는 트레일을 못 짠다. 최고점이 **언제** 나왔는지, 그 뒤 어떻게 되돌렸는지가
결과를 정한다. 그래서 거래마다 봉별 유리/불리 경로를 R 단위로 저장한 뒤 시뮬한다.

시뮬 규칙(라이브와 같은 순서):
  · 매 봉 불리 극값이 −1R 이하면 손절 (트레일 무장 전)
  · 유리 극값이 트리거 T 이상이면 무장, 이후 스톱 = (최고 유리 − D)
  · 무장 후 불리가 스톱 이하로 오면 그 값에 청산
  · 끝까지 안 걸리면 실제 실현값 사용
보수적으로 **같은 봉에서 유리·불리가 모두 닿으면 불리 우선**(낙관 편향 차단).

## 사후 최적화 방지
격자를 미리 등록하고(T 5개 × D 5개), 본표본에서는 **상한만** 본다.
0.37R 을 못 넘으면 이 축도 닫는다. 넘으면 홀드아웃 4관문으로 판정한다.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import run_live_parity  # noqa: E402

SYMS = ["BTCUSDT", "ETHUSDT"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "data", "axis", "_trail_paths.pkl")
TRIG = (0.5, 1.0, 1.5, 2.0, 2.5)      # 사전등록 — 사후에 넓히지 않는다
DIST = (0.3, 0.5, 0.75, 1.0, 1.5)
NEED = 0.37                            # SB 손익분기 부족분


def collect() -> list[dict]:
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            rows = pickle.load(f)
        print(f"  (캐시 재사용 — {len(rows)}건)", flush=True)
        return rows

    rows = []
    for sym in SYMS:
        print(f"  {sym} 경로 추출 중 …", flush=True)
        df5, kept, _ = run_live_parity(sym)
        hi = df5["high"].to_numpy(float)
        lo = df5["low"].to_numpy(float)
        for t in kept:
            e = float(t.entry)
            sl = float(getattr(t, "entry_sl", 0.0) or 0.0)
            risk = abs(e - sl)
            if risk <= 0 or e <= 0:
                continue
            i, j = int(t.entry_idx), min(int(t.exit_idx), len(df5) - 1)
            if j <= i:
                continue
            is_long = str(getattr(t.direction, "value", t.direction)).lower() == "long"
            h_, l_ = hi[i:j + 1], lo[i:j + 1]
            # 봉별 유리(fav)·불리(adv) 극값을 R 단위로. 둘 다 양수 방향.
            fav = (h_ - e) / risk if is_long else (e - l_) / risk
            adv = (e - l_) / risk if is_long else (h_ - e) / risk
            rows.append({
                "sym": sym, "dir": 1 if is_long else -1,
                "r": float(t.raw_pnl_pct) * e / risk,
                "fav": fav.astype(np.float32), "adv": adv.astype(np.float32),
            })
        print(f"    {sym} 누적 {len(rows)}건", flush=True)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(rows, f)
    return rows


def sim_trail(row: dict, trig: float, dist: float) -> float:
    """트레일 시뮬 — 같은 봉 동시 충족 시 불리 우선(보수적)."""
    fav, adv = row["fav"], row["adv"]
    peak = 0.0
    armed = False
    for k in range(len(fav)):
        # ① 무장 전 손절 (−1R)
        if not armed and adv[k] >= 1.0:
            return -1.0
        # ② 무장 후 스톱 — 불리를 먼저 본다
        if armed:
            stop = peak - dist
            # 스톱이 진입가 아래면 그대로 −1R 손절선이 유효
            if stop <= -1.0 and adv[k] >= 1.0:
                return -1.0
            if -adv[k] <= stop:
                return float(stop)
        # ③ 유리 갱신 · 무장
        if fav[k] > peak:
            peak = float(fav[k])
        if not armed and peak >= trig:
            armed = True
    return float(row["r"])          # 끝까지 안 걸리면 실제 실현값


def main() -> int:
    print("=== 트레일링 축 구조적 상한", flush=True)
    rows = collect()
    if not rows:
        print("  거래 0건", flush=True)
        return 0

    base = np.array([x["r"] for x in rows])
    n = len(base)
    print(f"\n  거래 {n}건 · 현행 건당 {base.mean():+.3f}R"
          f" (현행 설정 트리거 2.0R · 거리 1.5R)", flush=True)

    print(f"\n  {'':<8}" + "".join(f"{'D=' + format(d, '.2f'):>10}" for d in DIST), flush=True)
    best = None
    grid = {}
    for t in TRIG:
        cells = []
        for d in DIST:
            v = np.array([sim_trail(r, t, d) for r in rows])
            grid[(t, d)] = v
            delta = v.mean() - base.mean()
            cells.append(f"{delta:>+10.3f}")
            if best is None or v.mean() > best[2]:
                best = (t, d, v.mean())
        print(f"  T={t:<5.1f}" + "".join(cells), flush=True)

    print("\n  (표는 현행 대비 건당 R 변화)", flush=True)
    if best:
        t, d, m = best
        v = grid[(t, d)]
        print(f"\n  격자 최고 — 트리거 {t:.1f}R · 거리 {d:.2f}R → 건당 {m:+.3f}R"
              f" (현행 대비 {m - base.mean():+.3f}R)", flush=True)
        print(f"    승률 {100 * (v > 0).mean():.0f}% · 손절률 {100 * (v <= -0.999).mean():.0f}%",
              flush=True)
        ceil_ = m - base.mean()
        print(f"\n  판정 — 필요치 {NEED:.2f}R 대비 상한 {ceil_:+.3f}R", flush=True)
        print(f"    {'→ 열려 있다. 홀드아웃 4관문으로 판정 진행.' if ceil_ >= NEED else '→ 못 넘는다. 이 축도 닫는다.'}",
              flush=True)
        print("    ※ 격자 최고는 사후 argmax 다. 이 값 자체는 근거가 아니며"
              " 홀드아웃 통과가 유일한 근거다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
