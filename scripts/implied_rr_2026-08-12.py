"""#IMPLIED 2026-08-12 (2단계): implied_fvg 성적이 진짜인가, 손익비 필터의 착시인가.

## 1단계에서 나온 것
    검출 34,338 → 시간필터 6,038 → **min_rr 2.0 에서 5,960 탈락** → 진입 후보 68건
    손익비 분포: 중앙 **0.02** · 평균 0.13 · 2.0 이상 **1.1%**

손절이 ATR 바닥(1.5×)에 강제로 벌어지는데 implied_fvg 는 몸통 갭이라 구간이 좁다.
그래서 대부분 손익비가 안 나오고, **살아남은 1.1% 는 "손절이 우연히 가까웠던" 것**
일 수 있다. 그렇다면 +0.508R 은 소스의 힘이 아니라 **필터가 만든 선택 효과**다.

7/29 피보나치에서 같은 구조를 봤다 — FVG 갭 0.123% vs ATR 손절 1.9% 라 진입가를
움직여도 R 이 안 변했다("손잡이가 연결돼 있지 않다"). 방향만 반대고 원인은 같다.

## 가르는 법
손익비 문턱을 낮춰 표본을 늘렸을 때 **건당 R 이 유지되는가**.
    유지  → 소스가 진짜 좋다. 문턱이 좋은 것까지 버리고 있었다
    붕괴  → 필터가 만든 착시. 소스 자체는 평범하다

## 변형 (사전등록)
    min_rr 2.0(현행) · 1.5 · 1.2 · 1.0
implied_fvg **한 소스만** 다른 문턱을 쓰는 게 아니라, 전체 min_rr 을 낮춘 뒤
소스별로 갈라 본다(다른 소스가 어떻게 되는지도 같이 봐야 판단이 선다).

## 판정
implied_fvg 의 건당 R 이 표본 3배 이상에서도 양수를 유지하고 구간이 0 을 넘으면
"진짜"로 본다. 그 경우에만 홀드아웃으로 넘긴다.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from live_parity import live_cfg  # noqa: E402

from aurora_ict.backtest.replay import run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT"]
BARS = 120_000
GRID = (2.0, 1.5, 1.2, 1.0)
OUT = "data/axis/implied_rr.json"
SOURCES = ("turtle_soup", "implied_fvg", "mitigation_block", "rejection_block")
RNG = np.random.default_rng(20260812)
N_BOOT, MIN_N = 20000, 30


def ci(r):
    m = np.array([r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def run(sym: str, min_rr: float) -> list[dict]:
    df = _resample(_load_full(sym)).tail(BARS)
    # 킬존은 전면 개방 — 파트너 지시이자 표본 확보를 위해(성적 차이는 없음이 확인됨)
    cfg = live_cfg(sym, {"min_rr": min_rr, "nyse_gate": False})
    tl = cached_setup_timeline(df, cfg, sym)
    bt = run_backtest_from_timeline(df, tl, cfg)
    out = []
    for t in bt.trades:
        risk = abs(float(t.entry) - float(getattr(t, "entry_sl", 0.0) or 0.0))
        if risk <= 0 or t.entry <= 0:
            continue
        conf = tuple(t.confluences)
        src = ("mmbm" if "mmbm" in conf
               else next((s for s in SOURCES if s in conf), "fvg"))
        out.append({
            "sym": sym, "src": src,
            "r": float(t.raw_pnl_pct) * float(t.entry) / risk,
            "dir": str(getattr(t.direction, "value", t.direction)).lower(),
        })
    return out


def collect() -> dict:
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    res = {}
    for rr in GRID:
        t0 = time.time()
        print(f"  [min_rr {rr}] …", flush=True)
        res[str(rr)] = [r for s in PAIRS for r in run(s, rr)]
        print(f"    {len(res[str(rr)])}건 ({time.time() - t0:.0f}초)", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f)
    return res


def main() -> int:
    print(f"=== implied_fvg — 손익비 문턱별 (BTC+ETH 최근 {BARS:,}봉, 킬존 개방)",
          flush=True)
    res = collect()

    print(f"\n  {'min_rr':<9}{'전체':>7}{'전체R':>9}"
          f"{'implied':>9}{'impR':>9}   {'implied 95% 구간':<24}"
          f"{'fvg':>6}{'fvgR':>8}{'turtle':>8}{'turR':>8}", flush=True)
    for rr in GRID:
        rows = res.get(str(rr), [])
        if not rows:
            continue
        allr = np.array([x["r"] for x in rows])
        imp = np.array([x["r"] for x in rows if x["src"] == "implied_fvg"])
        fvg = np.array([x["r"] for x in rows if x["src"] == "fvg"])
        tur = np.array([x["r"] for x in rows if x["src"] == "turtle_soup"])
        lo, hi = ci(imp) if len(imp) >= MIN_N else (float("nan"),) * 2
        print(f"  {rr:<9.1f}{len(allr):>7}{allr.mean():>+9.3f}"
              f"{len(imp):>9}{(imp.mean() if len(imp) else float('nan')):>+9.3f}"
              f"   [{lo:+.3f} ~ {hi:+.3f}]      "
              f"{len(fvg):>6}{(fvg.mean() if len(fvg) else float('nan')):>+8.3f}"
              f"{len(tur):>8}{(tur.mean() if len(tur) else float('nan')):>+8.3f}",
              flush=True)

    print("\n  판정 — implied 표본이 3배 이상 늘어도 건당 R 이 양수를 유지하고", flush=True)
    print("  구간이 0 을 넘으면 '진짜'. 무너지면 손익비 필터가 만든 착시다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
