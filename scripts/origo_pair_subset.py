"""#AUTONOMOUS 2026-08-06: 페어 조합별 낙폭·생존 — BTC+ETH 만 vs 현행 7페어.

파트너: "알트 빼고 BTC 랑 ETH 만 해봐봐". 낙폭 해부에서 **LINK 가 최악 손실 10건 중
5건**을 차지한 것이 발단이다.

## 페어를 줄이면 두 가지가 동시에 일어난다 — 둘 다 봐야 한다
① 나쁜 페어가 빠져 손실이 준다 (기대 효과)
② **분산이 사라져 건당 노출이 커진다** — 동시보유가 줄면 자본을 덜 나누므로
   같은 배율이라도 실효 노출이 올라가 변동성이 커진다. 7페어 동시보유 평균 1.50
   → 2페어면 1.0 에 가까워지고 건당 size 가 60% → 90% 로 뛴다.
②를 빼고 ①만 보면 "알트 빼면 좋다"는 잘못된 결론이 나온다.

## 판정
거래 수가 크게 줄어 표본이 얇아지므로, 최종 수익 하나로 고르지 않는다.
**무작위 페어 조합**(같은 개수)과 비교해 "BTC+ETH 라서" 인지 "페어를 줄여서" 인지
가른다. 7페어에서 2개를 무작위로 뽑은 21개 조합 전부와 대조한다.
"""

from __future__ import annotations

import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import LIVE_BASE, PAIRS, run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

SIZE = LIVE_BASE["size_pct"]
LEV = 7.0
DD_PCT, DD_FACTOR = 0.25, 0.7
RUIN = 0.20
RNG = np.random.default_rng(20260806)
N_BOOT = 1000

_CACHE: dict[str, list] = {}


def rows_for(sym: str):
    if sym not in _CACHE:
        df5, kept, _ = run_live_parity(sym)
        idx = df5.index
        _CACHE[sym] = [
            {"sym": sym, "entry": idx[t.entry_idx],
             "exit": idx[min(t.exit_idx, len(idx) - 1)],
             "raw": float(t.raw_pnl_pct)}
            for t in kept
        ]
    return _CACHE[sym]


def build(syms):
    rows = [r for s in syms for r in rows_for(s)]
    rows.sort(key=lambda r: r["entry"])
    return rows


def sim(rows, *, shuffle_idx=None):
    """복리 — 동시보유로 size 분할 + DD 스로틀. (최종, 낙폭%, 파산, 동시보유평균)."""
    n = len(rows)
    if n < 5:
        return float("nan"), float("nan"), False, float("nan")
    conc = np.empty(n)
    for i, r in enumerate(rows):
        conc[i] = 1.0 + sum(
            1 for j, q in enumerate(rows)
            if j != i and q["entry"] <= r["exit"] and q["exit"] >= r["entry"]
        )
    order = range(n) if shuffle_idx is None else shuffle_idx
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i in order:
        sz = SIZE / conc[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        step = rows[i]["raw"] * sz * LEV - 2.0 * TAKER_FEE_PCT * sz * LEV
        eq *= (1.0 + step)
        if eq <= 0:
            return 0.0, 100.0, True, float(conc.mean())
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return eq, 100.0 * mdd, True, float(conc.mean())
    return eq, 100.0 * mdd, False, float(conc.mean())


def evaluate(label, syms):
    rows = build(syms)
    eq, mdd, ruin, conc = sim(rows)
    n = len(rows)
    fin = np.empty(N_BOOT)
    ruins = 0
    for k in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)       # 복원추출 — 미래 가정
        e, _, r_, _ = sim(rows, shuffle_idx=idx)
        fin[k] = e
        ruins += int(r_)
    p50, p5 = np.percentile(fin, [50, 5])
    print(f"  {label:<22}{n:>5}{conc:>7.2f}{eq:>9.2f}x{mdd:>7.1f}%"
          f"{p50:>9.2f}x{p5:>8.2f}x{100 * ruins / N_BOOT:>8.1f}%", flush=True)
    return {"eq": eq, "mdd": mdd, "p50": float(p50), "n": n}


def main() -> int:
    print("=== 페어 조합별 (7x · 동시보유 반영 · DD 스로틀 · 복원추출) ===", flush=True)
    print(f"  {'조합':<22}{'거래':>5}{'동시':>7}{'자산':>9}{'낙폭':>8}"
          f"{'부트중앙':>9}{'5%':>8}{'파산':>8}", flush=True)

    base = evaluate("현행 7페어", PAIRS)
    alts = [p for p in PAIRS if p not in ("BTCUSDT", "ETHUSDT")]
    be = evaluate("BTC+ETH 만", ["BTCUSDT", "ETHUSDT"])
    evaluate("LINK 만 제외(6)", [p for p in PAIRS if p != "LINKUSDT"])
    evaluate("BTC+ETH+SOL", ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    evaluate("알트만(5)", alts)

    print("\n### 2페어 조합 전수 — 'BTC+ETH 라서' 인가 '2개라서' 인가", flush=True)
    print(f"  {'조합':<22}{'거래':>5}{'동시':>7}{'자산':>9}{'낙폭':>8}"
          f"{'부트중앙':>9}{'5%':>8}{'파산':>8}", flush=True)
    scores = []
    for a, b in itertools.combinations(PAIRS, 2):
        r = evaluate(f"{a[:-4]}+{b[:-4]}", [a, b])
        scores.append((r["p50"], f"{a[:-4]}+{b[:-4]}", r))
    scores.sort(reverse=True)
    rank = [i for i, (_, name, _) in enumerate(scores, 1) if name == "BTC+ETH"]
    print(f"\n  BTC+ETH 는 21개 2페어 조합 중 {rank[0] if rank else '?'}위 "
          f"(중앙값 기준)", flush=True)
    print(f"  상위 3: {[(n, round(p, 2)) for p, n, _ in scores[:3]]}", flush=True)
    print(f"  하위 3: {[(n, round(p, 2)) for p, n, _ in scores[-3:]]}", flush=True)
    print(f"\n  현행 7페어 중앙 {base['p50']:.2f}배 / 낙폭 {base['mdd']:.1f}%"
          f"  ↔  BTC+ETH 중앙 {be['p50']:.2f}배 / 낙폭 {be['mdd']:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
