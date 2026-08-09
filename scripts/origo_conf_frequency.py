"""#AUTONOMOUS 2026-08-06: BTC+ETH 에서 confluence 문턱 1~5 — 빈도를 높일 수 있나.

파트너: "BTC, ETH 두 개로 가정하고 빈도를 높여보자. 지금 5점이잖아? 1,2,3,4점".

BTC+ETH 만 쓰면 5년 45건(월 0.75)으로 빈도가 너무 낮다. confluence 문턱을 낮추면
거래가 늘어나는데, **늘어난 거래가 쓸만한가**가 관건이다.

## 총량만 보면 안 되는 이유
문턱을 낮추면 거래가 늘어 총 수익도 대개 늘어난다. 그러나 그것이 "좋은 거래가
늘어서"인지 "그냥 많이 해서"인지는 총량으로 구분되지 않는다. 그래서 **증분**을 본다:

    conf 4 로 낮췄을 때 **새로 들어온 거래만** 따로 모아 성적을 낸다.

증분이 흑자면 문턱을 낮출 가치가 있고, 적자면 낮출수록 좋은 거래를 나쁜 거래로
희석하는 것이다. 주식 실험(8/6)에서 같은 방식으로 "게이트가 좋은 걸 거른 게 아니라
걸를 게 없었다"를 판별했다.

## 판정
① 증분 거래가 흑자인가(건당 R 기준 — 손절 폭이 달라도 비교 가능)
② 복리 자산·낙폭·파산확률이 개선되는가 (7x · 동시보유 · DD 스로틀)
③ 증분이 무작위 대비 유의한가 (순열검정 p<0.05)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import LIVE_BASE, run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

SYMS = ["BTCUSDT", "ETHUSDT"]
SIZE = LIVE_BASE["size_pct"]
LEV = 7.0
DD_PCT, DD_FACTOR = 0.25, 0.7
RUIN = 0.20
RNG = np.random.default_rng(20260806)
N_BOOT = 1000
N_PERM = 20000


def rows_for(conf: int):
    """confluence 문턱 conf 에서의 거래 — 키(진입시각+심볼)로 집합 비교 가능하게."""
    out = []
    for sym in SYMS:
        df5, kept, _ = run_live_parity(sym, {"min_confluence": conf})
        idx = df5.index
        for t in kept:
            ent = idx[t.entry_idx]
            risk = abs(float(t.entry) - float(getattr(t, "entry_sl", 0.0) or 0.0))
            r_mult = (float(t.raw_pnl_pct) * float(t.entry) / risk) if risk > 0 else 0.0
            out.append({
                "key": (sym, int(ent.value)),
                "sym": sym, "entry": ent,
                "exit": idx[min(t.exit_idx, len(idx) - 1)],
                "raw": float(t.raw_pnl_pct), "r": r_mult,
            })
    out.sort(key=lambda r: r["entry"])
    return out


def sim(rows, *, idx_order=None):
    n = len(rows)
    if n < 5:
        return float("nan"), float("nan"), False
    conc = np.empty(n)
    for i, r in enumerate(rows):
        conc[i] = 1.0 + sum(
            1 for j, q in enumerate(rows)
            if j != i and q["entry"] <= r["exit"] and q["exit"] >= r["entry"]
        )
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i in (range(n) if idx_order is None else idx_order):
        sz = SIZE / conc[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        step = rows[i]["raw"] * sz * LEV - 2.0 * TAKER_FEE_PCT * sz * LEV
        eq *= (1.0 + step)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return eq, 100.0 * mdd, True
    return eq, 100.0 * mdd, False


def main() -> int:
    sets = {c: rows_for(c) for c in (1, 2, 3, 4, 5)}
    print("=== BTC+ETH · confluence 문턱별 (7x · 동시보유 · DD 스로틀) ===", flush=True)
    print(f"  {'conf':<6}{'거래':>5}{'월빈도':>8}{'건당R':>8}{'승률':>6}"
          f"{'자산':>9}{'낙폭':>8}{'부트중앙':>9}{'5%':>8}{'파산':>8}", flush=True)

    span_days = (sets[1][-1]["exit"] - sets[1][0]["entry"]).days if sets[1] else 1
    months = max(span_days / 30.4, 1e-9)

    for c in (1, 2, 3, 4, 5):
        rows = sets[c]
        n = len(rows)
        if n < 5:
            print(f"  {c:<6}{n:>5}  표본부족", flush=True)
            continue
        rs = np.array([r["r"] for r in rows])
        e0, m0, _ = sim(rows)
        fin = np.empty(N_BOOT)
        ruin = 0
        for k in range(N_BOOT):
            order = RNG.integers(0, n, size=n)
            e, _, r_ = sim(rows, idx_order=order)
            fin[k] = e
            ruin += int(r_)
        p50, p5 = np.percentile(fin, [50, 5])
        print(f"  {c:<6}{n:>5}{n / months:>8.2f}{rs.mean():>8.2f}"
              f"{100 * (rs > 0).mean():>5.0f}%{e0:>8.2f}x{m0:>7.1f}%"
              f"{p50:>8.2f}x{p5:>7.2f}x{100 * ruin / N_BOOT:>7.1f}%", flush=True)

    print("\n### 증분 — 문턱을 한 단계 낮출 때 **새로 들어오는 거래만**", flush=True)
    print(f"  {'구간':<12}{'추가':>5}{'건당R':>8}{'승률':>6}{'합계R':>8}"
          f"{'무작위중앙':>11}{'p':>7}", flush=True)
    base_r = np.array([r["r"] for r in sets[5]])
    for c in (4, 3, 2, 1):
        keys_hi = {r["key"] for r in sets[c + 1]}
        add = [r for r in sets[c] if r["key"] not in keys_hi]
        if len(add) < 5:
            print(f"  {c + 1}→{c:<10}{len(add):>5}  표본부족", flush=True)
            continue
        ar = np.array([r["r"] for r in add])
        # 순열검정 — 같은 수를 전체(conf1)에서 무작위로 뽑았을 때의 합계 R 분포
        pool = np.array([r["r"] for r in sets[1]])
        obs = ar.sum()
        dist = np.array([RNG.choice(pool, size=len(ar), replace=False).sum()
                         for _ in range(N_PERM // 10)])
        p = float((dist >= obs).mean())
        print(f"  {c + 1}→{c:<10}{len(add):>5}{ar.mean():>8.2f}"
              f"{100 * (ar > 0).mean():>5.0f}%{obs:>8.1f}{np.median(dist):>11.1f}"
              f"{p:>7.3f}", flush=True)

    print(f"\n  참고: conf 5 기준선 건당 {base_r.mean():+.2f}R · "
          f"승률 {100 * (base_r > 0).mean():.0f}%", flush=True)
    print("  판정 — 증분이 흑자(건당 R>0)이고 낙폭·파산이 나빠지지 않아야 낮출 가치가 있다.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
