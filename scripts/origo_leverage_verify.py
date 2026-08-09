"""#AUTONOMOUS 2026-08-06: 레버리지 결론 재검증 4종 — 적용 전 확인.

파트너: "몇 번 더 테스트해보고 나서 적용시켜보자".

1차 스윕(origo_leverage_sweep.py)은 **매 거래가 시드의 90% 를 단독으로 쓴다**고
가정했다. 실제 라이브는 7페어에 분산되어 건당 노출이 훨씬 낮다. 그 차이가
결론을 바꾸는지가 이번 검증의 핵심이다.

## 4종

① **동시 보유 반영** — 거래의 진입~청산 구간이 겹치면 자본을 나눠 쓴다(실효 size
   = 90% / 동시보유수). 라이브에 가장 가까운 형태다.
② **DD 스로틀** — 계좌 낙폭이 25% 를 넘으면 이후 리스크 ×0.7 (라이브 설정값).
③ **거래 복원추출** — 순서 셔플이 아니라 거래를 **복원추출**한다. 셔플은 "같은
   거래 묶음의 다른 배열"만 보지만, 복원추출은 "이 분포에서 뽑은 다른 거래 묶음"
   을 본다. 미래를 가정할 때 더 정직하고 보수적이다.
④ **전·후반 분할** — 앞 절반과 뒤 절반에서 최적 배율이 다르면 그 결론은 시기에
   의존한다는 뜻이다(과최적 신호).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import LIVE_BASE, PAIRS, run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

RNG = np.random.default_rng(20260807)
N_BOOT = 2000
SIZE = LIVE_BASE["size_pct"]
RUIN = 0.20
DD_PCT, DD_FACTOR = 0.25, 0.7          # 라이브 origo_dd_throttle_pct / _factor
LEVS = (3, 5, 7, 10, 12, 15, 20)


def collect():
    """(진입ms, 청산ms, raw) — 동시 보유 판정을 위해 구간을 함께 가져온다."""
    rows = []
    for sym in PAIRS:
        df5, kept, _ = run_live_parity(sym)
        idx = df5.index
        for t in kept:
            rows.append((int(idx[t.entry_idx].value),
                         int(idx[min(t.exit_idx, len(idx) - 1)].value),
                         float(t.raw_pnl_pct)))
    rows.sort(key=lambda x: x[0])
    return rows


def concurrency(rows) -> np.ndarray:
    """각 거래가 열려 있는 동안의 **평균 동시 보유 수**(자기 포함)."""
    out = np.empty(len(rows))
    for i, (s, e, _) in enumerate(rows):
        overlap = sum(1 for j, (s2, e2, _) in enumerate(rows)
                      if j != i and s2 <= e and e2 >= s)
        out[i] = 1.0 + overlap
    return out


def sim(raws, lev, *, size_scale=None, days=None, dd_throttle=False):
    """복리 시뮬 — (최종배수, MDD%, 파산여부).

    size_scale: 거래별 size 배분(동시 보유 반영). None 이면 전액.
    """
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i, r in enumerate(raws):
        sz = SIZE if size_scale is None else SIZE * size_scale[i]
        if dd_throttle and eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        step = r * sz * lev - 2.0 * TAKER_FEE_PCT * sz * lev
        eq *= (1.0 + step)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return eq, 100.0 * mdd, True
    return eq, 100.0 * mdd, False


def table(title, raws, scale, *, dd=False, resample=False):
    print(f"\n### {title}", flush=True)
    print(f"  {'배율':<5}{'자산':>10}{'MDD':>8}{'부트중앙':>10}"
          f"{'5%분위':>9}{'파산확률':>10}", flush=True)
    best = None
    n = len(raws)
    for lev in LEVS:
        e0, m0, _ = sim(raws, lev, size_scale=scale, dd_throttle=dd)
        fin = np.empty(N_BOOT)
        ruin = 0
        for k in range(N_BOOT):
            idx = (RNG.integers(0, n, size=n) if resample
                   else RNG.permutation(n))
            sc = None if scale is None else scale[idx]
            e, _, r_ = sim(raws[idx], lev, size_scale=sc, dd_throttle=dd)
            fin[k] = e
            ruin += int(r_)
        p50, p5 = np.percentile(fin, [50, 5])
        if best is None or p50 > best[1]:
            best = (lev, p50)
        print(f"  {lev:<5.0f}{e0:>9.2f}x{m0:>7.1f}%{p50:>9.2f}x"
              f"{p5:>8.2f}x{100 * ruin / N_BOOT:>9.1f}%", flush=True)
    print(f"  → 중앙값 최대: {best[0]:.0f}x", flush=True)
    return best[0]


def main() -> int:
    rows = collect()
    raws = np.array([r for _, _, r in rows], float)
    conc = concurrency(rows)
    scale = 1.0 / conc

    print("=== Origo 레버리지 재검증 (적용 전) ===", flush=True)
    print(f"  거래 {len(raws)}건 · 동시보유 평균 {conc.mean():.2f}개 "
          f"(최대 {conc.max():.0f}) → 건당 실효 size 평균 "
          f"{100 * SIZE * scale.mean():.1f}%", flush=True)

    picks = []
    picks.append(table("① 동시 보유 반영 (라이브에 가장 가까움)", raws, scale))
    picks.append(table("② ① + DD 스로틀(-25% 시 ×0.7)", raws, scale, dd=True))
    picks.append(table("③ 거래 복원추출 (미래 가정 — 더 보수적)", raws, scale,
                       dd=True, resample=True))

    h = len(raws) // 2
    print("\n### ④ 전·후반 분할 (같은 답이 나와야 시기 의존이 아니다)", flush=True)
    for lab, sub, sc in (("전반", raws[:h], scale[:h]), ("후반", raws[h:], scale[h:])):
        best = None
        for lev in LEVS:
            fin = np.empty(500)
            for k in range(500):
                idx = RNG.integers(0, len(sub), size=len(sub))
                e, _, _ = sim(sub[idx], lev, size_scale=sc[idx], dd_throttle=True)
                fin[k] = e
            m = float(np.median(fin))
            if best is None or m > best[1]:
                best = (lev, m)
        print(f"  {lab}: 최적 {best[0]:.0f}x (중앙 {best[1]:.2f}배)", flush=True)
        picks.append(best[0])

    print(f"\n  종합 — 각 검증의 최적: {picks}", flush=True)
    print("  ※ 값이 흩어지면 '최적 하나'를 고르지 말고 **안전한 하한**을 택할 것.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
