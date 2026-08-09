"""#AUTONOMOUS 2026-08-09: MMBM 홀드아웃 — 탐색에 안 쓴 알트 5페어.

BTC+ETH 에서 MMBM 은 건당 +0.199R [+0.118 ~ +0.278], SB 대비 +0.260R (p=0.0002)
로 나왔다. 하지만 그 두 심볼은 배선 판정(8/6)에도 쓴 표본이라 **탐색에 오염**돼 있다.

바로 어제(8/8) confluence 재판정에서 본표본 통과 항목이 홀드아웃에서 **전멸**했다
(15개 변형 중 통과 0개, 효과가 1/3 로 줄거나 부호가 뒤집힘). 그러니 BTC+ETH 결과
하나로 "MMBM 은 좋다"고 결론 내리면 같은 함정을 반복한다.

## 판정
① 알트 5페어에서도 건당 R 95% 구간이 0 초과인가
② 심볼별 부호가 일관인가 (한 심볼이 전체를 업고 가는 것 아닌가)
③ 롱/숏 분리해도 양쪽 다 살아있나 — 상승장 롱 편향이면 국면이 번 것이다
④ 순열검정 — 무작위 진입 대비 유의한가

라이브 정합 조건(SMT✗ 스윕✗)으로만 잰다. 나머지 조합은 배선 근거가 아니다.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmbm_full as M  # noqa: E402

HOLDOUT = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
RNG = np.random.default_rng(20260809)
N_BOOT = 20000
N_PERM = 20000
MIN_N = 30          # 30건 미만은 판정하지 않는다 (표준 기준선)


def ci(r: np.ndarray) -> tuple[float, float]:
    m = np.array([r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> int:
    print("=== MMBM 홀드아웃 — 라이브 정합 조건(SMT✗ 스윕✗) · 알트 5페어", flush=True)
    print(f"  {'심볼':<10}{'거래':>6}{'월빈도':>8}{'건당R':>9}{'승률':>7}"
          f"   {'95% 구간':<20}{'판정':<8}", flush=True)

    per: dict[str, np.ndarray] = {}
    allr: list[float] = []
    alld: list[int] = []
    for sym in HOLDOUT:
        try:
            df, tr = M.backtest(sym, use_smt=False, require_sweep=False, detail=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym:<10} 실패 — {type(e).__name__}: {str(e)[:50]}", flush=True)
            continue
        if not tr:
            print(f"  {sym:<10}{0:>6}  진입 0건", flush=True)
            continue
        r = np.array([t[4] for t in tr], float)      # r_mult
        d = [int(t[2]) for t in tr]
        months = (tr[-1][3] - tr[0][0]) / 86400000 / 30.4
        per[sym] = r
        allr += r.tolist()
        alld += d
        if len(r) < MIN_N:
            print(f"  {sym:<10}{len(r):>6}{len(r) / max(months, 1e-9):>8.2f}"
                  f"{r.mean():>+9.3f}{100 * (r > 0).mean():>6.0f}%"
                  f"   {'':<20}{'표본부족':<8}", flush=True)
            continue
        lo, hi = ci(r)
        mark = "★0초과" if lo > 0 else ("적자" if hi < 0 else "0포함")
        print(f"  {sym:<10}{len(r):>6}{len(r) / max(months, 1e-9):>8.2f}"
              f"{r.mean():>+9.3f}{100 * (r > 0).mean():>6.0f}%"
              f"   [{lo:+.3f} ~ {hi:+.3f}]    {mark:<8}", flush=True)

    if not allr:
        print("\n  홀드아웃 진입 0건 — 판정 불가", flush=True)
        return 0

    r = np.array(allr)
    d = np.array(alld)
    lo, hi = ci(r)
    print(f"\n  합계 {len(r)}건 · 건당 {r.mean():+.3f}R · 승률 {100 * (r > 0).mean():.0f}%"
          f" · 95% [{lo:+.3f} ~ {hi:+.3f}]", flush=True)

    pos = sum(1 for s, v in per.items() if len(v) >= MIN_N and v.mean() > 0)
    judged = sum(1 for v in per.values() if len(v) >= MIN_N)
    print(f"  ② 심볼 일관성 — 판정가능 {judged}개 중 흑자 {pos}개", flush=True)

    print("  ③ 롱/숏", flush=True)
    for sgn, lab in ((1, "롱"), (-1, "숏")):
        sub = r[d == sgn]
        if len(sub) < MIN_N:
            print(f"     {lab}  {len(sub)}건 — 표본부족", flush=True)
            continue
        l2, h2 = ci(sub)
        print(f"     {lab}  {len(sub):>5}건  건당 {sub.mean():+.3f}R"
              f"  승률 {100 * (sub > 0).mean():>3.0f}%  [{l2:+.3f} ~ {h2:+.3f}]", flush=True)

    # ④ 순열 — 부호를 무작위로 뒤집어(귀무: 방향 정보 없음) 평균 R 분포와 비교
    obs = r.mean()
    dist = np.array([(r * RNG.choice([-1.0, 1.0], size=len(r))).mean()
                     for _ in range(N_PERM)])
    p = float((dist >= obs).mean())
    print(f"  ④ 부호순열 — 관측 {obs:+.3f}R · 무작위 {np.median(dist):+.3f}R · p={p:.4f}",
          flush=True)

    print("\n  판정 — ① 구간 0 초과 ② 심볼 과반 흑자 ③ 롱숏 양쪽 생존 ④ p<0.05"
          " 를 모두 넘어야 BTC+ETH 결과가 표본 밖에서도 성립한다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
