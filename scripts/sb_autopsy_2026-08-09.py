"""#AUTONOMOUS 2026-08-09: SB(Silver Bullet) 부검 — 왜 기댓값이 0 근처인가.

파트너 결정: "유지 + 원인 규명 먼저". 끄지 않고 원인을 찾는다.

## 지금까지
| 구성 | 건당R | 95% 구간 |
|---|---|---|
| SB 단독 | −0.067 | [−0.192 ~ +0.054] |
| SB − turtle | +0.171 | [−0.037 ~ +0.384] (홀드아웃 4관문 미통과) |
| MMBM 단독 | +0.199 | [+0.118 ~ +0.278] (홀드아웃 4관문 통과) |

turtle 하나를 빼는 걸로는 안 됐다. 더 근본을 본다.

## 세 갈래
① **Phase A vs B** — 8/8 재판정에서 A +0.140R / B −0.159R 이었다. Phase B 는
   4소스(turtle·mitigation·implied·rejection) 묶음이고 라이브 진입의 68% 다.
   turtle 단독이 아니라 **묶음 전체**를 빼면 4관문을 넘는가.
② **손익 산술** — 승률 25% · RR 2.50 이면 손익분기 RR 은 (1−0.25)/0.25 = 3.00 이다.
   즉 현재 구조는 **RR 이 0.5 모자라서** 마이너스다. 승률을 올릴지 RR 을 올릴지
   어느 쪽이 현실적인지 청산 사유 분포로 판단한다.
③ **flip_min_r 효과 검증** — 7/30 에 "flip 이 0.61R 에서 익절을 절단, EV 0.61→1.17R
   개선" 진단으로 #FLIP-MIN-R(최소 1.5R)을 배포했다. 라이브 2.2 실측(승률 46%·RR 0.94)
   의 기댓값은 0.46×0.94 − 0.54 = **−0.11R** 이고, 백테 2.3 은 −0.067R 이다.
   **개선폭이 예상(+1.17R)보다 훨씬 작다.** 그 진단이 맞았는지 청산 사유로 되짚는다.
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

RNG = np.random.default_rng(20260809)
N_BOOT = 20000
N_PERM = 20000
MIN_N = 30
SRC = (("본표본 BTC+ETH", "data/conf2/runs_main.json"),
       ("홀드아웃 알트5", "data/conf2/runs_holdout.json"))


def ci(r: np.ndarray) -> tuple[float, float]:
    m = np.array([r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def perm_diff(a: np.ndarray, b: np.ndarray) -> float:
    """b 가 a 보다 큰가 — 라벨 순열 p."""
    obs = b.mean() - a.mean()
    both = np.concatenate([a, b])
    na = len(a)
    d = np.empty(N_PERM)
    for k in range(N_PERM):
        p = RNG.permutation(both)
        d[k] = p[na:].mean() - p[:na].mean()
    return float((d >= obs).mean())


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return [t for t in d["trades"]["BASE"] if t.get("r") is not None and t["r"] == t["r"]]


# ------------------------------------------------------- ① Phase A vs B
def phase_split(name: str, rows: list[dict]) -> None:
    a = np.array([t["r"] for t in rows if t["phase"] == "A"])
    b = np.array([t["r"] for t in rows if t["phase"] == "B"])
    print(f"\n  [{name}]  Phase A {len(a)}건 · Phase B {len(b)}건", flush=True)
    if len(a) < MIN_N or len(b) < MIN_N:
        print("    표본부족", flush=True)
        return
    lo, hi = ci(a)
    p = perm_diff(b, a)
    print(f"    Phase A  {a.mean():+.3f}R  승률 {100 * (a > 0).mean():>3.0f}%"
          f"  [{lo:+.3f} ~ {hi:+.3f}]  {'★0 초과' if lo > 0 else '0 포함'}", flush=True)
    print(f"    Phase B  {b.mean():+.3f}R  승률 {100 * (b > 0).mean():>3.0f}%", flush=True)
    print(f"    A−B 차 {a.mean() - b.mean():+.3f}R · 라벨순열 p={p:.4f}"
          f"  {'유의' if p < 0.05 else '유의하지 않음'}", flush=True)

    # 심볼·방향 일관성
    syms = sorted({t["sym"] for t in rows})
    win = 0
    jud = 0
    for s in syms:
        ra = np.array([t["r"] for t in rows if t["sym"] == s and t["phase"] == "A"])
        rb = np.array([t["r"] for t in rows if t["sym"] == s and t["phase"] == "B"])
        if len(ra) < 10 or len(rb) < 10:
            continue
        jud += 1
        win += int(ra.mean() > rb.mean())
    print(f"    심볼 일관성 {win}/{jud} · ", end="", flush=True)
    ok = True
    for d_ in ("long", "short"):
        ra = np.array([t["r"] for t in rows if t["dir"] == d_ and t["phase"] == "A"])
        rb = np.array([t["r"] for t in rows if t["dir"] == d_ and t["phase"] == "B"])
        if len(ra) < MIN_N or len(rb) < MIN_N:
            print(f"{d_} 표본부족 ", end="", flush=True)
            continue
        good = ra.mean() > rb.mean()
        ok &= good
        print(f"{d_} A {ra.mean():+.3f} vs B {rb.mean():+.3f} {'✓' if good else '✗'}  ",
              end="", flush=True)
    print(f"\n    → 관문 ①{'✓' if lo > 0 else '✗'} ②{'✓' if jud and win >= (jud + 1) // 2 else '✗'}"
          f" ③{'✓' if ok else '✗'} ④{'✓' if p < 0.05 else '✗'}", flush=True)


# ------------------------------------------------------- ② 손익 산술
def arithmetic(name: str, rows: list[dict]) -> None:
    r = np.array([t["r"] for t in rows])
    w = r[r > 0]
    losses = r[r <= 0]
    wr = len(w) / len(r)
    rr = w.mean() / abs(losses.mean()) if len(losses) else float("nan")
    be_rr = (1 - wr) / wr if wr > 0 else float("inf")
    be_wr = 1 / (1 + rr) if rr == rr else float("nan")
    print(f"\n  [{name}]  승률 {100 * wr:.1f}% · RR {rr:.2f} · 건당 {r.mean():+.3f}R",
          flush=True)
    print(f"    손익분기 — 현재 승률이면 RR {be_rr:.2f} 필요 (지금 {rr:.2f}, "
          f"{'충족' if rr >= be_rr else f'{be_rr - rr:.2f} 부족'})", flush=True)
    print(f"               현재 RR 이면 승률 {100 * be_wr:.1f}% 필요 (지금 {100 * wr:.1f}%, "
          f"{'충족' if wr >= be_wr else f'{100 * (be_wr - wr):.1f}%p 부족'})", flush=True)

    # 청산 사유 분포 — 어디서 끝나는가
    oc = Counter(t["outcome"] for t in rows)
    print("    청산 사유 —", flush=True)
    for k, v in oc.most_common():
        sub = np.array([t["r"] for t in rows if t["outcome"] == k])
        print(f"      {k:<12}{v:>5}건 ({100 * v / len(rows):>4.1f}%)  "
              f"건당 {sub.mean():+.3f}R  기여 {sub.sum() / len(rows):+.3f}R", flush=True)

    # 승리 중 1R 미만 — 7/30 진단(86%)의 현재 값
    if len(w):
        small = (w < 1.0).mean()
        print(f"    승리 중 1R 미만 {100 * small:.0f}%  "
              f"(7/30 라이브 2.2 실측 86% → #FLIP-MIN-R 로 고친 그 지표)", flush=True)


def main() -> int:
    print("=== SB 부검 — 기댓값이 0 근처인 이유", flush=True)

    print("\n① Phase A(정통 FVG) vs Phase B(4소스 묶음) — 묶음째 빼면 넘는가")
    for name, path in SRC:
        phase_split(name, load(path))

    print("\n② 손익 산술 — 승률을 올릴 문제인가 RR 을 올릴 문제인가")
    for name, path in SRC:
        arithmetic(name, load(path))

    print("\n  판정 — Phase A 가 4관문을 넘으면 'Phase B 묶음 제거'가 다음 배포 후보다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
