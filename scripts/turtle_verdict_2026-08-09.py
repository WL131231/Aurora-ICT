"""#AUTONOMOUS 2026-08-09: turtle_soup 단독 재검정 — SB 본체를 살릴 수 있나.

## 배경
정합 기준선에서 SB(Silver Bullet) 본체는 건당 −0.067R [−0.192 ~ +0.054] 로
0 근처이거나 마이너스다. 반면 MMBM 은 BTC+ETH +0.199R · 알트 홀드아웃 +0.113R
(4관문 전부 통과)로 확실한 흑자다.

SB 를 통째로 끄는 안을 검토하기 **전에**, SB 안의 나쁜 부분만 제거하는 안을 먼저 본다.
8/8 재판정에서 `turtle_soup` 은 본표본 −0.409R · 홀드아웃 −0.141R 로 **양 표본 부호가
일관되게 음수**였고, 라이브 진입의 **61%** 를 차지한다. 빼면 SB 가 −0.067 → +0.179R
로 뒤집혔다. 다만 그때는 변형 15개 동시 검정이라 보정 문턱(0.0028)에 걸려 p=0.0077
로 탈락했다. **이제 단독 안건**이므로 문턱은 0.05 다.

## 방법 — 변형 비교가 아니라 직접 대조
BASE 와 NOTURTLE 은 표본이 중첩된다(같은 모집단에서 일부만 제거). 독립 표본 검정을
쓰면 안 된다. 대신 **같은 모집단 안에서 turtle 붙은 거래 vs 안 붙은 거래**를 직접
비교하고, 라벨 순열로 유의성을 잰다. 이게 "이 항목이 성과를 깎는가"의 정확한 질문이다.

MMBM 홀드아웃에 쓴 것과 같은 4관문을 적용한다.
"""

from __future__ import annotations

import json

import numpy as np

RNG = np.random.default_rng(20260809)
N_BOOT = 20000
N_PERM = 20000
MIN_N = 30


def ci(r: np.ndarray) -> tuple[float, float]:
    m = np.array([r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    out = []
    for t in d["trades"]["BASE"]:
        r = t.get("r")
        if r is None or r != r:      # NaN 제외
            continue
        out.append(t)
    return out


def block(name: str, rows: list[dict]) -> dict:
    """turtle 붙은 거래 vs 안 붙은 거래 직접 대조 + 4관문."""
    r = np.array([t["r"] for t in rows], float)
    tur = np.array([bool(t["flags"]["turtle_soup"]) for t in rows])
    a, b = r[tur], r[~tur]

    print(f"\n### {name}  (진입 {len(r)}건)", flush=True)
    print(f"  turtle 포함  {len(a):>5}건 ({100 * len(a) / len(r):>4.1f}%)  "
          f"건당 {a.mean():+.3f}R  승률 {100 * (a > 0).mean():>3.0f}%", flush=True)
    print(f"  turtle 제외  {len(b):>5}건 ({100 * len(b) / len(r):>4.1f}%)  "
          f"건당 {b.mean():+.3f}R  승률 {100 * (b > 0).mean():>3.0f}%", flush=True)

    if len(a) < MIN_N or len(b) < MIN_N:
        print("  표본부족 — 판정 불가", flush=True)
        return {}

    # ① 제거 후 구간이 0 초과인가
    lo, hi = ci(b)
    g1 = lo > 0
    print(f"  ① 제거 후 95% 구간  [{lo:+.3f} ~ {hi:+.3f}]  "
          f"{'★0 초과' if g1 else '0 포함'}", flush=True)

    # ④ 라벨 순열 — turtle 라벨을 무작위로 섞어 차이 분포와 비교
    obs = b.mean() - a.mean()
    both = np.concatenate([a, b])
    na = len(a)
    dist = np.empty(N_PERM)
    for k in range(N_PERM):
        p = RNG.permutation(both)
        dist[k] = p[na:].mean() - p[:na].mean()
    pv = float((dist >= obs).mean())
    g4 = pv < 0.05
    print(f"  ④ 라벨순열  차이 {obs:+.3f}R · 무작위 {np.median(dist):+.3f}R"
          f" · p={pv:.4f}  {'유의' if g4 else '유의하지 않음'}", flush=True)

    # ② 심볼 일관성 — 심볼별로도 turtle 이 더 나쁜가
    syms = sorted({t["sym"] for t in rows})
    worse, judged = 0, 0
    for s in syms:
        ra = np.array([t["r"] for t in rows if t["sym"] == s and t["flags"]["turtle_soup"]])
        rb = np.array([t["r"] for t in rows if t["sym"] == s and not t["flags"]["turtle_soup"]])
        if len(ra) < 10 or len(rb) < 10:
            continue
        judged += 1
        worse += int(ra.mean() < rb.mean())
        print(f"     {s:<10} 포함 {len(ra):>4}건 {ra.mean():+.3f}R  |  "
              f"제외 {len(rb):>4}건 {rb.mean():+.3f}R  "
              f"{'turtle 열위' if ra.mean() < rb.mean() else 'turtle 우위'}", flush=True)
    g2 = judged > 0 and worse >= (judged + 1) // 2
    print(f"  ② 심볼 일관성 — 판정가능 {judged}개 중 turtle 열위 {worse}개", flush=True)

    # ③ 롱/숏 — 제거 효과가 한쪽 방향에서만 나오는 건 아닌가
    g3 = True
    for sgn, lab in (("long", "롱"), ("short", "숏")):
        ra = np.array([t["r"] for t in rows if t["dir"] == sgn and t["flags"]["turtle_soup"]])
        rb = np.array([t["r"] for t in rows if t["dir"] == sgn and not t["flags"]["turtle_soup"]])
        if len(ra) < MIN_N or len(rb) < MIN_N:
            print(f"     {lab} 표본부족 (포함 {len(ra)} / 제외 {len(rb)})", flush=True)
            continue
        ok = rb.mean() > ra.mean()
        g3 &= ok
        print(f"     {lab}  포함 {ra.mean():+.3f}R  |  제외 {rb.mean():+.3f}R  "
              f"차 {rb.mean() - ra.mean():+.3f}R  {'✓' if ok else '✗'}", flush=True)

    print(f"  → 관문 ①{'✓' if g1 else '✗'} ②{'✓' if g2 else '✗'} "
          f"③{'✓' if g3 else '✗'} ④{'✓' if g4 else '✗'}", flush=True)
    return {"g1": g1, "g2": g2, "g3": g3, "g4": g4, "p": pv, "diff": obs}


def main() -> int:
    print("=== turtle_soup 단독 재검정 — 제거하면 SB 가 살아나는가", flush=True)
    print("  방법: 같은 모집단 안에서 turtle 포함 vs 제외 직접 대조 (표본 중첩이라"
          " 독립검정 불가) · 라벨순열 20,000회", flush=True)

    res = {}
    for name, path in (("본표본 BTC+ETH", "data/conf2/runs_main.json"),
                       ("홀드아웃 알트 5페어", "data/conf2/runs_holdout.json")):
        try:
            res[name] = block(name, load(path))
        except FileNotFoundError:
            print(f"\n  {name} — 파일 없음", flush=True)

    print("\n=== 종합", flush=True)
    ho = res.get("홀드아웃 알트 5페어") or {}
    mn = res.get("본표본 BTC+ETH") or {}
    if ho and mn:
        both = all(ho.get(g) for g in ("g1", "g2", "g3", "g4"))
        print(f"  본표본  차이 {mn['diff']:+.3f}R p={mn['p']:.4f}", flush=True)
        print(f"  홀드아웃 차이 {ho['diff']:+.3f}R p={ho['p']:.4f}", flush=True)
        print(f"  → 홀드아웃 4관문 {'전부 통과 — 제거 권고' if both else '미통과 — 보류'}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
