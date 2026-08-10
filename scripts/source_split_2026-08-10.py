"""#SRC-SPLIT 2026-08-10: 진입 소스별 기여도 — 묶음이 아니라 하나씩.

## 왜
파트너 지적: "실버불릿은 시간대잖아. 시간대가 문제가 아니라 로직 문제 아닐까."
정확한 지적이다. 시간대는 오늘 두 번 검증해 **차이 없음**이 확인됐고(킬존 확대
p=0.48, 미장 제한 유지), 문제는 진입 로직 쪽이다.

지금까지는 "Phase B 묶음"으로만 봤다(turtle+implied+mitigation+rejection).
묶음은 홀드아웃에서 p=0.12 로 떨어졌는데, **그중 하나가 나머지를 가렸을 수 있다.**
개별로 갈라 본 적이 없다.

## 소스 (배타적 — 한 거래는 한 소스)
    fvg              정통 Silver Bullet 로직 (가격 공백 되돌림)
    turtle_soup      거짓 돌파 되돌림
    implied_fvg      몸통 갭
    rejection_block  꼬리 거부 구간
    mitigation_block 미티게이션 블록 재테스트

## 판정 — MMBM 홀드아웃과 같은 4관문
    ① 건당 R 95% 구간이 0 초과(또는 제거 대상이면 0 미만)
    ② 심볼 일관성   ③ 롱/숏 양쪽   ④ 순열검정 p<0.05
본표본(BTC+ETH)과 홀드아웃(알트 5)에서 **부호가 같아야** 한다.
"""

from __future__ import annotations

import json

import numpy as np

RNG = np.random.default_rng(20260810)
N_BOOT = 20000
N_PERM = 20000
MIN_N = 30
SOURCES = ("turtle_soup", "implied_fvg", "mitigation_block", "rejection_block")
SRC = (("본표본 BTC+ETH", "data/conf2/runs_main.json"),
       ("홀드아웃 알트5", "data/conf2/runs_holdout.json"))


def ci(r: np.ndarray) -> tuple[float, float]:
    m = np.array([r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def perm(a: np.ndarray, b: np.ndarray) -> float:
    """b 평균이 a 평균보다 큰가 — 라벨 순열 p."""
    obs = b.mean() - a.mean()
    both = np.concatenate([a, b])
    na = len(a)
    d = np.empty(N_PERM)
    for k in range(N_PERM):
        p = RNG.permutation(both)
        d[k] = p[na:].mean() - p[:na].mean()
    return float((d >= obs).mean())


def src_of(t: dict) -> str:
    f = t["flags"]
    for s in SOURCES:
        if f.get(s):
            return s
    return "fvg"


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return [t for t in d["trades"]["BASE"]
            if t.get("r") is not None and t["r"] == t["r"]]


def block(name: str, rows: list[dict]) -> None:
    print(f"\n### {name}  (진입 {len(rows)}건)", flush=True)
    print(f"  {'소스':<18}{'거래':>6}{'비중':>7}{'건당R':>9}{'승률':>7}"
          f"   {'95% 구간':<22}{'심볼':>6}  {'롱/숏':<18}", flush=True)

    total = np.array([t["r"] for t in rows])
    syms = sorted({t["sym"] for t in rows})
    for s in ("fvg", *SOURCES):
        sub = [t for t in rows if src_of(t) == s]
        if len(sub) < 10:
            print(f"  {s:<18}{len(sub):>6}  표본부족", flush=True)
            continue
        r = np.array([t["r"] for t in sub])
        lo, hi = ci(r) if len(r) >= MIN_N else (float("nan"), float("nan"))
        mark = "★0초과" if lo > 0 else ("적자확정" if hi < 0 else "0포함")
        # 심볼 일관성 — 부호가 몇 개 심볼에서 같은가
        same = 0
        jud = 0
        for sy in syms:
            rr = np.array([t["r"] for t in sub if t["sym"] == sy])
            if len(rr) < 10:
                continue
            jud += 1
            same += int((rr.mean() > 0) == (r.mean() > 0))
        # 롱/숏
        parts = []
        for d_ in ("long", "short"):
            rr = np.array([t["r"] for t in sub if t["dir"] == d_])
            if len(rr) >= 10:
                parts.append(f"{'롱' if d_ == 'long' else '숏'} {rr.mean():+.2f}")
        print(f"  {s:<18}{len(r):>6}{100 * len(r) / len(rows):>6.1f}%"
              f"{r.mean():>+9.3f}{100 * (r > 0).mean():>6.0f}%"
              f"   [{lo:+.3f} ~ {hi:+.3f}] {mark:<8}{same}/{jud:<4}  "
              f"{' · '.join(parts):<18}", flush=True)

    # 각 소스를 뺐을 때 나머지가 어떻게 되나 (제거 후보 판정)
    print(f"\n  {'제거하면':<18}{'남는 거래':>9}{'건당R':>9}{'95% 구간':>24}{'순열 p':>9}",
          flush=True)
    for s in SOURCES:
        keep = np.array([t["r"] for t in rows if src_of(t) != s])
        drop = np.array([t["r"] for t in rows if src_of(t) == s])
        if len(keep) < MIN_N or len(drop) < MIN_N:
            continue
        lo, hi = ci(keep)
        p = perm(drop, keep)     # 남는 쪽이 빠지는 쪽보다 나은가
        print(f"  −{s:<17}{len(keep):>9}{keep.mean():>+9.3f}"
              f"   [{lo:+.3f} ~ {hi:+.3f}]{p:>9.4f}"
              f"  {'유의' if p < 0.05 else ''}", flush=True)
    print(f"  {'(제거 없음)':<18}{len(total):>9}{total.mean():>+9.3f}", flush=True)


def main() -> int:
    print("=== 진입 소스별 기여도 — 묶음이 아니라 하나씩", flush=True)
    print("  시간대는 검증 완료(차이 없음). 로직 쪽을 본다.", flush=True)
    for name, path in SRC:
        try:
            block(name, load(path))
        except FileNotFoundError:
            print(f"\n  {name} — 파일 없음", flush=True)
    print("\n  판정 — 본표본과 홀드아웃에서 **부호가 같고** 제거 시 순열 p<0.05 여야"
          " 제거 후보다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
