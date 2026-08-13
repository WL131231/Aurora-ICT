"""#TURTLE-3RD 2026-08-11: turtle_soup 제거 3차 검증 — 적대적.

## 여기까지
    1차 본표본 BTC+ETH  1081건 → 제거 시 836건 +0.152R · 순열 p=0.0000
    2차 홀드아웃 알트5  2397건 → 제거 시 1902건 +0.066R · 순열 p=0.0034
양쪽 다 제거 후 구간이 0 을 안 걸치고 부호도 같다(총 3,478건).

## 3차가 잡으려는 것
앞의 둘은 "turtle 을 빼면 성적이 오른다"를 보였을 뿐이다. **성적이 나쁜 거래
245건을 빼면 당연히 오른다.** turtle 이 특별한지 가리려면 다음이 필요하다:

① **플라시보** — 같은 수를 무작위로 빼도 비슷하게 오르는가.
   무작위 제거 분포에서 turtle 제거가 극단이어야 한다. (7/29 피보나치 확장에서
   플라시보도 흑자가 나와 기각한 전례 — 이후 필수 검증으로 승격됐다)
② **다중비교 보정** — 소스 4개(turtle/implied/mitigation/rejection)를 동시에 봤다.
   그중 하나가 우연히 유의할 확률을 Bonferroni 로 조인다.
③ **워크포워드** — 앞 절반에서 "turtle 이 나쁘다"를 정하고 **뒤 절반에서만** 검정.
   사후 선택 편향을 제거한다.
④ **복리·파산** — 건당 R 이 양수여도 경로가 죽을 수 있다. 라이브 사이징으로 시뮬.
⑤ **연도별 일관성** — 특정 해가 전체를 만든 것 아닌가.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

RNG = np.random.default_rng(20260811)
N_PERM = 20000
N_BOOT = 20000
SRC_FILES = (("본표본 BTC+ETH", "data/axis/src_wide_rows.json"),
             ("홀드아웃 알트5", "data/axis/src_wide_holdout.json"))
N_SOURCES_TESTED = 4          # turtle/implied/mitigation/rejection 동시 검정


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ci(r: np.ndarray) -> tuple[float, float]:
    m = np.array([r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def placebo(rows: list[dict]) -> tuple[float, float, float]:
    """① 무작위로 같은 수를 빼면? — turtle 제거가 그 분포에서 어디쯤인가."""
    r = np.array([x["r"] for x in rows])
    n_drop = sum(1 for x in rows if x["src"] == "turtle_soup")
    obs = np.array([x["r"] for x in rows if x["src"] != "turtle_soup"]).mean()
    n = len(r)
    dist = np.empty(N_PERM)
    for k in range(N_PERM):
        keep = RNG.permutation(n)[n_drop:]
        dist[k] = r[keep].mean()
    p = float((dist >= obs).mean())
    return obs, float(np.median(dist)), p


def walkforward(rows: list[dict]) -> tuple[float, float, int]:
    """③ 앞 절반에서 판단 → 뒤 절반에서만 검정."""
    half = len(rows) // 2
    a, b = rows[:half], rows[half:]
    ta = np.array([x["r"] for x in a if x["src"] == "turtle_soup"])
    if len(ta) < 10 or ta.mean() >= 0:
        return float("nan"), float("nan"), 0        # 앞 절반에서 근거 없음
    keep = np.array([x["r"] for x in b if x["src"] != "turtle_soup"])
    drop = np.array([x["r"] for x in b if x["src"] == "turtle_soup"])
    if len(keep) < 30 or len(drop) < 30:
        return float("nan"), float("nan"), 0
    obs = keep.mean() - drop.mean()
    both = np.concatenate([drop, keep])
    nd = len(drop)
    d = np.empty(N_PERM)
    for k in range(N_PERM):
        p_ = RNG.permutation(both)
        d[k] = p_[nd:].mean() - p_[:nd].mean()
    return float(obs), float((d >= obs).mean()), len(b)


def compound(rows: list[dict], drop_turtle: bool) -> tuple[float, float]:
    """④ 라이브 사이징 복리 — (최종 자산배수, 최대낙폭%)."""
    eq, peak, mdd = 1.0, 1.0, 0.0
    for x in rows:
        if drop_turtle and x["src"] == "turtle_soup":
            continue
        # 라이브 규약: 자산의 6%(MMBM 은 3%) 리스크 / 손절거리, 명목 상한 5.6배
        risk_pct = 0.03 if x["src"] == "mmbm" else 0.06
        # 손절거리는 R 정의상 1R — raw 를 R 로 환산해 두었으므로 notional 은
        # risk_pct 기준으로 직접 곱한다(sl_dist 가 원 데이터에 없음).
        pnl = x["r"] * risk_pct * eq - 2.0 * TAKER_FEE_PCT * eq * 3.0
        eq += pnl
        if eq <= 0:
            return 0.0, 100.0
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
    return eq, 100.0 * mdd


def main() -> int:
    print("=== turtle_soup 제거 3차 검증 (적대적)", flush=True)
    bonf = 0.05 / N_SOURCES_TESTED
    print(f"  다중비교 보정 문턱 = 0.05 / {N_SOURCES_TESTED} = {bonf:.4f}", flush=True)

    for name, path in SRC_FILES:
        if not os.path.exists(path):
            print(f"\n  {name} — 파일 없음", flush=True)
            continue
        rows = load(path)
        print(f"\n### {name}  ({len(rows)}건)", flush=True)

        # ① 플라시보
        obs, med, p_pl = placebo(rows)
        print(f"  ① 플라시보 — turtle 제거 {obs:+.4f}R vs 무작위 동수 제거 중앙"
              f" {med:+.4f}R · p={p_pl:.4f}"
              f"  {'통과' if p_pl < 0.05 else '★실패(무작위와 구분 안 됨)'}", flush=True)

        # ② 다중비교
        keep = np.array([x["r"] for x in rows if x["src"] != "turtle_soup"])
        drop = np.array([x["r"] for x in rows if x["src"] == "turtle_soup"])
        o2 = keep.mean() - drop.mean()
        both = np.concatenate([drop, keep])
        nd = len(drop)
        d2 = np.empty(N_PERM)
        for k in range(N_PERM):
            p_ = RNG.permutation(both)
            d2[k] = p_[nd:].mean() - p_[:nd].mean()
        p2 = float((d2 >= o2).mean())
        print(f"  ② 다중비교 — 순열 p={p2:.4f} vs 보정 문턱 {bonf:.4f}"
              f"  {'통과' if p2 < bonf else '실패'}", flush=True)

        # ③ 워크포워드
        wo, wp, nb = walkforward(rows)
        if nb:
            print(f"  ③ 워크포워드 — 뒤 절반 {nb}건 · 차 {wo:+.3f}R · p={wp:.4f}"
                  f"  {'통과' if wp < 0.05 else '실패'}", flush=True)
        else:
            print("  ③ 워크포워드 — 판정 불가(앞 절반 표본 부족)", flush=True)

        # ④ 복리
        e0, m0 = compound(rows, False)
        e1, m1 = compound(rows, True)
        print(f"  ④ 복리 — 현행 {e0:.2f}배(낙폭 {m0:.1f}%) → turtle 제거"
              f" {e1:.2f}배(낙폭 {m1:.1f}%)"
              f"  {'개선' if e1 > e0 and m1 <= m0 else '악화 또는 혼조'}", flush=True)

        # ⑤ 연도별
        print("  ⑤ 연도별 — turtle 건당R / 제거 후 전체", flush=True)
        ys = {}
        for x in rows:
            y = x.get("year")
            if y is None:
                continue
            ys.setdefault(y, []).append(x)
        if not ys:
            print("     (연도 정보 없음 — 원 데이터에 미포함)", flush=True)
        else:
            for y in sorted(ys):
                sub = ys[y]
                t = np.array([x["r"] for x in sub if x["src"] == "turtle_soup"])
                k = np.array([x["r"] for x in sub if x["src"] != "turtle_soup"])
                if len(t) < 10 or len(k) < 10:
                    continue
                print(f"     {y}  turtle {t.mean():+.3f}R ({len(t)}건) ·"
                      f" 제거 후 {k.mean():+.3f}R ({len(k)}건)", flush=True)

    print("\n  판정 — ①②③④ 를 양 표본에서 모두 넘어야 배포 후보다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
