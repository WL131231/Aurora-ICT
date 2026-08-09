"""#AUTONOMOUS 2026-08-07 — confluence **항목별 단독 기여도** (BTC+ETH 만).

## 왜

파트너 질문: "5등급 진입인데 4등급으로 낮추되 1+2+3+4 가 아니라 1+2+4+5 식으로
**섞어보자**. 진입 조건이 문제인지, 조건이 4개인 게 문제인지."

문턱만 낮추는 축은 이미 기각됐다(2026-08-07 직전 캠페인: 5→4 추가분 승률 25%,
p=0.989). 그러면 남은 질문은 **"점수를 구성하는 항목 하나하나가 진짜 값어치를
하는가"** 다. 값어치 없는 항목이 점수 자리를 차지하고 있으면, 그 항목이 곧
빈도를 막는 진범이다.

## 입력

conf_extract_2026-08-07.py 산출 `data/conf/trades_main.json` (min_confluence=0
재실행 · BTC 125 + ETH 133 = 258건). 홀드아웃 5페어는 **이 단계에서 열지 않는다**
(다음 단계 검증력 보존).

## 하는 일

1. **무조건부 비교** — 항목이 있는 거래 vs 없는 거래의 건당 R. 순열검정.
2. **점수 통제 비교** — 총점이 높은 거래가 좋은 건 당연하므로 통제가 필요하다.
   두 가지 통제를 다 본다(해석이 다르다):
   - (B1) **총점 고정** — 같은 총점 안에서 이 항목이 켜진 거래 vs 아닌 거래.
     "이 항목의 1점이 다른 항목의 1점보다 나은가"를 묻는다(대체재 비교).
   - (B2) **나머지 점수 고정** — 이 항목의 가중치를 뺀 점수가 같은 거래들끼리
     비교. "나머지가 같을 때 이 항목을 더하면 좋아지는가"를 묻는다(순수 증분).
     항목 고유 기여도는 B2 가 정답에 가깝다.
3. **macro 우선순위 분해** — high(+2) / normal(+1) / low(+0) 각각을 none 대비
   비교. high 가 정말 2점 값어치인지, normal 의 1점이 근거가 있는지.
4. **후보 룰 스크리닝** — 항목 기반 게이트를 사후 필터로 걸어 성적 비교.

## 통계 규약 (판정 기준 준수)

- 순열 20000회. 통계량은 **R 합 기반** — 두 군의 크기가 순열 아래서 고정이므로
  "선택군 ΣR" 과 "평균R 차이" 는 단조 동치다. 표에는 ΔR합(= 선택군이 전체 평균
  대비 벌어들인 초과 R)과 평균차를 같이 적는다. **복리 자산이득으로 판정하지
  않는다** (직전 캠페인에서 경로 운에 속아 p 0.026→0.105 로 무너진 사례).
- 다중비교: 시도한 전 검정 수를 세어 Bonferroni 문턱(0.05/시도수)을 병기한다.
- 30건 미만 셀은 ※로 표본부족을 명시한다.
- 연도별 ΣR 을 같이 찍어 몰빵 여부를 본다.

## [한계]

L1. (선행 단계에서 확인) conf0 집합은 conf5 집합의 상위집합이 **아니다**.
    replay 는 동시 포지션 1개라 문턱을 낮추면 앞쪽 저품질 거래가 뒤쪽 고품질
    거래를 밀어낸다(겹침 75~81%). 따라서 여기 후보 룰의 성적은 **스크리닝
    수치**이고, 진짜 판정은 _gate_pass 를 항목 기반으로 고쳐 재실행한 뒤에 한다.
L2. 청산 시각이 JSON 에 없어 **동시 보유 수를 실측할 수 없다**. 복리 열은
    size 0.9/2 (BTC·ETH 두 페어가 항상 겹친다는 보수 가정)로 계산하고,
    단독 가정(size 0.9)도 compound_full 로 같이 남긴다. 복리는 **참고 지표**이며
    판정에 쓰지 않는다.
L3. htf 는 백테 점수에 미반영(참고 컬럼). 여기서도 점수에 넣지 않고 별도 행으로
    기여도만 본다.
"""

from __future__ import annotations

import itertools
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

RNG = np.random.default_rng(20260807)
N_PERM = 20000
N_BOOT = 2000

SIZE = 0.9          # LIVE_BASE size_pct
LEV = 7.0           # 라이브 레버리지
CONC = 2.0          # 동시보유 보수 가정 — L2 참조 (BTC·ETH 항상 겹침)
RUIN = 0.20         # 시드 20% 파산 판정
DD_PCT, DD_FACTOR = 0.25, 0.7   # 라이브 dd_throttle

IN_PATH = "data/conf/trades_main.json"
OUT_PATH = "data/conf/single.json"

# 점수 가중 (silver_bullet + replay._boost_score 실측 — 선행 단계 검산 100% 일치)
W_MACRO = {"high": 2, "normal": 1, "low": 0, "none": 0}
ITEMS = ["ob", "macro", "sweep", "bias", "cisd", "po3"]


# ─────────────────────────────────────────────────────────── 로드/기본량

def load() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """main JSON 로드 → (레코드, R배열, raw배열, 연도배열)."""
    with open(IN_PATH, encoding="utf-8") as f:
        recs = json.load(f)
    recs.sort(key=lambda r: r["ts"])
    R = np.array([r["r"] for r in recs], float)
    raw = np.array([r["raw"] for r in recs], float)
    yr = np.array([pd.Timestamp(r["ts"]).year for r in recs], int)
    return recs, R, raw, yr


def own_weight(rec: dict, item: str) -> int:
    """그 거래에서 해당 항목이 실제로 먹은 점수(가중)."""
    if item == "macro":
        return W_MACRO[rec["flags"]["macro_pri"]]
    return 1 if rec["flags"][item] else 0


def span_months(recs: list[dict], mask: np.ndarray) -> float:
    """전체 데이터 기간(월) — 월빈도는 **표본 기간이 아니라 전체 기간**으로 나눈다.

    부분집합의 첫·마지막 거래로 나누면 드문 룰이 유리하게 보이는 착시가 생긴다.
    """
    ts = [pd.Timestamp(r["ts"]) for r in recs]
    return max((max(ts) - min(ts)).days / 30.44, 1e-9)


# ─────────────────────────────────────────────────────────── 순열검정

def perm_two_group(R: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    """두 군(있음/없음) 평균 R 차이의 순열검정.

    순열 아래 군 크기가 고정이므로 "선택군 ΣR" 과 "평균차" 는 단조 동치다.
    따라서 ΔR합 기준 검정과 같은 p 를 준다.

    Args:
        R: 거래별 R 배열.
        mask: 항목 보유 여부.

    Returns:
        (평균차 관측값, ΔR합 = 선택군이 전체평균 대비 번 초과 R, 양측 p).
    """
    n1 = int(mask.sum())
    n0 = len(R) - n1
    if n1 == 0 or n0 == 0:
        return float("nan"), float("nan"), float("nan")
    obs = R[mask].mean() - R[~mask].mean()
    dR = float(R[mask].sum() - n1 * R.mean())
    # 벡터화: 매 순열마다 n1 개를 뽑아 합만 본다(= 라벨 셔플과 동치).
    idx = np.argsort(RNG.random((N_PERM, len(R))), axis=1)[:, :n1]
    s1 = R[idx].sum(axis=1)
    null = s1 / n1 - (R.sum() - s1) / n0
    p = (1 + int(np.sum(np.abs(null) >= abs(obs) - 1e-12))) / (N_PERM + 1)
    return float(obs), dR, float(p)


def perm_stratified(R: np.ndarray, mask: np.ndarray,
                    strat: np.ndarray) -> tuple[float, float, int, list]:
    """층(strat) 안에서만 라벨을 섞는 층화 순열검정.

    통계량 T = Σ_s (n_s/N) · (평균R_있음,s − 평균R_없음,s).
    두 군이 다 존재하는 층만 쓴다.

    Returns:
        (T 관측값, 양측 p, 사용 거래수, 층별 상세 리스트).
    """
    keys = [k for k in np.unique(strat)
            if mask[strat == k].any() and (~mask[strat == k]).any()]
    if not keys:
        return float("nan"), float("nan"), 0, []
    groups, detail, tot = [], [], 0
    for k in keys:
        sel = strat == k
        rr, mm = R[sel], mask[sel]
        groups.append((rr, mm, int(mm.sum()), len(rr)))
        tot += len(rr)
        detail.append({"stratum": int(k), "n_with": int(mm.sum()),
                       "n_without": int((~mm).sum()),
                       "r_with": round(float(rr[mm].mean()), 4),
                       "r_without": round(float(rr[~mm].mean()), 4)})

    obs, null = 0.0, np.zeros(N_PERM)
    for rr, mm, n1, n in groups:
        n0 = n - n1
        obs += (n / tot) * (rr[mm].mean() - rr[~mm].mean())
        idx = np.argsort(RNG.random((N_PERM, n)), axis=1)[:, :n1]
        s1 = rr[idx].sum(axis=1)
        null += (n / tot) * (s1 / n1 - (rr.sum() - s1) / n0)
    p = (1 + int(np.sum(np.abs(null) >= abs(obs) - 1e-12))) / (N_PERM + 1)
    return float(obs), float(p), tot, detail


def perm_rule(R: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """룰 선택 부분집합 vs 같은 크기 무작위 선택 — ΔR합 순열검정.

    Returns:
        (ΔR합, 양측 p).
    """
    n1 = int(mask.sum())
    if n1 == 0 or n1 == len(R):
        return float("nan"), float("nan")
    obs = float(R[mask].sum() - n1 * R.mean())
    idx = np.argsort(RNG.random((N_PERM, len(R))), axis=1)[:, :n1]
    null = R[idx].sum(axis=1) - n1 * R.mean()
    p = (1 + int(np.sum(np.abs(null) >= abs(obs) - 1e-12))) / (N_PERM + 1)
    return obs, float(p)


# ─────────────────────────────────────────────────────────── 복리 시뮬

def sim(raws: np.ndarray, lev: float = LEV,
        conc: float = CONC) -> tuple[float, float, bool]:
    """복리 시뮬 — (최종 자산배수, MDD%, 파산여부). DD 스로틀 포함.

    origo_leverage_verify.sim 과 동일 식. size 는 동시보유 가정으로 나눈다(L2).
    """
    eq, peak, mdd = 1.0, 1.0, 0.0
    base = SIZE / conc
    for r in raws:
        sz = base * (DD_FACTOR if eq < peak * (1.0 - DD_PCT) else 1.0)
        eq *= 1.0 + (r * sz * lev - 2.0 * TAKER_FEE_PCT * sz * lev)
        if eq <= RUIN:
            return max(eq, 0.0), 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
    return eq, 100.0 * mdd, False


def boot(raws: np.ndarray) -> tuple[float, float]:
    """복원추출 부트스트랩 — (5%분위 자산배수, 파산확률%)."""
    n = len(raws)
    if n == 0:
        return float("nan"), float("nan")
    fin, ruin = np.empty(N_BOOT), 0
    for k in range(N_BOOT):
        e, _, r_ = sim(raws[RNG.integers(0, n, size=n)])
        fin[k], ruin = e, ruin + int(r_)
    return float(np.percentile(fin, 5)), 100.0 * ruin / N_BOOT


def profile(recs, R, raw, mask, name, rule, months) -> dict:
    """룰 하나의 성적표 dict."""
    n = int(mask.sum())
    if n == 0:
        return {"name": name, "rule": rule, "n": 0}
    rr, rw = R[mask], raw[mask]
    dR, p = perm_rule(R, mask)
    eq, mdd, ruined = sim(rw)
    p5, pruin = boot(rw)
    return {
        "name": name, "rule": rule, "n": n,
        "monthly": round(n / months, 3),
        "r_mean": round(float(rr.mean()), 4),
        "r_sum": round(float(rr.sum()), 2),
        "winrate": round(100.0 * float((rw > 0).mean()), 1),
        "compound": round(eq, 3), "mdd": round(mdd, 1), "ruined": bool(ruined),
        "boot_p5": round(p5, 3), "ruin_prob": round(pruin, 1),
        "dR_sum": round(dR, 2), "p_perm": round(p, 5),
        "small_sample": n < 30,
    }


# ─────────────────────────────────────────────────────────── 본문

def main() -> int:
    recs, R, raw, yr = load()
    months = span_months(recs, None)
    N = len(recs)
    print(f"=== confluence 항목별 단독 기여도 (BTC+ETH, min_confluence=0) ===")
    print(f"  거래 {N}건 · 기간 {recs[0]['ts'][:10]}~{recs[-1]['ts'][:10]} "
          f"({months:.1f}개월) · ΣR {R.sum():+.1f} · 평균R {R.mean():+.3f}",
          flush=True)

    flags = {k: np.array([r["flags"][k] for r in recs], bool) for k in ITEMS}
    pri = np.array([r["flags"]["macro_pri"] for r in recs])
    score = np.array([r["score"] for r in recs], int)
    own = {k: np.array([own_weight(r, k) for r in recs], int) for k in ITEMS}
    rest = {k: score - own[k] for k in ITEMS}
    htf = np.array([r["flags"]["htf_boost"] for r in recs], int)

    tests = 0
    result: dict = {"meta": {"n": N, "months": round(months, 1),
                             "r_sum": round(float(R.sum()), 2),
                             "r_mean": round(float(R.mean()), 4),
                             "n_perm": N_PERM}}

    # ── 1. 무조건부 항목별 비교 ────────────────────────────────────────
    print(f"\n{'=' * 92}\n[1] 무조건부 — 항목 있음 vs 없음 (건당 R)")
    print(f"  {'항목':<12}{'n(있음)':>8}{'평균R':>9}{'ΣR':>8}"
          f"{'n(없음)':>8}{'평균R':>9}{'차이':>9}{'ΔR합':>9}{'p':>9}", flush=True)
    marg = {}
    for k in ITEMS:
        m = flags[k]
        d, dR, p = perm_two_group(R, m)
        tests += 1
        note = " ※" if min(m.sum(), (~m).sum()) < 30 else ""
        marg[k] = {"n_with": int(m.sum()), "n_without": int((~m).sum()),
                   "r_mean_with": round(float(R[m].mean()), 4),
                   "r_mean_without": round(float(R[~m].mean()), 4),
                   "r_sum_with": round(float(R[m].sum()), 2),
                   "diff": round(d, 4), "dR_sum": round(dR, 2),
                   "p_perm": round(p, 5),
                   "small_sample": bool(min(m.sum(), (~m).sum()) < 30)}
        print(f"  {k:<12}{m.sum():>8}{R[m].mean():>+9.3f}{R[m].sum():>+8.1f}"
              f"{(~m).sum():>8}{R[~m].mean():>+9.3f}{d:>+9.3f}{dR:>+9.1f}"
              f"{p:>9.4f}{note}", flush=True)
    result["marginal"] = marg

    # ── 2. macro 우선순위 분해 ────────────────────────────────────────
    print(f"\n{'=' * 92}\n[2] macro 우선순위 분해 — high(+2) / normal(+1) / low(+0)")
    print("  각 우선순위를 macro 없음(none) 집단과 직접 비교한다.")
    print(f"  {'우선순위':<12}{'가중':>5}{'n':>6}{'평균R':>9}{'ΣR':>8}"
          f"{'vs none 차이':>13}{'ΔR합':>9}{'p':>9}", flush=True)
    none_m = pri == "none"
    macro_pri: dict = {}
    for p_ in ("high", "normal", "low"):
        m = pri == p_
        sub = m | none_m
        d, dR, pv = perm_two_group(R[sub], m[sub])
        tests += 1
        note = " ※" if m.sum() < 30 else ""
        macro_pri[p_] = {"weight": W_MACRO[p_], "n": int(m.sum()),
                         "r_mean": round(float(R[m].mean()), 4),
                         "r_sum": round(float(R[m].sum()), 2),
                         "diff_vs_none": round(d, 4), "dR_sum": round(dR, 2),
                         "p_perm": round(pv, 5),
                         "small_sample": bool(m.sum() < 30)}
        print(f"  {p_:<12}{W_MACRO[p_]:>5}{m.sum():>6}{R[m].mean():>+9.3f}"
              f"{R[m].sum():>+8.1f}{d:>+13.3f}{dR:>+9.1f}{pv:>9.4f}{note}",
              flush=True)
    macro_pri["none"] = {"weight": 0, "n": int(none_m.sum()),
                         "r_mean": round(float(R[none_m].mean()), 4),
                         "r_sum": round(float(R[none_m].sum()), 2)}
    print(f"  {'none':<12}{0:>5}{none_m.sum():>6}{R[none_m].mean():>+9.3f}"
          f"{R[none_m].sum():>+8.1f}", flush=True)
    result["macro_priority"] = macro_pri

    # ── 3. 점수 통제 비교 ─────────────────────────────────────────────
    print(f"\n{'=' * 92}\n[3-B1] 총점 통제 — 같은 총점 안에서 항목 유무")
    print("  해석: '이 항목의 점수가 다른 항목의 같은 점수보다 나은가'(대체재 비교)")
    print(f"  {'항목':<12}{'T(가중 평균차)':>16}{'사용 거래':>10}{'p':>9}  층별(총점:n있음/n없음)",
          flush=True)
    ctrl1 = {}
    for k in ITEMS:
        T, p, used, det = perm_stratified(R, flags[k], score)
        tests += 1
        ctrl1[k] = {"T": None if T != T else round(T, 4),
                    "p_perm": None if p != p else round(p, 5),
                    "n_used": used, "strata": det}
        cells = " ".join(f"{d['stratum']}:{d['n_with']}/{d['n_without']}"
                         for d in det)
        Ts = "  n/a" if T != T else f"{T:>+16.3f}"
        ps = "  n/a" if p != p else f"{p:>9.4f}"
        print(f"  {k:<12}{Ts}{used:>10}{ps}  {cells}", flush=True)
    result["control_total_score"] = ctrl1

    print(f"\n{'=' * 92}\n[3-B2] 나머지 점수 통제 — 이 항목 가중을 뺀 점수가 같은 거래끼리")
    print("  해석: '나머지가 같을 때 이 항목을 더하면 좋아지는가'(순수 증분·항목 고유 효과)")
    print(f"  {'항목':<12}{'T(가중 평균차)':>16}{'사용 거래':>10}{'p':>9}  층별(잔여점수:n있음/n없음)",
          flush=True)
    ctrl2 = {}
    for k in ITEMS:
        T, p, used, det = perm_stratified(R, flags[k], rest[k])
        tests += 1
        ctrl2[k] = {"T": None if T != T else round(T, 4),
                    "p_perm": None if p != p else round(p, 5),
                    "n_used": used, "strata": det}
        cells = " ".join(f"{d['stratum']}:{d['n_with']}/{d['n_without']}"
                         for d in det)
        Ts = "  n/a" if T != T else f"{T:>+16.3f}"
        ps = "  n/a" if p != p else f"{p:>9.4f}"
        print(f"  {k:<12}{Ts}{used:>10}{ps}  {cells}", flush=True)
    result["control_rest_score"] = ctrl2

    # macro high 만 따로 (잔여점수 통제) — 2점 값어치 검증
    mh = pri == "high"
    rest_mh = score - 2 * mh
    T, p, used, det = perm_stratified(R, mh, rest_mh)
    tests += 1
    result["macro_high_rest_controlled"] = {
        "T": None if T != T else round(T, 4),
        "p_perm": None if p != p else round(p, 5), "n_used": used,
        "strata": det}
    print(f"\n  [보조] macro_high 만 (잔여=총점-2 통제): T={T:+.3f} p={p:.4f} "
          f"n={used}", flush=True)

    # htf (점수 미반영·참고)
    hm = htf > 0
    d, dR, p = perm_two_group(R, hm)
    tests += 1
    result["htf_reference"] = {"n_with": int(hm.sum()), "diff": round(d, 4),
                               "dR_sum": round(dR, 2), "p_perm": round(p, 5),
                               "note": "백테 점수 미반영 — 참고값"}
    print(f"  [참고] htf_boost>0 (백테 점수 미반영): n={hm.sum()} "
          f"평균R={R[hm].mean():+.3f} vs {R[~hm].mean():+.3f} "
          f"차이={d:+.3f} p={p:.4f}", flush=True)

    # ── 4. 연도 일관성 (주요 항목) ────────────────────────────────────
    print(f"\n{'=' * 92}\n[4] 연도 일관성 — 항목 있음 집단의 연도별 ΣR / 평균R")
    years = sorted(set(yr.tolist()))
    print("  " + " " * 12 + "".join(f"{y:>14}" for y in years), flush=True)
    yearly = {}
    for k in ITEMS + ["macro_high"]:
        m = mh if k == "macro_high" else flags[k]
        cells, yy = [], {}
        for y in years:
            s = m & (yr == y)
            if s.sum() == 0:
                cells.append(f"{'-':>14}")
                yy[str(y)] = None
                continue
            cells.append(f"{R[s].sum():>+8.1f}/{s.sum():>4d}")
            yy[str(y)] = {"n": int(s.sum()), "r_sum": round(float(R[s].sum()), 2)}
        yearly[k] = yy
        print(f"  {k:<12}" + "".join(cells), flush=True)
    print("  (표기: ΣR/거래수)", flush=True)
    result["yearly"] = yearly

    # ── 5. 후보 룰 스크리닝 ───────────────────────────────────────────
    print(f"\n{'=' * 92}\n[5] 후보 룰 스크리닝 (사후 필터 — L1 때문에 판정 아님)")
    ob, sw, bi, ci, po = (flags["ob"], flags["sweep"], flags["bias"],
                          flags["cisd"], flags["po3"])
    ma = flags["macro"]
    macro_w = np.array([W_MACRO[p_] for p_ in pri], int)

    cands: list[tuple[str, str, np.ndarray]] = []
    cands.append(("현행", "score>=5", score >= 5))
    cands.append(("문턱4", "score>=4", score >= 4))
    cands.append(("문턱3", "score>=3", score >= 3))
    cands.append(("전체(문턱0)", "all", np.ones(N, bool)))

    # po3 제거 (99.2% 상수 → 실질 문턱 -1)
    s_nopo3 = score - po
    for th in (3, 4):
        cands.append((f"po3제외 문턱{th}", f"(score-po3)>={th}", s_nopo3 >= th))
    # macro high 를 1점으로 강등
    s_mflat = ob + ma.astype(int) + sw + bi + ci + po
    for th in (3, 4):
        cands.append((f"macro평탄 문턱{th}", f"(macro high도 1점)>={th}",
                      s_mflat >= th))
    # macro normal/low 를 0점으로 (high 만 인정)
    s_honly = ob + 2 * mh + sw + bi + ci + po
    for th in (4, 5):
        cands.append((f"macro=high만 문턱{th}", f"(normal/low 0점)>={th}",
                      s_honly >= th))
    # sweep 제외
    s_nosw = score - sw
    for th in (4, 5):
        cands.append((f"sweep제외 문턱{th}", f"(score-sweep)>={th}", s_nosw >= th))
    # macro_high 가중 상향(+3) — [2]에서 high 가 2점 이상의 값어치로 보였을 때의 후속
    cands.append(("macro_high3 문턱5", "(macro high=3점)>=5", score + mh >= 5))
    cands.append(("high3+sweep0 문턱4", "(high=3·sweep=0)>=4",
                  score + mh - sw >= 4))
    # bias 필수화
    cands.append(("bias 단독", "bias", bi))
    cands.append(("bias+문턱4", "bias & score>=4", bi & (score >= 4)))
    cands.append(("bias+macro_high", "bias & macro_high", bi & mh))

    # 파트너식 '항목 섞기' — 변별 4항목(ob/macro/sweep/bias) 부분집합 AND
    base4 = {"ob": ob, "macro": ma, "sweep": sw, "bias": bi}
    for r_ in (2, 3, 4):
        for combo in itertools.combinations(base4, r_):
            m = np.ones(N, bool)
            for c in combo:
                m &= base4[c]
            cands.append(("+".join(combo), " & ".join(combo), m))
    # k-of-4
    cnt4 = ob.astype(int) + ma.astype(int) + sw.astype(int) + bi.astype(int)
    for k_ in (2, 3):
        cands.append((f"4중{k_}개이상", f"count(ob,macro,sweep,bias)>={k_}",
                      cnt4 >= k_))

    rows = []
    for name, rule, m in cands:
        rows.append(profile(recs, R, raw, m, name, rule, months))
        tests += 1
    rows_sorted = sorted(rows, key=lambda r: -(r.get("r_sum") or -1e9))

    print(f"  {'후보':<18}{'n':>5}{'월빈도':>7}{'평균R':>8}{'ΣR':>8}{'승률':>6}"
          f"{'복리':>8}{'MDD':>7}{'5%분위':>8}{'파산%':>7}{'ΔR합':>8}{'p':>8}",
          flush=True)
    for r_ in rows_sorted:
        if not r_["n"]:
            continue
        note = " ※" if r_["small_sample"] else ""
        print(f"  {r_['name']:<18}{r_['n']:>5}{r_['monthly']:>7.2f}"
              f"{r_['r_mean']:>+8.3f}{r_['r_sum']:>+8.1f}{r_['winrate']:>5.0f}%"
              f"{r_['compound']:>7.2f}x{r_['mdd']:>6.1f}%{r_['boot_p5']:>7.2f}x"
              f"{r_['ruin_prob']:>6.1f}%{r_['dR_sum']:>+8.1f}{r_['p_perm']:>8.4f}"
              f"{note}", flush=True)
    result["candidates"] = rows_sorted

    # ── 5b. 현행 대비 증분 ────────────────────────────────────────────
    # 직전 캠페인 교훈: "문턱 5→4" 를 전체 성적으로 보면 흐려진다. 결정에 쓰이는
    # 것은 **새로 들어오는 거래(추가분)** 의 품질뿐이다. 그래서 룰마다 현행
    # (score>=5) 대비 추가분·제외분을 따로 떼어 검정한다.
    print(f"\n{'=' * 92}\n[5b] 현행(score>=5) 대비 증분 — 추가되는 거래만 따로 검정")
    print("  결정에 쓰이는 건 전체 성적이 아니라 **새로 들어오는 거래의 품질**이다.")
    cur = score >= 5
    print("  p 는 '추가분이 풀 평균 대비 다른가'(양측). ΔR합이 음수면 추가분이 "
          "무작위 선택보다 **나쁘다**는 뜻이다.")
    print(f"  {'후보':<18}{'추가n':>6}{'추가 평균R':>11}{'추가 ΣR':>9}{'추가 승률':>9}"
          f"{'ΔR합':>9}{'p(추가분)':>11}{'제외n':>6}{'제외 ΣR':>9}", flush=True)
    incs = []
    for name, rule, m in cands:
        add, drop = m & ~cur, cur & ~m
        if add.sum() == 0 and drop.sum() == 0:
            continue
        if add.sum() >= 1:
            dR, p = perm_rule(R, add)
            tests += 1
        else:
            dR, p = float("nan"), float("nan")
        rec = {"name": name, "rule": rule, "n_add": int(add.sum()),
               "r_mean_add": None if not add.sum() else round(float(R[add].mean()), 4),
               "r_sum_add": None if not add.sum() else round(float(R[add].sum()), 2),
               "winrate_add": None if not add.sum() else round(
                   100.0 * float((raw[add] > 0).mean()), 1),
               "dR_sum_add": None if dR != dR else round(dR, 2),
               "p_perm_add": None if p != p else round(p, 5),
               "n_drop": int(drop.sum()),
               "r_sum_drop": None if not drop.sum() else round(float(R[drop].sum()), 2),
               "small_sample": bool(add.sum() < 30)}
        incs.append(rec)
        note = " ※" if add.sum() < 30 else ""
        am = "  -" if not add.sum() else f"{R[add].mean():>+11.3f}"
        asum = "  -" if not add.sum() else f"{R[add].sum():>+9.1f}"
        awr = "  -" if not add.sum() else f"{100 * (raw[add] > 0).mean():>8.0f}%"
        ps = "  -" if p != p else f"{p:>11.4f}"
        dRs = "  -" if dR != dR else f"{dR:>+9.1f}"
        ds = "  -" if not drop.sum() else f"{R[drop].sum():>+9.1f}"
        print(f"  {name:<18}{add.sum():>6}{am}{asum}{awr}{dRs}{ps}"
              f"{drop.sum():>6}{ds}{note}", flush=True)
    result["increments"] = sorted(
        incs, key=lambda r: -(r["r_sum_add"] if r["r_sum_add"] is not None else -1e9))

    # ── 5c. score=4 골짜기 해부 ───────────────────────────────────────
    # 선행 단계 발견: 점수-수익이 단조가 아니고 score=4 가 두 세트 모두 음수다.
    # 가중 합산이 제대로 작동하면 나올 수 없는 모양이라 조합을 직접 뜯어본다.
    print(f"\n{'=' * 92}\n[5c] 점수대별 항목 조합 해부 (score 3·4·5) — 골짜기의 정체")
    combo_out: dict = {}
    for sc in (3, 4, 5):
        sel = score == sc
        print(f"\n  ── score={sc}: {sel.sum()}건 ΣR={R[sel].sum():+.1f} "
              f"평균R={R[sel].mean():+.3f}", flush=True)
        cc: dict[str, dict] = {}
        for i in np.where(sel)[0]:
            f_ = recs[i]["flags"]
            key = "+".join(
                ([f"macro_{f_['macro_pri']}"] if f_["macro_pri"] != "none" else [])
                + [k for k in ("ob", "sweep", "bias", "cisd", "po3") if f_[k]])
            cc.setdefault(key, {"n": 0, "r": 0.0, "w": 0})
            cc[key]["n"] += 1
            cc[key]["r"] += float(R[i])
            cc[key]["w"] += int(raw[i] > 0)
        for key, v in sorted(cc.items(), key=lambda x: -x[1]["n"]):
            note = " ※" if v["n"] < 30 else ""
            print(f"    {key:<46}{v['n']:>4}건 ΣR={v['r']:>+7.1f} "
                  f"평균R={v['r'] / v['n']:>+6.3f} 승률={100 * v['w'] / v['n']:>3.0f}%"
                  f"{note}", flush=True)
        combo_out[str(sc)] = {k: {"n": v["n"], "r_sum": round(v["r"], 2),
                                  "r_mean": round(v["r"] / v["n"], 4),
                                  "winrate": round(100 * v["w"] / v["n"], 1),
                                  "small_sample": v["n"] < 30}
                              for k, v in cc.items()}
    result["score_anatomy"] = combo_out

    # ── 6. 다중비교 ──────────────────────────────────────────────────
    bonf = 0.05 / tests
    result["meta"]["n_tests"] = tests
    result["meta"]["bonferroni"] = round(bonf, 6)
    print(f"\n{'=' * 92}\n[6] 다중비교")
    print(f"  이번 스크립트에서 수행한 순열검정 총 {tests}회 → "
          f"Bonferroni 문턱 0.05/{tests} = {bonf:.5f}", flush=True)
    survivors = []
    for k, v in marg.items():
        if v["p_perm"] < bonf:
            survivors.append(f"무조건부 {k} (p={v['p_perm']:.4f})")
    for k, v in ctrl2.items():
        if v["p_perm"] is not None and v["p_perm"] < bonf:
            survivors.append(f"잔여통제 {k} (p={v['p_perm']:.4f})")
    for r_ in rows_sorted:
        if r_.get("p_perm") is not None and r_["n"] and r_["p_perm"] < bonf:
            survivors.append(f"룰 {r_['name']} (p={r_['p_perm']:.4f})")
    print("  Bonferroni 생존: " + (", ".join(survivors) if survivors else "없음"),
          flush=True)
    result["meta"]["bonferroni_survivors"] = survivors

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(f"\n  저장 → {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
