"""#AUTONOMOUS 2026-08-07 — confluence **가중치 검증**.

## 왜

현행 confluence 는 항목별 가중 합산이고, 그 가중치(특히 macro high 의 +2)는
한 번도 검증된 적이 없다. 파트너 질문 "진입 조건이 문제인지, 조건이 4개인 게
문제인지"는 정확히 이 지점을 찌른다. 여기서 묻는 것:

  1. 가중치를 전부 +1 로 **평탄화**하면 성적이 어떻게 되나 (빈도 맞춰 비교).
  2. 항목별 **실측 기여도**로 가중치를 다시 매기면 현행보다 나은가.
  3. HTF weight 구간(4-9→+1 / 10-19→+2 / 20+→+3)이 성적과 단조인가.

## 데이터

conf_extract_2026-08-07.py 산출 data/conf/trades_main.json (BTC+ETH, 258건,
min_confluence=0 실행). **홀드아웃 5페어는 이 단계에서 열지 않는다** — 다음
단계 검증용이라 지금 보면 검증력이 사라진다.

## 방법 — 통계

- 통계량은 **ΔR합**(복리 자산이득으로 판정하면 경로 운에 속는다. 직전 캠페인에서
  p 0.026 → 0.105 로 무너진 사례).
- 순열검정 20000회. 귀무가설 = "점수는 R 과 무관" → 마스크는 고정하고 **R 을 섞는다**.
  선택 크기가 달라도 귀무분포가 크기 차이를 그대로 흡수하므로 유효하다.
- 두 종류의 p 를 나눠 본다:
    p_sel  = 이 규칙이 **무작위 선택보다** 나은가 (선별력).
    p_vs   = 이 규칙이 **현행 conf>=5 보다** 나은가 (머리끄덩이 맞대결). ← 판정용.
- 다중비교: 시도한 (체계 × 문턱) 전부를 세고 Bonferroni 문턱을 함께 찍는다.
- 30건 미만 셀은 표본부족으로 명시.

## [한계] — 결론에 반드시 달 것

L1. **사후 필터는 진짜 재실행이 아니다.** replay 는 동시 포지션 1개라 문턱을
    낮추면 앞쪽 저품질 거래가 뒤쪽 고품질 거래를 밀어낸다. conf5 실측 재실행은
    45건인데 이 데이터에 사후 conf>=5 를 걸면 32건이다(겹침 75~81%). 즉 여기
    숫자는 **스크리닝용 상대비교**이고, 절대값이 아니다.
L2. htf_w 는 근사이고 백테 점수에 **미반영**이다(라이브 전용). 게다가 아래에서
    보듯 이 근사는 포화돼 있어 라이브 계단(4-9/10-19/20+)을 그대로 검증하지 못한다.
L3. 복리 시뮬의 동시보유 판정에 쓸 **청산 시각이 데이터에 없다**. 같은 페어의
    다음 진입 시각(replay 는 exit_idx+1 로 점프하므로 청산의 상한)과 24시간 중
    작은 값으로 근사했다. 동시보유를 과대추정 → 사이즈 과소 → **보수적**.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

MAIN_PATH = "data/conf/trades_main.json"
OUT_PATH = "data/conf/weight.json"

RNG = np.random.default_rng(20260807)
N_PERM = 20000
N_BOOT = 2000

# 라이브 정합 복리 파라미터 (origo_leverage_verify.py sim 참고)
LEV = 7.0
SIZE = 0.9
DD_PCT, DD_FACTOR = 0.25, 0.7
RUIN = 0.20
MAX_HOLD_NS = 24 * 3600 * 10**9      # L3 근사 상한

MIN_N, MAX_N = 15, 150               # 문턱 스윕에서 볼 가치가 있는 거래수 범위

# ── 가중치 체계 (macro_high, macro_normal, macro_low, ob, sweep, bias, cisd, po3)
# 전부 **사전 선언**한다. 나중에 데이터 보고 몰래 추가하면 다중비교 계산이 거짓말이 된다.
SCHEMES: dict[str, tuple] = {
    "CUR":      (2, 1, 0, 1, 1, 1, 1, 1),   # 현행
    "FLAT6":    (1, 1, 1, 1, 1, 1, 1, 1),   # ①평탄화 (macro 우선순위 무시, 전부 +1)
    "FLAT4":    (1, 1, 1, 1, 1, 1, 0, 0),   # ①평탄화 + 상수항목(cisd·po3) 제거
    "MHI1":     (1, 1, 0, 1, 1, 1, 1, 1),   # macro high 를 +1 로 (=+2 주장 직접 검정)
    "MHI3":     (3, 1, 0, 1, 1, 1, 1, 1),   # macro high 를 +3 으로
    "BIAS2":    (2, 1, 0, 1, 1, 2, 1, 1),   # 실측 최대 기여 항목 bias 승격
    "NOSWEEP":  (2, 1, 0, 1, 0, 1, 1, 1),   # ②실측: sweep 기여 음수 → 가중 0
    "DATA":     (2, -1, -1, 0, 0, 2, 1, 0),  # ②실측 기여도 비례 (부호까지 반영)
}


# ────────────────────────────────────────────────────────────── 기본 유틸
def load_main() -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """main(BTC+ETH) 거래 로드 → (records, R배수, raw손익, 연도)."""
    with open(MAIN_PATH, encoding="utf-8") as fh:
        recs = json.load(fh)
    r = np.array([t["r"] for t in recs], float)
    raw = np.array([t["raw"] for t in recs], float)
    yr = np.array([int(t["ts"][:4]) for t in recs], int)
    return recs, r, raw, yr


def score_of(recs: list[dict], w: tuple) -> np.ndarray:
    """가중치 튜플로 confluence 점수 재계산.

    Args:
        recs: 거래 레코드.
        w: (macro_high, macro_normal, macro_low, ob, sweep, bias, cisd, po3).

    Returns:
        거래별 점수 배열.
    """
    m = {"high": w[0], "normal": w[1], "low": w[2], "none": 0}
    out = np.empty(len(recs), float)
    for i, t in enumerate(recs):
        f = t["flags"]
        out[i] = (m[f["macro_pri"]] + w[3] * f["ob"] + w[4] * f["sweep"]
                  + w[5] * f["bias"] + w[6] * f["cisd"] + w[7] * f["po3"])
    return out


def months_span(recs: list[dict], mask: np.ndarray) -> float:
    """선택 거래가 걸친 개월수 — 월빈도 계산용. 전체 기간 고정으로 잡는다."""
    ts = [pd.Timestamp(t["ts"]) for t in recs]
    return max((max(ts) - min(ts)).days / 30.44, 1e-9)


# ────────────────────────────────────────────────────────────── 순열검정
def perm_sel(r: np.ndarray, mask: np.ndarray, n_perm: int = N_PERM) -> float:
    """선별력 p — "이 마스크가 뽑은 ΣR 이 같은 크기 무작위 선택보다 큰가".

    귀무: 점수는 R 과 무관 → R 을 섞어도 ΣR 분포가 같다.
    """
    k = int(mask.sum())
    if k == 0:
        return 1.0
    obs = float(r[mask].sum())
    n = len(r)
    ge = 0
    for _ in range(n_perm):
        idx = RNG.permutation(n)[:k]
        if r[idx].sum() >= obs:
            ge += 1
    return (ge + 1) / (n_perm + 1)


def perm_vs(r: np.ndarray, mask: np.ndarray, base: np.ndarray,
            n_perm: int = N_PERM) -> tuple[float, float, float]:
    """맞대결 p — ΔΣR = ΣR(rule) - ΣR(base) 가 우연보다 큰가.

    마스크 둘 다 고정하고 R 만 섞는다. 두 마스크의 크기가 달라도 귀무분포가
    크기 차이(= Δn × 전체평균R)를 그대로 품으므로 비교가 유효하다.

    Returns:
        (관측 ΔΣR, 귀무 ΔΣR 평균, 단측 p).
    """
    obs = float(r[mask].sum() - r[base].sum())
    n = len(r)
    null = np.empty(n_perm)
    for i in range(n_perm):
        rp = r[RNG.permutation(n)]
        null[i] = rp[mask].sum() - rp[base].sum()
    p = (int((null >= obs).sum()) + 1) / (n_perm + 1)
    return obs, float(null.mean()), p


def perm_delta_mean(r: np.ndarray, mask: np.ndarray,
                    n_perm: int = N_PERM) -> tuple[float, float]:
    """항목 켜짐/꺼짐 평균R 차 + 양측 p (라벨 섞기).

    Returns:
        (관측 Δ평균R, 양측 p).
    """
    on, off = mask, ~mask
    if on.sum() == 0 or off.sum() == 0:
        return float("nan"), 1.0
    obs = float(r[on].mean() - r[off].mean())
    k = int(on.sum())
    n = len(r)
    ge = 0
    for _ in range(n_perm):
        idx = RNG.permutation(n)
        a = r[idx[:k]].mean()
        b = r[idx[k:]].mean()
        if abs(a - b) >= abs(obs):
            ge += 1
    return obs, (ge + 1) / (n_perm + 1)


# ────────────────────────────────────────────────────────────── 복리 시뮬
def exit_est(recs: list[dict]) -> np.ndarray:
    """청산 시각 근사 (L3) — 같은 페어 다음 진입과 진입+24h 중 이른 쪽."""
    ts = np.array([pd.Timestamp(t["ts"]).value for t in recs], np.int64)
    out = ts + MAX_HOLD_NS
    for sym in {t["sym"] for t in recs}:
        idx = np.array([i for i, t in enumerate(recs) if t["sym"] == sym])
        order = idx[np.argsort(ts[idx])]
        for a, b in zip(order[:-1], order[1:]):
            out[a] = min(out[a], ts[b])
    return out


def concurrency(starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """각 거래가 열린 동안의 동시 보유 수(자기 포함)."""
    out = np.empty(len(starts))
    for i in range(len(starts)):
        ov = int(((starts <= ends[i]) & (ends >= starts[i])).sum()) - 1
        out[i] = 1.0 + ov
    return out


def sim(raws: np.ndarray, scale: np.ndarray) -> tuple[float, float, bool]:
    """복리 시뮬 — (최종 자산배수, MDD%, 파산여부). 레버 7x · DD 스로틀 on."""
    eq = peak = 1.0
    mdd = 0.0
    for i, rr in enumerate(raws):
        sz = SIZE * scale[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        eq *= (1.0 + rr * sz * LEV - 2.0 * TAKER_FEE_PCT * sz * LEV)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return eq, 100.0 * mdd, True
    return eq, 100.0 * mdd, False


def compound(recs: list[dict], raw: np.ndarray, mask: np.ndarray,
             ends_all: np.ndarray) -> dict:
    """선택 거래로 복리 자산·MDD·파산확률·부트 5%분위."""
    sel = np.where(mask)[0]
    if len(sel) == 0:
        return {"compound": 1.0, "mdd": 0.0, "ruin": 0.0, "boot_p5": 1.0}
    starts = np.array([pd.Timestamp(recs[i]["ts"]).value for i in sel], np.int64)
    order = np.argsort(starts)
    sel = sel[order]
    starts = starts[order]
    ends = ends_all[sel]
    scale = 1.0 / concurrency(starts, ends)
    raws = raw[sel]
    eq, mdd, _ = sim(raws, scale)
    fin = np.empty(N_BOOT)
    ruin = 0
    n = len(raws)
    for k in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)      # 복원추출 = "이 분포의 다른 묶음"
        e, _, rn = sim(raws[idx], scale[idx])
        fin[k] = e
        ruin += int(rn)
    return {"compound": float(eq), "mdd": float(mdd),
            "ruin": 100.0 * ruin / N_BOOT, "boot_p5": float(np.percentile(fin, 5))}


# ────────────────────────────────────────────────────────────── 리포트 조각
def stat_row(recs, r, mask) -> dict:
    """거래수·월빈도·건당R·ΣR·승률."""
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "monthly": 0.0, "r_mean": 0.0, "r_sum": 0.0, "win": 0.0}
    mo = months_span(recs, mask)
    return {"n": n, "monthly": n / mo, "r_mean": float(r[mask].mean()),
            "r_sum": float(r[mask].sum()), "win": 100.0 * float((r[mask] > 0).mean())}


def part_a(recs, r, yr) -> dict:
    """항목별 한계 기여도 — 가중치를 논하기 전에 '기여가 있긴 한가'부터."""
    print("\n" + "=" * 82)
    print("[A] 항목별 실측 기여도 (main = BTC+ETH 258건, min_confluence=0)")
    print("=" * 82)
    items = {
        "ob":           np.array([t["flags"]["ob"] for t in recs], bool),
        "sweep":        np.array([t["flags"]["sweep"] for t in recs], bool),
        "bias":         np.array([t["flags"]["bias"] for t in recs], bool),
        "cisd":         np.array([t["flags"]["cisd"] for t in recs], bool),
        "po3":          np.array([t["flags"]["po3"] for t in recs], bool),
        "macro_high":   np.array([t["flags"]["macro_pri"] == "high" for t in recs], bool),
        "macro_normal": np.array([t["flags"]["macro_pri"] == "normal" for t in recs], bool),
        "macro_low":    np.array([t["flags"]["macro_pri"] == "low" for t in recs], bool),
    }
    cur_w = {"ob": 1, "sweep": 1, "bias": 1, "cisd": 1, "po3": 1,
             "macro_high": 2, "macro_normal": 1, "macro_low": 0}
    print(f"  {'항목':<13}{'현가중':>6}{'켜짐n':>6}{'출현율':>7}"
          f"{'켜짐평균R':>10}{'꺼짐평균R':>10}{'Δ평균R':>9}{'p(양측)':>9}   판정")
    out = {}
    n_test = len(items)
    bonf = 0.05 / n_test
    for k, m in items.items():
        if m.sum() < 2 or (~m).sum() < 2:
            print(f"  {k:<13}{cur_w[k]:>6}{int(m.sum()):>6}"
                  f"{100 * m.mean():>6.1f}%{'':>10}{'':>10}{'':>9}{'':>9}   상수·검정불가")
            out[k] = {"n": int(m.sum()), "note": "상수 — 변별력 없음"}
            continue
        dlt, p = perm_delta_mean(r, m)
        note = ("표본부족(<30)" if m.sum() < 30 or (~m).sum() < 30 else "")
        verdict = "기여 유의" if p < bonf else ("경계" if p < 0.05 else "기여 없음")
        print(f"  {k:<13}{cur_w[k]:>6}{int(m.sum()):>6}{100 * m.mean():>6.1f}%"
              f"{r[m].mean():>+10.3f}{r[~m].mean():>+10.3f}{dlt:>+9.3f}{p:>9.4f}"
              f"   {verdict} {note}")
        out[k] = {"n": int(m.sum()), "rate": float(m.mean()),
                  "r_mean_on": float(r[m].mean()), "r_mean_off": float(r[~m].mean()),
                  "delta": dlt, "p": p, "verdict": verdict}
    print(f"  ※ 항목 검정 {n_test}개 → Bonferroni 문턱 {bonf:.4f}")

    # 교락 확인 — macro_high 와 bias 가 서로를 대신 설명하는 건 아닌가.
    print("\n  [A-2] 교락 확인 (한쪽을 고정하고 다른 쪽 효과가 남는가)")
    mh, bi = items["macro_high"], items["bias"]
    for lab, sub, m2 in (("bias=1 안에서 macro_high", bi, mh),
                         ("bias=0 안에서 macro_high", ~bi, mh),
                         ("macro_high=1 안에서 bias", mh, bi),
                         ("macro_high=0 안에서 bias", ~mh, bi)):
        a, b = sub & m2, sub & ~m2
        if a.sum() < 2 or b.sum() < 2:
            print(f"    {lab:<26} 표본 부족 (n={int(a.sum())}/{int(b.sum())})")
            continue
        note = " ※표본부족" if min(a.sum(), b.sum()) < 30 else ""
        print(f"    {lab:<26} 켜짐 n={int(a.sum()):>3} {r[a].mean():+.3f} | "
              f"꺼짐 n={int(b.sum()):>3} {r[b].mean():+.3f} | "
              f"Δ={r[a].mean() - r[b].mean():+.3f}{note}")

    # 가중치 단조성 — macro 는 우선순위 3단이라 계단 검증이 가능한 유일한 항목.
    print("\n  [A-3] macro 우선순위 계단 (현행 high=+2 / normal=+1 / low=+0)")
    for pri, w in (("high", 2), ("normal", 1), ("low", 0), ("none", 0)):
        m = np.array([t["flags"]["macro_pri"] == pri for t in recs], bool)
        if not m.sum():
            continue
        note = " ※표본부족(<30)" if m.sum() < 30 else ""
        print(f"    {pri:<7} 가중 +{w}  n={int(m.sum()):>3}  "
              f"평균R={r[m].mean():+.3f}  ΣR={r[m].sum():+7.1f}  "
              f"승률={100 * (r[m] > 0).mean():>3.0f}%{note}")
    return out


def part_b(recs, r, yr, ends_all, raw) -> tuple[list[dict], int]:
    """가중치 체계 × 문턱 전수 — 현행 대비 맞대결."""
    print("\n" + "=" * 82)
    print("[B] 가중치 체계 × 문턱 전수 비교 (기준 = 현행 CUR/thr5)")
    print("=" * 82)
    base = score_of(recs, SCHEMES["CUR"]) >= 5
    b0 = stat_row(recs, r, base)
    print(f"  기준 CUR/thr5 : n={b0['n']} 월{b0['monthly']:.2f}회 "
          f"건당R={b0['r_mean']:+.3f} ΣR={b0['r_sum']:+.1f} 승률={b0['win']:.0f}%")
    print("     ※ L1 — 실제 conf5 재실행은 45건이다. 사후 필터라 32건으로 줄어든다."
          " 절대값이 아니라 **상대비교용**.")
    print(f"\n  {'체계':<9}{'문턱':>5}{'n':>5}{'월빈도':>7}{'건당R':>8}{'ΣR':>8}"
          f"{'승률':>6}{'ΔΣR':>8}{'귀무Δ':>7}{'p_vs':>8}{'p_sel':>8}")
    rows = []
    for name, w in SCHEMES.items():
        s = score_of(recs, w)
        for th in sorted(set(s.tolist())):
            mask = s >= th
            n = int(mask.sum())
            if not (MIN_N <= n <= MAX_N):
                continue
            st = stat_row(recs, r, mask)
            d, dnull, p_vs = perm_vs(r, mask, base)
            p_sel = perm_sel(r, mask)
            rows.append({"scheme": name, "thr": float(th), **st,
                         "d_rsum": d, "d_null": dnull,
                         "p_vs": p_vs, "p_sel": p_sel,
                         "w": list(w)})
            flag = " ※<30" if n < 30 else ""
            print(f"  {name:<9}{th:>5.0f}{n:>5}{st['monthly']:>7.2f}"
                  f"{st['r_mean']:>+8.3f}{st['r_sum']:>+8.1f}{st['win']:>5.0f}%"
                  f"{d:>+8.1f}{dnull:>+7.1f}{p_vs:>8.4f}{p_sel:>8.4f}{flag}")
    n_test = len(rows)
    print(f"\n  시도한 (체계 × 문턱) 조합 = {n_test}개  →  "
          f"Bonferroni 문턱 = 0.05/{n_test} = {0.05 / n_test:.5f}")
    print("  ※ 항목검정 8개까지 합치면 총 {}개 → {:.5f}".format(
        n_test + 8, 0.05 / (n_test + 8)))
    return rows, n_test


def part_c(recs, r) -> dict:
    """HTF weight 계단 단조성 — 라이브 전용 항목(백테 점수 미반영)."""
    print("\n" + "=" * 82)
    print("[C] HTF weight 구간 단조성 (라이브 전용 · 백테 점수 미반영)")
    print("=" * 82)
    hw = np.array([t["flags"]["htf_w"] for t in recs], float)
    hb = np.array([t["flags"]["htf_boost"] for t in recs], int)
    cnt = Counter(hb.tolist())
    print("  라이브 계단(4-9→+1 / 10-19→+2 / 20+→+3) 그대로 적용한 분포:")
    for b in (0, 1, 2, 3):
        m = hb == b
        if not m.sum():
            print(f"    boost +{b}: n=  0")
            continue
        note = " ※표본부족(<30)" if m.sum() < 30 else ""
        print(f"    boost +{b}: n={int(m.sum()):>4} ({100 * m.mean():>4.1f}%) "
              f"평균R={r[m].mean():+.3f} ΣR={r[m].sum():+7.1f}{note}")
    sat = 100.0 * cnt.get(3, 0) / len(recs)
    print(f"  → **포화 {sat:.1f}% 가 +3 구간**. 계단 상한 20 이 실측 htf_w 중앙값"
          f" {np.median(hw):.0f} 에 비해 터무니없이 낮다.")
    print("     즉 라이브 계단은 사실상 전 거래에 +3 을 주는 상수다 → 변별력 0,"
          " 단조성 검정 자체가 불가능.")

    print("\n  [C-2] 계단을 버리고 htf_w **5분위**로 다시 나눠 단조성을 본다:")
    q = np.percentile(hw, [20, 40, 60, 80])
    bins = np.digitize(hw, q)
    means = []
    for b in range(5):
        m = bins == b
        lo = hw[m].min() if m.sum() else 0
        hi = hw[m].max() if m.sum() else 0
        note = " ※표본부족(<30)" if m.sum() < 30 else ""
        means.append(float(r[m].mean()) if m.sum() else float("nan"))
        print(f"    Q{b + 1} (w {lo:>5.0f}~{hi:>5.0f}): n={int(m.sum()):>4} "
              f"평균R={means[-1]:+.3f} ΣR={r[m].sum():+7.1f}{note}")
    # 스피어만 상관 + 순열
    rk_w = pd.Series(hw).rank().to_numpy()
    rk_r = pd.Series(r).rank().to_numpy()
    rho = float(np.corrcoef(rk_w, rk_r)[0, 1])
    n = len(r)
    ge = 0
    for _ in range(N_PERM):
        if abs(np.corrcoef(rk_w, rk_r[RNG.permutation(n)])[0, 1]) >= abs(rho):
            ge += 1
    p_rho = (ge + 1) / (N_PERM + 1)
    mono = all(means[i] <= means[i + 1] for i in range(4))
    print(f"    스피어만 rho(htf_w, R) = {rho:+.4f}  순열 p = {p_rho:.4f}  "
          f"5분위 단조 = {'예' if mono else '아니오'}")
    print("    → HTF 지지가 클수록 성적이 좋다는 근거 " +
          ("있음" if p_rho < 0.05 and rho > 0 else "**없음**"))
    return {"saturated_pct": sat, "rho": rho, "p_rho": p_rho,
            "quintile_mean_r": means, "monotone": bool(mono),
            "median_htf_w": float(np.median(hw))}


def part_d(recs, r, raw, yr, ends_all, cands: list[dict], base_mask) -> list[dict]:
    """후보 3개 — 연도 일관성 · 추가/제외분 · 복리."""
    print("\n" + "=" * 82)
    print("[D] 후보 심층 (연도 일관성 + 추가·제외분 + 복리 시뮬)")
    print("=" * 82)
    b0 = stat_row(recs, r, base_mask)
    cb = compound(recs, raw, base_mask, ends_all)
    print(f"  [기준] CUR/thr5  n={b0['n']} 월{b0['monthly']:.2f} "
          f"건당R={b0['r_mean']:+.3f} ΣR={b0['r_sum']:+.1f} "
          f"복리={cb['compound']:.2f}x MDD={cb['mdd']:.1f}% "
          f"파산={cb['ruin']:.1f}% 부트5%={cb['boot_p5']:.2f}x")
    years = sorted(set(yr.tolist()))
    print("        연도별 ΣR: " + " ".join(
        f"{y}:{r[base_mask & (yr == y)].sum():+.1f}({int((base_mask & (yr == y)).sum())})"
        for y in years))

    out = []
    for c in cands:
        mask = score_of(recs, tuple(c["w"])) >= c["thr"]
        st = stat_row(recs, r, mask)
        cp = compound(recs, raw, mask, ends_all)
        add, drop = mask & ~base_mask, base_mask & ~mask
        print(f"\n  [{c['scheme']}/thr{c['thr']:.0f}]  가중={c['w']}")
        print(f"    n={st['n']} 월{st['monthly']:.2f} 건당R={st['r_mean']:+.3f} "
              f"ΣR={st['r_sum']:+.1f} 승률={st['win']:.0f}% | "
              f"복리={cp['compound']:.2f}x MDD={cp['mdd']:.1f}% "
              f"파산={cp['ruin']:.1f}% 부트5%={cp['boot_p5']:.2f}x")
        print(f"    vs 기준: ΔΣR={c['d_rsum']:+.1f} (귀무평균 {c['d_null']:+.1f}) "
              f"p_vs={c['p_vs']:.4f}")
        if add.sum():
            print(f"    **추가분** n={int(add.sum())} 평균R={r[add].mean():+.3f} "
                  f"ΣR={r[add].sum():+.1f} 승률={100 * (r[add] > 0).mean():.0f}% "
                  f"(전체 pool 평균 {r.mean():+.3f})"
                  f"{' ※표본부족(<30)' if add.sum() < 30 else ''}")
        else:
            print("    추가분 없음")
        if drop.sum():
            print(f"    제외분 n={int(drop.sum())} 평균R={r[drop].mean():+.3f} "
                  f"ΣR={r[drop].sum():+.1f}")
        else:
            print("    제외분 없음 (현행의 **순수 상위집합**)")
        yrs = {}
        line = []
        for y in years:
            m = mask & (yr == y)
            yrs[int(y)] = {"n": int(m.sum()), "r_sum": float(r[m].sum())}
            line.append(f"{y}:{r[m].sum():+.1f}({int(m.sum())})")
        print("    연도별 ΣR: " + " ".join(line))
        pos = sum(1 for y in years if yrs[int(y)]["r_sum"] > 0)
        print(f"    연도 일관성: {pos}/{len(years)}년 흑자"
              f"{'  → 몰빵 아님' if pos >= len(years) - 1 else '  → 편중 주의'}")
        out.append({
            "name": f"{c['scheme']}/thr{c['thr']:.0f}",
            "rule": rule_text(tuple(c["w"]), c["thr"]),
            "weights": c["w"], "thr": c["thr"],
            "n": st["n"], "monthly": st["monthly"], "r_mean": st["r_mean"],
            "r_sum": st["r_sum"], "win": st["win"],
            "compound": cp["compound"], "mdd": cp["mdd"],
            "ruin": cp["ruin"], "boot_p5": cp["boot_p5"],
            "p_perm": c["p_vs"], "p_sel": c["p_sel"],
            "d_rsum_vs_cur": c["d_rsum"], "d_rsum_null": c["d_null"],
            "added_n": int(add.sum()),
            "added_r_mean": float(r[add].mean()) if add.sum() else None,
            "dropped_n": int(drop.sum()),
            "dropped_r_mean": float(r[drop].mean()) if drop.sum() else None,
            "years": yrs,
        })
    return out, {"name": "CUR/thr5", "n": b0["n"], "monthly": b0["monthly"],
                 "r_mean": b0["r_mean"], "r_sum": b0["r_sum"], "win": b0["win"],
                 **cb}


def rule_text(w: tuple, thr: float) -> str:
    """가중치 튜플 → 사람이 읽는 규칙 문자열."""
    parts = [f"macro_high{w[0]:+g}", f"macro_normal{w[1]:+g}", f"macro_low{w[2]:+g}",
             f"ob{w[3]:+g}", f"sweep{w[4]:+g}", f"bias{w[5]:+g}",
             f"cisd{w[6]:+g}", f"po3{w[7]:+g}"]
    return " ".join(parts) + f"  >= {thr:g}"


def main() -> int:
    recs, r, raw, yr = load_main()
    print("=" * 82)
    print("confluence 가중치 검증 — main(BTC+ETH) 전용. 홀드아웃 5페어 미개봉.")
    print("=" * 82)
    print(f"  거래 {len(recs)}건 · 기간 {recs[0]['ts'][:10]} ~ {recs[-1]['ts'][:10]} "
          f"· 전체 평균R {r.mean():+.3f} · 승률 {100 * (r > 0).mean():.0f}%")

    a = part_a(recs, r, yr)
    ends_all = exit_est(recs)
    rows, n_test = part_b(recs, r, yr, ends_all, raw)

    # 후보 선정 — 파트너 지시: 홀드아웃 검증용으로 3개 이하.
    # 기준: (a) 현행 대비 ΔΣR>0, (b) 건당R 이 현행의 80% 이상(품질 유지),
    #       (c) n>=30(표본), (d) 서로 다른 가설을 대표할 것.
    base = score_of(recs, SCHEMES["CUR"]) >= 5
    b0 = stat_row(recs, r, base)
    pool = [x for x in rows
            if x["scheme"] != "CUR" and x["d_rsum"] > 0
            and x["r_mean"] >= 0.8 * b0["r_mean"] and x["n"] >= 30]
    pool.sort(key=lambda x: -x["d_rsum"])
    seen, cands = set(), []
    for x in pool:
        if x["scheme"] in seen:
            continue
        seen.add(x["scheme"])
        cands.append(x)
        if len(cands) == 3:
            break
    print("\n  [후보 선정 필터] ΔΣR>0 · 건당R >= 현행의 80% · n>=30 · 체계당 1개")
    print(f"    통과 {len(pool)}개 → 상위 3개 채택: "
          + ", ".join(f"{c['scheme']}/thr{c['thr']:.0f}" for c in cands))

    c = part_c(recs, r)
    cand_out, base_out = part_d(recs, r, raw, yr, ends_all, cands, base)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump({
            "generated": "2026-08-07",
            "dataset": "data/conf/trades_main.json (BTC+ETH 258건, min_confluence=0)",
            "note_L1": "사후 필터 — 실제 conf5 재실행은 45건, 사후는 32건. 상대비교용.",
            "baseline": base_out,
            "items": a,
            "htf": c,
            "grid": rows,
            "n_tests_grid": n_test,
            "n_tests_total": n_test + 8,
            "bonferroni": 0.05 / (n_test + 8),
            "candidates": cand_out,
        }, fh, ensure_ascii=False, indent=1)
    print(f"\n  저장 → {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
