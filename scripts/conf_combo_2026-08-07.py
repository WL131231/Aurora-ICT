"""#AUTONOMOUS 2026-08-07 — confluence **항목 조합(AND)** 전수 스크리닝.

## 파트너 질문

"지금은 5등급 진입인데 4등급으로 낮추되 1+2+3+4 가 아니라 1+2+4+5 / 1+3+4+5 식으로
**섞어보자**. 진입 조건이 문제인지, 조건이 4개인 게 문제인지."

→ 이 스크립트는 그 질문을 두 조각으로 갈라 답한다.
   (a) **개수 효과** — 요구 항목 수 k 가 늘수록 좋아지는가?
   (b) **조합 효과** — 같은 k 안에서 어떤 항목을 고르느냐가 더 중요한가?

## 항목 집합 (선행 단계 실측 — 브리핑의 5항목과 다르다)

백테(replay.py / live_parity)의 confluence 는 **6항목**이고 라이브의 htf 는 미구현이다:

    ob(+1) · macro(high+2/normal+1/low+0) · sweep(+1) · bias(+1) · cisd(+1) · po3(+1)

이 중 **po3 는 출현율 99.2% 로 사실상 상수**다(킬존 필터가 이미 AMD distribution
시간대만 남긴다). AND 조건에 po3 를 넣어봐야 거래가 0.8% 밖에 안 줄어 모든 조합이
쌍으로 중복된다. 그래서 po3 를 빼고

    ITEMS = (ob, macro, sweep, bias, cisd)

5항목으로 전수한다 — 파트너가 말한 "5종 중 k개" 구조와 정확히 맞는다.
k=2(10) · k=3(10) · k=4(5) · k=5(1) = **26가지**(사전 계획).
여기에 개수 효과 곡선의 왼쪽 끝을 보기 위해 k=1(5가지)를 **탐색적으로** 추가한다.
→ 실제 시도 31가지. 다중비교 문턱은 26 기준(0.05/26=0.00192)과 31 기준(0.00161)을
   모두 보고하고, 더불어 **max-T 순열 FWER p** 를 계산한다(Bonferroni 보다 정확).

## 통계 (판정 기준 4 준수 — 복리 자산이 아니라 R 로 판정)

관측 통계량 = **ΔR합** = ΣR(선택 거래) − n_선택 × 평균R(전체 258건).
즉 "같은 개수를 무작위로 뽑았을 때보다 R 을 얼마나 더 벌었나".
귀무가설 = 항목 플래그와 성과가 무관. 검정 = **거래별 R 을 셔플**(20000회).
이때 셔플 분포는 크기 n 의 무작위 부분집합 분포와 정확히 같다.
모든 조합이 **같은 셔플 draw** 를 공유하므로 max-T FWER 보정이 가능하다.

## 복리 평가 (라이브 정합 — 보고용, 판정용 아님)

레버 7x · size_pct 0.9 · 동시보유 분할 · DD 스로틀(-25%→×0.7) · 파산 시드 20%.
origo_leverage_verify.sim 이식.

## [한계] — 결론에 반드시 안고 갈 것

L1(★). 이 데이터는 min_confluence=0 재실행 산출이고, replay 는 동시 포지션 1개라
    진입 후 exit_idx+1 로 점프한다. 즉 conf0 집합은 conf5 집합의 상위집합이 **아니다**
    (실측 겹침 BTC 75% / ETH 81%). 그래서 여기 "현행(합산≥5)" 기준선은 32건이지만
    진짜 conf5 재실행은 45건이다. **본 표는 조합 스크리닝 전용**이고, 유망 조합은
    _gate_pass 를 항목 기반으로 고쳐 재실행 후 순열검정해야 확정된다.
L2. 청산 시각이 JSON 에 없어 동시보유 판정은 근사다 — 보유구간을
    [진입, min(진입+24h, 같은 페어 다음 진입)] 으로 잡았다(replay 가 청산 후에야
    다음 진입을 하므로 다음 진입은 청산 시각의 상한). 상한을 쓰므로 동시보유를
    **과대**추정 → 사이즈를 보수적으로 깎는 방향이다. 페어가 2개뿐이라 영향은 작다.
L3. holdout(5페어)은 이 단계에서 열지 않는다 — 다음 단계 검증용.
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

MAIN_PATH = "data/conf/trades_main.json"
OUT_PATH = "data/conf/combo.json"

# 계열 A = 사전 계획(브리핑 그대로). macro = 매크로 구간이면 켬(low 포함).
# 계열 B = 사후 추가. 1차 결과에서 **현행 합산>=5 집합의 96.9% 가 macro_high** 임을
#   발견해, macro 를 "high 우선순위일 때만" 으로 좁혀 다시 전수한다. 사후 탐색이므로
#   다중비교 계산에 반드시 포함한다(아래 unique family).
SETS = {
    "A": ("ob", "macro", "sweep", "bias", "cisd"),
    "B": ("ob", "macro_high", "sweep", "bias", "cisd"),
}
KS_PLANNED = (2, 3, 4, 5)          # 사전 계획 26가지 (계열 A)
KS_EXPLORE = (1,)                  # 탐색적 5가지 (개수 곡선의 왼쪽 끝)

N_PERM = 20_000
N_BOOT = 2_000
RNG = np.random.default_rng(20260807)

# 라이브 정합 자금관리 (LIVE_BASE + 라이브 강제값)
LEV = 7.0
SIZE = 0.9
DD_PCT, DD_FACTOR = 0.25, 0.7
RUIN = 0.20
HOLD_CAP_NS = 24 * 3600 * 1_000_000_000     # L2 근사 상한 24h


# ────────────────────────────── 로드 ──────────────────────────────

def load_main() -> dict:
    """trades_main.json → 넘파이 묶음.

    Returns:
        {r, raw, ts_ns, sym, year, flags(dict of bool array), score, scale}
    """
    with open(MAIN_PATH, encoding="utf-8") as f:
        recs = json.load(f)
    recs.sort(key=lambda x: x["ts"])
    ts = pd.to_datetime([x["ts"] for x in recs], utc=True)
    ts_ns = ts.astype("int64").to_numpy()
    d = {
        "n": len(recs),
        "r": np.array([x["r"] for x in recs], float),
        "raw": np.array([x["raw"] for x in recs], float),
        "win": np.array([x["raw"] > 0 for x in recs], bool),
        "ts_ns": ts_ns,
        "sym": np.array([x["sym"] for x in recs]),
        "year": ts.year.to_numpy(),
        "score": np.array([x["score"] for x in recs], int),
        "months": float((ts_ns[-1] - ts_ns[0]) / 1e9 / 86400 / 30.44),
    }
    flags = {}
    for k in ("ob", "macro", "sweep", "bias", "cisd", "po3"):
        flags[k] = np.array([bool(x["flags"][k]) for x in recs], bool)
    flags["macro_high"] = np.array(
        [x["flags"]["macro_pri"] == "high" for x in recs], bool)
    d["flags"] = flags
    d["scale"] = conc_scale(ts_ns, d["sym"])
    return d


def conc_scale(ts_ns: np.ndarray, sym: np.ndarray) -> np.ndarray:
    """동시보유 분할 배수 1/동시보유수 (L2 근사).

    보유구간 = [진입, min(진입+24h, 같은 페어의 다음 진입)].
    replay 는 청산 후에야 같은 페어의 다음 진입을 하므로 다음 진입 시각은
    청산 시각의 **상한**이다 → 동시보유 과대추정(보수적).

    Args:
        ts_ns: 진입 시각 (ns, 정렬됨).
        sym: 페어 이름.

    Returns:
        거래별 1/(평균 동시보유수) 배열.
    """
    n = len(ts_ns)
    end = ts_ns + HOLD_CAP_NS
    for s in set(sym.tolist()):
        idx = np.flatnonzero(sym == s)
        for a, b in zip(idx[:-1], idx[1:]):
            end[a] = min(end[a], ts_ns[b])
    out = np.empty(n)
    for i in range(n):
        ov = np.sum((ts_ns <= end[i]) & (end > ts_ns[i])) - 1
        out[i] = 1.0 / (1.0 + ov)
    return out


# ────────────────────────────── 복리 시뮬 ──────────────────────────────

def sim(raws: np.ndarray, scale: np.ndarray) -> tuple[float, float, bool]:
    """복리 시뮬 — (최종배수, MDD%, 파산여부). origo_leverage_verify.sim 이식."""
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i in range(len(raws)):
        sz = SIZE * scale[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        eq *= 1.0 + (raws[i] * sz * LEV - 2.0 * TAKER_FEE_PCT * sz * LEV)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return eq, 100.0 * mdd, True
    return eq, 100.0 * mdd, False


def boot(raws: np.ndarray, scale: np.ndarray) -> tuple[float, float]:
    """거래 복원추출 부트 — (5%분위 자산, 파산확률%)."""
    n = len(raws)
    if n == 0:
        return 0.0, 0.0
    fin = np.empty(N_BOOT)
    ruin = 0
    for k in range(N_BOOT):
        i = RNG.integers(0, n, size=n)
        e, _, rn = sim(raws[i], scale[i])
        fin[k] = e
        ruin += int(rn)
    return float(np.percentile(fin, 5)), 100.0 * ruin / N_BOOT


# ────────────────────────────── 조합 ──────────────────────────────

def mask_of(d: dict, combo: tuple[str, ...]) -> np.ndarray:
    """AND 규칙 마스크 — combo 의 모든 항목이 켜진 거래만 True."""
    m = np.ones(d["n"], bool)
    for k in combo:
        m &= d["flags"][k]
    return m


def evaluate(d: dict, m: np.ndarray, perm_r: np.ndarray) -> dict:
    """마스크 하나의 전체 지표.

    Args:
        d: load_main 산출.
        m: 선택 마스크.
        perm_r: (N_PERM, n) 셔플된 R 행렬.

    Returns:
        지표 dict (null 분포 포함 — max-T 보정에 쓴다).
    """
    n = int(m.sum())
    mu = float(d["r"].mean())
    if n == 0:
        return dict(n=0, monthly=0.0, r_mean=float("nan"), r_sum=0.0,
                    win=float("nan"), dr=0.0, compound=1.0, mdd=0.0,
                    p5=1.0, ruin=0.0, p_perm=1.0, z=float("-inf"),
                    null=np.zeros(N_PERM), by_year={})
    r = d["r"][m]
    r_sum = float(r.sum())
    dr = r_sum - n * mu
    e0, mdd, _ = sim(d["raw"][m], d["scale"][m])
    p5, ruin = boot(d["raw"][m], d["scale"][m])
    null = perm_r[:, m].sum(axis=1) - n * mu
    sd = float(null.std()) or 1e-12
    p = float((null >= dr - 1e-12).mean())
    by_year = {}
    for y in sorted(set(d["year"][m].tolist())):
        sel = m & (d["year"] == y)
        by_year[int(y)] = dict(n=int(sel.sum()), r_sum=float(d["r"][sel].sum()))
    return dict(
        n=n, monthly=n / d["months"], r_mean=float(r.mean()), r_sum=r_sum,
        win=float(d["win"][m].mean() * 100), dr=float(dr),
        compound=float(e0), mdd=float(mdd), p5=float(p5), ruin=float(ruin),
        p_perm=p, z=float((dr - null.mean()) / sd), null=null, by_year=by_year,
    )


def main() -> int:
    d = load_main()
    print(f"=== confluence 항목 조합(AND) 전수 — BTC+ETH {d['n']}건 "
          f"({d['months']:.1f}개월) ===", flush=True)
    print(f"  전체 평균R {d['r'].mean():+.4f} · ΣR {d['r'].sum():+.1f} · "
          f"승률 {100 * d['win'].mean():.0f}% · 월빈도 {d['n'] / d['months']:.2f}",
          flush=True)
    print(f"  동시보유 평균 {1 / d['scale'].mean():.2f}개 → 건당 실효 size "
          f"평균 {100 * SIZE * d['scale'].mean():.1f}% (L2 근사)", flush=True)
    print("  [항목 출현율] " + " · ".join(
        f"{k} {100 * d['flags'][k].mean():.1f}%"
        for k in ("ob", "macro", "macro_high", "sweep", "bias", "cisd", "po3")),
        flush=True)

    # 모든 조합이 공유하는 셔플 draw (max-T FWER 보정을 위해 필수)
    print(f"\n[순열] R 셔플 {N_PERM}회 생성 ...", flush=True)
    perm_r = np.empty((N_PERM, d["n"]))
    for j in range(N_PERM):
        perm_r[j] = RNG.permutation(d["r"])

    combos: list[tuple[str, str, tuple[str, ...]]] = []
    for fam_id, items in SETS.items():
        for k in KS_PLANNED + KS_EXPLORE:
            tag = ("planned" if (k in KS_PLANNED and fam_id == "A")
                   else "explore")
            for c in itertools.combinations(items, k):
                combos.append((fam_id, tag, c))

    # macro 를 포함하지 않는 조합은 계열 A/B 에서 **글자 그대로 같은 규칙**이라
    # 한 번만 센다(항목 집합 기준 dedupe). 서로 다른 조합이 우연히 같은 거래를
    # 고르는 경우(cisd 계열)는 별개 시도이므로 합치지 않는다.
    rows, seen = [], {}
    for fam_id, tag, c in combos:
        key = frozenset(c)
        if key in seen:
            seen[key]["fams"] += fam_id
            continue
        e = evaluate(d, mask_of(d, c), perm_r)
        e.update(tag=tag, k=len(c), name="+".join(c), rule=" AND ".join(c),
                 fams=fam_id)
        seen[key] = e
        rows.append(e)

    # 기준선 — 현행 합산 문턱들 (검색 대상 아님, 참조용)
    refs = []
    for thr in (3, 4, 5, 6):
        m = d["score"] >= thr
        e = evaluate(d, m, perm_r)
        e.update(tag="ref", k=-1, name=f"현행 합산>={thr}", rule=f"score>={thr}")
        refs.append(e)
    e_all = evaluate(d, np.ones(d["n"], bool), perm_r)
    e_all.update(tag="ref", k=-1, name="무필터(conf0)", rule="all")
    refs.append(e_all)

    # max-T FWER — 검색 대상(planned+explore)만 family 에 넣는다
    fam = [r for r in rows]
    Z = np.stack([(r["null"] - r["null"].mean()) / (r["null"].std() or 1e-12)
                  for r in fam])
    maxz = Z.max(axis=0)
    for r in fam:
        r["p_fwer"] = float((maxz >= r["z"] - 1e-12).mean())
    for r in refs:
        r["p_fwer"] = float("nan")

    n_planned = sum(1 for r in rows if r["tag"] == "planned")
    n_all = len(rows)                      # 중복 제거 후 **고유** 규칙 수
    bonf_p, bonf_a = 0.05 / n_planned, 0.05 / n_all

    # ────────────── 출력 ──────────────
    def show(title, items, *, sort=False):
        print(f"\n{'=' * 100}\n### {title}", flush=True)
        print(f"  {'규칙':<31}{'k':>2}{'n':>5}{'월':>6}{'평균R':>8}{'ΣR':>8}"
              f"{'ΔR':>8}{'승률':>6}{'복리':>10}{'MDD':>7}{'5%':>8}"
              f"{'파산':>7}{'p':>8}{'pFWER':>8}", flush=True)
        it = sorted(items, key=lambda r: -r["dr"]) if sort else items
        for r in it:
            flag = " ※<30" if 0 < r["n"] < 30 else ("  n=0" if r["n"] == 0 else "")
            pf = "  -  " if r["p_fwer"] != r["p_fwer"] else f"{r['p_fwer']:.4f}"
            print(f"  {r['name']:<31}{r['k'] if r['k'] > 0 else 0:>2}{r['n']:>5}"
                  f"{r['monthly']:>6.2f}{r['r_mean']:>+8.3f}{r['r_sum']:>+8.1f}"
                  f"{r['dr']:>+8.1f}{r['win']:>5.0f}%{r['compound']:>9.2f}x"
                  f"{r['mdd']:>6.0f}%{r['p5']:>7.2f}x{r['ruin']:>6.0f}%"
                  f"{r['p_perm']:>8.4f}{pf:>8}{flag}", flush=True)

    show("기준선 (참조 — 검색 family 밖)", refs)
    for k in (1, 2, 3, 4, 5):
        sub = [r for r in rows if r["k"] == k]
        tag = "탐색적(개수곡선 왼쪽 끝)" if k == 1 else "사전계획 A + 사후 B"
        show(f"k={k} ({tag}, 고유 {len(sub)}가지)", sub, sort=True)

    # ────────────── 개수 효과 vs 조합 효과 ──────────────
    print(f"\n{'=' * 100}\n### 개수 효과 vs 조합 효과 (파트너 질문 직답)", flush=True)
    print(f"  {'k':>2}{'조합수':>7}{'표본충분(n>=30)':>16}{'평균R 중앙':>11}"
          f"{'평균R 범위':>20}{'ΔR 중앙':>10}{'ΔR 범위':>18}", flush=True)
    kstat = {}
    for k in (1, 2, 3, 4, 5):
        sub = [r for r in rows if r["k"] == k and r["n"] >= 30]
        allsub = [r for r in rows if r["k"] == k]
        if not sub:
            print(f"  {k:>2}{len(allsub):>7}{0:>16}   — 전 조합 표본부족", flush=True)
            kstat[k] = None
            continue
        rm = np.array([r["r_mean"] for r in sub])
        dv = np.array([r["dr"] for r in sub])
        kstat[k] = dict(n_combo=len(sub), med=float(np.median(rm)),
                        lo=float(rm.min()), hi=float(rm.max()),
                        spread=float(rm.max() - rm.min()),
                        dmed=float(np.median(dv)))
        print(f"  {k:>2}{len(allsub):>7}{len(sub):>16}{np.median(rm):>+11.3f}"
              f"   {rm.min():+.3f} ~ {rm.max():+.3f}{np.median(dv):>+10.1f}"
              f"   {dv.min():+.1f} ~ {dv.max():+.1f}", flush=True)

    ok = [k for k in (1, 2, 3, 4, 5) if kstat.get(k)]
    if len(ok) >= 2:
        across = np.array([kstat[k]["med"] for k in ok])
        within = np.array([kstat[k]["spread"] for k in ok])
        print(f"\n  k 간 중앙값 변동폭(개수 효과) = {across.max() - across.min():+.3f} R", flush=True)
        print(f"  k 내 조합 간 최대 편차(조합 효과) = {within.max():+.3f} R "
              f"(k={ok[int(np.argmax(within))]})", flush=True)
        print("  → 조합 효과 / 개수 효과 = "
              f"{within.max() / max(across.max() - across.min(), 1e-9):.1f}배", flush=True)

    # ────────────── 연도 일관성 (상위 후보) ──────────────
    print(f"\n{'=' * 100}\n### 연도 일관성 — ΔR 상위 5개 + 기준선", flush=True)
    top = sorted([r for r in rows if r["n"] >= 30], key=lambda r: -r["dr"])[:5]
    years = sorted(set(d["year"].tolist()))
    print(f"  {'규칙':<31}" + "".join(f"{y:>11}" for y in years), flush=True)
    for r in top + [x for x in refs if x["rule"] == "score>=5"]:
        cells, pos = [], 0
        for y in years:
            v = r["by_year"].get(y)
            cells.append(f"{v['r_sum']:+.1f}/{v['n']}" if v else "-")
            pos += int(bool(v) and v["r_sum"] > 0)
        print(f"  {r['name']:<31}" + "".join(f"{c:>11}" for c in cells)
              + f"   흑자 {pos}/{len(years)}년", flush=True)

    # ────────────── 현행 규칙 해부 (★핵심 발견) ──────────────
    print(f"\n{'=' * 100}\n### 현행 합산>=5 규칙 해부 — 이게 정말 '5항목 가중합'인가?",
          flush=True)
    m5 = d["score"] >= 5
    fh, fb = d["flags"]["macro_high"], d["flags"]["bias"]
    print(f"  macro_high 출현율: 전체 {100 * fh.mean():.1f}% → "
          f"합산>=5 집합 안에서 {100 * fh[m5].mean():.1f}%", flush=True)
    print("  이유: po3 가 99.2% 로 +1 을 공짜로 얹고 macro_high 가 +2 → "
          "3점이 자동. 나머지 2점만 ob/sweep/bias 에서 채우면 된다.", flush=True)
    print("  macro_high 없이 5점을 만들려면 ob+macro_norm+sweep+bias 를 **전부** "
          "켜야 해서 사실상 안 나온다.", flush=True)
    print("\n  [합산>=5 집합의 실제 구성 패턴]", flush=True)
    keys = ("ob", "macro_high", "macro", "sweep", "bias", "cisd")
    pats: dict[tuple, list[int]] = {}
    for i in np.flatnonzero(m5):
        p = tuple(k for k in keys if d["flags"][k][i] and
                  not (k == "macro" and d["flags"]["macro_high"][i]))
        pats.setdefault(p, []).append(i)
    for p, ii in sorted(pats.items(), key=lambda x: -len(x[1])):
        print(f"    {len(ii):>3}건  평균R {np.mean(d['r'][ii]):+6.2f}  "
              f"{'+'.join(p)}", flush=True)

    mB = fh & fb
    print("\n  [현행 합산>=5  vs  macro_high+bias (2조건 AND)]", flush=True)
    print(f"    합산>=5 {int(m5.sum())}건 / macro_high+bias {int(mB.sum())}건 / "
          f"교집합 {int((m5 & mB).sum())}건 → 사실상 같은 규칙", flush=True)
    for lbl, m in (("2조건이 새로 잡는 것", mB & ~m5),
                   ("2조건이 놓치는 것", m5 & ~mB)):
        if m.sum():
            print(f"    {lbl:<20} n={int(m.sum()):>2}  ΣR {d['r'][m].sum():+6.1f}  "
                  f"평균R {d['r'][m].mean():+.3f}", flush=True)
    mx = fh & ~fb
    print(f"    bias 가 쳐내는 macro_high 거래: n={int(mx.sum())} "
          f"ΣR {d['r'][mx].sum():+.1f} 평균R {d['r'][mx].mean():+.3f} "
          "(※표본 5건 — bias 의 기여는 확정 불가)", flush=True)

    # ────────────── 다중비교 ──────────────
    print(f"\n{'=' * 100}\n### 다중비교", flush=True)
    print(f"  사전계획(계열 A, k=2..5) = {n_planned}가지 "
          f"→ Bonferroni 문턱 0.05/{n_planned} = {bonf_p:.5f}", flush=True)
    print(f"  실제 시도 = 계열 A 31 + 계열 B 31 = 62, 그중 macro 를 안 쓰는 15가지는 "
          f"A/B 가 같은 규칙 → **고유 {n_all}가지** "
          f"→ Bonferroni 문턱 0.05/{n_all} = {bonf_a:.5f}", flush=True)
    print("  ※ 실제로는 조합끼리 강하게 상관(포함관계)이라 Bonferroni 는 과보수적. "
          "max-T 순열 FWER p 를 함께 보고했다.", flush=True)
    sig_raw = [r for r in rows if r["p_perm"] < 0.05 and r["n"] >= 30]
    sig_b = [r for r in rows if r["p_perm"] < bonf_a and r["n"] >= 30]
    sig_f = [r for r in rows if r["p_fwer"] < 0.05 and r["n"] >= 30]
    print(f"  p<0.05 통과(표본충분): {len(sig_raw)}개 — "
          f"{[r['name'] for r in sig_raw] or '없음'}", flush=True)
    print(f"  Bonferroni({bonf_a:.5f}) 통과: {len(sig_b)}개 — "
          f"{[r['name'] for r in sig_b] or '없음'}", flush=True)
    print(f"  max-T FWER p<0.05 통과: {len(sig_f)}개 — "
          f"{[r['name'] for r in sig_f] or '없음'}", flush=True)
    thin = [r for r in rows if 0 < r["n"] < 30]
    print(f"  표본부족(<30) 셀: {len(thin)}/{n_all}개 "
          f"— cisd 포함 조합 {sum(1 for r in rows if 'cisd' in r['rule'])}개가 "
          "구조적으로 여기 해당(cisd 출현율 2.7%)", flush=True)

    # ────────────── 저장 ──────────────
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    def dump(r):
        return dict(name=r["name"], rule=r["rule"], tag=r["tag"],
                    fams=r.get("fams", ""), k=r["k"],
                    n=r["n"], monthly=round(r["monthly"], 3),
                    r_mean=round(r["r_mean"], 4) if r["n"] else None,
                    r_sum=round(r["r_sum"], 3), dr=round(r["dr"], 3),
                    win=round(r["win"], 1) if r["n"] else None,
                    compound=round(r["compound"], 4), mdd=round(r["mdd"], 2),
                    p5=round(r["p5"], 4), ruin=round(r["ruin"], 2),
                    p_perm=r["p_perm"],
                    p_fwer=None if r["p_fwer"] != r["p_fwer"] else r["p_fwer"],
                    thin=bool(0 < r["n"] < 30), by_year=r["by_year"])
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dict(
            meta=dict(source=MAIN_PATH, n_trades=d["n"], months=round(d["months"], 2),
                      item_sets={k: list(v) for k, v in SETS.items()},
                      n_planned=n_planned, n_tried_raw=62, n_unique=n_all,
                      bonferroni_planned=bonf_p, bonferroni_all=bonf_a,
                      n_perm=N_PERM, n_boot=N_BOOT, lev=LEV, size_pct=SIZE,
                      occurrence={k: round(float(d["flags"][k].mean()), 4)
                                  for k in ("ob", "macro", "macro_high",
                                            "sweep", "bias", "cisd", "po3")}),
            refs=[dump(r) for r in refs],
            candidates=[dump(r) for r in rows],
        ), f, ensure_ascii=False, indent=1)
    print(f"\n  저장 → {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
