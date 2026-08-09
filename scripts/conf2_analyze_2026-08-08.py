"""#PARITY-CONF 2026-08-08 — 정합 기준선 confluence **재판정** 분석.

conf2_rerun_2026-08-08.py 산출(data/conf2/runs_*.json)을 받아

  1) 항목별 기여도 (HTF·ote·turtle·implied 포함) — 현행 BASE 실현 거래 + POOL
  2) HTF supporting 기여도 + 가중치 계단(+1/+2/+3) 단조성
  3) 후보 재실행 성적표 (빈도·건당R·복리·낙폭·파산) — 사후 필터 아님
  4) 홀드아웃 5페어 동일 적용 (파라미터 재선택 없음)
  5) 다중비교 — 순열 p + Bonferroni 문턱 병기

## 통계 규약 (2026-07~08 확립)

- 통계량은 **ΔR합** = ΣR(선택) − n(선택) × 평균R(모집단). 순열 20000회
  (R 라벨 셔플). 복리 자산이득으로 판정하지 않는다 — 경로 운에 속는다.
- 후보 변형은 각자 **재실행**이라 POOL 의 부분집합이 아니다(슬롯 점유 차이).
  순열검정은 POOL 안에서 진입시각이 일치하는 거래만으로 마스크를 만들어 돌리고,
  **일치율을 같이 보고**한다. 재실행 실측치(빈도·복리)와 검정 p 는 출처가 다르다.
- 다중비교: 계열별 검정 수를 세어 Bonferroni 문턱(0.05/k)을 병기한다.
- 30건 미만은 ※ 로 표본부족 표시.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

RNG_SEED = 20260808
N_PERM = 20_000
N_BOOT = 2_000

LEV = 7.0            # LIVE_BASE leverage
SIZE = 0.9           # LIVE_BASE size_pct
DD_PCT, DD_FACTOR = 0.25, 0.7   # 라이브 dd_throttle
RUIN = 0.20          # 시드 20% 파산 판정

IN_DIR = OUT_DIR = "data/conf2"

ITEMS = ["ob", "sweep", "bias", "cisd", "po3", "ote", "smt", "dol_counter",
         "turtle_soup", "implied_fvg", "mitigation_block", "rejection_block"]
DERIVED = ["macro_high", "macro_normal", "macro_low", "macro_any",
           "htf>=1", "htf>=2", "htf>=3", "phase_b"]


# ───────────────────────────────────────────────────── 로드 / 파생량
def flags_matrix(recs: list[dict]) -> dict[str, np.ndarray]:
    """레코드 목록 → 항목 불리언 배열 dict (파생 항목 포함)."""
    f = {k: np.array([bool(r["flags"][k]) for r in recs], bool) for k in ITEMS}
    pri = np.array([r["macro_pri"] for r in recs])
    f["macro_high"] = pri == "high"
    f["macro_normal"] = pri == "normal"
    f["macro_low"] = pri == "low"
    f["macro_any"] = pri != "none"
    hb = np.array([int(r["htf_boost"]) for r in recs])
    for k in (1, 2, 3):
        f[f"htf>={k}"] = hb >= k
    f["phase_b"] = np.array([r["phase"] == "B" for r in recs], bool)
    return f


def pack(recs: list[dict]) -> dict:
    """레코드 목록 → 분석용 배열 묶음 (진입시각 정렬)."""
    recs = sorted(recs, key=lambda x: (x["ts"], x["sym"]))
    ts = pd.to_datetime([r["ts"] for r in recs], utc=True)
    ex = pd.to_datetime([r["exit_ts"] for r in recs], utc=True)
    return {
        "n": len(recs),
        "recs": recs,
        "r": np.array([r["r"] for r in recs], float),
        "raw": np.array([r["raw"] for r in recs], float),
        "score": np.array([r["score"] for r in recs], int),
        "htf_b": np.array([r["htf_boost"] for r in recs], int),
        "htf_w": np.array([r["htf_w"] for r in recs], float),
        "win": np.array([r["raw"] > 0 for r in recs], bool),
        "sym": np.array([r["sym"] for r in recs]),
        "year": ts.year.to_numpy(),
        "ts_ns": ts.astype("int64").to_numpy(),
        "ex_ns": ex.astype("int64").to_numpy(),
        "flags": flags_matrix(recs),
    }


def conc_scale(ts_ns: np.ndarray, ex_ns: np.ndarray) -> np.ndarray:
    """동시보유 분할 배수 = 1/(동시 오픈 수). 청산시각 **실측** 사용.

    라이브는 동시 보유 시 잔고를 나눠 쓴다. 진입 시점에 열려 있는 포지션 수로
    사이즈를 나눈다(페어 무관 — 계좌 하나).
    """
    n = len(ts_ns)
    out = np.empty(n)
    for i in range(n):
        out[i] = 1.0 / (1.0 + np.sum((ts_ns <= ts_ns[i]) & (ex_ns > ts_ns[i])) - 1)
    return out


def sim(raws: np.ndarray, scale: np.ndarray) -> tuple[float, float, bool]:
    """복리 시뮬 — (최종배수, MDD%, 파산여부). 레버 7 · size 0.9 · DD 스로틀."""
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


def boot(raws: np.ndarray, scale: np.ndarray, rng) -> tuple[float, float]:
    """거래 복원추출 부트 — (5%분위 자산, 파산확률%)."""
    n = len(raws)
    if n == 0:
        return 0.0, 0.0
    fin = np.empty(N_BOOT)
    ruin = 0
    for k in range(N_BOOT):
        i = rng.integers(0, n, size=n)
        e, _, rn = sim(raws[i], scale[i])
        fin[k] = e
        ruin += int(rn)
    return float(np.percentile(fin, 5)), 100.0 * ruin / N_BOOT


# ───────────────────────────────────────────────────── 순열검정
def _subset_sums(r_pool: np.ndarray, k: int, rng, n_perm: int = N_PERM,
                 chunk: int = 2000) -> np.ndarray:
    """모집단에서 크기 k 부분집합을 무작위로 뽑은 ΣR 분포 (비복원, 벡터화).

    귀무가설 "게이트는 아무 정보가 없다 = 모집단에서 k 개를 무작위로 고른 것과
    같다" 의 정확한 null. 행별 난수 argpartition 으로 k-부분집합을 한 번에 뽑는다.
    """
    n = len(r_pool)
    out = np.empty(n_perm)
    for s in range(0, n_perm, chunk):
        c = min(chunk, n_perm - s)
        u = rng.random((c, n))
        idx = np.argpartition(u, min(k, n - 1) - 1 if k < n else n - 1, axis=1)[:, :k]
        out[s:s + c] = r_pool[idx].sum(axis=1)
    return out


def perm_p(r_pool: np.ndarray, mask=None, rng=None, strata=None,
           obs_r: np.ndarray | None = None) -> tuple[float, float]:
    """ΔR합 통계량 + 순열 p.

    귀무 = 선택이 모집단에서 같은 개수를 무작위로 고른 것과 구별되지 않는다.
    mask 를 주면 모집단 내부 선택(항목 기여도), obs_r 을 주면 **재실행 결과**의
    R 배열을 직접 관측치로 쓴다(후보 변형 — 재실행이라 모집단 부분집합이 아니다).
    strata 를 주면 층(총점) 내부에서만 뽑아 층 효과를 통제한다.

    Returns:
        (ΔR합, p). 판정 불가면 (0.0, 1.0).
    """
    ok = ~np.isnan(r_pool)          # R 미측정(entry_sl 미기록) 거래는 전부 제외
    rp = r_pool[ok]
    if len(rp) < 5:
        return 0.0, 1.0
    if strata is not None and mask is not None:
        st, mk = strata[ok], mask[ok]
        obs, null, used = 0.0, np.zeros(N_PERM), False
        for s in np.unique(st):
            g = st == s
            rg, k = rp[g], int(mk[g].sum())
            if len(rg) < 2 or k == 0 or k == len(rg):
                continue
            used = True
            obs += float(rg[mk[g]].sum() - k * rg.mean())
            null += _subset_sums(rg, k, rng) - k * rg.mean()
        if not used:
            return 0.0, 1.0
        return obs, float((null >= obs - 1e-12).mean())
    mu = float(rp.mean())
    if obs_r is not None:
        o = obs_r[~np.isnan(obs_r)]
        k = len(o)
        if k == 0 or k >= len(rp):
            return 0.0, 1.0
        obs = float(o.sum() - k * mu)
    else:
        mk = mask[ok]
        k = int(mk.sum())
        if k == 0 or k == len(rp):
            return 0.0, 1.0
        obs = float(rp[mk].sum() - k * mu)
    null = _subset_sums(rp, k, rng) - k * mu
    return obs, float((null >= obs - 1e-12).mean())


def two_sample_perm(a: np.ndarray, b: np.ndarray, rng,
                    n_perm: int = N_PERM) -> tuple[float, float]:
    """두 표본 평균R 차(a−b) 양측 순열 p (라벨 셔플, 벡터화).

    ⚠️ a·b 가 서로 다른 재실행에서 나와 일부 거래를 공유한다(독립 아님).
    공유분은 두 표본을 양(+)의 상관으로 묶어 차의 참분산을 줄이므로, 독립 가정
    순열검정은 **보수적**이다(p 가 실제보다 크게 나온다).
    """
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return float("nan"), 1.0
    obs = float(a.mean() - b.mean())
    pool = np.concatenate([a, b])
    na, n = len(a), len(pool)
    cnt = 0
    for s in range(0, n_perm, 2000):
        c = min(2000, n_perm - s)
        u = rng.random((c, n))
        idx = np.argpartition(u, na - 1, axis=1)
        ma = pool[idx[:, :na]].mean(axis=1)
        mb = pool[idx[:, na:]].mean(axis=1)
        cnt += int(np.sum(np.abs(ma - mb) >= abs(obs) - 1e-12))
    return obs, cnt / n_perm


# ───────────────────────────────────────────────────── 지표
def metrics(d: dict, months: float, mask: np.ndarray | None = None,
            rng=None, n_pairs: int = 1) -> dict:
    """거래 묶음 → 성적 지표."""
    if mask is None:
        mask = np.ones(d["n"], bool)
    n = int(mask.sum())
    if n == 0:
        return dict(n=0, freq=0.0, r_mean=float("nan"), r_sum=0.0, win=float("nan"),
                    compound=1.0, mdd=0.0, p5=float("nan"), ruin=float("nan"),
                    by_year={}, by_sym={})
    r, raw = d["r"][mask], d["raw"][mask]
    sc = conc_scale(d["ts_ns"][mask], d["ex_ns"][mask])
    e0, mdd, _ = sim(raw, sc)
    p5, ruin = boot(raw, sc, rng) if rng is not None else (float("nan"), float("nan"))
    by_year = {int(y): dict(n=int(((d["year"] == y) & mask).sum()),
                            r_sum=float(np.nansum(d["r"][(d["year"] == y) & mask])))
               for y in sorted(set(d["year"][mask].tolist()))}
    by_sym = {s: dict(n=int(((d["sym"] == s) & mask).sum()),
                      r_sum=float(np.nansum(d["r"][(d["sym"] == s) & mask])))
              for s in sorted(set(d["sym"][mask].tolist()))}
    return dict(n=n, freq=n / months / n_pairs, r_mean=float(np.nanmean(r)),
                r_sum=float(np.nansum(r)), win=100.0 * float(d["win"][mask].mean()),
                compound=float(e0), mdd=float(mdd), p5=p5, ruin=ruin,
                by_year=by_year, by_sym=by_sym)


# ───────────────────────────────────────────────────── 섹션 1·2
def section_items(base: dict, pool: dict, rng, label: str) -> dict:
    """항목별 기여도 — 무조건부 + 총점 통제(층별 셔플)."""
    print(f"\n{'=' * 100}\n[1] 항목별 기여도 — {label}", flush=True)
    out = {}
    for pname, d in (("BASE(현행 문턱5)", base), ("POOL(문턱0)", pool)):
        if d["n"] == 0:
            continue
        keys = [k for k in ITEMS + DERIVED]
        tested = [k for k in keys
                  if 0 < int(d["flags"][k].sum()) < d["n"]]
        bonf = 0.05 / max(1, len(tested))
        print(f"\n  ── {pname}  n={d['n']}  평균R={np.nanmean(d['r']):+.3f}  "
              f"ΣR={np.nansum(d['r']):+.1f}   (검정 {len(tested)}개, "
              f"Bonferroni {bonf:.4f})", flush=True)
        print(f"  {'항목':<18}{'출현':>6}{'출현율':>7}{'R(있음)':>9}{'R(없음)':>9}"
              f"{'Δ평균R':>9}{'ΔR합':>8}{'p_perm':>8}{'p_점수통제':>10}", flush=True)
        res = {}
        for k in keys:
            m = d["flags"][k]
            non = int(m.sum())
            if non == 0 or non == d["n"]:
                print(f"  {k:<18}{non:>6}{100 * non / d['n']:>6.0f}%"
                      f"{'—':>9}{'—':>9}{'—':>9}{'—':>8}{'—':>8}{'—':>10}"
                      "   (변이 없음)", flush=True)
                res[k] = dict(n_on=non, rate=100 * non / d["n"], degenerate=True)
                continue
            r_on, r_off = d["r"][m], d["r"][~m]
            dr, p = perm_p(d["r"], m, rng)
            _, ps = perm_p(d["r"], m, rng, strata=d["score"])
            mark = "※" if non < 30 or (d["n"] - non) < 30 else " "
            print(f"  {k:<18}{non:>6}{100 * non / d['n']:>6.0f}%"
                  f"{np.nanmean(r_on):>+9.3f}{np.nanmean(r_off):>+9.3f}"
                  f"{np.nanmean(r_on) - np.nanmean(r_off):>+9.3f}{dr:>+8.1f}"
                  f"{p:>8.4f}{ps:>10.4f} {mark}", flush=True)
            res[k] = dict(n_on=non, rate=100 * non / d["n"],
                          r_on=float(np.nanmean(r_on)), r_off=float(np.nanmean(r_off)),
                          d_mean=float(np.nanmean(r_on) - np.nanmean(r_off)),
                          dr_sum=dr, p=p, p_strat=ps, small=bool(mark == "※"))
        out[pname] = dict(rows=res, bonferroni=bonf, n_tests=len(tested))
    return out


def section_htf(base: dict, pool: dict, rng) -> dict:
    """HTF supporting 전용 — 계단 단조성 + 문턱 의존성."""
    print(f"\n{'=' * 100}\n[2] HTF supporting 기여도 — 라이브 진입의 93% 에 붙는 항목",
          flush=True)
    out = {}
    for pname, d in (("BASE(문턱5)", base), ("POOL(문턱0)", pool)):
        if d["n"] == 0:
            continue
        print(f"\n  ── {pname}  n={d['n']}", flush=True)
        print(f"  {'htf boost':<12}{'건수':>7}{'비율':>7}{'평균R':>9}{'ΣR':>9}"
              f"{'승률':>7}", flush=True)
        rows = {}
        for b in (0, 1, 2, 3):
            m = d["htf_b"] == b
            if m.sum() == 0:
                print(f"  +{b:<11}{0:>7}{0.0:>6.1f}%{'—':>9}{'—':>9}{'—':>7}",
                      flush=True)
                rows[b] = dict(n=0)
                continue
            print(f"  +{b:<11}{int(m.sum()):>7}{100 * m.mean():>6.1f}%"
                  f"{np.nanmean(d['r'][m]):>+9.3f}{np.nansum(d['r'][m]):>+9.1f}"
                  f"{100 * d['win'][m].mean():>6.0f}%", flush=True)
            rows[b] = dict(n=int(m.sum()), r_mean=float(np.nanmean(d["r"][m])),
                           r_sum=float(np.nansum(d["r"][m])),
                           win=float(100 * d["win"][m].mean()))
        # 단조성 — 가중치(htf_w) 와 R 의 순위상관 + 순열 p
        ok = ~np.isnan(d["r"])
        if ok.sum() > 5 and len(np.unique(d["htf_w"][ok])) > 1:
            rho = float(pd.Series(d["htf_w"][ok]).corr(
                pd.Series(d["r"][ok]), method="spearman"))
            cnt = 0
            rv = d["r"][ok]
            wv = d["htf_w"][ok]
            for _ in range(2000):
                pr = float(pd.Series(wv).corr(
                    pd.Series(rng.permutation(rv)), method="spearman"))
                cnt += int(abs(pr) >= abs(rho) - 1e-12)
            print(f"    가중치(htf_w)–R 순위상관 rho={rho:+.3f} "
                  f"(양측 순열 p={cnt / 2000:.4f}, 2000회)", flush=True)
            out.setdefault("mono", {})[pname] = dict(rho=rho, p=cnt / 2000)
        # 가중치 구간
        print(f"    htf_w 분포: " + " ".join(
            f"{v}:{int((d['htf_w'] == v).sum())}"
            for v in sorted(set(d["htf_w"].tolist()))[:12]), flush=True)
        out[pname] = rows
    # 점수 구성 — "실효 문턱" 이 실제로 무엇인지 보여주는 표
    print("\n  ── 점수 구성 (BASE) — score = base + po3/cisd/ote… + htf(0~3) − dol(2)",
          flush=True)
    if base["n"]:
        sc = pd.Series(base["score"]).value_counts().sort_index()
        nh = pd.Series(base["score"] - base["htf_b"]).value_counts().sort_index()
        bs = pd.Series([r["base"] for r in base["recs"]]).value_counts().sort_index()
        print(f"    score      : " + " ".join(f"{k}:{v}" for k, v in sc.items()),
              flush=True)
        print(f"    score−htf  : " + " ".join(f"{k}:{v}" for k, v in nh.items()),
              flush=True)
        print(f"    base_score : " + " ".join(f"{k}:{v}" for k, v in bs.items()),
              flush=True)
        out["score_dist"] = {"score": {int(k): int(v) for k, v in sc.items()},
                             "score_minus_htf": {int(k): int(v) for k, v in nh.items()},
                             "base": {int(k): int(v) for k, v in bs.items()}}
    # POOL 에서 "htf 없으면 문턱5 미달" 인 거래 (= htf 덕에 들어간 진입)
    d = pool
    if d["n"]:
        dep = (d["score"] >= 5) & ((d["score"] - d["htf_b"]) < 5)
        own = (d["score"] - d["htf_b"]) >= 5
        print("\n  ── POOL: HTF 가 문턱을 넘겨준 진입 vs 자력 통과 진입", flush=True)
        for nm, m in (("HTF 덕에 통과", dep), ("자력 통과", own)):
            if m.sum():
                print(f"    {nm:<14}{int(m.sum()):>6}건  평균R "
                      f"{np.nanmean(d['r'][m]):>+.3f}  ΣR {np.nansum(d['r'][m]):>+.1f}  "
                      f"승률 {100 * d['win'][m].mean():>3.0f}%", flush=True)
        if dep.sum() and own.sum():
            diff, p = two_sample_perm(d["r"][dep], d["r"][own], rng)
            print(f"    차이 {diff:+.3f}R  양측 순열 p={p:.4f}", flush=True)
            out["htf_dependent_vs_own"] = dict(
                n_dep=int(dep.sum()), n_own=int(own.sum()),
                r_dep=float(np.nanmean(d["r"][dep])), r_own=float(np.nanmean(d["r"][own])),
                diff=diff, p=p)
    return out


# ───────────────────────────────────────────────────── 섹션 3·4
def section_variants(runs: dict, months: float, n_pairs: int, rng,
                     label: str) -> dict:
    """후보 재실행 성적표 + POOL 대조 순열검정."""
    print(f"\n{'=' * 100}\n[3] 후보 재실행 성적 — {label} "
          f"({n_pairs}페어 · {months:.1f}개월)", flush=True)
    packs = {vn: pack(tr) for vn, tr in runs["trades"].items()}
    pool = packs["POOL"]
    pool_ts = {(int(a), s) for a, s in zip(pool["ts_ns"], pool["sym"])}
    tested = [vn for vn in packs if vn not in runs["not_tested"]]
    bonf = 0.05 / len(tested)
    print(f"  검정 대상 {len(tested)}개 → Bonferroni 문턱 {bonf:.4f} "
          f"(순열 {N_PERM}회, 통계량 ΔR합)", flush=True)
    print(f"\n  {'변형':<10}{'n':>5}{'월빈도':>7}{'건당R':>8}{'승률':>6}{'ΣR':>8}"
          f"{'복리':>9}{'MDD%':>7}{'파산%':>7}{'부트5%':>8}{'ΔR합':>8}{'p_perm':>8}"
          f"{'일치':>6}", flush=True)
    out = {}
    pool_keys = {(int(x), y) for x, y in zip(pool["ts_ns"], pool["sym"])}
    for vn, d in packs.items():
        m = metrics(d, months, None, rng, n_pairs)
        # 재실행 결과이므로 POOL 의 부분집합이 아니다 → 관측치는 변형 자신의 R,
        # 귀무는 "POOL 에서 같은 개수를 무작위 선택". 겹침률은 참고로 같이 낸다.
        match = (sum(1 for x, y in zip(d["ts_ns"], d["sym"])
                     if (int(x), y) in pool_keys) / m["n"]) if m["n"] else 0.0
        dr, p = perm_p(pool["r"], rng=rng, obs_r=d["r"])
        print(f"  {vn:<10}{m['n']:>5}{m['freq']:>7.2f}{m['r_mean']:>+8.3f}"
              f"{m['win']:>5.0f}%{m['r_sum']:>+8.1f}{m['compound']:>9.3f}"
              f"{m['mdd']:>7.1f}{m['ruin']:>7.1f}{m['p5']:>8.3f}{dr:>+8.1f}"
              f"{p:>8.4f}{100 * match:>5.0f}%"
              + ("  ※" if m["n"] < 30 else ""), flush=True)
        out[vn] = dict(m, dr_pool=dr, p_perm=p, pool_match=match,
                       sig_bonf=bool(p <= bonf))
    out["_meta"] = dict(bonferroni=bonf, n_tests=len(tested), months=months,
                        n_pairs=n_pairs, pool_n=pool["n"])
    # BASE 대비 직접 비교 (빈도 정합 축 포함)
    print("\n  [BASE 대비 건당R 차 — 양측 순열, 거래 공유로 보수적]", flush=True)
    b = packs["BASE"]
    for vn in tested:
        d = packs[vn]
        if d["n"] < 5:
            continue
        diff, p = two_sample_perm(d["r"][~np.isnan(d["r"])],
                                  b["r"][~np.isnan(b["r"])], rng)
        print(f"    {vn:<10} Δ건당R {diff:>+7.3f}  p={p:.4f}"
              + ("  <= Bonferroni" if p <= bonf else ""), flush=True)
        out[vn]["diff_vs_base"] = diff
        out[vn]["p_vs_base"] = p
    return out


def spearman(a, b) -> float:
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def main() -> int:
    rng = np.random.default_rng(RNG_SEED)
    sfx = ""
    for a in sys.argv[1:]:
        if a.startswith("--bars="):
            sfx = f"_{a.split('=')[1]}"
    mp = os.path.join(IN_DIR, f"runs_main{sfx}.json")
    if not os.path.exists(mp):
        mp = os.path.join(IN_DIR, f"runs_dev{sfx}.json")
    with open(mp, encoding="utf-8") as f:
        main_runs = json.load(f)
    syms = [s for s in main_runs["meta"]]
    months = float(np.mean([main_runs["meta"][s]["_pair"]["months"] for s in syms]))
    base, pool = pack(main_runs["trades"]["BASE"]), pack(main_runs["trades"]["POOL"])

    out = {"meta": {"date": "2026-08-08", "main_syms": syms, "months": months,
                    "n_perm": N_PERM, "lev": LEV, "size": SIZE,
                    "note": "정합(2026-08-08 이식) 기준선. data/conf/*.json 은 "
                            "정합 이전이라 결론 근거 아님."}}
    out["items"] = section_items(base, pool, rng, "본 표본 " + "+".join(
        s.replace("USDT", "") for s in syms))
    out["htf"] = section_htf(base, pool, rng)
    out["main"] = section_variants(main_runs, months, len(syms), rng, "본 표본")

    hp = os.path.join(IN_DIR, f"runs_holdout{sfx}.json")
    if os.path.exists(hp):
        with open(hp, encoding="utf-8") as f:
            hold_runs = json.load(f)
        hsyms = [s for s in hold_runs["meta"]]
        hmonths = float(np.mean([hold_runs["meta"][s]["_pair"]["months"] for s in hsyms]))
        out["holdout"] = section_variants(hold_runs, hmonths, len(hsyms), rng,
                                          "홀드아웃 " + "·".join(
                                              s.replace("USDT", "") for s in hsyms))
        keys = [k for k in main_runs["trades"] if k not in ("_meta",)]
        sd = spearman([out["main"][k]["dr_pool"] for k in keys],
                      [out["holdout"][k]["dr_pool"] for k in keys])
        sr = spearman([out["main"][k]["r_mean"] for k in keys],
                      [out["holdout"][k]["r_mean"] for k in keys])
        print(f"\n[4] 주표본↔홀드아웃 순위일치 Spearman — ΔR합 {sd:+.3f} · "
              f"건당R {sr:+.3f}", flush=True)
        out["rank_agreement"] = {"spearman_dr": sd, "spearman_r_mean": sr}
        # 홀드아웃 항목 기여도(참고)
        out["items_holdout"] = section_items(
            pack(hold_runs["trades"]["BASE"]), pack(hold_runs["trades"]["POOL"]),
            rng, "홀드아웃(참고)")

    path = os.path.join(OUT_DIR, f"verdict{sfx}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n저장 → {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
