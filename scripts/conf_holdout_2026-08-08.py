"""#AUTONOMOUS 2026-08-08 — confluence 후보 **홀드아웃 검증**.

앞 3단계(단독기여도 / 조합AND / 가중치)가 BTC+ETH 258건에서 고른 후보를
**손대지 않고 그대로** SOL·XRP·DOGE·LINK·HYPE 597건에 적용한다.
파라미터 재선택 금지 — 재선택하는 순간 홀드아웃이 아니다.

## 판정 기준 (브리핑 6번)
홀드아웃 5페어는 2026-08-06 에 고정 페어에서 제외된 알트다(알트 단독 5년 0.86배).
그러므로 **절대 성적이 나쁜 것은 정상**이다. 판정은
  (a) 후보 간 **상대 순위**가 주 표본과 일치하는가 (Spearman)
  (b) **ΔR 부호(방향)**가 일치하는가
  (c) 주 표본에서 좋았는데 홀드아웃에서 ΔR 이 음수로 뒤집히는가 → 과최적 증거

## 통계
관측량 = ΔR합 = ΣR(선택) − n_선택 × 평균R(해당 풀 전체).
귀무 = 항목 플래그와 성과 무관. 거래별 R 셔플 20000회. 모든 후보가 같은 draw 를
공유하므로 max-T FWER 보정 가능.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

MAIN_PATH = "data/conf/trades_main.json"
HOLD_PATH = "data/conf/trades_holdout.json"
OUT_PATH = "data/conf/holdout.json"

N_PERM = 20_000
N_BOOT = 2_000
RNG = np.random.default_rng(20260808)

LEV = 7.0
SIZE = 0.9
DD_PCT, DD_FACTOR = 0.25, 0.7
RUIN = 0.20
HOLD_CAP_NS = 24 * 3600 * 1_000_000_000

# ── 후보 (앞 단계 산출물 그대로. 각 각도 최대 3개) ─────────────────────
# 가중치 스킴: (ob, macro_high, macro_normal, macro_low, sweep, bias, cisd, po3)
W_CUR = dict(ob=1, mh=2, mn=1, ml=0, sweep=1, bias=1, cisd=1, po3=1)
W_NOSWEEP = dict(W_CUR, sweep=0)
W_MHIONLY = dict(W_CUR, mn=0)
W_MHI3 = dict(W_CUR, mh=3)
W_DATA = dict(ob=0, mh=2, mn=-1, ml=-1, sweep=0, bias=2, cisd=1, po3=0)

CANDS = [
    # (키, 표시명, 출처각도, 종류, 사양, 주표본 사전보고 p)
    ("CUR5", "현행 score>=5", "기준선", "w", (W_CUR, 5), 0.00055),
    ("THR4", "문턱4 (기각 확인 대조)", "기준선", "w", (W_CUR, 4), 0.890),
    ("NOSWEEP5", "sweep제외 문턱5", "단독기여", "w", (W_NOSWEEP, 5), 0.0002),
    ("NOSWEEP4", "sweep제외 문턱4", "단독기여/가중치", "w", (W_NOSWEEP, 4), 0.00115),
    ("MHIONLY5", "macro=high만 문턱5", "단독기여", "w", (W_MHIONLY, 5), 0.0004),
    ("MHI3_6", "MHI3/thr6", "가중치", "w", (W_MHI3, 6), 0.204),
    ("DATA4", "DATA/thr4", "가중치", "w", (W_DATA, 4), 0.278),
    ("MH_BIAS", "macro_high AND bias", "조합", "and", ("macro_high", "bias"), 0.00035),
    ("MH", "macro_high 단독", "조합", "and", ("macro_high",), 0.00695),
]


def load(path: str) -> dict:
    """거래 JSON → 넘파이 묶음."""
    with open(path, encoding="utf-8") as f:
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
        "months": float((ts_ns[-1] - ts_ns[0]) / 1e9 / 86400 / 30.44),
    }
    f_ = {}
    for k in ("ob", "macro", "sweep", "bias", "cisd", "po3"):
        f_[k] = np.array([bool(x["flags"][k]) for x in recs], bool)
    pri = np.array([str(x["flags"].get("macro_pri")) for x in recs])
    f_["macro_high"] = pri == "high"
    f_["macro_normal"] = pri == "normal"
    f_["macro_low"] = pri == "low"
    d["flags"] = f_
    d["scale"] = conc_scale(ts_ns, d["sym"])
    return d


def conc_scale(ts_ns: np.ndarray, sym: np.ndarray) -> np.ndarray:
    """동시보유 분할 배수 1/동시보유수 (L2 근사, combo 스크립트와 동일)."""
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


def sim(raws: np.ndarray, scale: np.ndarray) -> tuple[float, float, bool]:
    """복리 시뮬 — (최종배수, MDD%, 파산여부)."""
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


def score_of(d: dict, w: dict) -> np.ndarray:
    """가중치 스킴으로 점수 재계산."""
    f = d["flags"]
    return (w["ob"] * f["ob"] + w["mh"] * f["macro_high"]
            + w["mn"] * f["macro_normal"] + w["ml"] * f["macro_low"]
            + w["sweep"] * f["sweep"] + w["bias"] * f["bias"]
            + w["cisd"] * f["cisd"] + w["po3"] * f["po3"]).astype(int)


def mask_of(d: dict, kind: str, spec) -> np.ndarray:
    """후보 사양 → 선택 마스크."""
    if kind == "w":
        w, thr = spec
        return score_of(d, w) >= thr
    m = np.ones(d["n"], bool)
    for k in spec:
        m &= d["flags"][k]
    return m


def evaluate(d: dict, m: np.ndarray, perm_r: np.ndarray) -> dict:
    """마스크 하나의 전체 지표 (null 분포 포함 — max-T 보정용)."""
    n = int(m.sum())
    mu = float(d["r"].mean())
    if n == 0:
        return dict(n=0, monthly=0.0, r_mean=float("nan"), r_sum=0.0,
                    win=float("nan"), dr=0.0, compound=1.0, mdd=0.0,
                    p5=0.0, ruin=0.0, p_perm=1.0, by_year={},
                    null=np.zeros(N_PERM))
    r = d["r"][m]
    r_sum = float(r.sum())
    dr = r_sum - n * mu
    e0, mdd, _ = sim(d["raw"][m], d["scale"][m])
    p5, ruin = boot(d["raw"][m], d["scale"][m])
    null = perm_r[:, m].sum(axis=1) - n * mu
    p = float((null >= dr - 1e-12).mean())
    by_year = {int(y): dict(n=int((m & (d["year"] == y)).sum()),
                            r_sum=float(d["r"][m & (d["year"] == y)].sum()))
               for y in sorted(set(d["year"][m].tolist()))}
    by_sym = {s: dict(n=int((m & (d["sym"] == s)).sum()),
                      r_sum=float(d["r"][m & (d["sym"] == s)].sum()))
              for s in sorted(set(d["sym"][m].tolist()))}
    return dict(n=n, monthly=n / d["months"], r_mean=float(r.mean()),
                r_sum=r_sum, win=float(d["win"][m].mean() * 100), dr=float(dr),
                compound=float(e0), mdd=float(mdd), p5=float(p5),
                ruin=float(ruin), p_perm=p, by_year=by_year, by_sym=by_sym,
                null=null)


def spearman(a: list[float], b: list[float]) -> float:
    """순위상관 (동점 평균순위)."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


def run(d: dict, label: str) -> dict:
    """한 풀(주표본 또는 홀드아웃)에 전 후보 적용."""
    perm_r = np.empty((N_PERM, d["n"]))
    rng = np.random.default_rng(20260808)
    for i in range(N_PERM):
        perm_r[i] = rng.permutation(d["r"])
    out = {}
    nulls = {}
    for key, name, angle, kind, spec, _p in CANDS:
        m = mask_of(d, kind, spec)
        res = evaluate(d, m, perm_r)
        nulls[key] = res.pop("null")
        res.update(name=name, angle=angle)
        out[key] = res
    # max-T FWER (기준선 2개 제외한 실검정 7개 대상)
    fam = [k for k, *_ in [(c[0],) for c in CANDS] if k not in ("CUR5", "THR4")]
    sd = {k: (nulls[k].std() or 1e-12) for k in fam}
    z = np.stack([nulls[k] / sd[k] for k in fam])
    maxz = z.max(axis=0)
    for k in fam:
        obs_z = out[k]["dr"] / sd[k]
        out[k]["p_fwer"] = float((maxz >= obs_z - 1e-12).mean())
    print(f"\n=== {label} — {d['n']}건 / {d['months']:.1f}개월 / "
          f"평균R {d['r'].mean():+.4f} / ΣR {d['r'].sum():+.1f} ===")
    return out


def main() -> int:
    main_d = load(MAIN_PATH)
    hold_d = load(HOLD_PATH)
    res_m = run(main_d, "주 표본 BTC+ETH (재계산 · 앞 단계 재현)")
    res_h = run(hold_d, "홀드아웃 SOL·XRP·DOGE·LINK·HYPE")

    hdr = ("{:<22} {:>4} {:>6} {:>8} {:>8} {:>6} {:>8} {:>7} {:>6} {:>7}")
    row = ("{:<22} {:>4d} {:>6.2f} {:>+8.3f} {:>+8.1f} {:>6.1f} {:>8.4f} "
           "{:>7.2f} {:>6.1f} {:>7.1f}")
    for label, res in (("주 표본", res_m), ("홀드아웃", res_h)):
        print(f"\n--- {label} ---")
        print(hdr.format("후보", "n", "월빈도", "건당R", "ΔR합", "승률",
                         "p_perm", "복리", "MDD%", "파산%"))
        for key, name, *_ in CANDS:
            r = res[key]
            if r["n"] == 0:
                print(f"{name:<22} 0건")
                continue
            print(row.format(name, r["n"], r["monthly"], r["r_mean"], r["dr"],
                             r["win"], r["p_perm"], r["compound"], r["mdd"],
                             r["ruin"]))

    keys = [c[0] for c in CANDS]
    sp_dr = spearman([res_m[k]["dr"] for k in keys],
                     [res_h[k]["dr"] for k in keys])
    sp_rm = spearman([res_m[k]["r_mean"] for k in keys],
                     [res_h[k]["r_mean"] for k in keys])
    print(f"\n순위일치 Spearman — ΔR합 {sp_dr:+.3f} · 건당R {sp_rm:+.3f}")

    out = {
        "meta": {
            "date": "2026-08-08",
            "main": dict(n=main_d["n"], months=main_d["months"],
                         r_mean=float(main_d["r"].mean()),
                         syms=sorted(set(main_d["sym"].tolist()))),
            "holdout": dict(n=hold_d["n"], months=hold_d["months"],
                            r_mean=float(hold_d["r"].mean()),
                            syms=sorted(set(hold_d["sym"].tolist()))),
            "n_perm": N_PERM, "n_cand": len(CANDS),
            "bonferroni": 0.05 / (len(CANDS) - 2),
            "statistic": "dR_sum (ΣR_sel − n_sel × mean_R_pool)",
        },
        "candidates": [dict(key=c[0], name=c[1], angle=c[2], kind=c[3],
                            spec=(list(c[4][0].items()) + [["thr", c[4][1]]]
                                  if c[3] == "w" else list(c[4])),
                            p_main_prior=c[5]) for c in CANDS],
        "main": res_m, "holdout": res_h,
        "rank_agreement": {"spearman_dr": sp_dr, "spearman_r_mean": sp_rm},
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n저장 {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
