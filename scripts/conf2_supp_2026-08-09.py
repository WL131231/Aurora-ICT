"""#PARITY-CONF 보강 2026-08-09 — 재판정에 빠져 있던 4가지를 채운다.

conf2_analyze 가 낸 표만으로는 답이 안 나오는 질문들이 있다. 여기서 채운다.

  [A] 정합 대조 먼저 — BASE 의 승률·RR·빈도를 라이브 실측과 나란히.
      (규칙 3: 어긋나면 아래 숫자는 신뢰하지 않는다)
  [B] HTF supporting 의 **인과 기여도** — 계단이 사실상 상수라면 "가점"이 아니라
      "문턱 오프셋"이다. 빈도정합 대조(BASE 문턱5+htf ↔ NOHTF2 문턱2−htf)로
      같은 크기 표본에서 성적이 같은지 본다. 같으면 기여도 0.
  [C] 점수(개수) 효과 — 2026-08-08 결론 "개수는 무관"의 정합 기준선 재검.
      score=5 vs >=6 등 층별 평균R + 순위상관.
  [D] 후보(macro_high AND bias) 의 **표본 밖 일반화** — 본표본/홀드아웃 분리 판정,
      연도별 부호, 상위 거래 제거 민감도(소수 대박에 얹힌 결론인지).

⚠️ 이 스크립트는 새 변형을 돌리지 않는다. conf2_rerun 산출(runs_*.json)만 읽는다.
   따라서 다중비교 분모는 conf2_rerun 사전등록(13개) 그대로이고, 여기서 새로
   더하는 검정은 [B]·[C]·[D] 의 8개다 — 아래 MULTIPLICITY 에 명시.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_spec = importlib.util.spec_from_file_location(
    "conf2_analyze",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "conf2_analyze_2026-08-08.py"),
)
A = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(A)

IN_DIR = OUT_DIR = "data/conf2"

# 라이브 실측 기준선 — 결론 신뢰도의 전제.
LIVE_WR, LIVE_RR = 46.0, 0.94        # 2026-05-28~07-30 Origo 2.2 청산 100건
LIVE_FREQ = 2.63                     # 건/페어/월 (fst 스냅샷 73건, 12유저)
LIVE_FREQ_RANGE = (2.26, 4.37)

# 이 스크립트가 새로 추가하는 검정 수 (Bonferroni 분모).
MULTIPLICITY = 8


def wr_rr(r: np.ndarray, raw: np.ndarray) -> tuple[float, float]:
    """승률(%) 과 RR(평균이익/|평균손실|) — 라이브 지표와 같은 정의."""
    w, l = raw[raw > 0], raw[raw < 0]
    rr = (w.mean() / abs(l.mean())) if len(w) and len(l) else float("nan")
    return 100.0 * len(w) / len(raw) if len(raw) else float("nan"), rr


def load(group: str) -> dict:
    with open(os.path.join(IN_DIR, f"runs_{group}.json"), encoding="utf-8") as f:
        return json.load(f)


def sec_a(runs: dict, label: str) -> dict:
    """[A] 정합 대조 — BASE 가 라이브와 같은 봇인가."""
    print(f"\n{'=' * 96}\n[A] 정합 대조 먼저 — {label}", flush=True)
    d = A.pack(runs["trades"]["BASE"])
    syms = [s for s in runs["meta"]]
    months = float(np.mean([runs["meta"][s]["_pair"]["months"] for s in syms]))
    wr, rr = wr_rr(d["r"], d["raw"])
    freq = d["n"] / months / len(syms)
    print(f"  {'':<14}{'백테(5년)':>12}{'라이브 실측':>14}{'비':>8}", flush=True)
    print(f"  {'진입 빈도':<14}{freq:>11.2f}{LIVE_FREQ:>13.2f}{freq / LIVE_FREQ:>8.2f}"
          f"   (라이브 유저별 {LIVE_FREQ_RANGE[0]:.2f}~{LIVE_FREQ_RANGE[1]:.2f})",
          flush=True)
    print(f"  {'승률':<14}{wr:>10.0f}%{LIVE_WR:>12.0f}%{wr / LIVE_WR:>8.2f}", flush=True)
    print(f"  {'RR':<14}{rr:>11.2f}{LIVE_RR:>13.2f}{rr / LIVE_RR:>8.2f}", flush=True)
    print(f"  건당R {np.nanmean(d['r']):+.3f} · ΣR {np.nansum(d['r']):+.1f} · n={d['n']}",
          flush=True)
    # 연도별 — 라이브 표본(2026 여름)과 가장 가까운 구간이 특히 중요.
    print("\n  [연도별 BASE]", flush=True)
    for y in sorted(set(d["year"].tolist())):
        m = d["year"] == y
        w2, r2 = wr_rr(d["r"][m], d["raw"][m])
        print(f"    {y}  n={int(m.sum()):>4}  승률 {w2:>3.0f}%  RR {r2:>5.2f}  "
              f"건당R {np.nanmean(d['r'][m]):>+6.3f}  ΣR {np.nansum(d['r'][m]):>+7.1f}",
              flush=True)
    return {"n": d["n"], "wr": wr, "rr": rr, "freq": freq, "months": months,
            "n_pairs": len(syms), "r_mean": float(np.nanmean(d["r"]))}


def sec_b(runs: dict, rng, label: str) -> dict:
    """[B] HTF supporting — 계단인가 상수인가, 그리고 빈도정합에서 기여가 남는가."""
    print(f"\n{'=' * 96}\n[B] HTF supporting 기여도 — {label}", flush=True)
    pool = A.pack(runs["trades"]["POOL"])
    base = A.pack(runs["trades"]["BASE"])
    print("  [B1] 계단(+1/+2/+3) 이 실제로 계단인가 — POOL 전수", flush=True)
    hb = pd.Series(pool["htf_b"]).value_counts().sort_index()
    for k, v in hb.items():
        m = pool["htf_b"] == k
        print(f"    boost +{k}: {v:>5}건 ({100 * v / pool['n']:>5.1f}%)  "
              f"평균R {np.nanmean(pool['r'][m]):>+6.3f}", flush=True)
    hw = pd.Series(pool["htf_w"])
    print(f"    가중치합 htf_w 분위: min={hw.min():.0f} "
          f"q10={hw.quantile(.1):.0f} q50={hw.quantile(.5):.0f} "
          f"q90={hw.quantile(.9):.0f} max={hw.max():.0f}", flush=True)
    print(f"    → 계단 경계는 2/10/20. q10={hw.quantile(.1):.0f} 이 이미 20 이상이면 "
          "'계단'이 아니라 **상수 +3**이다 (TF 가중치 15m2·1h4·2h6·4h10·1d20·1w40 을 "
          "지지 FVG **전부 합산**하기 때문).", flush=True)
    degen = float((pool["htf_b"] == 3).mean())

    print("\n  [B2] 빈도정합 대조 — 같은 거래 수에서 HTF 가 정보를 주는가", flush=True)
    print("    HTF 가 상수 +3 이면 '문턱5 + htf' 와 '문턱2 − htf' 는 같은 게이트다.", flush=True)
    rows = {}
    for vn in ("BASE", "NOHTF2", "NOHTF3", "NOHTF4", "NOHTF5"):
        if vn not in runs["trades"]:
            continue
        d = A.pack(runs["trades"][vn])
        if d["n"] == 0:
            continue
        w, r = wr_rr(d["r"], d["raw"])
        rows[vn] = dict(n=d["n"], r_mean=float(np.nanmean(d["r"])),
                        r_sum=float(np.nansum(d["r"])), wr=w, rr=r)
        print(f"    {vn:<8} n={d['n']:>5}  건당R {np.nanmean(d['r']):>+6.3f}  "
              f"ΣR {np.nansum(d['r']):>+7.1f}  승률 {w:>3.0f}%", flush=True)
    out = {"degenerate_plus3_rate": degen, "freq_matched": rows}
    if "NOHTF2" in rows:
        a, b = A.pack(runs["trades"]["NOHTF2"]), base
        diff, p = A.two_sample_perm(a["r"][~np.isnan(a["r"])],
                                    b["r"][~np.isnan(b["r"])], rng)
        print(f"    빈도정합 쌍 (BASE n={b['n']} ↔ NOHTF2 n={a['n']}): "
              f"Δ건당R {diff:+.3f}  양측 순열 p={p:.4f}", flush=True)
        print(f"    → p 가 크면 'HTF 는 선별 정보 0, 문턱 오프셋일 뿐' 이다.", flush=True)
        out["freq_matched_test"] = {"diff": diff, "p": p}

    print("\n  [B3] HTF 를 끄고 문턱을 올리면(선별 강화) 성적이 오르는가 — 계단 응답",
          flush=True)
    print("    이건 HTF 기여가 아니라 **문턱 자체의 기여**를 재는 것이다.", flush=True)
    return out


def sec_c(runs: dict, rng, label: str) -> dict:
    """[C] 점수(개수) 효과 — 2026-08-08 '개수는 무관' 결론 재검."""
    print(f"\n{'=' * 96}\n[C] 점수(항목 개수) 효과 — {label}", flush=True)
    out = {}
    for pname, key in (("POOL(문턱0)", "POOL"), ("BASE(문턱5)", "BASE")):
        d = A.pack(runs["trades"][key])
        if d["n"] == 0:
            continue
        print(f"\n  ── {pname}  n={d['n']}", flush=True)
        rows = {}
        for s in sorted(set(d["score"].tolist())):
            m = d["score"] == s
            if m.sum() == 0:
                continue
            rows[int(s)] = dict(n=int(m.sum()), r_mean=float(np.nanmean(d["r"][m])),
                                r_sum=float(np.nansum(d["r"][m])))
            print(f"    score={s:<3} n={int(m.sum()):>5}  건당R "
                  f"{np.nanmean(d['r'][m]):>+6.3f}  ΣR {np.nansum(d['r'][m]):>+7.1f}"
                  + ("  ※표본부족" if m.sum() < 30 else ""), flush=True)
        ok = ~np.isnan(d["r"])
        rho = float(pd.Series(d["score"][ok]).corr(pd.Series(d["r"][ok]),
                                                   method="spearman"))
        cnt, rv = 0, d["r"][ok]
        for _ in range(2000):
            pr = float(pd.Series(d["score"][ok]).corr(
                pd.Series(rng.permutation(rv)), method="spearman"))
            cnt += int(abs(pr) >= abs(rho) - 1e-12)
        print(f"    점수–R 순위상관 rho={rho:+.3f} (양측 순열 p={cnt / 2000:.4f})",
              flush=True)
        out[pname] = {"rows": rows, "rho": rho, "p": cnt / 2000}
    return out


def sec_d(main: dict, hold: dict | None, rng) -> dict:
    """[D] 후보 macro_high AND bias — 표본 밖 일반화 + 대박 의존도."""
    print(f"\n{'=' * 96}\n[D] 후보 재판정 — macro_high AND bias", flush=True)
    out = {}
    for gname, runs in (("본표본", main), ("홀드아웃", hold)):
        if runs is None:
            continue
        syms = [s for s in runs["meta"]]
        months = float(np.mean([runs["meta"][s]["_pair"]["months"] for s in syms]))
        pool = A.pack(runs["trades"]["POOL"])
        print(f"\n  ── {gname} ({len(syms)}페어 · {months:.1f}개월): "
              + " ".join(s.replace("USDT", "") for s in syms), flush=True)
        g = {}
        for vn in ("BASE", "MH_BIAS", "MH_BIAS5", "MH"):
            d = A.pack(runs["trades"][vn])
            if d["n"] == 0:
                continue
            m = A.metrics(d, months, None, rng, len(syms))
            w, rr = wr_rr(d["r"], d["raw"])
            dr, p = A.perm_p(pool["r"], rng=rng, obs_r=d["r"])
            # 상위 거래 제거 민감도 — 대박 3건을 빼도 살아남는가.
            r = np.sort(d["r"][~np.isnan(d["r"])])
            drop3 = float(r[:-3].mean()) if len(r) > 3 else float("nan")
            # 연도 부호
            ys = {int(y): float(np.nansum(d["r"][d["year"] == y]))
                  for y in sorted(set(d["year"].tolist()))}
            ypos = sum(1 for v in ys.values() if v > 0)
            # 페어 부호
            ss = {s: float(np.nansum(d["r"][d["sym"] == s]))
                  for s in sorted(set(d["sym"].tolist()))}
            spos = sum(1 for v in ss.values() if v > 0)
            print(f"    {vn:<9} n={m['n']:>4} 빈도{m['freq']:>5.2f}/월 "
                  f"건당R {m['r_mean']:>+6.3f} 승률{w:>3.0f}% RR{rr:>5.2f} "
                  f"복리 {m['compound']:>7.3f}x MDD {m['mdd']:>4.0f}% "
                  f"파산 {m['ruin']:>5.1f}% 부트5% {m['p5']:>6.3f} "
                  f"ΔR합 {dr:>+6.1f} p={p:.4f} 연도{ypos}/{len(ys)} 페어{spos}/{len(ss)}"
                  + ("  ※n<30" if m["n"] < 30 else ""), flush=True)
            print(f"              상위3건 제거 건당R {drop3:>+6.3f}"
                  f"   연도별ΣR " + " ".join(f"{k}:{v:+.1f}" for k, v in ys.items()),
                  flush=True)
            g[vn] = dict(m, wr=w, rr=rr, dr_pool=dr, p_perm=p, drop_top3_r=drop3,
                         by_year_sum=ys, y_pos=ypos, y_tot=len(ys),
                         by_sym_sum=ss, s_pos=spos, s_tot=len(ss))
        out[gname] = g
    return out


def sec_e(main: dict, hold: dict, rng) -> dict:
    """[E] 7페어 계좌 통합 — 배포 판단에 실제로 필요한 숫자.

    페어별 백테는 독립이지만 **계좌는 하나**다. 동시보유 분할·DD 스로틀·파산은
    전 페어 거래를 시간순으로 합쳐야 제대로 계산된다. 본표본+홀드아웃을 합쳐
    현행(BASE)과 후보를 같은 계좌에서 돌린다.

    ⚠️ 이건 **표본 밖 검정이 아니다** — 본표본이 섞여 있다. 판정은 [D] 홀드아웃으로
    하고, 여기 숫자는 "만약 성립한다면 계좌에 어떤 모양인가"에만 쓴다.
    """
    print(f"\n{'=' * 96}\n[E] 7페어 계좌 통합 (본표본+홀드아웃) — 규모 감각용", flush=True)
    syms = list(main["meta"]) + list(hold["meta"])
    months = float(np.mean([r["meta"][s]["_pair"]["months"]
                            for r in (main, hold) for s in r["meta"]]))
    out = {}
    for vn in ("BASE", "MH_BIAS", "MH_BIAS5", "MH", "NOTURTLE", "PHASEA"):
        if vn not in main["trades"] or vn not in hold["trades"]:
            continue
        d = A.pack(main["trades"][vn] + hold["trades"][vn])
        if d["n"] == 0:
            continue
        m = A.metrics(d, months, None, rng, len(syms))
        w, rr = wr_rr(d["r"], d["raw"])
        print(f"  {vn:<10} n={m['n']:>4} 빈도{m['freq']:>5.2f}/페어/월 "
              f"(계좌 {m['n'] / months:>5.2f}/월) 건당R {m['r_mean']:>+6.3f} "
              f"승률{w:>3.0f}% 복리 {m['compound']:>8.3f}x MDD {m['mdd']:>4.0f}% "
              f"파산 {m['ruin']:>5.1f}% 부트5% {m['p5']:>6.3f}", flush=True)
        out[vn] = dict(m, wr=w, rr=rr)
    out["_meta"] = {"syms": syms, "months": months}
    return out


def main_fn() -> int:
    rng = np.random.default_rng(20260809)
    hold_name = "holdout"
    for a in sys.argv[1:]:
        if a.startswith("--holdout="):
            hold_name = a.split("=")[1]
    m = load("main")
    h = load(hold_name) if os.path.exists(
        os.path.join(IN_DIR, f"runs_{hold_name}.json")) else None

    out: dict = {"date": "2026-08-09", "multiplicity_added": MULTIPLICITY,
                 "bonferroni_added": 0.05 / MULTIPLICITY}
    out["A_main"] = sec_a(m, "본표본 BTC+ETH")
    if h:
        out["A_holdout"] = sec_a(h, "홀드아웃 "
                                 + "·".join(s.replace("USDT", "") for s in h["meta"]))
    out["B"] = sec_b(m, rng, "본표본")
    if h:
        out["B_holdout"] = sec_b(h, rng, "홀드아웃")
    out["C"] = sec_c(m, rng, "본표본")
    if h:
        out["C_holdout"] = sec_c(h, rng, "홀드아웃")
    out["D"] = sec_d(m, h, rng)
    if h:
        out["E"] = sec_e(m, h, rng)
    print(f"\n{'=' * 96}\n[다중비교] 이 스크립트 추가 검정 {MULTIPLICITY}개 → "
          f"Bonferroni {0.05 / MULTIPLICITY:.4f}. conf2_rerun 사전등록 13개는 "
          f"별도 문턱 {0.05 / 13:.4f}.", flush=True)
    p = os.path.join(OUT_DIR, "supp.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"저장 → {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_fn())
