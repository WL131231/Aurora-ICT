"""#PARITY-CONF 2026-08-08 — 재판정에 **쓰는 그 거래**로 정합을 먼저 검증한다.

## 왜 별도로 또 검증하나

scripts/parity_verify_2026-08-08.py 는 정합 백테를 **따로 한 번 더** 돌려서
라이브와 대조한다. 그런데 confluence 재판정이 실제로 쓰는 거래는
conf2_rerun 이 뽑은 `runs_*.json` 의 **BASE 변형**이다. 둘이 같은 cfg(live_cfg)
· 같은 timeline 이라 원리상 동일해야 하지만, "결론의 근거가 된 바로 그 표본"이
라이브와 어긋나지 않는지는 그 표본 위에서 직접 보여야 한다. 규칙 3번
(정합 검증을 먼저 출력하고, 어긋나면 아래 숫자를 신뢰하지 않는다)의 이행이다.

판정은 parity_verify 와 **같은 함수**(Fisher 정확검정 + Newcombe 차 구간 +
9항목 Holm 보정)를 import 해서 쓴다 — 기준이 두 개가 되면 안 되므로.

## 산출물

    data/conf2/parity_of_verdict.json
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 파일명에 하이픈이 있어 일반 import 가 안 된다 — 경로로 직접 로드.
# (기준이 두 개가 되지 않도록 판정 함수는 반드시 parity_verify 것을 쓴다.)
import importlib.util as _ilu  # noqa: E402

_spec = _ilu.spec_from_file_location(
    "parity_verify_mod",
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "parity_verify_2026-08-08.py"))
_pv = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_pv)
LIVE_FREQ_PER_PAIR_MONTH = _pv.LIVE_FREQ_PER_PAIR_MONTH
LIVE_FREQ_RANGE = _pv.LIVE_FREQ_RANGE
LIVE_N = _pv.LIVE_N
LIVE_RATE = _pv.LIVE_RATE
compare = _pv.compare

IN_DIR = OUT_DIR = "data/conf2"


def recs_to_parity(recs: list[dict]) -> list[dict]:
    """conf2 레코드 → parity_verify.compare 가 기대하는 플래그 스키마.

    Args:
        recs: conf2_rerun rec_of 산출 (flags/macro_pri/htf_boost/phase 보유).
    Returns:
        LIVE_RATE 키를 bool 로 가진 레코드 목록 (+ ts/sym/phase/htf_boost).
    """
    out = []
    for r in recs:
        f = r["flags"]
        out.append({
            "htf_support_weight": int(r["htf_boost"]) > 0,
            "po3_distribution": bool(f["po3"]),
            "turtle_soup": bool(f["turtle_soup"]),
            "bias": bool(f["bias"]),
            "sweep": bool(f["sweep"]),
            "macro": r["macro_pri"] != "none",
            "ob": bool(f["ob"]),
            "implied_fvg": bool(f["implied_fvg"]),
            "cisd": bool(f["cisd"]),
            "ts": r["ts"], "sym": r["sym"], "phase": r["phase"],
            "htf_boost": int(r["htf_boost"]), "dir": r["dir"],
        })
    return out


def check(path: str, label: str) -> dict:
    """한 그룹(main/holdout)의 BASE 거래를 라이브 실측과 대조."""
    with open(path, encoding="utf-8") as f:
        runs = json.load(f)
    recs = recs_to_parity(runs["trades"]["BASE"])
    n = len(recs)
    print(f"\n{'=' * 92}\n[정합 검증] {label} — BASE(현행 문턱5) 진입 {n}건", flush=True)
    rows = compare(recs)
    print(f"\n  {'항목':<20}{'라이브':>8}{'백테':>8}{'차이':>10}"
          f"{'차 95%구간(%p)':>20}{'p':>9}  판정", flush=True)
    for r in rows:
        print(f"  {r['item']:<20}{r['live_pct']:>7.0f}%{r['bt_pct']:>7.0f}%"
              f"{r['diff_pp']:>+9.1f}%p"
              f"{r['diff_ci_lo_pp']:>+11.0f}~{r['diff_ci_hi_pp']:<+7.0f}"
              f"{r['p']:>9.4f}  {r['verdict']}", flush=True)
    nfail = sum(1 for r in rows if r["verdict"] == "FAIL")

    ph = Counter(r["phase"] for r in recs)
    print(f"\n  Phase A(FVG/IFVG) {ph['A']}건 {100 * ph['A'] / n:.1f}%  ·  "
          f"Phase B(정통소스) {ph['B']}건 {100 * ph['B'] / n:.1f}%  "
          "(라이브 B = turtle 61% + implied 7% = 68%)", flush=True)
    hb = Counter(r["htf_boost"] for r in recs)
    print("  htf boost 분포: " + "  ".join(
        f"+{k}:{hb.get(k, 0)}({100 * hb.get(k, 0) / n:.0f}%)" for k in (0, 1, 2, 3)),
        flush=True)
    dd = Counter(r["dir"] for r in recs)
    print(f"  방향: long {dd.get('long', 0)} / short {dd.get('short', 0)}", flush=True)

    freq = {}
    for sym in sorted({r["sym"] for r in recs}):
        m = runs["meta"][sym]["_pair"]["months"]
        c = sum(1 for r in recs if r["sym"] == sym)
        freq[sym] = c / m
        print(f"  {sym:<10}{c:>6}건 / {m:>5.1f}개월 = {c / m:.2f}건/월", flush=True)
    avg = float(np.mean(list(freq.values())))
    print(f"  → 평균 {avg:.2f}건/페어/월 · 라이브 실측 {LIVE_FREQ_PER_PAIR_MONTH:.2f} "
          f"(유저별 {LIVE_FREQ_RANGE[0]:.2f}~{LIVE_FREQ_RANGE[1]:.2f}) "
          f"= {avg / LIVE_FREQ_PER_PAIR_MONTH:.2f}배", flush=True)

    years = sorted({pd.Timestamp(r["ts"]).year for r in recs})
    print("\n  [연도별 출현율] — 국면 차이 vs 이식 오류 판별", flush=True)
    print("  {:<20}".format("항목") + "".join(f"{y:>8}" for y in years)
          + f"{'라이브':>9}", flush=True)
    ytbl = {}
    for k in LIVE_RATE:
        cells = []
        for y in years:
            rs = [r for r in recs if pd.Timestamp(r["ts"]).year == y]
            cells.append(100 * sum(1 for r in rs if r[k]) / len(rs) if rs else 0.0)
        ytbl[k] = {str(y): c for y, c in zip(years, cells)}
        print("  {:<20}".format(k) + "".join(f"{c:>7.0f}%" for c in cells)
              + f"{100 * LIVE_RATE[k]:>8.0f}%", flush=True)
    ny = [sum(1 for r in recs if pd.Timestamp(r["ts"]).year == y) for y in years]
    print("  {:<20}".format("(거래수)") + "".join(f"{c:>8d}" for c in ny)
          + f"{LIVE_N:>9d}", flush=True)

    print(f"\n  [판정] 허용범위 이탈(FAIL) 항목 {nfail}개", flush=True)
    return {"n": n, "items": rows, "n_fail": nfail,
            "phase_pct": {"A": 100 * ph["A"] / n, "B": 100 * ph["B"] / n},
            "htf_boost_pct": {str(k): 100 * hb.get(k, 0) / n for k in (0, 1, 2, 3)},
            "freq_per_month": freq, "freq_avg": avg,
            "freq_ratio_vs_live": avg / LIVE_FREQ_PER_PAIR_MONTH,
            "by_year": {"years": [str(y) for y in years], "n": ny, "rates": ytbl}}


def main() -> int:
    out = {}
    for g, label in (("main", "본 표본 BTC+ETH"), ("holdout", "홀드아웃 5페어")):
        p = os.path.join(IN_DIR, f"runs_{g}.json")
        if os.path.exists(p):
            out[g] = check(p, label)
    path = os.path.join(OUT_DIR, "parity_of_verdict.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n저장 → {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
