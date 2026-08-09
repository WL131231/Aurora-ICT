"""#PARITY-VERIFY 2026-08-08 — 이식된 백테가 **라이브와 같은 분포**를 내는지 실측 대조.

## 왜

2026-08-08 라이브 매매기록 실측으로 백테와 라이브가 **다른 봇**임이 드러났고,
replay.py 에 정합 이식을 했다. 이식했다고 끝이 아니다 — 백테가 라이브와 같은
거래를 뽑는지 **항목 출현율·점수 분포·진입 빈도**로 검증해야 한다. 어긋나면
이후 모든 연구(배포 후보 `macro_high AND bias` 재판정 포함)가 무의미해진다.

## 기준선 (라이브 실측)

Origo 진입 56건의 confluence 항목 출현율. 표본이 56건이므로 백테가 정확히
같은 값을 낼 수는 없다 → **라이브 비율의 Wilson 95% 신뢰구간**을 허용범위로 쓰고
구간 밖이면 FAIL 로 표시한다. 표본오차를 넘는 차이만 "이식이 틀렸다"로 본다.

## 항목 → 백테 필드 매핑

    라이브 태그               백테 출처
    htf_support_weight        Trade.boosts "htf_support_weight=..._boost+N"
    po3_distribution          Trade.boosts "po3"
    turtle_soup               Trade.confluences "turtle_soup"   (Phase B 소스)
    bias                      Trade.confluences "bias=..."      (Phase A)
    sweep                     Trade.confluences "sweep"         (Phase A)
    macro                     Trade.confluences "macro*"        (Phase A)
    ob                        Trade.confluences "ob=..."        (Phase A)
    implied_fvg               Trade.confluences "implied_fvg"   (Phase B 소스)
    cisd                      Trade.boosts "cisd"

Phase A(FVG/IFVG) 셋업과 Phase B(turtle/mitigation/implied/rejection) 셋업은
**배타적**이다. Phase B 셋업은 ob/macro/sweep/bias 가점 경로를 아예 안 탄다.
따라서 turtle 비율이 오르면 macro/ob/sweep/bias 비율은 자동으로 내려간다.

## 산출물

    data/parity/verify.json — 항목별 대조표 + 점수 분포 + 빈도 + 판정
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from live_parity import GAPS, live_cfg  # noqa: E402

from aurora_ict.backtest.replay import run_backtest_from_timeline  # noqa: E402

OUT_DIR = "data/parity"
PAIRS = ["BTCUSDT", "ETHUSDT"]

# ── 라이브 실측 기준선 (Origo 진입 56건) ──
LIVE_N = 56
LIVE_RATE: dict[str, float] = {
    "htf_support_weight": 52 / 56,
    "po3_distribution": 52 / 56,
    "turtle_soup": 34 / 56,
    "bias": 11 / 56,
    "sweep": 8 / 56,
    "macro": 4 / 56,
    "ob": 4 / 56,
    "implied_fvg": 4 / 56,
    "cisd": 3 / 56,
}
# 라이브 표본에 **0건**이던 항목 — 백테에서 크게 나오면 그것도 불일치다.
LIVE_ZERO = ["mitigation_block", "rejection_block", "ote", "dol_counter", "smt"]

# 라이브 Origo 2.2 진입 빈도 (data/live/fst_snapshots, 2026-07-21~07-30, 12유저).
# 유저별 (진입수 / 그 유저가 돌린 페어수 / 기간) 평균. MMBM 모델은 제외 — 백테
# 미이식(GAPS)이라 대조 대상이 아니다.
LIVE_FREQ_PER_PAIR_MONTH = 2.63
LIVE_FREQ_RANGE = (2.26, 4.37)   # 유저별 산포 (이상치 SI1E 26페어 제외)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """이항비율 Wilson 95% 신뢰구간.

    정규근사(Wald)는 p 가 0/1 에 가까울 때 구간이 음수로 새거나 0폭이 된다.
    라이브 93%(52/56) 같은 값이 정확히 그 구간이라 Wilson 을 쓴다.

    Args:
        k: 성공 수. n: 표본 수. z: 표준정규 분위 (1.96=95%).
    Returns:
        (하한, 상한).
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def flags_of(t) -> dict:
    """Trade → 라이브 태그 이름 기준 항목 플래그 + htf boost 크기.

    Args:
        t: replay.Trade (confluences / boosts 필드 보유).
    Returns:
        {태그: bool} + {"htf_boost": int, "macro_pri": str, "phase": "A"/"B"}.
    """
    conf = list(t.confluences)
    bs = list(t.boosts)
    f = {k: False for k in LIVE_RATE}
    f.update({k: False for k in LIVE_ZERO})
    htf_b = 0
    for b in bs:
        if b.startswith("htf_support_weight="):
            f["htf_support_weight"] = True
            htf_b = int(b.rsplit("+", 1)[1])
        elif b == "po3":
            f["po3_distribution"] = True
        elif b == "cisd":
            f["cisd"] = True
        elif b == "ote":
            f["ote"] = True
        elif b == "smt":
            f["smt"] = True
        elif b.startswith("dol_counter"):
            f["dol_counter"] = True
    macro_pri = "none"
    for c in conf:
        if c.startswith("ob="):
            f["ob"] = True
        elif c.startswith("macro_high="):
            macro_pri = "high"
        elif c.startswith("macro_low="):
            macro_pri = "low"
        elif c.startswith("macro="):
            macro_pri = "normal"
        elif c == "sweep":
            f["sweep"] = True
        elif c.startswith("bias="):
            f["bias"] = True
        elif c == "turtle_soup":
            f["turtle_soup"] = True
        elif c == "implied_fvg":
            f["implied_fvg"] = True
        elif c == "mitigation_block":
            f["mitigation_block"] = True
        elif c == "rejection_block":
            f["rejection_block"] = True
    # macro 는 low 도 "출현"으로 센다 (라이브 태그 기준 — 점수 0 이어도 기록됨).
    f["macro"] = macro_pri != "none"
    phase = "B" if (f["turtle_soup"] or f["implied_fvg"]
                    or f["mitigation_block"] or f["rejection_block"]) else "A"
    return {**f, "htf_boost": htf_b, "macro_pri": macro_pri, "phase": phase}


def r_of(t) -> float:
    """거래 1건 R 배수. entry_sl 미기록이면 nan."""
    risk = abs(t.entry - t.entry_sl)
    if risk <= 0 or t.entry <= 0:
        return float("nan")
    return float(t.raw_pnl_pct) / (risk / t.entry)


def run_pair(sym: str, bars: int = 0) -> tuple[list[dict], dict]:
    """한 페어 정합 백테 → 거래 레코드.

    Args:
        sym: 심볼.
        bars: >0 이면 df5 마지막 N봉만 사용 (빠른 예비 확인용. 캐시 키가 달라짐).
    Returns:
        (레코드 리스트, 기간 메타).
    """
    df5 = _resample(_load_full(sym))
    if bars > 0:
        df5 = df5.iloc[-bars:]
    cfg = live_cfg(sym)
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    recs = []
    for t in bt.trades:
        f = flags_of(t)
        recs.append({
            "sym": sym,
            "ts": df5.index[t.entry_idx].isoformat(),
            "dir": t.direction,
            "score": int(t.confluence_score),
            "base": int(t.base_score),
            "r": r_of(t),
            "raw": float(t.raw_pnl_pct),
            "net": float(t.net_pnl_pct),
            "outcome": t.outcome,
            **{k: bool(f[k]) for k in list(LIVE_RATE) + LIVE_ZERO},
            "htf_boost": f["htf_boost"],
            "macro_pri": f["macro_pri"],
            "phase": f["phase"],
        })
    months = (df5.index[-1] - df5.index[0]).days / 30.44
    meta = {"bars": len(df5), "start": df5.index[0].isoformat(),
            "end": df5.index[-1].isoformat(), "months": months,
            "setup_bars": sum(1 for x in tl if x is not None)}
    return recs, meta


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """두 비율 차(p1-p2)의 Newcombe 95% 신뢰구간 (Wilson 기반, method 10).

    단순 Wald 차 구간은 극단 비율(93%)에서 신뢰도가 무너진다. 라이브 n=56 ·
    백테 n=수백 처럼 표본크기가 크게 다를 때도 안정적인 Newcombe 를 쓴다.

    Args:
        k1, n1: 백테 성공/표본. k2, n2: 라이브 성공/표본.
    Returns:
        (하한, 상한) — 0 을 포함하면 "표본오차로 설명 가능".
    """
    p1, p2 = (k1 / n1 if n1 else 0.0), (k2 / n2 if n2 else 0.0)
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d - math.hypot(p1 - l1, u2 - p2), d + math.hypot(u1 - p1, p2 - l2))


def compare(recs: list[dict]) -> list[dict]:
    """항목별 백테 출현율 vs 라이브 — %p 차이 + 2표본 검정.

    ⚠️ 라이브 n=56 만 보고 판정하면 백테 표본이 작을 때 오판한다(백테 14건에서
    100% 가 나와도 라이브 93% 와 모순이 아니다). **양쪽 표본크기를 모두 반영하는**
    Fisher 정확검정 + Newcombe 차 구간으로 판정한다. 9항목 동시검정이므로
    Holm 보정을 함께 적용해 다중비교로 인한 거짓 FAIL 을 막는다.
    """
    from scipy.stats import fisher_exact
    n = len(recs)
    rows = []
    for k, live_p in LIVE_RATE.items():
        cnt = sum(1 for r in recs if r[k])
        kl = round(live_p * LIVE_N)
        bt_p = cnt / n if n else 0.0
        p_val = float(fisher_exact([[cnt, n - cnt], [kl, LIVE_N - kl]])[1])
        lo, hi = newcombe(cnt, n, kl, LIVE_N)
        rows.append({
            "item": k, "live_pct": 100 * live_p, "bt_pct": 100 * bt_p,
            "diff_pp": 100 * (bt_p - live_p), "bt_n": cnt, "live_n": kl,
            "diff_ci_lo_pp": 100 * lo, "diff_ci_hi_pp": 100 * hi,
            "p": p_val,
        })
    # Holm 보정 — p 오름차순으로 alpha/(m-i) 와 비교.
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p"])
    m = len(rows)
    rejected = True
    for rank, i in enumerate(order):
        thr = 0.05 / (m - rank)
        if rejected and rows[i]["p"] <= thr:
            rows[i]["verdict"] = "FAIL"
            rows[i]["holm_thr"] = thr
        else:
            rejected = False
            rows[i]["verdict"] = "OK"
            rows[i]["holm_thr"] = thr
    return rows


def report(name: str, recs: list[dict], meta: dict,
           months_by_sym: dict[str, float] | None = None) -> dict:
    """대조표 출력 + 결과 dict.

    Args:
        name: 표 제목.
        recs: 거래 레코드.
        meta: 페어별 기간 메타 (빈도 분모).
        months_by_sym: 주면 빈도 분모를 이걸로 대체 (연도별 부분표 등).
    """
    n = len(recs)
    print(f"\n{'=' * 84}\n[{name}] 백테 진입 {n}건", flush=True)
    if n == 0:
        return {"n": 0}
    rows = compare(recs)
    print(f"\n  {'항목':<20}{'라이브':>8}{'백테':>8}{'차이':>10}"
          f"{'차 95%구간(%p)':>20}{'p':>9}  판정", flush=True)
    for r in rows:
        print(f"  {r['item']:<20}{r['live_pct']:>7.0f}%{r['bt_pct']:>7.0f}%"
              f"{r['diff_pp']:>+9.1f}%p"
              f"{r['diff_ci_lo_pp']:>+11.0f}~{r['diff_ci_hi_pp']:<+7.0f}"
              f"{r['p']:>9.4f}  {r['verdict']}", flush=True)

    zero = {}
    print("\n  [라이브 0건 항목 — 백테에서 튀면 그것도 불일치]", flush=True)
    for k in LIVE_ZERO:
        c = sum(1 for r in recs if r[k])
        zero[k] = 100 * c / n
        print(f"    {k:<18} {c:>5}건  {100 * c / n:>5.1f}%", flush=True)

    print("\n  [Phase 구성] (Phase B = turtle/mitigation/implied/rejection)", flush=True)
    ph = Counter(r["phase"] for r in recs)
    print(f"    A(FVG/IFVG) {ph['A']:>5}건 {100 * ph['A'] / n:>5.1f}%   "
          f"B(정통소스) {ph['B']:>5}건 {100 * ph['B'] / n:>5.1f}%   "
          f"(라이브 B = turtle61%+implied7% = 68%)", flush=True)

    print("\n  [htf_support boost 크기] — turtle(base1)+po3(1) 이 문턱5를 넘으려면 +3 필요",
          flush=True)
    hb = Counter(r["htf_boost"] for r in recs)
    for k in (0, 1, 2, 3):
        print(f"    +{k}: {hb.get(k, 0):>5}건 {100 * hb.get(k, 0) / n:>5.1f}%", flush=True)

    print("\n  [confluence_score 분포] (문턱 5 — 라이브도 sub_ 강제 >=5)", flush=True)
    sc = Counter(r["score"] for r in recs)
    for k in sorted(sc):
        rs = [r for r in recs if r["score"] == k]
        rr = [r["r"] for r in rs if r["r"] == r["r"]]
        print(f"    score={k}: {sc[k]:>5}건 ({100 * sc[k] / n:>5.1f}%) "
              f"ΣR={np.nansum(rr):>8.1f} 평균R={np.nanmean(rr):>+6.3f}", flush=True)
    print(f"    평균 score = {np.mean([r['score'] for r in recs]):.2f}", flush=True)

    print("\n  [항목 동시출현] — 라이브에서 turtle 진입은 base1+po3 1 이라 "
          "htf+3 없이는 문턱5 불가", flush=True)
    pb = [r for r in recs if r["phase"] == "B"]
    co = {}
    if pb:
        co["phaseB_htf3_pct"] = 100 * sum(1 for r in pb if r["htf_boost"] == 3) / len(pb)
        co["phaseB_po3_pct"] = 100 * sum(1 for r in pb if r["po3_distribution"]) / len(pb)
        print(f"    Phase B 중 htf=+3 {co['phaseB_htf3_pct']:.1f}% · "
              f"po3 {co['phaseB_po3_pct']:.1f}%", flush=True)
    pa = [r for r in recs if r["phase"] == "A"]
    if pa:
        co["phaseA_htf3_pct"] = 100 * sum(1 for r in pa if r["htf_boost"] == 3) / len(pa)
        print(f"    Phase A 중 htf=+3 {co['phaseA_htf3_pct']:.1f}% "
              f"(A 는 ob/macro/sweep/bias 로 점수를 벌 수 있어 htf 의존이 낮아야 정상)",
              flush=True)

    print("\n  [진입 빈도]", flush=True)
    freq = {}
    for sym in sorted({r["sym"] for r in recs}):
        rs = [r for r in recs if r["sym"] == sym]
        m = (months_by_sym or {}).get(sym) or meta[sym]["months"]
        freq[sym] = len(rs) / m
        print(f"    {sym:<10} {len(rs):>5}건 / {m:>6.1f}개월 = "
              f"{len(rs) / m:>5.2f}건/월", flush=True)
    avg = float(np.mean(list(freq.values()))) if freq else 0.0
    ratio = avg / LIVE_FREQ_PER_PAIR_MONTH
    print(f"    라이브 실측 {LIVE_FREQ_PER_PAIR_MONTH:.2f}건/페어/월 "
          f"(유저별 {LIVE_FREQ_RANGE[0]:.2f}~{LIVE_FREQ_RANGE[1]:.2f}) "
          f"→ 백테/라이브 = {ratio:.2f}배", flush=True)

    print("\n  [방향]", flush=True)
    dd = Counter(r["dir"] for r in recs)
    print(f"    long {dd.get('long', 0)} / short {dd.get('short', 0)}", flush=True)

    nfail = sum(1 for r in rows if r["verdict"] == "FAIL")
    return {
        "n": n, "items": rows, "live_zero_pct": zero,
        "phase_pct": {"A": 100 * ph["A"] / n, "B": 100 * ph["B"] / n},
        "htf_boost_pct": {str(k): 100 * hb.get(k, 0) / n for k in (0, 1, 2, 3)},
        "score_dist": {str(k): sc[k] for k in sorted(sc)},
        "score_mean": float(np.mean([r["score"] for r in recs])),
        "freq_per_month": freq, "freq_avg": avg, "freq_ratio_vs_live": ratio,
        "dir": {"long": dd.get("long", 0), "short": dd.get("short", 0)},
        "cooccur": co, "n_fail": nfail,
    }


def main() -> int:
    bars = 0
    pairs = list(PAIRS)
    out_name = "verify"
    for a in sys.argv[1:]:
        if a.startswith("--bars="):
            bars = int(a.split("=")[1])
        elif a.startswith("--pairs="):
            pairs = a.split("=")[1].split(",")
        elif a.startswith("--out="):
            out_name = a.split("=")[1]
    globals()["PAIRS"] = pairs
    os.makedirs(OUT_DIR, exist_ok=True)
    all_recs: list[dict] = []
    meta: dict[str, dict] = {}
    for sym in PAIRS:
        print(f"[백테] {sym} ...", flush=True)
        r, m = run_pair(sym, bars)
        all_recs += r
        meta[sym] = m
        print(f"   → {len(r)}건  ({m['start'][:10]}~{m['end'][:10]}, "
              f"{m['months']:.1f}개월, setup봉 {m['setup_bars']})", flush=True)

    out = {"date": "2026-08-08", "bars_slice": bars, "meta": meta,
           "live_baseline": {"n": LIVE_N,
                             "rate_pct": {k: 100 * v for k, v in LIVE_RATE.items()},
                             "freq_per_pair_month": LIVE_FREQ_PER_PAIR_MONTH}}
    out["combined"] = report("BTC+ETH 합산", all_recs, meta)
    for sym in PAIRS:
        rs = [r for r in all_recs if r["sym"] == sym]
        if rs:
            out[sym] = report(sym, rs, meta)

    # 연도별 — 라이브 표본은 2026년 여름 한 구간이다. 5년 전체와 어긋나는 항목이
    # 있으면 그게 **이식 오류인지 국면 차이인지** 연도별 안정성으로 갈린다.
    print(f"\n{'=' * 84}\n[연도별 항목 출현율] — 국면 차이 vs 이식 오류 판별용", flush=True)
    years = sorted({pd.Timestamp(r["ts"]).year for r in all_recs})
    hdr = "  {:<20}".format("항목") + "".join(f"{y:>8}" for y in years) + f"{'라이브':>9}"
    print(hdr, flush=True)
    year_tbl: dict[str, dict] = {}
    for k in LIVE_RATE:
        cells = []
        for y in years:
            rs = [r for r in all_recs if pd.Timestamp(r["ts"]).year == y]
            cells.append(100 * sum(1 for r in rs if r[k]) / len(rs) if rs else 0.0)
        year_tbl[k] = {str(y): c for y, c in zip(years, cells)}
        print("  {:<20}".format(k) + "".join(f"{c:>7.0f}%" for c in cells)
              + f"{100 * LIVE_RATE[k]:>8.0f}%", flush=True)
    ny = [sum(1 for r in all_recs if pd.Timestamp(r["ts"]).year == y) for y in years]
    print("  {:<20}".format("(거래수)") + "".join(f"{c:>8d}" for c in ny)
          + f"{LIVE_N:>9d}", flush=True)
    out["by_year"] = {"years": [str(y) for y in years], "n": ny, "rates": year_tbl}

    # 2026년만 — 라이브 표본과 가장 가까운 구간(데이터는 2026-06-18 까지).
    r26 = [r for r in all_recs if pd.Timestamp(r["ts"]).year == 2026]
    if r26:
        m26 = {}
        for sym in PAIRS:
            e = pd.Timestamp(meta[sym]["end"])
            s = max(pd.Timestamp("2026-01-01T00:00:00+00:00"),
                    pd.Timestamp(meta[sym]["start"]))
            m26[sym] = max((e - s).days / 30.44, 1e-9)
        out["y2026"] = report("2026년만 (라이브 표본과 최근접 구간)", r26, meta, m26)

    out["gaps"] = list(GAPS)
    path = os.path.join(OUT_DIR, f"{out_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    tpath = os.path.join(OUT_DIR, f"{out_name}_trades.json")
    with open(tpath, "w", encoding="utf-8") as f:
        json.dump(all_recs, f, ensure_ascii=False)
    print(f"\n저장 → {path} / {tpath}", flush=True)

    nf = out["combined"].get("n_fail", 0)
    print(f"\n{'=' * 84}\n[판정] 합산 기준 허용범위 이탈 항목 {nf}개", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
