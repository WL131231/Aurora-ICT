"""Origo — 축별 연구에서 **반박을 견딘 손잡이들의 조합 검증** (2026-08-07).

## 배경
6개 축(SL·min_rr·사이징·청산·게이트·07/08AM)을 스윕한 뒤 6건의 독립 반박을 돌렸다.
반박 결과 "robust" 판정을 받은 **변경 후보는 0개**다. 반박을 살아남았지만
"uncertain" 으로 강등된 손잡이가 2개뿐이다:

  A) partial_tp_rr = 0   (부분익절 끄기)  — 청산 축
       · 청산축 보고서는 다중비교(Bonferroni 0.00128)와 연도제외 강건성으로 기각했다.
       · 반박 4는 조합 배포안의 나머지 두 축(trail 1.0 · flip 2.5)을 무효화했고,
         partial 축만 실체(56건 변경, 짝지은 p=0.0020, 부트 CI [+0.047,+0.208])로 남겼다.
       · 단 "두 파라미터화가 독립 확인"은 항등식이라 반박됨 → 확인은 1개 축뿐.

  B) regime 게이트 OFF   — 게이트 축
       · 게이트축 보고서는 잭나이프로 기각했으나 반박 5가 그 잭나이프를 무효화했다
         (R3: 기준선이 오히려 단일거래에 더 약함). 기각 사유가 사라져 "uncertain".

기각된 채로 남은 것: sl_dist_mult 전 후보 · min_rr 전 후보 · 사이징 14종 ·
trail/be/flip 단독 · 07/08AM 10종 · cond_align/nypm 게이트.
→ 조합 검증 대상은 A, B 둘뿐이다.

## 이 스크립트가 재는 것
1. A 단독 / B 단독 / A+B / (참고) A+B+반박된 축 을 **같은 하니스로 새로 백테**해서
   현행 기준선 대비 성적을 낸다.
2. **합이 부분의 합과 다른가** — ΔR합의 가법성을 직접 검사한다.
   A 는 청산 규칙이라 B 가 새로 들여보내는 거래에도 적용되므로 상호작용이 있다.
3. 반박 3의 교훈 반영: 복리 자산은 경로 운에 크게 흔들리므로 **1차 판정은 ΔR합**,
   복리는 보조 지표로만 쓴다.

## 하니스
live_parity.run_live_parity 경로. 게이트를 끄려면 게이트 이전 거래가 필요하므로
run_backtest_from_timeline 까지 직접 돌리고 LP.gate_* 를 사후 적용하되,
[0단계]에서 "세 게이트 전부 적용 = run_live_parity(kept)" 가 **거래 단위로 완전
일치**하는지 assert 한다. (게이트축 스크립트와 같은 방식이되 검증을 추가했다.)

## 측정 단위
R = (raw - 2×taker) / (|진입가-SL|/진입가). 복리는 raw 에 레버 7x·size 0.9·
동시보유 분할·DD 스로틀(-25%→×0.7)·파산(시드 20%) 적용.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import live_parity as LP  # noqa: E402
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from flip_ab_backtest import build_fvg_zones  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402
from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402


def _load_exit_mod():
    """청산축 스크립트를 그대로 임포트 — repartial/flip/시뮬을 재구현하지 않는다."""
    p = os.path.join(HERE, "axis_exit_2026-08-07.py")
    spec = importlib.util.spec_from_file_location("axis_exit_mod", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


EX = _load_exit_mod()

ROOT = os.path.dirname(HERE)
OUT_PATH = os.path.join(ROOT, "data", "axis", "combined.json")

PAIRS7 = LP.PAIRS
LIVE_PAIRS = ["BTCUSDT", "ETHUSDT"]

LEV, SIZE = 7.0, 0.9
RUIN, DD_PCT, DD_FACTOR = 0.20, 0.25, 0.7
N_PERM, N_BOOT = 20000, 2000
MIN_N = 30
FLIP_MIN_W = 4
RNG = np.random.default_rng(20260807)

BASE_EXIT = dict(trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, partial_tp_rr=1.5)
BASE_FLIP_R = LP.LIVE_FLIP_MIN_R      # 1.5
MONTHS = 59.0                         # 2021-07 ~ 2026-06


# ══════════════════════════════════════════ [0] 수집 (게이트 이전 + 게이트 플래그)
_PD: dict[str, EX.PairData] = {}
_CACHE: dict[tuple, dict] = {}


def collect(extra: dict) -> dict[str, list]:
    """설정 1개로 7페어 백테. 반환 sym -> [(trade, 게이트 플래그 dict)] (게이트 이전 전부).

    [0단계 검증] 세 게이트를 전부 적용한 결과가 run_live_parity 의 kept 와
    거래 단위로 완전 일치하는지 확인한다.
    """
    key = tuple(sorted(extra.items()))
    if key in _CACHE:
        return _CACHE[key]
    cfg_kw = dict(LP.LIVE_BASE)
    cfg_kw.update(extra)
    out = {}
    for sym in PAIRS7:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**cfg_kw)
        bt = run_backtest_from_timeline(df5, cached_setup_timeline(df5, cfg, sym), cfg)
        if sym not in _PD:
            _PD[sym] = EX.PairData(sym, df5)
        recs = []
        for t in bt.trades:
            ts = df5.index[t.entry_idx]
            recs.append((t, dict(
                regime=LP.gate_regime(t, sym),
                cond=LP.gate_cond_align(t, sym),
                nypm=LP.gate_nypm(ts),
            )))
        # 정합 검증 — 전 게이트 적용 == run_live_parity
        _, kept, _st = LP.run_live_parity(sym, extra)
        mine = [t for t, g in recs if g["regime"] and g["cond"] and g["nypm"]]
        assert len(mine) == len(kept), f"{sym} 게이트 정합 불일치 {len(mine)}!={len(kept)}"
        for a, b in zip(mine, kept):
            assert a.entry_idx == b.entry_idx and abs(a.raw_pnl_pct - b.raw_pnl_pct) < 1e-15
        out[sym] = recs
    _CACHE[key] = out
    return out


# ══════════════════════════════════════════ 시나리오 → 거래 행
def build(cfg_extra: dict, *, gates: tuple[bool, bool, bool], partial_tp_rr: float,
          pfrac: float, flip_r: float, pairs) -> list[dict]:
    """게이트 조합 + 청산 조합을 적용한 거래 행 목록."""
    by = collect(cfg_extra)
    g_reg, g_cond, g_ny = gates
    rows = []
    for sym in pairs:
        pd_ = _PD[sym]
        for t, gf in by[sym]:
            if g_reg and not gf["regime"]:
                continue
            if g_cond and not gf["cond"]:
                continue
            if g_ny and not gf["nypm"]:
                continue
            rf = EX._risk_frac(t)
            if rf <= 0:
                continue
            raw_p = EX.repartial(t, pd_, partial_tp_rr, pfrac) if partial_tp_rr > 0 \
                else t.raw_pnl_pct
            raw_f, ex_i, fired = EX.apply_flip_raw(pd_, t, flip_r)
            raw = raw_f if fired else raw_p
            ts = pd_.idx[t.entry_idx]
            rows.append(dict(
                sym=sym, ts=int(ts.value), ex_ts=int(pd_.idx[ex_i].value),
                raw=float(raw), r=float(EX._r_of(raw, rf)), year=int(ts.year),
                dir=("long" if EX._sign(t) > 0 else "short"),
                trend=float(t.entry_trend_pct or 0.0),
            ))
    rows.sort(key=lambda x: x["ts"])
    return rows


# ══════════════════════════════════════════ 복리 시뮬
def concurrency(rows) -> np.ndarray:
    s = np.array([x["ts"] for x in rows], dtype=np.int64)
    e = np.array([x["ex_ts"] for x in rows], dtype=np.int64)
    return np.array([int(np.count_nonzero((s <= e[i]) & (e >= s[i])))
                     for i in range(len(rows))], dtype=float)


def sim(raws, scale, lev=LEV):
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i in range(len(raws)):
        sz = SIZE * scale[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        eq *= (1.0 + raws[i] * sz * lev - 2.0 * TAKER_FEE_PCT * sz * lev)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return float(eq), 100.0 * mdd, True
    return float(eq), 100.0 * mdd, False


def compound(rows):
    raws = np.array([x["raw"] for x in rows])
    sc = 1.0 / concurrency(rows)
    eq, mdd, _ = sim(raws, sc)
    fin = np.empty(N_BOOT); ruin = 0
    n = len(raws)
    for k in range(N_BOOT):
        i = RNG.integers(0, n, size=n)
        e, _, r_ = sim(raws[i], sc[i])
        fin[k] = e; ruin += int(r_)
    p50, p5 = np.percentile(fin, [50, 5])
    return dict(compound=float(eq), mdd=float(mdd), ruin=float(100.0 * ruin / N_BOOT),
                boot_p50=float(p50), boot_p5=float(p5))


def summ(rows) -> dict:
    r = np.array([x["r"] for x in rows])
    w, lo = r[r > 0], r[r < 0]
    d = dict(n=len(r), r_mean=float(r.mean()), r_sum=float(r.sum()),
             r_se=float(r.std(ddof=1) / np.sqrt(len(r))),
             wr=float(100 * len(w) / len(r)),
             rr=float(w.mean() / abs(lo.mean())) if len(w) and len(lo) else float("nan"),
             per_month=len(r) / MONTHS)
    d.update(compound(rows))
    return d


# ══════════════════════════════════════════ 검정
def perm_paired(diffs, n=N_PERM):
    d = np.asarray(diffs)
    if d.size == 0:
        return float("nan"), float("nan")
    obs = d.mean()
    sgn = RNG.integers(0, 2, size=(n, d.size)) * 2 - 1
    dist = (sgn * d).mean(axis=1)
    return float(obs), float((int(np.count_nonzero(np.abs(dist) >= abs(obs) - 1e-12)) + 1) / (n + 1))


def perm_two(a, b, n=N_PERM):
    a, b = np.asarray(a), np.asarray(b)
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan")
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b]); na, tot = len(a), len(a) + len(b)
    order = RNG.random((n, tot)).argsort(axis=1)
    pm = pool[order]
    dist = pm[:, :na].mean(axis=1) - pm[:, na:].mean(axis=1)
    return float(obs), float((int(np.count_nonzero(np.abs(dist) >= abs(obs) - 1e-12)) + 1) / (n + 1))


def perm_rsum(cand, base, n=N_PERM):
    """ΔR합 순열검정 (반박 3 권고 — 복리 대신 R합을 통계량으로).

    귀무: 후보와 기준선 사이의 거래별 R 차이(공유분) 및 추가/소멸분의 부호가
    무작위. 공유분은 부호반전, 차이분(추가는 +R, 소멸은 -R)은 부호반전으로 섞는다.
    """
    mb = {(x["sym"], x["ts"]): x["r"] for x in base}
    mc = {(x["sym"], x["ts"]): x["r"] for x in cand}
    terms = []
    for k, v in mc.items():
        terms.append(v - mb[k] if k in mb else v)      # 공유 차이 / 추가분
    for k, v in mb.items():
        if k not in mc:
            terms.append(-v)                           # 소멸분
    t = np.asarray(terms)
    obs = t.sum()
    sgn = RNG.integers(0, 2, size=(n, t.size)) * 2 - 1
    dist = (sgn * t).sum(axis=1)
    return float(obs), float((int(np.count_nonzero(np.abs(dist) >= abs(obs) - 1e-12)) + 1) / (n + 1))


def split_sets(cand, base):
    mb = {(x["sym"], x["ts"]): x for x in base}
    mc = {(x["sym"], x["ts"]): x for x in cand}
    shared_d = [mc[k]["r"] - mb[k]["r"] for k in mc if k in mb]
    added = [mc[k] for k in mc if k not in mb]
    dropped = [mb[k] for k in mb if k not in mc]
    return shared_d, added, dropped


def rsum_terms(cand, base):
    """ΔR합을 거래 단위 항으로 분해. 반환 [(key, year, term)]."""
    mb = {(x["sym"], x["ts"]): x for x in base}
    mc = {(x["sym"], x["ts"]): x for x in cand}
    out = []
    for k, v in mc.items():
        if k in mb:
            out.append((k, v["year"], v["r"] - mb[k]["r"]))
        else:
            out.append((k, v["year"], v["r"]))
    for k, v in mb.items():
        if k not in mc:
            out.append((k, v["year"], -v["r"]))
    return out


def perm_terms(terms, n=N_PERM):
    t = np.asarray([x[2] for x in terms])
    if t.size == 0:
        return 0.0, float("nan")
    obs = t.sum()
    sgn = RNG.integers(0, 2, size=(n, t.size)) * 2 - 1
    dist = (sgn * t).sum(axis=1)
    return float(obs), float((int(np.count_nonzero(np.abs(dist) >= abs(obs) - 1e-12)) + 1) / (n + 1))


def robustness(cand, base):
    """연도제외 / 최대기여거래 제외 강건성 — 청산·게이트 축이 쓴 기각 잣대와 동일."""
    terms = rsum_terms(cand, base)
    yrs = sorted({y for _, y, _ in terms})
    per_year = {str(y): round(sum(v for _, yy, v in terms if yy == y), 3) for y in yrs}
    n_pos = sum(1 for v in per_year.values() if v > 0)
    tot = sum(v for _, _, v in terms)
    top_share = (max(per_year.values(), key=abs) / tot) if tot else float("nan")
    worst = None
    for y in yrs:
        sub = [x for x in terms if x[1] != y]
        d, p = perm_terms(sub)
        if worst is None or p > worst[2]:
            worst = (y, d, p)
    # 최대 기여 거래 1건 제외
    k_top = max(terms, key=lambda x: abs(x[2]))
    d_drop, p_drop = perm_terms([x for x in terms if x[0] != k_top[0]])
    return dict(year_dr=per_year, year_pos=f"{n_pos}/{len(yrs)}",
                year_top_share=float(top_share),
                loo_year=int(worst[0]), loo_year_drsum=worst[1], loo_year_p=worst[2],
                drop_top_drsum=d_drop, drop_top_p=p_drop,
                top_term=float(k_top[2]))


def years(rows) -> dict:
    o: dict[int, float] = {}
    for x in rows:
        o[x["year"]] = o.get(x["year"], 0.0) + x["r"]
    return {str(k): round(v, 2) for k, v in sorted(o.items())}


def cells(rows) -> dict:
    """국면×방향 셀 R합/건당R. 국면은 entry_trend_pct 부호+세기."""
    o: dict[str, list] = {}
    for x in rows:
        tr = x["trend"]
        reg = "상승" if tr > 0.4 else ("하락" if tr < -0.4 else "횡보")
        o.setdefault(f"{reg}×{'롱' if x['dir'] == 'long' else '숏'}", []).append(x["r"])
    return {k: dict(n=len(v), r=round(float(np.mean(v)), 3)) for k, v in sorted(o.items())}


def pr(s=""):
    print(s, flush=True)


# ══════════════════════════════════════════ main
def main() -> int:
    t0 = time.time()
    pr("=" * 96)
    pr("Origo — 반박 생존 손잡이 조합 검증  (2026-08-07)")
    pr("=" * 96)
    pr("반박에서 'robust' 판정 받은 변경 후보 = 0개.")
    pr("'uncertain' 으로 강등돼 살아남은 손잡이 = A) partial_tp_rr=0  B) regime 게이트 OFF")
    pr("→ 아래는 그 둘의 단독·조합 성적. (기각된 축은 참고행에만 표시)")

    EXIT_BASE = dict(BASE_EXIT)
    EXIT_A = dict(BASE_EXIT); EXIT_A["partial_tp_rr"] = 0.0
    EXIT_TRAIL = dict(EXIT_A); EXIT_TRAIL["trail_trigger"] = 1.0

    ALL_ON = (True, True, True)
    REG_OFF = (False, True, True)

    # (이름, cfg_extra, gates, partial_tp_rr, flip_r)
    SCEN = [
        ("기준선(현행)",            EXIT_BASE,  ALL_ON,  1.5, BASE_FLIP_R),
        ("A: partial 끔",           EXIT_A,     ALL_ON,  0.0, BASE_FLIP_R),
        ("B: regime OFF",           EXIT_BASE,  REG_OFF, 1.5, BASE_FLIP_R),
        ("A+B",                     EXIT_A,     REG_OFF, 0.0, BASE_FLIP_R),
        ("(참고) A+B+trail1.0+flip2.5", EXIT_TRAIL, REG_OFF, 0.0, 2.5),
    ]

    out = {"axis": "combined", "date": "2026-08-07",
           "robust_candidates": [],
           "surviving_uncertain": ["partial_tp_rr=0", "regime_gate_off"],
           "pairsets": {}}

    for pairs, tag in ((LIVE_PAIRS, "live2"), (PAIRS7, "power7")):
        pr("\n" + "=" * 96)
        pr(f"[{tag}] 페어 = {', '.join(p.replace('USDT', '') for p in pairs)}")
        pr("=" * 96)
        built = {}
        for name, cx, g, ptr, fr in SCEN:
            built[name] = build(cx, gates=g, partial_tp_rr=ptr, pfrac=0.5,
                                flip_r=fr, pairs=pairs)
        base = built["기준선(현행)"]
        bs = summ(base)
        pr(f"\n기준선: n={bs['n']} 월{bs['per_month']:.2f} 건당R{bs['r_mean']:+.4f} "
           f"(SE {bs['r_se']:.4f}) R합{bs['r_sum']:+.2f} 승률{bs['wr']:.1f}% "
           f"복리{bs['compound']:.2f}x MDD{bs['mdd']:.1f}% 파산{bs['ruin']:.1f}% "
           f"부트중앙{bs['boot_p50']:.2f}x 5%{bs['boot_p5']:.2f}x")
        pr(f"  연도R: {years(base)}")
        pr(f"  셀: {cells(base)}")

        rows_out = [dict(name="기준선(현행)", **bs, years=years(base), cells=cells(base))]
        pr("\n" + "-" * 96)
        pr(f"{'후보':30s} {'n':>4s} {'건당R':>8s} {'ΔR합':>8s} {'승률':>6s} "
           f"{'복리':>7s} {'MDD':>6s} {'파산':>6s} {'부트5%':>7s} {'p(ΔR합)':>9s}")
        pr("-" * 96)
        for name, cx, g, ptr, fr in SCEN[1:]:
            c = built[name]
            s = summ(c)
            shared_d, added, dropped = split_sets(c, base)
            d_rsum, p_rsum = perm_rsum(c, base)
            _, p_sh = perm_paired(shared_d) if shared_d else (0.0, float("nan"))
            add_r = [x["r"] for x in added]
            _, p_add = perm_two(add_r, [x["r"] for x in base]) if add_r else (0, float("nan"))
            rec = dict(name=name, **s, d_r=s["r_mean"] - bs["r_mean"], d_rsum=d_rsum,
                       p_rsum=p_rsum, n_shared=len(shared_d), n_added=len(added),
                       n_dropped=len(dropped),
                       shared_mean_d=(float(np.mean(shared_d)) if shared_d else 0.0),
                       p_shared=p_sh,
                       added_mean_r=(float(np.mean(add_r)) if add_r else float("nan")),
                       p_added=p_add, years=years(c), cells=cells(c))
            rec.update(robustness(c, base))
            rows_out.append(rec)
            pr(f"{name:30s} {s['n']:4d} {s['r_mean']:+8.4f} {d_rsum:+8.2f} "
               f"{s['wr']:5.1f}% {s['compound']:6.2f}x {s['mdd']:5.1f}% "
               f"{s['ruin']:5.1f}% {s['boot_p5']:6.2f}x {p_rsum:9.4f}")
            pr(f"{'':30s}   공유{len(shared_d)}건 ΔR평균{rec['shared_mean_d']:+.4f}"
               f"(p={p_sh:.4f}) / 추가{len(added)}건 건당R"
               f"{rec['added_mean_r'] if add_r else float('nan'):+.3f}(p={p_add:.4f})"
               f" / 소멸{len(dropped)}건")
            pr(f"{'':30s}   연도R: {years(c)}")
            pr(f"{'':30s}   셀: {cells(c)}")
            pr(f"{'':30s}   연도별ΔR합: {rec['year_dr']}  개선 {rec['year_pos']}  "
               f"최대연도비중 {rec['year_top_share'] * 100:.0f}%")
            pr(f"{'':30s}   [강건성] {rec['loo_year']}년 제외 → ΔR합 {rec['loo_year_drsum']:+.2f} "
               f"p={rec['loo_year_p']:.4f} / 최대기여거래({rec['top_term']:+.2f}R) 제외 → "
               f"ΔR합 {rec['drop_top_drsum']:+.2f} p={rec['drop_top_p']:.4f}")

        # ── 가법성 검사
        pr("\n" + "-" * 96)
        pr("[가법성] 합이 부분의 합과 같은가 — ΔR합 기준")
        dA = sum(x["r"] for x in built["A: partial 끔"]) - bs["r_sum"]
        dB = sum(x["r"] for x in built["B: regime OFF"]) - bs["r_sum"]
        dAB = sum(x["r"] for x in built["A+B"]) - bs["r_sum"]
        pr(f"  ΔR합(A)={dA:+.3f}  ΔR합(B)={dB:+.3f}  합={dA + dB:+.3f}  "
           f"실측 ΔR합(A+B)={dAB:+.3f}  상호작용={dAB - (dA + dB):+.3f}")
        # A 를 B 가 들여보낸 거래에만 적용했을 때의 기여
        b_rows = built["B: regime OFF"]
        ab_rows = built["A+B"]
        mb2 = {(x["sym"], x["ts"]): x["r"] for x in b_rows}
        extra_only = [(x["sym"], x["ts"]) for x in b_rows
                      if (x["sym"], x["ts"]) not in {(y["sym"], y["ts"]) for y in base}]
        mab = {(x["sym"], x["ts"]): x["r"] for x in ab_rows}
        d_extra = [mab[k] - mb2[k] for k in extra_only if k in mab]
        pr(f"  regime 추가분 {len(extra_only)}건에 A 를 적용한 효과: "
           f"ΔR합 {sum(d_extra):+.3f} / 건당 {np.mean(d_extra) if d_extra else 0:+.4f}")
        out["pairsets"][tag] = dict(
            rows=rows_out,
            additivity=dict(dA=dA, dB=dB, sum=dA + dB, dAB=dAB,
                            interaction=dAB - (dA + dB),
                            n_regime_added=len(extra_only),
                            d_extra_rsum=float(sum(d_extra)) if d_extra else 0.0),
        )

    out["n_scenarios_tried"] = len(SCEN) * 2
    out["bonferroni_alpha_this_run"] = 0.05 / (len(SCEN) - 1)
    # A 는 청산축 39개 설정 중에서 골라진 것이므로 그 축의 문턱을 그대로 이어받는다.
    out["bonferroni_alpha_exit_axis"] = 0.05 / 39
    out["verdicts"] = {
        "A: partial 끔": {
            "판정": "기각 유지 (재검토 1순위)",
            "근거": [
                "배포편성(BTC+ETH)에서는 청산축의 기각 사유였던 연도제외 강건성을 통과"
                "(2023 제외 ΔR합 +5.08 p=0.026, 최대기여거래 제외 +7.55 p=0.0037)",
                "그러나 7페어에서는 여전히 실패(2023 제외 p=0.110) — 청산축 기각 재현",
                "다중비교 문턱 미달: p=0.0018 > 0.05/39=0.00128 (청산축 39설정에서 선택된 축)",
                "효과가 2023 한 해에 42%(2페어)/60%(7페어) 몰려 있음",
            ],
        },
        "B: regime OFF": {
            "판정": "기각",
            "근거": [
                "ΔR합 p=0.29(2페어)/0.28(7페어) — 무의미",
                "추가분 건당R +0.330 vs 기준선 +0.643 — 들어오는 거래가 기존보다 나쁨",
                "연도 개선 3/6, 2025 제외 시 7페어 ΔR합이 -3.92 로 부호 반전(최대비중 150%)",
                "7페어 복리 3.74→2.86x · MDD 85.6→87.1% · 파산 6.7→8.9% 전부 악화",
                "2페어 부트 5%분위 1.69→1.48x 악화",
            ],
        },
        "A+B": {
            "판정": "기각",
            "근거": [
                "가법적 — 상호작용 -0.02(2페어)/+1.55(7페어, 총차의 6%). 상승효과 없음",
                "B 를 얹으면 p 가 나빠짐(0.0018→0.0135) 부트5%도 2.47→2.00x 로 희석",
                "2페어 연도제외 강건성 실패(2023 제외 p=0.107)",
            ],
        },
        "(참고) A+B+trail1.0+flip2.5": {
            "판정": "기각",
            "근거": ["trail 1.0·flip 2.5 는 반박 4에서 무효화된 축(p 하한 artifact / "
                     "BTC+ETH 에서 음수). 넣어도 2페어 p 는 0.0216 으로 A 단독보다 나쁘다."],
        },
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    pr(f"\n저장: {OUT_PATH}  ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
