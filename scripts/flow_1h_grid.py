"""Flow Engine 1h 백테스트 — 우리 표준 평가(복리·7x·동시보유·DD스로틀).

파트너 지시(2026-08-08): BTCUSDT+ETHUSDT 1h 에서
  · 진입모드 3종(Divergence Only / Momentum Cross / Both)
  · HTF 필터 on/off
  · TP/SL 모드 2종(ATR Dynamic / Fixed Percent)
= 3×2×2 = **12개 설정**을 전부 돌려 우리 표준 지표를 산출한다.
비용은 왕복 0.08%(우리) / 0.12%(Pine) 둘 다, 레버리지는 7x(주) / 10x(병기).

평가 규약은 scripts/origo_leverage_verify.py 의 sim 을 따른다.
  size_pct 0.9 · 동시보유 분할 · DD -25% 초과 시 리스크 ×0.7 · 시드 20% 이하 파산

★ 해석은 하지 않는다. 숫자만 산출해 data/flow/1h.json 으로 저장한다.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_engine import backtest, load  # noqa: E402

RES = "C:/Users/지영민/Desktop/Aurora-ICT-research"
PAIRS = ("BTCUSDT", "ETHUSDT")
TF = "1h"

SIZE = 0.9                      # LIVE_BASE["size_pct"]
DD_PCT, DD_FACTOR = 0.25, 0.7   # origo_dd_throttle_pct / _factor
RUIN = 0.20                     # 시드 20% 이하 = 파산
N_BOOT = 2000
RNG = np.random.default_rng(20260808)

# (레버리지, 왕복 수수료) — 7x/우리0.08% 가 주 지표, 나머지는 병기.
# 0bp 는 판정용이 아니라 **분해 진단**: 파산이 수수료 드래그 때문인지 분리한다.
COMBOS = ((7, 0.0008), (7, 0.0012), (10, 0.0008), (10, 0.0012), (7, 0.0))


def concurrency(s_ns: np.ndarray, e_ns: np.ndarray, half_open: bool = True):
    """각 거래가 열려 있는 동안의 동시 보유 수(자기 포함) — 벡터화.

    구간을 **반개구간 [진입, 청산)** 으로 본다(``half_open=True``, 기본).

    ⚠️ 왜 닫힌구간이 아닌가: 이 전략은 반대신호 시 청산 후 즉시 역진입한다
    (allow_reverse). 그러면 앞 거래의 청산봉 == 뒤 거래의 진입봉이라 닫힌구간
    으로 세면 **실제로는 겹치지 않는 연속 거래가 겹침으로 잡힌다**. 실측(BTC 1h
    Both/Fixed): 페어 내 엄격중첩 0건인데 경계접촉 748건 → 연쇄되면 동시보유가
    12까지 부풀고 건당 size 가 근거 없이 줄어든다. 페어 내 동시보유는 1개뿐이며
    진짜 동시보유는 BTC×ETH 뿐이므로 반개구간이 맞다.
    ``half_open=False`` 는 Origo 하니스와 같은 닫힌구간(대조용).
    """
    ss, ee = np.sort(s_ns), np.sort(e_ns)
    if half_open:
        a = np.searchsorted(ss, e_ns, side="left")    # s_j <  e_i
        b = np.searchsorted(ee, s_ns, side="right")   # e_j <= s_i
    else:
        a = np.searchsorted(ss, e_ns, side="right")   # s_j <= e_i
        b = np.searchsorted(ee, s_ns, side="left")    # e_j <  s_i
    return (a - b).astype(float)   # 자기 자신 포함


def sim_vec(raw_m: np.ndarray, sc_m: np.ndarray, lev: float, fee: float):
    """복리 시뮬을 **복제(replicate) 축으로 벡터화**. raw_m/sc_m 은 (K, n).

    Returns: (최종배수, MDD 비율, 파산여부, 파산 시점 거래번호(-1=생존),
              파산정지 없이 끝까지 간 자산) — 모두 (K,)

    ``eq_nr`` 은 파산 판정으로 멈추지 않고 끝까지 굴린 값이다. 12개 설정이 전부
    파산하면 최종자산이 파산 문턱에 고정되어 서로 구별이 안 되므로, 순위 비교용
    보조 지표로 함께 낸다(음수 방지 위해 0 에서 바닥).
    """
    k, n = raw_m.shape
    eq = np.ones(k)
    eq_nr = np.ones(k)
    peak_nr = np.ones(k)
    peak = np.ones(k)
    mdd = np.zeros(k)
    alive = np.ones(k, bool)
    ruined = np.zeros(k, bool)
    r_at = np.full(k, -1, dtype=int)
    for t in range(n):
        sz = SIZE * sc_m[:, t]
        sz = np.where(eq < peak * (1.0 - DD_PCT), sz * DD_FACTOR, sz)
        step = raw_m[:, t] * sz * lev - fee * sz * lev
        eq = np.where(alive, np.maximum(eq * (1.0 + step), 0.0), eq)
        sz2 = SIZE * sc_m[:, t]
        sz2 = np.where(eq_nr < peak_nr * (1.0 - DD_PCT), sz2 * DD_FACTOR, sz2)
        eq_nr = np.maximum(eq_nr * (1.0 + raw_m[:, t] * sz2 * lev
                                    - fee * sz2 * lev), 0.0)
        peak_nr = np.maximum(peak_nr, eq_nr)
        hit = alive & (eq <= RUIN)
        r_at = np.where(hit, t + 1, r_at)
        ruined |= hit
        alive &= ~hit
        peak = np.maximum(peak, eq)
        mdd = np.maximum(mdd, 1.0 - eq / peak)
    return eq, mdd, ruined, r_at, eq_nr


def run_variant(name: str, params: dict, cache: dict) -> dict:
    """한 설정을 BTC+ETH 통합 포트폴리오로 평가."""
    rows = []
    per_pair = {}
    span_lo, span_hi = None, None
    for sym in PAIRS:
        df = cache[sym]
        tr = backtest(df, **params)
        idx = df.index
        rr = np.array([t["r"] for t in tr], float) if tr else np.zeros(0)
        per_pair[sym] = dict(
            n=len(tr), r_mean=float(rr.mean()) if len(rr) else 0.0,
            r_sum=float(rr.sum()), win=float((rr > 0).mean()) if len(rr) else 0.0,
            r_med=float(np.median(rr)) if len(rr) else 0.0,
        )
        for t in tr:
            rows.append((int(idx[t["entry_idx"]].value),
                         int(idx[min(t["exit_idx"], len(idx) - 1)].value),
                         float(t["raw"]), float(t["r"])))
        span_lo = idx[0] if span_lo is None else min(span_lo, idx[0])
        span_hi = idx[-1] if span_hi is None else max(span_hi, idx[-1])

    rows.sort(key=lambda x: x[0])
    n = len(rows)
    months = (span_hi - span_lo).days / 30.44
    out = dict(name=name, params=params, n=n,
               monthly=round(n / months, 3), per_pair=per_pair)
    if n == 0:
        out.update(r_mean=0.0, r_sum=0.0, r_med=0.0, win=0.0,
                   compound=1.0, mdd=0.0, ruin=0.0, p5=1.0,
                   low_sample=True, by_combo={})
        return out

    s_ns = np.array([r[0] for r in rows])
    e_ns = np.array([r[1] for r in rows])
    raws = np.array([r[2] for r in rows])
    rs = np.array([r[3] for r in rows])
    conc = concurrency(s_ns, e_ns)
    conc_cl = concurrency(s_ns, e_ns, half_open=False)   # 대조용(Origo 방식)
    scale = 1.0 / conc

    out.update(r_mean=float(rs.mean()), r_med=float(np.median(rs)),
               r_sum=float(rs.sum()), win=float((rs > 0).mean()),
               conc_mean=float(conc.mean()), conc_max=float(conc.max()),
               conc_mean_closed=float(conc_cl.mean()),
               conc_max_closed=float(conc_cl.max()),
               eff_size_pct=float(100 * SIZE * (1.0 / conc).mean()),
               low_sample=bool(n < 30))

    # 부트스트랩 인덱스는 설정당 1회만 뽑아 (레버리지·비용) 간 비교를 공정하게
    bidx = RNG.integers(0, n, size=(N_BOOT, n))
    raw_b, sc_b = raws[bidx], scale[bidx]

    by = {}
    for lev, fee in COMBOS:
        eq0, mdd0, ru0, at0, nr0 = sim_vec(raws[None, :], scale[None, :], lev, fee)
        eqb, _, rub, _, nrb = sim_vec(raw_b, sc_b, lev, fee)
        key = f"{lev}x_{fee*10000:.0f}bp"
        by[key] = dict(
            compound=float(eq0[0]), mdd=float(mdd0[0]), ruin_actual=bool(ru0[0]),
            ruin=float(rub.mean()),
            p5=float(np.percentile(eqb, 5)), p50=float(np.percentile(eqb, 50)),
            ruin_at_trade=int(at0[0]), ruin_at_frac=(
                round(float(at0[0]) / n, 4) if at0[0] > 0 else None),
            compound_noruin=float(nr0[0]),
            p5_noruin=float(np.percentile(nrb, 5)),
            p50_noruin=float(np.percentile(nrb, 50)),
        )
    prim = by["7x_8bp"]
    out.update(compound=prim["compound"], mdd=prim["mdd"],
               ruin=prim["ruin"], p5=prim["p5"], by_combo=by)
    return out


def main() -> int:
    cache = {s: load(s, TF) for s in PAIRS}
    for s, d in cache.items():
        print(f"{s} {TF}: {len(d)}봉  {d.index[0]} ~ {d.index[-1]}", flush=True)

    variants = []
    for em in ("Divergence Only", "Momentum Cross", "Both"):
        for htf in (True, False):
            for mode in ("ATR Dynamic", "Fixed Percent"):
                p = dict(entry_mode=em, use_htf_filter=htf, tpsl_mode=mode)
                nm = f"{em} | HTF {'ON' if htf else 'OFF'} | {mode}"
                print(f"\n--- {nm}", flush=True)
                v = run_variant(nm, p, cache)
                variants.append(v)
                print(f"  n={v['n']} 월{v['monthly']:.2f} "
                      f"건당{v['r_mean']:+.3f}R 승률{v['win']*100:.1f}% "
                      f"ΣR{v['r_sum']:+.1f}", flush=True)
                for k, b in v.get("by_combo", {}).items():
                    at = b["ruin_at_trade"]
                    print(f"    {k}: 자산 {b['compound']:.4g}x · MDD "
                          f"{b['mdd']*100:.1f}% · 파산확률 {b['ruin']*100:.1f}% "
                          f"· p5 {b['p5']:.4g}x · "
                          f"{'파산 '+str(at)+'번째거래' if at > 0 else '생존'}"
                          f" · 무정지자산 {b['compound_noruin']:.3g}x", flush=True)

    os.makedirs(f"{RES}/data/flow", exist_ok=True)
    doc = dict(tf=TF, pairs=list(PAIRS), n_settings_tried=len(variants),
               leverage_primary=7, size_pct=SIZE,
               dd_throttle=[DD_PCT, DD_FACTOR], ruin_level=RUIN,
               n_boot=N_BOOT, variants=variants)
    path = f"{RES}/data/flow/1h.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
