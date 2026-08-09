"""#AUTONOMOUS 2026-08-08: Flow Engine vs 기존 봇(Origo/MMBM/Cursus) 비교 판정.

선행 3단계(15m·1h·4h)에서 살아남은 것은 **4h · Divergence Only · Fixed Percent**
하나뿐이다(나머지 33개 변형은 전부 파산 또는 건당 R 음수).
여기서는 "절대 성적"이 아니라 **우리 것 대비 도입 가치**를 판정한다.

## 무엇을 하나
1. Origo 현행(라이브 정합 conf5, BTC+ETH)과 나란히 같은 잣대로 비교.
2. **중복도** — Flow 진입이 Origo/MMBM 진입과 같은 심볼·방향·1시간 이내인가.
   겹치지 않으면 "빈도 보강"이라는 별도 가치가 있다(MMBM 선례).
3. **결합 시뮬** — Origo + Flow(순수 추가분) 복리·낙폭·파산.
4. Cursus(1h DualST) 신호와의 겹침 — 진입 시각 기준 간이 확인.
5. 판정 기준 1·2번(순열검정 / 국면×방향 기저 통제) 실측.
   ⚠ 2026-07-29 히든 다이버전스가 **기저 때문에** 기각된 선례가 있고
   Flow 도 다이버전스 계열이므로, 기저 통제 플라시보를 결정 근거로 삼는다.

## 평가 규약 (origo_leverage_verify.sim 과 동일)
7x · size_pct 0.9 · 동시보유 분할 · DD -25% → ×0.7 · 파산 = 시드 20% 이하
· 왕복 taker 0.08%(우리) / 0.12%(Pine) 병기 · 부트스트랩 복원추출 2000회.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow_engine as F  # noqa: E402
import mmbm_full as M  # noqa: E402
from live_parity import run_live_parity  # noqa: E402

RES = "C:/Users/지영민/Desktop/Aurora-ICT-research"
SYMS = ("BTCUSDT", "ETHUSDT")
LEV = 7.0
SIZE = 0.9
RUIN = 0.20
DD_PCT, DD_FACTOR = 0.25, 0.7
FEE_US, FEE_PINE = 0.0008, 0.0012
N_BOOT = 2000
N_PERM = 20000          # 판정기준 1번 (무작위 진입 대비)
N_PERM_BASE = 5000      # 판정기준 2번 (국면×방향 기저 통제)
DEDUP_MS = 60 * 60 * 1000       # 같은 기회로 볼 시간 창 (±1h)
RNG = np.random.default_rng(20260808)

# 선행 단계에서 유일하게 살아남은 설정
BEST = dict(entry_mode="Divergence Only", use_htf_filter=True,
            tpsl_mode="Fixed Percent")
BEST_NAME = "Flow 4h Div|HTF ON|Fixed%"
ALT = dict(entry_mode="Divergence Only", use_htf_filter=False,
           tpsl_mode="Fixed Percent")
ALT_NAME = "Flow 4h Div|HTF OFF|Fixed%"
TF = "4h"


# ══════════════════════════════════════════════════════════════════
# 1. 거래 수집 — 전부 같은 스키마 (sym, ent(ms), ex(ms), raw, r, dir)
# ══════════════════════════════════════════════════════════════════

CACHE = os.path.join(RES, "data/flow/_rows")


def cached(name: str, fn):
    """소스별 거래 수집 캐시 — 중간에 죽어도 다시 안 돌리게(Origo 가 매우 느리다)."""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, f"{name}.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    rows = fn()
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, default=float)
    print(f"  [캐시 저장] {name} {len(rows)}건", flush=True)
    return rows


def flow_rows(params: dict, tf: str = TF) -> list[dict]:
    out = []
    for sym in SYMS:
        df = F.load(sym, tf)
        for t in F.backtest(df, **params):
            out.append(dict(src="FLOW", sym=sym,
                            ent=int(t["ts"].value // 10**6),
                            ex=int(t["ex"].value // 10**6),
                            raw=float(t["raw"]), r=float(t["r"]),
                            dir=int(t["dir"])))
    out.sort(key=lambda x: x["ent"])
    return out


def origo_rows() -> list[dict]:
    """Origo 현행 = 라이브 정합(conf5 · regime · cond_align · nypm 게이트).

    페어 하나가 5분봉 5년 리플레이라 수 분 걸린다 → 페어별로 따로 캐시한다.
    """
    out = []
    for sym in SYMS:
        out.extend(cached(f"origo_{sym}", lambda s=sym: _origo_one(s)))
    out.sort(key=lambda x: x["ent"])
    return out


def _origo_one(sym: str) -> list[dict]:
    out = []
    if True:
        df5, kept, _ = run_live_parity(sym)
        idx = df5.index
        for t in kept:
            risk = abs(float(t.entry) - float(getattr(t, "entry_sl", 0.0) or 0.0))
            out.append(dict(
                src="ORIGO", sym=sym,
                ent=int(idx[t.entry_idx].value // 10**6),
                ex=int(idx[min(t.exit_idx, len(idx) - 1)].value // 10**6),
                raw=float(t.raw_pnl_pct),
                r=(float(t.raw_pnl_pct) * float(t.entry) / risk) if risk > 0 else 0.0,
                dir=1 if str(getattr(t.direction, "value", t.direction)).lower()
                    == "long" else -1))
    out.sort(key=lambda x: x["ent"])
    return out


def mmbm_rows() -> list[dict]:
    out = []
    for sym in SYMS:
        _df, tr = M.backtest(sym, detail=True)
        for ent_ms, _net, d, ex_ms, r_mult, gross in tr:
            out.append(dict(src="MMBM", sym=sym, ent=int(ent_ms), ex=int(ex_ms),
                            raw=float(gross), r=float(r_mult), dir=int(d)))
    out.sort(key=lambda x: x["ent"])
    return out


def cursus_signals() -> list[dict]:
    """Cursus 1h DualST 진입 신호 시각·방향 (신호봉 다음 봉 = 실제 진입)."""
    import dst_trend_bt_clamped as DST
    out = []
    for sym in SYMS:
        df = DST._load_1h(sym)
        sig = DST._signals(df)
        idx = sig.index
        b = sig["buy_sig"].to_numpy()
        s = sig["sell_sig"].to_numpy()
        for i in range(len(idx) - 1):
            if b[i]:
                out.append(dict(src="CURSUS", sym=sym,
                                ent=int(idx[i + 1].value // 10**6), dir=1))
            elif s[i]:
                out.append(dict(src="CURSUS", sym=sym,
                                ent=int(idx[i + 1].value // 10**6), dir=-1))
    out.sort(key=lambda x: x["ent"])
    return out


# ══════════════════════════════════════════════════════════════════
# 2. 평가 — 우리 표준 복리 시뮬
# ══════════════════════════════════════════════════════════════════

def _conc(rows) -> np.ndarray:
    n = len(rows)
    s = np.array([r["ent"] for r in rows], dtype=np.int64)
    e = np.array([r["ex"] for r in rows], dtype=np.int64)
    out = np.empty(n)
    for i in range(n):
        out[i] = int(((s <= e[i]) & (e >= s[i])).sum())
    return np.maximum(out, 1.0)


def sim(raw: np.ndarray, scale: np.ndarray, lev: float, fee: float):
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i in range(len(raw)):
        sz = SIZE * scale[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        eq *= (1.0 + raw[i] * sz * lev - fee * sz * lev)
        if eq <= 0.0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return eq, 100.0 * mdd, True
    return eq, 100.0 * mdd, False


def evaluate(rows, label, *, boot=N_BOOT, verbose=True) -> dict:
    if len(rows) < 5:
        if verbose:
            print(f"  {label:<26}{len(rows):>5}  표본부족(<5)", flush=True)
        return dict(name=label, n=len(rows), low_sample=True)
    rows = sorted(rows, key=lambda x: x["ent"])
    raw = np.array([x["raw"] for x in rows], float)
    r = np.array([x["r"] for x in rows], float)
    scale = 1.0 / _conc(rows)
    span_m = (max(x["ex"] for x in rows) - min(x["ent"] for x in rows)) / 86400000 / 30.44
    n = len(rows)

    res = dict(name=label, n=n, monthly=n / max(span_m, 1e-9),
               r_mean=float(r.mean()), r_med=float(np.median(r)),
               r_sum=float(r.sum()), win=float((r > 0).mean()),
               conc=float(_conc(rows).mean()), low_sample=bool(n < 30))
    for fee, tag in ((FEE_US, "0.08"), (FEE_PINE, "0.12")):
        eq0, mdd0, dead0 = sim(raw, scale, LEV, fee)
        fin = np.empty(boot)
        ruin = 0
        for k in range(boot):
            idx = RNG.integers(0, n, size=n)
            e_, _, d_ = sim(raw[idx], scale[idx], LEV, fee)
            fin[k] = e_
            ruin += int(d_)
        res[f"eq_{tag}"] = float(eq0)
        res[f"mdd_{tag}"] = float(mdd0)
        res[f"ruin_hist_{tag}"] = bool(dead0)
        res[f"ruin_{tag}"] = 100.0 * ruin / boot
        res[f"p50_{tag}"] = float(np.percentile(fin, 50))
        res[f"p5_{tag}"] = float(np.percentile(fin, 5))
    if verbose:
        print(f"  {label:<26}{n:>5}{res['monthly']:>7.2f}{r.mean():>8.3f}"
              f"{100 * res['win']:>5.0f}%{res['eq_0.08']:>9.2f}x"
              f"{res['mdd_0.08']:>7.1f}%{res['ruin_0.08']:>7.1f}%"
              f"{res['p5_0.08']:>8.2f}x{res['p50_0.08']:>8.2f}x", flush=True)
    return res


def header():
    print(f"  {'구성':<26}{'거래':>5}{'월':>7}{'건당R':>8}{'승률':>6}"
          f"{'자산':>10}{'낙폭':>8}{'파산':>8}{'p5':>8}{'부트중앙':>8}", flush=True)


# ══════════════════════════════════════════════════════════════════
# 3. 중복도
# ══════════════════════════════════════════════════════════════════

def dup_split(new_rows, base_rows, window=DEDUP_MS):
    """new 중 base 와 (같은 심볼·방향·|Δt|<=window) 인 것을 중복으로 분리."""
    by = {}
    for b in base_rows:
        by.setdefault((b["sym"], b["dir"]), []).append(b["ent"])
    for k in by:
        by[k] = np.array(sorted(by[k]), dtype=np.int64)
    dup, add = [], []
    for x in new_rows:
        arr = by.get((x["sym"], x["dir"]))
        hit = False
        if arr is not None and len(arr):
            j = int(np.searchsorted(arr, x["ent"]))
            for jj in (j - 1, j):
                if 0 <= jj < len(arr) and abs(int(arr[jj]) - x["ent"]) <= window:
                    hit = True
                    break
        (dup if hit else add).append(x)
    return dup, add


# ══════════════════════════════════════════════════════════════════
# 4. 플라시보 — 무작위 진입 (판정기준 1·2)
# ══════════════════════════════════════════════════════════════════

def placebo(params: dict, n_iter: int, *, regime_matched: bool, tf: str = TF):
    """같은 청산엔진·같은 신호 개수·같은 방향 비율의 **무작위 진입**.

    regime_matched=True 면 롱은 htf_bull 봉, 숏은 htf_bear 봉에서만 뽑는다
    → "상승장 롱 재확인" 기저를 통제한 비교 (판정기준 2번).
    """
    prep = []
    real_r, real_raw = [], []
    for sym in SYMS:
        df = F.load(sym, tf)
        sig_kw = {k: {**F.SIG_DEFAULTS, **params}[k] for k in F.SIG_DEFAULTS}
        sig = F.signals(df, **sig_kw)
        buy = sig["buy_sig"].to_numpy().copy()
        sell = sig["sell_sig"].to_numpy().copy()
        nb, ns = int(buy.sum()), int(sell.sum())
        lo = int(min(np.argmax(buy | sell), len(buy) - 1))
        hi = len(buy) - 2
        atr = F._atr(df, params.get("atr_period", 14)).to_numpy()
        ok = np.isfinite(atr)
        pool_all = np.array([i for i in range(lo, hi) if ok[i]], dtype=np.int64)
        hb = sig["htf_bull"].to_numpy()
        hs = sig["htf_bear"].to_numpy()
        pool_l = pool_all[hb[pool_all]] if regime_matched else pool_all
        pool_s = pool_all[hs[pool_all]] if regime_matched else pool_all
        prep.append((df, sig, buy, sell, nb, ns, pool_l, pool_s))
        for t in F.backtest(df, **params):
            real_r.append(t["r"])
            real_raw.append(t["raw"])

    obs_mean = float(np.mean(real_r))
    obs_sum = float(np.sum(real_r))
    real_signals = F.signals
    dist_mean = np.empty(n_iter)
    dist_sum = np.empty(n_iter)
    dist_n = np.empty(n_iter)
    try:
        for k in range(n_iter):
            rs, cnt = [], 0
            for df, sig, buy, sell, nb, ns, pool_l, pool_s in prep:
                buy[:] = False
                sell[:] = False
                if nb and len(pool_l):
                    buy[RNG.choice(pool_l, size=min(nb, len(pool_l)), replace=False)] = True
                if ns and len(pool_s):
                    pick = RNG.choice(pool_s, size=min(ns, len(pool_s)), replace=False)
                    sell[pick] = True
                    buy[pick] = False        # 충돌 시 숏 우선(임의·대칭)
                sig["buy_sig"] = buy
                sig["sell_sig"] = sell
                F.signals = lambda d, _s=sig, **kw: _s
                tr = F.backtest(df, **params)
                rs.extend(t["r"] for t in tr)
                cnt += len(tr)
            dist_mean[k] = np.mean(rs) if rs else 0.0
            dist_sum[k] = np.sum(rs) if rs else 0.0
            dist_n[k] = cnt
            if (k + 1) % 2000 == 0:
                print(f"    ... {k + 1}/{n_iter}", flush=True)
    finally:
        F.signals = real_signals

    p_mean = float((dist_mean >= obs_mean).mean())
    p_sum = float((dist_sum >= obs_sum).mean())
    return dict(n_iter=n_iter, regime_matched=regime_matched,
                obs_r_mean=obs_mean, obs_r_sum=obs_sum,
                plc_r_mean_med=float(np.median(dist_mean)),
                plc_r_mean_p95=float(np.percentile(dist_mean, 95)),
                plc_r_sum_med=float(np.median(dist_sum)),
                plc_r_sum_p95=float(np.percentile(dist_sum, 95)),
                plc_n_med=float(np.median(dist_n)),
                p_r_mean=p_mean, p_r_sum=p_sum,
                plc_pos_frac=float((dist_mean > 0).mean()))


# ══════════════════════════════════════════════════════════════════
# 5. main
# ══════════════════════════════════════════════════════════════════

def year_table(rows, label):
    print(f"\n  [{label} 연도별]", flush=True)
    ys = np.array([dt.datetime.utcfromtimestamp(x["ent"] / 1000).year for x in rows])
    out = {}
    for y in sorted(set(ys.tolist())):
        sub = [x for x, k in zip(rows, ys == y, strict=False) if k]
        r = np.array([x["r"] for x in sub], float)
        raw = np.array([x["raw"] for x in sub], float)
        sc = 1.0 / _conc(sub) if len(sub) > 1 else np.ones(len(sub))
        eq, mdd, dead = sim(raw, sc, LEV, FEE_US)
        flag = " ※표본부족" if len(sub) < 30 else ""
        print(f"   {y}  {len(sub):>4}건  건당 {r.mean():+.3f}R  ΣR {r.sum():+6.1f}"
              f"  승률 {100 * (r > 0).mean():>3.0f}%  자산 {eq:>6.2f}x"
              f"  낙폭 {mdd:>5.1f}%{'  파산' if dead else ''}{flag}", flush=True)
        out[str(y)] = dict(n=len(sub), r_mean=float(r.mean()), r_sum=float(r.sum()),
                           win=float((r > 0).mean()), eq=float(eq), mdd=float(mdd),
                           ruin=bool(dead), low_sample=bool(len(sub) < 30))
    return out


def main() -> int:
    res: dict = {"meta": dict(
        lev=LEV, size=SIZE, dd=[DD_PCT, DD_FACTOR], ruin=RUIN,
        fees=[FEE_US, FEE_PINE], n_boot=N_BOOT, pairs=list(SYMS), tf=TF,
        dedup_window_h=DEDUP_MS / 3600000)}

    print("=== 수집 ===", flush=True)
    flow = cached("flow_best", lambda: flow_rows(BEST))
    flow_alt = cached("flow_alt", lambda: flow_rows(ALT))
    org = cached("origo", origo_rows)
    # MMBM 은 5분봉 5년 구조탐지라 이번 실행 창에서 완주하지 못했다(2회 중단).
    # 캐시가 있으면 쓰고, 없으면 건너뛴다 → 결론에 "MMBM 대조 미실시"로 명시.
    mmb_p = os.path.join(CACHE, "mmbm.json")
    mmb = cached("mmbm", mmbm_rows) if os.path.exists(mmb_p) else []
    cur = cached("cursus", cursus_signals)
    print(f"  Flow(best) {len(flow)} · Flow(alt) {len(flow_alt)} · "
          f"Origo {len(org)} · MMBM {len(mmb)} · Cursus신호 {len(cur)}", flush=True)

    # ── ① 기존 봇 대비 ────────────────────────────────────────────
    print("\n=== ① 단독 성적 (7x · 우리 0.08% · 동시보유분할 · DD스로틀) ===",
          flush=True)
    header()
    res["origo"] = evaluate(org, "Origo 현행(conf5)")
    res["flow_best"] = evaluate(flow, BEST_NAME)
    res["flow_alt"] = evaluate(flow_alt, ALT_NAME)
    res["mmbm"] = evaluate(mmb, "MMBM(미배선)")

    # ── ② 중복도 ─────────────────────────────────────────────────
    print("\n=== ② 중복도 (같은 심볼·방향·±1시간) ===", flush=True)
    dup_o, add_o = dup_split(flow, org)
    dup_m, add_m = dup_split(flow, mmb)
    dup_c, add_c = dup_split(flow, cur)
    dup_all, add_all = dup_split(flow, org + mmb)
    ov = dict(
        vs_origo=dict(dup=len(dup_o), add=len(add_o),
                      pct=100 * len(dup_o) / max(len(flow), 1)),
        vs_mmbm=dict(dup=len(dup_m), add=len(add_m),
                     pct=100 * len(dup_m) / max(len(flow), 1)),
        vs_cursus=dict(dup=len(dup_c), add=len(add_c),
                       pct=100 * len(dup_c) / max(len(flow), 1)),
        vs_both=dict(dup=len(dup_all), add=len(add_all),
                     pct=100 * len(dup_all) / max(len(flow), 1)))
    for k, v in ov.items():
        print(f"  Flow {len(flow)}건 중 {k:<10} 중복 {v['dup']:>4}건 "
              f"({v['pct']:.1f}%) · 순수추가 {v['add']}건", flush=True)
    # 역방향(우리 기준: Origo 가 Flow 에 얼마나 잡히나)
    d_rev, a_rev = dup_split(org, flow)
    ov["origo_covered_by_flow"] = dict(dup=len(d_rev), add=len(a_rev),
                                       pct=100 * len(d_rev) / max(len(org), 1))
    print(f"  (역) Origo {len(org)}건 중 Flow 와 겹치는 것 {len(d_rev)}건 "
          f"({ov['origo_covered_by_flow']['pct']:.1f}%)", flush=True)
    res["overlap"] = ov

    print("\n  [순수 추가분 성적]", flush=True)
    header()
    res["flow_add_vs_origo"] = evaluate(add_o, "Flow 추가분(vs Origo)")

    # ── ③ 결합 ───────────────────────────────────────────────────
    print("\n=== ③ 결합 시뮬 (한 계좌 · 동시보유 분할) ===", flush=True)
    header()
    res["comb_origo_flow"] = evaluate(org + add_o, "Origo + Flow(추가분)")
    res["comb_origo_flow_all"] = evaluate(org + flow, "Origo + Flow(전체)")
    res["comb_origo_mmbm"] = evaluate(org + dup_split(mmb, org)[1],
                                      "(참고) Origo + MMBM추가분")
    res["comb_all3"] = evaluate(org + dup_split(mmb, org)[1]
                                + dup_split(flow, org + mmb)[1],
                                "Origo + MMBM + Flow")

    # ── 연도 일관성 ──────────────────────────────────────────────
    res["years_flow"] = year_table(flow, BEST_NAME)
    res["years_origo"] = year_table(org, "Origo 현행")
    res["years_comb"] = year_table(sorted(org + add_o, key=lambda x: x["ent"]),
                                   "Origo + Flow(추가분)")

    # ── 롱/숏 분리 ───────────────────────────────────────────────
    print("\n=== 롱/숏 분리 (기저 의심 점검) ===", flush=True)
    header()
    for d_, lab in ((1, "롱"), (-1, "숏")):
        res[f"flow_{lab}"] = evaluate([x for x in flow if x["dir"] == d_],
                                      f"Flow {lab}")
    for d_, lab in ((1, "롱"), (-1, "숏")):
        res[f"origo_{lab}"] = evaluate([x for x in org if x["dir"] == d_],
                                       f"Origo {lab}")

    with open(os.path.join(RES, "data/flow/compare.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print("\n중간 저장 완료 → data/flow/compare.json", flush=True)

    # ── ④ 플라시보 ───────────────────────────────────────────────
    print(f"\n=== ④ 무작위 진입 플라시보 (판정기준 1) · {N_PERM}회 ===", flush=True)
    p1 = placebo(BEST, N_PERM, regime_matched=False)
    res["placebo_random"] = p1
    print(f"  실측 건당 {p1['obs_r_mean']:+.4f}R / ΣR {p1['obs_r_sum']:+.1f}"
          f"  vs 무작위 중앙 {p1['plc_r_mean_med']:+.4f}R "
          f"(95%분위 {p1['plc_r_mean_p95']:+.4f}) → p={p1['p_r_mean']:.4f} "
          f"(ΣR p={p1['p_r_sum']:.4f})", flush=True)

    print(f"\n=== ⑤ 국면×방향 기저 통제 플라시보 (판정기준 2) · {N_PERM_BASE}회 ===",
          flush=True)
    p2 = placebo(BEST, N_PERM_BASE, regime_matched=True)
    res["placebo_regime"] = p2
    print(f"  실측 건당 {p2['obs_r_mean']:+.4f}R / ΣR {p2['obs_r_sum']:+.1f}"
          f"  vs 기저(HTF방향일치 무작위) 중앙 {p2['plc_r_mean_med']:+.4f}R "
          f"(95%분위 {p2['plc_r_mean_p95']:+.4f}) → p={p2['p_r_mean']:.4f} "
          f"(ΣR p={p2['p_r_sum']:.4f})", flush=True)
    print(f"  ※ 기저 자체가 흑자인 비율 {100 * p2['plc_pos_frac']:.1f}%", flush=True)

    with open(os.path.join(RES, "data/flow/compare.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print("\n최종 저장 → data/flow/compare.json", flush=True)
    return 0


def main_cached() -> int:
    """실제로 완주한 경로 — 캐시된 거래열(Flow·Cursus)만으로 측정.

    ★ 실행 기록 (2026-08-08)
      Origo 라이브정합 리플레이가 완주하지 못했다. 5분봉 521,135봉 × 2페어이고
      타임라인 캐시(116MB pickle) **역직렬화만 560초를 넘겼다**(동시 실행 python
      7개 경합). 3회 시도 모두 강제 중단 → Origo 거래열 미수집.
      그래서 ①은 과제 제시 기준선과 나란히 놓고, ②의 Origo 중복도는 우연 일치
      기대값으로 상한을 계산하고, ③ 결합 시뮬은 **미측정**으로 남긴다.
      MMBM(5분봉 구조탐지)도 같은 이유로 미실시.
    """
    res = {}
    flow = cached("flow_best", lambda: flow_rows(BEST))
    alt = cached("flow_alt", lambda: flow_rows(ALT))
    cur = cached("cursus", cursus_signals)
    header()
    res["flow_best"] = evaluate(flow, BEST_NAME)
    res["flow_alt"] = evaluate(alt, ALT_NAME)
    for d_, lab in ((1, "long"), (-1, "short")):
        res[f"flow_best_{lab}"] = evaluate([x for x in flow if x["dir"] == d_],
                                           f"Flow best {lab}")
    res["years_flow_best"] = year_table(flow, BEST_NAME)
    for name, rows in (("best", flow), ("alt", alt)):
        dup, add = dup_split(rows, cur)
        res[f"ov_cursus_{name}"] = dict(n=len(rows), dup=len(dup), add=len(add),
                                        pct=100 * len(dup) / len(rows))
    res["placebo_random"] = placebo(BEST, 4000, regime_matched=False)
    res["placebo_regime"] = placebo(BEST, 4000, regime_matched=True)
    with open(os.path.join(RES, "data/flow/compare.json"), "w",
              encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    return 0


if __name__ == "__main__":
    raise SystemExit(main_cached())
