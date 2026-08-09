"""#AUTONOMOUS 2026-08-08: Flow Engine **검증 배터리** (파트너 지시).

선행 3 TF 백테(15m/1h/4h × 12설정 = 36설정) 결과 유일한 생존 후보:
    **4h · Divergence Only · HTF ON · Fixed Percent**  (n=176 · +0.157R · 7x 3.21x)
차점: 4h · Divergence Only · HTF OFF · Fixed% (n=371 · +0.132R · 2.60x)
이 둘에 검증 배터리 5종을 건다.

배터리
  1) 순열검정 20000회 — 무작위 진입(같은 방향구성·같은 청산엔진) 대비
  2) **국면 × 방향 기저 통제** — 같은 국면·같은 방향의 무작위 진입과 나란히.
     ⚠️ 2026-07-29 히든 다이버전스가 정확히 이 기저에서 죽었다. 같은 계열이므로 필수.
  3) 연도 일관성
  4) 홀드아웃 5페어(SOL·XRP·DOGE·LINK·HYPE) — **파라미터 고정**
  5) 다이버전스 신호의 실제 기여 (div vs mom 분해)

핵심 도구 = **R-맵**. 각 봉 i 를 신호봉이라 치고 롱/숏 각각 진입했을 때의 R 을
전부 미리 계산해 둔다(청산 엔진은 flow_engine._walk_bar 그대로, 최대보유 CAP).
→ 순열검정의 "무작위 진입"이 실제 신호와 **완전히 같은 청산 규칙**을 쓰게 된다.
(실제 백테와 다른 점은 포지션 배타성·역신호 청산뿐. 후보 변형에서 reverse 는
 176건 중 2건(1.1%)이라 무시 가능 — 아래에서 실측 병기.)

국면 정의는 기존 연구와 동일: BTC 30일 변화율 ±15% → 상승/하락/횡보 (인과 shift).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_engine import (  # noqa: E402
    BT_DEFAULTS, SIG_DEFAULTS, _Pos, _atr, _tp_sl_pct, _walk_bar, backtest, load, signals,
)

RES = "C:/Users/지영민/Desktop/Aurora-ICT-research"
TF = "4h"
MAIN = ("BTCUSDT", "ETHUSDT")
HOLDOUT = ("SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT")
CAP = 120          # 최대보유 봉수(4h → 20일). 실측 최장 거래 32봉이라 여유 3.7배
N_PERM = 20000
SEED = 20260808
REGIME_THR = 0.15  # BTC 30일 변화율 문턱 — fib_regime_mtf.REGIME_THR 과 동일
N_SETTINGS = 36    # 3 TF × 12 설정 (선행 단계 전체)
ALPHA_B = 0.05 / N_SETTINGS

CAND = {
    "A: Div|HTF ON|Fixed%": dict(entry_mode="Divergence Only", use_htf_filter=True,
                                 tpsl_mode="Fixed Percent"),
    "B: Div|HTF OFF|Fixed%": dict(entry_mode="Divergence Only", use_htf_filter=False,
                                  tpsl_mode="Fixed Percent"),
}


# ══════════════════════════════════════════════════════════════════
# 1. R-맵 — 모든 봉 × 롱/숏 의 단독거래 R
# ══════════════════════════════════════════════════════════════════

def rmap(df: pd.DataFrame, o: dict) -> dict:
    """각 봉을 신호봉으로 간주한 단독거래 결과.

    Args:
        df: OHLCV (UTC index)
        o: SIG_DEFAULTS+BT_DEFAULTS 병합 옵션
    Returns:
        dict(r_long, r_short, dur_long, dur_short, valid, cap_hit)
        - r_*: 해당 봉 신호 → 다음 봉 시가 진입 → 청산까지의 R 배수
        - valid: ATR 유효 + 진입/감시봉 존재
    """
    n = len(df)
    O = df["open"].to_numpy(float)
    H = df["high"].to_numpy(float)
    L = df["low"].to_numpy(float)
    C = df["close"].to_numpy(float)
    atr = _atr(df, o["atr_period"]).to_numpy(float)
    alloc = np.array(o["alloc"], float) / 100.0
    slip = o["slip_pct"]

    r_l = np.full(n, np.nan)
    r_s = np.full(n, np.nan)
    d_l = np.zeros(n, int)
    d_s = np.zeros(n, int)
    cap_hit = 0

    for s_i in range(n - 2):
        if not np.isfinite(atr[s_i]):
            continue
        for is_long, ra, da in ((True, r_l, d_l), (False, r_s, d_s)):
            sgn = 1.0 if is_long else -1.0
            i = s_i + 1
            ref = C[s_i]
            tps_pct, sl_pct = _tp_sl_pct(atr[s_i], ref, o)
            fill_px = (O[i] if o["fill"] == "next_open" else C[s_i]) * (1 + sgn * slip)
            p = _Pos(is_long, ref, fill_px, tps_pct, sl_pct, o, i, s_i)
            realized: list[tuple[float, float]] = []
            end = min(n, i + CAP)
            j = i + o["exit_start_offset"]
            done = None
            while j < end:
                done = _walk_bar(p, O[j], H[j], L[j], C[j], realized, alloc, slip,
                                 o["intrabar"])
                if done is not None:
                    break
                j += 1
            if done is None:                     # CAP 도달 → 종가 청산
                j = min(j, n - 1)
                realized.append((max(1.0 - alloc[:p.hits].sum(), 0.0), C[j]))
                cap_hit += 1
            raw = sum(w * (px - p.fill) / p.fill * sgn for w, px in realized)
            risk = abs(p.ref - p.sl0_px) / p.ref
            ra[s_i] = raw / risk if risk > 0 else 0.0
            da[s_i] = j - i
    return dict(r_long=r_l, r_short=r_s, dur_long=d_l, dur_short=d_s,
                valid=np.isfinite(r_l) & np.isfinite(r_s), cap_hit=cap_hit)


# ══════════════════════════════════════════════════════════════════
# 2. 국면
# ══════════════════════════════════════════════════════════════════

def btc_regime_series() -> pd.Series:
    """BTC 30일 변화율 ±15% → 상승/하락/횡보. shift(1) 로 미래참조 차단."""
    d = load("BTCUSDT", "1h")
    chg = d["close"].pct_change(720).shift(1)
    return pd.Series(np.where(chg >= REGIME_THR, "상승",
                              np.where(chg <= -REGIME_THR, "하락", "횡보")),
                     index=d.index)


def map_regime(reg: pd.Series, idx: pd.DatetimeIndex) -> np.ndarray:
    return reg.reindex(idx, method="ffill").fillna("횡보").to_numpy()


# ══════════════════════════════════════════════════════════════════
# 3. 순열검정
# ══════════════════════════════════════════════════════════════════

def perm_global(obs_r: np.ndarray, obs_dir: np.ndarray, pool_l: np.ndarray,
                pool_s: np.ndarray, rng) -> dict:
    """무작위 진입 순열 — 같은 거래수·같은 롱/숏 구성, 진입 봉만 무작위."""
    nl = int((obs_dir > 0).sum())
    ns = int((obs_dir < 0).sum())
    n = nl + ns
    obs_mean = float(obs_r.mean())
    tot = np.zeros(N_PERM)
    if nl:
        tot += pool_l[rng.integers(0, len(pool_l), size=(N_PERM, nl))].sum(axis=1)
    if ns:
        tot += pool_s[rng.integers(0, len(pool_s), size=(N_PERM, ns))].sum(axis=1)
    null_mean = tot / n
    p = (1 + int((null_mean >= obs_mean).sum())) / (N_PERM + 1)
    return dict(n=n, n_long=nl, n_short=ns, obs_mean_r=obs_mean,
                obs_sum_r=float(obs_r.sum()), null_mean_r=float(null_mean.mean()),
                null_sd=float(null_mean.std()),
                delta_sum_r=float(obs_r.sum() - null_mean.mean() * n),
                null_p95=float(np.percentile(null_mean, 95)), p=p)


def perm_stratified(cells: dict, pools: dict, rng) -> dict:
    """국면×방향 **층화** 순열 — 무작위 진입도 같은 국면·같은 방향에서만 뽑는다.

    Args:
        cells: {(국면, 방향): 관측 R 배열}
        pools: {(국면, 방향): 그 층의 전체 봉 R 배열}
    Returns:
        전체 p + 셀별 표
    """
    n_tot = sum(len(v) for v in cells.values())
    obs_sum = sum(float(v.sum()) for v in cells.values())
    tot = np.zeros(N_PERM)
    per_cell = {}
    for key, obs in cells.items():
        pool = pools.get(key)
        k = len(obs)
        if pool is None or len(pool) == 0 or k == 0:
            per_cell[key] = dict(n=k, obs_mean=float(obs.mean()) if k else 0.0,
                                 base_mean=None, p=None, low_sample=k < 30,
                                 pool_n=0)
            continue
        draw = pool[rng.integers(0, len(pool), size=(N_PERM, k))]
        tot += draw.sum(axis=1)
        cm = draw.mean(axis=1)
        om = float(obs.mean())
        per_cell[key] = dict(
            n=k, obs_mean=om, base_mean=float(pool.mean()), pool_n=int(len(pool)),
            base_perm_mean=float(cm.mean()),
            delta=om - float(pool.mean()),
            p=(1 + int((cm >= om).sum())) / (N_PERM + 1),
            low_sample=bool(k < 30),
        )
    null_mean = tot / n_tot
    obs_mean = obs_sum / n_tot
    return dict(
        n=n_tot, obs_mean_r=obs_mean, null_mean_r=float(null_mean.mean()),
        delta_sum_r=float(obs_sum - null_mean.mean() * n_tot),
        p=(1 + int((null_mean >= obs_mean).sum())) / (N_PERM + 1),
        cells={f"{k[0]}|{'롱' if k[1] > 0 else '숏'}": v for k, v in per_cell.items()},
    )


# ══════════════════════════════════════════════════════════════════
# 4. 실행
# ══════════════════════════════════════════════════════════════════

def build(sym: str, params: dict, reg: pd.Series) -> dict:
    o = {**SIG_DEFAULTS, **BT_DEFAULTS, **params}
    df = load(sym, TF)
    sig = signals(df, **{k: o[k] for k in SIG_DEFAULTS})
    rm = rmap(df, o)
    return dict(sym=sym, df=df, sig=sig, rm=rm, reg=map_regime(reg, df.index),
                trades=backtest(df, **params), o=o)


def analyze(name: str, params: dict, syms, reg: pd.Series, rng) -> dict:
    print(f"\n{'=' * 72}\n[{name}] {params}  · 페어 {list(syms)}", flush=True)
    parts = [build(s, params, reg) for s in syms]

    # ---- 관측 거래 (실제 백테) ----
    obs_r, obs_dir, obs_reg, obs_year, obs_sym, obs_reason = [], [], [], [], [], []
    reverse_n = 0
    for b in parts:
        for t in b["trades"]:
            obs_r.append(t["r"])
            obs_dir.append(t["dir"])
            obs_reg.append(b["reg"][t["sig_idx"]])
            obs_year.append(t["ts"].year)
            obs_sym.append(b["sym"])
            obs_reason.append(t["reason"])
            reverse_n += int(t["outcome"] == "reverse")
    obs_r = np.array(obs_r, float)
    obs_dir = np.array(obs_dir, int)
    obs_reg = np.array(obs_reg)
    obs_year = np.array(obs_year)
    obs_sym = np.array(obs_sym)

    # ---- R-맵 기준 "신호봉 단독거래" (순열과 완전 동일 규칙) ----
    sa_r, sa_dir, sa_reg, sa_year, sa_sym = [], [], [], [], []
    for b in parts:
        v = b["rm"]["valid"]
        for arr, d in (("buy_sig", 1), ("sell_sig", -1)):
            idx = np.flatnonzero(b["sig"][arr].to_numpy() & v)
            src = b["rm"]["r_long"] if d > 0 else b["rm"]["r_short"]
            sa_r.append(src[idx])
            sa_dir.append(np.full(len(idx), d))
            sa_reg.append(b["reg"][idx])
            sa_year.append(b["df"].index[idx].year.to_numpy())
            sa_sym.append(np.full(len(idx), b["sym"]))
    sa_r = np.concatenate(sa_r)
    sa_dir = np.concatenate(sa_dir)
    sa_reg = np.concatenate(sa_reg)
    sa_year = np.concatenate(sa_year)
    sa_sym = np.concatenate(sa_sym)
    print(f"  실제백테 n={len(obs_r)} 건당 {obs_r.mean():+.4f}R ΣR {obs_r.sum():+.1f} "
          f"(reverse {reverse_n}건) | 단독거래(R-맵) n={len(sa_r)} "
          f"건당 {sa_r.mean():+.4f}R ΣR {sa_r.sum():+.1f}", flush=True)

    # ---- 풀 (무작위 진입 후보 봉) ----
    pool_l = np.concatenate([b["rm"]["r_long"][b["rm"]["valid"]] for b in parts])
    pool_s = np.concatenate([b["rm"]["r_short"][b["rm"]["valid"]] for b in parts])
    pool_reg = np.concatenate([b["reg"][b["rm"]["valid"]] for b in parts])
    cap_hit = sum(b["rm"]["cap_hit"] for b in parts)

    # ---- 1) 전역 순열검정 ----
    g_bt = perm_global(obs_r, obs_dir, pool_l, pool_s, rng)
    g_sa = perm_global(sa_r, sa_dir, pool_l, pool_s, rng)
    print(f"  [1] 순열검정 {N_PERM}회 (무작위 진입)")
    print(f"      실제백테 R : obs {g_bt['obs_mean_r']:+.4f} vs 기저 "
          f"{g_bt['null_mean_r']:+.4f} (sd {g_bt['null_sd']:.4f}) "
          f"ΔΣR {g_bt['delta_sum_r']:+.1f} → p={g_bt['p']:.4f}")
    print(f"      단독거래 R : obs {g_sa['obs_mean_r']:+.4f} vs 기저 "
          f"{g_sa['null_mean_r']:+.4f} ΔΣR {g_sa['delta_sum_r']:+.1f} "
          f"→ p={g_sa['p']:.4f}   ← 규칙 동일 비교(주 지표)")

    # ---- 2) 국면 × 방향 층화 ----
    cells, pools = {}, {}
    for rg in ("상승", "하락", "횡보"):
        for d in (1, -1):
            m = (sa_reg == rg) & (sa_dir == d)
            if m.sum():
                cells[(rg, d)] = sa_r[m]
            pm = pool_reg == rg
            pools[(rg, d)] = (pool_l if d > 0 else pool_s)[pm]
    st = perm_stratified(cells, pools, rng)
    print(f"  [2] 국면×방향 층화 순열 — 전체 obs {st['obs_mean_r']:+.4f} vs "
          f"층화기저 {st['null_mean_r']:+.4f} ΔΣR {st['delta_sum_r']:+.1f} "
          f"→ p={st['p']:.4f}")
    print(f"      {'셀':<10}{'n':>5}{'신호R':>10}{'기저R':>10}{'Δ':>9}{'p':>9}  풀n")
    for k, v in st["cells"].items():
        if v["base_mean"] is None:
            print(f"      {k:<10}{v['n']:>5}{'-':>10}")
            continue
        flag = " ※표본<30" if v["low_sample"] else ""
        print(f"      {k:<10}{v['n']:>5}{v['obs_mean']:>+10.4f}"
              f"{v['base_mean']:>+10.4f}{v['delta']:>+9.4f}{v['p']:>9.4f}"
              f"  {v['pool_n']}{flag}")

    # ---- 3) 연도 일관성 ----
    years = {}
    for y in sorted(set(sa_year.tolist())):
        m = sa_year == y
        pm_all = np.ones(len(pool_l), bool)  # 기저는 전기간(연도별 풀 분리는 아래)
        years[str(y)] = dict(n=int(m.sum()), sum_r=float(sa_r[m].sum()),
                             mean_r=float(sa_r[m].mean()),
                             win=float((sa_r[m] > 0).mean()),
                             low_sample=bool(m.sum() < 30))
        del pm_all
    # 연도별 기저(같은 해 봉만)
    yr_pool = np.concatenate([b["df"].index.year.to_numpy()[b["rm"]["valid"]]
                              for b in parts])
    print("  [3] 연도 일관성 (단독거래 기준)")
    print(f"      {'연도':<7}{'n':>5}{'건당R':>10}{'기저R':>10}{'ΣR':>9}{'승률':>8}")
    for y, v in years.items():
        ym = yr_pool == int(y)
        d_pos = (sa_dir[sa_year == int(y)] > 0)
        base = np.concatenate([pool_l[ym]] * 1)
        base_mix = (pool_l[ym].mean() * d_pos.mean()
                    + pool_s[ym].mean() * (1 - d_pos.mean())) if ym.sum() else np.nan
        v["base_mean_r"] = float(base_mix)
        del base
        print(f"      {y:<7}{v['n']:>5}{v['mean_r']:>+10.4f}{base_mix:>+10.4f}"
              f"{v['sum_r']:>+9.1f}{v['win'] * 100:>7.1f}%"
              f"{'  ※표본<30' if v['low_sample'] else ''}")
    ypos = sum(1 for v in years.values() if v["sum_r"] > 0)

    # ---- 롱/숏 분리 · 페어 분리 · 신호원 분해 ----
    def split(labels, values):
        out = {}
        for lab in sorted(set(labels.tolist())):
            m = labels == lab
            out[str(lab)] = dict(n=int(m.sum()), mean_r=float(values[m].mean()),
                                 sum_r=float(values[m].sum()),
                                 win=float((values[m] > 0).mean()),
                                 low_sample=bool(m.sum() < 30))
        return out

    ls = split(np.where(sa_dir > 0, "롱", "숏"), sa_r)
    ps = split(sa_sym, sa_r)
    print("  [보조] 롱/숏 · 페어 분리 (단독거래)")
    for k, v in {**ls, **ps}.items():
        print(f"      {k:<10} n={v['n']:>4} 건당 {v['mean_r']:+.4f}R "
              f"ΣR {v['sum_r']:+.1f} 승률 {v['win'] * 100:.1f}%"
              f"{'  ※표본<30' if v['low_sample'] else ''}")

    return dict(
        name=name, params=params, syms=list(syms),
        n_bt=len(obs_r), reverse_n=reverse_n, cap_hit=cap_hit,
        bt_mean_r=float(obs_r.mean()), bt_sum_r=float(obs_r.sum()),
        sa_n=int(len(sa_r)), sa_mean_r=float(sa_r.mean()), sa_sum_r=float(sa_r.sum()),
        perm_bt=g_bt, perm_sa=g_sa, strat=st, years=years, years_positive=ypos,
        long_short=ls, per_sym=ps,
        pool_n=int(len(pool_l)),
        pool_long_mean=float(pool_l.mean()), pool_short_mean=float(pool_s.mean()),
    )


def decompose(reg: pd.Series, rng) -> dict:
    """[5] 다이버전스 vs 모멘텀 크로스 기여 분해.

    'Both | HTF ON | Fixed%' 의 거래를 신호원(reason)별로 쪼갠다 +
    div/mom 단독 변형의 단독거래 R 을 나란히.
    """
    print(f"\n{'=' * 72}\n[5] 신호원 분해 — 다이버전스가 진짜 기여하는가", flush=True)
    out = {}
    both = dict(entry_mode="Both", use_htf_filter=True, tpsl_mode="Fixed Percent")
    rows = []
    for sym in MAIN:
        df = load(sym, TF)
        for t in backtest(df, **both):
            rows.append((t["reason"], t["r"]))
    by = {}
    for rs, r in rows:
        by.setdefault(rs, []).append(r)
    print("  Both|HTF ON|Fixed% 의 거래를 진입근거로 분해 (실제 백테)")
    for k, v in sorted(by.items()):
        a = np.array(v)
        print(f"      {k:<22} n={len(a):>4} 건당 {a.mean():+.4f}R ΣR {a.sum():+7.1f} "
              f"승률 {(a > 0).mean() * 100:.1f}%{'  ※표본<30' if len(a) < 30 else ''}")
        out[k] = dict(n=len(a), mean_r=float(a.mean()), sum_r=float(a.sum()),
                      win=float((a > 0).mean()), low_sample=bool(len(a) < 30))

    # 단독 변형 대조 (R-맵 기준 · 같은 청산엔진)
    print("  진입모드 단독 대조 (단독거래 R · HTF ON · Fixed%)")
    cmp_ = {}
    for em in ("Divergence Only", "Momentum Cross", "Both"):
        p = dict(entry_mode=em, use_htf_filter=True, tpsl_mode="Fixed Percent")
        rr, dd = [], []
        for sym in MAIN:
            o = {**SIG_DEFAULTS, **BT_DEFAULTS, **p}
            df = load(sym, TF)
            s = signals(df, **{k: o[k] for k in SIG_DEFAULTS})
            rm = rmap(df, o)
            v = rm["valid"]
            for arr, d in (("buy_sig", 1), ("sell_sig", -1)):
                idx = np.flatnonzero(s[arr].to_numpy() & v)
                rr.append((rm["r_long"] if d > 0 else rm["r_short"])[idx])
                dd.append(np.full(len(idx), d))
        rr = np.concatenate(rr)
        print(f"      {em:<22} n={len(rr):>4} 건당 {rr.mean():+.4f}R ΣR {rr.sum():+7.1f}")
        cmp_[em] = dict(n=int(len(rr)), mean_r=float(rr.mean()), sum_r=float(rr.sum()))
    return dict(by_reason=out, by_mode=cmp_)


def holdout(params: dict, reg: pd.Series, rng) -> dict:
    """[4] 홀드아웃 5페어 — 파라미터 완전 고정."""
    print(f"\n{'=' * 72}\n[4] 홀드아웃 5페어 (파라미터 고정) {params}", flush=True)
    res = {}
    for sym in HOLDOUT:
        try:
            r = analyze(f"홀드아웃 {sym}", params, (sym,), reg, rng)
        except Exception as e:                    # 데이터 부족 등
            print(f"  {sym}: 실패 {e}")
            continue
        res[sym] = r
    # 5페어 통합
    print("\n  ── 홀드아웃 5페어 통합 ──")
    pooled = analyze("홀드아웃 통합(5페어)", params, HOLDOUT, reg, rng)
    return dict(per_pair=res, pooled=pooled)


def main() -> int:
    rng = np.random.default_rng(SEED)
    reg = btc_regime_series()
    print(f"국면 정의: BTC 30일 변화율 ±{REGIME_THR * 100:.0f}% (shift1) — "
          f"{reg.value_counts().to_dict()}", flush=True)
    print(f"시도 설정 수 {N_SETTINGS} (3TF × 12) → Bonferroni α = {ALPHA_B:.5f}",
          flush=True)

    out: dict = dict(tf=TF, n_perm=N_PERM, seed=SEED, cap=CAP,
                     n_settings=N_SETTINGS, alpha_bonferroni=ALPHA_B,
                     regime_thr=REGIME_THR, candidates={})
    for nm, p in CAND.items():
        out["candidates"][nm] = analyze(nm, p, MAIN, reg, rng)

    out["decompose"] = decompose(reg, rng)
    out["holdout"] = holdout(CAND["A: Div|HTF ON|Fixed%"], reg, rng)

    os.makedirs(os.path.join(RES, "data/flow"), exist_ok=True)
    path = os.path.join(RES, "data/flow/battery.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n저장: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
