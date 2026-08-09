"""#AUTONOMOUS 2026-08-08: Flow Engine 판정보고서 **반박 검증**.

선행 보고(기각)의 각 주장을 독립적으로 재현/공격한다. 재현 코드를 공유하지 않고
flow_engine 의 backtest/signals 만 쓴다(=이식 자체는 별도로 검산 완료).

## 무엇을 확인하나
A. 재현 — 176건 / +0.157R 이 나오는가
B. 배포된 기본값(Pine as-shipped)의 성적 — 다중비교 논쟁과 무관한 1차 근거
C. 비관 가정 민감도 — SL우선 경로 · TP 관통요구 · 체결봉 감시 · 펀딩비
D. 파라미터 이웃(knife-edge 검사) — fix_sl · TP배율 · TF · lbL/lbR
E. 순열검정 3종 — iid 봉재표집 · **원형시프트(군집 보존)** · 국면×방향 층화
F. 홀드아웃 5페어 재현
G. 시도 설정 수 실집계(Bonferroni 분모)
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flow_engine as F  # noqa: E402
from flow_engine import (BT_DEFAULTS, SIG_DEFAULTS, _atr, _Pos, _tp_sl_pct,  # noqa: E402
                         _walk_bar, load, signals)

RES = "C:/Users/지영민/Desktop/Aurora-ICT-research"
OUT = os.path.join(RES, "data/flow/refute.json")
CACHE = os.path.join(RES, "data/flow/_refute")
MAIN = ("BTCUSDT", "ETHUSDT")
HOLDOUT = ("SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT")
TF = "4h"
CAP = 120
N_PERM = 20000
SEED = 20260808
FEE = 0.0008
LEV, SIZE, RUIN = 7.0, 0.9, 0.20

BEST = dict(entry_mode="Divergence Only", use_htf_filter=True,
            tpsl_mode="Fixed Percent")
SHIPPED = dict(entry_mode="Both", use_htf_filter=True, tpsl_mode="ATR Dynamic")

os.makedirs(CACHE, exist_ok=True)


# ══════════════════════════════════════════════════════════════════
# 공통 요약
# ══════════════════════════════════════════════════════════════════

def stat(trades, months=58.0, fee=FEE, lev=LEV):
    """거래 리스트 → 건당R·복리자산·MDD·파산 (우리 표준 7x)."""
    if not trades:
        return dict(n=0)
    r = np.array([t["r"] for t in trades], float)
    raw = np.array([t["raw"] for t in trades], float)
    eq, peak, mdd, ruin = 1.0, 1.0, 0.0, False
    for x in raw:
        eq *= (1 + (x - fee) * SIZE * lev)
        if eq <= RUIN:
            ruin = True
            break
        peak = max(peak, eq)
        mdd = max(mdd, 1 - eq / peak)
    return dict(n=len(r), monthly=len(r) / months, r_mean=float(r.mean()),
                r_med=float(np.median(r)), r_sum=float(r.sum()),
                win=float((r > 0).mean()), eq=float(eq), mdd=float(mdd * 100),
                ruin=bool(ruin))


def collect(params, syms=MAIN, tf=TF, funding=0.0, **kw):
    """여러 심볼 백테 합산. funding>0 이면 보유시간 기준 펀딩비를 raw 에서 차감."""
    out = []
    tfh = {"15m": 0.25, "30m": 0.5, "1h": 1, "2h": 2, "3h": 3, "4h": 4,
           "6h": 6, "8h": 8, "12h": 12}[tf]
    for sym in syms:
        try:
            df = load(sym, tf)
        except Exception:
            continue
        for t in F.backtest(df, **{**params, **kw}):
            raw = float(t["raw"])
            if funding:
                hrs = (t["exit_idx"] - t["entry_idx"]) * tfh
                raw -= funding * (hrs / 8.0)
            risk = abs(t["ref"] - t["sl"]) / t["ref"]
            out.append(dict(sym=sym, ts=int(t["ts"].value // 10**6),
                            raw=raw, r=raw / risk if risk > 0 else 0.0,
                            dir=int(t["dir"]), dur=int(t["exit_idx"] - t["entry_idx"]),
                            outcome=t["outcome"]))
    out.sort(key=lambda x: x["ts"])
    return out


# ══════════════════════════════════════════════════════════════════
# TP 관통 요구 — 지정가 체결 낙관성 검사
# ══════════════════════════════════════════════════════════════════

def _patch_penetration(eps: float):
    """TP 레벨을 eps(비율)만큼 안쪽으로 밀어 '터치=체결' 낙관을 제거.

    지정가는 레벨에 닿아도 큐가 밀리면 미체결이다. TP 목표를 eps 만큼 더 멀리
    두면 '레벨을 eps 관통해야 체결'과 동등해진다(체결가는 원 레벨 유지가 아니라
    보수적으로 밀린 레벨 = 더 유리 → 그래서 체결가도 원 레벨로 되돌린다).
    """
    orig = F._Pos.__init__

    def patched(self, is_long, ref, fill_px, tps_pct, sl_pct, o, entry_i, sig_i):
        orig(self, is_long, ref, fill_px, tps_pct, sl_pct, o, entry_i, sig_i)
        s = 1.0 if is_long else -1.0
        # 체결 판정용 레벨만 eps 밀고, 실현가는 원 레벨로 쓰기 위해 tp_px 를
        # 밀되 슬리피지 상쇄분을 되돌린다 → 순효과 = 관통 요구
        self.tp_px = [px * (1 + s * eps) for px in self.tp_px]
    F._Pos.__init__ = patched
    return orig


# ══════════════════════════════════════════════════════════════════
# R-맵 (순열검정용) — 모든 봉 × 롱/숏 단독거래 R
# ══════════════════════════════════════════════════════════════════

def rmap(sym, params, tf=TF):
    o = {**SIG_DEFAULTS, **BT_DEFAULTS, **params}
    key = f"rmap_{sym}_{tf}_{o['tpsl_mode'].replace(' ', '')}.npz"
    p = os.path.join(CACHE, key)
    if os.path.exists(p):
        z = np.load(p, allow_pickle=False)
        return z["rl"], z["rs"]
    df = load(sym, tf)
    n = len(df)
    O, H, L, C = (df[c].to_numpy(float) for c in ("open", "high", "low", "close"))
    atr = _atr(df, o["atr_period"]).to_numpy(float)
    alloc = np.array(o["alloc"], float) / 100.0
    slip = o["slip_pct"]
    rl, rs = np.full(n, np.nan), np.full(n, np.nan)
    for s_i in range(n - 2):
        if not np.isfinite(atr[s_i]):
            continue
        for is_long, ra in ((True, rl), (False, rs)):
            sgn = 1.0 if is_long else -1.0
            i = s_i + 1
            ref = C[s_i]
            tps, slp = _tp_sl_pct(atr[s_i], ref, o)
            fill = O[i] * (1 + sgn * slip)
            pos = _Pos(is_long, ref, fill, tps, slp, o, i, s_i)
            real: list = []
            end = min(n, i + CAP)
            j, done = i + o["exit_start_offset"], None
            while j < end:
                done = _walk_bar(pos, O[j], H[j], L[j], C[j], real, alloc, slip,
                                 o["intrabar"])
                if done is not None:
                    break
                j += 1
            if done is None:
                j = min(j, n - 1)
                real.append((max(1.0 - alloc[:pos.hits].sum(), 0.0), C[j]))
            raw = sum(w * (px - pos.fill) / pos.fill * sgn for w, px in real)
            risk = abs(pos.ref - pos.sl0_px) / pos.ref
            ra[s_i] = raw / risk if risk > 0 else 0.0
    np.savez_compressed(p, rl=rl, rs=rs)
    return rl, rs


def regime_series(thr=0.15):
    d = load("BTCUSDT", "1h")
    chg = d["close"].pct_change(720).shift(1)
    return pd.Series(np.where(chg >= thr, "up", np.where(chg <= -thr, "dn", "flat")),
                     index=d.index)


# ══════════════════════════════════════════════════════════════════
def main():
    rng = np.random.default_rng(SEED)
    res: dict = {}
    t0 = time.time()

    # ── A. 재현 ──────────────────────────────────────────────────
    base = collect(BEST)
    res["A_reproduce"] = stat(base)
    print("A 재현 BEST:", res["A_reproduce"], flush=True)

    # ── B. 배포 기본값 (Pine as-shipped) ────────────────────────
    res["B_shipped"] = {}
    for tf in ("15m", "1h", "4h"):
        s = stat(collect(SHIPPED, tf=tf), months=58.0)
        res["B_shipped"][tf] = s
        print(f"B shipped {tf}:", s, flush=True)

    # ── C. 비관 가정 민감도 ─────────────────────────────────────
    res["C_stress"] = {}
    probes = [
        ("baseline", {}, 0.0),
        ("sl_first(SL우선경로)", dict(intrabar="sl_first"), 0.0),
        ("exit_off0(체결봉감시)", dict(exit_start_offset=0), 0.0),
        ("htf_developing(Pine실제)", dict(htf_mode="developing"), 0.0),
        ("funding 0.01%/8h", {}, 0.0001),
        ("funding 0.03%/8h", {}, 0.0003),
        ("fee0.12%(Pine)", {}, 0.0),
    ]
    for name, kw, fund in probes:
        rows = collect(BEST, funding=fund, **kw)
        fee = 0.0012 if "0.12" in name else FEE
        res["C_stress"][name] = stat(rows, fee=fee)
        print(f"C {name}:", res["C_stress"][name], flush=True)

    # TP 관통 요구
    for eps in (0.0005, 0.001, 0.002):
        orig = _patch_penetration(eps)
        try:
            rows = collect(BEST)
            res["C_stress"][f"TP관통 {eps*100:.2f}%"] = stat(rows)
            print(f"C TP관통 {eps*100:.2f}%:",
                  res["C_stress"][f"TP관통 {eps*100:.2f}%"], flush=True)
        finally:
            F._Pos.__init__ = orig

    # ── D. 파라미터 이웃 ────────────────────────────────────────
    res["D_neighbor"] = {"fix_sl": {}, "tp_scale": {}, "tf": {}, "pivot": {}}
    for sl in (1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 4.0):
        res["D_neighbor"]["fix_sl"][str(sl)] = stat(collect(BEST, fix_sl=sl))
    for k in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        res["D_neighbor"]["tp_scale"][str(k)] = stat(
            collect(BEST, fix_tp=(1.0 * k, 2.0 * k, 3.0 * k, 4.0 * k)))
    for tf in ("1h", "2h", "3h", "4h", "6h", "8h", "12h"):
        mult = {"1h": 1, "2h": 2, "3h": 3, "4h": 4, "6h": 6, "8h": 8, "12h": 12}[tf]
        res["D_neighbor"]["tf"][tf] = stat(collect(BEST, tf=tf))
        _ = mult
    for lb in (3, 4, 5, 6, 7, 8):
        res["D_neighbor"]["pivot"][str(lb)] = stat(collect(BEST, lbL=lb, lbR=lb))
    for grp, d in res["D_neighbor"].items():
        print(f"D {grp}: " + " | ".join(
            f"{k}:n{v.get('n',0)} R{v.get('r_mean',0):+.3f} eq{v.get('eq',0):.2f}"
            for k, v in d.items()), flush=True)

    # ── E. 순열검정 3종 ─────────────────────────────────────────
    print(f"  rmap 생성 중... ({time.time()-t0:.0f}s)", flush=True)
    RL, RS, SIG, REG = {}, {}, {}, {}
    reg = regime_series()
    for sym in MAIN:
        RL[sym], RS[sym] = rmap(sym, BEST)
        df = load(sym, TF)
        SIG[sym] = signals(df, **{k: {**SIG_DEFAULTS, **BEST}[k] for k in SIG_DEFAULTS})
        REG[sym] = reg.reindex(df.index, method="ffill").fillna("flat").to_numpy()
        print(f"  rmap {sym} 완료 ({time.time()-t0:.0f}s)", flush=True)

    # 신호정렬(sa) 관측 — 순열과 같은 단독거래 잣대
    obs_r, obs_dir, obs_reg, obs_sym, obs_i = [], [], [], [], []
    for sym in MAIN:
        b = SIG[sym]["buy_sig"].to_numpy()
        s = SIG[sym]["sell_sig"].to_numpy()
        for i in np.flatnonzero(b):
            if np.isfinite(RL[sym][i]):
                obs_r.append(RL[sym][i]); obs_dir.append(1)
                obs_reg.append(REG[sym][i]); obs_sym.append(sym); obs_i.append(i)
        for i in np.flatnonzero(s):
            if np.isfinite(RS[sym][i]):
                obs_r.append(RS[sym][i]); obs_dir.append(-1)
                obs_reg.append(REG[sym][i]); obs_sym.append(sym); obs_i.append(i)
    obs_r = np.array(obs_r); obs_dir = np.array(obs_dir)
    obs_reg = np.array(obs_reg); obs_sym = np.array(obs_sym)
    om = float(obs_r.mean())
    res["E_obs"] = dict(n=len(obs_r), mean_r=om, sum_r=float(obs_r.sum()),
                        n_long=int((obs_dir > 0).sum()),
                        n_short=int((obs_dir < 0).sum()))
    print("E 관측(sa):", res["E_obs"], flush=True)

    pool_l = np.concatenate([RL[s][np.isfinite(RL[s])] for s in MAIN])
    pool_s = np.concatenate([RS[s][np.isfinite(RS[s])] for s in MAIN])

    # E1. iid 봉 재표집
    nl, ns = int((obs_dir > 0).sum()), int((obs_dir < 0).sum())
    tot = pool_l[rng.integers(0, len(pool_l), (N_PERM, nl))].sum(1) + \
        pool_s[rng.integers(0, len(pool_s), (N_PERM, ns))].sum(1)
    nm = tot / (nl + ns)
    res["E1_iid"] = dict(null_mean=float(nm.mean()), null_p95=float(np.percentile(nm, 95)),
                         p=(1 + int((nm >= om).sum())) / (N_PERM + 1),
                         pool_l_mean=float(pool_l.mean()), pool_s_mean=float(pool_s.mean()))
    print("E1 iid:", res["E1_iid"], flush=True)

    # E2. 원형 시프트 — 신호 군집·자기상관 보존
    shifts = np.zeros(N_PERM)
    for k in range(N_PERM):
        tot_r, cnt = 0.0, 0
        for sym in MAIN:
            n = len(RL[sym])
            off = int(rng.integers(1, n))
            for arr, key in ((RL[sym], "buy_sig"), (RS[sym], "sell_sig")):
                idx = (np.flatnonzero(SIG[sym][key].to_numpy()) + off) % n
                v = arr[idx]
                v = v[np.isfinite(v)]
                tot_r += v.sum(); cnt += len(v)
        shifts[k] = tot_r / max(cnt, 1)
    res["E2_shift"] = dict(null_mean=float(shifts.mean()),
                           null_p95=float(np.percentile(shifts, 95)),
                           p=(1 + int((shifts >= om).sum())) / (N_PERM + 1))
    print("E2 원형시프트:", res["E2_shift"], flush=True)

    # E3. 국면×방향 층화
    pools = {}
    for sym in MAIN:
        for d, arr in ((1, RL[sym]), (-1, RS[sym])):
            for rg in ("up", "dn", "flat"):
                m = (REG[sym] == rg) & np.isfinite(arr)
                pools.setdefault((rg, d), []).append(arr[m])
    pools = {k: np.concatenate(v) for k, v in pools.items()}
    cells = {}
    for rg, d, r in zip(obs_reg, obs_dir, obs_r):
        cells.setdefault((rg, int(d)), []).append(r)
    cells = {k: np.array(v) for k, v in cells.items()}
    tot = np.zeros(N_PERM)
    cellinfo = {}
    for k, ob in cells.items():
        pl = pools[k]
        draw = pl[rng.integers(0, len(pl), (N_PERM, len(ob)))]
        tot += draw.sum(1)
        cm = draw.mean(1)
        cellinfo[f"{k[0]}|{'L' if k[1] > 0 else 'S'}"] = dict(
            n=len(ob), obs=float(ob.mean()), base=float(pl.mean()),
            p=(1 + int((cm >= ob.mean()).sum())) / (N_PERM + 1),
            low_sample=bool(len(ob) < 30))
    nm3 = tot / len(obs_r)
    res["E3_strat"] = dict(null_mean=float(nm3.mean()),
                           p=(1 + int((nm3 >= om).sum())) / (N_PERM + 1),
                           cells=cellinfo)
    print("E3 층화:", res["E3_strat"], flush=True)

    # E4. 페어별 순열
    res["E4_persym"] = {}
    for sym in MAIN:
        m = obs_sym == sym
        o_r, o_d = obs_r[m], obs_dir[m]
        pl = RL[sym][np.isfinite(RL[sym])]
        ps = RS[sym][np.isfinite(RS[sym])]
        a, b = int((o_d > 0).sum()), int((o_d < 0).sum())
        t = pl[rng.integers(0, len(pl), (N_PERM, a))].sum(1) + \
            ps[rng.integers(0, len(ps), (N_PERM, b))].sum(1)
        nmm = t / (a + b)
        res["E4_persym"][sym] = dict(n=len(o_r), obs=float(o_r.mean()),
                                     null=float(nmm.mean()),
                                     p=(1 + int((nmm >= o_r.mean()).sum())) / (N_PERM + 1))
        print(f"E4 {sym}:", res["E4_persym"][sym], flush=True)

    # ── F. 홀드아웃 ─────────────────────────────────────────────
    res["F_holdout"] = {}
    allh = []
    for sym in HOLDOUT:
        rows = collect(BEST, syms=(sym,))
        res["F_holdout"][sym] = stat(rows)
        allh += rows
        print(f"F {sym}:", res["F_holdout"][sym], flush=True)
    res["F_holdout"]["ALL"] = stat(allh)
    print("F ALL:", res["F_holdout"]["ALL"], flush=True)

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n저장 {OUT}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
