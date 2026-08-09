"""#AUTONOMOUS 2026-08-08: Flow Engine 4h 백테 — 우리 표준 평가.

BTC+ETH 4h · 진입모드 3종 × HTF필터 on/off × TP/SL 2종 = 12 설정.
평가는 origo_leverage_verify.sim 규약(복리·size 0.9·동시보유 분할·DD 스로틀
-25%→×0.7·파산 20%)을 따르되, Flow Engine 은 독립 백테(flow_engine.backtest)다.

레버리지 7x 통일(우리 봇 비교) + Pine 기본 10x 병기.
비용 왕복 0.08%(우리) / 0.12%(Pine 0.06%×2) 둘 다.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_engine import backtest, load  # noqa: E402

RES = "C:/Users/지영민/Desktop/Aurora-ICT-research"
SYMS = ("BTCUSDT", "ETHUSDT")
TF = "4h"
SIZE = 0.9
RUIN = 0.20
DD_PCT, DD_FACTOR = 0.25, 0.7
N_BOOT = 2000
RNG = np.random.default_rng(20260808)
LEVS = (7, 10)
FEES = (0.0008, 0.0012)   # 왕복


def concurrency(spans: list[tuple[int, int]]) -> np.ndarray:
    """각 거래가 열려 있는 동안의 평균 동시 보유 수(자기 포함)."""
    n = len(spans)
    out = np.ones(n)
    s = np.array([a for a, _ in spans], dtype=np.int64)
    e = np.array([b for _, b in spans], dtype=np.int64)
    for i in range(n):
        ov = (s <= e[i]) & (e >= s[i])
        out[i] = ov.sum()          # 자기 포함
    return out


def sim_paths(raw: np.ndarray, scale: np.ndarray, lev: float, fee: float,
              order: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """복리 시뮬 — order 는 (B, n) 거래 인덱스 행렬. (최종자산, MDD, 파산) 벡터."""
    b, n = order.shape
    eq = np.ones(b)
    peak = np.ones(b)
    mdd = np.zeros(b)
    dead = np.zeros(b, dtype=bool)
    r_ = raw[order]
    sc_ = scale[order]
    for k in range(n):
        sz = SIZE * sc_[:, k]
        sz = np.where(eq < peak * (1.0 - DD_PCT), sz * DD_FACTOR, sz)
        step = r_[:, k] * sz * lev - fee * sz * lev
        new = eq * (1.0 + step)
        new = np.where(dead, eq, new)
        eq = np.maximum(new, 0.0)
        peak = np.maximum(peak, eq)
        mdd = np.maximum(mdd, 1.0 - eq / np.maximum(peak, 1e-12))
        newly = (~dead) & (eq <= RUIN)
        dead |= newly
    return eq, mdd * 100.0, dead


def evaluate(trades_all: list[dict]) -> dict:
    """거래 리스트(페어 통합, 시간순) → 우리 표준 지표."""
    trades_all = sorted(trades_all, key=lambda t: t["_s"])
    raw = np.array([t["raw"] for t in trades_all], float)
    r = np.array([t["r"] for t in trades_all], float)
    spans = [(t["_s"], t["_e"]) for t in trades_all]
    scale = 1.0 / concurrency(spans)
    n = len(raw)
    base = np.arange(n)[None, :]

    out: dict = {}
    for lev in LEVS:
        for fee in FEES:
            key = f"{lev}x_{fee*100:.2f}%"
            eq0, mdd0, dead0 = sim_paths(raw, scale, lev, fee, base)
            boot = RNG.integers(0, n, size=(N_BOOT, n))
            eqb, _, deadb = sim_paths(raw, scale, lev, fee, boot)
            out[key] = dict(
                compound=float(eq0[0]),
                mdd=float(mdd0[0]),
                ruin_hist=bool(dead0[0]),
                ruin=float(deadb.mean()),
                p5=float(np.percentile(eqb, 5)),
                p50=float(np.percentile(eqb, 50)),
            )
    out["_conc_mean"] = float(concurrency(spans).mean())
    out["_size_eff"] = float(SIZE * scale.mean())
    return out


def run(name: str, params: dict) -> dict:
    pool: list[dict] = []
    per_sym = {}
    months = 0.0
    for si, sym in enumerate(SYMS):
        df = load(sym, TF)
        months = max(months, (df.index[-1] - df.index[0]).days / 30.44)
        tr = backtest(df, **params)
        rr = np.array([t["r"] for t in tr], float) if tr else np.zeros(0)
        per_sym[sym] = dict(
            n=len(tr),
            r_mean=float(rr.mean()) if len(rr) else 0.0,
            r_sum=float(rr.sum()) if len(rr) else 0.0,
            win=float((rr > 0).mean()) if len(rr) else 0.0,
        )
        idx = df.index
        for t in tr:
            t["_s"] = int(idx[t["entry_idx"]].value)
            t["_e"] = int(idx[min(t["exit_idx"], len(idx) - 1)].value)
            t["_sym"] = sym
            pool.append(t)

    if not pool:
        return dict(name=name, params=params, n=0, note="거래 0건")

    r = np.array([t["r"] for t in pool], float)
    ev = evaluate(pool)
    # 연도 일관성용 원자료 (해석은 다음 단계)
    by_year: dict[str, list[float]] = {}
    for t in pool:
        by_year.setdefault(str(t["ts"].year), []).append(t["r"])
    years = {y: dict(n=len(v), r_sum=float(np.sum(v)), r_mean=float(np.mean(v)),
                     low_sample=bool(len(v) < 30))
             for y, v in sorted(by_year.items())}
    oc = {}
    for t in pool:
        oc[t["outcome"]] = oc.get(t["outcome"], 0) + 1

    res = dict(
        name=name,
        params=params,
        n=len(r),
        monthly=len(r) / months,               # BTC+ETH 합산
        monthly_per_pair=len(r) / months / len(SYMS),
        r_mean=float(r.mean()),
        r_med=float(np.median(r)),
        r_sum=float(r.sum()),
        win=float((r > 0).mean()),
        low_sample=bool(len(r) < 30),
        per_sym=per_sym,
        outcomes=oc,
        years=years,
        # 대표값 = 7x / 우리 0.08%
        compound=ev["7x_0.08%"]["compound"],
        mdd=ev["7x_0.08%"]["mdd"],
        ruin=ev["7x_0.08%"]["ruin"],
        p5=ev["7x_0.08%"]["p5"],
        lev_cost=ev,
    )
    print(f"[{name}] n={len(r)} 월{res['monthly']:.2f} R{r.mean():+.4f} "
          f"win{res['win']*100:.1f}% ΣR{r.sum():+.1f} | "
          f"7x/0.08: {res['compound']:.3f}x MDD{res['mdd']:.1f}% "
          f"ruin{res['ruin']*100:.1f}% p5 {res['p5']:.3f}x", flush=True)
    return res


def main() -> int:
    variants = []
    for em in ("Divergence Only", "Momentum Cross", "Both"):
        for htf in (True, False):
            for mode in ("ATR Dynamic", "Fixed Percent"):
                nm = (f"{em} | HTF {'ON' if htf else 'OFF'} | "
                      f"{'ATR' if mode == 'ATR Dynamic' else 'Fixed%'}")
                variants.append((nm, dict(entry_mode=em, use_htf_filter=htf,
                                          tpsl_mode=mode)))
    print(f"설정 {len(variants)}개 · 페어 {SYMS} · TF {TF}", flush=True)
    rows = [run(nm, p) for nm, p in variants]

    os.makedirs(os.path.join(RES, "data/flow"), exist_ok=True)
    out = dict(tf=TF, pairs=list(SYMS), n_settings=len(variants),
               size_pct=SIZE, dd_throttle=[DD_PCT, DD_FACTOR], ruin_level=RUIN,
               n_boot=N_BOOT, variants=rows)
    p = os.path.join(RES, "data/flow/4h.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1, default=float)
    print(f"\n저장: {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
