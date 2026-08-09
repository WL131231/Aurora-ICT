"""#AUTONOMOUS 2026-08-08: Flow Engine 15분봉 백테스트 (BTC+ETH 포트폴리오).

담당: 자율연구. 판정은 다음 단계 — 여기서는 **숫자만** 산출한다.

평가 규약(우리 표준, scripts/origo_leverage_verify.py 의 sim 을 따름):
  · 복리 · size_pct 0.9 · 레버리지 7x 통일(Pine 기본 10x 병기)
  · 동시보유 분할(겹치는 구간은 자본을 나눠 씀) · DD 스로틀(-25% → ×0.7)
  · 파산 = 시드 20% 이하 · 비용은 왕복 taker 0.08%(우리) / 0.12%(Pine) 둘 다
  · 부트스트랩 = 거래 복원추출 2000회 → 5%분위 · 파산확률
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flow_engine import backtest, load  # noqa: E402

RES = "C:/Users/지영민/Desktop/Aurora-ICT-research"
SYMS = ("BTCUSDT", "ETHUSDT")
TF = "15m"

RNG = np.random.default_rng(20260808)
N_BOOT = 2000
SIZE = 0.9          # live size_pct
RUIN = 0.20         # 시드 20% 이하 = 파산
DD_PCT, DD_FACTOR = 0.25, 0.7
FEES = {"0.08": 0.0008, "0.12": 0.0012}   # 왕복
LEVS = (1, 7, 10)   # 1x 는 참고용(파산이 레버리지 탓인지 원 엣지 탓인지 분리)


def concurrency(spans: np.ndarray) -> np.ndarray:
    """각 거래 구간이 겹치는 **평균 동시보유 수**(자기 포함). spans=(n,2) ns."""
    s, e = spans[:, 0], spans[:, 1]
    out = np.empty(len(spans))
    for i in range(len(spans)):
        out[i] = 1.0 + int(((s <= e[i]) & (e >= s[i])).sum()) - 1
    return out


def sim(raws, scale, lev, fee):
    """복리 시뮬 → (최종배수, MDD%, 파산여부, 파산거래번호).

    파산거래번호는 -1 이면 끝까지 생존. 12변형이 전부 파산하면 '자산'값이 전부
    파산 문턱(0.2 근처)에 붙어 변별력이 사라지므로, **언제** 파산했는지를 함께
    남긴다.
    """
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i, r in enumerate(raws):
        sz = SIZE * scale[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        step = r * sz * lev - fee * sz * lev
        eq *= (1.0 + step)
        if eq <= 0:
            return 0.0, 100.0, True, i
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return float(eq), 100.0 * mdd, True, i
    return float(eq), 100.0 * mdd, False, -1


def sim_nocap(raws, scale, lev, fee):
    """파산 컷 없이 끝까지 굴린 최종배수(참고치)."""
    eq = 1.0
    peak = 1.0
    for i, r in enumerate(raws):
        sz = SIZE * scale[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        step = r * sz * lev - fee * sz * lev
        eq *= max(1.0 + step, 1e-12)
        peak = max(peak, eq)
    return float(eq)


def boot(raws, scale, lev, fee, chunk: int = 500):
    """거래 복원추출 2000회 → (5%분위, 중앙, 파산확률%).

    복제본 축을 **벡터화**한다(경로 의존인 DD 스로틀·파산 판정은 시간축으로
    한 스텝씩 전진). 거래 1.1만건 × 2000복제를 파이썬 이중루프로 돌면 2e7 스텝이라
    실행 불가 — 스텝당 numpy 벡터연산으로 바꿔 시간축 n번만 돈다.
    """
    n = len(raws)
    fin = np.empty(N_BOOT)
    ruined_tot = 0
    done = 0
    while done < N_BOOT:
        b = min(chunk, N_BOOT - done)
        idx = RNG.integers(0, n, size=(b, n)).astype(np.int32)
        eq = np.ones(b)
        peak = np.ones(b)
        alive = np.ones(b, bool)
        for j in range(n):
            g = idx[:, j]
            sz = SIZE * scale[g]
            sz = np.where(eq < peak * (1.0 - DD_PCT), sz * DD_FACTOR, sz)
            step = raws[g] * sz * lev - fee * sz * lev
            eq = np.where(alive, np.maximum(eq * (1.0 + step), 1e-12), eq)
            np.maximum(peak, eq, out=peak)
            alive &= eq > RUIN
            if not alive.any():
                break
        fin[done:done + b] = eq
        ruined_tot += int((~alive).sum())
        done += b
    p5, p50 = np.percentile(fin, [5, 50])
    return float(p5), float(p50), 100.0 * ruined_tot / N_BOOT


# ── 변형 정의: 진입모드 3 × HTF 2 × TP/SL 2 = 12 ──────────────────────
ENTRY = ("Divergence Only", "Momentum Cross", "Both")
HTF = (True, False)
TPSL = ("ATR Dynamic", "Fixed Percent")


def main() -> int:
    t0 = time.time()
    dfs = {s: load(s, TF) for s in SYMS}
    for s, d in dfs.items():
        print(f"{s} {TF} — {len(d)}봉  {d.index[0]} ~ {d.index[-1]}", flush=True)
    months = max((d.index[-1] - d.index[0]).days for d in dfs.values()) / 30.44

    variants = []
    for em in ENTRY:
        for htf in HTF:
            for tm in TPSL:
                name = (f"{em} | HTF {'on' if htf else 'off'} | "
                        f"{'ATR' if tm == 'ATR Dynamic' else 'Fixed'}")
                params = dict(entry_mode=em, use_htf_filter=htf, tpsl_mode=tm)
                rows = []
                for s in SYMS:
                    for t in backtest(dfs[s], **params):
                        rows.append((int(t["ts"].value), int(t["ex"].value),
                                     float(t["raw"]), float(t["r"]), s))
                rows.sort(key=lambda x: x[0])
                n = len(rows)
                if n == 0:
                    variants.append(dict(name=name, params=params, n=0))
                    print(f"\n[{name}] 거래 0건", flush=True)
                    continue
                spans = np.array([[a, b] for a, b, _, _, _ in rows], dtype=np.int64)
                raws = np.array([c for _, _, c, _, _ in rows], float)
                rr = np.array([d for _, _, _, d, _ in rows], float)
                conc = concurrency(spans)
                scale = 1.0 / conc

                v = dict(
                    name=name, params=params, n=n,
                    n_by_sym={s: sum(1 for r in rows if r[4] == s) for s in SYMS},
                    monthly=n / months,
                    r_mean=float(rr.mean()), r_med=float(np.median(rr)),
                    r_sum=float(rr.sum()), win=float((rr > 0).mean()),
                    low_sample=bool(n < 30),
                    conc_mean=float(conc.mean()),
                    compound={}, mdd={}, ruin={}, p5={}, p50={},
                    ruin_actual={}, ruin_at_trade={}, ruin_at_date={},
                    compound_nocap={},
                )
                for lev in LEVS:
                    for fl, fee in FEES.items():
                        k = f"{lev}x_{fl}"
                        eq, mdd, ru, ri = sim(raws, scale, lev, fee)
                        p5, p50, rp = boot(raws, scale, lev, fee)
                        v["compound"][k] = eq
                        v["mdd"][k] = mdd
                        v["ruin"][k] = rp
                        v["p5"][k] = p5
                        v["p50"][k] = p50
                        v["ruin_actual"][k] = bool(ru)
                        v["ruin_at_trade"][k] = int(ri)
                        v["ruin_at_date"][k] = (
                            str(np.datetime64(rows[ri][0], "ns"))[:10] if ri >= 0
                            else None)
                        v["compound_nocap"][k] = sim_nocap(raws, scale, lev, fee)
                variants.append(v)

                print(f"\n[{name}] {n}건 (BTC {v['n_by_sym']['BTCUSDT']}/"
                      f"ETH {v['n_by_sym']['ETHUSDT']}) · 월 {v['monthly']:.2f}건 · "
                      f"건당 {v['r_mean']:+.4f}R (중앙 {v['r_med']:+.3f}) · "
                      f"승률 {v['win']*100:.1f}% · ΣR {v['r_sum']:+.1f}"
                      f"{'  ※표본부족' if v['low_sample'] else ''}", flush=True)
                for k in v["compound"]:
                    ra = (f" · 파산 {v['ruin_at_trade'][k]}번째({v['ruin_at_date'][k]})"
                          if v["ruin_actual"][k] else " · 생존")
                    print(f"   {k:>9}: 자산 {v['compound'][k]:.4f}x · "
                          f"MDD {v['mdd'][k]:.1f}% · 파산확률 {v['ruin'][k]:.1f}% · "
                          f"5%분위 {v['p5'][k]:.5f}x · 중앙 {v['p50'][k]:.5f}x · "
                          f"컷없음 {v['compound_nocap'][k]:.3e}x{ra}", flush=True)

    out = dict(tf=TF, symbols=list(SYMS), months=months,
               n_settings_tried=len(variants),
               bonferroni_alpha=0.05 / len(variants),
               eval=dict(size_pct=SIZE, levs=list(LEVS), fees_roundtrip=FEES,
                         dd_throttle=[DD_PCT, DD_FACTOR], ruin_level=RUIN,
                         n_boot=N_BOOT, concurrency_split=True),
               variants=variants)
    p = os.path.join(RES, "data/flow/15min.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {p} · {time.time()-t0:.0f}초", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
