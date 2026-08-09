"""#AUTONOMOUS 2026-08-06: MMBM 재검증 — BTC+ETH · 7x · 복리 판정.

## 왜 다시 하나
① 7/20 검증은 **하니스 정합(7/30) 이전**이다. 그때 라이브 기능 19개 중 10개가
   백테에 없다는 것이 드러났으므로 그 시점 결론은 재확인이 필요하다.
② 그 뒤 판단 기준 자체가 바뀌었다 — 단리 net 이 아니라 **복리 자산·낙폭·파산확률**.
③ 페어를 BTC+ETH 로 좁히는 안을 검토 중이고, 레버리지는 7x 로 내려갔다.

## 그리고 발견한 것
`mmbm_enabled` 가 **매니저에서 배선되지 않아 라이브에서 한 번도 돌지 않았다**
(기본 False, settings 에서 넘기는 곳 없음). 7/21 "활성 실측" 결정이 코드에
반영되지 않은 채 2주가 지났다. 이 재검증이 그 배선 여부를 정한다.

## 판정 — confluence 실험과 같은 잣대
빈도가 늘어도 **추가되는 거래가 나쁘면 의미가 없다**(문턱 5→4 에서 추가 68건이
승률 25% 였던 것처럼). 그래서 총량이 아니라 **증분**을 본다:
    · MMBM 단독 성적 (건당 R · 승률)
    · SB(conf5)와 시각 중복 제거 후 **순수 추가분**
    · 결합 시 복리 자산 · 낙폭 · 파산확률이 개선되는가
    · 순열검정 — 추가분이 무작위 대비 유의한가
"""

from __future__ import annotations

import datetime as dt
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmbm_full as M  # noqa: E402
from live_parity import LIVE_BASE, run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

SYMS = ["BTCUSDT", "ETHUSDT"]
SIZE = LIVE_BASE["size_pct"]
LEV = 7.0
DD_PCT, DD_FACTOR = 0.25, 0.7
RUIN = 0.20
RNG = np.random.default_rng(20260806)
N_BOOT = 1000
N_PERM = 5000
DEDUP_MS = 60 * 60 * 1000        # 1시간 내 같은 방향이면 같은 기회로 본다


def sb_rows():
    """Silver Bullet — 라이브 정합(conf5) 그대로."""
    out = []
    for sym in SYMS:
        df5, kept, _ = run_live_parity(sym)
        idx = df5.index
        for t in kept:
            risk = abs(float(t.entry) - float(getattr(t, "entry_sl", 0.0) or 0.0))
            out.append({
                "src": "SB", "sym": sym,
                "ent": int(idx[t.entry_idx].value // 10**6),
                "ex": int(idx[min(t.exit_idx, len(idx) - 1)].value // 10**6),
                "raw": float(t.raw_pnl_pct),
                "r": (float(t.raw_pnl_pct) * float(t.entry) / risk) if risk > 0 else 0.0,
                "dir": 1 if str(getattr(t.direction, "value", t.direction)).lower() == "long" else -1,
            })
    return out


def mmbm_rows():
    out = []
    for sym in SYMS:
        _df, tr = M.backtest(sym, detail=True)
        for ent_ms, _net_pct, d, ex_ms, r_mult, gross in tr:
            out.append({"src": "MMBM", "sym": sym, "ent": int(ent_ms),
                        "ex": int(ex_ms), "raw": float(gross),
                        "r": float(r_mult), "dir": int(d)})
    return out


def sim(rows):
    """복리 — 동시보유 size 분할 + DD 스로틀. (최종, 낙폭%, 파산)."""
    n = len(rows)
    if n < 5:
        return float("nan"), float("nan"), False
    rows = sorted(rows, key=lambda r: r["ent"])
    conc = np.empty(n)
    for i, r in enumerate(rows):
        conc[i] = 1.0 + sum(1 for j, q in enumerate(rows)
                            if j != i and q["ent"] <= r["ex"] and q["ex"] >= r["ent"])
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i, r in enumerate(rows):
        sz = SIZE / conc[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        eq *= (1.0 + r["raw"] * sz * LEV - 2.0 * TAKER_FEE_PCT * sz * LEV)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return eq, 100.0 * mdd, True
    return eq, 100.0 * mdd, False


def boot(rows):
    n = len(rows)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    fin, ruin = np.empty(N_BOOT), 0
    for k in range(N_BOOT):
        pick = [rows[i] for i in RNG.integers(0, n, size=n)]
        e, _, r_ = sim(pick)
        fin[k] = e
        ruin += int(r_)
    p50, p5 = np.percentile(fin, [50, 5])
    return float(p50), float(p5), 100.0 * ruin / N_BOOT


def describe(tag, rows):
    if len(rows) < 5:
        print(f"  {tag:<20}{len(rows):>5}  표본부족", flush=True)
        return
    r = np.array([x["r"] for x in rows])
    e0, m0, _ = sim(rows)
    p50, p5, pr = boot(rows)
    span = (max(x["ex"] for x in rows) - min(x["ent"] for x in rows)) / 86400000 / 30.4
    print(f"  {tag:<20}{len(rows):>5}{len(rows) / max(span, 1e-9):>8.2f}"
          f"{r.mean():>8.2f}{100 * (r > 0).mean():>5.0f}%"
          f"{e0:>9.2f}x{m0:>7.1f}%{p50:>9.2f}x{p5:>7.2f}x{pr:>7.1f}%", flush=True)


def main() -> int:
    sb = sb_rows()
    mm = mmbm_rows()
    # SB 와 같은 기회(같은 심볼·방향·1시간 내)는 제외 — 순수 추가분만
    sb_keys = {(x["sym"], x["dir"], x["ent"] // DEDUP_MS) for x in sb}
    add = [x for x in mm
           if (x["sym"], x["dir"], x["ent"] // DEDUP_MS) not in sb_keys]

    print("=== MMBM 재검증 (BTC+ETH · 7x · 동시보유 · DD 스로틀 · 복리) ===", flush=True)
    print(f"  {'구성':<20}{'거래':>5}{'월빈도':>8}{'건당R':>8}{'승률':>6}"
          f"{'자산':>9}{'낙폭':>8}{'부트중앙':>9}{'5%':>7}{'파산':>7}", flush=True)
    describe("SB 단독(현행)", sb)
    describe("MMBM 단독", mm)
    describe("MMBM 순수추가분", add)
    describe("SB + MMBM 결합", sb + add)

    print(f"\n  SB {len(sb)}건 · MMBM {len(mm)}건 중 SB 와 겹치지 않는 추가 "
          f"{len(add)}건 (중복 {len(mm) - len(add)}건 제거)", flush=True)

    if len(add) >= 5:
        ar = np.array([x["r"] for x in add])
        pool = np.array([x["r"] for x in mm])
        obs = ar.mean()
        dist = np.array([RNG.choice(pool, size=len(ar), replace=False).mean()
                         for _ in range(N_PERM)])
        p = float((dist >= obs).mean())
        print(f"  추가분 건당 {obs:+.3f}R · 무작위 중앙 {np.median(dist):+.3f}R"
              f" · p={p:.3f}", flush=True)

    print("\n  판정 — 추가분 건당 R>0 이고, 결합 시 낙폭·파산이 나빠지지 않아야 배선한다.",
          flush=True)
    return 0




def verify() -> int:
    """추가 검증 — 건당 R 신뢰구간 · 연도 일관성 · 롱숏 (main 이후 호출)."""
    sb, mm = sb_rows(), mmbm_rows()
    r = np.array([x["r"] for x in mm])
    n = len(r)

    print("\n=== 추가 검증 ===", flush=True)
    means = np.array([r[RNG.integers(0, n, size=n)].mean() for _ in range(20000)])
    lo, hi = np.percentile(means, [2.5, 97.5])
    print(f"① 건당 R 신뢰구간 — 평균 {r.mean():+.3f}R "
          f"[95% {lo:+.3f} ~ {hi:+.3f}]  → {'0 초과 확정' if lo > 0 else '0 포함(무의미)'}",
          flush=True)

    print("② 연도별 (7/20 기록: '횡보장 2025 약세' 재확인)", flush=True)
    ys = np.array([dt.datetime.utcfromtimestamp(x["ent"] / 1000).year for x in mm])
    for y in sorted(set(ys.tolist())):
        m = ys == y
        if m.sum() < 5:
            continue
        sub = [x for x, k in zip(mm, m, strict=False) if k]
        e, d, _ = sim(sub)
        print(f"   {y}  {m.sum():>4}건  건당 {r[m].mean():+.3f}R"
              f"  승률 {100 * (r[m] > 0).mean():>3.0f}%  자산 {e:>6.2f}x  낙폭 {d:>5.1f}%",
              flush=True)

    print("③ 롱/숏", flush=True)
    for d_, lab in ((1, "롱"), (-1, "숏")):
        sub = [x for x in mm if x["dir"] == d_]
        if len(sub) < 5:
            continue
        rr = np.array([x["r"] for x in sub])
        e, dd_, _ = sim(sub)
        print(f"   {lab}  {len(sub):>4}건  건당 {rr.mean():+.3f}R"
              f"  승률 {100 * (rr > 0).mean():>3.0f}%  자산 {e:>6.2f}x  낙폭 {dd_:>5.1f}%",
              flush=True)

    print("④ 결합 연도별 (SB+MMBM)", flush=True)
    comb = sb + mm
    yc = np.array([dt.datetime.utcfromtimestamp(x["ent"] / 1000).year for x in comb])
    for y in sorted(set(yc.tolist())):
        sub = [x for x, k in zip(comb, yc == y, strict=False) if k]
        if len(sub) < 5:
            continue
        e, d, _ = sim(sub)
        print(f"   {y}  {len(sub):>4}건  자산 {e:>6.2f}x  낙폭 {d:>5.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    main()
    raise SystemExit(verify())
