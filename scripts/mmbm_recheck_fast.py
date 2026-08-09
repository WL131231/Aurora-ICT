"""#AUTONOMOUS 2026-08-09: MMBM 재검증 — 정합 기준선 위에서 (벡터화판).

## 왜 다시 하나 (8/6 판정을 무효화한 사건)

8/6 배선(#415)의 근거는 이랬다:

    SB 단독    45건 · 월 0.77 · 건당 +0.686R
    MMBM 단독 862건 · 월 14.5 · 건당 +0.199R
    → SB 와 **중복 0건**, 빈도 20배

그런데 8/9 정합 2차 복구로 **그 SB 45건이 라이브의 1/9 였음**이 드러났다.
정합 후 SB 는 402건 · 건당 −0.067R 이다. 비교의 **양쪽이 다 바뀌었고**,
무엇보다 SB 가 9배 늘었으니 **중복이 0 일 수 없다**.

MMBM 은 지금 실계좌에서 돌고 있다. 판정이 나쁘면 끄는 PR 이 필요하다.

## 원본 대비 바뀐 것

① `sim()` 의 동시보유 계산이 O(n²) 순수 루프라, 표본이 45→1,264 건으로 늘자
   부트스트랩 1,000회가 25억 연산이 됐다(수 시간). numpy 브로드캐스팅으로 교체.
   **결과는 동일** — 아래 `_selftest_conc` 가 원본 정의와 일치함을 매번 확인한다.
② 무거운 시뮬레이션 **전에** 중복·건당R 부터 출력한다. 판정의 절반이 거기 있고,
   시뮬레이션이 실패해도 그 답은 건진다.
③ SB/MMBM 원천 데이터를 pickle 로 남겨 재실행이 즉시 되게 한다.
"""

from __future__ import annotations

import os
import pickle
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
RNG = np.random.default_rng(20260809)
N_BOOT = 1000
N_PERM = 5000
DEDUP_MS = 60 * 60 * 1000        # 1시간 내 같은 심볼·방향이면 같은 기회로 본다
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "data", "axis", "_mmbm_parity_rows.pkl")


# ---------------------------------------------------------------- 데이터 수집

def sb_rows() -> list[dict]:
    """Silver Bullet — 라이브 정합(conf5) 그대로. 정합 이식 후 402건."""
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


def mmbm_rows(*, use_smt: bool, require_sweep: bool) -> list[dict]:
    """MMBM 백테. 게이트 조합을 명시적으로 받는다.

    ⚠️ 2026-08-09 발견: 라이브 `detect_mmbm_setup` 은 **SMT 도 스윕도 쓰지 않는다**
    (조건 4개 = 신선 CHoCH · discount/premium · HTF 정합 · FVG). 그런데 백테
    `mmbm_full.backtest` 의 기본값은 둘 다 True 였고, 게다가 SMT 는 시그니처 불일치
    예외를 삼켜 조용히 꺼져 있었다. 8/6 배선 판정(862건)은 **스윕만 켜진 제3의 모델**을
    잰 것이다. 라이브 정합은 `use_smt=False, require_sweep=False`.
    """
    out = []
    for sym in SYMS:
        _df, tr = M.backtest(sym, use_smt=use_smt,
                             require_sweep=require_sweep, detail=True)
        for ent_ms, _net_pct, d, ex_ms, r_mult, gross in tr:
            out.append({"src": "MMBM", "sym": sym, "ent": int(ent_ms),
                        "ex": int(ex_ms), "raw": float(gross),
                        "r": float(r_mult), "dir": int(d)})
    return out


# 게이트 조합 3종 — 라이브 정합이 무엇인지 확정하기 위해 나란히 잰다.
VARIANTS = (
    ("라이브 정합(SMT✗ 스윕✗)", False, False),
    ("8/6 판정 조건(SMT✗ 스윕✓)", False, True),
    ("정통(SMT✓ 스윕✓)", True, True),
)


def load_rows() -> tuple[list[dict], dict[str, list[dict]]]:
    # SB 와 MMBM 을 따로 본다 — SB(정합 타임라인)가 가장 비싸므로, 게이트 조합만
    # 바꿔 다시 잴 때 SB 는 항상 재사용한다. 구형 캐시({sb, mm})도 sb 는 살려 쓴다.
    sb: list[dict] | None = None
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            d = pickle.load(f)
        if "mm_by" in d:
            print(f"  (캐시 재사용 — SB {len(d['sb'])}건 · MMBM 3조합)", flush=True)
            return d["sb"], d["mm_by"]
        sb = d.get("sb")
        if sb:
            print(f"  (구형 캐시 — SB {len(sb)}건만 재사용, MMBM 3조합 재계산)", flush=True)
    if sb is None:
        print("  원천 데이터 계산 중 …", flush=True)
        sb = sb_rows()
    print(f"  SB {len(sb)}건 확보 — MMBM 게이트 조합 3종 계산", flush=True)
    mm_by: dict[str, list[dict]] = {}
    for lab, smt, sweep in VARIANTS:
        mm_by[lab] = mmbm_rows(use_smt=smt, require_sweep=sweep)
        print(f"    {lab:<26} {len(mm_by[lab]):>5}건", flush=True)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump({"sb": sb, "mm_by": mm_by}, f)
    return sb, mm_by


# ---------------------------------------------------------------- 복리 시뮬

def _conc(ent: np.ndarray, ex: np.ndarray) -> np.ndarray:
    """동시보유 개수 — 원본의 O(n²) 루프를 브로드캐스팅으로.

    원본 정의: 자기 자신 제외하고 ``q.ent <= r.ex and q.ex >= r.ent`` 인 거래 수 + 1.
    """
    ov = (ent[None, :] <= ex[:, None]) & (ex[None, :] >= ent[:, None])
    np.fill_diagonal(ov, False)
    return 1.0 + ov.sum(axis=1)


def _selftest_conc() -> None:
    """벡터화가 원본 정의와 같은 값을 내는지 매 실행 확인 — 교체의 안전장치."""
    r = np.random.default_rng(7)
    e = r.integers(0, 100, 60).astype(float)
    x = e + r.integers(1, 30, 60)
    slow = np.array([1.0 + sum(1 for j in range(60)
                               if j != i and e[j] <= x[i] and x[j] >= e[i])
                     for i in range(60)])
    assert np.array_equal(slow, _conc(e, x)), "벡터화 conc 불일치"


def sim_arr(raw: np.ndarray, ent: np.ndarray, ex: np.ndarray):
    """복리 — 동시보유 size 분할 + DD 스로틀. (최종, 낙폭%, 파산)."""
    n = len(raw)
    if n < 5:
        return float("nan"), float("nan"), False
    o = np.argsort(ent, kind="stable")
    raw, ent, ex = raw[o], ent[o], ex[o]
    conc = _conc(ent, ex)
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i in range(n):
        sz = SIZE / conc[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        eq *= (1.0 + raw[i] * sz * LEV - 2.0 * TAKER_FEE_PCT * sz * LEV)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return eq, 100.0 * mdd, True
    return eq, 100.0 * mdd, False


def cols(rows: list[dict]):
    return (np.array([x["raw"] for x in rows], float),
            np.array([x["ent"] for x in rows], float),
            np.array([x["ex"] for x in rows], float))


def boot(rows: list[dict]):
    n = len(rows)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    raw, ent, ex = cols(rows)
    fin, ruin = np.empty(N_BOOT), 0
    for k in range(N_BOOT):
        p = RNG.integers(0, n, size=n)
        e, _, r_ = sim_arr(raw[p], ent[p], ex[p])
        fin[k] = e
        ruin += int(r_)
    p50, p5 = np.percentile(fin, [50, 5])
    return float(p50), float(p5), 100.0 * ruin / N_BOOT


# ---------------------------------------------------------------- 출력

def quick(tag: str, rows: list[dict]) -> None:
    """시뮬 없이 — 건수·빈도·건당R·승률만. 판정의 절반."""
    if len(rows) < 5:
        print(f"  {tag:<22}{len(rows):>6}  표본부족", flush=True)
        return
    r = np.array([x["r"] for x in rows])
    span = (max(x["ex"] for x in rows) - min(x["ent"] for x in rows)) / 86400000 / 30.4
    lo, hi = np.percentile([r[RNG.integers(0, len(r), len(r))].mean()
                            for _ in range(2000)], [2.5, 97.5])
    print(f"  {tag:<22}{len(rows):>6}{len(rows) / max(span, 1e-9):>8.2f}"
          f"{r.mean():>+9.3f}{100 * (r > 0).mean():>6.0f}%"
          f"   [{lo:+.3f} ~ {hi:+.3f}]", flush=True)


def heavy(tag: str, rows: list[dict]) -> None:
    if len(rows) < 5:
        return
    e0, m0, _ = sim_arr(*cols(rows))
    p50, p5, pr = boot(rows)
    print(f"  {tag:<22}{e0:>10.2f}x{m0:>8.1f}%{p50:>10.2f}x{p5:>8.2f}x{pr:>8.1f}%",
          flush=True)


def main() -> int:
    _selftest_conc()
    print("=== MMBM 재검증 — 정합 기준선 (BTC+ETH · 7x · 복리)", flush=True)
    sb, mm_by = load_rows()
    sb_keys = {(x["sym"], x["dir"], x["ent"] // DEDUP_MS) for x in sb}

    hdr = (f"  {'구성':<26}{'거래':>6}{'월빈도':>8}{'건당R':>9}{'승률':>7}"
           f"   {'건당R 95% 구간':<22}")
    print(f"\n{hdr}", flush=True)
    quick("SB 단독(현행 Origo)", sb)

    for lab, _s, _w in VARIANTS:
        mm = mm_by[lab]
        add = [x for x in mm
               if (x["sym"], x["dir"], x["ent"] // DEDUP_MS) not in sb_keys]
        print(f"\n### {lab}   (SB 와 중복 {len(mm) - len(add)}건)", flush=True)
        quick("  MMBM 단독", mm)
        quick("  순수추가분", add)
        quick("  SB + MMBM 결합", sb + add)

        if len(add) >= 5:
            ar = np.array([x["r"] for x in add])
            pool = np.array([x["r"] for x in mm])
            obs = ar.mean()
            # ① 추가분이 MMBM 전체 대비 특별한가 (중복 제거의 의미)
            d1 = np.array([RNG.choice(pool, size=len(ar), replace=False).mean()
                           for _ in range(N_PERM)])
            # ② MMBM 이 SB 대비 나은가 — 라벨 순열(두 표본 합쳐 무작위 분할)
            sr = np.array([x["r"] for x in sb])
            both = np.concatenate([ar, sr])
            obs2 = ar.mean() - sr.mean()
            d2 = np.empty(N_PERM)
            for k in range(N_PERM):
                p_ = RNG.permutation(both)
                d2[k] = p_[:len(ar)].mean() - p_[len(ar):].mean()
            print(f"    순열① 추가분 vs MMBM전체  p={float((d1 >= obs).mean()):.4f}", flush=True)
            print(f"    순열② MMBM vs SB 차 {obs2:+.3f}R  "
                  f"p={float((d2 >= obs2).mean()):.4f}", flush=True)

        print(f"    {'':<24}{'자산':>10}{'낙폭':>8}{'부트중앙':>10}{'5%':>8}{'파산':>8}",
              flush=True)
        for tag, rows in (("  MMBM 단독", mm), ("  SB + MMBM 결합", sb + add)):
            heavy(tag, rows)

    print("\n  판정 — **라이브 정합 행만** 배선 근거가 된다. 나머지 둘은 대조군.", flush=True)
    print("  기준: 추가분 건당R 95% 구간이 0 초과 + 결합이 SB 단독 대비 악화 없음.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
