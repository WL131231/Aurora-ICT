"""#AUTONOMOUS 2026-08-07 · 축: 사후 게이트 3종 재판정 (BTC+ETH 2페어 체제)

## 무엇을 왜 재는가

Origo 의 사후 게이트 3종은 전부 **7페어 시절 데이터로 잡은 값**이다.
    · gate_regime     — |진입시점 추세%| < 페어 q33 이면 차단 (2026-06-23)
    · gate_cond_align — 약/중추세(|추세|<q70)에서 역추세 진입만 차단 (2026-07-17)
    · gate_nypm       — NY_PM(17~21 UTC = 02~06 KST) 진입 차단 (2026-07-16,
                        근거는 "그 시간대 승률 0~1%" 2026-06-24 관측)
2026-08-06 에 라이브 페어가 **BTC+ETH 2개**로 줄었다. 위 임계값(q33/q70)은
7페어 롤링 분위에서 나온 것이고, NY_PM 근거도 7페어 라이브 실측이었다.
→ 남은 2페어 데이터만으로 다시 판정한다.

## 1단계 — 구조적 상한부터 (직전 피보 연구 교훈)

피보나치 레벨 연구는 "손잡이가 물리적으로 R 을 0.027R 밖에 못 움직인다"를
나중에야 알았다. 그래서 여기서는 스윕 전에 먼저 **게이트를 완전히 끄면 최대
몇 R 이 움직이는가**를 산출한다. 게이트는 이진 필터라 그 상한이 정확히
"차단된 거래들의 R 합"이다. 이 상한이 같은 표본에서 나오는 잡음
(차단 건수 × 건당 R 표준편차) 보다 작으면 스윕 자체가 무의미하다.

## 2~5단계

2. 게이트 8조합(2^3) 전수 — 거래수·건당R·복리자산·MDD·파산확률·순열 p
3. 임계값 재추정 — BTC+ETH 자체 분위로 다시 잡고 분위 스윕
4. NY_PM 시간대 재확인 — UTC 시간대별 건당 R (2026-06-24 근거가 지금도 유효한가)
5. 차단분 해부 — 국면(추세 방향)×진입방향 기저 통제, 연도 일관성

## 평가 규약 (파트너 지시 2026-08-07)

· 단리 net 금지. 레버리지 7x · size_pct 0.9 · **동시보유 반영**(구간 겹치면 size 분할)
  · DD 스로틀(-25% 초과 시 ×0.7) · 파산(시드 20% 이하) 판정 · 부트스트랩 중앙값/5%분위
· R 배수로 측정 (SL 폭이 변하면 %수익 비교가 왜곡되므로)
· 순열검정 20000회 · 표본 30건 미만 셀은 "표본부족" 명시

## 한계 (결론에 반드시 명시)

· 사후 필터다. 게이트가 막은 자리에서 라이브가 **다른 셋업을 잡았을 대체 효과**는
  반영되지 않는다(페어당 동시 1포지션 제약).
· ote_up_level 0.786 / sweep_gate / smart_size / daily_loss_limit / MMBM 2번째 모델은
  replay 미구현(live_parity.GAPS). 전 시나리오 공통이라 상대 비교에는 영향 작음.
· HTF FVG flip 은 사후 적용(min_r=LIVE_FLIP_MIN_R). 라이브 정합 경로.
"""

from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from flip_ab_backtest import _dir_of, _sl_of, build_fvg_zones, flip_target_at  # noqa: E402
from live_parity import (  # noqa: E402
    LIVE_BASE, LIVE_FLIP_MIN_R, REGIME_Q33, STRONG_Q70,
)

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402
from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

# ── 현행 라이브(2026-08-07) ──────────────────────────────────────────────
PAIRS = ["BTCUSDT", "ETHUSDT"]     # 2026-08-06 축소
LEV = 7.0                          # 라이브 레버리지
SIZE = LIVE_BASE["size_pct"]       # 0.9
RUIN_LEVEL = 0.20                  # 시드 20% 이하 = 파산 판정
DD_PCT, DD_FACTOR = 0.25, 0.7      # origo_dd_throttle_pct / _factor
FLIP_MIN_W = 4                     # flip target 최소 가중치(1h+) — 라이브 정합
NYPM_LIVE = (17, 21)               # 현행 차단 구간 (UTC)

N_PERM = 20000
N_BOOT = 2000
MIN_CELL = 30                      # 이 미만은 "표본부족"
RNG = np.random.default_rng(20260807)

OUT_JSON = "data/axis/gates.json"


# ═══════════════════ 0단계: 수집 ═══════════════════

def flip_apply(df5, t, zones, min_r: float):
    """HTF FVG flip 을 raw(가격변동 비율) 공간에서 사후 적용.

    live_parity 의 flip 경로(flip_verdict.apply_flip mode="minr")와 같은 판정이되,
    복리 시뮬에 레버리지를 직접 넣어야 하므로 net 이 아니라 raw 를 돌려준다.
    반환: (raw, 청산 idx, flip 발동 여부)
    """
    d = _dir_of(t)
    entry = float(t.entry)
    exit_i = min(t.exit_idx, len(df5) - 1)
    z = flip_target_at(zones, t.entry_idx, d, entry, FLIP_MIN_W)
    if z is None:
        return float(t.raw_pnl_pct), exit_i, False
    lo_i, hi_i = t.entry_idx + 1, exit_i
    if hi_i <= lo_i:
        return float(t.raw_pnl_pct), exit_i, False
    seg_hi = df5["high"].to_numpy()[lo_i:hi_i + 1]
    seg_lo = df5["low"].to_numpy()[lo_i:hi_i + 1]
    hit = np.flatnonzero((seg_lo <= z["hi"]) & (seg_hi >= z["lo"]))
    if not hit.size:
        return float(t.raw_pnl_pct), exit_i, False
    fpx = z["lo"] if d == 1 else z["hi"]
    raw = (fpx - entry) / entry * d
    risk = abs(entry - _sl_of(t, entry, d))
    r_at = (raw * entry / risk) if risk > 0 else 0.0
    if r_at >= min_r:                       # #FLIP-MIN-R — 이익 1.5R 이상일 때만 flip
        return raw, int(lo_i + hit[0]), True
    return float(t.raw_pnl_pct), exit_i, False


def collect() -> list[dict]:
    """페어별 **게이트 적용 전** 원본 거래 전부를 레코드로. flip 은 이미 적용."""
    recs: list[dict] = []
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**LIVE_BASE)
        bt = run_backtest_from_timeline(df5, cached_setup_timeline(df5, cfg, sym), cfg)
        zones = build_fvg_zones(df5)
        n_flip = 0
        for t in bt.trades:
            raw, xi, fired = flip_apply(df5, t, zones, LIVE_FLIP_MIN_R)
            n_flip += int(fired)
            entry = float(t.entry)
            sl = _sl_of(t, entry, _dir_of(t))
            risk_frac = abs(entry - sl) / entry
            if risk_frac <= 0:
                continue
            ts = df5.index[t.entry_idx]
            recs.append(dict(
                sym=sym,
                t_in=int(ts.value), t_out=int(df5.index[xi].value),
                year=int(ts.year), hour=int(ts.hour),
                d=_dir_of(t),
                trend=float(t.entry_trend_pct or 0.0),
                raw=raw, r=raw / risk_frac, risk_frac=risk_frac,
            ))
        print(f"  {sym}: 원본 {len(bt.trades)}건 / flip 발동 {n_flip}건 "
              f"/ zone {len(zones)}", flush=True)
    recs.sort(key=lambda x: x["t_in"])
    return recs


# ═══════════════════ 게이트 (파라미터화) ═══════════════════

def regime_floor(sym: str, mode, qtab: dict) -> float:
    """regime 게이트 하한. mode: "live"(라이브 상수) / "off" / float(분위 재추정)."""
    if mode == "off":
        return 0.0
    if mode == "live":
        return REGIME_Q33.get(sym, 0.0)
    return float(qtab[sym][mode])       # 재추정 분위 테이블


def strong_floor(sym: str, mode, qtab: dict) -> float:
    """cond_align 의 '강추세' 문턱. "live"/"off"/float(분위) / "all"(전 역추세 차단)."""
    if mode == "off":
        return 0.0
    if mode == "all":
        return float("inf")
    if mode == "live":
        return STRONG_Q70.get(sym, 0.0)
    return float(qtab[sym][mode])


def select(recs, *, regime="live", cond="live", nypm=NYPM_LIVE,
           qreg=None, qstr=None) -> list[dict]:
    """게이트 조합 적용 후 통과 거래. gate_* 의 판정 로직과 동일하게 재현.

    nypm 은 (시작,끝) 튜플 또는 **차단 시각 집합**(set) 또는 None(끔).
    """
    if isinstance(nypm, tuple):
        block_hours = set(range(nypm[0], nypm[1]))
    elif isinstance(nypm, (set, frozenset)):
        block_hours = set(nypm)
    else:
        block_hours = set()
    out = []
    for r in recs:
        fl = regime_floor(r["sym"], regime, qreg or {})
        if fl > 0 and abs(r["trend"]) < fl:
            continue
        st = strong_floor(r["sym"], cond, qstr or {})
        if st > 0:
            signed = r["trend"] * r["d"]
            if abs(r["trend"]) < st and signed < 0:   # 약/중추세 역추세 = 차단
                continue
        if r["hour"] in block_hours:
            continue
        out.append(r)
    return out


# ═══════════════════ 복리 시뮬 (동시보유·DD스로틀·파산) ═══════════════════

def concurrency(rows) -> np.ndarray:
    """각 거래가 열려 있는 동안의 평균 동시 보유 수(자기 포함). 라이브 자본 분할 재현."""
    out = np.empty(len(rows))
    for i, a in enumerate(rows):
        ov = sum(1 for j, b in enumerate(rows)
                 if j != i and b["t_in"] <= a["t_out"] and b["t_out"] >= a["t_in"])
        out[i] = 1.0 + ov
    return out


def sim(raws, scale, lev=LEV):
    """복리 시뮬 — (최종배수, MDD%, 파산여부). DD 스로틀 포함."""
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i, r in enumerate(raws):
        sz = SIZE * scale[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        step = r * sz * lev - 2.0 * TAKER_FEE_PCT * sz * lev
        eq *= (1.0 + step)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN_LEVEL:
            return eq, 100.0 * mdd, True
    return eq, 100.0 * mdd, False


def evaluate(rows) -> dict:
    """후보 1개 전체 평가 — 거래수·월빈도·건당R·승률·복리·MDD·파산·부트스트랩."""
    n = len(rows)
    if n < 2:
        return dict(n=n)
    raws = np.array([x["raw"] for x in rows])
    rs = np.array([x["r"] for x in rows])
    scale = 1.0 / concurrency(rows)
    fin, mdd0, ruin0 = sim(raws, scale)
    months = (rows[-1]["t_in"] - rows[0]["t_in"]) / 1e9 / 86400 / 30.44
    boots = np.empty(N_BOOT)
    nruin = 0
    for k in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        e, _, ru = sim(raws[idx], scale[idx])
        boots[k] = e
        nruin += int(ru)
    p50, p5 = np.percentile(boots, [50, 5])
    ys: dict[int, float] = {}
    for x in rows:
        ys[x["year"]] = ys.get(x["year"], 0.0) + x["r"]
    return dict(
        n=n, per_month=n / max(months, 1e-9),
        r_mean=float(rs.mean()), r_sum=float(rs.sum()), r_std=float(rs.std(ddof=1)),
        wr=float(100.0 * (rs > 0).mean()),
        compound=float(fin), mdd=float(mdd0), ruin_run=bool(ruin0),
        ruin=float(100.0 * nruin / N_BOOT),
        boot_p50=float(p50), boot_p5=float(p5),
        years={str(k): round(v, 2) for k, v in sorted(ys.items())},
        ypos=sum(1 for v in ys.values() if v > 0), ytot=len(ys),
    )


# ═══════════════════ 순열검정 ═══════════════════

def perm_p(group_a: np.ndarray, group_b: np.ndarray, n_perm: int = N_PERM) -> float:
    """두 그룹 건당 R 평균차의 양측 순열검정 p. 라벨을 무작위로 섞어 20000회."""
    if group_a.size == 0 or group_b.size == 0:
        return float("nan")
    pool = np.concatenate([group_a, group_b])
    na = group_a.size
    obs = abs(group_a.mean() - group_b.mean())
    rng = np.random.default_rng(12345)
    cnt = 0
    for _ in range(n_perm):
        p = rng.permutation(pool)
        if abs(p[:na].mean() - p[na:].mean()) >= obs - 1e-15:
            cnt += 1
    return (cnt + 1) / (n_perm + 1)


def diff_perm(base_rows, cand_rows) -> tuple[float, int]:
    """기준선과 후보의 **차집합 거래**가 공통 거래와 R 분포가 다른가.

    후보가 게이트를 끈 경우 차집합 = 새로 들어온 거래, 조인 경우 = 빠진 거래.
    반환 (p, 차집합 건수).
    """
    key = lambda x: (x["sym"], x["t_in"])  # noqa: E731
    kb = {key(x) for x in base_rows}
    kc = {key(x) for x in cand_rows}
    common = kb & kc
    diff = (kb | kc) - common
    all_rows = {key(x): x for x in base_rows} | {key(x): x for x in cand_rows}
    ga = np.array([all_rows[k]["r"] for k in diff])
    gb = np.array([all_rows[k]["r"] for k in common])
    return perm_p(ga, gb), int(ga.size)


# ═══════════════════ 출력 헬퍼 ═══════════════════

def row_line(name: str, e: dict, base: dict | None = None, p: float | None = None,
             ndiff: int | None = None) -> str:
    if e.get("n", 0) == 0:
        return f"  {name:<26} 거래 없음"
    d = "" if base is None else f"{e['r_mean'] - base['r_mean']:+6.3f}"
    ps = "  -  " if p is None or np.isnan(p) else f"{p:5.3f}"
    warn = "" if e["n"] >= MIN_CELL else " ⚠표본부족"
    return (f"  {name:<26}n={e['n']:4d} 월{e['per_month']:4.2f} "
            f"R={e['r_mean']:+6.3f}({d}) 승={e['wr']:3.0f}% "
            f"복리={e['compound']:9.2f}x MDD={e['mdd']:5.1f}% "
            f"파산={e['ruin']:5.1f}% 중앙={e['boot_p50']:8.2f}x "
            f"5%={e['boot_p5']:6.2f}x 연도{e['ypos']}/{e['ytot']} p={ps}"
            f"{'' if ndiff is None else f' Δn={ndiff}'}{warn}")


def main() -> int:
    print("=== 축: 게이트 3종 재판정 (BTC+ETH, 2026-08-07) ===\n", flush=True)
    print("[0단계] 원본 타임라인 수집 (게이트 적용 전, flip 1.5R 적용)", flush=True)
    recs = collect()
    print(f"  합계 원본 {len(recs)}건\n", flush=True)

    n_tried = 0            # 다중비교 카운터 — 시도한 조합 수를 전부 센다

    # ── 1단계: 구조적 상한 ────────────────────────────────────────────
    print("=" * 100, flush=True)
    print("[1단계] 구조적 상한 — 각 게이트가 R 을 최대 얼마나 움직일 수 있나", flush=True)
    print("  (게이트는 이진 필터 → 상한 = 차단된 거래들의 R 합. 이게 잡음보다 작으면 무의미)",
          flush=True)
    print("  ※ 차단 건수는 **한계 기여**(나머지 2종은 ON 유지). 중복 차단분은 제외된다.",
          flush=True)
    base_rows = select(recs)
    base_r = np.array([x["r"] for x in base_rows])
    print(f"\n  기준선(전부 ON) n={len(base_rows)} 건당R={base_r.mean():+.3f} "
          f"R합={base_r.sum():+.2f} 표준편차={base_r.std(ddof=1):.3f}", flush=True)
    print(f"\n  {'게이트':<14}{'차단':>6}{'차단 R합':>10}{'차단 건당R':>11}"
          f"{'잡음(1σ)':>10}{'상한/잡음':>10}  판정", flush=True)
    ceilings = {}
    for gname, kw in (("regime", dict(regime="off")),
                      ("cond_align", dict(cond="off")),
                      ("nypm", dict(nypm=None))):
        off_rows = select(recs, **kw)
        kb = {(x["sym"], x["t_in"]) for x in base_rows}
        blocked = [x for x in off_rows if (x["sym"], x["t_in"]) not in kb]
        if not blocked:
            continue
        br = np.array([x["r"] for x in blocked])
        # 잡음 = 같은 건수를 이 표본에서 무작위로 뽑았을 때 R합의 표준편차
        noise = base_r.std(ddof=1) * np.sqrt(len(br))
        ratio = abs(br.sum()) / noise
        verdict = "잡음 이하 → 스윕 무의미" if ratio < 1.0 else \
                  ("잡음권(1~2σ)" if ratio < 2.0 else "잡음 초과 → 검정 가치 있음")
        # 검정력 — 관측된 건당R 격차를 유의하게 잡으려면 그룹당 몇 건이 필요한가.
        # n ≈ 16σ²/Δ² (양측 α=0.05, 검정력 80%, 등분산 근사)
        delta = abs(br.mean() - base_r.mean())
        need = 16.0 * base_r.std(ddof=1) ** 2 / max(delta, 1e-9) ** 2
        ceilings[gname] = dict(n_blocked=int(len(br)), r_sum=float(br.sum()),
                               r_mean=float(br.mean()), r_raw_sum=float(
                                   sum(x["raw"] for x in blocked)),
                               noise_1sigma=float(noise), ratio=float(ratio),
                               delta_r=float(delta), n_needed=float(need),
                               verdict=verdict)
        print(f"  {gname:<14}{len(br):>6}{br.sum():>+10.2f}{br.mean():>+11.3f}"
              f"{noise:>10.2f}{ratio:>10.2f}  {verdict}", flush=True)
        print(f"                 └ 통과분과의 건당R 격차 {delta:.3f} → 이를 유의하게 "
              f"잡으려면 그룹당 {need:.0f}건 필요 "
              f"(현 빈도 월 {len(base_rows) / 59.0:.2f}건 기준 "
              f"{need / max(len(base_rows) / 59.0, 1e-9) / 12:.0f}년)", flush=True)

    tot_block = len(recs) - len(base_rows)
    print(f"\n  3종 전부 OFF 시 추가되는 거래 {tot_block}건 "
          f"(원본 {len(recs)} → 기준선 {len(base_rows)}, 차단율 "
          f"{100 * tot_block / len(recs):.0f}%)", flush=True)

    # ── 2단계: 8조합 전수 ─────────────────────────────────────────────
    print("\n" + "=" * 100, flush=True)
    print("[2단계] 게이트 8조합 전수 (2^3). 복리=레버7x·size0.9·동시보유분할·DD스로틀",
          flush=True)
    base_eval = evaluate(base_rows)
    print("\n" + row_line("★기준선 (전부 ON)", base_eval), flush=True)
    print(flush=True)
    combos = []
    for rg, cd, ny in itertools.product([True, False], repeat=3):
        if (rg, cd, ny) == (True, True, True):
            continue
        n_tried += 1
        kw = dict(regime="live" if rg else "off",
                  cond="live" if cd else "off",
                  nypm=NYPM_LIVE if ny else None)
        rows = select(recs, **kw)
        e = evaluate(rows)
        p, nd = diff_perm(base_rows, rows) if rows else (float("nan"), 0)
        label = "+".join([n for n, on in (("regime", rg), ("cond", cd), ("nypm", ny)) if on]) or "전부OFF"
        combos.append((label, kw, e, p, nd))
        print(row_line(f"ON: {label}", e, base_eval, p, nd), flush=True)

    # ── 3단계: 임계값 재추정 (BTC+ETH 자체 분위) ──────────────────────
    print("\n" + "=" * 100, flush=True)
    print("[3단계] 임계값 재추정 — BTC+ETH 셋업 분포로 분위를 다시 잡으면", flush=True)
    qs = [0.1, 0.2, 0.33, 0.4, 0.5, 0.6, 0.7, 0.8]
    qreg, qstr = {}, {}
    for sym in PAIRS:
        tr = np.array([abs(x["trend"]) for x in recs if x["sym"] == sym])
        qreg[sym] = {q: float(np.quantile(tr, q)) for q in qs}
        qstr[sym] = {q: float(np.quantile(tr, q)) for q in qs}
        print(f"  {sym} |추세%| 분위: " +
              " ".join(f"q{int(q * 100)}={qreg[sym][q]:.3f}" for q in qs), flush=True)
        print(f"    라이브 상수: regime_q33={REGIME_Q33[sym]:.3f} "
              f"strong_q70={STRONG_Q70[sym]:.3f}  ← 7페어 시절 롤링값", flush=True)

    print("\n  [3-A] regime 문턱 스윕 (cond·nypm 은 현행 유지)", flush=True)
    reg_sweep = []
    for m in ["off", "live", *qs]:
        n_tried += 1
        rows = select(recs, regime=m, qreg=qreg)
        e = evaluate(rows)
        p, nd = diff_perm(base_rows, rows) if rows else (float("nan"), 0)
        nm = f"regime={m if isinstance(m, str) else 'q' + str(int(m * 100))}"
        reg_sweep.append((nm, m, e, p, nd))
        print(row_line(nm, e, base_eval, p, nd), flush=True)

    print("\n  [3-B] cond_align 강추세 문턱 스윕 (regime·nypm 현행 유지)", flush=True)
    cond_sweep = []
    for m in ["off", "live", *qs, "all"]:
        n_tried += 1
        rows = select(recs, cond=m, qstr=qstr)
        e = evaluate(rows)
        p, nd = diff_perm(base_rows, rows) if rows else (float("nan"), 0)
        nm = f"cond={m if isinstance(m, str) else 'q' + str(int(m * 100))}"
        cond_sweep.append((nm, m, e, p, nd))
        print(row_line(nm, e, base_eval, p, nd), flush=True)

    # ── 4단계: NY_PM 재확인 ───────────────────────────────────────────
    print("\n" + "=" * 100, flush=True)
    print("[4단계] NY_PM 재확인 — 2026-06-24 '승률 0~1%' 근거가 지금도 유효한가", flush=True)
    print("  (regime·cond 는 ON 인 상태에서 nypm 만 끈 모집단 = 시간 필터의 순수 대상)",
          flush=True)
    pool = select(recs, nypm=None)
    print(f"\n  {'UTC시':<6}{'KST시':<8}{'건수':>6}{'건당R':>9}{'승률':>7}{'R합':>9}"
          f"{'건당R 95%CI(부트)':>22}", flush=True)
    hour_tab = {}
    rngh = np.random.default_rng(7)
    for h in range(24):
        cell = [x["r"] for x in pool if x["hour"] == h]
        if not cell:
            continue
        a = np.array(cell)
        mark = "  ← 현행 차단" if NYPM_LIVE[0] <= h < NYPM_LIVE[1] else ""
        # 건당 R 의 부트스트랩 95% CI — 시간대별 셀이 작아 점추정만으로는 못 읽는다.
        if a.size >= 3:
            bs = np.array([rngh.choice(a, a.size, replace=True).mean()
                           for _ in range(4000)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
        else:
            lo = hi = float("nan")
        hour_tab[h] = dict(n=int(a.size), r_mean=float(a.mean()),
                           wr=float(100 * (a > 0).mean()), r_sum=float(a.sum()),
                           ci_lo=float(lo), ci_hi=float(hi))
        print(f"  {h:<6}{(h + 9) % 24:<8}{a.size:>6}{a.mean():>+9.3f}"
              f"{100 * (a > 0).mean():>6.0f}%{a.sum():>+9.2f}"
              f"   [{lo:+6.3f},{hi:+6.3f}]{mark}", flush=True)

    nypm_rows = [x for x in pool if NYPM_LIVE[0] <= x["hour"] < NYPM_LIVE[1]]
    other = [x for x in pool if not (NYPM_LIVE[0] <= x["hour"] < NYPM_LIVE[1])]
    if nypm_rows and other:
        a = np.array([x["r"] for x in nypm_rows])
        b = np.array([x["r"] for x in other])
        p = perm_p(a, b)
        print(f"\n  NY_PM({NYPM_LIVE[0]}-{NYPM_LIVE[1]} UTC) n={a.size} "
              f"건당R={a.mean():+.3f} 승률={100 * (a > 0).mean():.0f}%", flush=True)
        print(f"  그 외          n={b.size} 건당R={b.mean():+.3f} "
              f"승률={100 * (b > 0).mean():.0f}%", flush=True)
        print(f"  순열검정 p={p:.4f} → "
              f"{'유의' if p < 0.05 else '유의하지 않음 (차이는 우연과 구분 불가)'}",
              flush=True)

    print("\n  [4-B] 차단 시각 변형 스윕 (차단할 UTC 시각 집합)", flush=True)
    ny_variants = [
        ("nypm=없음", None),
        ("nypm=17,18,19,20 (현행)", frozenset({17, 18, 19, 20})),
        ("nypm=17,18", frozenset({17, 18})),
        ("nypm=19,20", frozenset({19, 20})),
        ("nypm=17,18,20", frozenset({17, 18, 20})),
        ("nypm=17,18,20,21", frozenset({17, 18, 20, 21})),
        ("nypm=16~21 전부", frozenset({16, 17, 18, 19, 20, 21})),
        ("nypm=13,17,18,20,21", frozenset({13, 17, 18, 20, 21})),
    ]
    ny_sweep = []
    for nm, win in ny_variants:
        n_tried += 1
        rows = select(recs, nypm=win)
        e = evaluate(rows)
        pp, nd = diff_perm(base_rows, rows) if rows else (float("nan"), 0)
        ny_sweep.append((nm, win, e, pp, nd))
        print(row_line(nm, e, base_eval, pp, nd), flush=True)

    # ── 5단계: 차단분 해부 (국면×방향 기저 통제) ──────────────────────
    print("\n" + "=" * 100, flush=True)
    print("[5단계] 차단분 해부 — 걸러진 게 실제로 나쁜가 / 국면×방향 기저 통제", flush=True)
    for gname, kw in (("regime", dict(regime="off")),
                      ("cond_align", dict(cond="off")),
                      ("nypm", dict(nypm=None))):
        off_rows = select(recs, **kw)
        kb = {(x["sym"], x["t_in"]) for x in base_rows}
        blocked = [x for x in off_rows if (x["sym"], x["t_in"]) not in kb]
        if not blocked:
            continue
        br = np.array([x["r"] for x in blocked])
        braw = np.array([x["raw"] for x in blocked])
        baseraw = np.array([x["raw"] for x in base_rows])
        p = perm_p(br, base_r)
        p_raw = perm_p(braw, baseraw)
        flag = "" if br.size >= MIN_CELL else " ⚠표본부족"
        print(f"\n  · {gname} 차단분 n={br.size}{flag} 건당R={br.mean():+.3f} "
              f"(통과분 {base_r.mean():+.3f}) 승률={100 * (br > 0).mean():.0f}% "
              f"R합={br.sum():+.2f} 순열 p={p:.4f}", flush=True)
        # R 과 raw(가격변동%)는 SL 폭이 거래마다 달라 순위가 어긋날 수 있다.
        # 복리 자산은 raw 로 굴러가므로 둘 다 본다(판정 기준 4번 R, 실제 손익 raw).
        print(f"      raw(가격변동) 기준: 차단 {100 * braw.mean():+.3f}% "
              f"vs 통과 {100 * baseraw.mean():+.3f}% 순열 p={p_raw:.4f}", flush=True)
        print("      국면×방향 기저 통제 (추세부호 × 진입방향):", flush=True)
        for tsign, tname in ((1, "상승국면"), (-1, "하락국면")):
            for d, dname in ((1, "롱"), (-1, "숏")):
                cb = [x["r"] for x in blocked
                      if np.sign(x["trend"]) == tsign and x["d"] == d]
                cp = [x["r"] for x in base_rows
                      if np.sign(x["trend"]) == tsign and x["d"] == d]
                if not cb:
                    continue
                cbm = float(np.mean(cb))
                cpm = float(np.mean(cp)) if cp else float("nan")
                warn = " ⚠표본부족" if len(cb) < MIN_CELL else ""
                print(f"        {tname}·{dname:<2} 차단 n={len(cb):3d} R={cbm:+6.3f} "
                      f"| 통과 n={len(cp):3d} R={cpm:+6.3f}{warn}", flush=True)
        yb: dict[int, float] = {}
        for x in blocked:
            yb[x["year"]] = yb.get(x["year"], 0.0) + x["r"]
        top = max(yb.items(), key=lambda kv: abs(kv[1])) if yb else (0, 0.0)
        share = abs(top[1]) / max(sum(abs(v) for v in yb.values()), 1e-9)
        print("      연도별 차단분 R합: " +
              " ".join(f"{k}:{v:+.1f}" for k, v in sorted(yb.items())), flush=True)
        print(f"      최대 연도 {top[0]} 가 총 |R| 의 {100 * share:.0f}% "
              f"→ {'⚠️ 몰빵(연도 일관성 실패)' if share > 0.5 else '분산됨'}", flush=True)

    # ── 산출물 저장 ───────────────────────────────────────────────────
    def pack(name, params, e, p, nd):
        return dict(name=name, params=params, n=e.get("n", 0),
                    r_mean=e.get("r_mean"), compound=e.get("compound"),
                    mdd=e.get("mdd"), ruin=e.get("ruin"), p_perm=None if p is None or
                    (isinstance(p, float) and np.isnan(p)) else float(p),
                    n_diff=nd, per_month=e.get("per_month"), wr=e.get("wr"),
                    boot_p50=e.get("boot_p50"), boot_p5=e.get("boot_p5"),
                    ypos=e.get("ypos"), ytot=e.get("ytot"), years=e.get("years"))

    cands = []
    for label, kw, e, p, nd in combos:
        cands.append(pack(f"조합 ON={label}",
                          {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in kw.items()}, e, p, nd))
    for nm, m, e, p, nd in reg_sweep:
        cands.append(pack(nm, dict(regime=m), e, p, nd))
    for nm, m, e, p, nd in cond_sweep:
        cands.append(pack(nm, dict(cond=m), e, p, nd))
    for nm, w, e, p, nd in ny_sweep:
        cands.append(pack(nm, dict(nypm=sorted(w) if w else None), e, p, nd))

    # ── 다중비교 보정 판정 ────────────────────────────────────────────
    print("\n" + "=" * 100, flush=True)
    print(f"[다중비교] 시도한 조합 {n_tried}개 → Bonferroni 문턱 "
          f"a=0.05/{n_tried}={0.05 / n_tried:.5f}", flush=True)
    surv = [c for c in cands
            if c["p_perm"] is not None and c["p_perm"] < 0.05 / n_tried]
    raw05 = [c for c in cands if c["p_perm"] is not None and c["p_perm"] < 0.05]
    print(f"  무보정 p<0.05 통과: {len(raw05)}개 "
          f"(우연 기대치 {0.05 * n_tried:.1f}개)", flush=True)
    for c in raw05:
        print(f"    · {c['name']:<26} p={c['p_perm']:.4f} Δn={c['n_diff']} "
              f"n={c['n']} R={c['r_mean']:+.3f} 복리={c['compound']:.2f}x", flush=True)
    print(f"  Bonferroni 통과: {len(surv)}개"
          f"{'' if surv else '  → 살아남는 후보 없음'}", flush=True)

    # ── 6단계: 상위 후보 사후검증 (연도 일관성 + 잭나이프) ────────────
    print("\n" + "=" * 100, flush=True)
    print("[6단계] 상위 후보 사후검증 — 복리 상위 후보가 몇 건/몇 년에 기대고 있나",
          flush=True)
    shortlist = [
        ("nypm OFF (regime+cond ON)", dict(nypm=None)),
        ("nypm=17,18,20 (19시 허용)", dict(nypm=frozenset({17, 18, 20}))),
        ("nypm=17,18,20,21", dict(nypm=frozenset({17, 18, 20, 21}))),
        ("nypm=13,17,18,20,21", dict(nypm=frozenset({13, 17, 18, 20, 21}))),
        ("cond=q70 (재추정)", dict(cond=0.7)),
        ("regime=q70 (재추정)", dict(regime=0.7)),
        ("regime OFF", dict(regime="off")),
    ]
    kbase = {(x["sym"], x["t_in"]) for x in base_rows}
    post = {}
    for nm, kw in shortlist:
        rows = select(recs, qreg=qreg, qstr=qstr, **kw)
        e = evaluate(rows)
        if e.get("n", 0) < 2:
            continue
        kc = {(x["sym"], x["t_in"]) for x in rows}
        allr = {(x["sym"], x["t_in"]): x for x in recs}
        diff = (kbase | kc) - (kbase & kc)
        drows = [allr[k] for k in diff]
        yd: dict[int, float] = {}
        for x in drows:
            yd[x["year"]] = yd.get(x["year"], 0.0) + x["r"]
        tot = sum(abs(v) for v in yd.values())
        top = max(yd.items(), key=lambda kv: abs(kv[1])) if yd else (0, 0.0)
        share = abs(top[1]) / max(tot, 1e-9)
        # 잭나이프 — 차이집합에서 |R| 최대 1건을 빼면 복리가 얼마나 무너지나
        if drows:
            worst = max(drows, key=lambda x: abs(x["r"]))
            rows_j = [x for x in rows
                      if not (x["sym"] == worst["sym"] and x["t_in"] == worst["t_in"])]
            ej = evaluate(rows_j)
        else:
            ej = e
        post[nm] = dict(compound=e["compound"], boot_p50=e["boot_p50"],
                        boot_p5=e["boot_p5"], mdd=e["mdd"],
                        top_year=str(top[0]), top_year_share=float(share),
                        jack_compound=float(ej.get("compound", 0.0)),
                        n_diff=len(drows))
        print(f"\n  · {nm}", flush=True)
        print(f"      복리 {e['compound']:.2f}x (기준선 {base_eval['compound']:.2f}x) "
              f"부트중앙 {e['boot_p50']:.2f}x 5%분위 {e['boot_p5']:.2f}x "
              f"MDD {e['mdd']:.1f}%", flush=True)
        print(f"      차이집합 {len(drows)}건 연도별 R: " +
              " ".join(f"{k}:{v:+.1f}" for k, v in sorted(yd.items())), flush=True)
        print(f"      최대 연도 {top[0]} 가 총 |R| 의 {100 * share:.0f}% → "
              f"{'⚠️ 몰빵(연도 일관성 실패)' if share > 0.5 else '분산'}", flush=True)
        print(f"      차이집합 최대 1건 제거 시 복리 {e['compound']:.2f}x → "
              f"{ej.get('compound', 0):.2f}x "
              f"({'⚠️ 단일 거래 의존' if ej.get('compound', 0) < base_eval['compound'] else '유지'})",
              flush=True)
    payload_post = post

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    payload = dict(
        axis="게이트 3종 재판정 (regime q33 / cond_align q70 / NY_PM 제외) — BTC+ETH 2페어",
        baseline=pack("현행 (전부 ON)",
                      dict(regime="live", cond="live", nypm=list(NYPM_LIVE)),
                      base_eval, None, 0),
        ceilings=ceilings,
        hour_table={str(k): v for k, v in hour_tab.items()},
        candidates=cands,
        postcheck=payload_post,
        n_tried=n_tried,
        bonferroni_alpha=0.05 / n_tried,
        n_pass_raw05=len(raw05),
        n_pass_bonferroni=len(surv),
        notes=("BTC+ETH 5년(2021-07~2026-06) · 라이브 정합 하니스(live_parity) + "
               f"HTF FVG flip {LIVE_FLIP_MIN_R}R 사후적용 · 복리는 레버 {LEV}x·size {SIZE}"
               "·동시보유 자본분할·DD스로틀(-25%×0.7)·파산 시드20% · "
               f"순열 {N_PERM}회 양측 · 부트스트랩 {N_BOOT}회 복원추출 · "
               f"시도 조합 {n_tried}개(다중비교 무보정) · "
               "한계: 사후필터라 대체효과 미반영, ote_up_level/sweep_gate/smart_size/"
               "daily_loss_limit/MMBM 은 replay 미구현"),
    )
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    print(f"\n\n저장: {OUT_JSON}  (후보 {len(cands)}개 / 시도 조합 {n_tried}개)",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
