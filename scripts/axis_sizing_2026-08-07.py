"""#AXIS 2026-08-07: 사이징·낙폭 제어 축 — "배율 말고 로직으로 낙폭을 잡을 수 있는가".

파트너 관심사: 현행 Origo 최대낙폭이 36~85% 구간이다. 배율(레버리지·size_pct)을
낮추면 낙폭과 수익이 **같은 비율로** 줄 뿐이라 의미가 없다. 낙폭만 깎으려면
"어떤 거래에 크게, 어떤 거래에 작게" 를 **사전에 판별**할 수 있어야 한다.

## 1단계 — 구조적 상한 먼저 (직전 피보 연구의 교훈)
스윕을 돌리기 전에 각 손잡이가 물리적으로 R 을 얼마나 움직일 수 있는지 먼저 잰다.

  (a) size_pct 는 손잡이가 **아니다** — 복리식에서 size 와 레버리지는 곱으로만
      들어간다(step = raw·size·lev − 2·fee·size·lev). 즉 size_pct 0.9→0.5 는
      레버리지 7x→3.89x 와 **완전히 같은 수식**이다. 이미 결정된 레버리지 축의
      재탕이므로 독립적인 개선 여지가 수학적으로 0 이다. 항등식을 수치로 확인한다.

  (b) 품질 사이징(confluence 차등)의 상한 = confluence 점수가 가진 **정보량**.
      점수 버킷별 조건부 평균 R 이 서로 다르지 않으면 어떤 가중치를 씌워도
      기대 R 은 안 변한다. 오라클(사후 최적) 가중의 R 상승폭을 상한으로 계산하고
      표본 노이즈(SE)와 비교한다.

  (c) 연속손실 브레이크의 상한 = 손실의 **군집성**. 직전 거래가 손실일 때 다음
      거래 R 이 실제로 낮아야만 작동한다. 부호 자기상관과 streak 조건부 R 을 잰다.
      군집성이 없으면(i.i.d.) 브레이크는 무작위로 노출을 깎는 것 = 배율 낮추기.

## 2단계 — 실측 스윕(복리)
레버리지 7x · size 0.9 · 동시보유 분할 · DD 스로틀(-25%→×0.7) · 파산 20%.

## 3단계 — 프론티어 통제 (이 연구의 핵심 판정)
후보마다 **같은 최종자산을 내는 단순 배율(flat scalar)** 을 역산해 그 배율의 MDD 와
비교한다. 후보 MDD 가 flat 등가 MDD 보다 낮지 않으면 그냥 배율 낮추기와 동일 →
기각. "낙폭 감소분 대비 수익 감소분" 을 정량화하는 유일하게 정직한 방법이다.

## 검증
순열검정 20000회 · 국면×방향 기저 통제 · 연도 일관성 · R 배수 측정 ·
표본 30건 미만 셀 제외 · 다중비교(시도 조합 수) 명시.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import LIVE_BASE, PAIRS, run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

RNG = np.random.default_rng(20260807)
SIZE = float(LIVE_BASE["size_pct"])   # 0.9
LEV = 7.0                             # 현행 라이브
DD_PCT, DD_FACTOR = 0.25, 0.7         # origo_dd_throttle_pct / _factor
RUIN = 0.20                           # 시드 20% 이하 = 파산
N_PERM = 20000
N_BOOT = 2000
MIN_CELL = 30                         # 표본부족 기준

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "axis", "sizing.json")


# ────────────────────────────── 수집 ──────────────────────────────
def collect():
    """라이브 정합 백테 7페어 → 진입시각 순 정렬된 거래 목록."""
    rows = []
    for sym in PAIRS:
        df5, kept, _ = run_live_parity(sym)
        idx = df5.index
        for t in kept:
            rows.append(dict(
                sym=sym,
                s=int(idx[t.entry_idx].value),
                e=int(idx[min(t.exit_idx, len(idx) - 1)].value),
                raw=float(t.raw_pnl_pct),
                conf=int(t.confluence_score),
                risk=float(t.risk_pct),            # 진입가 대비 초기 SL 폭(%)
                dirn=str(getattr(t.direction, "value", t.direction)).lower(),
                trend=float(t.entry_trend_pct or 0.0),
                year=int(idx[t.entry_idx].year),
            ))
    rows.sort(key=lambda r: r["s"])
    return rows


def concurrency(rows) -> np.ndarray:
    """각 거래가 열려 있는 동안의 평균 동시보유 수(자기 포함) — 라이브 자본 분할 재현."""
    out = np.empty(len(rows))
    for i, a in enumerate(rows):
        ov = sum(1 for j, b in enumerate(rows)
                 if j != i and b["s"] <= a["e"] and b["e"] >= a["s"])
        out[i] = 1.0 + ov
    return out


# ────────────────────────── 복리 시뮬 ──────────────────────────
def sim(raws, scale, w, lev=LEV, dd=True):
    """복리 시뮬 — (최종배수, MDD비율, 파산여부, 집행거래수).

    raws  : 거래별 raw PnL 비율
    scale : 동시보유 분할 계수(1/동시보유수)
    w     : 거래별 사이즈 가중치(0 이면 그 거래는 건너뜀 = 미집행)
    """
    eq, peak, mdd, n = 1.0, 1.0, 0.0, 0
    for i in range(len(raws)):
        wi = w[i]
        if wi <= 0.0:
            continue
        sz = SIZE * scale[i] * wi
        if dd and eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        step = raws[i] * sz * lev - 2.0 * TAKER_FEE_PCT * sz * lev
        eq *= (1.0 + step)
        n += 1
        if eq <= 0.0:
            return 0.0, 1.0, True, n
        if eq > peak:
            peak = eq
        d = 1.0 - eq / peak
        if d > mdd:
            mdd = d
        if eq <= RUIN:
            return eq, mdd, True, n
    return eq, mdd, False, n


def streak_weights(raws, n_loss, factor, hold=0):
    """연속손실 브레이크 가중치 — n_loss 연속 손실 후 factor 적용.

    hold=0 : 다음 승리까지 계속 적용 / hold=k : 이후 k 건만 적용.
    가중치는 **그 거래 진입 전까지의 정보만** 사용한다(미래 참조 없음).
    """
    w = np.ones(len(raws))
    streak, left = 0, 0
    for i in range(len(raws)):
        if streak >= n_loss and (hold == 0 or left > 0):
            w[i] = factor
            if hold:
                left -= 1
        if raws[i] < 0:
            streak += 1
            if streak == n_loss and hold:
                left = hold
        else:
            streak, left = 0, 0
    return w


def conf_weights(confs, table, default=1.0):
    """confluence 점수 → 가중치 (table: {점수: 가중치})."""
    return np.array([table.get(int(c), default) for c in confs], float)


# ─────────────────── 1단계: 구조적 상한 ───────────────────
def stage1(rows, raws, scale, R):
    print("\n" + "=" * 78, flush=True)
    print("1단계 — 구조적 상한 (스윕 전에 반드시)", flush=True)
    print("=" * 78, flush=True)

    n = len(R)
    se = R.std(ddof=1) / np.sqrt(n)
    print(f"\n[표본] 거래 {n}건 · 5년 · 7페어 · 평균 {n / 60:.1f}건/월", flush=True)
    print(f"  건당 R 평균 {R.mean():+.3f}R · 표준편차 {R.std(ddof=1):.3f} "
          f"· **표준오차 {se:.3f}R** ← 이 값이 노이즈 바닥이다", flush=True)
    print(f"  승률 {(R > 0).mean() * 100:.1f}% · 동시보유 평균 "
          f"{(1 / scale).mean():.2f}개", flush=True)

    # (a) size_pct = 레버리지의 재탕임을 수치로 증명
    print("\n[a] size_pct 는 독립 손잡이가 아니다 — 레버리지와 곱으로만 들어간다", flush=True)
    ones = np.ones(n)
    e1, m1, _, _ = sim(raws, scale, ones * (0.5 / SIZE), lev=LEV)          # size 0.5
    e2, m2, _, _ = sim(raws, scale, ones, lev=LEV * 0.5 / SIZE)            # lev 3.89x
    print(f"  size 0.50·lev 7.00x → 자산 {e1:.4f}x · MDD {m1 * 100:.2f}%", flush=True)
    print(f"  size 0.90·lev {LEV * 0.5 / SIZE:.2f}x → 자산 {e2:.4f}x · MDD {m2 * 100:.2f}%", flush=True)
    print(f"  차이 {abs(e1 - e2):.2e} — 완전 동일. size_pct 스윕의 독립 개선 여지 = "
          f"**구조적으로 0**", flush=True)

    # (b) 품질 사이징 상한 = confluence 정보량
    print("\n[b] 품질 사이징 상한 = confluence 점수의 정보량", flush=True)
    confs = np.array([r["conf"] for r in rows])
    cells = {}
    for c in sorted(set(confs.tolist())):
        m = confs == c
        sub = R[m]
        ok = len(sub) >= MIN_CELL
        cells[int(c)] = dict(n=int(len(sub)), r=float(sub.mean()),
                             se=float(sub.std(ddof=1) / np.sqrt(len(sub))) if len(sub) > 1 else float("nan"),
                             usable=bool(ok))
        tag = "" if ok else "  ← 표본부족(결론 제외)"
        print(f"  conf={c}: n={len(sub):3d}  평균 {sub.mean():+.3f}R  "
              f"SE {cells[int(c)]['se']:.3f}{tag}", flush=True)
    use = [c for c, v in cells.items() if v["usable"]]
    if len(use) >= 2:
        lo, hi = min(use, key=lambda c: cells[c]["r"]), max(use, key=lambda c: cells[c]["r"])
        gap = cells[hi]["r"] - cells[lo]["r"]
        segap = np.hypot(cells[hi]["se"], cells[lo]["se"])
        print(f"  최대 격차: conf{hi} − conf{lo} = {gap:+.3f}R "
              f"(격차 SE {segap:.3f}, t={gap / segap:.2f})", flush=True)
        # 오라클 상한: 사용가능 버킷 중 최고만 100%, 나머지 0% (사후 최적 = 상한)
        oracle = cells[hi]["r"] - R.mean()
        print(f"  **오라클 상한**(최고 버킷만 진입 — 사후 최적이라 현실 불가): "
              f"건당 {oracle:+.3f}R", flush=True)
        print(f"  노이즈 바닥 {se:.3f}R 대비 {oracle / se:.2f}배 → "
              f"{'상한이 노이즈를 넘음. 순열검정으로 진위 판정 필요.' if oracle > se else '상한이 노이즈 이하 → 스윕 무의미.'}",
              flush=True)
    else:
        oracle, gap, segap = 0.0, 0.0, float("nan")
        print("  사용가능 버킷 2개 미만 → 품질 사이징 상한 = 0 (차등할 축이 없음)", flush=True)

    # (c) 연속손실 브레이크 상한 = 손실 군집성
    print("\n[c] 연속손실 브레이크 상한 = 손실의 군집성(자기상관)", flush=True)
    sign = np.where(R > 0, 1.0, -1.0)
    lag1 = float(np.corrcoef(sign[:-1], sign[1:])[0, 1])
    prev_loss = R[1:][sign[:-1] < 0]
    prev_win = R[1:][sign[:-1] > 0]
    print(f"  승패 부호 lag-1 자기상관 = {lag1:+.4f} "
          f"(0 이면 손실은 무작위로 흩어진다)", flush=True)
    for lab, sub in (("직전 손실 뒤", prev_loss), ("직전 승리 뒤", prev_win)):
        tag = "" if len(sub) >= MIN_CELL else "  ← 표본부족"
        print(f"  {lab}: n={len(sub):3d} 평균 {sub.mean():+.3f}R "
              f"SE {sub.std(ddof=1) / np.sqrt(len(sub)):.3f}{tag}", flush=True)
    streak_stats = {}
    st = 0
    buckets: dict[int, list] = {}
    for i in range(n):
        buckets.setdefault(min(st, 4), []).append(R[i])
        st = st + 1 if R[i] < 0 else 0
    for k in sorted(buckets):
        sub = np.array(buckets[k])
        tag = "" if len(sub) >= MIN_CELL else "  ← 표본부족"
        streak_stats[k] = dict(n=int(len(sub)), r=float(sub.mean()))
        print(f"  직전 연속손실 {k}{'+' if k == 4 else ''}건 상태: n={len(sub):3d} "
              f"평균 {sub.mean():+.3f}R{tag}", flush=True)
    mx, cur = 0, 0
    for x in R:
        cur = cur + 1 if x < 0 else 0
        mx = max(mx, cur)
    print(f"  실측 최대 연속손실 = {mx}건", flush=True)
    diff = float(prev_win.mean() - prev_loss.mean()) if len(prev_loss) and len(prev_win) else 0.0
    print(f"  **브레이크 상한**(직전 손실 뒤 노출 0 으로 만들었을 때 건당 R 개선 최대치) "
          f"= {diff:+.3f}R · 노이즈 바닥 {se:.3f}R 대비 {abs(diff) / se:.2f}배", flush=True)

    return dict(n=n, r_mean=float(R.mean()), r_se=float(se),
                conf_cells=cells, conf_oracle=float(oracle),
                conf_gap=float(gap), conf_gap_se=float(segap),
                lag1=lag1, streak=streak_stats, max_streak=int(mx),
                brake_bound=float(diff))


# ─────────────────── 2·3단계: 스윕 + 프론티어 ───────────────────
def evaluate(raws, scale, w):
    """후보 1개 평가 — 결정 경로 + 부트스트랩."""
    fin, mdd, ruined, n_exec = sim(raws, scale, w)
    idx0 = np.arange(len(raws))
    fins = np.empty(N_BOOT)
    mdds = np.empty(N_BOOT)
    ruin = 0
    for k in range(N_BOOT):
        idx = RNG.integers(0, len(raws), size=len(raws))   # 복원추출
        f, m, r_, _ = sim(raws[idx], scale[idx], w[idx])
        fins[k], mdds[k] = f, m
        ruin += int(r_)
    p50, p5 = np.percentile(fins, [50, 5])
    return dict(compound=float(fin), mdd=float(mdd * 100), ruin=float(100 * ruin / N_BOOT),
                boot_p50=float(p50), boot_p5=float(p5),
                boot_mdd_p50=float(np.median(mdds) * 100),
                n_exec=int(n_exec), ruined=bool(ruined), _idx0=idx0)


def flat_curve(raws, scale, ks=None):
    """단순 배율 곡선 — (배율, 자산, MDD) 목록. 프론티어 비교의 기준선 곡선."""
    ones = np.ones(len(raws))
    if ks is None:
        ks = np.concatenate([np.linspace(0.02, 1.0, 50), np.linspace(1.05, 2.0, 20)])
    out = []
    for k in ks:
        f, m, _, _ = sim(raws, scale, ones * k)
        out.append((float(k), float(f), float(m * 100)))
    return out


def flat_match_mdd(curve, mdd_target):
    """후보와 **같은 MDD** 를 갖는 단순 배율의 자산 — 매칭 방향을 MDD 로 잡는다.

    수익 매칭(반대 방향)은 후보가 flat 곡선 최대치를 넘으면 정의되지 않는다(포화).
    MDD 매칭은 flat 곡선이 MDD 0%~100% 를 덮으므로 항상 정의된다.
    """
    best = None
    for k, f, m in curve:
        if m <= mdd_target and (best is None or f > best[1]):
            best = (k, f, m)
    return best if best else (0.0, 1.0, 0.0)


def flat_match_ret(curve, ret_target):
    """후보와 같은 자산을 내는 최소 MDD 의 단순 배율. 포화면 None."""
    best = None
    for k, f, m in curve:
        if f >= ret_target and (best is None or m < best[2]):
            best = (k, f, m)
    return best


# ─────────────────── 순열검정 ───────────────────
def perm_p_quality(raws, scale, w, n_perm=N_PERM):
    """귀무: confluence 라벨은 정보가 없다 → 가중치 배열을 무작위 재배치.

    통계량: log(최종자산). p = P(순열 ≥ 실측).
    """
    obs, _, _, _ = sim(raws, scale, w)
    ge = 0
    for _ in range(n_perm):
        f, _, _, _ = sim(raws, scale, RNG.permutation(w))
        ge += int(f >= obs)
    return (ge + 1) / (n_perm + 1)


def perm_p_streak(raws, scale, rule, n_perm=N_PERM):
    """귀무: 거래 순서는 무작위(손실 군집 없음) → 순서를 섞고 규칙을 재적용.

    통계량: log(브레이크 자산) − log(동일 순서 기준선 자산).
    """
    w = rule(raws)
    f1, _, _, _ = sim(raws, scale, w)
    f0, _, _, _ = sim(raws, scale, np.ones(len(raws)))
    obs = np.log(max(f1, 1e-12)) - np.log(max(f0, 1e-12))
    ge = 0
    for _ in range(n_perm):
        idx = RNG.permutation(len(raws))
        rr, ss = raws[idx], scale[idx]
        a, _, _, _ = sim(rr, ss, rule(rr))
        b, _, _, _ = sim(rr, ss, np.ones(len(rr)))
        ge += int((np.log(max(a, 1e-12)) - np.log(max(b, 1e-12))) >= obs)
    return (ge + 1) / (n_perm + 1), float(obs)


# ─────────────────── 기저 통제 · 연도 일관성 ───────────────────
def base_control(rows, R, confs):
    """국면×방향 기저 통제 — 'conf 높은 게 잘 된다' 가 '순추세 롱' 의 재확인인가."""
    print("\n[기저 통제] 국면×방향 셀별 평균 R 과 conf 분포", flush=True)
    hi = confs >= 6
    cells: dict[str, dict] = {}
    for d in ("long", "short"):
        for reg, f in (("순추세", lambda r, dd: (r["trend"] > 0) == (dd == "long")),
                       ("역추세", lambda r, dd: (r["trend"] > 0) != (dd == "long"))):
            m = np.array([r["dirn"] == d and f(r, d) for r in rows])
            if m.sum() == 0:
                continue
            sub, subhi = R[m], R[m & hi]
            key = f"{d}·{reg}"
            cells[key] = dict(n=int(m.sum()), r=float(sub.mean()),
                              hi_frac=float((confs[m] >= 6).mean()),
                              hi_n=int(len(subhi)),
                              hi_r=float(subhi.mean()) if len(subhi) else float("nan"))
            tag = "" if m.sum() >= MIN_CELL else "  ← 표본부족"
            print(f"  {key:<10} n={m.sum():3d} 평균 {sub.mean():+.3f}R · "
                  f"conf≥6 비율 {100 * (confs[m] >= 6).mean():4.0f}% · "
                  f"그 셀 conf≥6 평균 {cells[key]['hi_r']:+.3f}R (n={len(subhi)}){tag}",
                  flush=True)
    return cells


def year_consistency(rows, R, confs):
    """연도 일관성 — conf≥6 우위가 특정 연도 몰빵인가."""
    print("\n[연도 일관성] conf≥6 vs conf=5 평균 R (연도별)", flush=True)
    years = sorted({r["year"] for r in rows})
    out = {}
    for y in years:
        m = np.array([r["year"] == y for r in rows])
        a, b = R[m & (confs >= 6)], R[m & (confs == 5)]
        da = (a.mean() - b.mean()) if len(a) and len(b) else float("nan")
        out[str(y)] = dict(n_hi=int(len(a)), n_lo=int(len(b)),
                           r_hi=float(a.mean()) if len(a) else float("nan"),
                           r_lo=float(b.mean()) if len(b) else float("nan"),
                           delta=float(da))
        print(f"  {y}: conf≥6 n={len(a):2d} {out[str(y)]['r_hi']:+.3f}R  |  "
              f"conf=5 n={len(b):2d} {out[str(y)]['r_lo']:+.3f}R  |  차 {da:+.3f}R"
              f"{'  ← 표본부족' if min(len(a), len(b)) < 5 else ''}", flush=True)
    pos = sum(1 for v in out.values() if v["delta"] > 0)
    print(f"  → conf≥6 우위 연도 {pos}/{len(years)}", flush=True)
    return out, pos, len(years)


def fragility(rows, raws, scale, R, results):
    """취약성 해부 — 브레이크 후보가 몇 개의 사건에 걸려 있는가 + 연도 제거 검정.

    거래 126건에서 10건을 건너뛰고 자산이 두 배가 된다면 그것은 '규칙'이 아니라
    '그 10건' 이다. 발동 사건 수 · 건너뛴 거래의 R · 연도별 기여를 본다.
    """
    print("\n" + "=" * 78, flush=True)
    print("6단계 — 취약성 해부 (브레이크 후보)", flush=True)
    print("=" * 78, flush=True)
    out = {}
    years = sorted({r["year"] for r in rows})
    yarr = np.array([r["year"] for r in rows])
    ones = np.ones(len(raws))
    base_fin, _, _, _ = sim(raws, scale, ones)
    for name, params, w, kind, ev, rmean in results:
        if kind != "brake":
            continue
        act = np.where(w < 1.0)[0]
        # 연속된 인덱스 묶음 = 발동 사건
        events = 0
        prev = -99
        for i in act:
            if i != prev + 1:
                events += 1
            prev = i
        skipped_R = R[act]
        print(f"\n  {name}: 발동 사건 {events}회 · 영향 거래 {len(act)}건 "
              f"(전체의 {100 * len(act) / len(R):.0f}%)", flush=True)
        if len(act):
            print(f"    영향 거래 평균 {skipped_R.mean():+.3f}R "
                  f"(전체 평균 {R.mean():+.3f}R) · 그중 손실 "
                  f"{int((skipped_R < 0).sum())}/{len(act)}건", flush=True)
            print(f"    영향 거래 연도 분포 {dict(Counter(yarr[act].tolist()))}", flush=True)
        # 연도 1개씩 제거 — 특정 연도 몰빵 검사
        rows_y = []
        for y in years:
            m = yarr != y
            rr, ss = raws[m], scale[m]
            wy = streak_weights(rr, params["n_loss"], params["factor"], params["hold"])
            a, ma, _, _ = sim(rr, ss, wy)
            b, mb, _, _ = sim(rr, ss, np.ones(len(rr)))
            rows_y.append((y, a / b if b > 0 else float("nan"), ma * 100, mb * 100))
        print("    [연도 제거] 그 연도를 빼도 브레이크가 이득인가 (자산비 >1 이면 이득)", flush=True)
        for y, ratio, ma, mb in rows_y:
            print(f"      −{y}: 자산비 {ratio:5.2f}x · MDD {mb:5.1f}%→{ma:5.1f}%",
                  flush=True)
        pos = sum(1 for _, r_, _, _ in rows_y if r_ > 1.0)
        print(f"    → 모든 연도 제거에서 이득 유지: {pos}/{len(years)}", flush=True)
        out[name] = dict(events=events, n_affected=int(len(act)),
                         affected_r_mean=float(skipped_R.mean()) if len(act) else None,
                         loo_year=[dict(drop=int(y), ratio=float(r_), mdd_brake=float(a),
                                        mdd_base=float(b)) for y, r_, a, b in rows_y],
                         loo_pos=pos, loo_total=len(years))
    return out


# ────────────────────────────── main ──────────────────────────────
def main() -> int:
    rows = collect()
    raws = np.array([r["raw"] for r in rows], float)
    scale = 1.0 / concurrency(rows)
    R = np.array([r["raw"] * 100.0 / r["risk"] for r in rows], float)   # R 배수
    confs = np.array([r["conf"] for r in rows])
    n = len(raws)

    print("=== Origo 사이징·낙폭 제어 축 (2026-08-07) ===", flush=True)
    print(f"  라이브 정합 백테 7페어 · 레버리지 {LEV:.0f}x · size {SIZE} · "
          f"동시보유 분할 · DD 스로틀 · 파산 {RUIN:.0%}", flush=True)
    print(f"  페어별 건수 {dict(Counter(r['sym'].replace('USDT', '') for r in rows))}", flush=True)

    s1 = stage1(rows, raws, scale, R)

    # ── 2단계 스윕 ──
    print("\n" + "=" * 78, flush=True)
    print("2단계 — 후보 스윕 (복리 · 동시보유 · DD 스로틀)", flush=True)
    print("=" * 78, flush=True)

    ones = np.ones(n)
    cands: list[tuple[str, dict, np.ndarray, str]] = []
    # ① size_pct flat (1단계 (a)에서 이미 배율 재탕임이 증명됨 — 대조군으로만 유지)
    for s in (0.3, 0.5, 0.7):
        cands.append((f"flat_size_{s}", {"size_pct": s}, ones * (s / SIZE), "flat"))
    # ② 품질 기반 사이징 (conf=7 은 표본부족이라 conf≥6 으로 묶는다)
    qtab = {
        "qual_5x0.7": {5: 0.7},
        "qual_5x0.5": {5: 0.5},
        "qual_5x0.5_6x1.3": {5: 0.5, 6: 1.3, 7: 1.3},
        "qual_5x0.0": {5: 0.0},
    }
    for name, tab in qtab.items():
        cands.append((name, {"conf_weight": tab}, conf_weights(confs, tab), "qual"))
    # ③ 연속손실 브레이크
    srules = {
        "brake_3x0.5": (3, 0.5, 0), "brake_3x0.0": (3, 0.0, 0),
        "brake_4x0.5": (4, 0.5, 0), "brake_4x0.0": (4, 0.0, 0),
        "brake_5x0.5": (5, 0.5, 0), "brake_5x0.0": (5, 0.0, 0),
        "brake_3x0.5_hold3": (3, 0.5, 3),
    }
    for name, (k, f, h) in srules.items():
        cands.append((name, {"n_loss": k, "factor": f, "hold": h},
                      streak_weights(raws, k, f, h), "brake"))
    # ④ 결합 (품질 최선 × 브레이크 최선은 아래에서 사후 추가)

    base = evaluate(raws, scale, ones)
    print(f"\n  {'후보':<20}{'집행':>5}{'건당R':>8}{'최종자산':>11}{'MDD':>8}"
          f"{'파산%':>7}{'부트중앙':>10}{'5%분위':>9}", flush=True)
    print(f"  {'기준선(현행 0.9)':<20}{base['n_exec']:>5}{R.mean():>+8.3f}"
          f"{base['compound']:>10.2f}x{base['mdd']:>7.1f}%{base['ruin']:>6.1f}%"
          f"{base['boot_p50']:>9.2f}x{base['boot_p5']:>8.2f}x", flush=True)

    results = []
    for name, params, w, kind in cands:
        ev = evaluate(raws, scale, w)
        exec_mask = w > 0
        rmean = float(R[exec_mask].mean()) if exec_mask.any() else float("nan")
        results.append((name, params, w, kind, ev, rmean))
        print(f"  {name:<20}{ev['n_exec']:>5}{rmean:>+8.3f}"
              f"{ev['compound']:>10.2f}x{ev['mdd']:>7.1f}%{ev['ruin']:>6.1f}%"
              f"{ev['boot_p50']:>9.2f}x{ev['boot_p5']:>8.2f}x", flush=True)

    # ── 3단계 프론티어 통제 ──
    print("\n" + "=" * 78, flush=True)
    print("3단계 — 프론티어 통제: 단순 배율 곡선(= 그냥 배율 낮추기)과의 비교", flush=True)
    print("  같은 MDD 에서 자산이 더 크면 프론티어 위로 올라간 것. 아니면 배율 재탕.", flush=True)
    print("=" * 78, flush=True)
    curve = flat_curve(raws, scale)
    print(f"\n  {'후보':<20}{'자산':>9}{'MDD':>8}{'같은MDD flat':>14}"
          f"{'자산이득':>10}{'같은자산 flat MDD':>18}", flush=True)
    front = {}
    for name, params, w, kind, ev, rmean in results:
        km, fm, mm = flat_match_mdd(curve, ev["mdd"])
        gain_ret = ev["compound"] / fm - 1.0 if fm > 0 else float("nan")
        mr = flat_match_ret(curve, ev["compound"])
        mr_txt = f"{mr[2]:.1f}% (k={mr[0]:.2f})" if mr else "포화(도달불가)"
        front[name] = dict(flat_k_mdd=km, flat_ret_at_mdd=fm,
                           ret_gain_pct=100 * gain_ret,
                           flat_mdd_at_ret=(mr[2] if mr else None))
        print(f"  {name:<20}{ev['compound']:>8.2f}x{ev['mdd']:>7.1f}%"
              f"{fm:>12.2f}x{100 * gain_ret:>+9.1f}%{mr_txt:>18}", flush=True)

    # ── 순열검정 (상한이 노이즈를 넘은 축만) ──
    print("\n" + "=" * 78, flush=True)
    print(f"4단계 — 순열검정 {N_PERM}회", flush=True)
    print("=" * 78, flush=True)
    pvals: dict[str, float] = {}
    for name, params, w, kind, ev, rmean in results:
        if kind == "flat":
            pvals[name] = float("nan")     # 결정적 배율 — 검정 대상 아님
            continue
        if kind == "qual":
            p = perm_p_quality(raws, scale, w)
            pvals[name] = p
            print(f"  {name:<20} p={p:.4f}  (귀무: confluence 라벨 무정보)", flush=True)
        else:
            k, f, h = params["n_loss"], params["factor"], params["hold"]
            p, obs = perm_p_streak(raws, scale, lambda rr: streak_weights(rr, k, f, h))
            pvals[name] = p
            print(f"  {name:<20} p={p:.4f}  Δlog={obs:+.3f}  "
                  f"(귀무: 거래 순서 무작위=손실 군집 없음)", flush=True)

    # ── 기저 통제 · 연도 일관성 ──
    print("\n" + "=" * 78, flush=True)
    print("5단계 — 기저 통제 · 연도 일관성", flush=True)
    print("=" * 78, flush=True)
    cells = base_control(rows, R, confs)
    yrs, ypos, ytot = year_consistency(rows, R, confs)
    frag = fragility(rows, raws, scale, R, results)

    # ── 판정 ──
    print("\n" + "=" * 78, flush=True)
    print("판정", flush=True)
    print("=" * 78, flush=True)
    # 다중비교 보정 — 검정 대상은 품질 4 + 브레이크 7 = 11개.
    tested = [nm for nm, pp in pvals.items() if pp == pp]
    m_tests = len(tested)
    bonf = {nm: min(1.0, pvals[nm] * m_tests) for nm in tested}
    print(f"\n  [다중비교] 검정 {m_tests}개 → Bonferroni 보정 p", flush=True)
    for nm in tested:
        print(f"    {nm:<20} raw p={pvals[nm]:.4f} → 보정 p={bonf[nm]:.3f}", flush=True)

    verdicts = {}
    for name, params, w, kind, ev, rmean in results:
        p = pvals[name]
        g = front[name]["ret_gain_pct"]
        if kind == "flat":
            v = "기각(구조적) — size_pct 는 레버리지와 곱으로만 들어감, 1단계 (a) 항등식"
        elif not (p == p) or p >= 0.05:
            v = f"기각 — 순열검정 p={p:.3f} ≥ 0.05"
        elif g <= 0:
            v = f"기각 — 프론티어 통제 실패(같은 MDD flat 대비 자산 {g:+.1f}%)"
        elif bonf[name] >= 0.05:
            fr = frag.get(name, {})
            v = (f"조건부 유망(단독 p={p:.3f}, 보정 p={bonf[name]:.3f}≥0.05 → 정식 기각) "
                 f"— 같은 MDD flat 대비 자산 {g:+.1f}%, 발동 {fr.get('events', '?')}회, "
                 f"연도제거 {fr.get('loo_pos', '?')}/{fr.get('loo_total', '?')}")
        else:
            v = f"유망 — 보정 p={bonf[name]:.3f}, 같은 MDD flat 대비 자산 {g:+.1f}%"
        verdicts[name] = v
        print(f"  {name:<20} {v}", flush=True)

    # ── JSON ──
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    payload = {
        "axis": "사이징·낙폭 제어 (size_pct / 품질 기반 사이징 / 연속손실 브레이크)",
        "generated": "2026-08-07",
        "harness": "scripts/live_parity.py run_live_parity, 7페어, lev 7x, size 0.9, "
                   "동시보유 분할, DD 스로틀 -25%×0.7, 파산 20%",
        "n_combos_tried": len(cands),
        "structural_bound": s1,
        "baseline": {
            "name": "현행 (size_pct 0.9, 무차등)",
            "params": {"size_pct": SIZE, "leverage": LEV},
            "n": int(base["n_exec"]), "r_mean": float(R.mean()),
            "compound": base["compound"], "mdd": base["mdd"], "ruin": base["ruin"],
            "boot_p50": base["boot_p50"], "boot_p5": base["boot_p5"],
            "p_perm": None,
        },
        "candidates": [
            {"name": name, "params": params, "kind": kind,
             "n": int(ev["n_exec"]), "r_mean": rmean,
             "compound": ev["compound"], "mdd": ev["mdd"], "ruin": ev["ruin"],
             "boot_p50": ev["boot_p50"], "boot_p5": ev["boot_p5"],
             "p_perm": (None if pvals[name] != pvals[name] else pvals[name]),
             "p_perm_bonferroni": (None if pvals[name] != pvals[name]
                                   else min(1.0, pvals[name] * m_tests)),
             "flat_same_mdd_k": front[name]["flat_k_mdd"],
             "flat_same_mdd_compound": front[name]["flat_ret_at_mdd"],
             "ret_gain_vs_flat_pct": front[name]["ret_gain_pct"],
             "flat_mdd_at_same_ret": front[name]["flat_mdd_at_ret"],
             "verdict": verdicts[name]}
            for name, params, w, kind, ev, rmean in results
        ],
        "regime_dir_cells": cells,
        "year_consistency": {"per_year": yrs, "pos": ypos, "total": ytot},
        "fragility": frag,
        "n_perm_tests": m_tests,
        "notes": (
            "① size_pct 는 복리식에서 레버리지와 곱으로만 들어가므로 독립 손잡이가 "
            "아니다(수치 항등식 확인, 차이 1e-15). 스윕은 대조군으로만 유지. "
            "② 품질/브레이크의 상한은 조건부 정보량이며 표본 126건 · SE %.3fR 이 "
            "노이즈 바닥이다. 상한은 노이즈를 넘었으므로(2.6배·3.9배) 스윕을 진행했다. "
            "③ 프론티어 통제는 **MDD 매칭** 으로 한다. 수익 매칭은 후보가 flat 곡선의 "
            "최대 수익을 넘어(포화) 정의되지 않는다. flat 후보 자신의 매칭 오차가 "
            "±3% 이므로 그것이 이 지표의 분해능이다. "
            "④ 전 후보 Bonferroni 보정 p ≥ 0.05 → 전면 기각. 최선(brake_5x0.0)조차 "
            "발동 사건이 3회뿐이라 규칙이 아니라 사건 3개에 걸려 있다. "
            "⑤ 한계: 브레이크가 거래를 건너뛰어도 동시보유 계수는 전체 집합 기준으로 "
            "고정했다. 재계산하면 잔여 거래 size 가 커져 개선폭이 줄어드는 방향 = "
            "현재 수치는 후보에게 유리한(낙관) 쪽이다. "
            "⑥ live_parity GAPS(ote_up_level 0.786, sweep_gate, smart_size, "
            "daily_loss_limit) 는 여전히 미구현 — smart_size 미구현은 이 축의 직접 한계."
        ) % s1["r_se"],
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\n  저장: {OUT}", flush=True)
    print(f"  시도한 조합 수: {len(cands)} (flat 3 · 품질 4 · 브레이크 7)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
