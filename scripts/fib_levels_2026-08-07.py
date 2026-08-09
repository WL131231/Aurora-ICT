"""피보나치 되돌림 진입 깊이(ote_level)별 전 구간 성적 비교 — 2026-08-07.

무엇을 재는가
  data/fib/lv{level}.json (Backtest 단계 산출물)에 저장된 레벨별 거래를 읽어
  레벨마다 거래수 / 건당R / 승률 / 합계R / 최대연속손실 을 계산한다.
  덧붙여 단조성(깊을수록 좋은가, 얕을수록 좋은가, 중간 봉우리인가, 무패턴인가)을
  통계로 판정하고, 현행 라이브 기본값 0.707 이 그 안에서 어디 있는지 표시한다.

왜 재는가
  현행 라이브는 0.707 기본이고 상승 국면에서만 0.786 을 쓴다(#REGIME-OTE, 2026-07-10).
  그 선택이 데이터로 뒷받침되는지, 아니면 다른 레벨이 더 낫거나 애초에 레벨 간
  차이가 노이즈 수준인지를 확인한다.

방법상 주의 (이 팀의 판정 기준 반영)
  1) 레벨별 거래 집합은 서로 독립이 아니다. 같은 FVG 셋업이 레벨만 바꿔 거의 그대로
     다시 나온다(진입가만 조금 다름). 그래서 레벨을 독립 표본으로 놓고 비교하면
     검정력이 왜곡된다. → (sym, ts) 로 셋업을 매칭해서 **대응표본(paired)** 으로도 본다.
  2) 순열검정은 대응 구조를 유지한 채(셋업 안에서 레벨 라벨만 섞어) 20000회 돌린다.
  3) 국면×방향 매칭 기저 통제를 위해 롱/숏, 심볼별로 쪼개서 같이 낸다.
  4) 비교는 전부 R 배수 기준. %수익(raw)은 SL 폭이 레벨마다 달라 왜곡되므로 참고용만.
  5) 표본 30건 미만 레벨은 "표본부족"으로 찍고 결론에서 뺀다.
"""

import glob
import json
import os
from collections import defaultdict, Counter
from datetime import datetime, timezone

import numpy as np

MIN_N = 30          # 이 미만이면 표본부족
N_PERM = 20000      # 순열검정 반복
RNG = np.random.default_rng(20260807)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data", "fib")


# ---------------------------------------------------------------- 로딩
def load_levels():
    """레벨별 거래 목록을 {level: [trade,...]} 로 읽는다. 시간순 정렬."""
    out = {}
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "lv*.json"))):
        d = json.load(open(path, encoding="utf-8"))
        trades = sorted(d["trades"], key=lambda t: t["ts"])
        out[float(d["level"])] = trades
    return out


def year_of(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year


# ---------------------------------------------------------------- 지표
def max_consec_loss(trades):
    """시간순으로 R<=0 이 몇 번 연달아 나왔는지의 최댓값."""
    best = cur = 0
    for t in sorted(trades, key=lambda x: x["ts"]):
        if t["r"] <= 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def stats(trades):
    r = np.array([t["r"] for t in trades], dtype=float)
    if len(r) == 0:
        return None
    wins = (r > 0).sum()
    # 건당R의 표준오차 (t 통계량 참고용)
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else float("nan")
    return {
        "n": len(r),
        "mean_r": r.mean(),
        "win": wins / len(r) * 100,
        "sum_r": r.sum(),
        "mdd_streak": max_consec_loss(trades),
        "se": se,
        "t": r.mean() / se if se and se == se and se > 0 else float("nan"),
        "raw_sum": sum(t["raw"] for t in trades) * 100,
    }


def fmt_row(label, s, note=""):
    return (f"{label:>10} | {s['n']:>4} | {s['mean_r']:>+7.3f} | {s['win']:>5.1f}% | "
            f"{s['sum_r']:>+8.2f} | {s['mdd_streak']:>4} | {s['t']:>+5.2f} | {note}")


# ---------------------------------------------------------------- 본체
def main():
    levels = load_levels()
    lv_keys = sorted(levels)

    print("=" * 96)
    print("피보나치 진입 깊이(ote_level)별 전 구간 성적 — Origo / BTC+ETH")
    all_ts = [t["ts"] for tr in levels.values() for t in tr]
    print(f"구간: {datetime.fromtimestamp(min(all_ts)/1000, tz=timezone.utc).date()} ~ "
          f"{datetime.fromtimestamp(max(all_ts)/1000, tz=timezone.utc).date()}")
    print("=" * 96)

    # ---- 1) 레벨별 전체 표
    print("\n[표1] 레벨별 전 구간 (R 배수 기준)")
    print(f"{'레벨':>10} | {'거래수':>4} | {'건당R':>7} | {'승률':>6} | "
          f"{'합계R':>8} | {'최대연손':>4} | {'t값':>5} | 비고")
    print("-" * 96)
    table = {}
    for lv in lv_keys:
        s = stats(levels[lv])
        table[lv] = s
        note = []
        if s["n"] < MIN_N:
            note.append("표본부족")
        if abs(lv - 0.707) < 1e-9:
            note.append("<<< 현행 라이브 기본값")
        if abs(lv - 0.786) < 1e-9:
            note.append("(상승 국면 전용, #REGIME-OTE)")
        print(fmt_row(f"{lv:.3f}", s, " ".join(note)))

    usable = [lv for lv in lv_keys if table[lv]["n"] >= MIN_N]
    print(f"\n유효 레벨(n>={MIN_N}): {[f'{x:.3f}' for x in usable]}")

    # ---- 2) 방향별 (국면×방향 기저 통제)
    print("\n[표2] 방향별 분해 — '상승장 롱' 같은 기저효과 확인용")
    print(f"{'레벨':>10} | {'롱n':>4} {'롱건당R':>8} | {'숏n':>4} {'숏건당R':>8}")
    print("-" * 96)
    for lv in lv_keys:
        lo = stats([t for t in levels[lv] if t["dir"] == "long"])
        sh = stats([t for t in levels[lv] if t["dir"] == "short"])
        print(f"{lv:>10.3f} | {lo['n']:>4} {lo['mean_r']:>+8.3f} | {sh['n']:>4} {sh['mean_r']:>+8.3f}")

    # ---- 3) 심볼별
    print("\n[표3] 심볼별 건당R")
    syms = sorted({t["sym"] for tr in levels.values() for t in tr})
    print(f"{'레벨':>10} | " + " | ".join(f"{s:>12}" for s in syms))
    print("-" * 96)
    for lv in lv_keys:
        cells = []
        for sm in syms:
            s = stats([t for t in levels[lv] if t["sym"] == sm])
            cells.append(f"n{s['n']:>3} {s['mean_r']:>+6.3f}" if s else "     -")
        print(f"{lv:>10.3f} | " + " | ".join(f"{c:>12}" for c in cells))

    # ---- 4) 연도 일관성
    print("\n[표4] 연도별 합계R (괄호는 거래수) — 특정 연도 몰빵 여부")
    years = sorted({year_of(t["ts"]) for tr in levels.values() for t in tr})
    print(f"{'레벨':>10} | " + " | ".join(f"{y:>13}" for y in years))
    print("-" * 96)
    for lv in lv_keys:
        cells = []
        for y in years:
            sub = [t for t in levels[lv] if year_of(t["ts"]) == y]
            cells.append(f"{sum(t['r'] for t in sub):>+7.2f}({len(sub):>2})" if sub else "      -    ")
        print(f"{lv:>10.3f} | " + " | ".join(f"{c:>13}" for c in cells))

    # ---- 5) 대응표본 구성: 같은 셋업(sym, ts)이 5개 레벨 전부에 존재하는 것만
    print("\n[표5] 대응표본(같은 셋업이 모든 레벨에 있는 건만) — 레벨 간 순수 비교")
    idx = defaultdict(dict)
    for lv in lv_keys:
        for t in levels[lv]:
            idx[(t["sym"], t["ts"])][lv] = t["r"]
    matched = [k for k, v in idx.items() if len(v) == len(lv_keys)]
    matched.sort(key=lambda k: k[1])
    M = np.array([[idx[k][lv] for lv in lv_keys] for k in matched], dtype=float)
    print(f"매칭된 셋업 수: {len(matched)}건 (레벨별 원 거래수 {[table[lv]['n'] for lv in lv_keys]})")
    print(f"{'레벨':>10} | {'건당R':>8} | {'승률':>6} | {'합계R':>8}")
    print("-" * 96)
    for j, lv in enumerate(lv_keys):
        col = M[:, j]
        print(f"{lv:>10.3f} | {col.mean():>+8.3f} | {(col > 0).mean()*100:>5.1f}% | {col.sum():>+8.2f}")

    if len(matched) < MIN_N:
        print("  ** 대응표본이 30건 미만 → 대응 비교는 표본부족")

    # ---- 6) 단조성 판정
    print("\n[판정A] 단조성 — 레벨 순서 vs 건당R")
    means = np.array([table[lv]["mean_r"] for lv in lv_keys])
    order = np.arange(len(lv_keys))
    # 스피어만(레벨 5개뿐이라 사실상 순위상관)
    rho_all = np.corrcoef(order, np.argsort(np.argsort(means)))[0, 1]
    print(f"  전체표본 건당R: {[f'{m:+.3f}' for m in means]}")
    print(f"  레벨순서-성적 순위상관(rho) = {rho_all:+.2f}")
    peak = lv_keys[int(np.argmax(means))]
    trough = lv_keys[int(np.argmin(means))]
    print(f"  최고 = {peak:.3f}, 최저 = {trough:.3f}")

    mm = M.mean(axis=0)
    rho_m = np.corrcoef(order, np.argsort(np.argsort(mm)))[0, 1]
    print(f"  대응표본 건당R: {[f'{m:+.3f}' for m in mm]}  rho = {rho_m:+.2f}")

    # 대응 순열검정: 각 셋업 안에서 레벨 라벨을 무작위로 섞는다.
    # (셋업 자체의 좋고 나쁨은 그대로 두고, '레벨이 성적을 만드는가'만 검정)
    def spearman_stat(col_means):
        rk = np.argsort(np.argsort(col_means))
        return np.corrcoef(order, rk)[0, 1]

    obs_rho = spearman_stat(mm)
    obs_spread = mm.max() - mm.min()
    cnt_rho = cnt_spread = 0
    for _ in range(N_PERM):
        P = RNG.permuted(M, axis=1)          # 행(셋업)마다 레벨 라벨 셔플
        pm = P.mean(axis=0)
        if abs(spearman_stat(pm)) >= abs(obs_rho) - 1e-12:
            cnt_rho += 1
        if (pm.max() - pm.min()) >= obs_spread - 1e-12:
            cnt_spread += 1
    p_rho = (cnt_rho + 1) / (N_PERM + 1)
    p_spread = (cnt_spread + 1) / (N_PERM + 1)
    print("\n[판정B] 대응 순열검정 (셋업 내 레벨 라벨 셔플, "
          f"{N_PERM}회)")
    print(f"  단조 추세 rho={obs_rho:+.2f}  → p = {p_rho:.4f}")
    print(f"  레벨 간 최대격차 {obs_spread:.3f}R → p = {p_spread:.4f}")

    # ---- 7) 현행 0.707 vs 각 레벨 대응 비교
    print("\n[판정C] 현행 0.707 대비 (대응표본, 부호섞기 순열검정)")
    j707 = lv_keys.index(0.707)
    print(f"{'상대레벨':>10} | {'ΔR/건':>8} | {'p':>7} | {'0.707 우세 건수':>14}")
    print("-" * 96)
    for j, lv in enumerate(lv_keys):
        if j == j707:
            continue
        d = M[:, j] - M[:, j707]
        obs = d.mean()
        # 대응 부호섞기: 두 레벨은 교환 가능하다는 귀무가설
        signs = RNG.choice([-1.0, 1.0], size=(N_PERM, len(d)))
        null = (signs * d).mean(axis=1)
        p = ((np.abs(null) >= abs(obs) - 1e-12).sum() + 1) / (N_PERM + 1)
        better707 = int((d < 0).sum())
        print(f"{lv:>10.3f} | {obs:>+8.3f} | {p:>7.4f} | {better707:>6}/{len(d)}")

    # ---- 8) 건당R이 0과 다른지 (레벨 자체의 엣지 유무)
    print("\n[판정D] 각 레벨 건당R이 0보다 큰가 (전체표본 부트스트랩 95% 구간)")
    print(f"{'레벨':>10} | {'건당R':>8} | {'95% CI':>22}")
    print("-" * 96)
    for lv in lv_keys:
        r = np.array([t["r"] for t in levels[lv]])
        bs = RNG.choice(r, size=(10000, len(r)), replace=True).mean(axis=1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"{lv:>10.3f} | {r.mean():>+8.3f} | [{lo:>+7.3f}, {hi:>+7.3f}]")

    # ---- 9) 비대응(전체표본) 순열검정
    # 대응검정은 '같은 셋업에서 진입가만 다를 때'의 기계적 차이까지 잡아낸다.
    # 실제로 우리가 알고 싶은 건 "레벨을 바꾸면 전체 성적이 달라지는가"이므로
    # 전체표본을 통째로 섞어 레벨 라벨을 재배정하는 검정도 같이 본다.
    print("\n[판정E] 비대응 순열검정 (전체표본, 레벨 라벨 무작위 재배정)")
    pool = np.array([t["r"] for lv in lv_keys for t in levels[lv]])
    sizes = [table[lv]["n"] for lv in lv_keys]
    obs_means = np.array([table[lv]["mean_r"] for lv in lv_keys])
    obs_spread2 = obs_means.max() - obs_means.min()
    obs_707 = table[0.707]["mean_r"] - np.mean(
        [table[lv]["mean_r"] for lv in lv_keys if lv != 0.707])
    c_sp = c_707 = 0
    for _ in range(N_PERM):
        p = RNG.permutation(pool)
        ms, off = [], 0
        for n in sizes:
            ms.append(p[off:off + n].mean())
            off += n
        ms = np.array(ms)
        if ms.max() - ms.min() >= obs_spread2 - 1e-12:
            c_sp += 1
        d707 = ms[j707] - np.mean(np.delete(ms, j707))
        if abs(d707) >= abs(obs_707) - 1e-12:
            c_707 += 1
    print(f"  레벨 간 최대격차 {obs_spread2:.3f}R → p = {(c_sp+1)/(N_PERM+1):.4f}")
    print(f"  0.707 - 나머지평균 = {obs_707:+.3f}R → p = {(c_707+1)/(N_PERM+1):.4f}")

    # ---- 10) 대응차이가 '기계적'인지 확인
    # 같은 셋업에서 레벨만 깊어지면 진입가가 유리해져 같은 SL/TP 대비 R이 자동으로
    # 조금 커진다. 그게 전부라면 outcome(청산 사유)은 레벨과 무관하게 같아야 한다.
    print("\n[판정F] 대응표본 차이가 기계적 진입가 효과인지 확인")
    same_outcome = 0
    for k in matched:
        oc = {lv: next(t["outcome"] for t in levels[lv]
                       if t["sym"] == k[0] and t["ts"] == k[1]) for lv in lv_keys}
        if len(set(oc.values())) == 1:
            same_outcome += 1
    d_all = M[:, -1] - M[:, 0]  # 0.886 - 0.500
    print(f"  매칭 셋업 {len(matched)}건 중 5개 레벨 outcome이 전부 동일: {same_outcome}건 "
          f"({same_outcome/len(matched)*100:.0f}%)")
    print(f"  0.886-0.500 건당차 중앙값 {np.median(d_all):+.4f}R, "
          f"평균 {d_all.mean():+.4f}R, |차이| 중앙값 {np.median(np.abs(d_all)):.4f}R")
    print(f"  참고: 레벨별 평균 |R| = {np.abs(M).mean():.3f}R "
          f"→ 차이는 그 {np.median(np.abs(d_all))/np.abs(M).mean()*100:.1f}% 수준")

    # ---- 11) 매칭에서 빠진 거래 (레벨별 고유 셋업)
    print("\n[판정G] 매칭에서 빠진 셋업 = 레벨에 따라 체결 여부가 갈린 건들")
    for lv in lv_keys:
        uniq = [t for t in levels[lv] if len(idx[(t["sym"], t["ts"])]) < len(lv_keys)]
        s = stats(uniq)
        print(f"  {lv:.3f}: 비매칭 {s['n']:>2}건, 건당R {s['mean_r']:>+6.3f}, "
              f"승률 {s['win']:>5.1f}%")

    # ---- 12) 참고: outcome 분포
    print("\n[참고] 레벨별 outcome 분포")
    for lv in lv_keys:
        print(f"  {lv:.3f}: {dict(Counter(t['outcome'] for t in levels[lv]))}")


if __name__ == "__main__":
    main()
