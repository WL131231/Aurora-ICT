"""#AUTONOMOUS 2026-08-07: 피보나치 되돌림 레벨(0.5~0.886) × **시장 국면** 교차 분석.

파트너 질문: "국면마다 다르게 쓰는 것도 생각해보고".
즉 **국면마다 최적 되돌림 레벨이 다른가**를 판정한다.
다르다면 현행 #REGIME-OTE(상승 국면 z>0.75 에서만 0.786, 그 외 0.707)를
바꿀 근거가 되고, 다르지 않다면 조건부 규칙 자체가 근거를 잃는다.

무엇을 재나
  data/fib/lv{레벨}.json (fib_level_run.py 가 run_live_parity 로 뽑은 라이브 정합
  거래)를 국면 라벨로 쪼개, **국면 × 레벨**의 건당 R배수·거래수를 낸다.
  %수익이 아니라 R배수로 재는 이유: 레벨마다 진입가가 달라 SL 폭이 달라지므로
  %수익 비교는 왜곡된다(팀 판정기준 4번).

왜 국면 정의를 여러 개 쓰나
  결론이 "국면 정의를 그렇게 골라서" 나온 것인지 확인하려고. 정의를 바꿔도
  같은 레벨이 이기면 신호, 정의마다 최적이 바뀌면 그건 잡음이다.
    (a) z : 일봉 20일 수익률 z-score > 0.75 상승 / < -0.75 하락 / 그 외 횡보.
            현행 라이브 #REGIME-OTE 와 **동일 계산식**(bot_ict_instance._regime_is_up:
            z = r20 / (일간수익률 20일 표준편차 × √20), 어제까지 마감 일봉).
    (b) sma: 일봉 종가 vs SMA20 vs SMA50 배열. 종가>20>50 상승 / 종가<20<50 하락 /
            그 외 횡보. z 와 달리 변동성으로 정규화하지 않는 순수 배열 판정.
    (c) csi: 횡보 상태 지수(chop_state_index, 로지스틱 인식모델) >= 0.6 이면 횡보,
            아니면 일봉 20일 변화율 부호로 상승/하락. 위 둘과 완전히 다른 축
            (변동성 수축·추세강도 부재를 본다).
  전부 **직전 완결 봉까지만** 써서 미래참조를 차단한다.

어떻게 판정하나 (팀 표준)
  · 국면별로 최적 레벨을 골라 쓰는 것이 단일 최적 레벨보다 건당 R을 얼마나
    올리는가(Δ)를 재고, 그 Δ가 **국면 라벨을 블록 셔플한 20000회**와 구분되는지
    순열검정한다(판정기준 1). 국면별 최적 선택은 구조상 Δ>=0 이므로,
    귀무분포도 >=0 이다 — 즉 "고르기만 해도 생기는 이득"과 대조하는 검정이다.
  · 국면 × 방향 교차표를 반드시 같이 낸다(판정기준 2). 게이트(htf align·cond_align)
    때문에 국면과 방향이 붙어 있으면 "국면 효과"는 "방향 효과"와 분리 불가다.
  · 연도별 분해(판정기준 3), 셀 표본 30건 미만이면 표본부족으로 명시(판정기준 5).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402

LEVELS = ["0.500", "0.618", "0.707", "0.886"]  # 실행 시 실제 파일로 덮어씀
PAIRS = ["BTCUSDT", "ETHUSDT"]
FIB_DIR = Path("data/fib")
N_PERM = 20000
BLOCK_DAYS = 30      # 블록 셔플 단위 — 국면은 수십일 지속하므로 일 단위 셔플은 과대검정
MIN_CELL = 5         # 이 미만인 셀은 국면별 선택에서 제외(전역 최적으로 폴백)
RNG_SEED = 20260807


# ────────────────────────────── 국면 라벨 ──────────────────────────────
def daily_close(sym: str) -> pd.Series:
    """1분봉 → 일봉 종가."""
    return _resample(_load_full(sym)).resample("1D").agg({"close": "last"}).dropna()["close"]


def label_z(c: pd.Series) -> pd.Series:
    """(a) 현행 #REGIME-OTE 와 동일: z = r20 / (일수익률 20일 std × √20)."""
    sig = c.pct_change().rolling(20).std()
    r20 = c / c.shift(20) - 1.0
    z = (r20 / (sig * np.sqrt(20))).shift(1)   # shift(1) = 어제 마감까지만 사용
    return pd.Series(np.where(z > 0.75, "상승", np.where(z < -0.75, "하락", "횡보")),
                     index=c.index)


def label_sma(c: pd.Series) -> pd.Series:
    """(b) 일봉 20/50 SMA 배열."""
    s20 = c.rolling(20).mean()
    s50 = c.rolling(50).mean()
    lab = np.where((c > s20) & (s20 > s50), "상승",
                   np.where((c < s20) & (s20 < s50), "하락", "횡보"))
    return pd.Series(lab, index=c.index).shift(1).fillna("횡보")


def label_csi(sym: str, c: pd.Series, model) -> pd.Series:
    """(c) CSI(횡보 인식 지수) 우선, 추세면 20일 변화율 부호로 상승/하락.

    CSI 는 확률이지만 일평균을 내면 0.3~0.5 에 몰려 고정 임계(0.6)로는 횡보가 0건이
    된다 → **자기 과거 분포의 상위 1/3**(확장창 66% 분위, 250일 이상 쌓인 뒤부터,
    shift 로 인과 유지)을 횡보로 본다. 페어별 스케일 차이도 자동 흡수된다.
    """
    from chop_state_index import csi_series, load_1h
    s = csi_series(load_1h(sym), model).shift(1)          # 1h 해상도, 인과 shift
    daily = s.resample("1D").mean().reindex(c.index).ffill()
    thr = daily.expanding(250).quantile(0.66).shift(1)     # 과거만으로 만든 임계
    chg = (c / c.shift(20) - 1.0).shift(1)
    lab = np.where(daily >= thr, "횡보", np.where(chg > 0, "상승", "하락"))
    return pd.Series(lab, index=c.index)


# ────────────────────────────── 데이터 조립 ──────────────────────────────
def load_trades(levels: list[str]) -> pd.DataFrame:
    rows = []
    for lv in levels:
        d = json.loads((FIB_DIR / f"lv{lv}.json").read_text(encoding="utf-8"))
        for t in d["trades"]:
            ts = pd.Timestamp(t["ts"], unit="ms", tz="UTC")
            rows.append(dict(lv=lv, sym=t["sym"], ts=ts, day=ts.normalize(),
                             r=float(t["r"]), dir=t["dir"], year=ts.year,
                             outcome=t.get("outcome", "")))
    return pd.DataFrame(rows)


def attach(D: pd.DataFrame, labmap: dict[str, pd.Series], name: str) -> pd.DataFrame:
    """거래 진입일 → 해당 페어의 국면 라벨."""
    out = []
    for sym, g in D.groupby("sym"):
        lab = labmap[sym]
        out.append(g.assign(**{name: lab.reindex(g.day).to_numpy()}))
    return pd.concat(out).sort_values("ts")


# ────────────────────────────── 통계량 ──────────────────────────────
def cell_table(D: pd.DataFrame, col: str) -> pd.DataFrame:
    """국면 × 레벨 건당 R / 거래수."""
    piv = D.pivot_table(index="lv", columns=col, values="r", aggfunc=["mean", "count"])
    return piv


def delta_stat(lv_masks, reg_arr, r_arr, glob_mean, best_g, regimes) -> tuple[float, dict]:
    """국면별 최적 레벨 선택이 단일 최적 레벨 대비 올리는 건당 R (Δ).

    · 전역 최적(best_g) = 레벨별 전체 건당 R 이 가장 큰 레벨. 국면 라벨과 무관하므로
      순열 중에도 바뀌지 않는다 → 미리 계산해 넘긴다.
    · 국면별 최적 = 셀 표본 MIN_CELL 이상 중 건당 R 최대 (미달이면 전역 최적 폴백).
    · Δ = (국면별 최적으로 고른 거래들의 건당 R) − (전역 최적 레벨의 건당 R)
    """
    picks = {}
    tot, cnt = 0.0, 0
    for g in regimes:
        m_g = reg_arr == g
        best_lv, best_v = None, -1e18
        for lv, m_lv in lv_masks.items():
            m = m_g & m_lv
            k = int(m.sum())
            if k >= MIN_CELL:
                v = float(r_arr[m].mean())
                if v > best_v:
                    best_lv, best_v = lv, v
        pick = best_lv if best_lv is not None else best_g
        picks[g] = pick
        sel = r_arr[m_g & lv_masks[pick]]
        tot += float(sel.sum()); cnt += len(sel)
    mean_sel = tot / cnt if cnt else 0.0
    return float(mean_sel - glob_mean), picks


def block_order(n: int, rng, block: int) -> np.ndarray:
    """0..n-1 을 block 크기 덩어리로 잘라 덩어리 순서만 섞은 인덱스.

    국면은 수십 일 지속하므로 하루씩 무작위로 섞으면 자기상관이 파괴돼
    귀무분포가 지나치게 좁아지고(=과대검정) p 가 부풀려진다. 덩어리 셔플로
    지속성을 보존한 채 '언제 어떤 국면이 왔는지'만 무작위화한다.
    """
    starts = np.arange(0, n, block)
    order = rng.permutation(len(starts))
    return np.concatenate([np.arange(s, min(s + block, n)) for s in starts[order]])


# ────────────────────────────── 출력 ──────────────────────────────
def report_def(D: pd.DataFrame, col: str, title: str, levels: list[str],
               labmap: dict[str, pd.Series], rng) -> None:
    regimes = ["상승", "횡보", "하락"]
    print(f"\n\n{'=' * 78}\n### 국면 정의 {title}\n{'=' * 78}", flush=True)

    # 1) 국면 × 레벨 건당 R / n
    print("\n[표1] 국면 × 레벨 — 건당R (n)   ※ n<30 은 표본부족", flush=True)
    hdr = f"  {'레벨':<8}" + "".join(f"{g:>16}" for g in regimes) + f"{'전체':>16}"
    print(hdr, flush=True)
    for lv in levels:
        row = f"  {lv:<8}"
        for g in regimes + [None]:
            m = (D.lv == lv) if g is None else ((D.lv == lv) & (D[col] == g))
            sub = D.loc[m, "r"]
            row += f"{sub.mean():>+10.3f}R({len(sub):>2d})" if len(sub) else f"{'-':>16}"
        print(row, flush=True)

    # 2) 국면 × 방향 (판정기준 2 — 기저 통제)
    print("\n[표2] 국면 × 방향 교차표 (전 레벨 합산) — 국면과 방향이 붙어 있는가",
          flush=True)
    ct = pd.crosstab(D[col], D["dir"])
    print("    " + ct.to_string().replace("\n", "\n    "), flush=True)
    coll = []
    for g in regimes:
        sub = D[D[col] == g]
        if len(sub) == 0:
            continue
        p = (sub["dir"] == "long").mean()
        coll.append(f"{g}: 롱 {100 * p:.0f}%")
    print("    → " + " / ".join(coll), flush=True)

    # 3) 국면별 최적 레벨과 Δ
    r_arr = D.r.to_numpy(dtype=float)
    reg_arr = D[col].to_numpy()
    lv_masks = {lv: (D.lv.to_numpy() == lv) for lv in levels}
    glob = {lv: float(r_arr[m].mean()) for lv, m in lv_masks.items()}
    best_g = max(glob, key=glob.get)
    d_obs, picks = delta_stat(lv_masks, reg_arr, r_arr, glob[best_g], best_g, regimes)
    print(f"\n[표3] 전역 최적 레벨 = {best_g} ({glob[best_g]:+.3f}R)", flush=True)
    for g in regimes:
        m = D[col] == g
        n_all = int(m.sum())
        pick = picks[g]
        cell = D.loc[m & (D.lv == pick), "r"]
        flag = "  ※표본부족" if len(cell) < 30 else ""
        print(f"    {g}: 최적 {pick}  건당 {cell.mean():+.3f}R  n={len(cell)}"
              f"  (국면 전체 거래 {n_all}){flag}", flush=True)
    print(f"\n    Δ(국면별 선택 − 전역 단일) = {d_obs:+.4f}R", flush=True)

    # 4) 순열검정 — 국면 라벨 블록 셔플
    #    거래일 → 일봉 인덱스 위치를 미리 뽑아두고, 매 순열마다 라벨 배열만 재배치.
    sym_arr = D.sym.to_numpy()
    pos: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in labmap:
        m = sym_arr == s
        loc = labmap[s].index.get_indexer(pd.DatetimeIndex(D.day.to_numpy()[m]))
        pos[s] = (m, loc)
    vals = {s: labmap[s].to_numpy() for s in labmap}
    ge = 0
    dist = np.empty(N_PERM)
    newreg = np.empty(len(D), dtype=object)
    for i in range(N_PERM):
        for s in labmap:
            m, loc = pos[s]
            order = block_order(len(vals[s]), rng, BLOCK_DAYS)
            newreg[m] = vals[s][order][loc]
        d_p, _ = delta_stat(lv_masks, newreg, r_arr, glob[best_g], best_g, regimes)
        dist[i] = d_p
        if d_p >= d_obs - 1e-12:
            ge += 1
    p = (ge + 1) / (N_PERM + 1)
    print(f"\n[표4] 순열검정 ({N_PERM}회, {BLOCK_DAYS}일 블록 셔플)", flush=True)
    print(f"    귀무 Δ 평균 {dist.mean():+.4f}R  중앙 {np.median(dist):+.4f}R  "
          f"95%분위 {np.quantile(dist, 0.95):+.4f}R", flush=True)
    print(f"    관측 Δ {d_obs:+.4f}R  →  p = {p:.4f}  "
          f"{'통과(p<0.05)' if p < 0.05 else '기각(p>=0.05) — 잡음과 구분 불가'}",
          flush=True)

    # 5) 현행 규칙(#REGIME-OTE) 직접 대조
    up = D[(D[col] == "상승") & (D.lv == "0.786")]["r"]
    rest = D[(D[col] != "상승") & (D.lv == "0.707")]["r"]
    cur = pd.concat([up, rest])
    base = D[D.lv == "0.707"]["r"]
    print("\n[표5] 현행 #REGIME-OTE(상승→0.786, 그 외→0.707) vs 순수 0.707", flush=True)
    print(f"    현행 조건부: 건당 {cur.mean():+.3f}R  n={len(cur)}  "
          f"(상승칸 {up.mean():+.3f}R n={len(up)})", flush=True)
    print(f"    순수 0.707 : 건당 {base.mean():+.3f}R  n={len(base)}", flush=True)
    print(f"    차이 = {cur.mean() - base.mean():+.3f}R", flush=True)

    # 5-b) 상승 국면 레벨별 부트스트랩 신뢰구간 — 현행 0.786 선택의 근거 점검.
    #      건당 R 의 95% CI 가 서로 겹치면 "레벨 차이"를 주장할 수 없다.
    print("\n[표5-b] 상승 국면 레벨별 건당R 95% 부트스트랩 CI (10000회 재표집)",
          flush=True)
    for lv in levels:
        v = D[(D[col] == "상승") & (D.lv == lv)]["r"].to_numpy(dtype=float)
        if len(v) < 3:
            print(f"    {lv}: n={len(v)} 표본부족", flush=True)
            continue
        bs = rng.choice(v, size=(10000, len(v)), replace=True).mean(axis=1)
        lo, hi = np.quantile(bs, [0.025, 0.975])
        mark = "  ← 현행 상승 국면 값" if lv == "0.786" else ""
        print(f"    {lv}: {v.mean():+.3f}R  CI[{lo:+.3f}, {hi:+.3f}]  n={len(v)}{mark}",
              flush=True)

    # 6) 연도 일관성 — 국면별 최적 레벨이 연도마다 유지되는가
    print("\n[표6] 연도 × 국면 — 그 해 최적 레벨(건당R, n>=3 인 셀만)", flush=True)
    for y in sorted(D.year.unique()):
        line = f"    {y}: "
        for g in regimes:
            sub = D[(D.year == y) & (D[col] == g)]
            cand = {lv: sub[sub.lv == lv]["r"] for lv in levels}
            cand = {k: v.mean() for k, v in cand.items() if len(v) >= 3}
            line += f"{g}={max(cand, key=cand.get)}({max(cand.values()):+.2f}R) " \
                if cand else f"{g}=표본부족 "
        print(line, flush=True)


def main() -> int:
    levels = sorted(p.stem[2:] for p in FIB_DIR.glob("lv*.json"))
    print(f"레벨 {levels} / 페어 {PAIRS}", flush=True)
    D = load_trades(levels)
    print(f"총 거래 {len(D)}건 (레벨 5개 합산, 레벨별 43~52건)", flush=True)

    closes = {s: daily_close(s) for s in PAIRS}
    defs: dict[str, dict[str, pd.Series]] = {
        "(a) z-score 20일 — 현행 #REGIME-OTE": {s: label_z(c) for s, c in closes.items()},
        "(b) 일봉 20/50 SMA 배열": {s: label_sma(c) for s, c in closes.items()},
    }
    try:
        from chop_state_index import fit_csi
        model = fit_csi(PAIRS)
        defs["(c) CSI 횡보지수 + 20일 부호"] = {
            s: label_csi(s, c, model) for s, c in closes.items()}
    except Exception as e:  # noqa: BLE001
        print(f"  CSI 정의 생략(실패): {e}", flush=True)

    rng = np.random.default_rng(RNG_SEED)
    for i, (title, labmap) in enumerate(defs.items()):
        col = f"reg{i}"
        Dx = attach(D, labmap, col).dropna(subset=[col])
        report_def(Dx, col, title, levels, labmap, rng)

    # 정의 간 라벨 일치도 — 결론이 정의에 의존하는지 판단 재료
    print(f"\n\n{'=' * 78}\n### 정의 간 라벨 일치도 (BTC 기준, 일 단위)\n{'=' * 78}",
          flush=True)
    keys = list(defs.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a = defs[keys[i]]["BTCUSDT"]
            b = defs[keys[j]]["BTCUSDT"]
            idx = a.index.intersection(b.index)
            agree = (a.reindex(idx) == b.reindex(idx)).mean()
            print(f"  {keys[i]} vs {keys[j]} : 일치 {100 * agree:.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
