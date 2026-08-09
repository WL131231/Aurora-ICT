"""#AUTONOMOUS 2026-08-07: 피보나치 되돌림 레벨의 **조건부 규칙** 탐색·검증.

## 무엇을 재나
현행 라이브(#REGIME-OTE, 2026-07-10)는 조건부다 —
기본 ote_level=0.707, 단 **상승 국면(일봉 20일 z>0.75)에서만 0.786**.
이 규칙이 (a) 재현되는지, (b) **고정 레벨 최선보다 실제로 나은지**,
(c) 다른 조건부 조합이 더 나은지, (d) 그게 **노이즈가 아닌지**를 잰다.

## 어떻게 재나
data/fib/lv{level}.json = 레벨별로 라이브 정합 백테(run_live_parity)를 통째로
다시 돌린 거래 장부다(진입가가 레벨마다 달라 사후 필터가 불가능하므로 재실행이
유일한 방법 — live_parity.GAPS 참조).

레벨마다 체결 여부·체결 시각이 달라 장부의 거래 수가 다르다(43~52건).
그래서 **신호 클러스터**로 묶는다 — 같은 심볼·같은 방향·진입시각 60분 이내면
같은 셋업으로 보고 하나의 신호로 취급, 그 신호의 레벨별 결과(R배수)를 붙인다.
어떤 레벨에서 결과가 없으면 = 그 깊이까지 안 왔다 = **미체결(거래 없음)**.

조건부 규칙 = 각 신호에 조건 라벨을 붙이고, 라벨마다 쓸 레벨을 정한 것.
그 규칙의 성적 = 신호별로 해당 레벨의 결과를 합산(미체결이면 0건).

## 검증
1. **순열검정 20000회** — 조건 라벨을 신호들에 **무작위 재배정**(개수 보존)하고
   같은 규칙을 다시 계산. 실제 규칙이 무작위 배정과 구분되는가.
2. **그리드 규칙은 max-통계량 순열검정** — 순열마다도 "그 순열에서의 최적 조합"을
   찾아 귀무분포를 만든다. 조합을 많이 뒤진 사실 자체를 검정에 포함시키는 것.
   (셀마다 독립적으로 최선 레벨을 고르면 총합이 최대이므로 분리 계산이 가능하다)
3. **국면×방향 매칭 기저** 표를 함께 출력 — "상승장 롱이 잘 된다"의 재확인인지 확인.
4. **연도별 분해** — 특정 연도 몰빵인지.
5. R배수로만 측정 — 레벨마다 SL 폭이 달라 %수익 비교는 왜곡된다.

## 표본 경고
5년 BTC+ETH 전체가 신호 60개 수준이다. 조건으로 나누면 셀당 10~25건 —
팀 기준(<30건 = 표본부족)에 걸린다. 결론은 이 한계를 달고 나간다.

사용: cd Aurora-ICT-research && PYTHONIOENCODING=utf-8 PYTHONPATH=src \
      python scripts/fib_conditional_2026-08-07.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full  # noqa: E402

LEVELS = ["0.500", "0.618", "0.707", "0.786", "0.886"]
LVF = {lv: float(lv) for lv in LEVELS}
PAIRS = ["BTCUSDT", "ETHUSDT"]
FIB = Path("data/fib")
RNG = np.random.default_rng(20260807)
N_PERM = 20000
CLUSTER_MS = 60 * 60 * 1000          # 같은 셋업으로 볼 진입시각 허용 간격(60분)


# ────────────────────────────── 국면·변동성 라벨 ──────────────────────────────
def daily_labels(sym: str) -> pd.DataFrame:
    """일자별 국면(상승/횡보/하락)·변동성(고/저) 라벨. 전부 '어제까지' 정보만 사용."""
    df = _load_full(sym)
    d = df["close"].resample("1D").last().dropna()
    ret = d.pct_change()
    sig = ret.rolling(20).std()
    r20 = d.pct_change(20)
    # regime_edge_lab.py(7/10)와 동일 정의: 20일 수익률을 20일σ×√20 으로 정규화
    z = (r20 / (sig * np.sqrt(20))).shift(1)
    regime = pd.Series(
        np.where(z > 0.75, "상승", np.where(z < -0.75, "하락", "횡보")),
        index=d.index, dtype=object)
    regime[z.isna()] = "미정"

    # 변동성 2분할 — 확장(expanding) 중앙값 기준. 전체 중앙값은 미래참조라 금지.
    s_prev = sig.shift(1)
    med = s_prev.expanding(min_periods=60).median()
    vol = pd.Series(np.where(s_prev > med, "고변동", "저변동"), index=d.index, dtype=object)
    vol[s_prev.isna() | med.isna()] = "미정"
    return pd.DataFrame({"regime": regime, "vol": vol, "z": z, "sig": s_prev})


# ────────────────────────────── 신호 클러스터 ──────────────────────────────
def build_signals() -> list[dict]:
    """레벨별 장부를 신호 단위로 묶는다. 반환: 신호 목록(레벨별 R 배수 포함)."""
    rows = []
    for lv in LEVELS:
        book = json.loads((FIB / f"lv{lv}.json").read_text(encoding="utf-8"))
        for t in book["trades"]:
            rows.append((t["sym"], t["dir"], t["ts"], lv, float(t["r"])))
    rows.sort(key=lambda x: (x[0], x[1], x[2]))

    sigs: list[dict] = []
    cur: dict | None = None
    for sym, d, ts, lv, r in rows:
        if (cur is not None and cur["sym"] == sym and cur["dir"] == d
                and ts - cur["last"] <= CLUSTER_MS and lv not in cur["r"]):
            cur["r"][lv] = r
            cur["last"] = ts
        else:
            cur = {"sym": sym, "dir": d, "ts": ts, "last": ts, "r": {lv: r}}
            sigs.append(cur)

    lab = {s: daily_labels(s) for s in PAIRS}
    for g in sigs:
        day = pd.Timestamp(g["ts"], unit="ms", tz="UTC").normalize()
        L = lab[g["sym"]]
        row = L.loc[day] if day in L.index else None
        g["regime"] = "미정" if row is None else str(row["regime"])
        g["vol"] = "미정" if row is None else str(row["vol"])
        g["year"] = pd.Timestamp(g["ts"], unit="ms", tz="UTC").year
    sigs.sort(key=lambda g: g["ts"])
    return sigs


# ────────────────────────────── 규칙 평가 ──────────────────────────────
def eval_rule(sigs: list[dict], labels: list[str], mapping: dict[str, str]) -> dict:
    """라벨→레벨 매핑 규칙의 성적. R배수 기준."""
    rs, yrs = [], {}
    for g, lab in zip(sigs, labels):
        lv = mapping.get(lab)
        if lv is None or lv not in g["r"]:
            continue                     # 미체결 = 거래 없음
        r = g["r"][lv]
        rs.append(r)
        yrs[g["year"]] = yrs.get(g["year"], 0.0) + r
    if not rs:
        return dict(n=0, sum=0.0, mean=0.0, wr=0.0, yrs={})
    a = np.array(rs)
    return dict(n=len(a), sum=float(a.sum()), mean=float(a.mean()),
                wr=100.0 * float((a > 0).mean()), yrs=yrs)


def cellsum_matrix(sigs: list[dict], labels: list[str],
                   cells: list[str]) -> np.ndarray:
    """[셀 × 레벨] 총 R 행렬. 셀마다 독립적으로 최선 레벨을 고를 수 있게 한다."""
    ci = {c: i for i, c in enumerate(cells)}
    M = np.zeros((len(cells), len(LEVELS)))
    for g, lab in zip(sigs, labels):
        if lab not in ci:
            continue
        for j, lv in enumerate(LEVELS):
            if lv in g["r"]:
                M[ci[lab], j] += g["r"][lv]
    return M


def perm_fixed_rule(sigs, labels, mapping, n_perm=N_PERM) -> tuple[float, float]:
    """고정 조건부 규칙 순열검정 — 라벨을 무작위 재배정(개수 보존). 반환 (p, 귀무평균)."""
    obs = eval_rule(sigs, labels, mapping)["sum"]
    arr = np.array(labels, dtype=object)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = eval_rule(sigs, list(RNG.permutation(arr)), mapping)["sum"]
    p = (1.0 + float((null >= obs).sum())) / (n_perm + 1.0)
    return p, float(null.mean())


def perm_grid_rule(sigs, labels, cells, n_perm=N_PERM):
    """그리드(셀별 최적 레벨) max-통계량 순열검정.

    반환: (최적 매핑, 관측 총R, p, 귀무 max 평균, 고정최선 총R)
    """
    M = cellsum_matrix(sigs, labels, cells)
    best_j = M.argmax(axis=1)
    obs = float(M.max(axis=1).sum())
    mapping = {c: LEVELS[best_j[i]] for i, c in enumerate(cells)}
    fixed_best = float(M.sum(axis=0).max())          # 전 셀 동일 레벨 중 최선

    arr = np.array(labels, dtype=object)
    null = np.empty(n_perm)
    for k in range(n_perm):
        null[k] = float(cellsum_matrix(sigs, list(RNG.permutation(arr)),
                                       cells).max(axis=1).sum())
    p = (1.0 + float((null >= obs).sum())) / (n_perm + 1.0)
    return mapping, obs, p, float(null.mean()), fixed_best


# ────────────────────────────── 출력 ──────────────────────────────
def show(tag: str, s: dict, base_sum: float | None = None) -> None:
    d = "" if base_sum is None else f" Δ={s['sum'] - base_sum:+7.2f}R"
    print(f"  {tag:34s} n={s['n']:3d}  총={s['sum']:+8.2f}R  "
          f"건당={s['mean']:+6.3f}R  승률={s['wr']:3.0f}%{d}", flush=True)


def year_line(s: dict) -> str:
    return " ".join(f"{y}:{v:+.1f}" for y, v in sorted(s["yrs"].items()))


def main() -> int:
    sigs = build_signals()
    print(f"신호(클러스터) 총 {len(sigs)}건 · 페어 {','.join(PAIRS)} · "
          f"{pd.Timestamp(sigs[0]['ts'], unit='ms', tz='UTC').date()} ~ "
          f"{pd.Timestamp(sigs[-1]['ts'], unit='ms', tz='UTC').date()}\n", flush=True)

    import collections
    print("[신호 구성]", flush=True)
    print("  레벨별 체결수:",
          {lv: sum(1 for g in sigs if lv in g["r"]) for lv in LEVELS}, flush=True)
    print("  국면분포:", dict(collections.Counter(g["regime"] for g in sigs)), flush=True)
    print("  변동성분포:", dict(collections.Counter(g["vol"] for g in sigs)), flush=True)
    print("  방향분포:", dict(collections.Counter(g["dir"] for g in sigs)), flush=True)

    # ── 1. 고정 레벨 기준선 ──
    print("\n[1] 고정 레벨 (조건 없음)", flush=True)
    fixed = {}
    for lv in LEVELS:
        fixed[lv] = eval_rule(sigs, ["*"] * len(sigs), {"*": lv})
        show(f"고정 {lv}", fixed[lv])
    for lv in LEVELS:
        print(f"      {lv} 연도별: {year_line(fixed[lv])}", flush=True)
    best_fixed_lv = max(LEVELS, key=lambda lv: fixed[lv]["sum"])
    base = fixed[best_fixed_lv]["sum"]
    print(f"  → 고정 최선 = {best_fixed_lv} ({base:+.2f}R). 이하 Δ는 이 값 대비.",
          flush=True)
    print(f"  → 현행 기본레벨 0.707 단독 = {fixed['0.707']['sum']:+.2f}R", flush=True)

    # ── 2. 매칭 기저: 국면×방향 (고정 0.707) ──
    print("\n[2] 매칭 기저 통제 — 고정 0.707 의 국면×방향 성적", flush=True)
    for rg in ["상승", "횡보", "하락", "미정"]:
        for dr in ["long", "short"]:
            sub = [g["r"]["0.707"] for g in sigs
                   if g["regime"] == rg and g["dir"] == dr and "0.707" in g["r"]]
            if not sub:
                continue
            a = np.array(sub)
            print(f"  {rg} × {dr:5s}  n={len(a):3d}  총={a.sum():+7.2f}R  "
                  f"건당={a.mean():+6.3f}R", flush=True)

    # ── 2b. 국면 셀별 레벨 성적 — 현행 규칙(상승→0.786)의 근거 자체를 확인 ──
    print("\n[2b] 국면 셀 × 레벨 (총R / 건당R / n) — 현행 규칙의 근거 점검", flush=True)
    for rg in ["상승", "횡보", "하락"]:
        parts = []
        for lv in LEVELS:
            sub = [g["r"][lv] for g in sigs if g["regime"] == rg and lv in g["r"]]
            if sub:
                a = np.array(sub)
                parts.append(f"{lv}: {a.sum():+6.2f}R/{a.mean():+5.2f}R/n{len(a)}")
            else:
                parts.append(f"{lv}: -")
        print(f"  {rg}\n      " + "\n      ".join(parts), flush=True)

    # ── 3. 조건부 규칙들 ──
    reg = [g["regime"] for g in sigs]
    vol = [g["vol"] for g in sigs]
    dr = [g["dir"] for g in sigs]
    reg_cells = ["상승", "횡보", "하락", "미정"]

    print("\n[3] 조건부 규칙 — 사전 지정(가설 1개, 그리드 아님)", flush=True)
    results = []

    # C0: 현행 라이브 규칙
    m_cur = {"상승": "0.786", "횡보": "0.707", "하락": "0.707", "미정": "0.707"}
    s_cur = eval_rule(sigs, reg, m_cur)
    show("C0 현행(상승0.786/그외0.707)", s_cur, base)
    p_cur, nm_cur = perm_fixed_rule(sigs, reg, m_cur)
    print(f"      순열 p={p_cur:.4f} (귀무평균 {nm_cur:+.2f}R) · "
          f"연도별 {year_line(s_cur)}", flush=True)
    results.append(("C0 현행", s_cur, p_cur))

    # C0R: 역방향(플라시보 대조) — 상승에 얕은 레벨
    m_rev = {"상승": "0.707", "횡보": "0.786", "하락": "0.786", "미정": "0.786"}
    s_rev = eval_rule(sigs, reg, m_rev)
    show("C0R 역방향 대조", s_rev, base)
    p_rev, _ = perm_fixed_rule(sigs, reg, m_rev)
    print(f"      순열 p={p_rev:.4f}", flush=True)
    results.append(("C0R 역방향", s_rev, p_rev))

    print("\n[4] 조건부 규칙 — 그리드 탐색 (max-통계량 순열검정으로 다중비교 통제)",
          flush=True)

    grids = [
        ("G1 국면3분할(+미정)", reg, reg_cells, 5 ** 4),
        ("G2 변동성2분할(+미정)", vol, ["고변동", "저변동", "미정"], 5 ** 3),
        ("G3 방향2분할", dr, ["long", "short"], 5 ** 2),
        ("G4 국면×방향", [f"{a}|{b}" for a, b in zip(reg, dr)],
         sorted({f"{a}|{b}" for a, b in zip(reg, dr)}), None),
        ("G5 국면×변동성", [f"{a}|{b}" for a, b in zip(reg, vol)],
         sorted({f"{a}|{b}" for a, b in zip(reg, vol)}), None),
    ]
    for name, labs, cells, ncomb in grids:
        nc = ncomb if ncomb else 5 ** len(cells)
        mapping, obs, p, nullmean, fx = perm_grid_rule(sigs, labs, cells)
        s = eval_rule(sigs, labs, mapping)
        print(f"  {name}  조합수={nc}  셀={len(cells)}", flush=True)
        cnt = collections.Counter(labs)
        print("      최적매핑: " + " / ".join(
            f"{c}→{mapping[c]}(n={cnt.get(c, 0)})" for c in cells), flush=True)
        show("      → 성적", s, base)
        print(f"      max-통계량 순열 p={p:.4f} "
              f"(귀무 max 평균 {nullmean:+.2f}R, 관측 {obs:+.2f}R)", flush=True)
        print(f"      연도별 {year_line(s)}", flush=True)
        results.append((name, s, p))

    # ── 5. 요약표 ──
    print("\n[5] 요약 (Δ = 고정 최선 대비)", flush=True)
    print(f"  {'규칙':26s} {'n':>4s} {'총R':>9s} {'건당R':>8s} "
          f"{'승률':>5s} {'Δ':>9s} {'순열p':>8s} {'판정':>6s}", flush=True)
    for lv in LEVELS:
        s = fixed[lv]
        print(f"  {'고정 ' + lv:26s} {s['n']:4d} {s['sum']:+9.2f} "
              f"{s['mean']:+8.3f} {s['wr']:4.0f}% {s['sum'] - base:+9.2f} "
              f"{'-':>8s} {'-':>6s}", flush=True)
    for name, s, p in results:
        v = "통과" if p < 0.05 else "기각"
        print(f"  {name:26s} {s['n']:4d} {s['sum']:+9.2f} {s['mean']:+8.3f} "
              f"{s['wr']:4.0f}% {s['sum'] - base:+9.2f} {p:8.4f} {v:>6s}", flush=True)

    print("\n[다중비교 고지] 이번 실행에서 시도한 조건부 규칙 = "
          f"사전지정 2개(C0·C0R) + 그리드 5종. 그리드가 실제로 뒤진 조합 수 = "
          f"{' + '.join(str(5 ** len(c)) for _, _, c, _ in grids)}. "
          "그리드 p 는 max-통계량 순열이라 이 탐색량이 이미 반영돼 있다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
