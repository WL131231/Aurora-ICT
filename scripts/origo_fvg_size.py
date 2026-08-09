"""#AUTONOMOUS 2026-08-02: Origo FVG 크기가 성적을 가르는가 (탐색 + 검증).

발단은 HA 실험이다. HA 로 FVG 를 찾으면 354→47건으로 무너지는데, **살아남은 47건의
질이 오히려 좋았다**(승률 47→56%, RR 1.94→2.17, MDD 533→151). HA 가 갭을 메우니까
살아남은 건 "평활해도 안 메워질 만큼 큰 갭" 이다. 그렇다면 HA 로 갈아탈 게 아니라
**현행 FVG 에 크기 축을 보는 것**으로 같은 정보를 얻을 수 있다 — 좌표 문제도 없고
셋업도 안 깎인다.

측정에서 지킬 것:
① **R 단위로 잰다.** Origo 는 SL 이 구조 기반(ATR×4)이라 큰 갭 셋업은 SL 도 멀다.
   %수익이 커 보여도 R 로는 같을 수 있다(7/30 에 Trade 에 entry_sl 을 넣은 이유).
② **변동성 중복 확인.** 갭 크기는 변동성 대리변수인데 우리는 이미 regime_filter
   (변동성 q33 하단 차단)를 쓴다. 상관이 높으면 같은 걸 두 번 거르는 것이다.
③ **순열검정.** 상위 분위만 남기면 뭘 기준으로 하든 분산이 준다. 무작위 분할과
   구분되지 않으면 기각이다(HA 방향 신호가 방금 이걸로 기각됐다 — p=0.37).

기각 기준(먼저 박아둔다): 분위별 단조성이 없거나 · regime 상관 0.7 초과 ·
순열검정 p>=0.05 · R 기준으로 차이 소멸 → 어느 하나라도 걸리면 끝.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import PAIRS, run_live_parity  # noqa: E402

RNG = np.random.default_rng(20260802)
N_PERM = 20000


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR — 갭 크기 정규화 기준(페어·시기 무관 비교를 위해)."""
    h, lo, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([h - lo, (h - pc).abs(), (lo - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def collect():
    """거래별 (R배수, 갭/ATR, 변동성, 방향, 연도) 수집.

    갭 크기는 셋업의 진입가와 손절가 거리로 근사한다 — Origo 의 SL 은 FVG 구조에서
    나오므로 이 거리가 곧 셋업이 딛고 선 구조의 크기다. (FVG 원본 좌표를 timeline
    에서 다시 꺼내는 것보다 정합적이다 — 실제 주문에 쓰인 값이라서.)
    """
    rows = []
    for sym in PAIRS:
        df5, kept, _ = run_live_parity(sym)
        a = atr(df5)
        vol = (df5["close"].pct_change().rolling(288).std() * 100.0)  # 하루치 변동성
        for t in kept:
            i = t.entry_idx
            ts = df5.index[i]
            av = float(a.iloc[i]) if i < len(a) else np.nan
            if not np.isfinite(av) or av <= 0:
                continue
            risk = abs(float(t.entry) - float(t.entry_sl))
            if risk <= 0:
                continue
            r = float(t.raw_pnl_pct) * float(t.entry) / risk
            rows.append({
                "r": r, "net": float(t.net_pnl_pct),
                "size": risk / av,                       # 구조 크기 / ATR
                "vol": float(vol.iloc[i]) if i < len(vol) else np.nan,
                "long": str(getattr(t.direction, "value", t.direction)).lower() == "long",
                "year": ts.year, "sym": sym,
            })
    return rows


def perm(vals: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """순열검정 — (관측 평균차, p값)."""
    if mask.sum() < 5 or (~mask).sum() < 5:
        return float("nan"), float("nan")
    obs = vals[mask].mean() - vals[~mask].mean()
    k = int(mask.sum())
    d = np.empty(N_PERM)
    for i in range(N_PERM):
        idx = RNG.permutation(len(vals))
        d[i] = vals[idx[:k]].mean() - vals[idx[k:]].mean()
    return float(obs), float((np.abs(d) >= abs(obs)).mean())


def main() -> int:
    rows = collect()
    n = len(rows)
    print(f"=== Origo FVG 구조 크기 × 성적 === (라이브 정합 {n}건)\n", flush=True)
    if n < 40:
        print("표본 부족 — 판정 불가", flush=True)
        return 1

    size = np.array([r["size"] for r in rows])
    rmul = np.array([r["r"] for r in rows])
    net = np.array([r["net"] for r in rows]) * 100.0
    vol = np.array([r["vol"] for r in rows])

    print("① 크기 5분위별 성적 (R 배수 = SL 대비)", flush=True)
    print(f"  {'분위':<6}{'구조/ATR':>10}{'n':>5}{'건당R':>8}{'승률':>6}{'건당net':>9}",
          flush=True)
    qs = np.quantile(size, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for i in range(5):
        lo_, hi_ = qs[i], qs[i + 1]
        m = (size >= lo_) & (size <= hi_ if i == 4 else size < hi_)
        if m.sum() < 3:
            print(f"  Q{i + 1:<5}{lo_:>10.2f}{m.sum():>5} 표본부족", flush=True)
            continue
        wr = 100.0 * (rmul[m] > 0).mean()
        print(f"  Q{i + 1:<5}{lo_:>6.2f}~{hi_:<4.2f}{m.sum():>4}{rmul[m].mean():>8.2f}"
              f"{wr:>5.0f}%{net[m].mean():>8.2f}%", flush=True)

    ok = np.isfinite(vol)
    corr = float(np.corrcoef(size[ok], vol[ok])[0, 1]) if ok.sum() > 10 else float("nan")
    print(f"\n② 변동성 중복  corr(구조크기, 변동성) = {corr:+.3f}"
          f"  → {'중복 — 기존 regime 게이트와 같은 축' if abs(corr) > 0.7 else '독립적'}",
          flush=True)

    print("\n③ 순열검정 (상위 X% vs 나머지)", flush=True)
    for pct in (0.5, 0.6, 0.7, 0.8):
        thr = np.quantile(size, pct)
        m = size >= thr
        o_r, p_r = perm(rmul, m)
        o_n, p_n = perm(net, m)
        print(f"  상위 {100 * (1 - pct):>2.0f}%  R차이 {o_r:+.3f} (p={p_r:.3f})"
              f"   net차이 {o_n:+.2f}%p (p={p_n:.3f})"
              f"   {'유의' if p_r < 0.05 else '무의미'}", flush=True)

    print("\n④ 롱/숏 분리 (상위 50% 기준)", flush=True)
    thr = np.quantile(size, 0.5)
    for lab, sel in (("롱", True), ("숏", False)):
        idx = np.array([r["long"] == sel for r in rows])
        if idx.sum() < 10:
            continue
        big = idx & (size >= thr)
        sml = idx & (size < thr)
        if big.sum() < 5 or sml.sum() < 5:
            continue
        print(f"  {lab}  큰구조 {rmul[big].mean():+.2f}R ({big.sum()}건)"
              f"   작은구조 {rmul[sml].mean():+.2f}R ({sml.sum()}건)"
              f"   차이 {rmul[big].mean() - rmul[sml].mean():+.2f}R", flush=True)

    print("\n⑤ 연도 일관성 (상위 50% − 하위 50%, R)", flush=True)
    for y in sorted({r["year"] for r in rows}):
        m = np.array([r["year"] == y for r in rows])
        b, s = m & (size >= thr), m & (size < thr)
        if b.sum() < 3 or s.sum() < 3:
            print(f"  {y}  표본부족", flush=True)
            continue
        print(f"  {y}  {rmul[b].mean() - rmul[s].mean():+.2f}R"
              f"  (큰 {b.sum()}건 / 작은 {s.sum()}건)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
