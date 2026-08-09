"""#AUTONOMOUS 2026-08-02: Origo HA 방향 신호 검증 — 진짜인가, 기저인가.

1차 탐색에서 HA 방향 일치 거래가 건당 +10~11%p 나았다(1h·4h 모두). 배포로 가기 전에
**기각 가능한 형태**로 세 가지를 확인한다. 7/29 에 확립한 검증 3종과 같은 취지다.

① 순열검정 — 같은 표본을 무작위로 같은 비율로 갈랐을 때 이만한 차이가 얼마나 흔한가.
   Origo 는 건당 편차가 크다(승률 47% · RR 1.94). 126건에서 10%p 차이는 우연일 수
   있고, 그렇다면 여기서 끝이다.

② 국면·방향 매칭 기저 — **7/29 히든 다이버전스를 기각시킨 결정타**. HA 방향은 결국
   추세 방향이라 "상승장 롱이 잘 된다"를 재확인한 것일 수 있다. 국면(일봉 추세) ×
   방향 칸 안에서도 차이가 남는지 본다. 안 남으면 HA 는 기여가 없다.

③ 지표 특이성 — HA 대신 **단순 EMA 방향**으로 같은 분할을 하면? 같은 차이가 나오면
   "HA 가 좋다"가 아니라 "두 추세 지표가 엇갈릴 때 나쁘다"는 일반 현상이고, 이미
   align 게이트가 하는 일과 겹친다.

⚠️ 불일치 거래도 흑자(+6.98%/+8.49%)라는 점이 중요하다. 게이트로 자르면 알파를
   버린다 — 살아남는다면 처방은 게이트가 아니라 **사이징**이어야 한다.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import PAIRS, run_live_parity, stat  # noqa: E402
from origo_ha_direction import ha_direction  # noqa: E402

RNG = np.random.default_rng(20260802)
N_PERM = 20000


def regime_direction(df5: pd.DataFrame, win: int = 20) -> pd.Series:
    """일봉 국면 — 20일 종가 z-score 부호. +1 상승 / -1 하락 (완결봉만)."""
    d1 = df5["close"].resample("1D").last().dropna()
    ret = d1.pct_change()
    z = (d1 - d1.rolling(win).mean()) / d1.rolling(win).std().replace(0, np.nan)
    _ = ret
    s = np.sign(z).shift(1).fillna(0.0)
    return s.reindex(df5.index, method="ffill")


def ema_direction(df5: pd.DataFrame, tf: str = "1h", span: int = 20) -> pd.Series:
    """상위 TF EMA 기울기 방향 — HA 와 대조할 단순 지표."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    htf = df5.resample(tf).agg(agg).dropna()
    ema = htf["close"].ewm(span=span, adjust=False).mean()
    d = np.sign(ema.diff()).shift(1).fillna(0.0)
    return d.reindex(df5.index, method="ffill")


def collect(tf: str):
    """거래별 (net, HA일치, EMA일치, 국면, 롱여부) 수집."""
    out = []
    for sym in PAIRS:
        df5, kept, _ = run_live_parity(sym)
        hd = ha_direction(df5, tf)
        ed = ema_direction(df5, tf)
        rg = regime_direction(df5)
        for t in kept:
            ts = df5.index[t.entry_idx]
            h_, e_, r_ = hd.get(ts, np.nan), ed.get(ts, np.nan), rg.get(ts, np.nan)
            if not np.isfinite(h_):
                continue
            long_ = str(getattr(t.direction, "value", t.direction)).lower() == "long"
            out.append({
                "net": float(t.net_pnl_pct),
                "ha_ok": (h_ > 0) == long_,
                "ema_ok": ((e_ > 0) == long_) if np.isfinite(e_) and e_ != 0 else None,
                "regime": int(r_) if np.isfinite(r_) else 0,
                "long": long_, "sym": sym, "ts": ts,
            })
    return out


def perm_test(rows, key: str) -> tuple[float, float, float]:
    """순열검정 — (실제 건당 차이, p값, 무작위 차이의 95분위)."""
    v = np.array([r["net"] for r in rows], float)
    m = np.array([bool(r[key]) for r in rows])
    if m.sum() < 5 or (~m).sum() < 5:
        return float("nan"), float("nan"), float("nan")
    obs = v[m].mean() - v[~m].mean()
    k = int(m.sum())
    diffs = np.empty(N_PERM)
    for i in range(N_PERM):
        idx = RNG.permutation(len(v))
        diffs[i] = v[idx[:k]].mean() - v[idx[k:]].mean()
    p = float((np.abs(diffs) >= abs(obs)).mean())
    return float(obs), p, float(np.percentile(np.abs(diffs), 95))


def main() -> int:
    print("=== Origo HA 방향 신호 검증 ===", flush=True)
    for tf in ("1h", "4h"):
        rows = collect(tf)
        print(f"\n[HTF = {tf}]  거래 {len(rows)}건", flush=True)

        obs, p, q95 = perm_test(rows, "ha_ok")
        print(f"  ① 순열검정  건당차이 {obs:+.2f}%p · p={p:.3f} "
              f"· 무작위 95분위 {q95:.2f}%p  → {'유의' if p < 0.05 else '**유의하지 않음**'}",
              flush=True)

        er = [r for r in rows if r["ema_ok"] is not None]
        obs_e, p_e, _ = perm_test(er, "ema_ok")
        print(f"  ③ 지표특이성  EMA 방향으로 같은 분할: 건당차이 {obs_e:+.2f}%p "
              f"· p={p_e:.3f}  (HA {obs:+.2f}%p 와 비교)", flush=True)

        print("  ② 국면×방향 칸 안에서:", flush=True)
        for rg, rl in ((1, "상승장"), (-1, "하락장")):
            for lo, ll in ((True, "롱"), (False, "숏")):
                cell = [r for r in rows if r["regime"] == rg and r["long"] == lo]
                ok = [r["net"] for r in cell if r["ha_ok"]]
                no = [r["net"] for r in cell if not r["ha_ok"]]
                if len(ok) < 3 or len(no) < 3:
                    print(f"     {rl} {ll:<2} 표본부족 (일치 {len(ok)} / 불일치 {len(no)})",
                          flush=True)
                    continue
                print(f"     {rl} {ll:<2} 일치 {np.mean(ok):+7.2f}% ({len(ok):3d}건)"
                      f"  불일치 {np.mean(no):+7.2f}% ({len(no):3d}건)"
                      f"  차이 {np.mean(ok) - np.mean(no):+7.2f}%p", flush=True)

        s_ok = stat([(r["ts"], r["net"], r["sym"]) for r in rows if r["ha_ok"]])
        s_no = stat([(r["ts"], r["net"], r["sym"]) for r in rows if not r["ha_ok"]])
        if s_ok and s_no:
            print(f"  참고: 불일치도 net {s_no['net']:+.1f}% (건당 "
                  f"{s_no['net'] / max(s_no['n'], 1):+.2f}%) — 자르면 알파를 버린다",
                  flush=True)
    print("\n판정: ①이 유의하지 않거나 ②에서 칸별 차이가 사라지면 기각.", flush=True)
    print("      ③에서 EMA 가 비슷하면 HA 특유의 정보가 아니다(align 과 중복).",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
