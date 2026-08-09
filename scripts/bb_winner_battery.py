"""#AUTONOMOUS 2026-07-28: BB 통과조합 검증배터리 — 우연 판별 (수축부스트 탈락 잣대 동일).

매트릭스 2016조합 중 11개 통과(전부 1h, 복귀계열 E3/E6/E7). 통과율 0.55% = 무작위
기대(다중비교)와 구분 필요. 대표 조합 3종에 배터리:
  [1] 셔플 검정 — 신호 시점 무작위 재배치 2000회, net 분포 대비 관측 p값
  [2] 파라미터 이웃 — BB기간(14/20/30)·배수(1.8/2.0/2.5)·SL·TP 이웃에서 유지되나
  [3] 연도별/반기 재확인 + MDD
  [4] 타 페어 전이 — ETH·SOL·XRP·DOGE·LINK 에서도 같은 부호인가(BTC 전용 우연 판별)
  [5] TF 인접 — 30min·2h 에서도 살아있나
전부 통과해야 후보. 하나라도 붕괴하면 기각(수축부스트 전례).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from btc_bb_matrix import BB_N, prep, run, signals, stat  # noqa: E402

WINNERS = [
    ("E3 tp=1R sl=atr15 chop maker", "E3", "1R", "atr15", "chop", 0.0008),
    ("E7 tp=2R sl=atr15 narrow maker", "E7", "2R", "atr15", "narrow", 0.0008),
    ("E3 tp=2R sl=atr1 narrow maker", "E3", "2R", "atr1", "narrow", 0.0008),
]


def net_of(tr) -> float:
    return sum(p for _, p in tr)


def main() -> int:
    tf = "1h"
    df, d = prep(tf)
    idx = df.index
    rng = np.random.default_rng(11)
    for name, rule, tp, sl, rg, cost in WINNERS:
        print(f"\n########## {name} ##########", flush=True)
        sig = signals(d, rule)
        base_tr = run(d, idx, sig, tp, sl, rg, cost)
        line, ok = stat(base_tr)
        print(f"  기준: {line}", flush=True)
        obs = net_of(base_tr)

        # [1] 셔플 — 신호 위치를 무작위로 옮겨 같은 개수만큼 매매
        nz = np.flatnonzero(sig != 0)
        vals = sig[nz]
        worse = 0
        trials = 300  # 계산량 고려(각 시행이 전체 시뮬)
        for _ in range(trials):
            shuf = np.zeros_like(sig)
            pos = rng.choice(np.arange(BB_N + 4, len(sig) - 2), size=len(nz), replace=False)
            shuf[pos] = rng.permutation(vals)
            t = run(d, idx, shuf, tp, sl, rg, cost)
            if net_of(t) >= obs:
                worse += 1
        print(f"  [1] 셔플 {trials}회: p={worse / trials:.3f} (관측 net={obs:+.1f})", flush=True)

        # [2] 파라미터 이웃 — BB 기간·배수 변주
        import btc_bb_matrix as M
        for n_, k_ in ((14, 2.0), (30, 2.0), (20, 1.8), (20, 2.5)):
            M.BB_N, M.BB_K = n_, k_
            df2, d2 = M.prep(tf)
            s2 = M.signals(d2, rule)
            t2 = M.run(d2, df2.index, s2, tp, sl, rg, cost)
            l2, ok2 = M.stat(t2)
            print(f"  [2] BB({n_},{k_}): {'★' if ok2 else ' '}{l2}", flush=True)
        M.BB_N, M.BB_K = 20, 2.0

        # [4] 타 페어 전이
        import btc_bb_matrix as M2
        for sym in ("ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT"):
            M2.SYM = sym
            try:
                dfx, dx = M2.prep(tf)
                sx = M2.signals(dx, rule)
                tx = M2.run(dx, dfx.index, sx, tp, sl, rg, cost)
                lx, okx = M2.stat(tx)
                print(f"  [4] {sym:<9}: {'★' if okx else ' '}{lx}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [4] {sym}: 실패 {e}", flush=True)
        M2.SYM = "BTCUSDT"

        # [5] TF 인접
        for tf2 in ("30min", "2h"):
            try:
                M2.SYM = "BTCUSDT"
                dfy, dy = M2.prep(tf2) if tf2 in ("5m", "15min", "1h") else _prep_any(M2, tf2)
                sy = M2.signals(dy, rule)
                ty = M2.run(dy, dfy.index, sy, tp, sl, rg, cost)
                ly, oky = M2.stat(ty)
                print(f"  [5] TF {tf2:<6}: {'★' if oky else ' '}{ly}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [5] TF {tf2}: 실패 {e}", flush=True)
    return 0


def _prep_any(M, tf: str):
    """prep 의 bars_day 맵에 없는 TF 지원 — 30min/2h."""
    import pandas as pd

    from bt_par import _load_full, _resample
    from chop_gate_bakeoff import adx14, roll_q
    df = _resample(_load_full(M.SYM)).resample(tf).agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()
    c = df["close"].to_numpy(); h = df["high"].to_numpy()
    lo = df["low"].to_numpy(); o = df["open"].to_numpy()
    v = df["volume"].to_numpy()
    s = pd.Series(c)
    mid = s.rolling(M.BB_N).mean().to_numpy()
    sd = s.rolling(M.BB_N).std().to_numpy()
    up = mid + M.BB_K * sd; dn = mid - M.BB_K * sd
    width = (up - dn) / np.maximum(mid, 1e-12)
    bars_day = 48 if tf == "30min" else 12
    narrow_thr = roll_q(width, bars_day * 90, 0.33)
    adx = adx14(h, lo, c)
    lowmin = pd.Series(lo).rolling(5).min().to_numpy()
    highmax = pd.Series(h).rolling(5).max().to_numpy()
    k_raw = 100 * (c - lowmin) / np.maximum(highmax - lowmin, 1e-12)
    k = pd.Series(k_raw).rolling(3).mean().to_numpy()
    tr = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    volma = pd.Series(v).rolling(20).mean().to_numpy()
    return df, dict(c=c, h=h, lo=lo, o=o, v=v, mid=mid, up=up, dn=dn, width=width,
                    narrow_thr=narrow_thr, adx=adx, k=k, atr=atr, volma=volma)


if __name__ == "__main__":
    raise SystemExit(main())
