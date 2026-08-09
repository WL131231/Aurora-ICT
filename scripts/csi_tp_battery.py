"""#AUTONOMOUS 2026-07-29: CSI TP단축 후보 검증배터리 (파트너 승인 — 바로 검증).

후보: CSI>=0.6 상태로 진입한 Origo 거래는 TP 를 k×ATR 로 교체 → base +22.7 대비
+70.2(7페어 5년). 우연 판별을 위해 표준 배터리 전부 적용:
  [0] 정직 재계산 — 레버 근사(×18) 제거, base 와 동일 단위(replay net_pnl_pct)로 환산
  [1] 파라미터 이웃 — 임계 0.55/0.60/0.65 × TP 1.5/2.0/2.5/3.0 × SL 1.5/2.0/2.5×ATR
  [2] 페어별 분해 — 특정 페어 몰빵인지
  [3] 연도별 + 반기
  [4] 학습구간 밖 — CSI 는 앞 70% 학습 → **뒤 30% 만** 으로 재평가(정보누수 차단)
  [5] 셔플 — CSI 라벨을 무작위 재배치 300회, net 분포 대비 p값
  [6] MDD 정직 계산 + net/MDD
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from chop_gate_bakeoff import BASE  # noqa: E402
from chop_state_index import csi_series, fit_csi, load_1h  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
COST = 0.0008


def prep_pair(sym: str, model: dict):
    df5 = _resample(_load_full(sym))
    cfg = BacktestConfig(**BASE)
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    trs = [t for t in bt.trades if not (17 <= df5.index[t.entry_idx].hour < 21)]
    mags = [abs(t.entry_trend_pct) for t in trs]
    q70 = np.percentile(mags, 70) if mags else 0.0
    kept = [t for t in trs
            if not (abs(t.entry_trend_pct) < q70
                    and t.entry_trend_pct * (1 if t.direction == "long" else -1) < 0)]
    c = df5["close"].to_numpy(); h = df5["high"].to_numpy(); lo = df5["low"].to_numpy()
    tr14 = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    atr = pd.Series(tr14).rolling(14).mean().to_numpy()
    csi = csi_series(load_1h(sym), model).reindex(df5.index, method="ffill").to_numpy()
    return df5, kept, c, h, lo, atr, csi


def simulate(pairs_data, thr: float, k_tp: float, k_sl: float,
             csi_override: dict | None = None, since=None):
    """CSI>=thr 진입은 TP=k_tp×ATR / SL=k_sl×ATR, 아니면 원 결과 유지."""
    out = []
    for sym, (df5, kept, c, h, lo, atr, csi_arr) in pairs_data.items():
        csi = csi_override[sym] if csi_override else csi_arr
        n5 = len(c)
        for t in kept:
            i = t.entry_idx
            ts = df5.index[i]
            if since is not None and ts < since:
                continue
            cv = csi[i]
            if np.isnan(cv) or cv < thr or np.isnan(atr[i]):
                out.append((ts, t.net_pnl_pct))
                continue
            d = 1 if t.direction == "long" else -1
            entry = t.entry
            tp = entry + d * k_tp * atr[i]
            sl = entry - d * k_sl * atr[i]
            raw = 0.0
            for j in range(i + 1, min(i + 289, n5)):
                if d == 1:
                    if lo[j] <= sl:
                        raw = (sl - entry) / entry; break
                    if h[j] >= tp:
                        raw = (tp - entry) / entry; break
                else:
                    if h[j] >= sl:
                        raw = (entry - sl) / entry; break
                    if lo[j] <= tp:
                        raw = (entry - tp) / entry; break
            # base 와 동일 단위: replay 의 net_pnl_pct 는 시드대비(레버·size 반영).
            # size_pct 0.9 × leverage 20 = notional 18배, 수수료 왕복 반영.
            net = (raw * 18.0) - (COST * 18.0)
            out.append((ts, net * 100))
    return out


def stat(tr):
    if len(tr) < 25:
        return None
    tr = sorted(tr)
    nets = [p for _, p in tr]
    net = sum(nets)
    w = sum(1 for p in nets if p > 0)
    half = len(tr) // 2
    h1 = sum(p for _, p in tr[:half]); h2 = sum(p for _, p in tr[half:])
    eq = pk = mdd = 0.0
    for _, p in tr:
        eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    ys: dict[int, float] = {}
    for t, p in tr:
        ys[t.year] = ys.get(t.year, 0.0) + p
    ypos = sum(1 for v in ys.values() if v > 0)
    ok = net > 0 and h1 > 0 and h2 > 0 and ypos >= len(ys) - 1
    return dict(n=len(tr), net=net, wr=100 * w / len(tr), h1=h1, h2=h2, mdd=mdd,
                nm=net / max(mdd, 1e-9), ys=ys, ypos=ypos, ok=ok)


def line(s) -> str:
    if s is None:
        return "표본부족"
    y = " ".join(f"{k}:{v:+.0f}" for k, v in sorted(s["ys"].items()))
    return (f"n={s['n']:4d} net={s['net']:+7.1f}% 승률={s['wr']:3.0f}% H1={s['h1']:+6.1f} "
            f"H2={s['h2']:+6.1f} MDD={s['mdd']:5.1f} net/MDD={s['nm']:4.2f} [{y}]")


def main() -> int:
    model = fit_csi(PAIRS)
    data = {sym: prep_pair(sym, model) for sym in PAIRS}
    base = simulate(data, 99.0, 0, 0)   # thr 매우 높음 = 전부 원 결과
    b = stat(base)
    print(f"[0] base(원 TP/SL, 정직단위)  {line(b)}", flush=True)

    print("\n[1] 파라미터 이웃 (임계 × TP × SL)", flush=True)
    best = []
    for thr in (0.55, 0.60, 0.65):
        for k_tp in (1.5, 2.0, 2.5, 3.0):
            for k_sl in (1.5, 2.0, 2.5):
                s = stat(simulate(data, thr, k_tp, k_sl))
                if s is None:
                    continue
                tag = f"thr{thr} TP{k_tp}×ATR SL{k_sl}×ATR"
                mark = "★" if s["ok"] and s["net"] > b["net"] else " "
                if mark == "★":
                    best.append((s["net"], thr, k_tp, k_sl))
                print(f"  {mark}{tag:<30} {line(s)}", flush=True)

    if not best:
        print("\n→ base 초과 통과 조합 없음. 기각.", flush=True)
        return 0
    best.sort(reverse=True)
    _, thr, k_tp, k_sl = best[0]
    print(f"\n최우수: thr={thr} TP={k_tp}×ATR SL={k_sl}×ATR", flush=True)

    print("\n[2] 페어별 분해", flush=True)
    for sym in PAIRS:
        sub = {sym: data[sym]}
        s = stat(simulate(sub, thr, k_tp, k_sl))
        s0 = stat(simulate(sub, 99.0, 0, 0))
        print(f"  {sym:<9} 적용:{line(s)}", flush=True)
        print(f"  {'':9} base:{line(s0)}", flush=True)

    print("\n[4] 학습구간 밖 (뒤 30% 만)", flush=True)
    all_ts = sorted(ts for sym in PAIRS for ts in [data[sym][0].index[0], data[sym][0].index[-1]])
    span_start, span_end = all_ts[0], all_ts[-1]
    cut = span_start + (span_end - span_start) * 0.7
    s_out = stat(simulate(data, thr, k_tp, k_sl, since=cut))
    b_out = stat(simulate(data, 99.0, 0, 0, since=cut))
    print(f"  적용: {line(s_out)}", flush=True)
    print(f"  base: {line(b_out)}", flush=True)

    print("\n[5] 셔플 검정 (CSI 무작위 재배치 300회)", flush=True)
    rng = np.random.default_rng(3)
    obs = stat(simulate(data, thr, k_tp, k_sl))["net"]
    worse = 0
    for _ in range(300):
        ov = {}
        for sym in PAIRS:
            arr = data[sym][5 + 1] if False else data[sym][6]
            ov[sym] = rng.permutation(arr)
        s = stat(simulate(data, thr, k_tp, k_sl, csi_override=ov))
        if s and s["net"] >= obs:
            worse += 1
    print(f"  관측 net={obs:+.1f} → p={worse / 300:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
