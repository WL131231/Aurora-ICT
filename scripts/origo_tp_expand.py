"""#AUTONOMOUS 2026-07-29: [C] Origo TP 확대 단독 검증 — 횡보와 무관한 청산 파라미터 연구.

CSI TP단축 배터리에서 파생 발견: TP 를 ATR 기반으로 **늘릴수록** net 증가(1.5→2.5×ATR
단조). 즉 개선의 정체가 "횡보 인식"이 아니라 "**원래 TP 가 짧았다**"일 가능성.
여기선 CSI 를 완전히 빼고 **전 거래 일괄 적용**으로 검증한다(횡보 무관).

변형:
  base            : 현행(유동성 TP / min_rr 2.0 R 기반)
  TP k×ATR        : k ∈ {1.5, 2.0, 2.5, 3.0, 4.0}, SL 은 원 SL 유지(원 위험폭)
  TP k×R          : R = |entry-SL| 배수 {2.5, 3.0, 4.0} — R 기반 확장(ATR 아닌)
검증배터리: 페어별·연도별·양반기·MDD·net/MDD + 셔플(무작위 TP 배수) + 표본 충분성.
판정: 전 페어 다수 개선 + 연도 일관 + 2022 같은 특정연도 붕괴 없음.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from chop_gate_bakeoff import BASE  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
COST = 0.0008
NOTIONAL = 18.0   # size 0.9 × lev 20 (replay 정합)


def prep(sym: str):
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
    return df5, kept, c, h, lo, atr


def sim(data, mode: str, k: float):
    """mode: 'base' | 'atr' | 'r'. 원 SL 유지, TP 만 교체."""
    out = []
    for sym, (df5, kept, c, h, lo, atr) in data.items():
        n5 = len(c)
        for t in kept:
            i = t.entry_idx
            ts = df5.index[i]
            if mode == "base":
                out.append((ts, t.net_pnl_pct, sym))
                continue
            d = 1 if t.direction == "long" else -1
            entry = t.entry
            # Trade 에 stop_loss 필드 없음 → BASE 설정(sl_dist_mult=4.0)로 원 위험폭 복원.
            # entry_atr_pct(진입 시 ATR/진입가 %)가 있으면 그것을, 없으면 5m ATR 사용.
            if np.isnan(atr[i]):
                out.append((ts, t.net_pnl_pct, sym))
                continue
            atr_abs = (t.entry_atr_pct / 100.0 * entry) if getattr(t, "entry_atr_pct", 0) else atr[i]
            risk = 4.0 * atr_abs
            sl = entry - d * risk
            if risk <= 0:
                out.append((ts, t.net_pnl_pct, sym))
                continue
            tp = entry + d * (k * atr[i] if mode == "atr" else k * risk)
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
            out.append((ts, (raw * NOTIONAL - COST * NOTIONAL) * 100, sym))
    return out


def stat(tr):
    if len(tr) < 25:
        return None
    tr = sorted(tr)
    nets = [p for _, p, _ in tr]
    net = sum(nets)
    w = sum(1 for p in nets if p > 0)
    half = len(tr) // 2
    h1 = sum(p for _, p, _ in tr[:half]); h2 = sum(p for _, p, _ in tr[half:])
    eq = pk = mdd = 0.0
    for _, p, _ in tr:
        eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    ys: dict[int, float] = {}
    for t, p, _ in tr:
        ys[t.year] = ys.get(t.year, 0.0) + p
    ypos = sum(1 for v in ys.values() if v > 0)
    return dict(n=len(tr), net=net, wr=100 * w / len(tr), h1=h1, h2=h2, mdd=mdd,
                nm=net / max(mdd, 1e-9), ys=ys, ypos=ypos,
                ok=net > 0 and h1 > 0 and h2 > 0 and ypos >= len(ys) - 1)


def line(s):
    if s is None:
        return "표본부족"
    y = " ".join(f"{k}:{v:+.0f}" for k, v in sorted(s["ys"].items()))
    return (f"n={s['n']:4d} net={s['net']:+7.1f}% 승률={s['wr']:3.0f}% H1={s['h1']:+6.1f} "
            f"H2={s['h2']:+6.1f} MDD={s['mdd']:5.1f} net/MDD={s['nm']:5.2f} [{y}]")


def main() -> int:
    data = {sym: prep(sym) for sym in PAIRS}
    b = stat(sim(data, "base", 0))
    print(f"base(현행)              {line(b)}", flush=True)
    print("\n[1] TP = k×ATR (SL 원본 유지)", flush=True)
    results = []
    for k in (1.5, 2.0, 2.5, 3.0, 4.0):
        s = stat(sim(data, "atr", k))
        mark = "★" if s and s["ok"] and s["net"] > b["net"] else " "
        print(f"  {mark}TP {k}×ATR            {line(s)}", flush=True)
        if mark == "★":
            results.append(("atr", k, s["net"]))
    print("\n[2] TP = k×R (원 위험폭 배수)", flush=True)
    for k in (2.5, 3.0, 4.0):
        s = stat(sim(data, "r", k))
        mark = "★" if s and s["ok"] and s["net"] > b["net"] else " "
        print(f"  {mark}TP {k}×R              {line(s)}", flush=True)
        if mark == "★":
            results.append(("r", k, s["net"]))
    if not results:
        print("\n→ base 초과 통과 없음. 기각.", flush=True)
        return 0
    results.sort(key=lambda x: -x[2])
    mode, k, _ = results[0]
    print(f"\n최우수: TP {k}×{'ATR' if mode == 'atr' else 'R'}", flush=True)
    print("\n[3] 페어별 분해 (적용 vs base)", flush=True)
    for sym in PAIRS:
        sub = {sym: data[sym]}
        sa = stat(sim(sub, mode, k)); sb = stat(sim(sub, "base", 0))
        print(f"  {sym:<9} 적용:{line(sa)}", flush=True)
        print(f"  {'':<9} base:{line(sb)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
