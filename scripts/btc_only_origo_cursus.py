"""#AUTONOMOUS 2026-07-28: BTC 단독 심층 — Origo / Cursus 두 트랙 (파트너 지시).

지금까지 7페어 합산만 봤다. BTC 만 떼어 5년 정직 검증:
  트랙1 Origo — 현행 2.2 설정 기준 + 파라미터 이웃(등급·RR·SL배수·ttl) 그리드.
    라이브게이트(NY_PM 제외 + cond_align) 적용, 연도별·양반기·MDD.
  트랙2 Cursus — 정직 하니스(플립=시가 청산, 래칫 올바른 쪽만)로 진입×청산 그리드.
    유령체결 버그 수정본 기준. BTC 만의 최적 조합이 있는지.
판정: 5년 net>0 + 양반기 흑자 + 연도 다수 흑자 → 후보. 전멸이면 BTC 전용 튜닝 여지
없음으로 종결.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import dst_trend_bt as dst  # noqa: E402
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from chop_gate_bakeoff import BASE, adx14  # noqa: E402

from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct  # noqa: E402
from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

SYM = "BTCUSDT"
LEV = 10.0  # Cursus BTC 라이브 레버
FUND = getattr(dst, "FUNDING_PER_HOUR", 0.0)


# ---------------- 트랙1: Origo ----------------
def origo_variant(df5, cfgd: dict) -> list[tuple[pd.Timestamp, float]]:
    cfg = BacktestConfig(**cfgd)
    tl = cached_setup_timeline(df5, cfg, SYM)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    trs = [t for t in bt.trades if not (17 <= df5.index[t.entry_idx].hour < 21)]
    mags = [abs(t.entry_trend_pct) for t in trs]
    q70 = np.percentile(mags, 70) if mags else 0.0
    out = []
    for t in trs:
        sgn = 1.0 if t.direction == "long" else -1.0
        if abs(t.entry_trend_pct) < q70 and t.entry_trend_pct * sgn < 0:
            continue
        out.append((df5.index[t.entry_idx], t.net_pnl_pct))
    return out


def stat(tr: list[tuple[pd.Timestamp, float]]) -> str:
    if not tr:
        return "n=0"
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
    yearly = " ".join(f"{y}:{v:+.1f}" for y, v in sorted(ys.items()))
    flag = "★" if (net > 0 and h1 > 0 and h2 > 0 and ypos >= len(ys) - 1) else " "
    return (f"{flag}n={len(tr):4d} net={net:+7.1f} 승률={100 * w / len(tr):3.0f}% "
            f"H1={h1:+6.1f} H2={h2:+6.1f} MDD={mdd:5.1f} [{yearly}]")


def track_origo(df5) -> None:
    print("\n===== 트랙1: Origo BTC 단독 =====", flush=True)
    print(f"현행 2.2         {stat(origo_variant(df5, BASE))}", flush=True)
    grid = []
    for mc in (4, 5, 6):
        for rr in (2.0, 2.5, 3.0):
            grid.append(dict(min_confluence=mc, min_rr=rr))
    for sl in (3.0, 4.0, 5.0):
        grid.append(dict(sl_dist_mult=sl))
    for ttl in (4, 6, 9):
        grid.append(dict(entry_ttl_bars=ttl))
    for ote in (0.618, 0.707, 0.786):
        grid.append(dict(ote_level=ote))
    for g in grid:
        cfgd = {**BASE, **g}
        tag = " ".join(f"{k}={v}" for k, v in g.items())
        try:
            print(f"{tag:<32} {stat(origo_variant(df5, cfgd))}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{tag:<32} 실패 {e}", flush=True)


# ---------------- 트랙2: Cursus ----------------
def cursus_variant(df1h, m1, m2, atr_n, tmult) -> list[tuple[pd.Timestamp, float]]:
    """정직 트레일 시뮬 (유령체결 수정본)."""
    out = df1h.copy()
    out["st1"] = dst._supertrend(out, m1, atr_n)
    out["st2"] = dst._supertrend(out, m2, atr_n)
    src = out["close"]
    bull = (src > out["st1"]) & (src > out["st2"])
    bear = (src < out["st1"]) & (src < out["st2"])
    buy = np.concatenate([[False], (bull & ~bull.shift(1, fill_value=False)).to_numpy()[:-1]])
    sell = np.concatenate([[False], (bear & ~bear.shift(1, fill_value=False)).to_numpy()[:-1]])
    trail = dst._supertrend(out, tmult, atr_n).to_numpy()
    h = out["high"].to_numpy(); lo = out["low"].to_numpy()
    c = out["close"].to_numpy(); o = out["open"].to_numpy()
    idx = out.index
    trades = []
    side = 0
    entry = stop = 0.0
    e_i = 0
    for i in range(1, len(c)):
        ln = trail[i]
        if side != 0:
            d = side
            hit = (d == 1 and lo[i] <= stop) or (d == -1 and h[i] >= stop)
            rev = (d == 1 and sell[i]) or (d == -1 and buy[i])
            flip = (not np.isnan(ln)) and ((d == 1 and ln > c[i]) or (d == -1 and ln < c[i]))
            if hit or rev or flip:
                exit_raw = stop if hit else o[i]
                slp = slip_pct(h[i], lo[i], c[i])
                px = apply_slippage(exit_raw, "long" if d == 1 else "short", "exit", slp)
                raw = (px - entry) / entry * d
                net, _ = apply_costs(raw, 0.9, LEV)
                net -= (i - e_i) * FUND * 0.9 * LEV
                trades.append((idx[i], net * 100))
                side = 0
            else:
                if d == 1 and (not np.isnan(ln)) and ln <= c[i]:
                    stop = max(stop, ln)
                elif d == -1 and (not np.isnan(ln)) and ln >= c[i]:
                    stop = min(stop, ln)
        if side == 0:
            d = 1 if buy[i] else (-1 if sell[i] else 0)
            if d != 0:
                slp = slip_pct(h[i], lo[i], c[i])
                entry = apply_slippage(o[i], "long" if d == 1 else "short", "entry", slp)
                side = d
                base_sl = entry * (1 - 0.02 * d)
                stop = base_sl if np.isnan(ln) else (
                    max(min(ln, entry * 0.999), base_sl) if d == 1
                    else min(max(ln, entry * 1.001), base_sl))
                e_i = i
    return trades


def track_cursus(df1h) -> None:
    print("\n===== 트랙2: Cursus BTC 단독 (정직 하니스) =====", flush=True)
    for m1 in (1.5, 2.0, 3.0):
        for m2 in (3.0, 4.0, 6.0):
            if m1 >= m2:
                continue
            for atr_n in (10, 14, 21):
                for tm in (3.0, 6.0, 8.0):
                    tag = f"ST{m1}/{m2} ATR{atr_n} t{tm}"
                    try:
                        print(f"{tag:<24} {stat(cursus_variant(df1h, m1, m2, atr_n, tm))}",
                              flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"{tag:<24} 실패 {e}", flush=True)


def main() -> int:
    df5 = _resample(_load_full(SYM))
    track_origo(df5)
    track_cursus(dst._load_1h(SYM))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
