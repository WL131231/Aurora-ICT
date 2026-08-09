"""#AUTONOMOUS 2026-07-27: Cursus 재검증 2단계 — 상위 조합 5m 정밀 + 배터리.

1단계 스크린(정직 1h): ST×/4.0 + trail6 클러스터가 고원 형성(19/72 통과, 상위 7중 6).
현행 라이브(ST2/3+래더)는 -2784 최하위권. 2단계 = 상위 4조합을 신호 1h·체결 5m
경로로 재검증: 트레일 라인 래칫(올바른 쪽만)·플립=5m 시가 청산·초기 SL 2% 바닥·
REVERSE·수수료+슬리피지+펀딩. 배터리: 연도별·페어별·H1/H2·MDD.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import dst_trend_bt as dst  # noqa: E402

from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct  # noqa: E402

COMBOS = [
    ("ST3/4 ATR14 t6", 3.0, 4.0, 14, 6.0),
    ("ST2/4 ATR14 t6", 2.0, 4.0, 14, 6.0),
    ("ST1.5/4 ATR14 t6", 1.5, 4.0, 14, 6.0),
    ("ST2/6 ATR21 t3", 2.0, 6.0, 21, 3.0),
]
FUND = getattr(dst, "FUNDING_PER_HOUR", 0.0)


def run_pair(sym: str, m1, m2, atr_n, tmult) -> list[tuple[pd.Timestamp, float, str]]:
    lev = 10.0 if sym.startswith("BTC") else 7.0
    from bt_par import _load_full, _resample
    df1h = dst._load_1h(sym)
    out = df1h.copy()
    out["st1"] = dst._supertrend(out, m1, atr_n)
    out["st2"] = dst._supertrend(out, m2, atr_n)
    src = out["close"]
    bull = (src > out["st1"]) & (src > out["st2"])
    bear = (src < out["st1"]) & (src < out["st2"])
    buy1 = (bull & ~bull.shift(1, fill_value=False)).to_numpy()
    sell1 = (bear & ~bear.shift(1, fill_value=False)).to_numpy()
    trail1 = dst._supertrend(out, tmult, atr_n).to_numpy()
    c1 = out["close"].to_numpy()
    lag = out.index + pd.Timedelta(hours=1)
    df5 = _resample(_load_full(sym))
    df5 = df5[(df5.index >= out.index[0]) & (df5.index <= out.index[-1] + pd.Timedelta(hours=1))]
    o5 = df5["open"].to_numpy(); h5 = df5["high"].to_numpy(); lo5 = df5["low"].to_numpy()
    idx5 = df5.index

    def to5(a, fill=np.nan):
        return pd.Series(a, index=lag).reindex(idx5, method="ffill").fillna(fill).to_numpy()

    buy5 = to5(buy1, False)
    sell5 = to5(sell1, False)
    line5 = to5(trail1)
    c1_5 = to5(c1)  # 직전 완결 1h 종가 (플립 판정 기준)
    sid5 = to5(np.arange(len(out), dtype=float), -1)
    n = len(o5)
    trades = []
    side = 0
    entry = stop = 0.0
    e_i = 0
    last_sid = -1.0
    for i in range(n):
        new_sig = 0
        if sid5[i] != last_sid and sid5[i] >= 0:
            if bool(buy5[i]):
                new_sig = 1
            elif bool(sell5[i]):
                new_sig = -1
            last_sid = sid5[i]
        ts = idx5[i]
        ln = line5[i]
        if side != 0:
            d = side
            rev = new_sig == -d
            flip = (not np.isnan(ln)) and ((d == 1 and ln > c1_5[i]) or (d == -1 and ln < c1_5[i]))
            hit = (d == 1 and lo5[i] <= stop) or (d == -1 and h5[i] >= stop)
            if hit or rev or flip:
                exit_raw = stop if hit else o5[i]
                slp = slip_pct(h5[i], lo5[i], o5[i])
                px = apply_slippage(exit_raw, "long" if d == 1 else "short", "exit", slp)
                raw = (px - entry) / entry * d
                net, _ = apply_costs(raw, 0.9, lev)
                net -= ((i - e_i) / 12.0) * FUND * 0.9 * lev  # 5m→시간 환산 펀딩
                trades.append((ts, net * 100, sym))
                side = 0
            else:
                if d == 1 and (not np.isnan(ln)) and ln <= c1_5[i]:
                    stop = max(stop, ln)
                elif d == -1 and (not np.isnan(ln)) and ln >= c1_5[i]:
                    stop = min(stop, ln)
        if side == 0 and new_sig != 0:
            slp = slip_pct(h5[i], lo5[i], o5[i])
            entry = apply_slippage(o5[i], "long" if new_sig == 1 else "short", "entry", slp)
            side = new_sig
            base_sl = entry * (1 - 0.02 * side)
            if not np.isnan(ln):
                stop = max(min(ln, entry * 0.999), base_sl) if side == 1 else \
                       min(max(ln, entry * 1.001), base_sl)
            else:
                stop = base_sl
            e_i = i
    return trades


def report(tag: str, allt: list) -> None:
    allt.sort()
    nets = [p for _, p, _ in allt]
    net = sum(nets)
    w = sum(1 for p in nets if p > 0)
    half = len(allt) // 2
    h1 = sum(p for _, p, _ in allt[:half]); h2 = sum(p for _, p, _ in allt[half:])
    eq = pk = mdd = 0.0
    for _, p, _ in allt:
        eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    ys: dict[int, float] = {}
    for t, p, _ in allt:
        ys[t.year] = ys.get(t.year, 0.0) + p
    yearly = " ".join(f"{y}:{v:+.0f}" for y, v in sorted(ys.items()))
    print(f"\n■ {tag}: n={len(allt)} net={net:+.1f}% 승률={100 * w / len(allt):.0f}% "
          f"H1={h1:+.1f} H2={h2:+.1f} MDD={mdd:.1f} net/MDD={net / max(mdd, 1e-9):.2f}", flush=True)
    print(f"  연도별: {yearly}", flush=True)
    per = {}
    for _, p, s in allt:
        per[s] = per.get(s, 0.0) + p
    print("  페어별: " + " ".join(f"{k.replace('USDT',''):s}:{v:+.0f}" for k, v in per.items()),
          flush=True)


def main() -> int:
    for tag, m1, m2, atr_n, tm in COMBOS:
        allt = []
        for sym in dst.PAIRS:
            allt += run_pair(sym, m1, m2, atr_n, tm)
        report(tag, allt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
