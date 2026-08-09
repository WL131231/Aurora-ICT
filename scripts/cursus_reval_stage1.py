"""#AUTONOMOUS 2026-07-27: Cursus 재검증 1단계 — 정직 하니스 굵은 스크린 (파트너 승인).

배경: 6/25 출범검증이 유령체결 버그(트레일ST 플립 시 반대편 라인가 청산)로 부풀림 —
정직 수정 시 전 연도 음수. 엔진을 백지에서 재탐색한다.

1단계: 1h 봉 정직 시뮬(플립=시가 청산, 래칫은 올바른 쪽만)로 진입×청산×SL 그리드
스크린. 비용 = 기존 cost 모듈(수수료+슬리피지+펀딩) 동일.
  진입: ST1×{1.5,2,3} × ST2×{3,4,6}(ST1<ST2 만) × ATR{10,14,21}
  청산: trail×{3,6}(정직 래칫) / ladder(SL2% 4분할 1~4% — 1h 봉이라 SL우선 보수)
  SL:   trail 엔진은 초기 2% 바닥 고정.
통과 기준(스크린): 5년 net>0 + H1/H2 모두 >0. 통과자만 2단계(5m 정밀+배터리).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import dst_trend_bt as dst  # noqa: E402 — _load_1h/_atr/_supertrend/cost 상수 재사용

from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct  # noqa: E402

PAIRS = dst.PAIRS
FUND = dst.FUNDING_PER_HOUR if hasattr(dst, "FUNDING_PER_HOUR") else 0.0


def signals(df: pd.DataFrame, m1: float, m2: float, atr_n: int) -> pd.DataFrame:
    out = df.copy()
    out["st1"] = dst._supertrend(out, m1, atr_n)
    out["st2"] = dst._supertrend(out, m2, atr_n)
    src = out["close"]
    bull = (src > out["st1"]) & (src > out["st2"])
    bear = (src < out["st1"]) & (src < out["st2"])
    out["buy_sig"] = bull & ~bull.shift(1, fill_value=False)
    out["sell_sig"] = bear & ~bear.shift(1, fill_value=False)
    return out


def run_trail(df, m1, m2, atr_n, trail_mult, lev) -> list[tuple[pd.Timestamp, float]]:
    """정직 트레일 시뮬 — 플립=다음봉 시가 청산, 래칫 올바른 쪽만, 초기 SL 2% 바닥."""
    sig = signals(df, m1, m2, atr_n)
    trail = dst._supertrend(sig, trail_mult, atr_n).to_numpy()
    h = sig["high"].to_numpy(); lo = sig["low"].to_numpy(); c = sig["close"].to_numpy()
    o = sig["open"].to_numpy(); idx = sig.index
    buy = np.concatenate([[False], sig["buy_sig"].to_numpy()[:-1]])
    sell = np.concatenate([[False], sig["sell_sig"].to_numpy()[:-1]])
    trades = []
    side = 0
    entry = stop = 0.0
    e_i = 0
    for i in range(1, len(c)):
        s_now = trail[i]
        if side != 0:
            if side == 1:
                hit = lo[i] <= stop
                rev = bool(sell[i])
                flip = (not np.isnan(s_now)) and s_now > c[i]
            else:
                hit = h[i] >= stop
                rev = bool(buy[i])
                flip = (not np.isnan(s_now)) and s_now < c[i]
            if hit or rev or flip:
                exit_raw = stop if hit else o[i]
                slp = slip_pct(h[i], lo[i], c[i])
                px = apply_slippage(exit_raw, "long" if side == 1 else "short", "exit", slp)
                raw = (px - entry) / entry * side
                net, _ = apply_costs(raw, 0.9, lev)
                net -= (i - e_i) * FUND * 0.9 * lev
                trades.append((idx[i], net * 100))
                side = 0
            else:
                if side == 1 and (not np.isnan(s_now)) and s_now <= c[i]:
                    stop = max(stop, s_now)
                elif side == -1 and (not np.isnan(s_now)) and s_now >= c[i]:
                    stop = min(stop, s_now)
        if side == 0:
            d = 1 if buy[i] else (-1 if sell[i] else 0)
            if d != 0:
                slp = slip_pct(h[i], lo[i], c[i])
                entry = apply_slippage(o[i], "long" if d == 1 else "short", "entry", slp)
                side = d
                base_sl = entry * (1 - 0.02 * d)
                stop = base_sl if np.isnan(s_now) else (
                    max(min(s_now, entry * 0.999), base_sl) if d == 1
                    else min(max(s_now, entry * 1.001), base_sl))
                e_i = i
    return trades


def run_ladder(df, m1, m2, atr_n, lev) -> list[tuple[pd.Timestamp, float]]:
    """래더(현행 라이브형) 1h 보수 시뮬 — SL2%, TP1~4% 4분할, 래더, REVERSE."""
    sig = signals(df, m1, m2, atr_n)
    h = sig["high"].to_numpy(); lo = sig["low"].to_numpy(); c = sig["close"].to_numpy()
    o = sig["open"].to_numpy(); idx = sig.index
    buy = np.concatenate([[False], sig["buy_sig"].to_numpy()[:-1]])
    sell = np.concatenate([[False], sig["sell_sig"].to_numpy()[:-1]])
    trades = []
    side = 0
    entry = sl = 0.0
    tps = []
    filled = []
    rem = 1.0
    for i in range(1, len(c)):
        if side != 0:
            d = side
            if (d == 1 and sell[i]) or (d == -1 and buy[i]):
                raw = (o[i] - entry) / entry * d
                net, _ = apply_costs(raw, 0.9, lev)
                trades.append((idx[i], net * rem * 100))
                side = 0
            elif (d == 1 and lo[i] <= sl) or (d == -1 and h[i] >= sl):
                raw = (sl - entry) / entry * d
                net, _ = apply_costs(raw, 0.9, lev)
                trades.append((idx[i], net * rem * 100))
                side = 0
            else:
                for k in range(4):
                    if filled[k] or rem <= 0:
                        continue
                    if (d == 1 and h[i] >= tps[k]) or (d == -1 and lo[i] <= tps[k]):
                        raw = (tps[k] - entry) / entry * d
                        net, _ = apply_costs(raw, 0.9, lev)
                        trades.append((idx[i], net * 0.25 * 100))
                        filled[k] = True
                        rem = max(0.0, rem - 0.25)
                        if k == 1:
                            sl = tps[0]
                        elif k == 2:
                            sl = tps[1]
                if rem <= 1e-9:
                    side = 0
        if side == 0:
            d = 1 if buy[i] else (-1 if sell[i] else 0)
            if d != 0:
                entry = o[i]
                side = d
                sl = entry * (1 - 0.02 * d)
                tps = [entry * (1 + lv * d) for lv in (0.01, 0.02, 0.03, 0.04)]
                filled = [False] * 4
                rem = 1.0
    return trades


def stats(tr):
    if not tr:
        return None
    tr.sort()
    nets = [p for _, p in tr]
    net = sum(nets)
    half = len(tr) // 2
    h1 = sum(p for _, p in tr[:half]); h2 = sum(p for _, p in tr[half:])
    eq = pk = mdd = 0.0
    for _, p in tr:
        eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    return dict(n=len(tr), net=net, h1=h1, h2=h2, mdd=mdd)


def main() -> int:
    combos = []
    for m1 in (1.5, 2.0, 3.0):
        for m2 in (3.0, 4.0, 6.0):
            if m1 >= m2:
                continue
            for atr_n in (10, 14, 21):
                combos.append((m1, m2, atr_n))
    dfs = {sym: dst._load_1h(sym) for sym in PAIRS}
    levs = {sym: (10.0 if sym.startswith("BTC") else 7.0) for sym in PAIRS}
    results = []
    for (m1, m2, atr_n) in combos:
        for exit_name in ("trail3", "trail6", "ladder"):
            allt = []
            for sym in PAIRS:
                if exit_name == "ladder":
                    allt += run_ladder(dfs[sym], m1, m2, atr_n, levs[sym])
                else:
                    tm = 3.0 if exit_name == "trail3" else 6.0
                    allt += run_trail(dfs[sym], m1, m2, atr_n, tm, levs[sym])
            st = stats(allt)
            if st is None:
                continue
            tag = f"ST{m1}/{m2} ATR{atr_n} {exit_name}"
            ok = st["net"] > 0 and st["h1"] > 0 and st["h2"] > 0
            results.append((st["net"], tag, st, ok))
            print(f"{'★' if ok else ' '}{tag:<26} n={st['n']:5d} net={st['net']:+9.1f} "
                  f"H1={st['h1']:+8.1f} H2={st['h2']:+8.1f} MDD={st['mdd']:7.1f}", flush=True)
    print("\n=== 상위 10 (net 순) ===", flush=True)
    for net, tag, st, ok in sorted(results, reverse=True)[:10]:
        print(f"{'★' if ok else ' '}{tag:<26} net={net:+9.1f} H1={st['h1']:+8.1f} H2={st['h2']:+8.1f}",
              flush=True)
    npass = sum(1 for *_, ok in results if ok)
    print(f"\n통과(5y+ & 양반기+): {npass}/{len(results)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
