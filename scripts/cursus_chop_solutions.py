"""#AUTONOMOUS 2026-07-27: Cursus 횡보 해법 3계열 비교 (파트너: 횡보 최우선, 2~3개 추리기).

배경: 라이브 해부 — Cursus 저ADX 출혈(avg -1.96/건) vs 고ADX 독식(+1.93). 단순
진입skip 게이트는 REVERSE 체인 특성상 추세 다리를 놓쳐 악화(실측). 해법 후보:
  C1 사이즈 변조 — 저ADX 진입은 비중 축소(체인 유지, 출혈만 감쇠)
      C1a: ADX<20 → 0.5x / C1b: ADX<20 → 0.25x / C1c: 계단(18↓0.25, 22↓0.5)
  C2 상태 재진입 게이트 — 저ADX skip 하되, ADX 회복 시 "정렬 상태"로 즉시 재진입
      (크로스오버 재대기 안 함 — 체인 공백 문제 해소) C2a: thr20 / C2b: thr22
  C3 저ADX 조기익절 모드 — 저ADX 진입은 TP 그리드 절반(0.5/1/1.5/2%)+TP2 전량
      (횡보 좁은 진폭에 맞춘 빠른 회수) C3a: thr20
엔진: 신호 1h · 체결 5m 경로(라이브 정합 검증본). 표: 종합+연도별+H1/H2+MDD,
+2026-06-15 이후(라이브 대조 구간) 별도 컬럼.
"""
from __future__ import annotations

import sys

import dst_trend_bt as dst
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from chop_gate_bakeoff import adx14  # noqa: E402

from aurora.backtest.cost import apply_costs  # noqa: E402

SL_PCT = 0.02
TP_LV = (0.01, 0.02, 0.03, 0.04)
TP_FRAC = 0.25
LIVE_CMP = pd.Timestamp("2026-06-15", tz="UTC")


def run_pair(sym: str, mode: str) -> list[tuple[pd.Timestamp, float]]:
    """모드별 시뮬 — (청산시각, 시드대비 net%). 신호 1h·체결 5m."""
    lev = 10.0 if sym.startswith("BTC") else 7.0
    from bt_par import _load_full, _resample
    df1h = dst._load_1h(sym)
    sig = dst._signals(df1h)
    h1, l1, c1 = (sig[k].to_numpy() for k in ("high", "low", "close"))
    adx1 = adx14(h1, l1, c1)
    st1 = sig["st1"].to_numpy(); st2 = sig["st2"].to_numpy()
    bull_state = (c1 > st1) & (c1 > st2)
    bear_state = (c1 < st1) & (c1 < st2)
    def _persist(state, k):
        r = state.copy()
        for j in range(1, k):
            r = r & np.roll(state, j)
        r[:k] = False
        # "직전 봉까지 k봉 연속 정렬" 이면서 그 시작 봉에서 신규 성립 = 확인 진입 신호
        fresh = r & ~np.concatenate([[False], r[:-1]])
        return fresh
    conf_buy = {2: _persist(bull_state, 2), 3: _persist(bull_state, 3)}
    conf_sell = {2: _persist(bear_state, 2), 3: _persist(bear_state, 3)}
    lag = sig.index + pd.Timedelta(hours=1)
    df5 = _resample(_load_full(sym))
    df5 = df5[(df5.index >= sig.index[0]) & (df5.index <= sig.index[-1] + pd.Timedelta(hours=1))]
    o5 = df5["open"].to_numpy(); h5 = df5["high"].to_numpy(); lo5 = df5["low"].to_numpy()
    idx5 = df5.index

    def to5(arr, fill=np.nan):
        return pd.Series(arr, index=lag).reindex(idx5, method="ffill").fillna(fill).to_numpy()

    # 방향별 원시 밴드(hl2 ∓ mult×ATR) — 포지션 안에서 래칫(롱: 하단밴드 최고치).
    _atr1 = dst._atr(sig, dst.ATR_PERIOD).to_numpy()
    _hl2 = (h1 + l1) / 2
    lo_band3 = _hl2 - 3.0 * _atr1; up_band3 = _hl2 + 3.0 * _atr1
    lo_band6 = _hl2 - 6.0 * _atr1; up_band6 = _hl2 + 6.0 * _atr1
    lb3_5 = to5(lo_band3); ub3_5 = to5(up_band3)
    lb6_5 = to5(lo_band6); ub6_5 = to5(up_band6)
    if mode in ("F2", "F3"):
        kk = 2 if mode == "F2" else 3
        buy5 = to5(conf_buy[kk], False)
        sell5 = to5(conf_sell[kk], False)
    else:
        buy5 = to5(sig["buy_sig"].to_numpy(), False)
        sell5 = to5(sig["sell_sig"].to_numpy(), False)
    adx5 = to5(adx1)
    bull5 = to5(bull_state, False)
    bear5 = to5(bear_state, False)
    sig_id5 = to5(np.arange(len(sig), dtype=float), -1)
    n = len(o5)
    trades: list[tuple[pd.Timestamp, float]] = []

    thr = 22.0 if mode in ("C2b",) else 20.0
    pos = 0
    entry = sl = 0.0
    tps: list[float] = []
    fracs: list[float] = []
    filled: list[bool] = []
    rem = 1.0
    scale = 1.0
    last_sig_id = -1.0
    gate_blocked_dir = 0  # C2: skip 했던 방향 (ADX 회복 시 상태 재진입용)

    def close_part(ts, px, frac, d):
        raw = (px - entry) / entry * d
        net, _f = apply_costs(raw, 0.9, lev)
        trades.append((ts, net * frac * scale * 100))

    def open_pos(d, px, low_adx):
        nonlocal pos, entry, sl, tps, filled, rem, scale, fracs
        pos = d
        entry = px
        sl = entry * (1 - SL_PCT * d)
        scale = 1.0
        lv_grid = TP_LV
        fr = [TP_FRAC] * 4
        if low_adx:
            if mode == "C1a":
                scale = 0.5
            elif mode == "C1b":
                scale = 0.25
            elif mode == "C1c":
                pass  # 계단 — open 호출부에서 scale 지정
            elif mode == "C3a":
                lv_grid = (0.005, 0.01, 0.015, 0.02)
                fr = [0.5, 0.5, 0.0, 0.0]  # TP2 전량
        tps = [entry * (1 + lv * d) for lv in lv_grid]
        fracs = fr
        filled = [False] * 4
        rem = 1.0

    for i in range(n):
        new_sig = 0
        if sig_id5[i] != last_sig_id and sig_id5[i] >= 0:
            if bool(buy5[i]):
                new_sig = 1
            elif bool(sell5[i]):
                new_sig = -1
            last_sig_id = sig_id5[i]
        ts = idx5[i]
        a = adx5[i]
        low_adx = (not np.isnan(a)) and a < thr
        if pos == 0:
            d = new_sig
            # C2: 이전에 skip 한 방향이 있고 ADX 회복 + 정렬 유지면 상태 재진입.
            if mode.startswith("C2") and d == 0 and gate_blocked_dir != 0 and not low_adx:
                if (gate_blocked_dir == 1 and bull5[i]) or (gate_blocked_dir == -1 and bear5[i]):
                    d = gate_blocked_dir
            if d != 0:
                if mode.startswith("C2") and low_adx:
                    gate_blocked_dir = d  # skip + 기억
                else:
                    if mode == "C1c":
                        open_pos(d, o5[i], low_adx)
                        scale = 0.25 if (not np.isnan(a) and a < 18) else (
                            0.5 if low_adx else 1.0)
                    else:
                        open_pos(d, o5[i], low_adx)
                    gate_blocked_dir = 0
            continue
        d = pos
        if new_sig == -d:
            close_part(ts, o5[i], rem, d)
            pos = 0
            if mode.startswith("C2") and low_adx:
                gate_blocked_dir = -d
            else:
                if mode == "C1c":
                    open_pos(-d, o5[i], low_adx)
                    scale = 0.25 if (not np.isnan(a) and a < 18) else (
                        0.5 if low_adx else 1.0)
                else:
                    open_pos(-d, o5[i], low_adx)
                gate_blocked_dir = 0
            continue
        if mode in ("T3", "T6"):
            # 트레일 청산 — 방향별 밴드 래칫: 롱은 하단밴드 최고치(상향만), 숏은
            # 상단밴드 최저치(하향만). 초기 SL(2%)이 바닥. TP 없음.
            band = (lb3_5[i] if mode == "T3" else lb6_5[i]) if d == 1 else                    (ub3_5[i] if mode == "T3" else ub6_5[i])
            if not np.isnan(band):
                sl = max(sl, band) if d == 1 else min(sl, band)
            if (d == 1 and lo5[i] <= sl) or (d == -1 and h5[i] >= sl):
                close_part(ts, sl, rem, d)
                pos = 0
            continue
        if (d == 1 and lo5[i] <= sl) or (d == -1 and h5[i] >= sl):
            close_part(ts, sl, rem, d)
            pos = 0
            continue
        for k2 in range(4):
            if filled[k2] or rem <= 0 or fracs[k2] <= 0:
                continue
            hit = (d == 1 and h5[i] >= tps[k2]) or (d == -1 and lo5[i] <= tps[k2])
            if hit:
                close_part(ts, tps[k2], min(fracs[k2], rem), d)
                filled[k2] = True
                rem = max(0.0, rem - fracs[k2])
                if k2 == 1:
                    sl = tps[0]
                elif k2 == 2:
                    sl = tps[1]
        if rem <= 1e-9:
            pos = 0
    return trades


def table(rows: dict[str, list]) -> None:
    for name, tr in rows.items():
        if not tr:
            print(f"{name:<6} 거래 0", flush=True)
            continue
        tr.sort()
        nets = [p for _, p in tr]
        net = sum(nets)
        w = sum(1 for p in nets if p > 0)
        half = len(tr) // 2
        h1 = sum(p for _, p in tr[:half]); h2 = sum(p for _, p in tr[half:])
        eq = pk = mdd = 0.0
        for _, p in tr:
            eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
        recent = sum(p for t, p in tr if t >= LIVE_CMP)
        ys: dict[int, float] = {}
        for t, p in tr:
            ys[t.year] = ys.get(t.year, 0.0) + p
        yearly = " ".join(f"{y}:{v:+.0f}" for y, v in sorted(ys.items()))
        print(f"{name:<6} n={len(tr):5d} net={net:+8.1f}% H1={h1:+7.1f} H2={h2:+7.1f} "
              f"MDD={mdd:6.1f} 최근45d={recent:+6.1f} [{yearly}]", flush=True)


def main() -> int:
    modes = ["base", "F2", "F3"]
    agg: dict[str, list] = {m: [] for m in modes}
    for sym in dst.PAIRS:
        for m in modes:
            agg[m] += run_pair(sym, m if m != "base" else "none")
        print(f"{sym} 완료", flush=True)
    print("\n===== 종합 (7페어 합산, 시드% — 레버 반영) =====", flush=True)
    table(agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
