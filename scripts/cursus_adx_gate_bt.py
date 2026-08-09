"""#AUTONOMOUS 2026-07-27: Cursus ADX 게이트 백테 — 횡보 출혈 해결 후보 (파트너 최우선).

라이브 해부 결과: Cursus 청산 154건 중 진입 시 ADX(1h)<20 구간 avg **-1.96**/건
(-41.2), ADX>25 avg +1.93 (+206) — 추세봇의 횡보 취약이 라이브 손실의 실체.
→ "ADX<임계면 Cursus 신규진입 skip" 게이트를 5년 7페어 원본엔진 시뮬로 정식 검증.

엔진(라이브 정합, bot_trend_instance 스펙): Dual ST(×2/×3) 정렬 돌파 진입(다음봉
시가) · 고정 SL 2% · 4분할 TP +1/2/3/4% ×25% · TP래더(TP2→SL=TP1, TP3→SL=TP2,
TP4 전량) · REVERSE(반대 신호 시 잔량 청산 후 반대 진입). 신호·지표는 dst_trend_bt
재사용(lookahead 방지 동일), 비용 aurora.backtest.cost.

게이트: ADX14(1h, 직전 완결봉) < thr → 신규 진입만 skip (보유 관리·REVERSE 청산은
유지, 단 REVERSE 재진입은 게이트 적용). thr ∈ {없음, 18, 20, 22, 25}.
배터리: 페어별·연도별·H1/H2·MDD — 전 항목 표로.
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


def run_pair(sym: str, adx_thr: float | None) -> list[tuple[pd.Timestamp, float]]:
    """1페어 시뮬 — 신호 1h · 체결 5m 경로(라이브 폴링 정합). (청산시각, 시드대비 net%)."""
    lev = 10.0 if sym.startswith("BTC") else 7.0  # 라이브 Cursus 레버 정합
    from bt_par import _load_full, _resample
    df1h = dst._load_1h(sym)
    sig = dst._signals(df1h)
    h1, l1, c1 = (sig[k].to_numpy() for k in ("high", "low", "close"))
    adx1 = adx14(h1, l1, c1)
    # 1h 신호/ADX — "그 봉 종가 확정 후" 유효 → 다음 1h 구간의 5m 봉들에 매핑.
    sig_buy = pd.Series(sig["buy_sig"].to_numpy(), index=sig.index)
    sig_sell = pd.Series(sig["sell_sig"].to_numpy(), index=sig.index)
    adx_s = pd.Series(adx1, index=sig.index)
    df5 = _resample(_load_full(sym))
    df5 = df5[(df5.index >= sig.index[0]) & (df5.index <= sig.index[-1] + pd.Timedelta(hours=1))]
    o5 = df5["open"].to_numpy(); h5 = df5["high"].to_numpy(); lo5 = df5["low"].to_numpy()
    idx5 = df5.index
    # 각 5m 봉에 "직전 완결 1h 봉" 신호/ADX 부여 (완결 = 봉시각+1h <= 현재).
    lag = sig.index + pd.Timedelta(hours=1)
    buy5 = pd.Series(sig_buy.to_numpy(), index=lag).reindex(idx5, method="ffill").fillna(False).to_numpy()
    sell5 = pd.Series(sig_sell.to_numpy(), index=lag).reindex(idx5, method="ffill").fillna(False).to_numpy()
    adx5 = pd.Series(adx_s.to_numpy(), index=lag).reindex(idx5, method="ffill").to_numpy()
    # 신호는 1h당 1회만 실행 — 신호 봉 전환 감지용.
    sig_id5 = pd.Series(range(len(sig)), index=lag).reindex(idx5, method="ffill").fillna(-1).to_numpy()
    n = len(o5)
    trades: list[tuple[pd.Timestamp, float]] = []

    def gate_ok(i: int) -> bool:
        return adx_thr is None or (not np.isnan(adx5[i]) and adx5[i] >= adx_thr)

    pos = 0
    entry = sl = 0.0
    tps: list[float] = []
    filled: list[bool] = []
    rem = 1.0
    last_sig_id = -1

    def close_part(ts, px, frac, d):
        raw = (px - entry) / entry * d
        net, _fee = apply_costs(raw, 0.9, lev)
        trades.append((ts, net * frac * 100))

    def open_pos(d, px):
        nonlocal pos, entry, sl, tps, filled, rem
        pos = d
        entry = px
        sl = entry * (1 - SL_PCT * d)
        tps = [entry * (1 + lv * d) for lv in TP_LV]
        filled = [False] * 4
        rem = 1.0

    for i in range(n):
        new_sig = 0
        if sig_id5[i] != last_sig_id and sig_id5[i] >= 0:
            k = int(sig_id5[i])
            if bool(sig_buy.iloc[k]):
                new_sig = 1
            elif bool(sig_sell.iloc[k]):
                new_sig = -1
            last_sig_id = sig_id5[i]
        ts = idx5[i]
        if pos == 0:
            if new_sig != 0 and gate_ok(i):
                open_pos(new_sig, o5[i])
            continue
        d = pos
        # REVERSE — 새 반대 신호가 이 5m 봉에서 유효해지면 시가로 잔량 청산 후 재진입.
        if new_sig == -d:
            close_part(ts, o5[i], rem, d)
            pos = 0
            if gate_ok(i):
                open_pos(-d, o5[i])
            continue
        # 5m 봉 내 SL/TP — SL 우선(보수)이나 5m 폭이라 왜곡 미미.
        if (d == 1 and lo5[i] <= sl) or (d == -1 and h5[i] >= sl):
            close_part(ts, sl, rem, d)
            pos = 0
            continue
        for k2 in range(4):
            if filled[k2] or rem <= 0:
                continue
            hit = (d == 1 and h5[i] >= tps[k2]) or (d == -1 and lo5[i] <= tps[k2])
            if hit:
                close_part(ts, tps[k2], TP_FRAC, d)
                filled[k2] = True
                rem = max(0.0, rem - TP_FRAC)
                if k2 == 1:
                    sl = tps[0]
                elif k2 == 2:
                    sl = tps[1]
        if all(filled) or rem <= 1e-9:
            pos = 0
    return trades


def table(rows_by_variant: dict[str, list[tuple[pd.Timestamp, float]]]) -> None:
    for name, tr in rows_by_variant.items():
        if not tr:
            print(f"{name:<8} 거래 0", flush=True)
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
        ys: dict[int, float] = {}
        for t, p in tr:
            ys[t.year] = ys.get(t.year, 0.0) + p
        yearly = " ".join(f"{y}:{v:+.0f}" for y, v in sorted(ys.items()))
        print(f"{name:<8} n={len(tr):5d} net={net:+8.1f}% 승률={100 * w / len(tr):3.0f}% "
              f"H1={h1:+7.1f} H2={h2:+7.1f} MDD={mdd:6.1f} net/MDD={net / max(mdd, 1e-9):5.2f} "
              f"[{yearly}]", flush=True)


def main() -> int:
    variants: dict[str, float | None] = {"base": None, "ADX>=18": 18, "ADX>=20": 20,
                                          "ADX>=22": 22, "ADX>=25": 25}
    agg: dict[str, list] = {k: [] for k in variants}
    for sym in dst.PAIRS:
        print(f"\n===== {sym} =====", flush=True)
        per = {}
        for name, thr in variants.items():
            tr = run_pair(sym, thr)
            per[name] = tr
            agg[name] += tr
        table(per)
    print("\n===== 종합 (7페어 합산) =====", flush=True)
    table(agg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
