"""Cursus REVERSE(반대신호 청산) 영향 검증 — FST #1 자율연구.

FST 2026-07-01: 라이브 Cursus RR 0.24(추세봇 역설). flip_close(반대신호 청산)
6건이 -216.7 로 손실 집중 → 트레일에 걸리기 전 반대신호로 크게 깨짐. 백테에서
REVERSE 청산의 기여도를 분리 검증한다.

비교(1h, 7페어, trail_mult=6=라이브 설정):
    - 현행(hit+rev): 트레일 스탑 OR 반대신호로 청산 + 역진입 (dst._run).
    - no-rev(hit만): 반대신호 무시, 트레일 스탑까지 보유. 추세 반전을 트레일이
      뒤늦게 잡으나 반대신호 조기청산의 큰 손실은 회피 가능.
    - rev-loss만: 반대신호가 떴을 때 현재 미실현이 손실이면 청산(손실 확대 차단),
      이익이면 무시(추세 지속 베팅). REVERSE 의 선택적 적용.

진입/트레일/비용/lookahead 방지는 dst_trend_bt 와 100% 동일(_signals/_supertrend).
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
import dst_trend_bt as dst  # noqa: E402
import numpy as np  # noqa: E402


def _run_variant(df, trail_mult=6.0, mode="norev"):
    """REVERSE 처리만 바꾼 트레일 시뮬. mode: norev / revloss.

    mode='norev'  : 반대신호 무시 — 트레일 스탑 히트만 청산.
    mode='revloss': 반대신호 시 미실현 손실이면 청산, 이익이면 무시.
    """
    sig = dst._signals(df)
    trail = dst._supertrend(sig, trail_mult, dst.ATR_PERIOD)
    h = sig["high"].values
    low = sig["low"].values
    c = sig["close"].values
    o = sig["open"].values
    stop_arr = trail.values
    years = sig.index.year.values
    buy = np.concatenate([[False], sig["buy_sig"].values[:-1]])
    sell = np.concatenate([[False], sig["sell_sig"].values[:-1]])
    trades: list[tuple[float, int]] = []
    side = None
    entry = stop = 0.0
    entry_i = 0
    for i in range(1, len(c)):
        s_now = stop_arr[i]
        if np.isnan(s_now):
            continue
        if side is not None:
            hit = low[i] <= stop if side == "long" else h[i] >= stop
            rev = bool(sell[i]) if side == "long" else bool(buy[i])
            do_rev = False
            if rev and not hit:
                # 반대신호 시 미실현 손익(다음봉 시가 기준) 부호로 선택적 청산.
                unreal = (o[i] - entry) / entry
                if side == "short":
                    unreal = -unreal
                if mode == "norev":
                    do_rev = False
                elif mode == "revloss":
                    do_rev = unreal < 0  # 손실이면 끊고, 이익이면 추세 지속
            if hit or do_rev:
                exit_raw = stop if hit else o[i]
                slp = dst.slip_pct(h[i], low[i], c[i])
                exit_px = dst.apply_slippage(exit_raw, side, "exit", slp)
                raw = (exit_px - entry) / entry
                if side == "short":
                    raw = -raw
                net, _ = dst.apply_costs(raw, dst.SIZE_PCT, dst.LEVERAGE)
                net -= (i - entry_i) * dst.FUNDING_PER_HOUR * dst.SIZE_PCT * dst.LEVERAGE
                trades.append((net, int(years[i])))
                side = None
            elif side == "long":
                stop = max(stop, s_now)
            else:
                stop = min(stop, s_now)
        if side is None:
            if buy[i]:
                slp = dst.slip_pct(h[i], low[i], c[i])
                entry = dst.apply_slippage(o[i], "long", "entry", slp)
                side = "long"; stop = s_now; entry_i = i
            elif sell[i]:
                slp = dst.slip_pct(h[i], low[i], c[i])
                entry = dst.apply_slippage(o[i], "short", "entry", slp)
                side = "short"; stop = s_now; entry_i = i
    return trades


def _agg(data, fn):
    agg = {"net": 0.0, "wr": 0.0, "rr": 0.0, "n": 0}
    nz = 0
    for d in data.values():
        tr = fn(d)
        s = dst._stats([t[0] for t in tr])
        for k in ("net", "wr", "rr"):
            agg[k] += s[k]
        agg["n"] += int(s["n"]); nz += 1
    agg["nz"] = max(nz, 1)
    return agg


def main() -> int:
    data = {}
    for sym in dst.PAIRS:
        try:
            d = dst._load_1h(sym)
            if len(d) >= 200:
                data[sym] = d
        except Exception as e:  # noqa: BLE001
            print(f"(로드 실패 {sym}: {e})")

    lines = [
        "===== Cursus REVERSE 영향 검증 (1h, 7페어, trail x6, 시드1000, 20x) =====",
        "진입·트레일·비용 dst_trend_bt 동일. REVERSE 청산 처리만 변형.",
        "",
        f"{'방식':<26}{'USDT':>9}{'승률':>7}{'RR':>6}{'거래':>8}",
    ]
    variants = [
        ("현행 (hit + REVERSE)", lambda d: dst._run(d, 6.0)),
        ("no-rev (트레일만)", lambda d: _run_variant(d, 6.0, "norev")),
        ("rev-loss만 (손실시만 청산)", lambda d: _run_variant(d, 6.0, "revloss")),
    ]
    for label, fn in variants:
        a = _agg(data, fn); nz = a["nz"]
        lines.append(
            f"{label:<22}{a['net'] * dst.SEED:>+9.0f}"
            f"{a['wr'] / nz:>6.0f}%{a['rr'] / nz:>6.2f}{a['n']:>8d}")

    txt = "\n".join(lines)
    with open("reverse_impact_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE")
    except UnicodeEncodeError:
        print("(결과는 reverse_impact_result.txt)\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
