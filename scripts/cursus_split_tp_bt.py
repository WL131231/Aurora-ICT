"""Cursus 4분할 TP 되돌리기 검증 — 분할익절+트레일 vs 트레일 단독.

파트너(6/27): Cursus 원본(외부개발자 매매기법.py)의 4단계 분할 TP(1~4%)를 우리가
백테 RR<1 적자라 제거하고 트레일 단독으로 갔는데, 되돌렸을 때(분할 복원) 실제 성과를
재검증. dst_trend_bt 모듈(동일 데이터·지표·비용·lookahead 방지) 재사용.

비교:
    - 트레일 단독(현행): ST 트레일(x6) 청산만. RR 비대칭(수익 추세 끝까지).
    - 4분할 TP + 트레일: 진입 후 +1/2/3/4%(또는 ATR×) 도달 시 25%씩 부분 익절,
      잔량은 트레일 SL. 수익을 일찍 확보하나 추세 끝까지 못 먹어 RR 저하 가능.

진입 신호·트레일·lookahead 방지는 dst_trend_bt 와 100% 동일(_signals/_supertrend).
부분 청산은 각 청산분 비중(size×frac)으로 수수료까지 비례 반영.

실행: cwd=Aurora-ICT-research, PYTHONPATH=../Aurora-ICT/src.
담당: 지영민.
"""
from __future__ import annotations

import sys

import dst_trend_bt as dst
import numpy as np

from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct


def _run_split(
    df,
    *,
    trail_mult: float = 6.0,
    tp_levels: tuple[float, ...] = (0.01, 0.02, 0.03, 0.04),
    tp_frac: float = 0.25,
    use_atr_tp: bool = False,
    atr_mults: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0),
) -> list[tuple[float, int]]:
    """4분할 TP + 트레일 시뮬 — 부분 청산 net(비율, 시드 대비) 리스트.

    Args:
        df: 1h OHLCV.
        trail_mult: 잔량 트레일 ST 폭.
        tp_levels: 고정% TP 레벨(롱 +/숏 -). use_atr_tp=False 일 때.
        tp_frac: 각 TP 청산 비중(0.25 = 4분할).
        use_atr_tp: True 면 tp_levels 대신 entry ± ATR×atr_mults.
        atr_mults: ATR 기반 TP 배수.
    """
    sig = dst._signals(df)
    trail = dst._supertrend(sig, trail_mult, dst.ATR_PERIOD)
    atr = dst._atr(sig, dst.ATR_PERIOD)
    h = sig["high"].values
    low = sig["low"].values
    c = sig["close"].values
    o = sig["open"].values
    stop_arr = trail.values
    atrv = atr.values
    years = sig.index.year.values
    buy = np.concatenate([[False], sig["buy_sig"].values[:-1]])
    sell = np.concatenate([[False], sig["sell_sig"].values[:-1]])

    n_tp = len(atr_mults) if use_atr_tp else len(tp_levels)
    trades: list[tuple[float, int]] = []
    side: str | None = None
    entry = stop = 0.0
    entry_i = 0
    remaining = 0.0
    tp_px: list[float] = []
    tp_done: list[bool] = []

    def _book(exit_raw: float, frac: float, i: int) -> None:
        slp = slip_pct(h[i], low[i], c[i])
        exit_px = apply_slippage(exit_raw, side, "exit", slp)  # type: ignore[arg-type]
        raw = (exit_px - entry) / entry
        if side == "short":
            raw = -raw
        net, _ = apply_costs(raw, dst.SIZE_PCT * frac, dst.LEVERAGE)
        net -= (i - entry_i) * dst.FUNDING_PER_HOUR * dst.SIZE_PCT * frac * dst.LEVERAGE
        trades.append((net, int(years[i])))

    for i in range(1, len(c)):
        s_now = stop_arr[i]
        if np.isnan(s_now):
            continue
        if side is not None:
            if side == "long":
                hit_sl = low[i] <= stop
                rev = bool(sell[i])
            else:
                hit_sl = h[i] >= stop
                rev = bool(buy[i])
            if hit_sl or rev:
                # 잔량 전부 청산(스탑가 or 반대신호 다음봉 시가).
                exit_raw = stop if hit_sl else o[i]
                _book(exit_raw, remaining, i)
                side = None
                continue
            # 미달성 TP 부분 청산.
            for k in range(n_tp):
                if tp_done[k]:
                    continue
                reached = (h[i] >= tp_px[k]) if side == "long" else (low[i] <= tp_px[k])
                if reached:
                    _book(tp_px[k], tp_frac, i)
                    tp_done[k] = True
                    remaining = max(0.0, remaining - tp_frac)
            if remaining <= 1e-9:
                side = None
                continue
            # 트레일 갱신(다음 봉용).
            stop = max(stop, s_now) if side == "long" else min(stop, s_now)
        if side is None:
            new_side = "long" if buy[i] else ("short" if sell[i] else None)
            if new_side is not None:
                slp = slip_pct(h[i], low[i], c[i])
                entry = apply_slippage(o[i], new_side, "entry", slp)
                side = new_side
                stop = s_now
                entry_i = i
                remaining = 1.0
                tp_done = [False] * n_tp
                a = atrv[i] if not np.isnan(atrv[i]) else 0.0
                if use_atr_tp:
                    tp_px = [
                        entry + m * a if new_side == "long" else entry - m * a
                        for m in atr_mults
                    ]
                else:
                    tp_px = [
                        entry * (1 + p) if new_side == "long" else entry * (1 - p)
                        for p in tp_levels
                    ]
    return trades


def main() -> int:
    data = {}
    for sym in dst.PAIRS:
        try:
            d = dst._load_1h(sym)
            if len(d) >= 200:
                data[sym] = d
        except Exception as e:  # noqa: BLE001
            print(f"(로드 실패 {sym}: {e})")

    def _agg_trail() -> dict:
        agg = {"net": 0.0, "wr": 0.0, "rr": 0.0, "n": 0}
        nz = 0
        for d in data.values():
            tr = dst._run(d, 6.0)
            s = dst._stats([t[0] for t in tr])
            for k in ("net", "wr", "rr"):
                agg[k] += s[k]
            agg["n"] += int(s["n"])
            nz += 1
        agg["nz"] = max(nz, 1)
        return agg

    def _agg_split(**kw) -> dict:
        agg = {"net": 0.0, "wr": 0.0, "rr": 0.0, "n": 0}
        yearly: dict[int, float] = {}
        nz = 0
        for d in data.values():
            tr = _run_split(d, **kw)
            s = dst._stats([t[0] for t in tr])
            for k in ("net", "wr", "rr"):
                agg[k] += s[k]
            agg["n"] += int(s["n"])
            nz += 1
            for net, yr in tr:
                yearly[yr] = yearly.get(yr, 0.0) + net
        agg["nz"] = max(nz, 1)
        agg["yearly"] = yearly
        return agg

    lines = [
        "===== Cursus 4분할 TP 되돌리기 검증 (1h, 7페어, 시드1000, 20x) =====",
        "진입·트레일·비용은 dst_trend_bt 동일. 분할 TP 복원 시 성과 비교.",
        "",
        f"{'방식':<26}{'USDT':>9}{'승률':>7}{'RR':>6}{'거래':>8}",
    ]
    a = _agg_trail()
    nz = a["nz"]
    lines.append(
        f"{'트레일 단독 x6 (현행)':<22}{a['net'] * dst.SEED:>+9.0f}"
        f"{a['wr'] / nz:>6.0f}%{a['rr'] / nz:>6.2f}{a['n']:>8d}")

    # 4분할 % TP (원본 1~4%) + 트레일.
    a = _agg_split(trail_mult=6.0, tp_levels=(0.01, 0.02, 0.03, 0.04))
    nz = a["nz"]
    lines.append(
        f"{'4분할 %TP(1~4%)+트레일x6':<22}{a['net'] * dst.SEED:>+9.0f}"
        f"{a['wr'] / nz:>6.0f}%{a['rr'] / nz:>6.2f}{a['n']:>8d}")
    split_yearly = a["yearly"]

    # 4분할 ATR TP(1~4×) + 트레일 — 페어 변동성 정합.
    a = _agg_split(trail_mult=6.0, use_atr_tp=True, atr_mults=(1.0, 2.0, 3.0, 4.0))
    nz = a["nz"]
    lines.append(
        f"{'4분할 ATR_TP(1~4x)+트레일x6':<20}{a['net'] * dst.SEED:>+9.0f}"
        f"{a['wr'] / nz:>6.0f}%{a['rr'] / nz:>6.2f}{a['n']:>8d}")

    # 넓은 ATR TP(2~8×) — 추세 끝까지 일부 보존.
    a = _agg_split(trail_mult=6.0, use_atr_tp=True, atr_mults=(2.0, 4.0, 6.0, 8.0))
    nz = a["nz"]
    lines.append(
        f"{'4분할 ATR_TP(2~8x)+트레일x6':<20}{a['net'] * dst.SEED:>+9.0f}"
        f"{a['wr'] / nz:>6.0f}%{a['rr'] / nz:>6.2f}{a['n']:>8d}")

    lines.append("")
    lines.append("--- 4분할 %TP(1~4%) 연도별 net ---")
    for yr in sorted(split_yearly):
        lines.append(f"  {yr}: {split_yearly[yr] * dst.SEED:+9.0f} USDT")

    txt = "\n".join(lines)
    with open("cursus_split_tp_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE")
    except UnicodeEncodeError:
        print("(결과는 cursus_split_tp_result.txt)\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
