"""볼린저 돌파(브레이크아웃) 백테 — 횡보 후 변동성 확장 추세 추종.

파트너(6/27): 평균회귀가 엣지 없으니([[bb-meanrev-research-2026-06-27]]) 그 역(逆),
즉 밴드 이탈을 '추세 신호'로 보는 돌파가 엣지인지 검증. BB squeeze(밴드폭 수축=
횡보)에서 밴드 돌파 시 추세 추종 진입 + ATR 트레일 청산(Cursus 와 같은 RR 비대칭).

전략:
    - squeeze: 밴드폭 (upper-lower)/mid 가 최근 N봉 중 하위 percentile(변동성 수축=횡보).
    - 진입: squeeze 상태에서 종가가 상단밴드 위 돌파 → 롱 / 하단 아래 → 숏(양방향).
    - 청산: ATR 트레일 스탑(chandelier) — 추세 방향으로만 끌어 손실 타이트/수익 추세끝까지.
    - 고정 TP 없음, 물타기 없음, SL 무조건. RR 비대칭(Cursus 원칙 공유).

검증: 7페어 5년, 시드1000, 20x, size0.9, aurora.backtest.cost. squeeze ON/OFF + 트레일폭
그리드. lookahead 방지: 신호 1봉 지연+다음봉 시가 진입, 직전봉 확정 stop 으로 장중 청산.

실행: cwd=Aurora-ICT-research, PYTHONPATH=../Aurora-ICT/src, argv[1]=TF분(기본 60).
담당: 지영민.
"""
from __future__ import annotations

import sys

import bb_meanrev_bt as bb
import numpy as np
import pandas as pd

from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct

SQ_LOOKBACK = 100      # 밴드폭 percentile 산출 롤링 윈도우
SQ_PCT = 0.3           # 하위 30% 이하 밴드폭 = squeeze(횡보) 판정


def _load_tf(sym: str, tf_min: int) -> pd.DataFrame:
    """1m parquet → tf_min 분봉 OHLCV 리샘플."""
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    rule = f"{tf_min}min"
    o = df["open"].resample(rule).first()
    h = df["high"].resample(rule).max()
    lo = df["low"].resample(rule).min()
    c = df["close"].resample(rule).last()
    v = df["volume"].resample(rule).sum()
    return pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "volume": v},
    ).dropna()


def _run_breakout(
    df: pd.DataFrame,
    *,
    trail_mult: float = 4.0,
    use_squeeze: bool = True,
    fee_pct: float = 0.0004,
    funding_per_bar: float = 0.0001 / 8,
) -> list[tuple[float, int]]:
    """BB squeeze 돌파 + ATR 트레일 시뮬 — net_pnl(비율)+연도 리스트.

    Args:
        df: OHLCV.
        trail_mult: ATR 트레일 폭(chandelier). 좁으면 휩쏘, 넓으면 추세 끝까지.
        use_squeeze: True 면 squeeze(밴드폭 하위 percentile) 상태에서만 돌파 진입.
        fee_pct: taker 수수료(0 이면 maker 가정).
        funding_per_bar: 봉당 펀딩 비율.
    """
    mid, up, lo = bb._bbands(df, bb.BB_PERIOD, bb.BB_MULT)
    atr = bb._atr(df, bb.ATR_PERIOD)
    bw = (up - lo) / mid
    sq_thr = bw.rolling(SQ_LOOKBACK).quantile(SQ_PCT)
    c = df["close"].values
    h = df["high"].values
    low = df["low"].values
    o = df["open"].values
    upv = up.values
    lov = lo.values
    atrv = atr.values
    bwv = bw.values
    sqv = sq_thr.values
    years = df.index.year.values

    trades: list[tuple[float, int]] = []
    side: str | None = None
    entry = stop = 0.0
    entry_i = 0
    for i in range(1, len(c)):
        a_now = atrv[i]
        if np.isnan(a_now):
            continue
        if side is not None:
            # #LOOKAHEAD 방지: 직전 봉까지 확정된 trail stop 으로 현재 봉 장중 청산
            # 먼저 체크, 청산 안 되면 현재 봉 종가로 트레일 갱신(다음 봉 반영).
            if side == "long":
                hit = low[i] <= stop
            else:
                hit = h[i] >= stop
            if hit:
                slp = slip_pct(h[i], low[i], c[i])
                exit_px = apply_slippage(stop, side, "exit", slp)
                raw = (exit_px - entry) / entry
                if side == "short":
                    raw = -raw
                net, _ = apply_costs(raw, bb.SIZE_PCT, bb.LEVERAGE, fee_pct=fee_pct)
                net -= (i - entry_i) * funding_per_bar * bb.SIZE_PCT * bb.LEVERAGE
                trades.append((net, int(years[i])))
                side = None
            elif side == "long":
                stop = max(stop, c[i] - trail_mult * a_now)
            else:
                stop = min(stop, c[i] + trail_mult * a_now)
        if side is None:
            j = i - 1
            if np.isnan(upv[j]) or np.isnan(atrv[j]):
                continue
            squeezed = (not use_squeeze) or (
                not np.isnan(sqv[j]) and bwv[j] <= sqv[j]
            )
            if not squeezed:
                continue
            long_sig = c[j] > upv[j]
            short_sig = c[j] < lov[j]
            if long_sig:
                slp = slip_pct(h[i], low[i], c[i])
                entry = apply_slippage(o[i], "long", "entry", slp)
                side = "long"
                stop = entry - trail_mult * atrv[j]
                entry_i = i
            elif short_sig:
                slp = slip_pct(h[i], low[i], c[i])
                entry = apply_slippage(o[i], "short", "entry", slp)
                side = "short"
                stop = entry + trail_mult * atrv[j]
                entry_i = i
    return trades


def main() -> int:
    tf_min = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    funding_per_bar = (0.0001 / 8) * (tf_min / 60.0)
    data = {}
    for sym in bb.PAIRS:
        try:
            d = _load_tf(sym, tf_min)
            if len(d) >= 500:
                data[sym] = d
        except Exception as e:  # noqa: BLE001
            print(f"(로드 실패 {sym}: {e})")

    lines = [
        f"===== 볼린저 돌파(브레이크아웃) 백테 {tf_min}m (7페어, 시드1000, 20x) =====",
        "squeeze(밴드폭 하위30%)에서 밴드 돌파 추세추종 + ATR 트레일. RR 비대칭.",
        "",
        f"{'squeeze':>8} {'트레일':>6} {'수수료':>6} {'USDT':>9} {'승률':>6} {'RR':>5} "
        f"{'연속손절':>8} {'거래':>7}",
    ]

    def _agg(trail_mult: float, use_squeeze: bool, fee_pct: float) -> dict:
        agg = {"net": 0.0, "wr": 0.0, "rr": 0.0, "streak": 0.0, "n": 0}
        yearly: dict[int, float] = {}
        nz = 0
        for d in data.values():
            tr = _run_breakout(d, trail_mult=trail_mult, use_squeeze=use_squeeze,
                               fee_pct=fee_pct, funding_per_bar=funding_per_bar)
            s = bb._stats([t[0] for t in tr])
            for k in ("net", "wr", "rr", "streak"):
                agg[k] += s[k]
            agg["n"] += int(s["n"])
            nz += 1
            for net, yr in tr:
                yearly[yr] = yearly.get(yr, 0.0) + net
        agg["nz"] = max(nz, 1)
        agg["yearly"] = yearly
        return agg

    best = None
    for use_squeeze in (True, False):
        for trail_mult in (3.0, 4.0, 6.0):
            a = _agg(trail_mult, use_squeeze, 0.0004)
            nz = a["nz"]
            tag = "ON" if use_squeeze else "OFF"
            lines.append(
                f"{tag:>8} x{trail_mult:<5.1f} {'taker':>6} "
                f"{a['net'] * bb.SEED:+9.0f} {a['wr'] / nz:5.0f}% {a['rr'] / nz:5.2f} "
                f"{a['streak'] / nz:7.1f}회 {a['n']:7d}")
            if best is None or a["net"] > best[0]:
                best = (a["net"], trail_mult, use_squeeze, a)

    # 최선 조합 maker(수수료0) + 연도별 + walk-forward.
    if best is not None:
        _, b_tm, b_sq, _ = best
        a0 = _agg(b_tm, b_sq, 0.0)
        nz = a0["nz"]
        lines.append("")
        lines.append(
            f"--- 최선(squeeze {'ON' if b_sq else 'OFF'}, x{b_tm:.1f}) maker(수수료0) ---")
        lines.append(
            f"  net {a0['net'] * bb.SEED:+9.0f} USDT  승률 {a0['wr'] / nz:4.0f}%  "
            f"RR {a0['rr'] / nz:4.2f}  거래 {a0['n']}")
        lines.append("")
        lines.append("--- 최선 조합 연도별 net (taker) ---")
        _, _, _, b_a = best
        for yr in sorted(b_a["yearly"]):
            lines.append(f"  {yr}: {b_a['yearly'][yr] * bb.SEED:+9.0f} USDT")
        lines.append("")
        lines.append("--- Walk-forward (앞60% in / 뒤40% out, taker) ---")
        lines.append(f"  {'트레일':<7}{'in-net':>11}{'out-net':>11}")
        for tm in (3.0, 4.0, 6.0):
            innet = outnet = 0.0
            for d in data.values():
                k = int(len(d) * 0.6)
                innet += bb._stats([t[0] for t in _run_breakout(
                    d.iloc[:k], trail_mult=tm, use_squeeze=b_sq,
                    funding_per_bar=funding_per_bar)])["net"]
                outnet += bb._stats([t[0] for t in _run_breakout(
                    d.iloc[k:], trail_mult=tm, use_squeeze=b_sq,
                    funding_per_bar=funding_per_bar)])["net"]
            lines.append(f"  x{tm:<6.1f}{innet * bb.SEED:>+11.0f}{outnet * bb.SEED:>+11.0f}")

    txt = "\n".join(lines)
    out = f"bb_breakout_{tf_min}m_result.txt"
    with open(out, "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + f"\n→ {out}\nDONE")
    except UnicodeEncodeError:
        print(f"(결과는 {out})\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
