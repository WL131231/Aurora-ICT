"""DualST 추세형 백테 — 트레일링 중심(ST 라인 스탑) + 양방향, 1h, 우리원칙 버전.

파트너 결정(2026-06-25): Origo(ICT 단타) 외 추세형 봇 1개 추가. 외부개발자 매매기법.py
의 Dual SuperTrend 신호를 채용하되 4단계 분할TP(백테 적자·RR<1)는 제거 → 트레일링
중심으로 RR 비대칭 복원이 핵심.

전략:
    - 진입: close 가 ST1(ATR14×2.0) & ST2(ATR14×3.0) 둘 다 위 = 롱 / 둘 다 아래 = 숏.
      정렬이 새로 발생한 봉(돌파)에서 진입.
    - 청산: ST 라인(트레일 스탑)을 추세 방향으로만 끌어올리다 종가/저고가가 깨면 청산.
      반대 신호(REVERSE)면 청산 후 같은 봉 역진입.
    - 별도 고정 TP 없음 — ST 라인이 곧 트레일 스탑이라 손실=타이트, 수익=추세 끝까지(RR 비대칭).

검증: 7페어(고정7) 5년 1h, 시드 1000, 20x, size 0.9. 평가 = net USDT / 최대DD / 승률 /
RR(avgWin/avgLoss) / 연속손절 / 거래수. 수수료·슬리피지는 aurora.backtest.cost 재사용
(taker 0.04%×2 + 변동성 슬리피지) → Origo 백테와 동일 기준.

스탑 라인 변형: st1(타이트 ×2.0) vs st2(넓은 ×3.0) 비교.
실행: cwd=Aurora-ICT-research, PYTHONPATH=../Aurora-ICT/src.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
LEVERAGE = 20.0
SIZE_PCT = 0.9
ATR_PERIOD = 14
ST1_MULT = 2.0
ST2_MULT = 3.0
# Bybit 평균 펀딩 8h당 0.01% → 시간당. 추세형 장기보유 비용(보유 봉수×차감).
FUNDING_PER_HOUR = 0.0001 / 8


def _load_1h(sym: str) -> pd.DataFrame:
    """1m parquet → 1h OHLCV 리샘플 (data/{sym}_1m_full.parquet)."""
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    o = df["open"].resample("60min").first()
    h = df["high"].resample("60min").max()
    lo = df["low"].resample("60min").min()
    c = df["close"].resample("60min").last()
    v = df["volume"].resample("60min").sum()
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c, "volume": v}).dropna()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR (매매기법.py 차용)."""
    h, low, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - low), (h - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _supertrend(df: pd.DataFrame, mult: float, period: int) -> pd.Series:
    """SuperTrend 라인 (매매기법.py 차용) — dir -1=상승추세/1=하락추세."""
    hl2 = (df["high"] + df["low"]) / 2.0
    a = _atr(df, period)
    upper = (hl2 + mult * a).values
    lower = (hl2 - mult * a).values
    close = df["close"].values
    av = a.values
    n = len(df)
    up = upper.copy()
    lo = lower.copy()
    st = np.full(n, np.nan)
    dir_ = np.ones(n)
    for i in range(1, n):
        if np.isnan(av[i]):
            continue
        if not np.isnan(lo[i - 1]):
            lo[i] = lo[i] if (lo[i] > lo[i - 1] or close[i - 1] < lo[i - 1]) else lo[i - 1]
        if not np.isnan(up[i - 1]):
            up[i] = up[i] if (up[i] < up[i - 1] or close[i - 1] > up[i - 1]) else up[i - 1]
        if np.isnan(st[i - 1]) or st[i - 1] == up[i - 1]:
            dir_[i] = -1 if close[i] > up[i] else 1
        else:
            dir_[i] = 1 if close[i] < lo[i] else -1
        st[i] = lo[i] if dir_[i] == -1 else up[i]
    return pd.Series(st, index=df.index)


def _signals(df: pd.DataFrame) -> pd.DataFrame:
    """Dual SuperTrend 신호 — 둘 다 정렬 돌파 시 buy/sell_sig."""
    out = df.copy()
    out["st1"] = _supertrend(out, ST1_MULT, ATR_PERIOD)
    out["st2"] = _supertrend(out, ST2_MULT, ATR_PERIOD)
    src = out["close"]
    bull = (src > out["st1"]) & (src > out["st2"])
    bear = (src < out["st1"]) & (src < out["st2"])
    out["buy_sig"] = bull & ~bull.shift(1, fill_value=False)
    out["sell_sig"] = bear & ~bear.shift(1, fill_value=False)
    return out


def _run(df: pd.DataFrame, trail_mult: float = 3.0) -> list[tuple[float, int]]:
    """트레일링 중심 시뮬 — 진입(ST1×2 & ST2×3 정렬 돌파) 후 별도 트레일 ST(trail_mult)
    로 청산. 트레일 폭이 좁으면 휩쏘에 자주 잘려 RR↓, 넓으면 큰 추세 끝까지 먹어 RR↑.
    net_pnl(비율) 리스트 반환.
    """
    sig = _signals(df)
    trail = _supertrend(sig, trail_mult, ATR_PERIOD)
    h = sig["high"].values
    low = sig["low"].values
    c = sig["close"].values
    stop_arr = trail.values
    o = sig["open"].values
    years = sig.index.year.values
    # #LOOKAHEAD 보수화: buy/sell 신호는 봉 종가 확정 후에야 발생 → 1봉 지연시켜 다음
    # 봉 시가에 진입·역청산(실거래 정합, 같은 봉 종가 체결의 낙관 제거).
    buy = np.concatenate([[False], sig["buy_sig"].values[:-1]])
    sell = np.concatenate([[False], sig["sell_sig"].values[:-1]])
    trades: list[tuple[float, int]] = []
    side: str | None = None
    entry = 0.0
    stop = 0.0
    entry_i = 0
    for i in range(1, len(c)):
        s_now = stop_arr[i]
        if np.isnan(s_now):
            continue
        if side is not None:
            # #LOOKAHEAD 방지: 직전 봉까지 확정된 stop 으로 현재 봉 청산을 먼저 체크하고,
            # 청산 안 됐을 때만 현재 봉 ST 로 트레일 스탑 갱신(다음 봉부터 반영). ST[i] 는
            # i봉 종가 확정 후에야 알 수 있으므로 같은 봉 장중 청산 판정에 쓰면 미래정보 누수.
            if side == "long":
                hit = low[i] <= stop
                rev = bool(sell[i])
            else:
                hit = h[i] >= stop
                rev = bool(buy[i])
            # #PHANTOM-FIX2: 보유 중 트레일ST 플립(라인이 반대편으로 점프)이면
            # 라인가 청산은 체결 불가 — 이번 봉 시가로 정직 청산.
            flip = (side == "long" and s_now > c[i]) or (side == "short" and s_now < c[i])
            if hit or rev or flip:
                exit_raw = stop if hit else o[i]   # 스탑히트=스탑가, 반대신호/플립=시가
                slp = slip_pct(h[i], low[i], c[i])
                exit_px = apply_slippage(exit_raw, side, "exit", slp)
                raw = (exit_px - entry) / entry
                if side == "short":
                    raw = -raw
                net, _ = apply_costs(raw, SIZE_PCT, LEVERAGE)
                net -= (i - entry_i) * FUNDING_PER_HOUR * SIZE_PCT * LEVERAGE  # 펀딩비
                trades.append((net, int(years[i])))
                side = None
            elif side == "long":
                if s_now <= c[i]:
                    stop = max(stop, s_now)   # 올바른 쪽일 때만 래칫
            else:
                if s_now >= c[i]:
                    stop = min(stop, s_now)
        # 청산 직후 같은 봉의 반대 신호면 자동 역진입(REVERSE) — 아래 분기가 처리.
        if side is None:
            if buy[i]:
                slp = slip_pct(h[i], low[i], c[i])
                entry = apply_slippage(o[i], "long", "entry", slp)
                side = "long"
                # #PHANTOM-FIX: 트레일ST 가 아직 반대편(진입가 위)이면 체결 불가능한
                # 스탑 — 올바른 쪽(진입가 아래)으로 클램프(2% 바닥).
                stop = min(s_now, entry * 0.98) if not np.isnan(s_now) else entry * 0.98
                entry_i = i
            elif sell[i]:
                slp = slip_pct(h[i], low[i], c[i])
                entry = apply_slippage(o[i], "short", "entry", slp)
                side = "short"
                stop = max(s_now, entry * 1.02) if not np.isnan(s_now) else entry * 1.02
                entry_i = i
    return trades


def _stats(trades: list[float]) -> dict[str, float]:
    """net(비율 누적)/MDD/승률/RR/연속손절/거래수."""
    if not trades:
        return {"n": 0, "net": 0.0, "mdd": 0.0, "wr": 0.0, "rr": 0.0, "streak": 0.0}
    cum = peak = mdd = 0.0
    streak = maxstreak = nwin = 0
    gw = gl = 0.0
    for p in trades:
        cum += p
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
        if p > 0:
            nwin += 1
            gw += p
            streak = 0
        else:
            streak += 1
            maxstreak = max(maxstreak, streak)
            gl += abs(p)
    n = len(trades)
    nloss = n - nwin
    rr = (gw / nwin) / (gl / nloss) if nwin and nloss else 0.0
    return {"n": n, "net": cum, "mdd": mdd, "wr": nwin / n * 100, "rr": rr,
            "streak": float(maxstreak)}


def main() -> int:
    lines = ["===== DualST 추세형 백테 (1h, 트레일폭 변형, 7페어, 시드1000, 20x) =====",
             f"{'트레일':<6} {'USDT':>8} {'최대DD':>8} {'승률':>6} {'RR':>5} "
             f"{'연속손절':>8} {'거래':>6}"]
    yearly: dict[int, float] = {}
    for trail_mult in (3.0, 4.0, 5.0, 6.0, 8.0):
        agg = {"net": 0.0, "mdd": 0.0, "n": 0, "wr": 0.0, "rr": 0.0, "streak": 0.0}
        nz = 0
        for sym in PAIRS:
            df1h = _load_1h(sym)
            if len(df1h) < 100:
                continue
            tr = _run(df1h, trail_mult)
            s = _stats([t[0] for t in tr])
            agg["net"] += s["net"]
            agg["mdd"] += s["mdd"]
            agg["n"] += int(s["n"])
            agg["wr"] += s["wr"]
            agg["rr"] += s["rr"]
            agg["streak"] += s["streak"]
            nz += 1
            if trail_mult == 6.0:   # 균형점(net+RR) — 연도별 분해로 추세장 의존 확인
                for net, yr in tr:
                    yearly[yr] = yearly.get(yr, 0.0) + net
        if nz:
            lines.append(
                f"x{trail_mult:<5.1f}{agg['net'] * SEED:+8.0f} {agg['mdd'] * SEED:7.0f}↓ "
                f"{agg['wr'] / nz:5.0f}% {agg['rr'] / nz:5.2f} "
                f"{agg['streak'] / nz:7.1f}회 {agg['n']:6d}")
    lines.append("")
    lines.append("--- 연도별 net (trail x6, 7페어 합산, 펀딩 반영) ---")
    for yr in sorted(yearly):
        lines.append(f"  {yr}: {yearly[yr] * SEED:+9.0f} USDT")
    # Walk-forward: 앞 60%(in-sample)에서 좋던 trail 이 뒤 40%(out, 미관측 구간)에서도
    # 버티는지 = 과최적 판별. in 최고가 out 에서 무너지면 그 trail 은 곡선맞춤(curve-fit).
    lines.append("")
    lines.append("--- Walk-forward (앞60% in / 뒤40% out, 7페어 합산 net USDT) ---")
    lines.append(f"  {'trail':<7}{'in-net':>11}{'out-net':>11}")
    for tm in (4.0, 5.0, 6.0, 8.0):
        innet = outnet = 0.0
        for sym in PAIRS:
            df1h = _load_1h(sym)
            if len(df1h) < 200:
                continue
            k = int(len(df1h) * 0.6)
            innet += _stats([t[0] for t in _run(df1h.iloc[:k], tm)])["net"]
            outnet += _stats([t[0] for t in _run(df1h.iloc[k:], tm)])["net"]
        lines.append(f"  x{tm:<6.1f}{innet * SEED:>+11.0f}{outnet * SEED:>+11.0f}")
    txt = "\n".join(lines)
    with open("dst_trend_bt_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE")
    except UnicodeEncodeError:
        print("(결과는 dst_trend_bt_result.txt)\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
