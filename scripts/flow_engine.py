"""Flow Engine (외부 개발자 제공 Pine v6 전략) → 파이썬 이식.

원본: ``C:\\Users\\지영민\\Desktop\\Flow_Engine_Webhook.pine`` (837줄)

이 모듈은 **판정용 독립 백테스트**다. Origo/Cursus 하니스(run_live_parity 등)와
공유하는 코드는 없다. 다만 평가 단위(raw = 총손익 비율, R = 초기리스크 배수)는
우리 표준에 맞춘다.

## look-ahead 차단 3중 장치
1. **피봇 확정 지연** — ``ta.pivothigh(x, 5, 5)`` 가 봉 i 에서 값을 주면 그 피봇은
   봉 i-5 에 있다. 원본 Pine 도 봉 i 에서 신호를 내므로 **원본 자체는 미래를 보지
   않는다**. 이식본도 피봇 위치 p 가 아니라 **확정봉 p+lbR** 에서만 신호를 켠다.
2. **일봉 HTF shift** — 기본 ``htf_mode="closed"``: 장중 봉 t 는 **완결된 직전
   일봉**만 본다. (참고용으로 Pine 의 lookahead_off 실제 동작인 "형성 중 일봉"
   모드 ``"developing"`` 도 제공 — 이것도 미래참조는 아니지만 더 낙관적이다.)
3. **신호봉 다음 봉 체결** — 신호는 봉 s 마감 시 확정, 체결은 봉 s+1 시가.
   TP/SL 감시는 그보다 한 봉 더 뒤(``exit_start_offset=1``)부터 → 같은 봉
   진입+청산이 원천적으로 불가능.

## 우리 판정과 Pine 의 차이 (이식 시 판단이 갈린 지점)
아래 항목은 모두 옵션으로 뚫어놨고, 기본값은 **보수적인 쪽**을 골랐다.
자세한 근거는 각 옵션 docstring 참조.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ══════════════════════════════════════════════════════════════════
# 1. Pine 내장함수 재현
# ══════════════════════════════════════════════════════════════════


def _ema(s: pd.Series, n: int) -> pd.Series:
    """Pine ``ta.ema`` — alpha=2/(n+1) 재귀. 초기값은 첫 n봉 SMA 로 시드.

    Pine 은 워밍업 구간에서 na 를 내고 SMA 로 시드한다. 5년 데이터에서 시드 차이는
    무의미하지만, prefix 재계산 인과성 검증(verify_causality)이 정확히 일치하려면
    시드 규칙이 결정적이어야 하므로 명시적으로 SMA 시드를 쓴다.
    """
    return pd.Series(_recur(s.to_numpy(dtype=float), n, 2.0 / (n + 1.0)),
                     index=s.index)


def _recur(v: np.ndarray, n: int, a: float) -> np.ndarray:
    """alpha 재귀 공통부 — **선행 NaN 을 건너뛰고** 첫 유효 n개 SMA 로 시드.

    ⚠️ 이식 초기 버그: 선행 NaN 을 그대로 시드에 넣으면 NaN 이 끝까지 전파되어
    trend_smooth 가 전 구간 NaN → 모멘텀 크로스 신호가 통째로 죽는다. Pine 은
    소스가 na 인 동안 ema 를 시작하지 않으므로, 첫 유효값부터 시드해야 정합.
    """
    out = np.full(len(v), np.nan)
    fin = np.flatnonzero(np.isfinite(v))
    if len(fin) < n:
        return out
    s0 = fin[n - 1]
    out[s0] = v[fin[:n]].mean()
    for i in range(s0 + 1, len(v)):
        out[i] = a * v[i] + (1 - a) * out[i - 1] if np.isfinite(v[i]) else out[i - 1]
    return out


def _rma(s: pd.Series, n: int) -> pd.Series:
    """Pine ``ta.rma`` (Wilder) — alpha=1/n, 첫 n봉 SMA 시드. RSI/ATR 의 기반."""
    return pd.Series(_recur(s.to_numpy(dtype=float), n, 1.0 / n), index=s.index)


def _rsi(close: pd.Series, n: int) -> pd.Series:
    """Pine ``ta.rsi`` — 상승분/하락분의 RMA 비율."""
    d = close.diff()
    up = _rma(d.clip(lower=0).fillna(0.0), n)
    dn = _rma((-d).clip(lower=0).fillna(0.0), n)
    rs = up / dn.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out[dn == 0] = 100.0
    out[(up == 0) & (dn == 0)] = 50.0
    return out


def _atr(df: pd.DataFrame, n: int) -> pd.Series:
    """Pine ``ta.atr`` — True Range 의 RMA."""
    pc = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return _rma(tr, n)


def _pivots(x: np.ndarray, lbl: int, lbr: int, kind: str) -> np.ndarray:
    """Pine ``ta.pivothigh/pivotlow`` 의 피봇 **위치** 인덱스 배열.

    판단이 갈린 지점 ①: **동률(tie) 처리**. Pine 의 정확한 동률 규칙은 문서화되어
    있지 않다(평탄 구간에서 na 를 내는 것으로 알려짐). 여기서는 **좌우 모두 강부등호**
    를 요구한다. mom_norm 은 연속 실수라 정확한 동률은 사실상 발생하지 않으므로
    이 선택이 결과를 바꿀 여지는 없다(실측: 동률로 탈락한 후보 0건 수준).
    """
    n = len(x)
    if n < lbl + lbr + 1:
        return np.array([], dtype=int)
    w = np.lib.stride_tricks.sliding_window_view(x, lbl + lbr + 1)  # (n-lbl-lbr, W)
    c = w[:, lbl]
    left, right = w[:, :lbl], w[:, lbl + 1:]
    with np.errstate(invalid="ignore"):
        if kind == "high":
            ok = (c > left.max(axis=1)) & (c > right.max(axis=1))
        else:
            ok = (c < left.min(axis=1)) & (c < right.min(axis=1))
    ok &= np.isfinite(w).all(axis=1)
    return np.flatnonzero(ok) + lbl


# ══════════════════════════════════════════════════════════════════
# 2. HTF (일봉) 바이어스
# ══════════════════════════════════════════════════════════════════


def _htf_bias(df: pd.DataFrame, htf: str = "1D", mode: str = "closed"):
    """일봉 추세 바이어스 (htf_bull / htf_bear) 를 장중 봉에 매핑.

    판단이 갈린 지점 ②: **Pine 의 request.security 실제 동작 vs 안전판**.
    ``request.security(sym,"D",close)`` 는 lookahead_off 기본값에서 **형성 중인**
    일봉 값을 준다(= 현재 장중 close). 이건 미래참조는 아니지만, 일봉 EMA 가
    장중에 계속 움직여 신호가 리페인트처럼 보인다.
    - ``mode="closed"`` (기본): **완결된 직전 일봉**만 사용 → 확실히 안전.
    - ``mode="developing"``: Pine 원본 재현(형성 중 일봉 close = 현재 close 로
      일봉 EMA 를 1스텝 전진). 원본과의 대조용.

    반환: (bull, bear, src_end_ns) — src_end_ns 는 각 봉이 참조한 일봉의 **종료
    시각**(ns). 검증 (b) 에서 이 값이 봉 시각을 넘지 않는지 확인한다.
    """
    d = df["close"].resample(htf).last().dropna()
    de_f = _ema(d, 21)
    de_s = _ema(d, 50)

    day = df.index.floor(htf)
    if mode == "closed":
        # 직전 완결 일봉 값으로 shift
        c_p = d.shift(1).reindex(day).to_numpy(dtype=float)
        f_p = de_f.shift(1).reindex(day).to_numpy(dtype=float)
        s_p = de_s.shift(1).reindex(day).to_numpy(dtype=float)
        bull = (c_p > f_p) & (f_p > s_p)
        bear = (c_p < f_p) & (f_p < s_p)
        # 참조한 일봉의 종료시각 = 당일 시작 시각 (직전 일봉의 끝)
        src_end = day
    else:
        a21, a50 = 2 / 22.0, 2 / 51.0
        f_p = de_f.shift(1).reindex(day).to_numpy(dtype=float)
        s_p = de_s.shift(1).reindex(day).to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        f_dev = a21 * c + (1 - a21) * f_p
        s_dev = a50 * c + (1 - a50) * s_p
        bull = (c > f_dev) & (f_dev > s_dev)
        bear = (c < f_dev) & (f_dev < s_dev)
        src_end = df.index  # 현재 봉까지 (미래 아님)

    bull = pd.Series(np.where(np.isnan(f_p), False, bull), index=df.index)
    bear = pd.Series(np.where(np.isnan(f_p), False, bear), index=df.index)
    return bull.astype(bool), bear.astype(bool), src_end


# ══════════════════════════════════════════════════════════════════
# 3. 신호
# ══════════════════════════════════════════════════════════════════

SIG_DEFAULTS: dict[str, Any] = dict(
    mom_len=14,
    trend_len=50,
    ob_level=70.0,
    os_level=-70.0,
    lbL=5,
    lbR=5,
    htf="1D",
    htf_mode="closed",
    entry_mode="Both",          # "Divergence Only" | "Momentum Cross" | "Both"
    use_htf_filter=True,
    direction="Both",           # "Long Only" | "Short Only" | "Both"
)


def signals(df: pd.DataFrame, **kw) -> pd.DataFrame:
    """Pine 7~8장(142~249행)의 신호를 그대로 재현.

    반환 컬럼: buy_sig / sell_sig / div_long / div_short / mom_long / mom_short /
    mom_norm / osc_line / trend_smooth / htf_bull / htf_bear / div_pivot_idx
    (div 신호가 켜진 봉에서 그 근거 피봇의 **위치 인덱스**. 검증 (a) 용).

    ``df`` 는 UTC DatetimeIndex + open/high/low/close/volume.
    **순수 함수**다 — 앞부분만 잘라 넣어도 같은 결과가 나와야 한다(인과성 검증).
    """
    o = {**SIG_DEFAULTS, **kw}
    n = len(df)
    close = df["close"]

    # --- Momentum ---
    mom_raw = (close / close.shift(o["mom_len"]) - 1.0) * 100.0
    mom_max = mom_raw.abs().rolling(100, min_periods=100).max()
    mom_norm = np.where(
        (mom_max.to_numpy() != 0) & np.isfinite(mom_max.to_numpy()),
        mom_raw.to_numpy() / mom_max.to_numpy() * 100.0,
        0.0,
    )
    mom_norm = pd.Series(mom_norm, index=df.index)
    # Pine 의 var/na 전파와 달리 numpy 는 nan 을 남기므로, 워밍업 구간을 명시적으로 막는다
    warm = ~np.isfinite(mom_max.to_numpy())
    mom_norm[warm] = np.nan

    # --- OB/OS ---
    # ★★ 원본 결함: osc_line = (RSI-50)*1.4 이므로 값역이 **정확히 [-70, +70]**.
    #    ob_level=+70 / os_level=-70 은 그 값역의 **양 끝점**이라, crossover(osc,-70)
    #    은 RSI[1]<=0, crossunder(osc,+70) 은 RSI[1]>=100 을 요구한다 → 사실상 불가능.
    #    실측(BTC/ETH × 15m/1h/4h 6조합, 5년): OS크로스 0건 · OB크로스 0건.
    #    즉 mom_long/mom_short 의 **두 번째 절(OB/OS 반등)은 통째로 죽은 코드**이고
    #    실제로는 (mom_norm 0선 교차 ∧ trend_smooth 부호) 만 작동한다.
    osc_line = (_rsi(close, o["mom_len"]) - 50.0) * 1.4

    # --- Trend ---
    ema_fast = _ema(close, 21)
    ema_slow = _ema(close, o["trend_len"])
    trend_smooth = _ema((ema_fast - ema_slow) / ema_slow * 1000.0, 8)

    # --- HTF ---
    htf_bull, htf_bear, src_end = _htf_bias(df, o["htf"], o["htf_mode"])

    # --- Divergence (피봇 확정봉에서만 켜진다) ---
    mn = mom_norm.to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    lbl, lbr = o["lbL"], o["lbR"]

    bear_div = np.zeros(n, bool)
    bull_div = np.zeros(n, bool)
    piv_idx = np.full(n, -1, dtype=int)   # 검증용: 근거 피봇의 위치

    prev_ph_mom = prev_ph_price = None
    for p in _pivots(mn, lbl, lbr, "high"):
        c = p + lbr                      # ★ 확정봉 = 피봇 + lbR
        if c >= n:
            break
        if prev_ph_mom is not None and hi[p] > prev_ph_price and mn[p] < prev_ph_mom:
            bear_div[c] = True
            piv_idx[c] = p
        prev_ph_mom, prev_ph_price = mn[p], hi[p]

    prev_pl_mom = prev_pl_price = None
    for p in _pivots(mn, lbl, lbr, "low"):
        c = p + lbr
        if c >= n:
            break
        if prev_pl_mom is not None and lo[p] < prev_pl_price and mn[p] > prev_pl_mom:
            bull_div[c] = True
            piv_idx[c] = p
        prev_pl_mom, prev_pl_price = mn[p], lo[p]

    # --- Momentum Cross ---
    mnp = mom_norm.shift(1)
    ocp = osc_line.shift(1)
    cross_up = (mom_norm > 0) & (mnp <= 0)
    cross_dn = (mom_norm < 0) & (mnp >= 0)
    osc_from_os = (osc_line > o["os_level"]) & (ocp <= o["os_level"])
    osc_from_ob = (osc_line < o["ob_level"]) & (ocp >= o["ob_level"])

    mom_long = (cross_up & (trend_smooth > 0)) | (osc_from_os & (mom_norm > -50))
    mom_short = (cross_dn & (trend_smooth < 0)) | (osc_from_ob & (mom_norm < 50))
    mom_long = mom_long.fillna(False)
    mom_short = mom_short.fillna(False)

    div_long = pd.Series(bull_div, index=df.index)
    div_short = pd.Series(bear_div, index=df.index)

    em = o["entry_mode"]
    if em == "Divergence Only":
        raw_long, raw_short = div_long, div_short
    elif em == "Momentum Cross":
        raw_long, raw_short = mom_long, mom_short
    else:
        raw_long, raw_short = div_long | mom_long, div_short | mom_short

    # HTF 필터: 중립이면 통과, 역방향일 때만 차단
    if o["use_htf_filter"]:
        htf_long_ok = ~htf_bear
        htf_short_ok = ~htf_bull
    else:
        htf_long_ok = pd.Series(True, index=df.index)
        htf_short_ok = pd.Series(True, index=df.index)

    dir_long_ok = o["direction"] != "Short Only"
    dir_short_ok = o["direction"] != "Long Only"

    out = pd.DataFrame(index=df.index)
    out["buy_sig"] = (raw_long & htf_long_ok & dir_long_ok).fillna(False)
    out["sell_sig"] = (raw_short & htf_short_ok & dir_short_ok).fillna(False)
    out["div_long"], out["div_short"] = div_long, div_short
    out["mom_long"], out["mom_short"] = mom_long, mom_short
    out["mom_norm"], out["osc_line"] = mom_norm, osc_line
    out["trend_smooth"] = trend_smooth
    out["htf_bull"], out["htf_bear"] = htf_bull, htf_bear
    out["div_pivot_idx"] = piv_idx
    out.attrs["htf_src_end"] = src_end
    return out


# ══════════════════════════════════════════════════════════════════
# 4. TP/SL · 트레일링
# ══════════════════════════════════════════════════════════════════

BT_DEFAULTS: dict[str, Any] = dict(
    # --- TP/SL ---
    tpsl_mode="ATR Dynamic",     # "ATR Dynamic" | "Fixed Percent" | "Manual Percent"
    atr_period=14,
    atr_tp=(1.0, 2.0, 3.0, 4.0),
    atr_sl=1.5,
    fix_tp=(1.0, 2.0, 3.0, 4.0),
    fix_sl=2.0,
    alloc=(25.0, 25.0, 25.0, 25.0),
    # --- 트레일링 ---
    use_trailing=True,
    trail_type="Moving Target",   # Moving Target|Moving 2-Target|Breakeven|
                                  # Percent Below Triggers|Percent Below Highest
    trail_trigger_mode="Target",  # "Target" | "Percentage"
    trail_trigger=2,
    trail_trigger_pct=2.0,
    trail_pct=1.0,
    # --- 체결 모델 ---
    fill="next_open",             # "next_open"(기본,보수) | "close"(Pine 원본)
    exit_start_offset=1,          # 체결봉 + N 부터 TP/SL 감시 (기본 1 = 같은봉 청산 불가)
    intrabar="tv",                # "tv"(봉방향 경로) | "sl_first"(비관)
    slip_pct=0.0002,              # 시장가/스톱 체결 슬리피지 (지정가 TP 는 0)
    allow_reverse=True,           # 반대신호 시 청산 후 역진입 (Pine 동작)
)


def _tp_sl_pct(row_atr: float, ref: float, o: dict) -> tuple[list[float], float]:
    """진입 스냅샷의 TP1~4 / SL 을 **가격 대비 %** 로. Pine 10장(288~316행)."""
    if o["tpsl_mode"] == "ATR Dynamic":
        tps = [row_atr * m / ref * 100.0 for m in o["atr_tp"]]
        sl = row_atr * o["atr_sl"] / ref * 100.0
    else:
        tps = list(o["fix_tp"])
        sl = o["fix_sl"]
    return tps, sl


class _Pos:
    """포지션 상태 — Pine 13·14장(366~517행)의 var 들을 그대로 들고 있다."""

    __slots__ = ("is_long", "ref", "fill", "tp_px", "sl0_px", "hits",
                 "trail_active", "extreme", "o", "entry_i", "sig_i")

    def __init__(self, is_long, ref, fill_px, tps_pct, sl_pct, o, entry_i, sig_i):
        s = 1.0 if is_long else -1.0
        self.is_long = is_long
        self.ref = ref
        self.fill = fill_px
        self.tp_px = [ref * (1 + s * p / 100.0) for p in tps_pct]
        self.sl0_px = ref * (1 - s * sl_pct / 100.0)
        self.hits = 0
        self.trail_active = False
        self.extreme = None      # Percent Below Highest 용 최고/최저
        self.o = o
        self.entry_i = entry_i
        self.sig_i = sig_i

    def dyn_sl(self) -> float:
        """Pine ``f_dynamic_sl_long/short`` 완전 이식 (트레일 5종)."""
        o, s = self.o, (1.0 if self.is_long else -1.0)
        tt, h = o["trail_type"], self.hits
        eff = max(o["trail_trigger"], 2) if tt == "Moving 2-Target" else o["trail_trigger"]
        if tt in ("Moving Target", "Moving 2-Target"):
            trig = h >= eff
        elif o["trail_trigger_mode"] == "Percentage":
            trig = self.trail_active
        else:
            trig = h >= eff

        if not o["use_trailing"] or not trig:
            return self.sl0_px
        if tt == "Breakeven":
            return self.ref
        if tt == "Moving Target":
            if h >= 4:
                return self.tp_px[2]
            if h >= 3:
                return self.tp_px[1]
            if h >= 2:
                return self.tp_px[0]
            return self.ref
        if tt == "Moving 2-Target":
            if h >= 4:
                return self.tp_px[1]
            if h >= 3:
                return self.tp_px[0]
            return self.ref
        if tt == "Percent Below Triggers":
            if h >= 1:
                trigger_price = self.tp_px[min(h, 4) - 1]
            else:
                trigger_price = self.ref * (1 + s * o["trail_trigger_pct"] / 100.0)
            return trigger_price * (1 - s * o["trail_pct"] / 100.0)
        # Percent Below Highest
        ref = self.extreme if self.extreme is not None else self.ref
        return ref * (1 - s * o["trail_pct"] / 100.0)


# ══════════════════════════════════════════════════════════════════
# 5. 백테스트
# ══════════════════════════════════════════════════════════════════


def backtest(df: pd.DataFrame, **kw) -> list[dict]:
    """Flow Engine 독립 백테스트.

    반환: 거래 dict 리스트
        ts       진입 체결 시각 (UTC)
        ex       청산 완료 시각
        raw      **총손익 비율** (레버리지·수수료 전, 슬리피지 반영).
                 Origo replay 의 raw_pnl_pct 와 같은 단위 (0.012 = +1.2%)
        r        R 배수 = raw / (초기 SL 거리 비율)
        dir      +1 롱 / -1 숏
        outcome  'tp4'|'sl'|'trail'|'reverse'|'eod'
        entry    실제 체결가 (슬리피지 반영)
        sl       초기 SL 가격
        (부가) ref, tp_hits, entry_idx, exit_idx, sig_idx, reason

    ★ look-ahead 차단
      · 신호는 봉 s 마감 확정 → 체결은 봉 s+1 시가 (fill="next_open")
      · TP/SL 감시는 봉 s+1+exit_start_offset 부터 (기본 1 → s+2)
      · 일봉 HTF 는 직전 완결 일봉
    """
    o = {**SIG_DEFAULTS, **BT_DEFAULTS, **kw}
    sig = signals(df, **{k: o[k] for k in SIG_DEFAULTS})
    atr = _atr(df, o["atr_period"]).to_numpy(dtype=float)

    ts = df.index
    O = df["open"].to_numpy(dtype=float)
    H = df["high"].to_numpy(dtype=float)
    L = df["low"].to_numpy(dtype=float)
    C = df["close"].to_numpy(dtype=float)
    buy = sig["buy_sig"].to_numpy()
    sell = sig["sell_sig"].to_numpy()
    dl = sig["div_long"].to_numpy()
    ds = sig["div_short"].to_numpy()
    ml = sig["mom_long"].to_numpy()
    ms = sig["mom_short"].to_numpy()

    n = len(df)
    alloc = np.array(o["alloc"], float) / 100.0
    slip = o["slip_pct"]
    trades: list[dict] = []

    pos: _Pos | None = None
    realized: list[tuple[float, float]] = []   # (비중, 청산가)
    pend: tuple[int, bool] | None = None       # (신호봉 idx, is_long)
    watch_from = 0

    def _open_trade(p: _Pos, exit_i: int, outcome: str):
        s = 1.0 if p.is_long else -1.0
        raw = sum(w * (px - p.fill) / p.fill * s for w, px in realized)
        risk = abs(p.ref - p.sl0_px) / p.ref
        trades.append(dict(
            ts=ts[p.entry_i], ex=ts[exit_i], raw=float(raw),
            r=float(raw / risk) if risk > 0 else 0.0,
            dir=int(s), outcome=outcome, entry=float(p.fill), sl=float(p.sl0_px),
            ref=float(p.ref), tp_hits=int(p.hits),
            entry_idx=int(p.entry_i), exit_idx=int(exit_i), sig_idx=int(p.sig_i),
            reason=("divergence+momentum" if (dl[p.sig_i] and ml[p.sig_i]) or
                    (ds[p.sig_i] and ms[p.sig_i])
                    else ("divergence" if (dl[p.sig_i] or ds[p.sig_i])
                          else "momentum_cross")),
        ))

    for i in range(n):
        # ── (1) 대기 주문 체결: 신호봉 s 의 다음 봉 시가 ──────────────
        if pend is not None and i == pend[0] + 1:
            s_i, is_long = pend
            pend = None
            s = 1.0 if is_long else -1.0
            # 반대 포지션이면 먼저 시장가 청산 (Pine: strategy.close → entry)
            if pos is not None and pos.is_long != is_long:
                px = O[i] * (1 - (1.0 if pos.is_long else -1.0) * slip)
                realized.append((max(1.0 - alloc[:pos.hits].sum(), 0.0), px))
                _open_trade(pos, i, "reverse")
                pos = None
            if pos is None and np.isfinite(atr[s_i]):
                ref = C[s_i]                      # Pine entry_price := close
                tps_pct, sl_pct = _tp_sl_pct(atr[s_i], ref, o)
                fill_px = (O[i] if o["fill"] == "next_open" else C[s_i]) * (1 + s * slip)
                pos = _Pos(is_long, ref, fill_px, tps_pct, sl_pct, o, i, s_i)
                realized = []
                watch_from = i + o["exit_start_offset"]

        # ── (2) TP/SL 감시 (봉 내부 경로 모델) ───────────────────────
        if pos is not None and i >= watch_from:
            done = _walk_bar(pos, O[i], H[i], L[i], C[i], realized, alloc, slip,
                             o["intrabar"])
            if done is not None:
                _open_trade(pos, i, done)
                pos = None

        # ── (3) 신호 발생 (봉 마감 확정) ─────────────────────────────
        if pend is None and i + 1 < n:
            if buy[i] and (pos is None or not pos.is_long):
                if pos is None or o["allow_reverse"]:
                    pend = (i, True)
            elif sell[i] and (pos is None or pos.is_long):
                if pos is None or o["allow_reverse"]:
                    pend = (i, False)

    if pos is not None:
        realized.append((max(1.0 - alloc[:pos.hits].sum(), 0.0), C[n - 1]))
        _open_trade(pos, n - 1, "eod")
    return trades


def _walk_bar(p: _Pos, o_, h_, l_, c_, realized, alloc, slip, mode) -> str | None:
    """한 봉 안에서 가격이 지나간 **경로**를 따라 TP/SL 체결을 판정.

    판단이 갈린 지점 ③: **봉 내부 순서**. Pine(TradingView) 브로커 에뮬레이터는
    바 방향으로 경로를 가정한다 — 양봉 O→L→H→C, 음봉 O→H→L→C. ``mode="tv"``
    가 그것이고, ``mode="sl_first"`` 는 SL 을 무조건 먼저 보는 비관 모델이다.
    갭(시가가 이미 레벨 너머)이면 **시가 체결**로 처리한다(레벨 체결보다 불리).
    """
    s = 1.0 if p.is_long else -1.0

    def hit_tp(level_px: float) -> bool:
        return (level_px >= p.tp_px[p.hits]) if p.is_long else (level_px <= p.tp_px[p.hits])

    def hit_sl(level_px: float, sl: float) -> bool:
        return (level_px <= sl) if p.is_long else (level_px >= sl)

    def take_tp(px: float):
        """지정가 체결 → 슬리피지 0. Pine 도 limit 은 레벨 체결."""
        realized.append((alloc[p.hits], px))
        p.hits += 1

    def take_sl(px: float):
        realized.append((max(1.0 - alloc[:p.hits].sum(), 0.0), px * (1 - s * slip)))

    # --- 갭: 시가가 이미 넘었으면 시가 체결 ---
    sl = p.dyn_sl()
    if hit_sl(o_, sl):
        take_sl(o_)
        return "sl" if p.hits == 0 else "trail"
    while p.hits < 4 and hit_tp(o_):
        take_tp(o_)
        if p.hits >= 4:
            return "tp4"
        sl = p.dyn_sl()
        if hit_sl(o_, sl):
            take_sl(o_)
            return "trail"

    if mode == "sl_first":
        pts = [o_, (l_ if p.is_long else h_), (h_ if p.is_long else l_), c_]
    else:
        pts = [o_, l_, h_, c_] if c_ >= o_ else [o_, h_, l_, c_]

    for a, b in zip(pts[:-1], pts[1:]):
        up = b >= a
        if (up and p.is_long) or (not up and not p.is_long):
            # 이익 방향 다리 → TP 캐스케이드
            if p.o["trail_trigger_mode"] == "Percentage" and not p.trail_active:
                trg = p.ref * (1 + s * p.o["trail_trigger_pct"] / 100.0)
                if (b >= trg) if p.is_long else (b <= trg):
                    p.trail_active = True
            if p.o["trail_type"] == "Percent Below Highest":
                trig = (p.trail_active if p.o["trail_trigger_mode"] == "Percentage"
                        else p.hits >= p.o["trail_trigger"])
                if trig:
                    p.extreme = b if p.extreme is None else (
                        max(p.extreme, b) if p.is_long else min(p.extreme, b))
            while p.hits < 4 and hit_tp(b):
                take_tp(p.tp_px[p.hits])   # 지정가 → 레벨 그대로 체결
                if p.hits >= 4:
                    return "tp4"
                if p.o["trail_type"] == "Percent Below Highest":
                    trig = (p.trail_active if p.o["trail_trigger_mode"] == "Percentage"
                            else p.hits >= p.o["trail_trigger"])
                    if trig:
                        p.extreme = b if p.extreme is None else (
                            max(p.extreme, b) if p.is_long else min(p.extreme, b))
            sl = p.dyn_sl()
            if hit_sl(b, sl):        # 트레일 SL 이 올라와 현재가를 추월한 경우
                take_sl(b)
                return "trail"
        else:
            sl = p.dyn_sl()
            if hit_sl(b, sl):
                take_sl(sl)
                return "sl" if p.hits == 0 else "trail"
    return None


# ══════════════════════════════════════════════════════════════════
# 6. 데이터 · 평가 유틸
# ══════════════════════════════════════════════════════════════════

_RES = "C:/Users/지영민/Desktop/Aurora-ICT-research"


def load(sym: str, tf: str = "1h") -> pd.DataFrame:
    """1분 원본 → 지정 TF 리샘플 (bt_par._load_full / _resample 과 동일 규약)."""
    d = pd.read_parquet(os.path.join(_RES, f"data/{sym}_1m_full.parquet"))
    d.index = pd.DatetimeIndex(pd.to_datetime(d["timestamp"], unit="ms", utc=True))
    d = d[["open", "high", "low", "close", "volume"]]
    if tf in ("1m", "1min"):
        return d
    r = {"5m": "5min", "15m": "15min", "30m": "30min",
         "1h": "1h", "4h": "4h", "1d": "1D"}.get(tf, tf)
    return pd.DataFrame({
        "open": d["open"].resample(r).first(),
        "high": d["high"].resample(r).max(),
        "low": d["low"].resample(r).min(),
        "close": d["close"].resample(r).last(),
        "volume": d["volume"].resample(r).sum(),
    }).dropna()


def summarize(trades: list[dict], df: pd.DataFrame, label: str = "") -> dict:
    """우리 표준 요약 — 거래수·월빈도·건당R·승률·복리자산·MDD (7x / 10x)."""
    if not trades:
        print(f"  {label}: 거래 0건")
        return {}
    r = np.array([t["r"] for t in trades])
    raw = np.array([t["raw"] for t in trades])
    months = (df.index[-1] - df.index[0]).days / 30.44
    res = dict(n=len(r), per_month=len(r) / months, mean_r=r.mean(),
               med_r=float(np.median(r)), win=float((r > 0).mean()), sum_r=r.sum())
    print(f"\n  [{label}] 거래 {len(r)}건 · 월 {len(r)/months:.2f}건 · "
          f"건당 {r.mean():+.3f}R (중앙 {np.median(r):+.3f}R) · "
          f"승률 {(r>0).mean()*100:.1f}% · ΣR {r.sum():+.1f}")
    for lev in (7, 10):
        for fee, fl in ((0.0008, "우리 0.08%"), (0.0012, "Pine 0.12%")):
            eq, peak, mdd, ruin = 1.0, 1.0, 0.0, False
            for x in raw:
                step = x * 0.9 * lev - fee * 0.9 * lev
                eq *= (1 + step)
                if eq <= 0.20:
                    ruin = True
                    break
                peak = max(peak, eq)
                mdd = max(mdd, 1 - eq / peak)
            print(f"    {lev:>2}x · {fl}: 자산 {eq:8.3f}x · MDD {mdd*100:5.1f}%"
                  f"{' · 파산' if ruin else ''}")
            res[f"eq{lev}_{int(fee*10000)}"] = eq
    return res


# ══════════════════════════════════════════════════════════════════
# 7. look-ahead 검증 3종
# ══════════════════════════════════════════════════════════════════


def verify(df: pd.DataFrame, trades: list[dict], n_probe: int = 250, **kw) -> None:
    """(a) 피봇 확정 지연 · (b) 일봉 미래참조 · (c) 같은 봉 진입+청산."""
    o = {**SIG_DEFAULTS, **kw}
    sig = signals(df, **{k: o[k] for k in SIG_DEFAULTS})
    print("\n=== look-ahead 검증 ===")

    # (a-1) div 신호는 반드시 근거 피봇 + lbR 봉에서만
    piv = sig["div_pivot_idx"].to_numpy()
    idx = np.where(piv >= 0)[0]
    lag = idx - piv[idx]
    ok = int((lag == o["lbR"]).sum())
    print(f"  (a-1) 피봇 기반 신호 {len(idx)}건 — 확정 지연 = lbR({o['lbR']})봉 "
          f"{ok}건 / 최소 {lag.min() if len(lag) else 0} · 최대 "
          f"{lag.max() if len(lag) else 0} → {'통과' if ok == len(idx) else '실패'}")

    # (a-2) 인과성: 앞부분만 잘라 재계산해도 마지막 봉 신호가 동일해야 한다.
    #       ★ 표본의 절반은 **실제 신호가 켜진 봉**에서 뽑는다 — 무작위 봉만 보면
    #         신호가 거의 없어(5년 중 ~1%) 검증이 헛돈다.
    rng = np.random.default_rng(20260808)
    sig_bars = np.flatnonzero(
        (sig["buy_sig"].to_numpy() | sig["sell_sig"].to_numpy())
        & (np.arange(len(df)) >= 500))
    half = min(n_probe // 2, len(sig_bars))
    ks = np.concatenate([
        rng.choice(sig_bars, size=half, replace=False),
        rng.choice(np.arange(500, len(df)), size=n_probe - half, replace=False),
    ])
    mism = 0
    hit = 0
    for k in ks:
        sub = signals(df.iloc[: k + 1], **{kk: o[kk] for kk in SIG_DEFAULTS})
        b0, s0 = sig["buy_sig"].iat[k], sig["sell_sig"].iat[k]
        b1, s1 = sub["buy_sig"].iat[-1], sub["sell_sig"].iat[-1]
        if (b0 != b1) or (s0 != s1):
            mism += 1
        hit += int(b0 or s0)
    print(f"  (a-2) prefix 재계산 인과성: {len(ks)}개 봉 표본(신호봉 {hit}개 포함) "
          f"— 불일치 {mism}건 → {'통과' if mism == 0 else '실패'}")

    # (b) 일봉 HTF: 참조한 일봉 종료시각이 현재 봉 시각을 넘지 않아야
    src = pd.DatetimeIndex(sig.attrs["htf_src_end"])
    delta = (src - df.index).total_seconds().to_numpy()
    print(f"  (b)   일봉 참조 {len(df)}봉 — 참조 일봉 종료시각 - 현재봉시각: "
          f"최대 {delta.max()/3600:+.2f}시간 → "
          f"{'통과(미래 없음)' if delta.max() <= 0 else '실패'}")

    # (c) 같은 봉 진입+청산
    same = sum(1 for t in trades if t["entry_idx"] == t["exit_idx"])
    lag_min = min((t["entry_idx"] - t["sig_idx"]) for t in trades) if trades else 0
    print(f"  (c)   거래 {len(trades)}건 — 같은 봉 진입+청산 {same}건 · "
          f"신호→체결 최소지연 {lag_min}봉 → "
          f"{'통과' if same == 0 and lag_min >= 1 else '실패'}")


# ══════════════════════════════════════════════════════════════════

def main() -> int:
    df = load("BTCUSDT", "1h")
    print(f"BTCUSDT 1h — {len(df)}봉  {df.index[0]} ~ {df.index[-1]}")
    tr = backtest(df)
    verify(df, tr)
    print("\n=== 기본 설정 (Both · HTF필터 · ATR Dynamic · Moving Target) ===")
    summarize(tr, df, "BTC 1h 기본(보수)")

    # 보수적 선택이 얼마나 비싼지 — Pine 원본 동작에 가까운 쪽과 대조.
    # (셋 다 look-ahead 는 아니다. 낙관/보수의 차이일 뿐)
    print("\n=== Pine 원본 동작 대조 (look-ahead 아님 · 낙관도 차이) ===")
    for lab, kw in (
        ("HTF 형성중 일봉(Pine 실제)", dict(htf_mode="developing")),
        ("신호봉 종가 체결(Pine 실제)", dict(fill="close")),
        ("체결봉부터 TP/SL 감시", dict(exit_start_offset=0)),
        ("SL 우선(비관)", dict(intrabar="sl_first")),
        ("슬리피지 0", dict(slip_pct=0.0)),
        ("HTF 필터 OFF", dict(use_htf_filter=False)),
        ("Divergence Only", dict(entry_mode="Divergence Only")),
        ("Momentum Cross Only", dict(entry_mode="Momentum Cross")),
    ):
        summarize(backtest(df, **kw), df, lab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
