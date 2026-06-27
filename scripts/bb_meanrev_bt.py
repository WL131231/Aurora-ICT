"""볼린저밴드 평균회귀 백테 — 횡보장 전용 매매 연구 (투트랙 3번째 후보).

파트너 자율연구 요청(2026-06-27): Origo(ICT 단타)·Cursus(DualST 추세)가 약한
횡보장을 메우는 볼린저밴드 평균회귀 전략을 백테로 검증. 타사 볼린저 봇 분석
([[bollinger-bot-2026-06-25]])에서 진입신호(BB+STO)만 차용하고, 그쪽의 치명적
약점(마틴게일·무손절 물타기)은 **완전히 배제** — 타이트 고정 SL + 단일 진입.

전략:
    - 횡보 판단: ADX(추세 강도) < 임계일 때만 진입. 추세장(ADX 높음)에선 가격이
      밴드를 타고 추세를 지속해 평균회귀가 학살당하므로 진입 금지가 핵심.
    - 진입: 종가가 하단밴드 이탈 + 스토캐스틱 과매도 → 롱(평균회귀 기대).
            종가가 상단밴드 이탈 + 스토캐스틱 과매수 → 숏.
    - 청산: 중심선(SMA20) 도달 = 평균회귀 완료 → 익절. ATR 기반 고정 SL → 손절.
            ttl 봉 내 미청산이면 종가 청산(평균회귀는 빠르게 끝나야 정상).
    - **물타기 없음**: 포지션당 1회 진입, SL 무조건 실행. 타사 봇과 정반대.

검증: 7페어(고정7) 5년 1h, 시드 1000, 20x, size 0.9. 평가 = net USDT / 최대DD /
승률 / RR / 거래수 + 연도별 + walk-forward(과최적 판별). 수수료·슬리피지·펀딩은
aurora.backtest.cost 재사용 → Origo/Cursus 백테와 동일 기준.

lookahead 방지: 신호는 봉 종가 확정 후 → 다음 봉 시가 진입. 청산 타겟(중심선)·SL 은
직전 봉까지 확정된 값으로 현재 봉 장중(high/low) 히트 판정(같은 봉 종가 체결 낙관 제거).

실행: cwd=Aurora-ICT-research, PYTHONPATH=../Aurora-ICT/src.
담당: 지영민 (횡보장 평균회귀 연구).
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
BB_PERIOD = 20
BB_MULT = 2.0
ADX_PERIOD = 14
STO_PERIOD = 14
STO_OS = 20.0   # 스토캐스틱 과매도 — 롱 진입 확인
STO_OB = 80.0   # 과매수 — 숏 진입 확인
TTL_BARS = 48   # 1h 기준 2일 — 평균회귀 미완 시 종가 청산
FUNDING_PER_HOUR = 0.0001 / 8  # Bybit 평균 펀딩 8h 0.01% → 시간당


def _load_1h(sym: str) -> pd.DataFrame:
    """1m parquet → 1h OHLCV 리샘플 (data/{sym}_1m_full.parquet)."""
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    o = df["open"].resample("60min").first()
    h = df["high"].resample("60min").max()
    lo = df["low"].resample("60min").min()
    c = df["close"].resample("60min").last()
    v = df["volume"].resample("60min").sum()
    return pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "volume": v},
    ).dropna()


def _atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR."""
    h, low, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - low), (h - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _bbands(df: pd.DataFrame, period: int, mult: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """볼린저밴드 — (중심선 SMA, 상단 +mult·σ, 하단 -mult·σ)."""
    mid = df["close"].rolling(period).mean()
    sd = df["close"].rolling(period).std(ddof=0)
    return mid, mid + mult * sd, mid - mult * sd


def _adx(df: pd.DataFrame, period: int) -> pd.Series:
    """ADX(추세 강도) — Wilder. 낮을수록 횡보, 높을수록 추세."""
    h, low, c = df["high"], df["low"], df["close"]
    up = h.diff()
    dn = -low.diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pc = c.shift(1)
    tr = pd.concat([(h - low), (h - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(
        alpha=1 / period, adjust=False, min_periods=period).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _stoch(df: pd.DataFrame, period: int) -> pd.Series:
    """스토캐스틱 %K — (close-최저)/(최고-최저)×100. 과매수/매도 확인용."""
    ll = df["low"].rolling(period).min()
    hh = df["high"].rolling(period).max()
    return 100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)


def _run(
    df: pd.DataFrame,
    *,
    adx_thr: float = 25.0,
    sl_mult: float = 2.0,
    use_sto: bool = True,
    ttl: int = TTL_BARS,
    entry_mult: float = BB_MULT,
    exit_mode: str = "mid",
    fee_pct: float = 0.0004,
) -> list[tuple[float, int]]:
    """볼린저 평균회귀 시뮬 — net_pnl(비율) + 연도 리스트 반환.

    Args:
        df: 1h OHLCV.
        adx_thr: 횡보 판정 ADX 임계(미만이면 진입 허용). 0 이면 필터 없음.
        sl_mult: SL = entry ∓ ATR×sl_mult (롱/숏).
        use_sto: True 면 스토캐스틱 과매수/매도 확인을 진입 조건에 추가.
        ttl: 보유 최대 봉 수(초과 시 종가 청산).
        entry_mult: 진입 밴드 σ 배수(깊은 이탈일수록 반등 폭 확보). 청산용 밴드는
            BB_MULT(2σ) 고정.
        exit_mode: 익절 타겟 — "mid"(중심선) / "opp"(반대 밴드, 풀스윙→RR↑).
    """
    mid, up, lo = _bbands(df, BB_PERIOD, BB_MULT)         # 청산 타겟용(2σ)
    _, up_e, lo_e = _bbands(df, BB_PERIOD, entry_mult)    # 진입 트리거용
    adx = _adx(df, ADX_PERIOD)
    sto = _stoch(df, STO_PERIOD)
    atr = _atr(df, ATR_PERIOD)
    c = df["close"].values
    h = df["high"].values
    low = df["low"].values
    o = df["open"].values
    midv = mid.values
    upv = up.values
    lov = lo.values
    up_ev = up_e.values
    lo_ev = lo_e.values
    adxv = adx.values
    stov = sto.values
    atrv = atr.values
    years = df.index.year.values

    trades: list[tuple[float, int]] = []
    side: str | None = None
    entry = stop = target = 0.0
    entry_i = 0
    for i in range(1, len(c)):
        if side is not None:
            # #LOOKAHEAD 방지: 직전 봉까지 확정된 target·stop 으로 현재 봉 장중
            # (high/low) 히트 판정. SL·TP 동시 히트면 보수적으로 SL 우선.
            if side == "long":
                hit_sl = low[i] <= stop
                hit_tp = h[i] >= target
            else:
                hit_sl = h[i] >= stop
                hit_tp = low[i] <= target
            exit_raw: float | None = None
            if hit_sl:
                exit_raw = stop
            elif hit_tp:
                exit_raw = target
            elif (i - entry_i) >= ttl:
                exit_raw = c[i]
            if exit_raw is not None:
                slp = slip_pct(h[i], low[i], c[i])
                exit_px = apply_slippage(exit_raw, side, "exit", slp)
                raw = (exit_px - entry) / entry
                if side == "short":
                    raw = -raw
                net, _ = apply_costs(raw, SIZE_PCT, LEVERAGE, fee_pct=fee_pct)
                net -= (i - entry_i) * FUNDING_PER_HOUR * SIZE_PCT * LEVERAGE
                trades.append((net, int(years[i])))
                side = None
            else:
                # 청산 안 됨 → 다음 봉용 target 갱신(현재 봉 확정 밴드/중심선).
                if exit_mode == "opp":
                    nt = upv[i] if side == "long" else lov[i]
                else:
                    nt = midv[i]
                if not np.isnan(nt):
                    target = float(nt)
        if side is None:
            # 진입: 직전 봉(j=i-1) 확정 신호 → 현재 봉 시가 진입(낙관 제거).
            j = i - 1
            if np.isnan(lo_ev[j]) or np.isnan(adxv[j]) or np.isnan(atrv[j]):
                continue
            ranging = (adxv[j] < adx_thr) if adx_thr > 0 else True
            if not ranging:
                continue
            sto_ok_long = (not use_sto) or (not np.isnan(stov[j]) and stov[j] < STO_OS)
            sto_ok_short = (not use_sto) or (not np.isnan(stov[j]) and stov[j] > STO_OB)
            long_sig = c[j] < lo_ev[j] and sto_ok_long
            short_sig = c[j] > up_ev[j] and sto_ok_short
            if long_sig:
                slp = slip_pct(h[i], low[i], c[i])
                entry = apply_slippage(o[i], "long", "entry", slp)
                side = "long"
                stop = entry - sl_mult * atrv[j]
                target = float(upv[j] if exit_mode == "opp" else midv[j])
                entry_i = i
            elif short_sig:
                slp = slip_pct(h[i], low[i], c[i])
                entry = apply_slippage(o[i], "short", "entry", slp)
                side = "short"
                stop = entry + sl_mult * atrv[j]
                target = float(lov[j] if exit_mode == "opp" else midv[j])
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
    lines = [
        "===== 볼린저 평균회귀 백테 (1h, 7페어, 시드1000, 20x, size0.9) =====",
        "BB(20,2σ) 하단/상단 이탈 + STO(14) 과매수매도, 청산=중심선 / SL=ATR×n / ttl48",
        "",
        "--- 그리드: ADX 횡보필터 × SL폭 (STO 확인 ON) ---",
        f"{'ADX<':>5} {'SL':>5} {'USDT':>8} {'최대DD':>8} {'승률':>6} {'RR':>5} "
        f"{'연속손절':>8} {'거래':>6}",
    ]
    # 페어별 1h 데이터 1회 로드 캐시(그리드 반복 재사용).
    data = {}
    for sym in PAIRS:
        try:
            d = _load_1h(sym)
            if len(d) >= 200:
                data[sym] = d
        except Exception as e:  # noqa: BLE001
            lines.append(f"  (로드 실패 {sym}: {e})")

    def _agg(**kw: object) -> dict[str, float]:
        agg = {"net": 0.0, "mdd": 0.0, "n": 0, "wr": 0.0, "rr": 0.0, "streak": 0.0}
        nz = 0
        yearly: dict[int, float] = {}
        for d in data.values():
            tr = _run(d, **kw)  # type: ignore[arg-type]
            s = _stats([t[0] for t in tr])
            for k in ("net", "mdd", "wr", "rr", "streak"):
                agg[k] += s[k]
            agg["n"] += int(s["n"])
            nz += 1
            for net, yr in tr:
                yearly[yr] = yearly.get(yr, 0.0) + net
        agg["_nz"] = max(nz, 1)
        agg["_yearly"] = yearly  # type: ignore[assignment]
        return agg

    best = None
    for adx_thr in (0.0, 20.0, 25.0):
        for sl_mult in (1.5, 2.0, 3.0):
            a = _agg(adx_thr=adx_thr, sl_mult=sl_mult, use_sto=True)
            nz = a["_nz"]
            lines.append(
                f"{adx_thr:5.0f} {sl_mult:5.1f} {a['net'] * SEED:+8.0f} "
                f"{a['mdd'] * SEED:7.0f}↓ {a['wr'] / nz:5.0f}% {a['rr'] / nz:5.2f} "
                f"{a['streak'] / nz:7.1f}회 {a['n']:6d}")
            if best is None or a["net"] > best[0]:
                best = (a["net"], adx_thr, sl_mult)

    b_adx = best[1] if best else 20.0
    b_sl = best[2] if best else 2.0

    # 실험: 익절 반대밴드(풀스윙, RR↑) × 진입 깊이(entry σ). 평균회귀 RR<1 극복 시도.
    lines.append("")
    lines.append(f"--- 익절모드 × 진입깊이 (ADX<{b_adx:.0f}, SL×{b_sl:.1f}) ---")
    lines.append(
        f"{'익절':>5} {'진입σ':>5} {'USDT':>8} {'승률':>6} {'RR':>5} "
        f"{'연속손절':>8} {'거래':>6}")
    for exit_mode in ("mid", "opp"):
        for entry_mult in (2.0, 2.5, 3.0):
            a = _agg(adx_thr=b_adx, sl_mult=b_sl, use_sto=True,
                     entry_mult=entry_mult, exit_mode=exit_mode)
            nz = a["_nz"]
            lines.append(
                f"{exit_mode:>5} {entry_mult:5.1f} {a['net'] * SEED:+8.0f} "
                f"{a['wr'] / nz:5.0f}% {a['rr'] / nz:5.2f} "
                f"{a['streak'] / nz:7.1f}회 {a['n']:6d}")

    # STO 확인 ON vs OFF (기본 mid 익절).
    lines.append("")
    lines.append(f"--- STO 확인 ON vs OFF (ADX<{b_adx:.0f}, SL×{b_sl:.1f}, 중심선익절) ---")
    for use_sto in (True, False):
        a = _agg(adx_thr=b_adx, sl_mult=b_sl, use_sto=use_sto)
        nz = a["_nz"]
        tag = "ON " if use_sto else "OFF"
        lines.append(
            f"  STO {tag}: {a['net'] * SEED:+8.0f} USDT  승률 {a['wr'] / nz:4.0f}%  "
            f"RR {a['rr'] / nz:4.2f}  거래 {a['n']}")

    # 결정적 실험: maker 지정가(수수료 0) 가정 — 평균회귀 엣지가 taker 수수료 탓인지,
    # raw 자체가 무엇인지 판별. 흑자 전환이면 '지정가 진입이면 가능', 여전히 적자면
    # raw 엣지 자체가 없음(시장 추세성). 슬리피지는 보수적으로 유지(미체결 리스크 대용).
    lines.append("")
    lines.append("--- 수수료 0(maker 지정가 가정) — raw 엣지 판별 ---")
    lines.append(f"{'익절':>5} {'진입σ':>5} {'USDT':>9} {'승률':>6} {'RR':>5} {'거래':>6}")
    for exit_mode in ("mid", "opp"):
        for entry_mult in (2.0, 2.5, 3.0):
            a = _agg(adx_thr=b_adx, sl_mult=b_sl, use_sto=True, entry_mult=entry_mult,
                     exit_mode=exit_mode, fee_pct=0.0)
            nz = a["_nz"]
            lines.append(
                f"{exit_mode:>5} {entry_mult:5.1f} {a['net'] * SEED:+9.0f} "
                f"{a['wr'] / nz:5.0f}% {a['rr'] / nz:5.2f} {a['n']:6d}")

    txt = "\n".join(lines)
    with open("bb_meanrev_bt_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE")
    except UnicodeEncodeError:
        print("(결과는 bb_meanrev_bt_result.txt)\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
