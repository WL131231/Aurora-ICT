"""#AUTONOMOUS 2026-07-27: 횡보 게이트 대회 — ADX vs CHOP vs ER vs BBW (1단계).

파트너 승인 플랜: 횡보 대응 1단계 = "어느 지표가 횡보를 가장 잘 잡나" 정면 비교.
Origo 5년 라이브게이트 거래(NY_PM 제외 + cond_align)에 각 게이트를 skip 필터로
걸어 — "횡보 손실을 얼마나 걸러내고 추세 수익을 얼마나 안 죽이나" 측정.

게이트(전부 1h 지표, 직전 완결봉만 사용 — 인과):
  - ADX(14) < 20 / < 25          : 고전 추세강도. 낮으면 횡보.
  - CHOP(14) > 61.8 / > 55       : 횡보 측정 전용 지표. 높으면 횡보.
  - ER(24)  < 롤링90일 q33       : Kaufman 효율비(1일 환산). 낮으면 횡보.
  - BBW(20,2σ) < 롤링90일 q33    : 밴드폭 수축 = 저변동 횡보.
판정: skip 된 거래의 net 이 크게 음수(손실 제거)이면서 keep net 이 base 대비
유지·개선 + 연도별 robust 한 게이트가 승자 → 2단계(스윕 페이드)의 횡보 정의로 사용.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

FIXED = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
    trail_trigger=2.0, trail_dist=1.5, partial_tp_rr=1.5, partial_be=True,
)
ROLL_Q = 24 * 90  # 롤링 분위 창 — 1h 봉 90일


def _wilder_smooth(x: np.ndarray, n: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[:n] = np.nan
    if len(x) <= n:
        return out
    out[n] = np.nansum(x[1:n + 1])
    for i in range(n + 1, len(x)):
        out[i] = out[i - 1] - out[i - 1] / n + x[i]
    return out


def adx14(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    """Wilder ADX — 추세강도 (낮으면 횡보)."""
    up = np.concatenate([[0.0], np.diff(h)])
    dn = np.concatenate([[0.0], -np.diff(lo)])
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    atr = _wilder_smooth(tr, n)
    pdi = 100 * _wilder_smooth(plus_dm, n) / (atr + 1e-12)
    mdi = 100 * _wilder_smooth(minus_dm, n) / (atr + 1e-12)
    dx = 100 * np.abs(pdi - mdi) / (pdi + mdi + 1e-12)
    adx = np.full_like(dx, np.nan)
    start = 2 * n
    if len(dx) > start:
        adx[start] = np.nanmean(dx[n:start + 1])
        for i in range(start + 1, len(dx)):
            adx[i] = (adx[i - 1] * (n - 1) + dx[i]) / n
    return adx


def chop14(h: np.ndarray, lo: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    """Choppiness Index — 높으면 횡보(≥61.8), 낮으면 추세(≤38.2)."""
    tr = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    s = pd.Series(tr).rolling(n).sum().to_numpy()
    hh = pd.Series(h).rolling(n).max().to_numpy()
    ll = pd.Series(lo).rolling(n).min().to_numpy()
    rng = np.maximum(hh - ll, 1e-12)
    return 100 * np.log10(np.maximum(s, 1e-12) / rng) / np.log10(n)


def er24(c: np.ndarray, n: int = 24) -> np.ndarray:
    """Kaufman 효율비 (1h×24 = 1일) — 낮으면 횡보."""
    ch = np.abs(pd.Series(c).diff(n).to_numpy())
    vol = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(n).sum().to_numpy()
    return np.clip(ch / (vol + 1e-12), 0, 1)


def bbw20(c: np.ndarray, n: int = 20, k: float = 2.0) -> np.ndarray:
    """볼린저 밴드폭 (상대) — 좁으면 저변동 횡보."""
    s = pd.Series(c)
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return ((2 * k * sd) / (ma + 1e-12)).to_numpy()


def roll_q(x: np.ndarray, win: int, q: float) -> np.ndarray:
    """롤링 분위(인과) — 각 시점의 과거 win 개 기준 q 분위값."""
    return pd.Series(x).rolling(win, min_periods=win // 3).quantile(q).to_numpy()


def collect(sym: str) -> list[dict]:
    """라이브게이트 Origo 거래 + 진입 시점 4개 게이트 값(직전 완결 1h봉)."""
    df5 = _resample(_load_full(sym))
    cfg = BacktestConfig(**BASE)
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    # 1h 지표 — 직전 완결봉만(shift 1) 5m 로 인과 매핑.
    d1h = df5.resample("1h").agg(
        {"high": "max", "low": "min", "close": "last"}).dropna()
    h1, l1, c1 = (d1h[k].to_numpy() for k in ("high", "low", "close"))
    ind = pd.DataFrame({
        "adx": adx14(h1, l1, c1),
        "chop": chop14(h1, l1, c1),
        "er": er24(c1),
        "bbw": bbw20(c1),
    }, index=d1h.index)
    ind["er_q33"] = roll_q(ind["er"].to_numpy(), ROLL_Q, 0.33)
    ind["bbw_q33"] = roll_q(ind["bbw"].to_numpy(), ROLL_Q, 0.33)
    ind = ind.shift(1).reindex(df5.index, method="ffill")
    # 라이브게이트 재현 — NY_PM(17-21 UTC) 제외 + cond_align(q70 약추세 역행 skip).
    mags = [abs(t.entry_trend_pct) for t in bt.trades
            if not (17 <= df5.index[t.entry_idx].hour < 21)]
    q70 = np.percentile(mags, 70) if mags else 0.0
    out = []
    for t in bt.trades:
        hh = df5.index[t.entry_idx].hour
        if 17 <= hh < 21:
            continue
        sgn = 1.0 if t.direction == "long" else -1.0
        if abs(t.entry_trend_pct) < q70 and t.entry_trend_pct * sgn < 0:
            continue
        row = ind.iloc[t.entry_idx]
        if row.isna().any():
            continue
        out.append(dict(
            ts=df5.index[t.entry_idx], net=t.net_pnl_pct, sym=sym,
            adx=row["adx"], chop=row["chop"],
            er=row["er"], er_q33=row["er_q33"],
            bbw=row["bbw"], bbw_q33=row["bbw_q33"],
        ))
    return out


def stat(g: list[dict]) -> str:
    n = len(g)
    if not n:
        return "n=   0"
    net = sum(x["net"] for x in g)
    w = sum(1 for x in g if x["net"] > 0)
    return f"n={n:4d} net={net:+8.1f} 승률={100 * w / n:3.0f}% avg={net / n:+.3f}"


def yearly(g: list[dict]) -> str:
    ys: dict[int, float] = {}
    for x in g:
        ys[x["ts"].year] = ys.get(x["ts"].year, 0.0) + x["net"]
    return " ".join(f"{y}:{v:+.0f}" for y, v in sorted(ys.items()))


GATES = {
    "ADX<20": lambda x: x["adx"] < 20,
    "ADX<25": lambda x: x["adx"] < 25,
    "CHOP>61.8": lambda x: x["chop"] > 61.8,
    "CHOP>55": lambda x: x["chop"] > 55,
    "ER<q33": lambda x: x["er"] < x["er_q33"],
    "BBW<q33": lambda x: x["bbw"] < x["bbw_q33"],
}


def main() -> int:
    allt: list[dict] = []
    for sym in FIXED:
        rows = collect(sym)
        allt += rows
        print(f"{sym}: {stat(rows)}", flush=True)
    print(f"\nbase 전체: {stat(allt)}  연도별[{yearly(allt)}]\n", flush=True)
    print("=== 게이트 대회 — '횡보 판정 시 진입 skip' 적용 결과 ===", flush=True)
    print(f"{'게이트':<10} | skip 된 거래(제거된 net)          | keep 결과                              | keep 연도별")
    for name, fn in GATES.items():
        skipped = [x for x in allt if fn(x)]
        kept = [x for x in allt if not fn(x)]
        print(f"{name:<10} | {stat(skipped):<33} | {stat(kept):<38} | {yearly(kept)}", flush=True)
    print("\n→ 승자 기준: skip net 큰 음수 + keep net 유지·개선 + 연도별 일관", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
