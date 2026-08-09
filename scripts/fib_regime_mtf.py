"""#AUTONOMOUS 2026-07-29: 임펄스 — 멀티TF(15m·1h) × **시장 국면 분해** (파트너 지적 반영).

파트너 지적 2가지:
 ① 임펄스는 1h 에서만 나오는 게 아니다. 우리는 단타 로직 → **15m·1h 둘 다** 봐야.
 ② 5년치면 상승만 있는 게 아니다 → 그런데 앞선 배터리에서 **숏 -284%** 였다.
    2022 대폭락·2026 하락이 분명히 있었는데 숏이 못 번 것 = 검증의 구멍.

핵심 질문: **하락장에서 숏이 버는가?**
  상승장 롱이 버는 건 크립토 상승편향으로 설명된다(플라시보도 흑자였다).
  하락장 숏이 벌어야 방향 대칭 엣지 = 진짜다.

국면 라벨(BTC 기준, 전 페어 공통 — BTC 가 시장 대표):
  BTC 1h 종가의 30일(720봉) 변화율
    >= +15% : 상승장   /   <= -15% : 하락장   /   그 사이 : 횡보장
출력: TF × 국면 × 방향(롱/숏) 매트릭스 + 각 칸의 플라시보 대조.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from fib_trend_impulse import candle_impulses  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
NOTIONAL = 18.0
COST = 0.0011
REGIME_THR = 0.15


def load_tf(sym: str, tf: str):
    df = _resample(_load_full(sym)).resample(tf).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    c = df["close"].to_numpy(); h = df["high"].to_numpy()
    lo = df["low"].to_numpy(); o = df["open"].to_numpy()
    tr = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    return df, o, c, h, lo, atr


def btc_regime(tf: str) -> pd.Series:
    """BTC 30일 변화율로 시장 국면 — 전 페어 공통 적용. 인과(직전 완결봉)."""
    df = _resample(_load_full("BTCUSDT")).resample("1h").agg({"close": "last"}).dropna()
    c = df["close"]
    chg = c.pct_change(720).shift(1)      # 30일 = 720시간, shift 로 미래참조 차단
    reg = pd.Series(np.where(chg >= REGIME_THR, "상승",
                             np.where(chg <= -REGIME_THR, "하락", "횡보")), index=df.index)
    return reg


def ema_ctx(c):
    s = pd.Series(c)
    e200 = s.ewm(span=200, adjust=False).mean().shift(1).to_numpy()
    cp = np.concatenate([[np.nan], c[:-1]])
    return np.where(cp > e200, 1, np.where(cp < e200, -1, 0))


def run(data, reg_map, tf, win, k, ext=3.0, hold=240, placebo_rng=None):
    """반환: [(ts, pnl, sym, d, regime)]"""
    out = []
    for sym, (df, o, c, h, lo, atr) in data.items():
        n = len(c)
        ctx = ema_ctx(c)
        reg = reg_map.reindex(df.index, method="ffill").to_numpy()
        imps = candle_impulses(o, c, h, lo, atr, win, k, 0.45, 0.6, 0.25, win * 2)
        if placebo_rng is not None:
            legs = [(abs(e - s) / max(c[i], 1e-12), d) for (i, s, e, d) in imps]
            lowv, highv = win + 210, n - hold - 2
            if highv <= lowv or not legs:
                continue
            picks = sorted(placebo_rng.choice(np.arange(lowv, highv),
                                              size=min(len(legs), highv - lowv), replace=False))
            imps = [(idx, c[idx] - d * legr * c[idx], c[idx], d)
                    for idx, (legr, d) in zip(picks, legs)]
        busy = -1
        for (i, start, end, d) in imps:
            if i <= busy:
                continue
            if placebo_rng is None and ctx[i] != d:
                continue
            leg = abs(end - start)
            if leg <= 0:
                continue
            fill = i + 1
            if fill >= n - 1:
                continue
            entry = o[fill]; sl = start
            risk = abs(entry - sl)
            if risk <= 0 or risk / entry < 0.002:
                continue
            tp = start + d * ext * leg
            if (d == 1 and tp <= entry) or (d == -1 and tp >= entry):
                continue
            raw = 0.0
            exit_j = min(fill + hold, n - 1)
            for j in range(fill + 1, exit_j + 1):
                if d == 1:
                    if lo[j] <= sl:
                        raw = (sl - entry) / entry; exit_j = j; break
                    if h[j] >= tp:
                        raw = (tp - entry) / entry; exit_j = j; break
                else:
                    if h[j] >= sl:
                        raw = (entry - sl) / entry; exit_j = j; break
                    if lo[j] <= tp:
                        raw = (entry - tp) / entry; exit_j = j; break
            else:
                raw = ((c[exit_j] - entry) / entry) * d
            rg = reg[fill] if fill < len(reg) else "횡보"
            out.append((df.index[fill], (raw - COST) * NOTIONAL * 100, sym, d,
                        rg if isinstance(rg, str) else "횡보"))
            busy = exit_j
    return out


def matrix(trades, label):
    print(f"\n----- {label} -----", flush=True)
    print(f"  {'국면':<6} {'방향':<5} {'n':>6} {'net':>10} {'승률':>6} {'평균/건':>9}", flush=True)
    tot = {}
    for rg in ("상승", "횡보", "하락"):
        for d, dn in ((1, "롱"), (-1, "숏")):
            sub = [p for _, p, _, dd, r in trades if dd == d and r == rg]
            if len(sub) < 20:
                print(f"  {rg:<6} {dn:<5} {len(sub):6d}  표본부족", flush=True)
                continue
            net = sum(sub)
            wr = 100 * sum(1 for x in sub if x > 0) / len(sub)
            print(f"  {rg:<6} {dn:<5} {len(sub):6d} {net:+10.1f} {wr:5.0f}% {net / len(sub):+8.2f}",
                  flush=True)
            tot[(rg, dn)] = net
    return tot


def main() -> int:
    print("국면 라벨: BTC 30일 변화율 ±15% (인과)", flush=True)
    # 주의: pandas 에서 "15m" 은 15**개월**. 분은 "15min" 이다(7/29 버그 — 15m 이 5봉이었다).
    for tf, win, k in (("15min", 12, 3.0), ("1h", 12, 3.0)):
        print(f"\n\n================ TF {tf} (win{win} ATR×{k}) ================", flush=True)
        data = {s: load_tf(s, tf) for s in PAIRS}
        reg = btc_regime(tf)
        tr = run(data, reg, tf, win, k)
        print(f"총 {len(tr)}건", flush=True)
        matrix(tr, f"{tf} 진짜 임펄스")
        pl = run(data, reg, tf, win, k, placebo_rng=np.random.default_rng(0))
        matrix(pl, f"{tf} 플라시보(무작위 시점)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
