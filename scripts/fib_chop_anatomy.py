"""#AUTONOMOUS 2026-07-29: "횡보장 롱" 해부 — 파트너 가설 검증 + 15m 문턱 수정.

발견: 국면×방향 6칸 중 **횡보장 롱만** 플라시보 대비 뚜렷한 우위(+5593 vs -1947).
파트너 가설: "축적됐다가 **분출**하는 걸 임펄스로 잡는 것 아니냐?"

주의 — '횡보'가 두 연구에서 다른 뜻:
  · 분출탐지(7/29 오전) 의 횡보 = CSI 기준 **12시간** 스케일 (전부 음수로 기각)
  · 이번 매트릭스의 횡보장 = BTC **30일** 변화율 ±15% (장기 추세 부재)
  스케일 60배 차이라 모순이 아닐 수 있다. 직접 재서 확인한다.

해부 항목:
  A 진입 직전 축적도 — 24·72봉 레인지폭 / ATR 비. 국면별 비교(분출이면 횡보장이 좁아야)
  B CSI 교차 — 12시간 인식기가 그 시점을 횡보로 봤는가
  C 청산 사유 분포 — TP / SL / 만료 (진짜 분출이면 TP 비중이 높아야)
  D 연도·페어 분포 — 특정 시기 몰림이면 우연 의심
  E 다중검정 방어 — 횡보장 롱만 플라시보 시드 5개로 재대조
그리고 [15m 수정] 몸통·방향 문턱을 TF 별로 재탐색(1h 기준값이 15m 에선 0건이었다).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from fib_regime_mtf import COST, NOTIONAL, btc_regime, ema_ctx, load_tf  # noqa: E402
from fib_trend_impulse import candle_impulses  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]


def run_detail(data, reg_map, win=12, k=3.0, ext=3.0, hold=240, body=0.45,
               dirt=0.6, placebo_rng=None):
    """상세 기록 — 청산사유·축적도 포함."""
    out = []
    for sym, (df, o, c, h, lo, atr) in data.items():
        n = len(c)
        ctx = ema_ctx(c)
        reg = reg_map.reindex(df.index, method="ffill").to_numpy()
        imps = candle_impulses(o, c, h, lo, atr, win, k, body, dirt, 0.25, win * 2)
        if placebo_rng is not None:
            legs = [(abs(e - s) / max(c[i], 1e-12), d) for (i, s, e, d) in imps]
            a, b = win + 210, n - hold - 2
            if b <= a or not legs:
                continue
            picks = sorted(placebo_rng.choice(np.arange(a, b),
                                              size=min(len(legs), b - a), replace=False))
            imps = [(i2, c[i2] - d * lr * c[i2], c[i2], d) for i2, (lr, d) in zip(picks, legs)]
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
            # 축적도 — 임펄스 **시작 전** 24/72봉 레인지폭 / ATR
            p0 = i - win
            acc24 = acc72 = np.nan
            if p0 - 72 >= 0 and atr[p0] and not np.isnan(atr[p0]) and atr[p0] > 0:
                acc24 = (h[p0 - 24:p0].max() - lo[p0 - 24:p0].min()) / atr[p0]
                acc72 = (h[p0 - 72:p0].max() - lo[p0 - 72:p0].min()) / atr[p0]
            raw = 0.0; why = "만료"
            exit_j = min(fill + hold, n - 1)
            for j in range(fill + 1, exit_j + 1):
                if d == 1:
                    if lo[j] <= sl:
                        raw = (sl - entry) / entry; exit_j = j; why = "SL"; break
                    if h[j] >= tp:
                        raw = (tp - entry) / entry; exit_j = j; why = "TP"; break
                else:
                    if h[j] >= sl:
                        raw = (entry - sl) / entry; exit_j = j; why = "SL"; break
                    if lo[j] <= tp:
                        raw = (entry - tp) / entry; exit_j = j; why = "TP"; break
            else:
                raw = ((c[exit_j] - entry) / entry) * d
            rg = reg[fill] if fill < len(reg) else "횡보"
            out.append(dict(ts=df.index[fill], pnl=(raw - COST) * NOTIONAL * 100, sym=sym,
                            d=d, reg=rg if isinstance(rg, str) else "횡보", why=why,
                            acc24=acc24, acc72=acc72, bars=exit_j - fill))
            busy = exit_j
    return pd.DataFrame(out)


def main() -> int:
    data = {s: load_tf(s, "1h") for s in PAIRS}
    reg = btc_regime("1h")
    D = run_detail(data, reg)
    CL = D[(D.reg == "횡보") & (D.d == 1)]          # 관심 칸
    print(f"===== 횡보장 롱 {len(CL)}건 (net {CL.pnl.sum():+.1f}) =====", flush=True)

    print("\n[A] 진입 직전 축적도 (임펄스 시작 전 레인지폭 / ATR) — 낮을수록 축적", flush=True)
    print(f"  {'국면·방향':<12} {'n':>5} {'24봉':>8} {'72봉':>8}", flush=True)
    for rg in ("상승", "횡보", "하락"):
        for d, dn in ((1, "롱"), (-1, "숏")):
            s = D[(D.reg == rg) & (D.d == d)]
            if len(s) < 20:
                continue
            print(f"  {rg + ' ' + dn:<12} {len(s):5d} {s.acc24.mean():8.2f} {s.acc72.mean():8.2f}",
                  flush=True)

    print("\n[C] 청산 사유 분포", flush=True)
    print(f"  {'국면·방향':<12} {'n':>5} {'TP%':>6} {'SL%':>6} {'만료%':>6} {'평균보유(봉)':>12}",
          flush=True)
    for rg in ("상승", "횡보", "하락"):
        for d, dn in ((1, "롱"), (-1, "숏")):
            s = D[(D.reg == rg) & (D.d == d)]
            if len(s) < 20:
                continue
            n = len(s)
            print(f"  {rg + ' ' + dn:<12} {n:5d} {100 * (s.why == 'TP').mean():5.0f}% "
                  f"{100 * (s.why == 'SL').mean():5.0f}% {100 * (s.why == '만료').mean():5.0f}% "
                  f"{s.bars.mean():11.0f}", flush=True)

    print("\n[D] 횡보장 롱 — 연도별 / 페어별", flush=True)
    print("  연도: " + " ".join(f"{y}:{v:+.0f}({c}건)" for (y, v, c) in
                                [(y, g.pnl.sum(), len(g)) for y, g in CL.groupby(CL.ts.dt.year)]),
          flush=True)
    print("  페어: " + " ".join(f"{s.replace('USDT', '')}:{g.pnl.sum():+.0f}({len(g)})"
                                for s, g in CL.groupby("sym")), flush=True)

    print("\n[E] 횡보장 롱만 — 플라시보 5시드 재대조", flush=True)
    for seed in range(5):
        P = run_detail(data, reg, placebo_rng=np.random.default_rng(seed))
        pc = P[(P.reg == "횡보") & (P.d == 1)]
        print(f"  seed{seed}: n={len(pc):4d} net={pc.pnl.sum():+9.1f} "
              f"승률={100 * (pc.pnl > 0).mean():3.0f}%", flush=True)

    print("\n\n===== [15m 문턱 재탐색] =====", flush=True)
    d15 = {s: load_tf(s, "15min") for s in PAIRS}
    reg15 = btc_regime("15m")
    for body in (0.25, 0.35, 0.45):
        for dirt in (0.5, 0.6):
            for k in (3.0, 4.0):
                E = run_detail(d15, reg15, win=12, k=k, body=body, dirt=dirt, hold=240)
                if E.empty:
                    print(f"  몸통{body} 방향{dirt} ATR×{k}: 0건", flush=True)
                    continue
                cl = E[(E.reg == "횡보") & (E.d == 1)]
                print(f"  몸통{body} 방향{dirt} ATR×{k}: 전체 n={len(E):5d} net={E.pnl.sum():+9.1f} "
                      f"| 횡보롱 n={len(cl):4d} net={cl.pnl.sum():+8.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
