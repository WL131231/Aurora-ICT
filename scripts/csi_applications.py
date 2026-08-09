"""#AUTONOMOUS 2026-07-29: CSI(횡보 상태 지수) 응용 4갈래 백테 (파트너 지시).

CSI = 사후 인식 복합 모델(리프트 1.6배, 검증구간 확인). 예측이 아닌 **현재 상태**.
파트너 지시 4갈래:
  [1] 사후대응A — CSI 높으면 Origo 사이즈 축소(진입은 유지, 리스크만 축소)
  [2] 사후대응B — CSI + 연속손실 결합(상태 인식 + 실제 손실 사실 → 오탐 감소)
  [3] 사후대응C — CSI 높으면 **보유 중 관리** 변경(TP 단축 = 조기 익절)
  [4] 볼밴 재도전 — "예측이 아닌 인식" 으로 진입 조건 교체. 기존 볼밴 실패가 예측
      의존 탓인지 규명(파트너 가설). CSI>임계 상태에서만 E3/E7 복귀 진입.
기준선: Origo BTC 라이브게이트 base / 볼밴은 무조건 진입.
판정: net>0 + 양반기 + 연도 다수 → ★. 승자는 이후 배터리.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import btc_bb_matrix as BBM  # noqa: E402
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from chop_gate_bakeoff import BASE  # noqa: E402
from chop_state_index import csi_series, fit_csi, load_1h  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]


def stat(tr, min_n: int = 25) -> tuple[str, bool]:
    if len(tr) < min_n:
        return f"n={len(tr):4d} (표본부족)", False
    tr = sorted(tr)
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
    ypos = sum(1 for v in ys.values() if v > 0)
    ok = net > 0 and h1 > 0 and h2 > 0 and ypos >= len(ys) - 1
    yearly = " ".join(f"{y}:{v:+.0f}" for y, v in sorted(ys.items()))
    return (f"n={len(tr):4d} net={net:+7.1f}% 승률={100 * w / len(tr):3.0f}% "
            f"H1={h1:+6.1f} H2={h2:+6.1f} MDD={mdd:5.1f} [{yearly}]"), ok


def origo_trades(sym: str):
    """라이브게이트 Origo 거래 + 진입 idx/시각/방향/진입가/원 net."""
    df5 = _resample(_load_full(sym))
    cfg = BacktestConfig(**BASE)
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    trs = [t for t in bt.trades if not (17 <= df5.index[t.entry_idx].hour < 21)]
    mags = [abs(t.entry_trend_pct) for t in trs]
    q70 = np.percentile(mags, 70) if mags else 0.0
    kept = []
    for t in trs:
        sgn = 1.0 if t.direction == "long" else -1.0
        if abs(t.entry_trend_pct) < q70 and t.entry_trend_pct * sgn < 0:
            continue
        kept.append(t)
    return df5, kept


def main() -> int:
    model = fit_csi(PAIRS)
    csi_by_sym = {}
    for sym in PAIRS:
        csi_by_sym[sym] = csi_series(load_1h(sym), model)

    # ---------- [1] CSI 사이즈 축소 ----------
    print("\n########## [1] CSI 높으면 사이즈 축소 (Origo 7페어) ##########", flush=True)
    allbase = []
    per_trade = []  # (ts, net, csi)
    for sym in PAIRS:
        df5, kept = origo_trades(sym)
        csi5 = csi_by_sym[sym].reindex(df5.index, method="ffill")
        for t in kept:
            ts = df5.index[t.entry_idx]
            c = csi5.iloc[t.entry_idx]
            allbase.append((ts, t.net_pnl_pct))
            per_trade.append((ts, t.net_pnl_pct, float(c) if not pd.isna(c) else np.nan))
    line, _ = stat(allbase)
    print(f"  base(변조없음)          {line}", flush=True)
    for thr in (0.5, 0.6, 0.7):
        for scale in (0.5, 0.25, 0.0):
            adj = [(ts, n * (scale if (not np.isnan(c)) and c >= thr else 1.0))
                   for ts, n, c in per_trade]
            line, ok = stat(adj)
            print(f"  {'★' if ok else ' '}CSI>={thr} → ×{scale:<4}   {line}", flush=True)

    # ---------- [2] CSI + 연속손실 ----------
    print("\n########## [2] CSI + 연속손실 결합 ##########", flush=True)
    for thr in (0.5, 0.6):
        for nloss in (2, 3):
            for scale in (0.5, 0.0):
                out = []
                streak = 0
                for ts, n, c in sorted(per_trade):
                    hot = (not np.isnan(c)) and c >= thr and streak >= nloss
                    out.append((ts, n * (scale if hot else 1.0)))
                    streak = streak + 1 if n < 0 else 0
                line, ok = stat(out)
                print(f"  {'★' if ok else ' '}CSI>={thr} & {nloss}연패 → ×{scale:<4} {line}",
                      flush=True)

    # ---------- [3] CSI 보유 중 관리(TP 단축) ----------
    print("\n########## [3] CSI 높으면 보유 중 TP 단축 ##########", flush=True)
    for thr in (0.5, 0.6):
        for k in (1.0, 1.5, 2.0):
            out = []
            for sym in PAIRS:
                df5, kept = origo_trades(sym)
                c5 = df5["close"].to_numpy(); h5 = df5["high"].to_numpy()
                lo5 = df5["low"].to_numpy()
                n5 = len(c5)
                tr14 = np.concatenate([[h5[0] - lo5[0]], np.maximum(
                    h5[1:] - lo5[1:], np.maximum(np.abs(h5[1:] - c5[:-1]),
                                                 np.abs(lo5[1:] - c5[:-1])))])
                atr = pd.Series(tr14).rolling(14).mean().to_numpy()
                csi5 = csi_by_sym[sym].reindex(df5.index, method="ffill").to_numpy()
                for t in kept:
                    i = t.entry_idx
                    ts = df5.index[i]
                    cv = csi5[i]
                    if np.isnan(cv) or cv < thr or np.isnan(atr[i]):
                        out.append((ts, t.net_pnl_pct))   # 평소 로직 그대로
                        continue
                    d = 1 if t.direction == "long" else -1
                    entry = t.entry
                    sl = t.entry - d * abs(t.entry - t.stop_loss) if hasattr(t, "stop_loss") else None
                    tp = entry + d * k * atr[i]
                    slp = entry - d * 2.0 * atr[i]        # 손절은 원 위험폭 근사(2×ATR)
                    netv = 0.0
                    for j in range(i + 1, min(i + 289, n5)):
                        if d == 1:
                            if lo5[j] <= slp:
                                netv = (slp - entry) / entry; break
                            if h5[j] >= tp:
                                netv = (tp - entry) / entry; break
                        else:
                            if h5[j] >= slp:
                                netv = (entry - slp) / entry; break
                            if lo5[j] <= tp:
                                netv = (entry - tp) / entry; break
                    out.append((ts, (netv - 0.0008) * 100 * 18))  # 레버 근사(시드% 정합)
            line, ok = stat(out)
            print(f"  {'★' if ok else ' '}CSI>={thr} → TP {k}×ATR  {line}", flush=True)

    # ---------- [4] 볼밴 × CSI(인식) ----------
    print("\n########## [4] 볼린저 × CSI 상태 인식 (BTC 1h) ##########", flush=True)
    BBM.SYM = "BTCUSDT"
    df, d = BBM.prep("1h")
    idx = df.index
    csi = csi_by_sym["BTCUSDT"].reindex(idx).to_numpy()
    for rule in ("E3", "E7"):
        sig_raw = BBM.signals(d, rule)
        for thr in (0.5, 0.6, 0.7):
            m = (~np.isnan(csi)) & (csi >= thr)
            sig = np.where(m, sig_raw, 0.0)
            for tp in ("1R", "2R"):
                for sl in ("atr1", "atr15"):
                    tr = BBM.run(d, idx, sig, tp, sl, "all", 0.0008)
                    line, ok = stat(tr)
                    tag = f"{rule} CSI>={thr} tp={tp} sl={sl} maker"
                    print(f"  {'★' if ok else ' '}{tag:<36} {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
