"""#AUTONOMOUS 2026-07-29: CSI 멀티TF — 1H 정확도 + 5M 순발력 (파트너 지시).

파트너: "1시간봉으로 정확도를, 5분봉으로 순발력을". 현행 CSI 는 1h 전용이라 최대
1시간 지연이 있다. 5m 재료를 더해 **상태 변화를 빨리 감지**하면서 1h 의 안정성을 유지.

구성:
  CSI_1h   : 기존(1h 재료 8종) — 안정적이나 최대 1h 지연
  CSI_5m   : 동일 재료를 5m 로 계산(ADX·CHOP·BBW·기울기·볼륨비·시각·요일 + 1h ADX)
             → 5분마다 갱신(순발력), 노이즈 큼
  CSI_MTF  : 두 확률의 결합 — 아래 4가지 방식 비교
      M1 평균        : (csi1h + csi5m)/2
      M2 가중        : 0.7×csi1h + 0.3×csi5m (1h 우선)
      M3 AND        : 둘 다 임계 초과 시만 (정밀도 우선)
      M4 1h확인+5m전환: 1h 가 임계 초과 상태에서 5m 이 먼저 이탈하면 즉시 해제
                        (진입은 신중, 이탈은 빠르게 — 순발력의 실질 의미)
평가: 사후 인식 정답(5m 기준 과거 144봉=12h 실현추세)에 대한 정밀도·리프트 +
      **상태 전환 지연 측정**(1h 대비 몇 분 빨리 감지하는가).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from chop_gate_bakeoff import adx14, chop14  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
CHOP_THR = 0.30
H_5M = 144        # 5m×144 = 12시간 (1h 지평 12봉과 동일 시간)


def feats(df: pd.DataFrame, adx_hi: pd.Series | None = None) -> pd.DataFrame:
    c = df["close"].to_numpy(); h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    v = df["volume"].to_numpy() if "volume" in df else np.ones(len(c))
    s = pd.Series(c)
    ma = s.rolling(20).mean().to_numpy(); sd = s.rolling(20).std().to_numpy()
    bbw = (4.0 * sd) / np.maximum(ma, 1e-12)
    out = pd.DataFrame(index=df.index)
    out["adx"] = adx14(h, lo, c)
    out["chop"] = chop14(h, lo, c)
    out["bbw"] = bbw
    out["bbw_slope"] = pd.Series(bbw).pct_change(5).to_numpy()
    out["volr"] = v / np.maximum(pd.Series(v).rolling(20).mean().to_numpy(), 1e-12)
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek
    out["adx_hi"] = (adx_hi.reindex(df.index, method="ffill").to_numpy()
                     if adx_hi is not None else out["adx"].to_numpy())
    return out


F = ["adx", "chop", "bbw", "bbw_slope", "volr", "hour", "dow", "adx_hi"]


def logistic_fit(X, y, iters=600, lr=0.5):
    mu, sg = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sg
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        g = p - y
        w -= lr * (Z.T @ g) / len(y)
        b -= lr * g.mean()
    return w, b, mu, sg


def apply_model(m, X):
    w, b, mu, sg = m
    return 1 / (1 + np.exp(-np.clip(((X - mu) / sg) @ w + b, -30, 30)))


def build_pair(sym: str):
    df5 = _resample(_load_full(sym))
    df1 = df5.resample("1h").agg({"open": "first", "high": "max", "low": "min",
                                  "close": "last", "volume": "sum"}).dropna()
    a1 = pd.Series(adx14(df1["high"].to_numpy(), df1["low"].to_numpy(),
                         df1["close"].to_numpy()), index=df1.index).shift(1)
    d4 = df1.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    a4 = pd.Series(adx14(d4["high"].to_numpy(), d4["low"].to_numpy(),
                         d4["close"].to_numpy()), index=d4.index).shift(1)
    f1 = feats(df1, a4)
    f5 = feats(df5, a1)      # 5m 재료 + 상위(1h) ADX
    # 정답 — 5m 격자에서 과거 12시간 실현추세
    c5 = df5["close"].to_numpy(); h5 = df5["high"].to_numpy(); l5 = df5["low"].to_numpy()
    hi = pd.Series(h5).rolling(H_5M).max().to_numpy()
    lo_ = pd.Series(l5).rolling(H_5M).min().to_numpy()
    prev = pd.Series(c5).shift(H_5M).to_numpy()
    y5 = pd.Series((np.abs(c5 - prev) / np.maximum(hi - lo_, 1e-12)) < CHOP_THR,
                   index=df5.index)
    return df5, f1.shift(1), f5, y5   # f1 은 완결 1h 만 사용(인과)


def main() -> int:
    tr1X, tr1y, tr5X, tr5y = [], [], [], []
    store = {}
    for sym in PAIRS:
        df5, f1, f5, y5 = build_pair(sym)
        # 1h 학습 라벨 — 1h 격자에서 과거 12봉
        d1 = df5.resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        c1 = d1["close"].to_numpy()
        hi1 = pd.Series(d1["high"].to_numpy()).rolling(12).max().to_numpy()
        lo1 = pd.Series(d1["low"].to_numpy()).rolling(12).min().to_numpy()
        pv1 = pd.Series(c1).shift(12).to_numpy()
        y1 = pd.Series((np.abs(c1 - pv1) / np.maximum(hi1 - lo1, 1e-12)) < CHOP_THR,
                       index=d1.index)
        m1 = pd.concat([f1, y1.rename("y")], axis=1).dropna()
        m5 = pd.concat([f5, y5.rename("y")], axis=1).dropna()
        cut1 = int(len(m1) * 0.7); cut5 = int(len(m5) * 0.7)
        tr1X.append(m1[F].to_numpy()[:cut1]); tr1y.append(m1["y"].to_numpy()[:cut1].astype(int))
        tr5X.append(m5[F].to_numpy()[:cut5]); tr5y.append(m5["y"].to_numpy()[:cut5].astype(int))
        store[sym] = (m1, m5, cut1, cut5, df5)
    M1 = logistic_fit(np.vstack(tr1X), np.concatenate(tr1y))
    M5 = logistic_fit(np.vstack(tr5X), np.concatenate(tr5y))
    print("모델 학습 완료 (1h / 5m)", flush=True)

    # 검증 — 5m 격자 통합 평가
    rows = []
    for sym in PAIRS:
        m1, m5, cut1, cut5, df5 = store[sym]
        p1 = pd.Series(apply_model(M1, m1[F].to_numpy()), index=m1.index)
        p5 = pd.Series(apply_model(M5, m5[F].to_numpy()), index=m5.index)
        te5 = m5.iloc[cut5:]
        p1_on5 = p1.reindex(te5.index, method="ffill")
        d = pd.DataFrame({"p1": p1_on5, "p5": p5.reindex(te5.index),
                          "y": te5["y"].astype(bool)}).dropna()
        rows.append(d)
    D = pd.concat(rows)
    base = D.y.mean()
    print(f"\n검증 표본 {len(D):,} · 기저율 {100 * base:.1f}%", flush=True)
    print(f"  {'방식':<22} {'빈도':>6} {'정밀도':>7} {'리프트':>6}", flush=True)

    def ev(mask, label):
        mask = mask.to_numpy() if hasattr(mask, "to_numpy") else mask
        if mask.sum() < 500:
            print(f"  {label:<22} 표본부족", flush=True)
            return
        prec = D.y.to_numpy()[mask].mean()
        print(f"  {label:<22} {100 * mask.mean():5.1f}% {100 * prec:6.1f}% {prec / base:5.2f}x",
              flush=True)

    for q in (0.90, 0.80):
        t1 = D.p1.quantile(q); t5 = D.p5.quantile(q)
        ev(D.p1 >= t1, f"1h 단독 상위{100 - int(q * 100)}%")
        ev(D.p5 >= t5, f"5m 단독 상위{100 - int(q * 100)}%")
        mtf_avg = (D.p1 + D.p5) / 2
        ev(mtf_avg >= mtf_avg.quantile(q), f"M1 평균 상위{100 - int(q * 100)}%")
        mtf_w = 0.7 * D.p1 + 0.3 * D.p5
        ev(mtf_w >= mtf_w.quantile(q), f"M2 가중7:3 상위{100 - int(q * 100)}%")
        ev((D.p1 >= t1) & (D.p5 >= t5), f"M3 AND 상위{100 - int(q * 100)}%")

    # 상태 전환 지연 — 1h 판정이 켜지는 시점 vs 5m/MTF 가 먼저 켜지는 시점
    print("\n[전환 순발력] 횡보 상태 진입/이탈 감지 시차(분)", flush=True)
    lags_in, lags_out = [], []
    for sym in PAIRS:
        m1, m5, cut1, cut5, df5 = store[sym]
        p1 = pd.Series(apply_model(M1, m1[F].to_numpy()), index=m1.index)
        p5 = pd.Series(apply_model(M5, m5[F].to_numpy()), index=m5.index)
        te = m5.iloc[cut5:].index
        s1 = (p1.reindex(te, method="ffill") >= 0.6).astype(int)
        s5 = (p5.reindex(te) >= 0.6).astype(int)
        # 1h 가 0→1 되는 지점마다, 그 직전 5m 가 언제 1 이 됐는지
        ups = np.flatnonzero((s1.diff() == 1).to_numpy())
        for u in ups[:2000]:
            back = s5.to_numpy()[max(0, u - 24):u + 1]
            if back.size and back[0] == 0 and back.max() == 1:
                first = int(np.argmax(back == 1))
                lags_in.append((len(back) - 1 - first) * 5)
        downs = np.flatnonzero((s1.diff() == -1).to_numpy())
        for dn in downs[:2000]:
            back = s5.to_numpy()[max(0, dn - 24):dn + 1]
            if back.size and back[0] == 1 and back.min() == 0:
                first = int(np.argmax(back == 0))
                lags_out.append((len(back) - 1 - first) * 5)
    if lags_in:
        print(f"  진입 감지: 5m 이 평균 {np.mean(lags_in):.0f}분 먼저 (중앙값 {np.median(lags_in):.0f}분, n={len(lags_in)})",
              flush=True)
    if lags_out:
        print(f"  이탈 감지: 5m 이 평균 {np.mean(lags_out):.0f}분 먼저 (중앙값 {np.median(lags_out):.0f}분, n={len(lags_out)})",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
