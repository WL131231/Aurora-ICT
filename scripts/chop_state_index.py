"""#AUTONOMOUS 2026-07-29: 횡보 상태 지수(CSI) — 사후 인식 복합 모델 모듈 (파트너 승인).

7/29 측정: 미래 횡보 **예측**은 전 지표 1.00배(불가), 현재 횡보 **인식**은 복합 모델
1.64배(AUC 0.710) 가능. 이 인식 모델을 재사용 지표로 고정한다.

CSI = 로지스틱(현재 상태 재료) → 0~1 확률. 높을수록 "지금 횡보 상태".
재료(전부 직전 완결봉, 인과): ADX14 · CHOP14 · BBW · BBW 5봉 변화율 · 거래량비 ·
  시각(UTC) · 요일 · 4h ADX.
학습: 시계열 앞 70% 만 사용(뒤 30% 는 검증 전용). 페어별 학습 X — 전 페어 통합
계수(과적합 억제, 신규 페어 전이 가능).
사용: fit_csi() 로 계수 학습 → csi_series(df) 로 임의 구간 확률 산출.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from chop_gate_bakeoff import adx14, chop14, roll_q  # noqa: E402

FEATS = ["adx", "chop", "bbw", "bbw_slope", "volr", "hour", "dow", "adx4h"]
H_STATE = 12        # 사후 인식 지평 — 12봉(=12시간). 24→12 로 단축 시
                    # 정밀도 53.7%→61.0% (7/29 정밀도 실험). 짧을수록 현재
                    # 상태가 구간을 잘 대변.
CHOP_THR = 0.30


def features_1h(df1h: pd.DataFrame) -> pd.DataFrame:
    """1h OHLCV → CSI 재료 (인과: 전부 현재봉까지 확정 정보)."""
    c = df1h["close"].to_numpy(); h = df1h["high"].to_numpy()
    lo = df1h["low"].to_numpy()
    v = df1h["volume"].to_numpy() if "volume" in df1h else np.ones(len(c))
    s = pd.Series(c)
    mid = s.rolling(20).mean().to_numpy()
    sd = s.rolling(20).std().to_numpy()
    bbw = (4.0 * sd) / np.maximum(mid, 1e-12)
    out = pd.DataFrame(index=df1h.index)
    out["adx"] = adx14(h, lo, c)
    out["chop"] = chop14(h, lo, c)
    out["bbw"] = bbw
    out["bbw_slope"] = pd.Series(bbw).pct_change(5).to_numpy()
    out["volr"] = v / np.maximum(pd.Series(v).rolling(20).mean().to_numpy(), 1e-12)
    out["hour"] = df1h.index.hour
    out["dow"] = df1h.index.dayofweek
    d4 = df1h.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    a4 = pd.Series(adx14(d4["high"].to_numpy(), d4["low"].to_numpy(),
                         d4["close"].to_numpy()), index=d4.index).shift(1)
    out["adx4h"] = a4.reindex(df1h.index, method="ffill").to_numpy()
    # 사후 정답(학습용) — 과거 H봉 실현추세
    hi = pd.Series(h).rolling(H_STATE).max().to_numpy()
    lo_ = pd.Series(lo).rolling(H_STATE).min().to_numpy()
    prev = pd.Series(c).shift(H_STATE).to_numpy()
    out["now_chop"] = (np.abs(c - prev) / np.maximum(hi - lo_, 1e-12)) < CHOP_THR
    return out


def load_1h(sym: str) -> pd.DataFrame:
    return _resample(_load_full(sym)).resample("1h").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()


def fit_csi(pairs: list[str], train_frac: float = 0.7) -> dict:
    """전 페어 통합 로지스틱 학습 — 시계열 앞 train_frac 만 사용."""
    Xs, ys = [], []
    for sym in pairs:
        f = features_1h(load_1h(sym)).dropna()
        cut = int(len(f) * train_frac)
        f = f.iloc[:cut]
        Xs.append(f[FEATS].astype(float).to_numpy())
        ys.append(f["now_chop"].to_numpy().astype(int))
    X = np.vstack(Xs); y = np.concatenate(ys)
    mu, sg = X.mean(0), X.std(0) + 1e-9
    Xn = (X - mu) / sg
    w = np.zeros(Xn.shape[1]); b = 0.0
    for _ in range(600):
        p = 1 / (1 + np.exp(-np.clip(Xn @ w + b, -30, 30)))
        g = p - y
        w -= 0.5 * (Xn.T @ g) / len(y)
        b -= 0.5 * g.mean()
    return dict(w=w, b=b, mu=mu, sg=sg)


def csi_series(df1h: pd.DataFrame, model: dict) -> pd.Series:
    """1h df → CSI 확률 시계열 (NaN 구간은 NaN)."""
    f = features_1h(df1h)
    X = f[FEATS].astype(float).to_numpy()
    ok = ~np.isnan(X).any(axis=1)
    z = np.full(len(f), np.nan)
    Xn = (X[ok] - model["mu"]) / model["sg"]
    z[ok] = 1 / (1 + np.exp(-np.clip(Xn @ model["w"] + model["b"], -30, 30)))
    return pd.Series(z, index=f.index)


def main() -> int:
    pairs = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
    m = fit_csi(pairs)
    print("계수(표준화):", " ".join(f"{k}={v:+.2f}" for k, v in zip(FEATS, m["w"])), flush=True)
    print(f"절편 {m['b']:+.3f}", flush=True)
    # 검증(뒤 30%)에서 정밀도/리프트 재확인
    for sym in pairs[:3]:
        df = load_1h(sym)
        f = features_1h(df).dropna()
        cut = int(len(f) * 0.7)
        te = f.iloc[cut:]
        csi = csi_series(df, m).reindex(te.index)
        y = te["now_chop"].to_numpy().astype(bool)
        base = y.mean()
        for q in (0.8, 0.7):
            thr = csi.quantile(q)
            sel = (csi >= thr).to_numpy()
            print(f"  {sym} CSI>{thr:.2f}: 빈도{100 * sel.mean():4.1f}% "
                  f"정밀도{100 * y[sel].mean():5.1f}% 리프트{y[sel].mean() / base:.2f}x",
                  flush=True)
    np.savez("csi_model.npz", **m)
    print("→ csi_model.npz 저장", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
