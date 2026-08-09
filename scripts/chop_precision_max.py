"""#AUTONOMOUS 2026-07-29: 횡보 인식 정밀도 극대화 (파트너: "최대한 높게, 정밀하게").

현재: ADX<20 단독 51.5%, 복합 로지스틱 54%(리프트 1.64x). 이걸 끌어올린다.

개선 축:
  [A] 재료 확장 — 기존 8개 + 신규:
      · 실현변동성 비(단기/장기 표준편차 — 변동성 압축의 다른 표현)
      · 방향 전환 빈도(부호 바뀜 횟수 / N봉 — 톱질의 직접 측정)
      · 고저폭 대비 종가 위치 분산
      · 연속 도지형 비율(|종가-시가|/고저폭 작은 봉)
      · 상위TF(4h) BBW·CHOP
      · 최근 스윙 고저 갱신 실패 횟수(구조적 정체)
  [B] 지평 최적화 — 인식 지평 H=12/24/48 중 어디서 가장 정밀한가
  [C] 임계 상향 — 확신 상위 5%/10%만 취할 때 정밀도(적게 판정하되 정확하게)
  [D] 비선형 — 재료 구간화(binning) + 상호작용 항 추가한 확장 로지스틱
  [E] 앙상블 — 여러 지평 모델의 합의(3개 중 3개 동의 시만 판정)
평가: 검증구간(뒤 30%) 정밀도·리프트·판정빈도. 목표 = 정밀도 65%+ (리프트 1.9x+).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from chop_gate_bakeoff import adx14, chop14, roll_q  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
CHOP_THR = 0.30


def build(sym: str, H: int) -> pd.DataFrame:
    df = _resample(_load_full(sym)).resample("1h").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()
    o = df["open"].to_numpy(); c = df["close"].to_numpy()
    h = df["high"].to_numpy(); lo = df["low"].to_numpy(); v = df["volume"].to_numpy()
    s = pd.Series(c)
    out = pd.DataFrame(index=df.index)
    # 기존 재료
    out["adx"] = adx14(h, lo, c)
    out["chop"] = chop14(h, lo, c)
    ma = s.rolling(20).mean().to_numpy(); sd = s.rolling(20).std().to_numpy()
    bbw = (4.0 * sd) / np.maximum(ma, 1e-12)
    out["bbw"] = bbw
    out["bbw_slope"] = pd.Series(bbw).pct_change(5).to_numpy()
    out["volr"] = v / np.maximum(pd.Series(v).rolling(20).mean().to_numpy(), 1e-12)
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek
    d4 = df.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    out["adx4h"] = pd.Series(adx14(d4["high"].to_numpy(), d4["low"].to_numpy(),
                                   d4["close"].to_numpy()), index=d4.index).shift(1)\
        .reindex(df.index, method="ffill").to_numpy()
    # [A] 신규 재료
    r = s.pct_change()
    out["vol_ratio"] = (r.rolling(6).std() / (r.rolling(48).std() + 1e-12)).to_numpy()
    sign = np.sign(r.fillna(0).to_numpy())
    flip = (sign[1:] * sign[:-1] < 0).astype(float)
    out["flip_rate"] = np.concatenate([[np.nan], pd.Series(flip).rolling(24).mean().to_numpy()])
    body = np.abs(c - o) / np.maximum(h - lo, 1e-12)
    out["doji_rate"] = pd.Series((body < 0.25).astype(float)).rolling(24).mean().to_numpy()
    clpos = (c - lo) / np.maximum(h - lo, 1e-12)
    out["clpos_std"] = pd.Series(clpos).rolling(24).std().to_numpy()
    sd4 = pd.Series(d4["close"].to_numpy()).rolling(20).std().to_numpy()
    ma4 = pd.Series(d4["close"].to_numpy()).rolling(20).mean().to_numpy()
    out["bbw4h"] = pd.Series((4 * sd4) / np.maximum(ma4, 1e-12), index=d4.index).shift(1)\
        .reindex(df.index, method="ffill").to_numpy()
    out["chop4h"] = pd.Series(chop14(d4["high"].to_numpy(), d4["low"].to_numpy(),
                                     d4["close"].to_numpy()), index=d4.index).shift(1)\
        .reindex(df.index, method="ffill").to_numpy()
    hh = pd.Series(h).rolling(24).max().to_numpy()
    ll = pd.Series(lo).rolling(24).min().to_numpy()
    out["range_pos"] = (c - ll) / np.maximum(hh - ll, 1e-12)
    out["range_pct"] = (hh - ll) / np.maximum(c, 1e-12)
    # 정답 — 과거 H봉 실현추세(사후 인식)
    hi_h = pd.Series(h).rolling(H).max().to_numpy()
    lo_h = pd.Series(lo).rolling(H).min().to_numpy()
    prev = s.shift(H).to_numpy()
    out["y"] = (np.abs(c - prev) / np.maximum(hi_h - lo_h, 1e-12)) < CHOP_THR
    return out.replace([np.inf, -np.inf], np.nan).dropna()


FEAT_BASE = ["adx", "chop", "bbw", "bbw_slope", "volr", "hour", "dow", "adx4h"]
FEAT_NEW = FEAT_BASE + ["vol_ratio", "flip_rate", "doji_rate", "clpos_std",
                        "bbw4h", "chop4h", "range_pos", "range_pct"]


def logistic(Xtr, ytr, iters=800, lr=0.5):
    mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-9
    Z = (Xtr - mu) / sg
    w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(Z @ w + b, -30, 30)))
        g = p - ytr
        w -= lr * (Z.T @ g) / len(ytr)
        b -= lr * g.mean()
    return w, b, mu, sg


def predict(m, X):
    w, b, mu, sg = m
    return 1 / (1 + np.exp(-np.clip(((X - mu) / sg) @ w + b, -30, 30)))


def evaluate(feats: list[str], H: int, label: str, inter: bool = False) -> dict:
    Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
    for sym in PAIRS:
        d = build(sym, H)
        cut = int(len(d) * 0.7)
        X = d[feats].astype(float).to_numpy()
        if inter:  # 상호작용 — adx×bbw, adx×flip 등 핵심 곱항
            extra = np.column_stack([
                d["adx"].to_numpy() * d["bbw"].to_numpy(),
                d["adx"].to_numpy() * d["flip_rate"].to_numpy() if "flip_rate" in d else d["adx"].to_numpy(),
                d["bbw"].to_numpy() * d["vol_ratio"].to_numpy() if "vol_ratio" in d else d["bbw"].to_numpy(),
            ])
            X = np.hstack([X, extra])
        y = d["y"].to_numpy().astype(int)
        Xtr_l.append(X[:cut]); ytr_l.append(y[:cut])
        Xte_l.append(X[cut:]); yte_l.append(y[cut:])
    Xtr = np.vstack(Xtr_l); ytr = np.concatenate(ytr_l)
    Xte = np.vstack(Xte_l); yte = np.concatenate(yte_l)
    m = logistic(Xtr, ytr)
    p = predict(m, Xte)
    base = yte.mean()
    res = dict(label=label, H=H, base=base)
    for q in (0.95, 0.90, 0.80, 0.70):
        thr = np.quantile(p, q)
        sel = p >= thr
        prec = yte[sel].mean()
        res[f"q{int(q * 100)}"] = (100 * sel.mean(), 100 * prec, prec / base)
    return res


def show(r: dict) -> None:
    print(f"\n  {r['label']} (H={r['H']}, 기저율 {100 * r['base']:.1f}%)", flush=True)
    for q in (95, 90, 80, 70):
        f, prec, lift = r[f"q{q}"]
        print(f"    상위{100 - q:2d}%: 빈도{f:5.1f}% 정밀도{prec:5.1f}% 리프트{lift:.2f}x",
              flush=True)


def main() -> int:
    print("=== [기준] 기존 8재료 ===", flush=True)
    for H in (12, 24, 48):
        show(evaluate(FEAT_BASE, H, "기존8"))
    print("\n=== [A] 재료 확장 16개 ===", flush=True)
    for H in (12, 24, 48):
        show(evaluate(FEAT_NEW, H, "확장16"))
    print("\n=== [D] 확장 + 상호작용항 ===", flush=True)
    for H in (24, 48):
        show(evaluate(FEAT_NEW, H, "확장16+교차", inter=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
