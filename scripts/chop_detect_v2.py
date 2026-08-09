"""#AUTONOMOUS 2026-07-29: 횡보 구분 정면 재도전 — 사후인식 vs 사전예측 (파트너 지시).

7/28 결과: 표준 지표(ADX·스퀴즈·CHOP)의 **미래 횡보 예측력 = 리프트 1.00배(정보 0)**.
파트너: "횡보 구분부터 잡고 가자" → 두 갈래로 분해해 한계를 정확히 규명한다.

[A] 사후 인식 — "지금 횡보 중인가?" (동시점 판정)
    정답 = 현재 시점 **과거** H봉의 실현추세 < 0.30. 지표가 이걸 맞추는지.
    (맞으면: 예측은 못 해도 '진입 중' 인식은 가능 → 사후대응 설계 근거)
[B] 사전 예측 — "앞으로 H봉이 횡보일까?" (미래 판정)
    정답 = 미래 H봉 실현추세 < 0.30. 7/28 재현 + **신규 재료** 투입:
      · 지속성: 현재 사후횡보 여부(횡보는 관성이 있나?)
      · 거래량: vol/20봉평균 하위
      · 시간대: UTC 시간(아시아 세션 = 저변동 통설)
      · 요일: 주말(토·일)
      · 상위TF: 4h ADX·BBW
      · 변동성 압축 추세: BBW 5봉 변화율(수축 진행 중)
      · 복합: 로지스틱 회귀(전 재료, 시계열 분할 학습/검증 — 과적합 방지)
측정: 정밀도·리프트·AUC. 리프트>1.10 이면 실용 후보.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from chop_gate_bakeoff import adx14, chop14, roll_q  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
TF = "1h"
H = 24          # 예측/인식 지평 (1h×24 = 1일)
CHOP_THR = 0.30


def realized_trend(h, lo, c, H, future=True):
    """구간 실현 추세도 = |순변화| / 고저폭. future=False 면 과거 H봉."""
    s_h, s_l, s_c = pd.Series(h), pd.Series(lo), pd.Series(c)
    if future:
        hi = s_h.shift(-1).rolling(H).max().shift(-(H - 1)).to_numpy()
        lo_ = s_l.shift(-1).rolling(H).min().shift(-(H - 1)).to_numpy()
        end = s_c.shift(-H).to_numpy()
        start = c
    else:
        hi = s_h.rolling(H).max().to_numpy()
        lo_ = s_l.rolling(H).min().to_numpy()
        end = c
        start = s_c.shift(H).to_numpy()
    return np.abs(end - start) / np.maximum(hi - lo_, 1e-12)


def build(sym: str) -> pd.DataFrame:
    df = _resample(_load_full(sym)).resample(TF).agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}).dropna()
    c = df["close"].to_numpy(); h = df["high"].to_numpy()
    lo = df["low"].to_numpy(); v = df["volume"].to_numpy()
    s = pd.Series(c)
    mid = s.rolling(20).mean().to_numpy()
    sd = s.rolling(20).std().to_numpy()
    bbw = (4.0 * sd) / np.maximum(mid, 1e-12)
    out = pd.DataFrame(index=df.index)
    out["adx"] = adx14(h, lo, c)
    out["chop"] = chop14(h, lo, c)
    out["bbw"] = bbw
    out["bbw_q33"] = roll_q(bbw, 24 * 90, 0.33)
    out["bbw_slope"] = pd.Series(bbw).pct_change(5).to_numpy()      # 수축 진행중?
    out["volr"] = v / np.maximum(pd.Series(v).rolling(20).mean().to_numpy(), 1e-12)
    out["hour"] = df.index.hour
    out["dow"] = df.index.dayofweek
    # 상위TF (4h) — 직전 완결 4h 봉
    d4 = df.resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    a4 = pd.Series(adx14(d4["high"].to_numpy(), d4["low"].to_numpy(),
                         d4["close"].to_numpy()), index=d4.index).shift(1)
    out["adx4h"] = a4.reindex(df.index, method="ffill").to_numpy()
    # 정답
    out["past_tr"] = realized_trend(h, lo, c, H, future=False)
    out["fut_tr"] = realized_trend(h, lo, c, H, future=True)
    out["now_chop"] = out["past_tr"] < CHOP_THR       # 사후 인식 정답
    out["fut_chop"] = out["fut_tr"] < CHOP_THR        # 사전 예측 정답
    out["sym"] = sym
    return out.dropna()


def report(d: pd.DataFrame, target: str, title: str) -> None:
    y = d[target].to_numpy().astype(bool)
    base = y.mean()
    print(f"\n===== {title} =====", flush=True)
    print(f"  기저율 {base * 100:.1f}%  표본 {len(d):,}", flush=True)
    print(f"  {'판정':<26} {'빈도':>6} {'정밀도':>7} {'리프트':>6}", flush=True)
    preds = {
        "ADX<20": d.adx < 20,
        "ADX<25": d.adx < 25,
        "스퀴즈q33": d.bbw < d.bbw_q33,
        "CHOP>55": d.chop > 55,
        "현재횡보(지속성)": d.now_chop if target == "fut_chop" else pd.Series(False, index=d.index),
        "거래량<0.8x": d.volr < 0.8,
        "아시아(0-8utc)": d.hour < 8,
        "주말": d.dow >= 5,
        "4h ADX<20": d.adx4h < 20,
        "BBW 수축진행(<-10%)": d.bbw_slope < -0.10,
        "지속성+ADX<20": (d.now_chop & (d.adx < 20)) if target == "fut_chop" else pd.Series(False, index=d.index),
        "지속성+스퀴즈": (d.now_chop & (d.bbw < d.bbw_q33)) if target == "fut_chop" else pd.Series(False, index=d.index),
    }
    for k, p in preds.items():
        p = p.to_numpy().astype(bool)
        if p.sum() < 200:
            continue
        prec = y[p].mean()
        print(f"  {k:<26} {100 * p.mean():5.1f}% {100 * prec:6.1f}% {prec / base:5.2f}x",
              flush=True)


def logistic_eval(d: pd.DataFrame, target: str) -> None:
    """복합 모델 — 시계열 분할(앞 70% 학습 / 뒤 30% 검증), 과적합 방지."""
    feats = ["adx", "chop", "bbw", "bbw_slope", "volr", "hour", "dow", "adx4h"]
    if target == "fut_chop":
        feats.append("now_chop")
    d = d.sort_index()
    X = d[feats].astype(float).to_numpy()
    y = d[target].to_numpy().astype(int)
    cut = int(len(d) * 0.7)
    Xtr, Xte, ytr, yte = X[:cut], X[cut:], y[:cut], y[cut:]
    mu, sg = Xtr.mean(0), Xtr.std(0) + 1e-9
    Xtr = (Xtr - mu) / sg; Xte = (Xte - mu) / sg
    # 간단 로지스틱 (경사하강 — 외부 의존 없이)
    w = np.zeros(Xtr.shape[1]); b = 0.0
    for _ in range(400):
        z = Xtr @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        g = p - ytr
        w -= 0.5 * (Xtr.T @ g) / len(ytr)
        b -= 0.5 * g.mean()
    pte = 1 / (1 + np.exp(-np.clip(Xte @ w + b, -30, 30)))
    base = yte.mean()
    # AUC (rank 기반)
    order = np.argsort(pte)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(pte) + 1)
    npos, nneg = yte.sum(), len(yte) - yte.sum()
    auc = (ranks[yte == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else 0.5
    print(f"\n  [복합 로지스틱] 검증 표본 {len(yte):,}  기저율 {base * 100:.1f}%  AUC={auc:.3f}",
          flush=True)
    for q in (0.90, 0.80, 0.70):
        thr = np.quantile(pte, q)
        sel = pte >= thr
        prec = yte[sel].mean()
        print(f"    상위{100 * (1 - q):.0f}% 확신구간: 빈도{100 * sel.mean():4.1f}% "
              f"정밀도{100 * prec:5.1f}% 리프트{prec / base:.2f}x", flush=True)


def main() -> int:
    ds = []
    for sym in PAIRS:
        try:
            ds.append(build(sym))
        except Exception as e:  # noqa: BLE001
            print(f"{sym} 실패: {e}", flush=True)
    d = pd.concat(ds)
    print(f"총 표본 {len(d):,} ({TF}, 지평 {H}봉)", flush=True)
    report(d, "now_chop", "[A] 사후 인식 — 지금 횡보 중인가")
    logistic_eval(d, "now_chop")
    report(d, "fut_chop", "[B] 사전 예측 — 앞으로 횡보일까")
    logistic_eval(d, "fut_chop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
