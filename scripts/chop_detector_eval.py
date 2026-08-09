"""#AUTONOMOUS 2026-07-28: 횡보 판정 지표 자체의 정확도 평가 (파트너: "판단 자체는 어때?").

지금까지 지표를 매매결과로만 간접 평가했다. 여기선 **판정 자체의 예측력**을 직접 측정.

정답(사후 실현 국면) — 판정 시점 이후 H봉 구간에서:
  · 실현추세 = |종가변화| / (구간 고저폭)          … 방향성 (Kaufman ER 사후판)
  · 횡보 정답 = 실현추세 < 0.30 (= 왔다갔다만 하고 순변화 미미)
  · 추세 정답 = 실현추세 > 0.60
지표(판정 시점, 직전 완결봉만 — 인과):
  ADX14 <18/<20/<25 · BBW 스퀴즈 q20/q33 · CHOP14 >55/>61.8
  + 조합(ADX<20 AND 스퀴즈q33 / ADX<20 OR CHOP>55)
측정: 정밀도(판정=횡보 중 실제 횡보 비율) · 재현율 · F1 · 리프트(=정밀도/기저율) ·
      추세오판율(횡보라 했는데 실제 추세) · 판정빈도.
비교 기준선: 무작위 판정(기저율) — 리프트 1.0 이면 지표가 정보 없음.
페어 7종 × TF(5m·1h) × H(12·48봉).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from chop_gate_bakeoff import adx14, chop14, roll_q  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]


def build(sym: str, tf: str, H: int):
    df = _resample(_load_full(sym))
    if tf != "5m":
        df = df.resample(tf).agg({"high": "max", "low": "min", "close": "last"}).dropna()
    c = df["close"].to_numpy(); h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    n = len(c)
    bars_day = 288 if tf == "5m" else 24
    # 지표 (판정 시점 i — 직전 완결봉까지의 정보만)
    adx = adx14(h, lo, c)
    chop = chop14(h, lo, c)
    s = pd.Series(c)
    mid = s.rolling(20).mean().to_numpy()
    sd = s.rolling(20).std().to_numpy()
    bbw = (4.0 * sd) / np.maximum(mid, 1e-12)
    q20 = roll_q(bbw, bars_day * 90, 0.20)
    q33 = roll_q(bbw, bars_day * 90, 0.33)
    # 사후 정답 — i+1..i+H 구간의 실현 추세도
    fut_hi = pd.Series(h).shift(-1).rolling(H).max().shift(-(H - 1)).to_numpy()
    fut_lo = pd.Series(lo).shift(-1).rolling(H).min().shift(-(H - 1)).to_numpy()
    fut_c = pd.Series(c).shift(-H).to_numpy()
    rng = np.maximum(fut_hi - fut_lo, 1e-12)
    realized = np.abs(fut_c - c) / rng          # 0=완전 횡보, 1=일직선 추세
    valid = ~np.isnan(realized) & ~np.isnan(adx) & ~np.isnan(chop) & ~np.isnan(q33)
    truth_chop = (realized < 0.30) & valid
    truth_trend = (realized > 0.60) & valid
    preds = {
        "ADX<18": adx < 18,
        "ADX<20": adx < 20,
        "ADX<25": adx < 25,
        "스퀴즈q20": bbw < q20,
        "스퀴즈q33": bbw < q33,
        "CHOP>55": chop > 55,
        "CHOP>61.8": chop > 61.8,
        "ADX<20 AND 스퀴즈q33": (adx < 20) & (bbw < q33),
        "ADX<20 OR CHOP>55": (adx < 20) | (chop > 55),
    }
    return {k: (v & valid) for k, v in preds.items()}, truth_chop, truth_trend, valid


def main() -> int:
    for tf, H in (("5m", 48), ("1h", 12), ("1h", 48)):
        print(f"\n########## TF {tf} · 예측지평 {H}봉 ##########", flush=True)
        agg_pred: dict[str, np.ndarray] = {}
        agg_chop = []
        agg_trend = []
        agg_valid = []
        for sym in PAIRS:
            try:
                preds, tc, tt, va = build(sym, tf, H)
            except Exception as e:  # noqa: BLE001
                print(f"  {sym} 실패: {e}", flush=True)
                continue
            for k, v in preds.items():
                agg_pred.setdefault(k, []).append(v)
            agg_chop.append(tc); agg_trend.append(tt); agg_valid.append(va)
        if not agg_chop:
            continue
        tc = np.concatenate(agg_chop); tt = np.concatenate(agg_trend)
        va = np.concatenate(agg_valid)
        base = tc[va].mean()  # 기저율 — 무작위로 찍었을 때 횡보 맞출 확률
        print(f"  기저율(실제 횡보 비율) = {base * 100:.1f}%  (표본 {va.sum():,})", flush=True)
        print(f"  {'지표':<22} {'판정빈도':>7} {'정밀도':>7} {'재현율':>7} {'F1':>6} "
              f"{'리프트':>6} {'추세오판':>8}", flush=True)
        for k in agg_pred:
            p = np.concatenate(agg_pred[k])
            npred = p.sum()
            if npred < 100:
                continue
            prec = tc[p].mean()
            rec = p[tc].mean() if tc.sum() else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            lift = prec / base if base else 0.0
            trend_err = tt[p].mean()
            print(f"  {k:<22} {100 * npred / va.sum():6.1f}% {100 * prec:6.1f}% "
                  f"{100 * rec:6.1f}% {100 * f1:5.1f}% {lift:5.2f}x {100 * trend_err:7.1f}%",
                  flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
