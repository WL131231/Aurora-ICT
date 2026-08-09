"""국면(추세/반등) 판정 '눈' 분류력 — 단독 3개 + 결합 4개 = 7가지 분리도 측정.

파트너(2026-06-15): EMA스프레드(1)·구조 BOS비율(2)·ATR(3) 를 각각 + 1+2/2+3/1+3/
1+2+3 결합까지 다 비교. 게이트 연결 전에 "어느 눈(조합)이 추세(IN)/반등(OUT)을 제일
잘 가르는지" 정량 스크리닝.

방법: 세 지표 모두 "추세=높음" 방향. 전체(IN+OUT) 기준 표준화(z) 후, 조합은 z 평균.
분리도 d = (IN중앙 - OUT중앙)/pooled_std. d 클수록 좋은 눈(추세에서 높고 반등에서 낮음).

사용: PYTHONPATH=src python scripts/regime_probe.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

FIXED7 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
IN_PHASES = [
    ("2021-11", "2021-11-10", "2021-12-22"),
    ("2023-01", "2023-01-05", "2023-02-16"),
    ("2024-02", "2024-01-25", "2024-03-08"),
]
OUT_PHASES = [
    ("2024-08", "2024-08-01", "2024-09-12"),
    ("2026", "2026-04-28", "2026-06-10"),
]
ALIGN_PERIODS = (60, 120, 200, 350, 480, 620)


def _resample(d: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    o = d["open"].resample(rule).first()
    h = d["high"].resample(rule).max()
    lo = d["low"].resample(rule).min()
    c = d["close"].resample(rule).last()
    return pd.DataFrame({"open": o, "high": h, "low": lo, "close": c}).dropna()


def _load(sym: str, s: str, e: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[s:e]


def ema_spread(df5: pd.DataFrame) -> pd.Series:
    """1h EMA(60~620) 간격/close. 추세=발산(큼). look-ahead 방지 shift."""
    h1 = df5["close"].resample("1h").last()
    em = pd.concat(
        [h1.ewm(span=p, adjust=False).mean() for p in ALIGN_PERIODS], axis=1,
    )
    spread = (em.max(axis=1) - em.min(axis=1)) / h1
    return spread.shift(1).reindex(df5.index, method="ffill")


def atr_pct(df5: pd.DataFrame, n: int = 14) -> pd.Series:
    """5m ATR/close. 변동성 — 추세에서 약간 큼."""
    pc = df5["close"].shift()
    tr = pd.concat(
        [df5["high"] - df5["low"], (df5["high"] - pc).abs(), (df5["low"] - pc).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n).mean() / df5["close"]


def struct_score(df5: pd.DataFrame, lookback: int = 288) -> pd.Series:
    """최근 lookback(5m봉≈1일) BOS 비율 = BOS/(BOS+CHoCH). 추세=1쪽(BOS 우세)."""
    from aurora_ict.indicators.structure import (
        StructureType,
        detect_structure_events,
    )
    from aurora_ict.indicators.swing_points import detect_swing_points

    swings = detect_swing_points(df5, left=2, right=2)
    events = detect_structure_events(df5, swings)
    bos = np.zeros(len(df5))
    ch = np.zeros(len(df5))
    for ev in events:
        if ev.type in (StructureType.BOS_BULLISH, StructureType.BOS_BEARISH):
            bos[ev.idx] = 1
        elif ev.type in (StructureType.CHOCH_BULLISH, StructureType.CHOCH_BEARISH):
            ch[ev.idx] = 1
    bos_r = pd.Series(bos, index=df5.index).rolling(lookback).sum()
    ch_r = pd.Series(ch, index=df5.index).rolling(lookback).sum()
    tot = bos_r + ch_r
    return bos_r / tot.where(tot > 0, np.nan)


def _collect_rows(phases) -> np.ndarray:
    """봉별 [spread, atr, struct] 행렬 (유효봉만)."""
    rows = []
    for sym in FIXED7:
        for _pn, s, e in phases:
            d1 = _load(sym, s, e)
            if len(d1) < 5000:
                continue
            df5 = _resample(d1)
            if len(df5) < 700:
                continue
            sp = ema_spread(df5).to_numpy()
            at = atr_pct(df5).to_numpy()
            st = struct_score(df5).to_numpy()
            m = np.isfinite(sp) & np.isfinite(at) & np.isfinite(st)
            rows.append(np.column_stack([sp[m], at[m], st[m]]))
    return np.vstack(rows)


def main() -> int:
    print("=== 국면 눈 분리도: 단독 3 + 결합 4 (d 클수록 추세/반등 잘 가름) ===", flush=True)
    inn = _collect_rows(IN_PHASES)
    out = _collect_rows(OUT_PHASES)
    alld = np.vstack([inn, out])
    mu, sd = alld.mean(0), alld.std(0)
    inz = (inn - mu) / sd
    outz = (out - mu) / sd
    # col: 0=spread(1), 1=atr(3), 2=struct(2)
    combos = {
        "1_EMA스프레드": [0],
        "2_구조BOS":     [2],
        "3_ATR":         [1],
        "1+2":           [0, 2],
        "2+3":           [2, 1],
        "1+3":           [0, 1],
        "1+2+3":         [0, 1, 2],
    }
    rank = []
    for name, cols in combos.items():
        iv = inz[:, cols].mean(1)
        ov = outz[:, cols].mean(1)
        pooled = np.sqrt((iv.std() ** 2 + ov.std() ** 2) / 2) or 1.0
        d = (np.median(iv) - np.median(ov)) / pooled
        rank.append((name, d, np.median(iv), np.median(ov)))
        print(
            f"  {name:14s} d={d:+.3f}  IN_med={np.median(iv):+.3f} OUT_med={np.median(ov):+.3f}",
            flush=True,
        )
    rank.sort(key=lambda r: -r[1])
    print(f"\n순위(분리도): {' > '.join(f'{n}({d:+.2f})' for n, d, _, _ in rank)}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
