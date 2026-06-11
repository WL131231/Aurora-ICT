"""FSD-style 학습 데이터셋 생성 — 백테스트 후보 setup 전수 + 결과 라벨.

테슬라 shadow mode 의 백테스트 버전: 봇이 *볼 수 있었던 모든 setup*(거른 것
포함)에 특징 + 시뮬 결과(라벨)를 붙여 학습 데이터로. (pair, phase)별 CSV 저장.

라벨: #EDGE-V2 청산 규칙(SLx3, TP 원RR 유지, ttl 120분 체결)으로 시뮬 —
win(tp)=1 / loss(sl)=0. 미체결/eod 는 제외. 같은 setup(ts+방향)은 첫 등장
1회만 (봇이 행동할 시점).

사용: python scripts/ml_dataset_gen.py BTCUSDT
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

from aurora_ict.backtest.replay import (
    BacktestConfig,
    _precompute_align_score,
    _simulate_exit,
    _simulate_fill,
    build_setup_timeline,
)
from aurora_ict.strategy.silver_bullet import Direction

SYM = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"

PHASES = [
    ("in", "2021-11", "2021-11-10", "2021-12-22"),
    ("in", "2023-01", "2023-01-05", "2023-02-16"),
    ("in", "2024-02", "2024-01-25", "2024-03-08"),
    ("out", "2024-08", "2024-08-01", "2024-09-12"),
    ("out", "2026", "2026-04-28", "2026-06-10"),
]

CFG = BacktestConfig(min_rr=2.0, disable_time_filter=False)  # 후보 폭넓게(rr2.0+)
SL_MULT = 3.0
TTL = 120


def load_slice(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{SYM}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def main() -> int:
    out_dir = Path("data/ml")
    out_dir.mkdir(exist_ok=True)
    for grp, pname, s, e in PHASES:
        df = load_slice(s, e)
        tl = build_setup_timeline(df, CFG)
        align = _precompute_align_score(df, CFG.htf_align_periods).to_numpy()
        opens = df["open"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        idx = df.index
        seen: set = set()
        rows = []
        for i in range(CFG.window, len(df) - 1):
            item = tl[i]
            if item is None:
                continue
            setup, bars_since = item
            key = (setup.ts_ms, setup.direction.value)
            if key in seen:
                continue
            seen.add(key)
            if bars_since > CFG.setup_stale_bars:
                continue
            # 라벨 시뮬 — EDGE-V2 청산 규칙.
            entry = setup.entry
            risk = abs(entry - setup.stop_loss)
            if risk <= 0 or entry <= 0:
                continue
            rr0 = abs(setup.take_profit - entry) / risk
            nr = risk * SL_MULT
            if setup.direction is Direction.LONG:
                sl, tp = entry - nr, entry + nr * rr0
            else:
                sl, tp = entry + nr, entry - nr * rr0
            fill = _simulate_fill(highs, lows, i, setup.direction, entry, TTL)
            if fill is None:
                continue
            exit_idx, _px, outcome = _simulate_exit(
                opens, highs, lows, closes, fill, setup.direction, sl, tp, CFG,
            )
            if outcome == "eod":
                continue
            a = align[i]
            ts = idx[i]
            rows.append({
                "sym": SYM, "grp": grp, "phase": pname,
                "score": setup.confluence_score,
                "rr": round(setup.risk_reward, 3),
                "dir_long": 1 if setup.direction is Direction.LONG else 0,
                "window": setup.window,
                "source": getattr(setup.source, "value", str(setup.source)),
                "sl_dist_pct": round(risk / entry * 100, 4),
                "align_score": float(a) if a == a else 0.0,
                "align_valid": 1 if a == a else 0,
                "hour_utc": ts.hour,
                "bars_since": bars_since,
                "label_win": 1 if outcome == "tp" else 0,
                "r_result": rr0 if outcome == "tp" else -1.0,
            })
        out = out_dir / f"{SYM}_{pname}_{grp}.csv"
        with out.open("w", newline="", encoding="utf-8") as f:
            if rows:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"[{SYM} {pname}] {len(rows)} samples -> {out}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
