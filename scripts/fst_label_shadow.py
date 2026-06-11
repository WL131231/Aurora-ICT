"""FST 2단계 — shadow 기록 자동 라벨링 (사후 가격으로 SL/TP 채점).

봇이 기록한 ``shadow_setups.jsonl``(진입/거름 후보, #280)의 각 setup 에 대해
로컬 1m 가격(parquet)으로 "그 자리에 들어갔다면 어떻게 됐나"를 시뮬:
ttl(기본 120분) 내 limit 체결 → SL/TP 먼저 닿는 쪽 판정 → 라벨 CSV +
verdict 별 요약(거른 게 맞았나) 출력.

사용:
    python scripts/fst_label_shadow.py <shadow_setups.jsonl> [--ttl 120]

전제: data/{SYMBOL}USDT_1m_full.parquet 가 해당 기간을 커버해야 함
(없거나 기간 밖이면 그 행은 skip — skipped 카운트로 보고).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

_PARQ_CACHE: dict[str, pd.DataFrame | None] = {}


def _load_1m(symbol: str) -> pd.DataFrame | None:
    """"BTC/USDT:USDT" → data/BTCUSDT_1m_full.parquet (캐시)."""
    base = symbol.split("/")[0]
    key = f"{base}USDT"
    if key not in _PARQ_CACHE:
        p = Path(f"data/{key}_1m_full.parquet")
        if not p.exists():
            _PARQ_CACHE[key] = None
        else:
            df = pd.read_parquet(p)
            df.index = pd.DatetimeIndex(
                pd.to_datetime(df["timestamp"], unit="ms", utc=True),
            )
            _PARQ_CACHE[key] = df[["open", "high", "low", "close"]]
    return _PARQ_CACHE[key]


def _label_one(rec: dict, ttl_bars: int) -> str | None:
    """단일 shadow 행 라벨 — "tp"/"sl"/"unfilled"/None(데이터 없음)."""
    df = _load_1m(rec["symbol"])
    if df is None:
        return None
    ts = pd.to_datetime(rec["ts_ms"], unit="ms", utc=True)
    seg = df.loc[ts:]
    if len(seg) < 5:
        return None
    entry = float(rec["entry"])
    sl = float(rec["stop_loss"])
    tp = float(rec["take_profit"])
    is_long = rec["direction"] == "long"
    highs = seg["high"].to_numpy()
    lows = seg["low"].to_numpy()
    # 체결: ttl 봉 내 limit 도달.
    fill = None
    for j in range(1, min(ttl_bars + 1, len(seg))):
        if (is_long and lows[j] <= entry) or (not is_long and highs[j] >= entry):
            fill = j
            break
    if fill is None:
        return "unfilled"
    for j in range(fill + 1, len(seg)):
        hit_sl = lows[j] <= sl if is_long else highs[j] >= sl
        hit_tp = highs[j] >= tp if is_long else lows[j] <= tp
        if hit_sl:  # 동시 도달 시 SL 우선(보수)
            return "sl"
        if hit_tp:
            return "tp"
    return None  # 아직 미결(데이터 끝)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", help="shadow_setups.jsonl 경로")
    ap.add_argument("--ttl", type=int, default=120, help="체결 대기 분 (기본 120)")
    args = ap.parse_args()

    rows = []
    skipped = 0
    with open(args.jsonl, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            label = _label_one(rec, args.ttl)
            if label is None:
                skipped += 1
                continue
            risk = abs(rec["entry"] - rec["stop_loss"])
            rr0 = abs(rec["take_profit"] - rec["entry"]) / risk if risk else 0.0
            rec["label"] = label
            rec["r_result"] = rr0 if label == "tp" else (-1.0 if label == "sl" else 0.0)
            rows.append(rec)

    out = Path(args.jsonl).with_suffix(".labeled.csv")
    if rows:
        keys = list(rows[0].keys())
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    print(f"라벨 완료 {len(rows)}건 (skip {skipped}) → {out}")

    # verdict 별 요약 — "거른 게 맞았나".
    agg: dict[str, list] = defaultdict(lambda: [0, 0, 0, 0.0])  # n, tp, sl, R
    for r in rows:
        a = agg[r["verdict"]]
        a[0] += 1
        if r["label"] == "tp":
            a[1] += 1
        elif r["label"] == "sl":
            a[2] += 1
        a[3] += r["r_result"]
    print("\nverdict        n   tp   sl   놓친/번 R")
    for v, (n, tp, sl, rr) in sorted(agg.items()):
        print(f"{v:13s} {n:4d} {tp:4d} {sl:4d}  {rr:+8.1f}")
    print("\n해석: taken 의 R>0 = 룰이 일하는 중 / grade_skip 등의 R>0 크면 "
          "거른 자리에 먹을 게 있었다는 뜻 (FST 학습 가치).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
