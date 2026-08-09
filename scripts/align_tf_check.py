"""align TF 불일치 검증 — 5분봉 vs 1h봉 EMA(60~620) 정렬 score 분포 비교.

라이브는 1h봉 align(htf_ema_bias_tf=1h), 백테 replay 는 5분봉(trade TF) align.
같은 periods(60~620)·threshold(2)인데 TF가 12배 달라 방향이 갈린다는 가설 검증.
최근 구간에서 1h align 이 숏(score<=-2) 편향이고 5분이 균형이면 → 라이브 숏91% 의
근본이 TF 불일치임을 확정. _full(1m 2년)로 1h EMA620(26일) 충분히 계산.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/align_tf_check.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
PERIODS = (60, 120, 200, 350, 480, 620)
T = 2  # threshold
RECENT_DAYS = 30  # 분포 집계 구간


def _resample_tf(df1m, rule):
    return pd.DataFrame({
        "close": df1m["close"].resample(rule).last(),
    }).dropna()


def _align_series(closes: pd.Series):
    """각 봉 align score (인접 EMA 쌍 정/역배열 ±1 누적, 범위 -5~+5)."""
    emas = {p: closes.ewm(span=p, adjust=False).mean() for p in PERIODS}
    score = pd.Series(0, index=closes.index)
    for a, b in zip(PERIODS[:-1], PERIODS[1:]):
        score = score + (emas[a] > emas[b]).astype(int) - (emas[a] < emas[b]).astype(int)
    return score


def _dist(scores: pd.Series, label):
    n = len(scores)
    if n == 0:
        return f"{label}: 데이터 없음"
    short = (scores <= -T).sum()
    long = (scores >= T).sum()
    neu = n - short - long
    return (f"{label}: 평균{scores.mean():+.2f}  "
            f"숏(<=-{T}) {100 * short / n:.0f}% / 중립 {100 * neu / n:.0f}% / 롱(>={T}) {100 * long / n:.0f}%")


def main() -> int:
    print(f"=== align TF 검증: 5분 vs 1h EMA{PERIODS} 정렬 (최근 {RECENT_DAYS}일) ===", flush=True)
    agg5 = []
    agg1h = []
    for sym in PAIRS:
        df1m = _load_full(sym)
        last = df1m.index[-1]
        cut = last - pd.Timedelta(days=RECENT_DAYS)
        s5 = _align_series(_resample_tf(df1m, "5min")["close"])
        s1h = _align_series(_resample_tf(df1m, "1h")["close"])
        s5r = s5[s5.index >= cut]
        s1hr = s1h[s1h.index >= cut]
        agg5.append(s5r)
        agg1h.append(s1hr)
        print(f"  {sym}", flush=True)
        print(f"    {_dist(s5r, '5분 ')}", flush=True)
        print(f"    {_dist(s1hr, '1h  ')}", flush=True)

    all5 = pd.concat(agg5)
    all1h = pd.concat(agg1h)
    lines = [
        "",
        f"=== 7페어 합산 (최근 {RECENT_DAYS}일, 데이터 마지막 {df1m.index[-1]}) ===",
        "  " + _dist(all5, "5분 align "),
        "  " + _dist(all1h, "1h  align "),
        "",
        "라이브 실측: 진입 숏 91% / 롱 9%",
        "백테(5분 align): 롱50/숏50 (live_replay_fresh)",
    ]
    s5_short = 100 * (all5 <= -T).sum() / len(all5)
    s1h_short = 100 * (all1h <= -T).sum() / len(all1h)
    if s1h_short > s5_short + 15:
        lines.append(f"\n→ ✅ TF 불일치 확정: 1h align 숏편향 {s1h_short:.0f}% >> 5분 {s5_short:.0f}%. "
                     f"라이브 1h align 이 숏 고착의 근본. 처방: align TF 를 trade TF(5분)로 일치.")
    else:
        lines.append(f"\n→ 1h 숏편향 {s1h_short:.0f}% vs 5분 {s5_short:.0f}% — TF 차이 작음. 다른 원인 재검토.")

    txt = "\n".join(lines)
    with open("align_tf_check_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print(txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 align_tf_check_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
