"""SL 축 연구 보조 — min_sl_distance_pct 변형 타임라인을 미리 병렬 빌드해 둔다.

min_sl_distance_pct 는 bt_par 캐시 키에 들어가므로 값을 바꾸면 타임라인을
새로 만들어야 한다(페어당 10분 이상). 본 스크립트는 그 빌드만 백그라운드로
미리 돌려 캐시에 넣는 용도다. 결과 분석은 axis_sl-atr-floor_2026-08-07.py 가 한다.
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from live_parity import LIVE_BASE  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig  # noqa: E402

# 스윕 대상: 현행 0.001 대비 더 큰 최소 SL 거리(= 구조적으로 좁은 셋업 배제)
JOBS = [(s, v) for v in (0.002, 0.003, 0.005) for s in ("BTCUSDT", "ETHUSDT")]


def build(job):
    sym, v = job
    t0 = time.time()
    df5 = _resample(_load_full(sym))
    cfg = BacktestConfig(**{**LIVE_BASE, "min_sl_distance_pct": v})
    tl = cached_setup_timeline(df5, cfg, sym)
    n = sum(1 for x in tl if x is not None)
    return f"{sym} min_sl={v} 셋업봉={n} {time.time() - t0:.0f}초"


if __name__ == "__main__":
    with Pool(3) as p:
        for line in p.imap_unordered(build, JOBS):
            print(line, flush=True)
