"""정합(v2) 타임라인 5년 사전 빌드 — 7페어, 워커 4개.

2026-08-08 정합 이식으로 detect 캐시 키가 v2 로 바뀌어 기존 캐시가 전부 무효다.
confluence 재판정(본/홀드아웃)이 쓸 7페어 timeline 을 미리 만들어 둔다.
워커를 4로 묶는 이유: 가용 RAM 이 ~11GB 라 7 프로세스 동시는 스왑 위험.
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SYMS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]


def _build(sym: str) -> str:
    from bt_par import _load_full, _resample, cached_setup_timeline
    from live_parity import live_cfg
    t0 = time.time()
    df5 = _resample(_load_full(sym))
    cfg = live_cfg(sym)
    tl = cached_setup_timeline(df5, cfg, sym)
    n_set = sum(1 for x in tl if x is not None)
    return (f"{sym} bars={len(df5)} setup_bars={n_set} "
            f"{time.time() - t0:.0f}s {time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    syms = sys.argv[1:] or SYMS
    with Pool(min(4, len(syms)), maxtasksperchild=1) as p:
        for r in p.imap_unordered(_build, syms):
            print(r, flush=True)
    print("DONE", flush=True)
