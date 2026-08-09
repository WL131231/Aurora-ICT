"""정합 타임라인 사전 빌드 — verify 단계에서 캐시 hit 하도록 미리 만들어 둔다.

2026-08-08 정합 이식으로 detect 캐시 키가 v2 로 바뀌어 전 캐시가 무효다.
BTC/ETH 5년 timeline 을 페어 병렬로 빌드한다(페어당 ~2시간).
"""
from __future__ import annotations

import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _build(sym: str) -> str:
    from bt_par import _load_full, _resample, cached_setup_timeline
    from live_parity import live_cfg
    t0 = time.time()
    df5 = _resample(_load_full(sym))
    cfg = live_cfg(sym)
    tl = cached_setup_timeline(df5, cfg, sym)
    n_set = sum(1 for x in tl if x is not None)
    return f"{sym} bars={len(df5)} setup_bars={n_set} {time.time() - t0:.0f}s"


if __name__ == "__main__":
    syms = sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]
    with Pool(len(syms)) as p:
        for r in p.imap_unordered(_build, syms):
            print(r, flush=True)
