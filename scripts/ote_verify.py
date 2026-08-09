"""피보나치 OTE confluence 검증 — 현행(cisd+po3) vs +OTE, 7페어 net·승률·거래수.

ICT OTE(직전 임펄스 swing leg 의 0.618~0.786 되돌림) 구간 진입 시 confluence
+1. apply_ote 는 _boost_score(재생 단계)라 timeline 재사용. OTE 가점이 진입 질
(net·승률)을 올리는지, 게이트 통과로 빈도가 어떻게 변하는지 실측. ICT 정통
편입 가치 판단(파트너 6/18). cisd+po3, ttl6(BTC만 12), conf4, rr2.5, sl x3, size 0.9.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/ote_verify.py
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402

from aurora_ict.backtest.replay import (  # noqa: E402
    BacktestConfig,
    build_setup_timeline,
    run_backtest_from_timeline,
)

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)
VARIANTS = {"BASE": {}, "+OTE": {"apply_ote": True}}


def _pair(sym):
    """한 페어: timeline 1회 → BASE/+OTE 재생 → {v:(net,nwin,n)}."""
    df5 = _resample(_load_full(sym))
    if len(df5) < 1400:
        return (sym, {})
    ttl = 12 if sym == "BTCUSDT" else 6
    base_cfg = {**BASE, "entry_ttl_bars": ttl}
    tl = build_setup_timeline(df5, BacktestConfig(**base_cfg))
    res = {}
    for v, extra in VARIANTS.items():
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**{**base_cfg, **extra}))
        net = sum(t.net_pnl_pct for t in bt.trades)
        nwin = sum(1 for t in bt.trades if t.net_pnl_pct > 0)
        res[v] = (net, nwin, len(bt.trades))
    return (sym, res)


def main() -> int:
    # 순차 빌드 — timeline(페어당 setup 수만개)이 커서 Pool 동시 빌드가 OOM.
    # 1페어씩 빌드→재생→해제하면 메모리 1페어분만 점유(느리지만 안정).
    rows = []
    for sym in PAIRS:
        rows.append(_pair(sym))
        print(f"  [{len(rows)}/{len(PAIRS)}] {sym} done", flush=True)

    lines = ["===== 피보나치 OTE 검증 (현행 vs +OTE, 7페어 5년) ====="]
    lines.append(f"{'페어':<9} {'BASE net/승률/거래':>24}   {'+OTE net/승률/거래':>24}")
    agg = {v: [0.0, 0, 0] for v in VARIANTS}
    for sym, res in rows:
        if not res:
            continue
        cells = []
        for v in VARIANTS:
            net, nwin, n = res[v]
            wr = (nwin / n * 100) if n else 0.0
            cells.append(f"{net:+7.1f}/{wr:4.1f}%/{n:4d}")
            agg[v][0] += net; agg[v][1] += nwin; agg[v][2] += n
        lines.append(f"{sym:<9} " + "   ".join(cells))
    lines.append("-" * 72)
    cells = []
    for v in VARIANTS:
        net, nwin, n = agg[v]
        wr = (nwin / n * 100) if n else 0.0
        cells.append(f"{net:+7.1f}/{wr:4.1f}%/{n:4d}")
    lines.append(f"{'합계':<9} " + "   ".join(cells))

    bn, bw, bc = agg["BASE"]
    on, ow, oc = agg["+OTE"]
    bwr = (bw / bc * 100) if bc else 0.0
    owr = (ow / oc * 100) if oc else 0.0
    lines.append(f"\n[+OTE 효과]  net {on - bn:+.1f}%p,  승률 {owr - bwr:+.1f}%p,  거래 {oc - bc:+d}건")
    lines.append("※ OTE 가점으로 게이트 통과 늘면 거래↑. net·승률 같이 오르면 ICT OTE 편입 가치.")

    txt = "\n".join(lines)
    with open("ote_verify_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 ote_verify_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
