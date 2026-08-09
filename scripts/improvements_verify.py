"""개선안 종합 검증 — baseline vs 역추세동적SL vs 분할TP(ladder) vs 둘다.

3가지를 7페어 합산 net·승률·거래수로 비교(동일 timeline 재사용 — 셋 다 재생 단계):
  BASE : sl_dist_mult 3.0 고정, ladder off (현행 Origo 1)
  A(CT): 역추세(signed_trend<0) 진입만 sl_dist_mult 4.0, 나머지 3.0 (#CT-SL)
  B(LAD): 분할 TP — 손익 10%/20%/원목표 3단(alloc 1/3) + 20% 도달 시 SL 본전+4%
  A+B : 둘 다

ladder 는 net 일부 희생 대신 승률·확정수익↑ 가설 검증(파트너 6/18). cisd+po3,
ttl6(BTC만 12), conf4, rr2.5, size 0.9, lev 20.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/improvements_verify.py
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
    leverage=20.0,
)
# 분할 alloc 1/3·1/3·(나머지 1/3 원TP). be: 2번째 도달 시 본전+4%. X·Y 동일 조건.
_LAD = {"ladder_tp": True, "ladder_alloc": (0.3334, 0.3333),
        "ladder_be_pnl": 4.0, "ladder_be_after": 2}
VARIANTS = {
    "BASE": {},
    "A_CT": {"sl_dist_mult_ct": 4.0, "ct_trend_threshold": 0.0},
    # X: 절대 손익% 10/20/원TP 3단
    "B_X_pnl": {**_LAD, "ladder_mode": "pnl", "ladder_levels_pnl": (10.0, 20.0)},
    # Y: 원 TP 거리 3등분 (1/3·2/3·원TP)
    "B_Y_frac": {**_LAD, "ladder_mode": "tpfrac", "ladder_tp_fracs": (0.3333, 0.6667)},
    # A + 더 나은 ladder 는 결과 보고 추가 판단 (여기선 A+X 표기)
    "AB_X": {"sl_dist_mult_ct": 4.0, "ct_trend_threshold": 0.0,
             **_LAD, "ladder_mode": "pnl", "ladder_levels_pnl": (10.0, 20.0)},
}


def _pair(sym):
    """한 페어: timeline 1회(BTC만 ttl12) → variant별 재생 → {v:(net,nwin,n)}."""
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
    # 워커 4로 제한(5 variant × 큰 timeline → 메모리 절약) + 페어별 진행 출력.
    rows = []
    with Pool(min(4, len(PAIRS))) as p:
        for sym, res in p.imap_unordered(_pair, PAIRS):
            print(f"  [{len(rows) + 1}/{len(PAIRS)}] {sym} done", flush=True)
            rows.append((sym, res))

    lines = ["===== 개선안 종합 검증 (7페어 5년, net/승률/거래수) ====="]
    lines.append(f"{'페어':<9} " + "  ".join(f"{v:>18}" for v in VARIANTS))
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
        lines.append(f"{sym:<9} " + "  ".join(cells))
    lines.append("-" * 90)
    cells = []
    for v in VARIANTS:
        net, nwin, n = agg[v]
        wr = (nwin / n * 100) if n else 0.0
        cells.append(f"{net:+7.1f}/{wr:4.1f}%/{n:4d}")
    lines.append(f"{'합계':<9} " + "  ".join(cells))

    base_net = agg["BASE"][0]
    base_wr = (agg["BASE"][1] / agg["BASE"][2] * 100) if agg["BASE"][2] else 0.0
    lines.append(f"\n[BASE 대비 변화]  (각 셀: net합%/승률/거래수)")
    for v in VARIANTS:
        if v == "BASE":
            continue
        net, nwin, n = agg[v]
        wr = (nwin / n * 100) if n else 0.0
        lines.append(f"  {v:<6}: net {net - base_net:+.1f}%p, 승률 {wr - base_wr:+.1f}%p")
    lines.append("\n※ ladder(B)는 net↓ 예상이나 승률·확정수익↑면 구독제 상품성 trade-off 판단.")

    txt = "\n".join(lines)
    with open("improvements_verify_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 improvements_verify_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
