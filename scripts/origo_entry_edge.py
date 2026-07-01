"""Origo 진입 엣지 광역 스윕 — FST #1 자율연구.

FST 2026-07-01: Origo 는 청산(trailing)으로 RR 을 1.9 까지 올려도 net 적자 →
근본 병목은 진입 엣지(이기는 판 부족). 진입 파라미터를 단변수로 광역 스윕해
어느 축이 net/RR 을 흑자로 돌리는지 탐색. 청산은 진입 순수 비교 위해 현행
고정(tp_rr_override=1.0)으로 고정.

진입 축(단변수, 나머지 BASE 고정):
    - min_confluence: 신호 요건 개수(높을수록 엄격)
    - htf_align_threshold: 상위 EMA 정렬 강도
    - ote_level: 되돌림 진입 깊이
    - sl_dist_mult: 손절 거리
    - setup_stale_bars: setup 신선도(오래된 setup 폐기)
    - disable_time_filter: 킬존/시간 필터 on/off
    - min_rr: 최소 손익비 필터

진입 파라미터가 바뀌면 timeline 재빌드(캐시). 7페어 합산.
사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/origo_entry_edge.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=1.0,
)
# 축 이름 -> 값 리스트 (BASE 값 포함, 첫 원소가 현행이면 굳이 표시)
SWEEPS = {
    "min_confluence": [3, 4, 5, 6, 7],
    "htf_align_threshold": [1, 2, 3, 4, 5],
    "ote_level": [0.5, 0.618, 0.707, 0.786],
    "sl_dist_mult": [2.0, 2.5, 3.0, 3.5, 4.0],
    "setup_stale_bars": [2, 3, 4, 6],
    "disable_time_filter": [False, True],
    "min_rr": [1.5, 2.0, 2.5, 3.0],
}


def _metrics(trades, days):
    ts = list(trades)
    if not ts:
        return None
    cum = peak = mdd = 0.0
    for t in sorted(ts, key=lambda t: t.exit_idx):
        cum += t.net_pnl_pct
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    wins = [t.net_pnl_pct for t in ts if t.net_pnl_pct > 0]
    losses = [t.net_pnl_pct for t in ts if t.net_pnl_pct < 0]
    n = len(ts)
    return dict(cum=cum, mdd=mdd, nwin=len(wins), n=n,
                gw=sum(wins), gl=sum(losses), days=days)


def _eval(overrides):
    """7페어 합산 — (usdt, mdd, wr, rr, n, freq)."""
    tot = {"cum": 0.0, "mdd": 0.0, "nwin": 0, "n": 0, "gw": 0.0, "gl": 0.0, "days": 0.0}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        days = len(df5) / 288.0
        ttl = 12 if sym == "BTCUSDT" else 6
        cfg = {**BASE, "entry_ttl_bars": ttl, **overrides}
        tl = cached_setup_timeline(df5, BacktestConfig(**cfg), sym)
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
        m = _metrics(bt.trades, days)
        if not m:
            continue
        for k in ("cum", "mdd", "nwin", "n", "gw", "gl", "days"):
            tot[k] += m[k]
    n = tot["n"]
    wr = tot["nwin"] / n * 100 if n else 0.0
    aw = tot["gw"] / tot["nwin"] if tot["nwin"] else 0.0
    al = tot["gl"] / (n - tot["nwin"]) if (n - tot["nwin"]) else 0.0
    rr = aw / -al if al < 0 else 0.0
    return dict(usdt=tot["cum"] * SEED / 100, mdd=tot["mdd"] * SEED / 100,
                wr=wr, rr=rr, n=n, freq=n / (tot["days"] / len(PAIRS)) if tot["days"] else 0)


def main() -> int:
    lines = ["===== Origo 진입 엣지 광역 스윕 (7페어 5년, 청산=현행고정tp1) =====",
             "BASE: conf4 htf2 ote0.707 sl3 stale3 킬존ON rr2.0. 단변수 스윕.",
             ""]
    # 기준선
    b = _eval({})
    lines.append(f"[BASE] usdt={b['usdt']:+.0f} DD={b['mdd']:.0f} 승={b['wr']:.0f}% "
                 f"RR={b['rr']:.2f} n={b['n']} freq={b['freq']:.2f}/일")
    print(lines[-1], flush=True)
    for axis, vals in SWEEPS.items():
        lines.append(f"\n[{axis}]")
        lines.append(f"  {'값':<8}{'USDT':>8}{'DD':>7}{'승률':>6}{'RR':>6}{'거래':>7}{'빈도':>7}")
        for v in vals:
            r = _eval({axis: v})
            row = (f"  {str(v):<8}{r['usdt']:>+8.0f}{r['mdd']:>7.0f}{r['wr']:>5.0f}%"
                   f"{r['rr']:>6.2f}{r['n']:>7d}{r['freq']:>6.2f}회")
            lines.append(row)
            print(f"{axis}={v} usdt={r['usdt']:+.0f} rr={r['rr']:.2f}", flush=True)

    txt = "\n".join(lines)
    with open("origo_entry_edge_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\nDONE (결과: origo_entry_edge_result.txt)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
