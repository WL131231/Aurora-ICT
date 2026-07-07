"""레인지 인식 관망 게이트 — "3번 판단" 증류 2호 (FST #4 자율연구, 2026-07-07).

실측: 6/26~30 축적 레인지(BTC 58~61k)에서 봇이 숏 39개, 매일 -33~-68 출혈.
ICT 정통 "draw 불명확한 박스에선 자리가 아니다"를 룰로: 직전 M일 고저 폭이
종가 대비 X% 미만이면(압축 레인지) 다음날 진입 전체 차단.

주의: 압축은 확장의 전조(스퀴즈)이기도 함 — 과차단하면 돌파 초입을 놓침.
M=5 고정, X 스윕으로 이득/비용 곡선 확인. 타임라인 필터(재생만).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/range_gate.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
M_DAYS = 5
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=5.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)
X_PCTS = [0.0, 3.0, 4.0, 5.0, 6.0]  # 0 = 게이트 없음(기준). BTC 5일폭 X% 미만 차단.


def _range_flags(df5, x_pct: float):
    """일봉 직전 M일 (고-저)/종가 폭 < x% → 다음날 차단 마스크."""
    d = df5.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    width = (d["high"].rolling(M_DAYS).max() - d["low"].rolling(M_DAYS).min()) \
        / d["close"] * 100
    blocked_next = width.shift(1) < x_pct  # 어제까지 M일 폭 기준 → 오늘 차단
    return d.index, blocked_next.fillna(False).values


def _filtered(tl, df5, days_idx, blocked):
    day_of = days_idx.searchsorted(df5.index.normalize(), side="right") - 1
    out = list(tl)
    for i, item in enumerate(out):
        if item is not None and day_of[i] >= 0 and blocked[day_of[i]]:
            out[i] = None
    return out


def _metrics(trades):
    ts = list(trades)
    wins = [t.net_pnl_pct for t in ts if t.net_pnl_pct > 0]
    cum = peak = mdd = 0.0
    for t in sorted(ts, key=lambda t: t.exit_idx):
        cum += t.net_pnl_pct
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return dict(cum=cum, mdd=mdd, n=len(ts), nwin=len(wins))


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl})
    tl = cached_setup_timeline(df5, cfg, sym)
    out = {}
    for x in X_PCTS:
        if x <= 0:
            ftl = tl
        else:
            days_idx, blocked = _range_flags(df5, x)
            ftl = _filtered(tl, df5, days_idx, blocked)
        label = "기준(게이트없음)" if x <= 0 else f"5일폭<{x:.0f}% 차단"
        out[label] = _metrics(run_backtest_from_timeline(df5, ftl, cfg).trades)
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    labels = ["기준(게이트없음)"] + [f"5일폭<{x:.0f}% 차단" for x in X_PCTS if x > 0]
    lines = ["===== 레인지 관망 게이트 (7페어 5년, BASE=Origo1.5 정합 BE 포함) =====",
             f"{'변형':<20}{'USDT':>8}{'DD':>7}{'승률':>6}{'거래':>7}"]
    for label in labels:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0, "nwin": 0}
        for _sym, out in results:
            m = out[label]
            for k in tot:
                tot[k] += m[k]
        wr = tot["nwin"] / tot["n"] * 100 if tot["n"] else 0.0
        lines.append(f"{label:<18}{tot['cum'] * SEED / 100:>+8.0f}"
                     f"{tot['mdd'] * SEED / 100:>7.0f}{wr:>5.0f}%{tot['n']:>7d}")
    txt = "\n".join(lines)
    with open("range_gate_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
