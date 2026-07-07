"""증류 6호 — Premium/Discount 게이트 (FST #4, 2026-07-08. 정통 간극 마지막 후보).

정통: dealing range(최근 스윙 고저)를 50% 로 갈라 매수는 discount(하반),
매도는 premium(상반)에서만. 현행 OTE(0.707)는 FVG '내부' 되돌림 깊이라
레인지 레벨의 P/D 와 별개 — 롤링 N일 고저 기준 P/D 게이트를 추가해 검증.
    LONG: 진입가가 N일 레인지 하위 50% 일 때만 / SHORT: 상위 50% 일 때만.
N=5/10/20. 재생만(캐시).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/premium_discount_gate.py
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
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=5.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)
NDAYS = [5, 10, 20]


def _pd_filtered(tl, df5, ndays: int):
    """N일 롤링 고저 50% 기준 — 롱=discount만 / 숏=premium만."""
    from aurora_ict.strategy.silver_bullet import Direction
    bars = ndays * 288
    hi = df5["high"].rolling(bars, min_periods=bars // 2).max().values
    lo = df5["low"].rolling(bars, min_periods=bars // 2).min().values
    close = df5["close"].values
    out = list(tl)
    for i, item in enumerate(out):
        if item is None or np.isnan(hi[i]) or hi[i] <= lo[i]:
            continue
        mid = (hi[i] + lo[i]) / 2.0
        setup = item[0]
        if setup.direction is Direction.LONG and close[i] > mid:
            out[i] = None          # premium 에서 롱 금지
        elif setup.direction is Direction.SHORT and close[i] < mid:
            out[i] = None          # discount 에서 숏 금지
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
    out = {"기준(P/D없음)": _metrics(run_backtest_from_timeline(df5, tl, cfg).trades)}
    for n in NDAYS:
        ftl = _pd_filtered(tl, df5, n)
        out[f"P/D {n}일"] = _metrics(run_backtest_from_timeline(df5, ftl, cfg).trades)
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    labels = ["기준(P/D없음)"] + [f"P/D {n}일" for n in NDAYS]
    lines = ["===== Premium/Discount 게이트 (7페어 5년, BASE=Origo 1.5) =====",
             f"{'변형':<14}{'USDT':>8}{'DD':>7}{'승률':>6}{'거래':>7}"]
    for label in labels:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0, "nwin": 0}
        for _sym, out in results:
            m = out[label]
            for k in tot:
                tot[k] += m[k]
        wr = tot["nwin"] / tot["n"] * 100 if tot["n"] else 0.0
        lines.append(f"{label:<12}{tot['cum'] * SEED / 100:>+8.0f}"
                     f"{tot['mdd'] * SEED / 100:>7.0f}{wr:>5.0f}%{tot['n']:>7d}")
    txt = "\n".join(lines)
    with open("premium_discount_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
