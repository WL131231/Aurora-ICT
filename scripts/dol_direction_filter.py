"""증류 3호 — draw-on-liquidity(DOL) 방향 필터 (FST #4 자율연구, 2026-07-07).

ICT 정통: 가격은 "아직 안 걷은 유동성"을 향해 간다 (ERL→IRL). 스윕 게이트
(#365)가 "스윕 직후 역방향 차단"이라면, 이건 상시 방향 선호를 룰로:
    최근에 걷힌(스윕된) 극단의 **반대쪽**이 draw → 그 방향만 진입 허용.
    - 직전 N일(10) 저점이 최근 K일 내 스윕됨 → draw = 위 → LONG only
    - 고점 스윕 → draw = 아래 → SHORT only
    - 양쪽 다/둘 다 아님 → 필터 없음 (기존 동작)
스윕 게이트와 차이: 반전 마감 조건 없음(스윕 사실만), 차단이 아니라 '선호'
(반대만 거름). K=3/5/10 스윕. 고정7 캐시 타임라인 재생만.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/dol_direction_filter.py
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
LOOKBACK_D = 10
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=5.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)
KDAYS = [3, 5, 10]


def _dol_filtered(tl, df5, k: int):
    """최근 K일 내 스윕된 극단의 반대 방향만 허용한 타임라인 복사."""
    from aurora_ict.strategy.silver_bullet import Direction
    d = df5.resample("1D").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    lo_n = d["low"].shift(1).rolling(LOOKBACK_D).min()
    hi_n = d["high"].shift(1).rolling(LOOKBACK_D).max()
    ssl_sweep = (d["low"] < lo_n).values      # 저점 스윕일 (반전 마감 조건 없음)
    bsl_sweep = (d["high"] > hi_n).values     # 고점 스윕일
    n = len(d)
    # 오늘 기준 "최근 K일(어제까지) 내 스윕" 플래그
    recent_ssl = np.zeros(n, dtype=bool)
    recent_bsl = np.zeros(n, dtype=bool)
    for i in range(n):
        a = max(0, i - k)
        recent_ssl[i] = ssl_sweep[a:i].any()
        recent_bsl[i] = bsl_sweep[a:i].any()
    day_of = d.index.searchsorted(df5.index.normalize(), side="right") - 1
    out = list(tl)
    for i, item in enumerate(out):
        if item is None or day_of[i] < 0:
            continue
        di = day_of[i]
        ssl_r, bsl_r = recent_ssl[di], recent_bsl[di]
        if ssl_r == bsl_r:      # 둘 다/둘 다 아님 — 필터 없음
            continue
        setup = item[0]
        if ssl_r and setup.direction is Direction.SHORT:   # draw=위 → 숏 제거
            out[i] = None
        elif bsl_r and setup.direction is Direction.LONG:  # draw=아래 → 롱 제거
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
    out = {"기준(필터없음)": _metrics(run_backtest_from_timeline(df5, tl, cfg).trades)}
    for k in KDAYS:
        ftl = _dol_filtered(tl, df5, k)
        out[f"DOL K={k}일"] = _metrics(run_backtest_from_timeline(df5, ftl, cfg).trades)
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    labels = ["기준(필터없음)"] + [f"DOL K={k}일" for k in KDAYS]
    lines = ["===== DOL 방향 필터 (7페어 5년, BASE=Origo 1.5 풀구성) =====",
             f"{'변형':<16}{'USDT':>8}{'DD':>7}{'승률':>6}{'거래':>7}"]
    for label in labels:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0, "nwin": 0}
        for _sym, out in results:
            m = out[label]
            for kk in tot:
                tot[kk] += m[kk]
        wr = tot["nwin"] / tot["n"] * 100 if tot["n"] else 0.0
        lines.append(f"{label:<14}{tot['cum'] * SEED / 100:>+8.0f}"
                     f"{tot['mdd'] * SEED / 100:>7.0f}{wr:>5.0f}%{tot['n']:>7d}")
    txt = "\n".join(lines)
    with open("dol_direction_filter_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
