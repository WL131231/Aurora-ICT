"""스윕-반전 bias 게이트 실험 — 7/1 사례 일반화 (FST #3 자율연구, 2026-07-07).

실측: 7/1 BTC 일봉이 6/25 저점(58043)을 스윕(57756)하고 강반전 마감 → 이후
+11% 상승. 그러나 봇(EMA align bias)은 6/26~7/4 숏만 치다 7/4 -165 전멸.
EMA 는 유동성 이벤트보다 3~5일 늦음 — ICT 정통은 "주요 저점 스윕+변위 반전 =
롱 bias 전환"을 즉시 읽는다.

실험: 일봉에서 스윕-반전일 감지 →
    - SSL 스윕(저점이 직전 N일 최저 하회 + 종가 회복): 이후 K일 SHORT 차단
    - BSL 스윕(고점이 직전 N일 최고 상회 + 종가 반락): 이후 K일 LONG 차단
타임라인 방향 필터(재생만)로 5년 검증. N=10, K=2/3/5.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/sweep_bias_gate.py
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
    trail_trigger=2.0, trail_dist=1.5,
)
KDAYS = [2, 3, 5]


def _sweep_flags(df5):
    """일봉 스윕-반전 감지 → 5m 인덱스별 (block_short, block_long) 마스크.

    SSL 스윕일: low < 직전 N일 최저 && close > (low + 0.5*(high-low)) (상반부 마감).
    BSL 스윕일: high > 직전 N일 최고 && close < (low + 0.5*(high-low)).
    다음날부터 K일 차단 (당일 진입은 유지 — 종가 확정 후 인지 가능).
    """
    d = df5.resample("1D").agg({"open": "first", "high": "max",
                                "low": "min", "close": "last"}).dropna()
    lo_n = d["low"].shift(1).rolling(LOOKBACK_D).min()
    hi_n = d["high"].shift(1).rolling(LOOKBACK_D).max()
    mid = d["low"] + (d["high"] - d["low"]) * 0.5
    ssl_sweep = (d["low"] < lo_n) & (d["close"] > mid)
    bsl_sweep = (d["high"] > hi_n) & (d["close"] < mid)
    return d.index, ssl_sweep.values, bsl_sweep.values


def _dir_filtered(tl, df5, days_idx, ssl, bsl, k: int):
    """스윕-반전 다음날부터 K일간 역방향 setup 제거한 타임라인 복사."""
    from aurora_ict.strategy.silver_bullet import Direction
    block_short = np.zeros(len(days_idx), dtype=bool)
    block_long = np.zeros(len(days_idx), dtype=bool)
    for i in range(len(days_idx)):
        if ssl[i]:
            block_short[i + 1:i + 1 + k] = True
        if bsl[i]:
            block_long[i + 1:i + 1 + k] = True
    day_of = days_idx.searchsorted(df5.index.normalize(), side="right") - 1
    out = list(tl)
    for i, item in enumerate(out):
        if item is None:
            continue
        di = day_of[i]
        if di < 0:
            continue
        setup = item[0]
        if setup.direction is Direction.SHORT and block_short[di]:
            out[i] = None
        elif setup.direction is Direction.LONG and block_long[di]:
            out[i] = None
    return out


def _metrics(bt_trades):
    ts = list(bt_trades)
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
    days_idx, ssl, bsl = _sweep_flags(df5)
    out = {"기준(게이트없음)": _metrics(run_backtest_from_timeline(df5, tl, cfg).trades)}
    for k in KDAYS:
        ftl = _dir_filtered(tl, df5, days_idx, ssl, bsl, k)
        out[f"스윕게이트 K={k}일"] = _metrics(run_backtest_from_timeline(df5, ftl, cfg).trades)
    print(f"  {sym} done (스윕일 SSL {int(ssl.sum())} / BSL {int(bsl.sum())})", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    labels = ["기준(게이트없음)"] + [f"스윕게이트 K={k}일" for k in KDAYS]
    lines = ["===== 스윕-반전 bias 게이트 (7페어 5년, BASE=Origo1.4+BE후보아님) =====",
             f"{'변형':<20}{'USDT':>8}{'DD':>7}{'승률':>6}{'거래':>7}"]
    for label in labels:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0, "nwin": 0}
        for _sym, out in results:
            m = out[label]
            for kk in tot:
                tot[kk] += m[kk]
        wr = tot["nwin"] / tot["n"] * 100 if tot["n"] else 0.0
        lines.append(f"{label:<18}{tot['cum'] * SEED / 100:>+8.0f}"
                     f"{tot['mdd'] * SEED / 100:>7.0f}{wr:>5.0f}%{tot['n']:>7d}")
    txt = "\n".join(lines)
    with open("sweep_bias_gate_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
