"""BE@1R 잠금 강건성 검증 — be_trigger 이웃 그리드 + 전/후반 walk-forward.

이익잠금 그리드에서 BE@1R+trail2/1.5 가 +278 로 배포 기준(+240) 상회.
단 BE@0.5(+162)·lock0.3(+178) 급락이라 (1.0, 0.0) 이 과최적화 봉우리인지 확인:
    - be_trigger 0.75/1.0/1.25/1.5 (lock 0 고정) 이웃 스캔
    - 최선 후보를 데이터 전반부/후반부 분리 재검 (walk-forward)
사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/be_lock_robust.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=5.0,
    trail_trigger=2.0, trail_dist=1.5,
)
TRIGGERS = [0.0, 0.75, 1.0, 1.25, 1.5]  # 0.0 = BE 없음(배포 기준)


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
    base_cfg = {**BASE, "entry_ttl_bars": ttl}
    tl = cached_setup_timeline(df5, BacktestConfig(**base_cfg), sym)
    half = len(df5) // 2
    out = {}
    for bt_r in TRIGGERS:
        cfg = BacktestConfig(**{**base_cfg, "be_trigger": bt_r, "be_lock": 0.0})
        bt = run_backtest_from_timeline(df5, tl, cfg)
        ts = list(bt.trades)
        out[f"BE@{bt_r}"] = _metrics(ts)
        # walk-forward: exit_idx 로 전/후반 분리
        out[f"BE@{bt_r}|전반"] = _metrics([t for t in ts if t.exit_idx < half])
        out[f"BE@{bt_r}|후반"] = _metrics([t for t in ts if t.exit_idx >= half])
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    lines = ["===== BE 잠금 강건성 (7페어 5년, trail2/1.5 고정, lock=0) =====",
             f"{'변형':<16}{'USDT':>8}{'DD':>7}{'거래':>7}   {'전반USDT':>9}{'후반USDT':>9}"]
    for bt_r in TRIGGERS:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0}
        h1 = h2 = 0.0
        for _sym, out in results:
            m = out[f"BE@{bt_r}"]
            tot["cum"] += m["cum"]; tot["mdd"] += m["mdd"]; tot["n"] += m["n"]
            h1 += out[f"BE@{bt_r}|전반"]["cum"]
            h2 += out[f"BE@{bt_r}|후반"]["cum"]
        label = "배포기준(BE없음)" if bt_r == 0.0 else f"BE@{bt_r}R"
        lines.append(f"{label:<14}{tot['cum'] * SEED / 100:>+8.0f}{tot['mdd'] * SEED / 100:>7.0f}"
                     f"{tot['n']:>7d}   {h1 * SEED / 100:>+9.0f}{h2 * SEED / 100:>+9.0f}")
    txt = "\n".join(lines)
    with open("be_lock_robust_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
