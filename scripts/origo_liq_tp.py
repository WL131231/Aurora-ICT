"""증류 4호 — 유동성 풀 타깃 TP vs R기하 청산 (FST #4, 파트너 최우선 관심).

정통 ICT: TP = 다음 미스윕 유동성(BSL/SSL). setup.take_profit 이 원래 이 값인데
라이브/백테 모두 tp_rr_override(R기하)로 덮어써 왔다 — override 를 벗기고(0.0)
정통 청산을 현행(트레일+BE)과 정면 비교. 하이브리드(유동성TP+트레일/BE)도 포함.

7페어 5년, rr2.0 캐시 타임라인 재생만.
사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/origo_liq_tp.py
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
    ote_level=0.707, min_rr=2.0,
)
VARIANTS = [
    ("기준: trail2/1.5+BE (배포 1.5)",
     dict(tp_rr_override=5.0, trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0)),
    ("유동성TP 단독 (정통)", dict(tp_rr_override=0.0)),
    ("유동성TP + BE@1R", dict(tp_rr_override=0.0, be_trigger=1.0, be_lock=0.0)),
    ("유동성TP + partial1R+BE", dict(tp_rr_override=0.0, partial_tp_rr=1.0,
                                     partial_be=True)),
    ("유동성TP + trail2/1.5 + BE",
     dict(tp_rr_override=0.0, trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0)),
]


def _metrics(trades):
    ts = list(trades)
    wins = [t.net_pnl_pct for t in ts if t.net_pnl_pct > 0]
    losses = [t.net_pnl_pct for t in ts if t.net_pnl_pct < 0]
    cum = peak = mdd = 0.0
    for t in sorted(ts, key=lambda t: t.exit_idx):
        cum += t.net_pnl_pct
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return dict(cum=cum, mdd=mdd, n=len(ts), nwin=len(wins),
                gw=sum(wins), gl=sum(losses))


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    base_cfg = {**BASE, "entry_ttl_bars": ttl}
    tl = cached_setup_timeline(df5, BacktestConfig(**base_cfg), sym)
    half = len(df5) // 2
    out = {}
    for label, ov in VARIANTS:
        cfg = BacktestConfig(**{**base_cfg, **ov})
        bt = run_backtest_from_timeline(df5, tl, cfg)
        m = _metrics(bt.trades)
        m["h1"] = sum(t.net_pnl_pct for t in bt.trades if t.exit_idx < half)
        m["h2"] = sum(t.net_pnl_pct for t in bt.trades if t.exit_idx >= half)
        out[label] = m
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    lines = ["===== 유동성 풀 타깃 TP vs R기하 (7페어 5년, conf5/SLx4/rr2.0) =====",
             f"{'변형':<28}{'USDT':>8}{'DD':>7}{'승률':>6}{'RR':>6}{'거래':>7}"
             f"{'전반':>8}{'후반':>8}"]
    for label, _ in VARIANTS:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0, "nwin": 0, "gw": 0.0, "gl": 0.0,
               "h1": 0.0, "h2": 0.0}
        for _sym, out in results:
            m = out[label]
            for k in tot:
                tot[k] += m[k]
        n = tot["n"]
        wr = tot["nwin"] / n * 100 if n else 0.0
        aw = tot["gw"] / tot["nwin"] if tot["nwin"] else 0.0
        al = tot["gl"] / (n - tot["nwin"]) if (n - tot["nwin"]) else 0.0
        rr = aw / -al if al < 0 else 0.0
        lines.append(f"{label:<26}{tot['cum'] * SEED / 100:>+8.0f}"
                     f"{tot['mdd'] * SEED / 100:>7.0f}{wr:>5.0f}%{rr:>6.2f}{n:>7d}"
                     f"{tot['h1'] * SEED / 100:>+8.0f}{tot['h2'] * SEED / 100:>+8.0f}")
    txt = "\n".join(lines)
    with open("origo_liq_tp_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
