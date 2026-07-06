"""이익 잠금(profit-capture) 그리드 — 파트너 4번 과제 (2026-07-07).

파트너: "수익 구간인데 TP 가 멀어 손절 터진다. TP 를 낮추면? (최대 40% ROI@20x
= 가격 2%)". MFE 분석과 짝 — 백테로 이익잠금 계열을 전수 비교:
    - 짧은 고정 TP (0.75R/1.0R/1.5R) — TP 자체를 낮추기
    - 조기 트레일 (0.5~1.5 trigger) — 활성화를 앞당기기
    - BE 잠금 (be_trigger/be_lock) — 트레일 유지 + 본전/이익 잠금만 추가
    - 분할 익절 + 트레일 하이브리드
R = SLx4 적용 후 거리(페어별 가격 ~1.2-3%) → 1R ≈ ROI@20x 24~60%.
BASE = 배포 Origo 1.4 (conf5/SLx4/rr2.0). 7페어 5년, 전부 재생(캐시).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/origo_profit_lock_grid.py
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
    ("기준: trail 2.0/1.5 (배포)", dict(tp_rr_override=5.0, trail_trigger=2.0, trail_dist=1.5)),
    # -- TP 낮추기 (고정) --
    ("고정TP 0.75R", dict(tp_rr_override=0.75)),
    ("고정TP 1.0R", dict(tp_rr_override=1.0)),
    ("고정TP 1.5R", dict(tp_rr_override=1.5)),
    # -- 조기 트레일 --
    ("trail 0.5/0.5", dict(tp_rr_override=5.0, trail_trigger=0.5, trail_dist=0.5)),
    ("trail 1.0/0.5", dict(tp_rr_override=5.0, trail_trigger=1.0, trail_dist=0.5)),
    ("trail 1.0/0.75", dict(tp_rr_override=5.0, trail_trigger=1.0, trail_dist=0.75)),
    ("trail 1.0/1.0", dict(tp_rr_override=5.0, trail_trigger=1.0, trail_dist=1.0)),
    # -- BE 잠금 + 배포 트레일 유지 --
    ("BE@0.5R + trail2/1.5", dict(tp_rr_override=5.0, trail_trigger=2.0, trail_dist=1.5,
                                  be_trigger=0.5, be_lock=0.0)),
    ("BE@1R + trail2/1.5", dict(tp_rr_override=5.0, trail_trigger=2.0, trail_dist=1.5,
                                be_trigger=1.0, be_lock=0.0)),
    ("BE@1R lock0.3 + trail2/1.5", dict(tp_rr_override=5.0, trail_trigger=2.0,
                                        trail_dist=1.5, be_trigger=1.0, be_lock=0.3)),
    # -- 분할 + 트레일 하이브리드 --
    ("partial 1.0R+BE + trail2/1.5", dict(tp_rr_override=5.0, trail_trigger=2.0,
                                          trail_dist=1.5, partial_tp_rr=1.0,
                                          partial_be=True)),
    ("partial 0.75R+BE + trail2/1.5", dict(tp_rr_override=5.0, trail_trigger=2.0,
                                           trail_dist=1.5, partial_tp_rr=0.75,
                                           partial_be=True)),
]


def _metrics(bt):
    ts = list(bt.trades)
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
    out = {"_days": len(df5) / 288.0}
    for label, ov in VARIANTS:
        cfg = BacktestConfig(**{**base_cfg, **ov})
        out[label] = _metrics(run_backtest_from_timeline(df5, tl, cfg))
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    days_tot = sum(out["_days"] for _s, out in results)
    lines = ["===== Origo 이익잠금 그리드 (7페어 5년, ROI@20x: 1R≈24~60%) =====",
             "BASE = conf5/SLx4/rr2.0.",
             "",
             f"{'변형':<30}{'USDT':>8}{'DD':>7}{'승률':>6}{'RR':>6}{'거래':>7}"]
    for label, _ in VARIANTS:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0, "nwin": 0, "gw": 0.0, "gl": 0.0}
        for _sym, out in results:
            m = out.get(label)
            if m:
                for k in tot:
                    tot[k] += m[k]
        n = tot["n"]
        wr = tot["nwin"] / n * 100 if n else 0.0
        aw = tot["gw"] / tot["nwin"] if tot["nwin"] else 0.0
        al = tot["gl"] / (n - tot["nwin"]) if (n - tot["nwin"]) else 0.0
        rr = aw / -al if al < 0 else 0.0
        lines.append(f"{label:<28}{tot['cum'] * SEED / 100:>+8.0f}"
                     f"{tot['mdd'] * SEED / 100:>7.0f}{wr:>5.0f}%{rr:>6.2f}{n:>7d}")
    txt = "\n".join(lines)
    with open("origo_profit_lock_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
