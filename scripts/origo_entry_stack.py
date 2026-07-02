"""Origo 진입 엣지 조합(스택) 확인 — FST #1 자율연구 후속.

단변수 스윕(origo_entry_edge.py) 결과: min_confluence 4->5 로 흑자 전환(+58),
sl_dist_mult 넓힐수록 개선(4.0 -43), htf_align 4~5(-92), 킬존필터 필수.
단변수로 각각 개선된 축을 조합했을 때 시너지(흑자 확대)가 나는지, 아니면
빈도가 과도하게 죽는지(과최적화) 확인한다. 청산은 진입 순수비교 위해 현행
고정(tp_rr_override=1.0). 참고로 trail 2.0/1.5(RR1.9) 조합도 1개 병행.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/origo_entry_stack.py
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
# (라벨, override) — 단변수 흑자 축 조합
VARIANTS = [
    ("BASE(conf4/sl3/htf2)", {}),
    ("conf5", dict(min_confluence=5)),
    ("conf5+sl4", dict(min_confluence=5, sl_dist_mult=4.0)),
    ("conf5+sl4+htf4", dict(min_confluence=5, sl_dist_mult=4.0, htf_align_threshold=4)),
    ("conf6+sl4+htf4", dict(min_confluence=6, sl_dist_mult=4.0, htf_align_threshold=4)),
    ("conf5+sl4+htf4+trail2/1.5",
     dict(min_confluence=5, sl_dist_mult=4.0, htf_align_threshold=4,
          tp_rr_override=5.0, trail_trigger=2.0, trail_dist=1.5)),
]


def _metrics(trades):
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
    return dict(cum=cum, mdd=mdd, nwin=len(wins), n=len(ts),
                gw=sum(wins), gl=sum(losses))


def _eval(ov):
    tot = {"cum": 0.0, "mdd": 0.0, "nwin": 0, "n": 0, "gw": 0.0, "gl": 0.0, "days": 0.0}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        days = len(df5) / 288.0
        ttl = 12 if sym == "BTCUSDT" else 6
        cfg = {**BASE, "entry_ttl_bars": ttl, **ov}
        tl = cached_setup_timeline(df5, BacktestConfig(**cfg), sym)
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
        m = _metrics(bt.trades)
        if not m:
            continue
        for k in ("cum", "mdd", "nwin", "n", "gw", "gl"):
            tot[k] += m[k]
        tot["days"] += days
    n = tot["n"]
    wr = tot["nwin"] / n * 100 if n else 0.0
    aw = tot["gw"] / tot["nwin"] if tot["nwin"] else 0.0
    al = tot["gl"] / (n - tot["nwin"]) if (n - tot["nwin"]) else 0.0
    rr = aw / -al if al < 0 else 0.0
    freq = n / (tot["days"] / len(PAIRS)) if tot["days"] else 0
    return dict(usdt=tot["cum"] * SEED / 100, mdd=tot["mdd"] * SEED / 100,
                wr=wr, rr=rr, n=n, freq=freq)


def main() -> int:
    lines = ["===== Origo 진입 엣지 조합 확인 (7페어 5년, 시드1000) =====",
             f"{'방식':<28}{'USDT':>8}{'DD':>7}{'승률':>6}{'RR':>6}{'거래':>7}{'빈도':>7}"]
    for label, ov in VARIANTS:
        r = _eval(ov)
        line = (f"{label:<26}{r['usdt']:>+8.0f}{r['mdd']:>7.0f}{r['wr']:>5.0f}%"
                f"{r['rr']:>6.2f}{r['n']:>7d}{r['freq']:>6.2f}회")
        lines.append(line)
        print(f"{label} usdt={r['usdt']:+.0f} rr={r['rr']:.2f} n={r['n']} freq={r['freq']:.2f}", flush=True)
    txt = "\n".join(lines)
    with open("origo_entry_stack_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\nDONE (결과: origo_entry_stack_result.txt)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
