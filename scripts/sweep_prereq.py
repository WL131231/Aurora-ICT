"""증류 5호 — 선행 스윕 '필수 조건' 승격 실험 (FST #4, 2026-07-08).

정통 실버불릿 시퀀스는 [유동성 스윕 확인 → MSS → 첫 FVG] 순서가 전제.
현행은 sweep 이 confluence 가점(+1)일 뿐 필수가 아님 — sweep 성분이 없는
setup 을 제거(전제조건 승격)하면 정통에 가까워지는데 net 이 지켜지는지 검증.
비교로 CISD(MSS 계열) 필수 승격, 둘 다 필수도 함께.

타임라인 setup.confluences 문자열 필터 → 재생만(캐시).
사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/sweep_prereq.py
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
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)
FILTERS = [
    ("기준(가점제)", None),
    ("sweep 필수", ["sweep"]),
    ("cisd 필수", ["cisd"]),
    ("sweep+cisd 필수", ["sweep", "cisd"]),
]


def _req_filtered(tl, required: list[str]):
    out = list(tl)
    for i, item in enumerate(out):
        if item is None:
            continue
        confs = " ".join(str(c).lower() for c in (item[0].confluences or []))
        if not all(r in confs for r in required):
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
    for label, req in FILTERS:
        ftl = tl if req is None else _req_filtered(tl, req)
        out[label] = _metrics(run_backtest_from_timeline(df5, ftl, cfg).trades)
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    lines = ["===== 선행 스윕/CISD 필수 승격 (7페어 5년, BASE=Origo 1.5) =====",
             f"{'변형':<18}{'USDT':>8}{'DD':>7}{'승률':>6}{'거래':>7}"]
    for label, _req in FILTERS:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0, "nwin": 0}
        for _sym, out in results:
            m = out[label]
            for k in tot:
                tot[k] += m[k]
        wr = tot["nwin"] / tot["n"] * 100 if tot["n"] else 0.0
        lines.append(f"{label:<16}{tot['cum'] * SEED / 100:>+8.0f}"
                     f"{tot['mdd'] * SEED / 100:>7.0f}{wr:>5.0f}%{tot['n']:>7d}")
    txt = "\n".join(lines)
    with open("sweep_prereq_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
