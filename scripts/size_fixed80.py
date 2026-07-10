"""사이징 실험 — 점수 판단(40~90%) vs 80% 고정 (파트너 요청, 2026-07-10).

라이브 판단식: confluence 점수별 notional (0→40 / 1→55 / 2→70 / 3+→80~90%).
net 은 노셔널에 선형(수수료도 노셔널 비례)이라, 재생 트레이드의 net 을
size 비율로 재가중해 비교한다. 표기 = 시드% (단리, 20배 반영).

가설: conf5 게이트 하에선 전 트레이드 점수>=5 → 판단식이 항상 최고 구간
→ 고정과 사실상 동일할 것. 점수 분포도 함께 출력해 검증.

BASE = Origo 1.6 배포 구성. 7페어 5년, 캐시 재생.
사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/size_fixed80.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BT_SIZE = 0.9  # 재생 기준 노셔널 (재가중의 분모)
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=BT_SIZE,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)


def _score_size(score: int) -> float:
    """라이브 판단식 — 0→40% / 1→55% / 2→70% / 3+→80% (base40+step15, max80)."""
    return min(0.40 + 0.15 * max(0, score), 0.80)


def _agg(trades, size_fn):
    cum = peak = mdd = 0.0
    for t in sorted(trades, key=lambda t: t.exit_idx):
        cum += t.net_pnl_pct * (size_fn(t.confluence_score) / BT_SIZE)
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return cum, mdd


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl})
    tl = cached_setup_timeline(df5, cfg, sym)
    ts = list(run_backtest_from_timeline(df5, tl, cfg).trades)
    scores = Counter(t.confluence_score for t in ts)
    out = {}
    for label, fn in [
        ("판단식 40~80% (현행 로직)", _score_size),
        ("80% 고정", lambda s: 0.80),
        ("90% 고정 (백테 기준)", lambda s: 0.90),
    ]:
        out[label] = _agg(ts, fn)
    print(f"  {sym} done n={len(ts)}", flush=True)
    return sym, out, scores


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    all_scores = Counter()
    for _s, _o, sc in results:
        all_scores.update(sc)
    lines = ["===== 사이징: 점수 판단 vs 80% 고정 (7페어 5년, 시드% 단리, 20배 반영) =====",
             f"진입 트레이드 confluence 점수 분포: {dict(sorted(all_scores.items()))}",
             "",
             f"{'방식':<26}{'시드%(단리)':>12}{'최대DD(시드%)':>14}"]
    for label in ["판단식 40~80% (현행 로직)", "80% 고정", "90% 고정 (백테 기준)"]:
        cum = sum(o[label][0] for _s, o, _c in results)
        mdd = sum(o[label][1] for _s, o, _c in results)
        lines.append(f"{label:<24}{cum * 100:>+11.0f}%{mdd * 100:>13.0f}%")
    txt = "\n".join(lines)
    with open("size_fixed80_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
