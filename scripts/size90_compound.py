"""시드 90% 물량(×20배) 투입 — 복리+청산 시뮬 (파트너 요청, 2026-07-10).

"아예 시드의 90%를 투입하면?" 은 단리 합산으로 답하면 왜곡됨(고정 노셔널 단리는
DD>100% 가 수치로만 존재). 실제로는 판마다 시드가 변하고(복리), 큰 손절이
연속되면 계좌가 죽는다(청산). 7페어 트레이드를 청산시각 순으로 병합해 하나의
시드에 순차 적용:
    equity *= 1 + net_i × (m / 18)      # net_i 는 노셔널 18배(0.9×20) 기준
    equity ≤ 시드의 5% → 사망(청산) 처리
비교 배수 m: 18(=시드90%×20배, 요청안) / 9(45%) / 3.6(≈현행 리스크6% 평균 노셔널)
/ 1.8(9%). 근사 한계: 페어 동시 보유를 순차로 단순화(동시 손절 클러스터는 실제가
더 위험) — 즉 아래 결과는 90% 안에 유리한 편향.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/size90_compound.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BT_NOTIONAL = 18.0  # 재생 net 기준 노셔널 (0.9×20)
MULTS = [("시드90%×20배 (요청안, m=18)", 18.0),
         ("시드45%×20배 (m=9)", 9.0),
         ("현행 리스크6% 근사 (m≈3.6)", 3.6),
         ("보수 (m=1.8)", 1.8)]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl})
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    # (청산시각, net) — 시각은 df 인덱스로
    return [(df5.index[t.exit_idx], t.net_pnl_pct) for t in bt.trades]


def main() -> int:
    with Pool(4) as p:
        per_pair = p.map(_pair_worker, PAIRS)
    stream = sorted([x for lst in per_pair for x in lst], key=lambda x: x[0])
    lines = ["===== 시드 90% 물량 복리 시뮬 (7페어 5년 병합, 시드 1.0 시작) =====",
             f"트레이드 {len(stream)}건, 승률 {sum(1 for _t, n in stream if n > 0) / len(stream) * 100:.0f}%",
             "",
             f"{'방식':<26}{'최종 시드배수':>12}{'최대DD':>8}{'생존':>8}"]
    for label, m in MULTS:
        eq = 1.0
        peak = 1.0
        mdd = 0.0
        dead_at = None
        for ts, net in stream:
            eq *= (1.0 + net * (m / BT_NOTIONAL))
            peak = max(peak, eq)
            mdd = max(mdd, 1.0 - eq / peak)
            if eq <= 0.05:
                dead_at = ts
                break
        surv = f"사망 {str(dead_at)[:10]}" if dead_at else "생존"
        lines.append(f"{label:<24}{eq:>11.2f}x{mdd * 100:>7.0f}%{surv:>14}")
    txt = "\n".join(lines)
    with open("size90_compound_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
