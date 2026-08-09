"""횡보 임계 라이브 이식 검증 — q33(사후분위) vs 고정 절대임계 / 페어별 효과.

백테 횡보컷=페어별 |entry_trend| 하위33% 분위(사후계산, 라이브 불가). 라이브용
대안: ①고정 공통 절대임계(|trend|<X% 차단) ②페어별 고정임계. 어느 게 q33 후처리
만큼 net흑자+DD↓+체감승률 내는지 검증. 분할1.0/be 포함(최종후보 구성). 먼저 각
페어 q33 절대값을 출력(고정임계 후보 가늠) → 고정임계 스윕.

  최종후보(0.707/swing/분할1.0-be) × 횡보컷 [q33 / 고정 0.10/0.15/0.20/0.25%].

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/regime_live_thresh.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
OTE = 0.707
MIN_RR = 2.0
FIXED = [0.10, 0.15, 0.20, 0.25]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3, entry_ttl_bars=6,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    partial_tp_rr=1.0, partial_be=True,
)


def _stats(trades):
    ts = sorted(trades, key=lambda t: t.exit_idx)
    if not ts:
        return [0.0, 0.0, 0, 0, 0]
    cum = peak = mdd = 0.0
    nloss = 0
    streak = maxstreak = 0
    for t in ts:
        p = t.net_pnl_pct
        cum += p
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
        if p < 0:
            nloss += 1; streak += 1; maxstreak = max(maxstreak, streak)
        else:
            streak = 0
    n = len(ts)
    return [cum, mdd, n, (n - nloss) / n * 100, maxstreak]


def main() -> int:
    keys = ["none", "q33"] + [f"fix{f}" for f in FIXED]
    agg = {k: [0.0, 0.0, 0, 0, 0] for k in keys}  # net,mdd,n,feelN,streak합
    q33vals = {}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        detect_cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR}
        tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), sym)
        cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR, "tp_rr_override": 0.0}
        trades = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg)).trades
        if len(trades) < 9:
            continue
        absten = sorted(abs(t.entry_trend_pct) for t in trades)
        q33 = absten[len(trades) // 3]
        q33vals[sym] = q33
        cuts = {"none": -1.0, "q33": q33, **{f"fix{f}": f for f in FIXED}}
        for k, thr in cuts.items():
            sub = [t for t in trades if abs(t.entry_trend_pct) >= thr]
            net, mdd, n, feel, stk = _stats(sub)
            a = agg[k]
            a[0] += net; a[1] += mdd; a[2] += n; a[3] += round(feel * n / 100); a[4] += stk
        print(f"  {sym} q33={q33:.3f} done", flush=True)

    npair = len(PAIRS)
    lines = ["===== 횡보 임계 라이브 이식 (0.707/swing/분할1.0-be, 7페어, 시드1000) =====",
             "[페어별 q33 절대 |추세%| 값] (고정임계 후보 가늠)"]
    for sym in PAIRS:
        if sym in q33vals:
            lines.append(f"  {sym:<10} {q33vals[sym]:.3f}%")
    lines.append(f"\n  {'횡보컷':<10} {'USDT':>7} {'최대DD':>7} {'체감승률':>8} {'연속손절':>8} {'거래':>6}")
    for k in keys:
        net, mdd, n, feelN, stk = agg[k]
        if not n:
            continue
        label = "횡보생략" if k == "none" else ("q33(사후)" if k == "q33" else k.replace("fix", "고정") + "%")
        lines.append(f"  {label:<10} {net * SEED / 100:+7.0f} {mdd * SEED / 100:6.0f}↓ {feelN / n * 100:7.0f}% {stk / npair:6.1f}회 {n:6d}")
    lines.append("\n※ 고정임계가 q33(사후)만큼 net흑자+체감승률+연속손절↓ 면 라이브 이식 가능(고정값 채택).")
    lines.append("  q33값이 페어마다 크게 다르면 → 고정 공통은 부적합, 페어별 임계나 ATR정규화 필요.")

    txt = "\n".join(lines)
    with open("regime_live_thresh_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 regime_live_thresh_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
