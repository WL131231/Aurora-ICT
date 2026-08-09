"""분할익절 체감승률 검증 — 라이브 후보(0.707/swing/횡보회피33%)에 partial_tp 얹기.

파트너 우려(6/23): 승률30%면 상용 판매 시 "연속 빨간색 화면" 때문에 구독 이탈.
실제 승률은 못올리나(작은TP=net적자 증명), 분할익절+본전이동(partial_be)으로
TP1(1R) 닿는 거래를 양수(초록)로 전환 → 체감승률↑ + 연속손절↓. net은 runner
50%가 보존. partial 변형별로 체감지표 측정:
  - 순손실거래%(net<0=빨강), 큰손실거래%(net<-0.5%=진한빨강)
  - 페어별 max 연속손절(연속 빨강 최대), net USDT, DD
비교: off(현행후보) vs partial[1.0/0.5/1.5]+be. 7페어, 횡보회피33% 후처리. 시드1000.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/partial_feel.py
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
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3, entry_ttl_bars=6,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)
# (라벨, partial_tp_rr, partial_be)
VARIANTS = [
    ("off(현행후보)", 0.0, False),
    ("분할1.0/be", 1.0, True),
    ("분할0.5/be", 0.5, True),
    ("분할1.5/be", 1.5, True),
]


def _stats(trades):
    """net, mdd, n, 순손실%, 큰손실%, max연속손절, 비손실(체감승)%"""
    ts = sorted(trades, key=lambda t: t.exit_idx)
    if not ts:
        return [0.0] * 7
    cum = peak = mdd = 0.0
    nloss = nbigloss = 0
    streak = maxstreak = 0
    for t in ts:
        p = t.net_pnl_pct
        cum += p
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
        if p < 0:
            nloss += 1
            streak += 1
            maxstreak = max(maxstreak, streak)
        else:
            streak = 0
        if p < -0.5:
            nbigloss += 1
    n = len(ts)
    return [cum, mdd, n, nloss / n * 100, nbigloss / n * 100, maxstreak, (n - nloss) / n * 100]


def main() -> int:
    # 라벨 -> 합산 [net, mdd, n, lossN, biglossN, maxstreak합, 비손실N], days
    agg = {v[0]: [0.0, 0.0, 0, 0, 0, 0, 0] for v in VARIANTS}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        detect_cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR}
        tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), sym)
        for label, ptp, pbe in VARIANTS:
            cfg = {**BASE, "ote_level": OTE, "min_rr": MIN_RR, "tp_rr_override": 0.0,
                   "partial_tp_rr": ptp, "partial_be": pbe}
            trades = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg)).trades
            if len(trades) < 9:
                continue
            # 횡보회피33% (이 페어 |entry_trend| 하위33% 차단)
            thr = sorted(abs(t.entry_trend_pct) for t in trades)[len(trades) // 3]
            sub = [t for t in trades if abs(t.entry_trend_pct) >= thr]
            net, mdd, n, lp, blp, mstk, feelwr = _stats(sub)
            a = agg[label]
            a[0] += net; a[1] += mdd; a[2] += n
            a[3] += round(lp * n / 100); a[4] += round(blp * n / 100)
            a[5] += mstk; a[6] += round(feelwr * n / 100)
        print(f"  {sym} done", flush=True)

    npair = len(PAIRS)
    lines = ["===== 분할익절 체감승률 검증 (0.707/swing/횡보회피33%, 7페어, 시드1000) =====",
             f"  {'변형':<14} {'USDT':>7} {'최대DD':>7} {'체감승률':>8} {'순손실%':>8} {'큰손실%':>8} {'연속손절':>8}"]
    for label, _, _ in VARIANTS:
        net, mdd, n, lossN, biglossN, mstk, feelN = agg[label]
        if not n:
            continue
        feelwr = feelN / n * 100
        lossp = lossN / n * 100
        bigp = biglossN / n * 100
        avgstk = mstk / npair
        lines.append(f"  {label:<14} {net * SEED / 100:+7.0f} {mdd * SEED / 100:6.0f}↓ {feelwr:7.0f}% {lossp:7.0f}% {bigp:7.0f}% {avgstk:6.1f}회")
    lines.append("\n※ 체감승률=비손실거래%(초록). 분할익절이 체감승률↑·큰손실%↓·연속손절↓ 면 이탈방어 효과.")
    lines.append("  연속손절=페어별 max 연속 빨강의 평균(낮을수록 화면 덜 무섭다).")

    txt = "\n".join(lines)
    with open("partial_feel_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 partial_feel_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
