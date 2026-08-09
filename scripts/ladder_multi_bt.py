"""다단계 분할TP 백테 — 현재(1.5R 1단계) vs A(고정ROI 10/20/30/40) vs B(swing 4등분).

파트너(6/24): Dual ST 4분할TP 차용 검토. replay 의 ladder_tp(이미 구현)로 두 방식 비교:
- A 고정 ROI%: ladder_mode=pnl, levels=10/20/30/40(레버20x → 가격 0.5/1/1.5/2%), alloc 25%×4
- B swing 4등분: ladder_mode=tpfrac, fracs=0.25/0.5/0.75/1.0(진입~원TP 거리 25%씩)
둘 다 1단계 도달 후 본전 근사(be_after=1). 현재 1.5R/be 와 net/체감승률/RR/연속손절 비교.
0.707 캐시 timeline 재사용(청산만 다름). 시드 1000.
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
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True, ote_level=0.707,
    min_confluence=4, min_rr=2.0, sl_dist_mult=3.0, setup_stale_bars=3, entry_ttl_bars=6,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9, leverage=20.0,
)
VARIANTS = [
    ("현재 1.5R/be", dict(tp_rr_override=0.0, partial_tp_rr=1.5, partial_be=True)),
    # 옵션1: 20% ROI(20x→가격1%)에서 전량 청산 — 최대 TP 고정.
    ("옵션1 TP20%고정전량", dict(
        tp_rr_override=0.0, ladder_tp=True, ladder_mode="pnl",
        ladder_levels_pnl=(20.0,), ladder_alloc=(1.0,),
        ladder_be_after=99, ladder_be_pnl=0.0)),
    # 옵션2: 20%서 50% 반익 + SL 을 본전+4%ROI(가격0.2%)로 + 나머지 50% swing TP 유지.
    ("옵션2 20%반익+본전4%+swing", dict(
        tp_rr_override=0.0, ladder_tp=True, ladder_mode="pnl",
        ladder_levels_pnl=(20.0,), ladder_alloc=(0.5,),
        ladder_be_after=1, ladder_be_pnl=4.0)),
]


def _stats(trades):
    ts = sorted(trades, key=lambda t: t.exit_idx)
    if not ts:
        return [0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0]
    cum = peak = mdd = 0.0
    nloss = streak = maxstreak = 0
    gw = gl = 0.0
    nwin = 0
    for t in ts:
        p = t.net_pnl_pct
        cum += p
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
        if p > 0:
            nwin += 1; gw += p; streak = 0
        else:
            nloss += 1; streak += 1; maxstreak = max(maxstreak, streak); gl += abs(p)
    n = len(ts)
    return [cum, mdd, n, nwin, maxstreak, n - nloss, gw / nwin if nwin else 0,
            gl / nloss if nloss else 0]


def main() -> int:
    agg = {v[0]: [0.0, 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0] for v in VARIANTS}  # +days
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        days = len(df5) / 288.0
        tl = cached_setup_timeline(df5, BacktestConfig(**{**BASE, "tp_rr_override": 0.0}), sym)
        for label, extra in VARIANTS:
            bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**{**BASE, **extra}))
            net, mdd, n, nwin, mstk, feel, aw, al = _stats(bt.trades)
            a = agg[label]
            a[0] += net; a[1] += mdd; a[2] += n; a[3] += nwin
            a[4] += mstk; a[5] += feel; a[6] += aw; a[7] += al; a[8] += days
        print(f"  {sym} done", flush=True)

    lines = ["===== 다단계 분할TP 비교 (0.707/횡보전, 7페어, 시드1000) =====",
             f"{'변형':<22} {'USDT':>7} {'최대DD':>7} {'체감승률':>8} {'연속손절':>8} {'RR':>5} {'거래':>6}"]
    for label, _ in VARIANTS:
        net, mdd, n, nwin, mstk, feel, aw, al, days = agg[label]
        feelwr = feel / n * 100 if n else 0
        rr = (aw / len([1])) / (al / len([1])) if al else 0  # aw,al 은 페어합산 평균합
        rr = aw / al if al else 0
        avgstk = mstk / len(PAIRS)
        lines.append(f"{label:<22} {net * SEED / 100:+7.0f} {mdd * SEED / 100:6.0f}↓ "
                     f"{feelwr:7.0f}% {avgstk:6.1f}회 {rr:5.2f} {n:6d}")
    lines.append("\n※ net 흑자 유지하며 체감승률↑·연속손절↓·RR 보존(고래 교훈) 인 방식 채택.")
    lines.append("  A=고정ROI(가격 0.5/1/1.5/2%) / B=원TP 4등분(RR 비례). 둘 다 1단계후 본전근사.")

    txt = "\n".join(lines)
    with open("ladder_multi_bt_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE")
    except UnicodeEncodeError:
        print("(결과는 ladder_multi_bt_result.txt)\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
