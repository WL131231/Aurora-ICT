"""타점(진입 깊이) 연구 — ote_level 별 net·승률·SL률·체결률.

파트너 진단(6/19): 방향은 맞는데 ①타점 ②먼TP ③구조 중 하나가 문제. 타점 검증.
현행 라이브 = FVG mean(ote_level 0.5) 진입. 더 깊은 OTE(0.62~0.79) 되돌림에서
진입하면 더 좋은 가격 → SL 여유·RR↑·SL피격↓? 단 깊이 안 닿으면 미체결(체결률↓).
ote_level 은 setup entry 결정(detect)이라 timeline 재빌드. 7페어 5년 Origo 1.1.

  ote_level: 0.5(현행 mean) / 0.62 / 0.705(OTE sweet) / 0.79
각 net·승률·거래수(체결률)·SL_HIT 비율. 깊을수록 승률↑·SL률↓·체결↓ 기대.
net 유지+승률↑+SL률↓면 타점 개선 → silver_bullet 진입 깊이 옵션 추가 가치.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/entry_depth_research.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402

from aurora_ict.backtest.replay import (  # noqa: E402
    BacktestConfig,
    build_setup_timeline,
    run_backtest_from_timeline,
)

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
OTE_LEVELS = [0.5, 0.618, 0.707, 0.786, 0.886]  # 핵심 5레벨(1차 0.5/0.618 + 2차 0.707~0.886)
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def _pair(sym):
    df5 = _resample(_load_full(sym))
    if len(df5) < 1400:
        return (sym, {}, 1.0)
    df5 = df5.iloc[len(df5) // 2:]  # 최근 절반(라이브 관련 + 빌드 시간 절반)
    days = len(df5) / 288.0
    ttl = 12 if sym == "BTCUSDT" else 6
    out = {}
    for ote in OTE_LEVELS:
        # ote_level 은 entry 위치(detect) 결정 → ote 마다 timeline 빌드.
        cfg = {**BASE, "ote_level": ote, "entry_ttl_bars": ttl}
        tl = build_setup_timeline(df5, BacktestConfig(**cfg))
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
        n = len(bt.trades)
        net = sum(t.net_pnl_pct for t in bt.trades)
        nwin = sum(1 for t in bt.trades if t.net_pnl_pct > 0)
        # SL_HIT 비율: net<0 비율로 근사 (outcome 미노출 시)
        nloss = n - nwin
        out[ote] = (net, nwin, n, nloss)
    return (sym, out, days)


def main() -> int:
    rows = []
    for sym in PAIRS:
        rows.append(_pair(sym))
        print(f"  [{len(rows)}/{len(PAIRS)}] {sym} done", flush=True)
    total_days = sum(d for _, o, d in rows if o)
    agg = {ote: [0.0, 0, 0, 0] for ote in OTE_LEVELS}
    for _, out, _ in rows:
        for ote, (net, nwin, n, nloss) in out.items():
            a = agg[ote]
            a[0] += net; a[1] += nwin; a[2] += n; a[3] += nloss

    lines = ["===== 타점(진입깊이 ote_level) 연구 (7페어 5년 Origo 1.1) ====="]
    lines.append(f"{'ote_level':<12} {'net%':>8} {'승률':>6} {'거래':>6} {'1일빈도':>8} {'손절률':>7}")
    for ote in OTE_LEVELS:
        net, nwin, n, nloss = agg[ote]
        wr = (nwin / n * 100) if n else 0.0
        freq = n / total_days * 7 if total_days else 0.0
        slr = (nloss / n * 100) if n else 0.0
        label = f"{ote}(현행)" if ote == 0.5 else f"{ote}"
        lines.append(f"{label:<12} {net:+8.1f} {wr:5.1f}% {n:5d} {freq:7.2f}회 {slr:6.1f}%")

    lines.append("\n※ 깊을수록(0.79) entry 가격 유리 → 승률↑·손절률↓ 기대, 체결(빈도)↓.")
    lines.append("  net 유지 + 승률↑ + 손절률↓ 면 '타점 개선' → silver_bullet 진입깊이 옵션 추가 가치.")
    base_net = agg[0.5][0]
    base_wr = (agg[0.5][1] / agg[0.5][2] * 100) if agg[0.5][2] else 0.0
    lines.append(f"\n[현행(0.5) 대비]")
    for ote in OTE_LEVELS:
        if ote == 0.5:
            continue
        net, nwin, n, nloss = agg[ote]
        wr = (nwin / n * 100) if n else 0.0
        lines.append(f"  {ote}: net {net - base_net:+.1f}%p, 승률 {wr - base_wr:+.1f}%p")

    txt = "\n".join(lines)
    with open("entry_depth_research_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 entry_depth_research_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
