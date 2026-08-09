"""타점(진입깊이 0.886) robust 검증 — 0.5(현행) vs 0.886 전·후반 일관성.

entry_depth(최근절반) 결과: ote 0.886 이 net+17.9·승률30.4%·손절69.6% 로 현행
0.5(+12.4·28.3%·71.7%) 대비 전면 우위. 단 최근절반이라 min_rr 처럼 시기 의존일
수 → 전·후반 모두 0.886 우위면 robust(배포 가치). ote_level 은 detect 인자라
재빌드(2레벨×전후반×7페어=28빌드). cisd+po3·conf4·rr2.5·ttl6·sl x3·킬존.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/ote_depth_robust.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import (  # noqa: E402
    BacktestConfig,
    run_backtest_from_timeline,
)

# DOGE 제외 — 전반 0.886 timeline 빌드 무한루프(detect 버그 의심, 별도 규명). 6페어로 robust.
PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "LINKUSDT", "HYPEUSDT"]
OTES = [0.5, 0.886]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def _pair(sym):
    df5 = _resample(_load_full(sym))
    if len(df5) < 1400:
        return []
    half = len(df5) // 2
    ttl = 12 if sym == "BTCUSDT" else 6
    out = []
    for seg_label, seg in (("전반", df5.iloc[:half]), ("후반", df5.iloc[half:])):
        for ote in OTES:
            cfg = {**BASE, "ote_level": ote, "entry_ttl_bars": ttl}
            tl = cached_setup_timeline(seg, BacktestConfig(**cfg), sym)
            bt = run_backtest_from_timeline(seg, tl, BacktestConfig(**cfg))
            n = len(bt.trades)
            net = sum(t.net_pnl_pct for t in bt.trades)
            nwin = sum(1 for t in bt.trades if t.net_pnl_pct > 0)
            out.append((seg_label, ote, net, nwin, n))
    return out


def main() -> int:
    rows = []
    for sym in PAIRS:
        rows.extend(_pair(sym))
        print(f"  {sym} done", flush=True)

    lines = ["===== 타점 0.886 robust (0.5 vs 0.886 전·후반, 7페어) ====="]
    best = {}
    for seg in ("전반", "후반"):
        lines.append(f"\n--- {seg} ---")
        agg = {}
        for sl, ote, net, nwin, n in rows:
            if sl != seg:
                continue
            a = agg.setdefault(ote, [0.0, 0, 0])
            a[0] += net; a[1] += nwin; a[2] += n
        bn, bote = -1e9, None
        for ote in OTES:
            net, nwin, n = agg.get(ote, [0.0, 0, 0])
            wr = (nwin / n * 100) if n else 0.0
            slr = ((n - nwin) / n * 100) if n else 0.0
            mark = " ←현행" if ote == 0.5 else ""
            lines.append(f"  ote{ote}: net{net:+7.1f} 승{wr:4.1f}% 손절{slr:4.1f}% 거래{n:4d}{mark}")
            if net > bn:
                bn, bote = net, ote
        best[seg] = bote
        lines.append(f"  → {seg} 최적 ote{bote}")

    lines.append("\n===== robust 판정 =====")
    ok = best.get("전반") == best.get("후반") == 0.886
    lines.append(f"  전반 ote{best.get('전반')} / 후반 ote{best.get('후반')}")
    if ok:
        lines.append("  → ✅ 0.886 전후반 robust 우위! 진입 깊이 0.886 배포 가치 (타점 개선 확정).")
    elif best.get("전반") == best.get("후반"):
        lines.append(f"  → 전후반 일치(ote{best.get('전반')})이나 0.886 아님.")
    else:
        lines.append("  → ⚠ 불일치 — min_rr 처럼 시기 의존. 신중.")

    txt = "\n".join(lines)
    with open("ote_depth_robust_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 ote_depth_robust_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
