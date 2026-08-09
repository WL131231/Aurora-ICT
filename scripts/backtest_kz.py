"""킬존별 봇 강점 — 백테 5년 trade 진입 시각을 킬존 분류. 캐시 재사용(재빌드 X).

파트너(6/24): 우리 봇이 어느 킬존 시간대에 강한지. 백테 BASE 는 sub 윈도우(킬존)
에만 진입하므로 trade 진입시각(entry_idx→df.index, UTC)을 ICT 킬존으로 분류해
킬존별 net/승률/빈도 집계. 라이브 후보(0.707/분할1.5/be) 청산 기준. 0.707 캐시
재사용이라 24h 재빌드 불필요. 라이브(표본작음)보다 5년 대표본이라 신뢰성↑.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True, ote_level=0.707,
    min_confluence=4, min_rr=2.0, sl_dist_mult=3.0, setup_stale_bars=3, entry_ttl_bars=6,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    partial_tp_rr=1.5, partial_be=True,
)


def kz(h: int) -> str:
    # 봇 UI 킬존(KST=UTC+9). London 16-18KST=07-09UTC, NY_AM 20-22=11-13,
    # LDN마감 23-01=14-16, NY_PM 02-05=17-20, Asian 08-12:50=23-03.
    if 7 <= h < 10:
        return "London(16-18KST)"
    if 11 <= h < 13:
        return "NY_AM(20-22KST)"
    if 14 <= h < 16:
        return "LDN_Close(23-01KST)"
    if 17 <= h < 21:
        return "NY_PM(02-05KST)"
    if h >= 23 or h < 4:
        return "Asian(08-12KST)"
    return "기타(장간)"


ORDER = ["Asian(08-12KST)", "London(16-18KST)", "NY_AM(20-22KST)",
         "LDN_Close(23-01KST)", "NY_PM(02-05KST)", "기타(장간)"]


def main() -> int:
    agg = {k: [0.0, 0, 0] for k in ORDER}  # net, n, nwin
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        tl = cached_setup_timeline(df5, BacktestConfig(**BASE), sym)
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**BASE))
        for t in bt.trades:
            h = df5.index[t.entry_idx].hour
            k = kz(h)
            a = agg[k]
            a[0] += t.net_pnl_pct
            a[1] += 1
            if t.net_pnl_pct > 0:
                a[2] += 1
        print(f"  {sym} done ({len(bt.trades)})", flush=True)

    total_n = sum(a[1] for a in agg.values())
    lines = ["===== 킬존별 봇 강점 (백테 5년, 0.707/분할1.5/be, 7페어) =====",
             f"{'킬존':<20} {'net%':>9} {'거래':>6} {'비중':>6} {'승률':>6} {'평균':>7}"]
    for k in ORDER:
        net, n, nwin = agg[k]
        if n == 0:
            continue
        wr = nwin / n * 100
        share = n / total_n * 100 if total_n else 0
        lines.append(f"{k:<20} {net:+9.1f} {n:6d} {share:5.0f}% {wr:5.0f}% {net / n:+7.3f}")
    lines.append(f"\n총 {total_n}거래. ※ net·평균·승률 높은 킬존 = 봇이 강한 장(5년 대표본).")
    lines.append("  라이브(Origo1.1 표본작음): NY_PM 최악·비킬존 흑자였음 — 백테 대조 확인.")

    txt = "\n".join(lines)
    with open("backtest_kz_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE")
    except UnicodeEncodeError:
        print("(결과는 backtest_kz_result.txt)\nDONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
