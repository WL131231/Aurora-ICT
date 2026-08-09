"""#FST5 2026-07-16: 24시간 모드(referral) 킬존별 5년 백테 — Asian 게이트 검증.

라이브 진단: Origo 1.8 referral(24h) 유저가 Asian 킬존서 승률 0%/-47.
1.7 검증 BASE 는 disable_time_filter=False(킬존만)라 Asian 진입이 아예 없었음 —
즉 24h 모드는 무검증 구간. 여기선 disable_time_filter=True(24h)로 라이브 referral
동작을 재현, 전 진입을 킬존 분류해 Asian 이 5년 대표본서도 적자인지 확인한다.

통과 조건: Asian net 음수 & 제거 시 총 EV 개선 → Origo 1.9(referral 도 Asian 차단).
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

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
# 1.7 검증 BASE 그대로 — 단 disable_time_filter=True(24h referral 재현).
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=True, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
)


def kz(h: int) -> str:
    # UTC hour → 봇 UI 킬존(KST=UTC+9). backtest_kz.py 와 동일 매핑.
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
    agg = {k: [0.0, 0, 0] for k in ORDER}  # net%, n, nwin
    total_net = 0.0
    total_n = 0
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        for t in bt.trades:
            h = df5.index[t.entry_idx].hour
            a = agg[kz(h)]
            a[0] += t.net_pnl_pct
            a[1] += 1
            if t.net_pnl_pct > 0:
                a[2] += 1
            total_net += t.net_pnl_pct
            total_n += 1
        print(f"  {sym} done ({len(bt.trades)})")

    print("\n===== 24h 모드(referral) 킬존별 5년 백테 (1.7 BASE, 7페어) =====")
    print(f"{'킬존':<24}{'net%':>8}{'거래':>7}{'승률':>7}{'평균%':>9}")
    asian_net = asian_n = 0
    for k in ORDER:
        net, n, nwin = agg[k]
        if n == 0:
            continue
        wr = 100 * nwin / n
        avg = net / n
        print(f"{k:<24}{net:>+8.1f}{n:>7}{wr:>6.0f}%{avg:>+9.3f}")
        if k.startswith("Asian"):
            asian_net, asian_n = net, n
    print(f"\n총 {total_n}거래 net={total_net:+.1f}%")
    print(f"Asian 제거 시: net={total_net - asian_net:+.1f}% "
          f"(Asian {asian_n}건 {asian_net:+.1f}% 제거)")
    print("판정: Asian net 음수 & 제거 시 총 net 개선 → Asian 게이트 정당")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
