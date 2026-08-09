"""#FST5 2026-07-16: 시간필터 A/B — 동일 config, disable_time_filter만 토글.

killzone 변수 격리: 다른 파라미터 전부 고정하고 disable_time_filter 만 True(24h)/
False(킬존한정) 두 번 돌려, 24h 가 추가로 무는 시간대(Asian/off/NY_PM)가
순엣지를 갉아먹는지 직접 비교. 킬존별 분해 동반.
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
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
)


def kz(h: int) -> str:
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
    return "off(장간)"


ORDER = ["Asian(08-12KST)", "London(16-18KST)", "NY_AM(20-22KST)",
         "LDN_Close(23-01KST)", "NY_PM(02-05KST)", "off(장간)"]


def run(dtf: bool):
    agg = {k: [0.0, 0, 0] for k in ORDER}
    tot_net = tot_n = 0
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**{**BASE, "disable_time_filter": dtf})
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        for t in bt.trades:
            h = df5.index[t.entry_idx].hour
            a = agg[kz(h)]
            a[0] += t.net_pnl_pct; a[1] += 1
            if t.net_pnl_pct > 0:
                a[2] += 1
            tot_net += t.net_pnl_pct; tot_n += 1
    return agg, tot_net, tot_n


def show(label, agg, tot_net, tot_n):
    print(f"\n===== {label} =====")
    print(f"{'킬존':<24}{'net%':>8}{'거래':>7}{'승률':>7}{'평균%':>9}")
    for k in ORDER:
        net, n, nwin = agg[k]
        if n == 0:
            print(f"{k:<24}{'-':>8}{0:>7}")
            continue
        print(f"{k:<24}{net:>+8.1f}{n:>7}{100*nwin/n:>6.0f}%{net/n:>+9.3f}")
    print(f"총 {tot_n}거래 net={tot_net:+.1f}%")


def main() -> int:
    a_off, n_off, cnt_off = run(False)   # 킬존 한정 (구독 모드)
    a_on, n_on, cnt_on = run(True)       # 24h (referral 모드)
    show("킬존 한정 (disable_time_filter=False, 구독 모드)", a_off, n_off, cnt_off)
    show("24시간 (disable_time_filter=True, referral 모드)", a_on, n_on, cnt_on)
    print("\n===== 차이 (24h - 킬존한정) — 24h가 추가로 무는 거래의 순효과 =====")
    print(f"{'킬존':<24}{'Δnet%':>9}{'Δ거래':>8}")
    for k in ORDER:
        dn = a_on[k][0] - a_off[k][0]
        dc = a_on[k][1] - a_off[k][1]
        print(f"{k:<24}{dn:>+9.1f}{dc:>+8}")
    print(f"\n총 순효과: net {n_on - n_off:+.1f}% / 거래 {cnt_on - cnt_off:+d}")
    print("→ 24h 추가분이 음수면 referral 24h 개방이 손해 = 킬존 한정이 정답")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
