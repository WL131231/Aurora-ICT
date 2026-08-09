"""#FST7 2026-07-17 자율연구 ③: 청산(exit) 측 국면별 스윕 — 촙 생존 청산법 탐색.

진입/게이트 축은 소진(NY_PM·cond_align·regime_filter). 미탐색 최대영역=청산.
라이브 2.0 청산(trail 2.0/1.5 + 분할 1.5R + BE) 기준, 청산 파라미터를 ER(장중
효율비) 버킷별로 스윕. 특히 극톱질(ER<q20, 현재장) 버킷서 개선되는 청산 탐색.

가설: 촙에선 되돌림 전 빠른 익절/조기 BE 가 SL 회귀를 막나?
청산 파라미터는 run-time(캐시 재사용) → 직렬 빠름. 2.0 게이트는 진입기반이라
청산과 독립 → 각 exit 변형 결과에 동일 post-filter 적용.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
# 라이브 2.0 청산 정합 base
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=True, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, entry_ttl_bars=6,
    trail_trigger=2.0, trail_dist=1.5, partial_tp_rr=1.5, partial_be=True,
)
ER_N = 48

# 청산 변형 (base 대비 override)
EXITS = {
    "base(tr2.0/1.5,p1.5,BE)": {},
    "빠른분할 p1.0": {"partial_tp_rr": 1.0},
    "빠른분할 p0.8": {"partial_tp_rr": 0.8},
    "조기BE be0.5": {"be_lock": 0.5},
    "조기BE be1.0": {"be_lock": 1.0},
    "빠른trail tr1.5": {"trail_trigger": 1.5},
    "타이트trail d1.0": {"trail_dist": 1.0},
    "빠른trail+타이트": {"trail_trigger": 1.5, "trail_dist": 1.0},
    "p1.0+be0.5": {"partial_tp_rr": 1.0, "be_lock": 0.5},
    "p1.0+trail1.5": {"partial_tp_rr": 1.0, "trail_trigger": 1.5},
}


def er_at(closes, idx, n=ER_N):
    if idx < n:
        return 1.0
    seg = closes[idx - n:idx + 1]
    path = np.abs(np.diff(seg)).sum()
    return abs(seg[-1] - seg[0]) / path if path > 0 else 0.0


def gate_keep(t, df5, strong_floor):
    """2.0 게이트(NY_PM 제외 + cond_align) — 진입기반, 청산과 독립."""
    h = df5.index[t.entry_idx].hour
    if 17 <= h < 21:
        return False
    sign = 1.0 if t.direction == "long" else -1.0
    signed = t.entry_trend_pct * sign
    mag = abs(t.entry_trend_pct)
    if mag < strong_floor and signed < 0:  # 약/중추세 역추세 차단
        return False
    return True


def main() -> int:
    # 페어별 df/closes/timeline/강추세floor 준비
    prep = {}
    q70 = {}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        cfg0 = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg0, sym)
        prep[sym] = (df5, df5["close"].to_numpy(), tl)
        bt0 = run_backtest_from_timeline(df5, tl, cfg0)
        mags = [abs(t.entry_trend_pct) for t in bt0.trades
                if not (17 <= df5.index[t.entry_idx].hour < 21)]
        q70[sym] = np.percentile(mags, 70) if mags else 0.0

    # ER 임계 (전 base trade)
    all_er = []
    for sym in PAIRS:
        df5, closes, tl = prep[sym]
        bt0 = run_backtest_from_timeline(df5, tl, BacktestConfig(**BASE))
        for t in bt0.trades:
            if gate_keep(t, df5, q70[sym]):
                all_er.append(er_at(closes, t.entry_idx))
    er_q20 = np.percentile(all_er, 20)

    print(f"극톱질 ER<{er_q20:.3f} (현재장 아날로그) | 2.0 게이트 적용\n")
    print(f"{'청산변형':<24}{'총net':>8}{'극톱질net':>10}{'추세net':>9}{'거래':>6}")
    base_tot = base_chop = None
    for ename, ov in EXITS.items():
        cfg = BacktestConfig(**{**BASE, **ov})
        tot = chop = trend = n = 0
        for sym in PAIRS:
            df5, closes, tl = prep[sym]
            bt = run_backtest_from_timeline(df5, tl, cfg)
            for t in bt.trades:
                if not gate_keep(t, df5, q70[sym]):
                    continue
                er = er_at(closes, t.entry_idx)
                tot += t.net_pnl_pct; n += 1
                if er < er_q20:
                    chop += t.net_pnl_pct
                else:
                    trend += t.net_pnl_pct
        if base_tot is None:
            base_tot, base_chop = tot, chop
        dc = "" if ename.startswith("base") else f" (촙Δ{chop-base_chop:+.1f})"
        print(f"{ename:<24}{tot:>+8.1f}{chop:>+10.1f}{trend:>+9.1f}{n:>6}{dc}")
    print("\n→ 극톱질net 이 base보다 큰 청산 = 현재장 청산 개선책 (총net 유지하며)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
