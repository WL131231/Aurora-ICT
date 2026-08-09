"""#AUTONOMOUS 2026-07-20: 자기자본곡선 필터 — 장 의존도 낮추기.

파트너: "연속 손실/수익이 찍힌다, 장을 더 이기는 조합". 라이브 pnl lag1 자기상관
+0.65(강한 clustering, 나쁜국면 지속). 처방 가설=equity-curve filter: 전략 자신의
최근 성과가 나쁘면(롤링 pnl<0) 진입 skip, 회복하면 복귀 → 나쁜 streak 꼬리 절단.

먼저 백테(단일봇, 멀티유저 혼입 없음)에서 clustering 재현 확인 → equity 필터
(윈도우W 최근 pnl 합<thr 면 skip) 스윕 → net·MDD·streak·walk-forward.
페어별 독립 equity(각 페어 자기 최근성과로 자기 진입 게이트).
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

FIXED = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
    trail_trigger=2.0, trail_dist=1.5, partial_tp_rr=1.5, partial_be=True,
)


def gated_trades(bt, df5):
    """2.0 게이트(NY_PM+cond_align) 후 (entry_time, net) 시간순."""
    mags = [abs(t.entry_trend_pct) for t in bt.trades
            if not (17 <= df5.index[t.entry_idx].hour < 21)]
    q70 = np.percentile(mags, 70) if mags else 0.0
    out = []
    for t in bt.trades:
        h = df5.index[t.entry_idx].hour
        if 17 <= h < 21:
            continue
        sign = 1.0 if t.direction == "long" else -1.0
        if abs(t.entry_trend_pct) < q70 and t.entry_trend_pct * sign < 0:
            continue
        out.append((df5.index[t.entry_idx].value, t.net_pnl_pct))
    return out


def mdd(cum):
    peak = -1e9
    md = 0.0
    for v in cum:
        peak = max(peak, v)
        md = min(md, v - peak)
    return md


def apply_filter(trades, W, thr):
    """페어별 equity 필터: 최근 W개 pnl 합 < thr 면 skip. skip 도 이력엔 미포함
    (skip 이라 결과 없음). 반환: 통과 trade net 리스트."""
    hist = []
    kept = []
    for _t, net in trades:
        recent = sum(hist[-W:]) if len(hist) >= W else 0.0
        if len(hist) >= W and recent < thr:
            # skip — 진입 안 함(성과 없음), 이력엔 미반영(관측 못함)
            continue
        kept.append(net)
        hist.append(net)
    return kept


def main() -> int:
    # 페어별 trade 수집
    per = {}
    for sym in FIXED:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        per[sym] = gated_trades(bt, df5)

    # 1) 백테 clustering 확인 (페어별 pnl lag1 자기상관 평균)
    print("=== 백테 단일봇 clustering 확인 (페어별 pnl lag1 자기상관) ===")
    acs = []
    for sym in FIXED:
        nets = [n for _, n in per[sym]]
        if len(nets) > 5:
            a = np.corrcoef(np.array(nets[:-1]), np.array(nets[1:]))[0, 1]
            acs.append(a)
            print(f"  {sym:<9} n={len(nets):3d} lag1자기상관={a:+.3f}")
    print(f"  평균 자기상관={np.nanmean(acs):+.3f} (양수면 clustering 진짜)\n")

    # 2) equity 필터 스윕 (페어별 독립 적용 → 합산)
    def eval_cfg(W, thr):
        allkept = []
        for sym in FIXED:
            allkept += apply_filter(per[sym], W, thr) if W else [n for _, n in per[sym]]
        net = sum(allkept)
        n = len(allkept)
        w = sum(1 for x in allkept if x > 0)
        cum = np.cumsum(allkept)
        return net, n, (100 * w / n if n else 0), mdd(cum)

    print("=== equity-curve 필터 스윕 (W=최근거래수, thr=롤링합 문턱) ===")
    print(f"{'설정':<20}{'net%':>8}{'거래':>6}{'승률':>7}{'MDD%':>8}{'net/MDD':>9}")
    bn, bnn, bw, bmd = eval_cfg(0, 0)
    print(f"{'base(필터없음)':<20}{bn:>+8.1f}{bnn:>6}{bw:>6.0f}%{bmd:>+8.1f}{bn/abs(bmd) if bmd else 0:>9.2f}")
    for W in (3, 5, 8):
        for thr in (0.0, -2.0):
            net, n, wr, md = eval_cfg(W, thr)
            print(f"{'W'+str(W)+' thr'+str(thr):<20}{net:>+8.1f}{n:>6}{wr:>6.0f}%{md:>+8.1f}"
                  f"{net/abs(md) if md else 0:>9.2f}")
    print("\n→ net 유지·상승하며 MDD 축소(net/MDD↑) = 장 의존도 낮춤 성공")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
