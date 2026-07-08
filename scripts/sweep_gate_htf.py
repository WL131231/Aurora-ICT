"""사건인식 HTF 확장 — 1h/2h/4h 스윕-반전 게이트 (파트너 지시, 2026-07-08).

배포된 일봉 스윕-반전 게이트(#365: 10일 극단 스윕+반전 마감 → 2일 역방향 차단)
를 인트라데이 TF 로 확장 검증. 같은 규칙을 TF 봉에 적용:
    - TF 봉이 직전 3일 상당 봉들의 최저를 하회 스윕 + 상반부 마감 → K시간 SHORT 차단
    - 고점 대칭 → LONG 차단
기준선 = 일봉 게이트(K=2일, 라이브 상태) 적용 타임라인. 변형 = 기준선 위에
TF 게이트 추가(1h/2h/4h 단독 + 전부). K = 12h / 24h.
BASE = Origo 1.6 (유동성TP + trail 2/1.5 + BE@1R). 재생만(캐시).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/sweep_gate_htf.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
LOOKBACK_DAYS_D = 10   # 일봉 게이트 (배포값)
BLOCK_DAYS_D = 2
LOOKBACK_DAYS_TF = 3   # 인트라데이 게이트 룩백 (3일 상당 봉수)
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)
TFS = {"1h": 60, "2h": 120, "4h": 240}
BLOCK_HOURS = [12, 24]


def _sweep_block_mask(df5, rule: str, minutes: int, lookback_bars: int, block_bars_5m: int):
    """TF 스윕-반전 → 5m 인덱스 차단 마스크 (block_short, block_long).

    스윕-반전 봉 '마감 직후'부터 block_bars_5m(5m 봉수) 동안 차단.
    """
    d = df5.resample(rule).agg({"high": "max", "low": "min", "close": "last"}).dropna()
    lo_n = d["low"].shift(1).rolling(lookback_bars).min()
    hi_n = d["high"].shift(1).rolling(lookback_bars).max()
    mid = (d["low"] + (d["high"] - d["low"]) * 0.5)
    ssl = ((d["low"] < lo_n) & (d["close"] > mid)).values
    bsl = ((d["high"] > hi_n) & (d["close"] < mid)).values
    n5 = len(df5)
    bs = np.zeros(n5, dtype=bool)
    bl = np.zeros(n5, dtype=bool)
    # TF 봉 마감 시각 → 5m 위치
    close_pos = df5.index.searchsorted(d.index + np.timedelta64(minutes, "m"))
    for j in range(len(d)):
        if not (ssl[j] or bsl[j]):
            continue
        a = close_pos[j]
        b = min(n5, a + block_bars_5m)
        if ssl[j]:
            bs[a:b] = True
        if bsl[j]:
            bl[a:b] = True
    return bs, bl


def _apply_masks(tl, masks):
    """masks = [(block_short, block_long), ...] — OR 결합해 setup 제거."""
    from aurora_ict.strategy.silver_bullet import Direction
    out = list(tl)
    for i, item in enumerate(out):
        if item is None:
            continue
        d = item[0].direction
        for bs, bl in masks:
            if (d is Direction.SHORT and bs[i]) or (d is Direction.LONG and bl[i]):
                out[i] = None
                break
    return out


def _metrics(trades):
    ts = list(trades)
    wins = [t.net_pnl_pct for t in ts if t.net_pnl_pct > 0]
    cum = peak = mdd = 0.0
    for t in sorted(ts, key=lambda t: t.exit_idx):
        cum += t.net_pnl_pct
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return dict(cum=cum, mdd=mdd, n=len(ts), nwin=len(wins))


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl})
    tl = cached_setup_timeline(df5, cfg, sym)
    # 기준선 = 배포 상태(일봉 게이트 K=2일)
    daily_mask = _sweep_block_mask(df5, "1D", 1440, LOOKBACK_DAYS_D,
                                   BLOCK_DAYS_D * 288)
    tl_base = _apply_masks(tl, [daily_mask])
    out = {"기준(일봉 게이트=라이브)": _metrics(
        run_backtest_from_timeline(df5, tl_base, cfg).trades)}
    for tf, minutes in TFS.items():
        lb = LOOKBACK_DAYS_TF * (1440 // minutes)
        for bh in BLOCK_HOURS:
            m = _sweep_block_mask(df5, tf, minutes, lb, bh * 12)
            ftl = _apply_masks(tl, [daily_mask, m])
            out[f"+{tf} 게이트 K={bh}h"] = _metrics(
                run_backtest_from_timeline(df5, ftl, cfg).trades)
    # 전 TF 결합 (K=24h)
    all_masks = [daily_mask] + [
        _sweep_block_mask(df5, tf, minutes, LOOKBACK_DAYS_TF * (1440 // minutes), 24 * 12)
        for tf, minutes in TFS.items()
    ]
    out["+1h+2h+4h (K=24h)"] = _metrics(
        run_backtest_from_timeline(df5, _apply_masks(tl, all_masks), cfg).trades)
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    labels = ["기준(일봉 게이트=라이브)"]
    for tf in TFS:
        for bh in BLOCK_HOURS:
            labels.append(f"+{tf} 게이트 K={bh}h")
    labels.append("+1h+2h+4h (K=24h)")
    lines = ["===== 사건인식 HTF 확장 (7페어 5년, BASE=Origo 1.6 정합) =====",
             f"{'변형':<24}{'USDT':>8}{'DD':>7}{'승률':>6}{'거래':>7}"]
    for label in labels:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0, "nwin": 0}
        for _sym, out in results:
            m = out[label]
            for k in tot:
                tot[k] += m[k]
        wr = tot["nwin"] / tot["n"] * 100 if tot["n"] else 0.0
        lines.append(f"{label:<22}{tot['cum'] * SEED / 100:>+8.0f}"
                     f"{tot['mdd'] * SEED / 100:>7.0f}{wr:>5.0f}%{tot['n']:>7d}")
    txt = "\n".join(lines)
    with open("sweep_gate_htf_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
