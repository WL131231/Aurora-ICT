"""분할 진입 시뮬 — 1차 0.786 + 2차 0.886, 반등 시 2차 취소 (파트너 설계, 2026-07-10).

규칙:
    setup 채택(0.786 타임라인·게이트 통과) 후 TTL 안에서
    E1 = fvg OTE 0.786 터치 시 50% / E2 = 0.886 터치 시 50%.
    E1 체결 후 가격이 OTE 0.5(변형: 0.382) 를 트레이드 방향으로 되찾으면 E2 취소.
    SL = setup 구조 SL (x4 확장 동일 규칙, 혼합 진입가 기준 리스크 재산정).
    청산 = 1.6 정합: 유동성 TP + BE@1R + 트레일 2R/1.5R (혼합 진입가 기준),
    동일봉 SL 우선(비관). 수수료/슬리피지/펀딩 반영.

변형: 단일 0.786(대조군, 동일 심으로) / 분할+취소0.5 / 분할+취소0.382 / 분할 무취소.
출력: 국면 버킷 + 반기 + 합계 (시드% 단리).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/split_entry_sim.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from regime_edge_lab import REGIMES, classify_days  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SL_MULT = 4.0
TRAIL_TRG, TRAIL_DST, BE_TRG = 2.0, 1.5, 1.0
HOLD_MAX_BARS = 288 * 4   # 안전 상한 4일 (트레일 미청산 방치 방지)
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=SL_MULT, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.786, min_rr=2.0, tp_rr_override=0.0,
    trail_trigger=TRAIL_TRG, trail_dist=TRAIL_DST, be_trigger=BE_TRG, be_lock=0.0,
)


def _sim_trade(o, h, low, c, i0, ttl, direction, lv1, lv2, cancel_lv, sl0, tp,
               split: bool):
    """한 setup 의 분할 체결+청산 시뮬. 반환 (net_fraction, 체결여부)."""
    from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct
    sign = 1.0 if direction == "long" else -1.0
    e1 = e2 = None
    e2_cancelled = not split
    n = len(c)
    j = i0
    # --- 체결 단계 (TTL 내) ---
    while j < min(i0 + ttl, n - 1):
        j += 1
        touch1 = low[j] <= lv1 if sign > 0 else h[j] >= lv1
        touch2 = low[j] <= lv2 if sign > 0 else h[j] >= lv2
        bounce = (h[j] >= cancel_lv) if sign > 0 else (low[j] <= cancel_lv)
        if e1 is None and touch1:
            e1 = lv1
            if split and touch2:      # 같은 봉 관통 — 둘 다 체결로 처리
                e2 = lv2
            j_fill = j
            break
        if e1 is None and bounce:
            continue                   # 미체결 반등은 무시(체결 전 취소 개념 없음)
    if e1 is None:
        return None
    # E2 대기 (E1 체결 후): 취소 조건 vs 터치 — 같은 봉이면 취소 우선(보수).
    entry_frac = [(e1, 0.5 if split else 1.0)]
    k = j_fill
    while split and e2 is None and not e2_cancelled and k < min(i0 + ttl, n - 1):
        k += 1
        bounce = (h[k] >= cancel_lv) if sign > 0 else (low[k] <= cancel_lv)
        touch2 = low[k] <= lv2 if sign > 0 else h[k] >= lv2
        if bounce:
            e2_cancelled = True
            break
        if touch2:
            e2 = lv2
            break
    if e2 is not None:
        entry_frac.append((e2, 0.5))
    tot_frac = sum(f for _e, f in entry_frac)
    blend = sum(e * f for e, f in entry_frac) / tot_frac
    risk = abs(blend - sl0)
    if risk <= 0:
        return None
    # --- 청산 단계 ---
    stop = sl0
    be_done = False
    trail_on = False
    peak = blend
    start = max(j_fill, k if e2 is not None else j_fill)
    for m in range(start + 1, min(start + HOLD_MAX_BARS, n)):
        hit_sl = low[m] <= stop if sign > 0 else h[m] >= stop
        if hit_sl:
            exit_px, mkt = stop, False
            break
        hit_tp = h[m] >= tp if sign > 0 else low[m] <= tp
        if hit_tp:
            exit_px, mkt = tp, False
            break
        prof = (h[m] - blend) if sign > 0 else (blend - low[m])
        if not be_done and prof >= risk * BE_TRG:
            stop = blend if (sign > 0 and blend > stop) or (sign < 0 and blend < stop) else stop
            be_done = True
        if prof >= risk * TRAIL_TRG:
            trail_on = True
        if trail_on:
            peak = max(peak, h[m]) if sign > 0 else min(peak, low[m])
            t_stop = peak - sign * risk * TRAIL_DST
            if (sign > 0 and t_stop > stop) or (sign < 0 and t_stop < stop):
                stop = t_stop
    else:
        m = min(start + HOLD_MAX_BARS, n) - 1
        exit_px, mkt = c[m], True
    slp = slip_pct(h[m], low[m], c[m])
    px = apply_slippage(exit_px, "long" if sign > 0 else "short", "exit", slp) \
        if mkt else exit_px
    net = 0.0
    for e, f in entry_frac:
        raw = (px - e) / e * sign
        part, _ = apply_costs(raw, 0.9 * f, 20.0)
        part -= (m - j_fill) / 12 * (0.0001 / 8) * 0.9 * f * 20.0
        net += part
    return net, m


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl})
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)   # 게이트 통과 setup 식별용
    o = df5["open"].values
    h = df5["high"].values
    low = df5["low"].values
    c = df5["close"].values
    days_idx, labels = classify_days(df5)
    day_of = days_idx.searchsorted(df5.index.normalize(), side="right") - 1
    half = len(df5) // 2

    # 트레이드 → 원 setup 역추적 (entry_idx 직전 ttl 창의 timeline 항목)
    jobs = []
    for t in bt.trades:
        for j in range(t.entry_idx, max(t.entry_idx - ttl - 1, 0), -1):
            item = tl[j]
            if item is not None and item[0].direction.value == str(t.direction):
                s = item[0]
                if s.fvg is not None:
                    jobs.append((j, s))
                break

    def run_variant(split, cancel_ote):
        buckets = {r: [0.0, 0] for r in REGIMES}
        halves = [0.0, 0.0]
        for j, s in jobs:
            sign = 1.0 if s.direction.value == "long" else -1.0
            lv1 = s.fvg.ote_threshold(0.786)
            lv2 = s.fvg.ote_threshold(0.886)
            # cancel_ote=None = 무취소 (절대 안 닿는 레벨)
            cancel_lv = (s.fvg.ote_threshold(cancel_ote) if cancel_ote
                         else (float("inf") if sign > 0 else float("-inf")))
            sl_dist0 = abs(s.entry - s.stop_loss) * SL_MULT
            sl0 = lv1 - sign * sl_dist0     # 구조 SL x4 (1차 기준 — 엔진 근사)
            r = _sim_trade(o, h, low, c, j, ttl, s.direction.value,
                           lv1, lv2, cancel_lv, sl0, s.take_profit, split)
            if r is None:
                continue
            net, m = r
            di = day_of[j]
            lab = labels[di] if di >= 0 else "횡보"
            buckets[lab][0] += net
            buckets[lab][1] += 1
            halves[0 if j < half else 1] += net
        return buckets, halves

    out = {
        "단일 0.786 (대조)": run_variant(False, 0.5),
        "분할 취소0.5": run_variant(True, 0.5),
        "분할 취소0.382": run_variant(True, 0.382),
        "분할 무취소": run_variant(True, None),
    }
    print(f"  {sym} done jobs={len(jobs)}", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    labels = ["단일 0.786 (대조)", "분할 취소0.5", "분할 취소0.382", "분할 무취소"]
    lines = ["===== 분할 진입 0.786+0.886 (7페어 5년, 시드% 단리, 커스텀 체결심) =====",
             f"{'변형':<18}" + "".join(f"{r:>11}" for r in REGIMES)
             + f"{'합계':>10}{'전/후반':>15}"]
    for label in labels:
        tot = {r: [0.0, 0] for r in REGIMES}
        hh = [0.0, 0.0]
        for _s, out in results:
            b, hv = out[label]
            for r in REGIMES:
                tot[r][0] += b[r][0]
                tot[r][1] += b[r][1]
            hh[0] += hv[0]
            hh[1] += hv[1]
        g = sum(v[0] for v in tot.values())
        seg = "".join(f"{tot[r][0] * 100:>+8.0f}({tot[r][1]:>2})" for r in REGIMES)
        lines.append(f"{label:<18}{seg}{g * 100:>+9.0f}%"
                     f"{hh[0] * 100:>+7.0f}/{hh[1] * 100:+.0f}%")
    txt = "\n".join(lines)
    with open("split_entry_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
