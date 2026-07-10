"""2순위 — bias 근본 개선: 구조(MSS/변위) 기반 vs EMA align (2026-07-10 밤).

문제: EMA align 은 유동성 이벤트 대비 3~5일 지연 (7/4 숏 전멸의 뿌리).
스윕 게이트(#365)는 '사건 직후 차단'일 뿐 평시 방향은 여전히 EMA.

후보 bias (전부 일봉 후행 전용 — 라이브 이식 가능):
    MSS bias: 3봉 피벗 스윙 추적 — 종가가 최근 스윙고점 돌파=bull /
              스윙저점 이탈=bear. 반대 돌파까지 유지.
    변위 bias: 일봉 몸통 > 1.5×ATR20 인 날의 방향. 반대 변위까지 유지.

변형 (full-run + 방향 필터):
    A 기준(align, 1.6 정합)
    B align OFF + MSS bias 필터 (교체)
    C align 유지 + MSS 필터 추가 (겹침 — 둘 다 허용해야 진입)
    D align OFF + 변위 bias 필터
    E align OFF + MSS·변위 합의(둘 다 같은 방향일 때만)
버킷: 국면 × 반기 + 합계. 표기 = 시드% 단리.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/bias_structure_lab.py
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
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)


def mss_bias(d):
    """일봉 MSS bias — 3봉 피벗 스윙 돌파 추적. 반환: 일별 'bull'/'bear'/None.

    어제까지의 마감 일봉만 사용 (오늘 bias 는 어제 종가 기준 — lookahead 없음).
    """
    h = d["high"].values
    low = d["low"].values
    c = d["close"].values
    n = len(d)
    bias = [None] * n
    last_sh = last_sl = None
    cur = None
    for i in range(n):
        # 오늘 bias = 어제까지 정보
        bias[i] = cur
        if i < 2:
            continue
        # 어제(i-1) 종가로 돌파 판정 → 오늘부터 반영되도록 순서 유지
        j = i - 1
        if last_sh is not None and c[j] > last_sh:
            cur = "bull"
        elif last_sl is not None and c[j] < last_sl:
            cur = "bear"
        # 피벗 확정 (j-1 기준 3봉)
        if 1 <= j - 1 and j < n:
            k = j - 1
            if h[k] > h[k - 1] and h[k] > h[k + 1 - 1 + 1 - 1] if False else False:
                pass
        k = j - 1
        if k >= 1 and k + 1 <= j:
            if h[k] > h[k - 1] and h[k] > h[k + 1]:
                last_sh = h[k]
            if low[k] < low[k - 1] and low[k] < low[k + 1]:
                last_sl = low[k]
    return bias


def disp_bias(d):
    """변위 bias — 몸통 > 1.5×ATR20 일봉의 방향 유지."""
    o = d["open"].values
    c = d["close"].values
    h = d["high"].values
    low = d["low"].values
    n = len(d)
    tr = np.maximum(h[1:] - low[1:],
                    np.maximum(abs(h[1:] - c[:-1]), abs(low[1:] - c[:-1])))
    atr = np.full(n, np.nan)
    for i in range(21, n):
        atr[i] = tr[i - 20:i].mean()
    bias = [None] * n
    cur = None
    for i in range(n):
        bias[i] = cur          # 오늘 = 어제까지 정보
        j = i                   # 오늘 봉은 미완 — 어제 봉으로 갱신
        if j >= 1 and not np.isnan(atr[j - 1]):
            body = c[j - 1] - o[j - 1]
            if abs(body) > 1.5 * atr[j - 1]:
                cur = "bull" if body > 0 else "bear"
    return bias


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    from aurora_ict.strategy.silver_bullet import Direction
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    d = df5.resample("1D").agg({"open": "first", "high": "max",
                                "low": "min", "close": "last"}).dropna()
    days_idx = d.index
    day_of = days_idx.searchsorted(df5.index.normalize(), side="right") - 1
    reg_idx, reg_labels = classify_days(df5)
    reg_of = reg_idx.searchsorted(df5.index.normalize(), side="right") - 1
    half = len(df5) // 2
    mb = mss_bias(d)
    db = disp_bias(d)

    def allowed(i, direction, use_mss, use_disp, need_both=False):
        di = day_of[i]
        if di < 0:
            return True
        want = "bull" if direction is Direction.LONG else "bear"
        oks = []
        if use_mss:
            oks.append(mb[di] is None or mb[di] == want)
        if use_disp:
            oks.append(db[di] is None or db[di] == want)
        if not oks:
            return True
        return all(oks) if need_both else all(oks)

    def run(cfg_ov, filt=None):
        cfg = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl, **cfg_ov})
        tl = cached_setup_timeline(df5, cfg, sym)
        if filt is not None:
            tl2 = list(tl)
            for i, item in enumerate(tl2):
                if item is not None and not filt(i, item[0].direction):
                    tl2[i] = None
            tl = tl2
        bt = run_backtest_from_timeline(df5, tl, cfg)
        buckets = {r: [0.0, 0] for r in REGIMES}
        halves = [0.0, 0.0]
        for t in bt.trades:
            ri = reg_of[t.entry_idx]
            lab = reg_labels[ri] if ri >= 0 else "횡보"
            buckets[lab][0] += t.net_pnl_pct
            buckets[lab][1] += 1
            halves[0 if t.entry_idx < half else 1] += t.net_pnl_pct
        return buckets, halves

    out = {}
    out["A 기준(align)"] = run({})
    out["B MSS 교체"] = run({"htf_ema_bias": "off"},
                          lambda i, dd: allowed(i, dd, True, False))
    out["C align+MSS 겹침"] = run({}, lambda i, dd: allowed(i, dd, True, False))
    out["D 변위 교체"] = run({"htf_ema_bias": "off"},
                          lambda i, dd: allowed(i, dd, False, True))
    out["E MSS+변위 합의 교체"] = run({"htf_ema_bias": "off"},
                                lambda i, dd: allowed(i, dd, True, True,
                                                      need_both=True))
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    labels = ["A 기준(align)", "B MSS 교체", "C align+MSS 겹침", "D 변위 교체",
              "E MSS+변위 합의 교체"]
    lines = ["===== bias 구조 실험 (7페어 5년, 시드% 단리) =====",
             f"{'변형':<20}" + "".join(f"{r:>11}" for r in REGIMES)
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
        lines.append(f"{label:<20}{seg}{g * 100:>+9.0f}%"
                     f"{hh[0] * 100:>+7.0f}/{hh[1] * 100:+.0f}%")
    txt = "\n".join(lines)
    with open("bias_structure_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
