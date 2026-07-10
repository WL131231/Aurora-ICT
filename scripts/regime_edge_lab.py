"""국면 4분할 엣지 랩 — 하락/횡보/상승/횡보→급변 (파트너 지시, 2026-07-10).

목적: 엣지 후보들을 국면별로 나눠 평가 — 전체 합에선 기각됐던 방법이 특정
국면에서 유효한지(=국면 적응형 게이트 후보) 재심. 오퍼스 시절 방법 재검 포함.

국면 분류 (후행 전용 — 라이브 구현 가능):
    일봉 20일 수익률 z (일간수익률 20일 표준편차×√20 정규화, 어제까지)
    z > +0.75 = 상승 / z < -0.75 = 하락 / 그 외 = 횡보.
    전이: 직전 5일+ 횡보 지속 후 |일간수익률| > 2.5×20일σ 인 날부터 3일
    = 횡보→급등/급락 (전이가 기본 라벨보다 우선).

변형 (전부 재생/필터 — 캐시):
    1 기준(1.6) / 2 htf_align=4 재검 / 3 CT-SL(순추세 x3·역추세 x4, Origo1.1 방식)
    4 국면-방향 게이트(하락장 롱금지+상승장 숏금지) / 5 전이 구간만 sweep 필수
    6 NY_PM 진입 제외.
표기 = 시드% (단리, 20배 반영).
사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/regime_edge_lab.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0,
    trail_trigger=2.0, trail_dist=1.5, be_trigger=1.0, be_lock=0.0,
)
REGIMES = ["하락", "횡보", "상승", "전이↑", "전이↓"]


def classify_days(df5):
    """일 단위 국면 라벨 (후행 전용)."""
    d = df5.resample("1D").agg({"close": "last"}).dropna()
    ret = d["close"].pct_change()
    sig = ret.rolling(20).std()
    r20 = d["close"].pct_change(20)
    z = (r20 / (sig * np.sqrt(20))).shift(1)          # 어제까지 정보
    base = np.where(z > 0.75, "상승", np.where(z < -0.75, "하락", "횡보"))
    # 전이: 직전 5일+ 횡보 지속 후 급변일부터 3일
    label = base.copy().astype(object)
    range_streak = 0
    burst_left = 0
    burst_dir = ""
    thr = (2.5 * sig).shift(1)
    for i in range(len(d)):
        if burst_left > 0:
            label[i] = burst_dir
            burst_left -= 1
            continue
        if base[i] == "횡보":
            range_streak += 1
        if (range_streak >= 5 and not np.isnan(thr.iloc[i])
                and abs(ret.iloc[i]) > thr.iloc[i]):
            burst_dir = "전이↑" if ret.iloc[i] > 0 else "전이↓"
            label[i] = burst_dir
            burst_left = 2
            range_streak = 0
        elif base[i] != "횡보":
            range_streak = 0
    return d.index, label


def _day_of(df5, days_idx):
    return days_idx.searchsorted(df5.index.normalize(), side="right") - 1


def _ny_pm_mask(df5):
    """NY_PM(02~05 KST) 진입 시간 마스크."""
    hours = (df5.index + timedelta(hours=9)).hour
    return (hours >= 2) & (hours < 5)


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    from aurora_ict.strategy.silver_bullet import Direction
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    cfg0 = BacktestConfig(**{**BASE, "entry_ttl_bars": ttl})
    tl = cached_setup_timeline(df5, cfg0, sym)
    days_idx, labels = classify_days(df5)
    day_of = _day_of(df5, days_idx)
    npm = _ny_pm_mask(df5)

    def tl_filter(fn):
        out = list(tl)
        for i, item in enumerate(out):
            if item is None or day_of[i] < 0:
                continue
            if not fn(i, item[0], labels[day_of[i]]):
                out[i] = None
        return out

    def run(cfg=None, ftl=None):
        bt = run_backtest_from_timeline(df5, ftl if ftl is not None else tl,
                                        cfg if cfg is not None else cfg0)
        # 국면 버킷 (진입봉 기준)
        buckets = {r: [0.0, 0, 0] for r in REGIMES}   # [net, n, wins]
        for t in bt.trades:
            di = day_of[t.entry_idx]
            if di < 0:
                continue
            b = buckets[labels[di]]
            b[0] += t.net_pnl_pct
            b[1] += 1
            b[2] += 1 if t.net_pnl_pct > 0 else 0
        return buckets

    out = {}
    out["1 기준(1.6)"] = run()
    out["2 htf_align=4 재검"] = run(cfg=BacktestConfig(
        **{**BASE, "entry_ttl_bars": ttl, "htf_align_threshold": 4}))
    out["3 CT-SL x3/역추세x4"] = run(cfg=BacktestConfig(
        **{**BASE, "entry_ttl_bars": ttl, "sl_dist_mult": 3.0,
           "sl_dist_mult_ct": 4.0, "ct_trend_threshold": 0.0}))
    out["4 국면-방향 게이트"] = run(ftl=tl_filter(
        lambda i, s, lab: not ((lab == "하락" and s.direction is Direction.LONG)
                               or (lab == "상승" and s.direction is Direction.SHORT))))
    out["5 전이만 sweep 필수"] = run(ftl=tl_filter(
        lambda i, s, lab: (not lab.startswith("전이"))
        or any("sweep" in str(c).lower() for c in (s.confluences or []))))
    out["6 NY_PM 제외"] = run(ftl=tl_filter(lambda i, s, lab: not npm[i]))
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    labels = ["1 기준(1.6)", "2 htf_align=4 재검", "3 CT-SL x3/역추세x4",
              "4 국면-방향 게이트", "5 전이만 sweep 필수", "6 NY_PM 제외"]
    lines = ["===== 국면 4분할 엣지 랩 (7페어 5년, 시드% 단리·20배 반영) =====", ""]
    for label in labels:
        tot = {r: [0.0, 0, 0] for r in REGIMES}
        for _sym, out in results:
            for r in REGIMES:
                b = out[label][r]
                tot[r][0] += b[0]
                tot[r][1] += b[1]
                tot[r][2] += b[2]
        row = [f"{label:<22}"]
        g_net = g_n = 0.0
        for r in REGIMES:
            net, n, w = tot[r]
            g_net += net
            g_n += n
            wr = w / n * 100 if n else 0
            row.append(f"{r} {net * 100:+.0f}%({n}/{wr:.0f}%)")
        row.append(f"| 합계 {g_net * 100:+.0f}% ({int(g_n)}건)")
        lines.append("  ".join(row))
    txt = "\n".join(lines)
    with open("regime_edge_lab_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
