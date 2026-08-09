"""#FST6 2026-07-16: 횡보 게이트 강화 스윕 — 무추세장 입장료 절감 연구.

라이브 진단: 2주 극단 횡보(ER 1-7%)에서 추세모델 Origo 가 톱질당함. 라이브
regime_filter 는 |entry_trend_pct|<q33 floor 스킵(크기만, 방향 무시). 각 trade 에
entry_trend_pct 기록되므로 5년 백테 trade 를 후처리로 여러 촙 게이트 변형 평가.

베이스 = 1.9(NY_PM 제외) 위. 변형:
  - mag_qNN: |trend| 하위 NN% floor (q33 현행 → q40/50/60 상향)
  - align  : signed_trend>=0 (진입방향이 20봉 추세와 정합) — 역추세 진입 제거
  - align+mag: 둘 다
목표: net 개선 + 빈도 과도감소 없이 + 페어 robust.
"""
from __future__ import annotations

import os
import sys

import numpy as np

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
    apply_cisd=True, apply_po3=True, disable_time_filter=True, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
)


def is_nypm_utc(h: int) -> bool:
    return 17 <= h < 21  # NY_PM = 02-05 KST


def collect():
    """페어별 trade (entry_trend_pct, direction부호, net, is_nypm) 수집."""
    out = {}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        recs = []
        for t in bt.trades:
            h = df5.index[t.entry_idx].hour
            sign = 1.0 if t.direction == "long" else -1.0
            recs.append((t.entry_trend_pct, sign, t.net_pnl_pct, is_nypm_utc(h)))
        out[sym] = recs
    return out


def evalv(data, keep):
    """keep(rec)->bool 필터로 페어별 net 집계. 반환=(총net, 총거래, 승률, 페어net dict)."""
    tot_net = tot_n = tot_w = 0
    per = {}
    for sym, recs in data.items():
        kept = [r for r in recs if keep(r)]
        net = sum(r[2] for r in kept)
        per[sym] = net
        tot_net += net; tot_n += len(kept)
        tot_w += sum(1 for r in kept if r[2] > 0)
    wr = 100 * tot_w / tot_n if tot_n else 0
    return tot_net, tot_n, wr, per


def main() -> int:
    data = collect()
    # 페어별 |trend| 분위 floor 사전계산 (NY_PM 제외 후 기준).
    floors = {}
    for q in (33, 40, 50, 60):
        floors[q] = {}
        for sym, recs in data.items():
            mags = [abs(r[0]) for r in recs if not r[3]]
            floors[q][sym] = np.percentile(mags, q) if mags else 0.0

    # 베이스 = NY_PM 제외 (1.9).
    base = lambda r: not r[3]  # noqa: E731
    variants = [("1.9 base(NY_PM제외)", base)]
    for q in (40, 50, 60):
        variants.append((f"mag_q{q}",
                         lambda r, q=q: base(r) and abs(r[0]) >= floors[q][_cur[0]]))
    variants.append(("align(정합)", lambda r: base(r) and r[0] * r[1] > 0))
    for q in (40, 50):
        variants.append((f"align+mag_q{q}",
                         lambda r, q=q: base(r) and r[0] * r[1] > 0
                         and abs(r[0]) >= floors[q][_cur[0]]))

    print(f"{'변형':<22}{'net%':>8}{'거래':>7}{'승률':>7}{'페어+':>7}")
    base_net = None
    for label, fn in variants:
        # 페어별 floor 참조 위해 sym 컨텍스트 필요 → per-pair 재구현.
        tot_net = tot_n = tot_w = 0
        pos_pairs = 0
        base_per = {}
        for sym, recs in data.items():
            _cur[0] = sym
            kept = [r for r in recs if fn(r)]
            net = sum(r[2] for r in kept)
            base_per[sym] = net
            if net > 0:
                pos_pairs += 1
            tot_net += net; tot_n += len(kept)
            tot_w += sum(1 for r in kept if r[2] > 0)
        wr = 100 * tot_w / tot_n if tot_n else 0
        if base_net is None:
            base_net = tot_net
        mark = f" (Δ{tot_net-base_net:+.1f})" if label != variants[0][0] else ""
        print(f"{label:<22}{tot_net:>+8.1f}{tot_n:>7}{wr:>6.0f}%{pos_pairs:>6}/7{mark}")
    print("\n목표: base 대비 net↑ + 거래 과도감소 없음(빈도) + 페어+ 5/7 이상(robust)")
    return 0


_cur = [None]  # per-pair floor 참조용 (클로저 sym 컨텍스트)

if __name__ == "__main__":
    raise SystemExit(main())
