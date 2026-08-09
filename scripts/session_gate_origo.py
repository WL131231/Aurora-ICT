"""#AUTONOMOUS 2026-07-29: 세션(시간대) 게이트 — 현행 Origo 5년 백테 분해 (파트너 지시 ③ 재구성).

파트너 원 요청은 "07/08AM 모델" 재검증이었으나 7/20 기록 확인 결과 그것은 독립 모델이
아니라 **MMBM 반전 로직을 NY 오전에 한정한 버전**이고, MMBM 은 7/27 완전 기각
(라이브 24청산 전패, 재검증 -181%, 원 검증 재현 실패=부기 착시). 모체가 죽었으므로
그대로 되살릴 근거가 없다. 대신 그 안의 **살릴 수 있는 질문**만 분리:

    "AM 세션이 다른 시간대보다 실제로 좋은가?" — MMBM 과 무관하게 **현행 Origo** 에
    바로 적용 가능. 6/24 킬존 연구(NY_PM 최악)·nypm_gate_verify.py 의 연장선.

방법: 현행 라이브 정합 설정으로 5년 7페어 백테 → 각 거래를 **진입 시각(UTC)** 으로
분류해 세션별 net·승률·건당. 그리고 각 세션 제외 시 총 net 이 개선되는지(페어 과반 포함).
세션 정의(UTC / KST):
  ASIA      00-07 / 09-16      LONDON    07-11 / 16-20
  NY_AM     11-14 / 20-23      ← ICT "AM 프라임타임"·07/08AM 모델이 노린 구간
  NY_LUNCH  14-17 / 23-02      NY_PM     17-21 / 02-06      LATE  21-24 / 06-09
판정: 특정 세션 제외가 총 net 개선 + **페어 과반 개선** 이어야 채택(요행 배제).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
# 라이브 정합 — disable_time_filter=True 로 전 시간대 진입을 재현한 뒤 사후 분류.
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=True, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
)
SESSIONS = [
    ("ASIA", 0, 7), ("LONDON", 7, 11), ("NY_AM", 11, 14),
    ("NY_LUNCH", 14, 17), ("NY_PM", 17, 21), ("LATE", 21, 24),
]


def sess_of(h: int) -> str:
    for name, a, b in SESSIONS:
        if a <= h < b:
            return name
    return "LATE"


def main() -> int:
    per_sym: dict[str, dict[str, list[float]]] = {}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**BASE)
        tl = cached_setup_timeline(df5, cfg, sym)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        d: dict[str, list[float]] = {n: [] for n, _, _ in SESSIONS}
        for t in bt.trades:
            d[sess_of(df5.index[t.entry_idx].hour)].append(t.net_pnl_pct)
        per_sym[sym] = d
        print(f"  {sym} 거래 {len(bt.trades)}건 로드", flush=True)

    print("\n===== 세션별 성과 (5년 7페어 합산) =====", flush=True)
    print(f"  {'세션':<10} {'KST':<9} {'n':>6} {'net':>10} {'승률':>6} {'건당':>8}", flush=True)
    kst = {"ASIA": "09-16", "LONDON": "16-20", "NY_AM": "20-23",
           "NY_LUNCH": "23-02", "NY_PM": "02-06", "LATE": "06-09"}
    totals = {}
    for name, _, _ in SESSIONS:
        vals = [v for s in PAIRS for v in per_sym[s][name]]
        if not vals:
            continue
        net = sum(vals)
        totals[name] = net
        wr = 100 * sum(1 for v in vals if v > 0) / len(vals)
        print(f"  {name:<10} {kst[name]:<9} {len(vals):6d} {net:+10.1f} {wr:5.0f}% "
              f"{net / len(vals):+7.3f}", flush=True)

    grand = sum(totals.values())
    print(f"\n  전체 net {grand:+.1f}%", flush=True)

    print("\n===== 세션 제외 효과 (제외 시 총 net) =====", flush=True)
    print(f"  {'제외 세션':<12} {'제외후net':>10} {'개선':>9} {'페어개선':>9}", flush=True)
    for name, _, _ in SESSIONS:
        after = grand - totals.get(name, 0.0)
        imp = 0
        for s in PAIRS:
            a = sum(v for k in per_sym[s] for v in per_sym[s][k])
            b = a - sum(per_sym[s][name])
            if b > a:
                imp += 1
        mark = "★" if after > grand and imp >= 4 else " "
        print(f" {mark}{name:<12} {after:+10.1f} {after - grand:+9.1f} {imp:>6}/7", flush=True)

    print("\n===== NY_AM 단독 채택 시 (AM 만 매매) =====", flush=True)
    am = totals.get("NY_AM", 0.0)
    n_am = sum(len(per_sym[s]["NY_AM"]) for s in PAIRS)
    n_all = sum(len(per_sym[s][k]) for s in PAIRS for k in per_sym[s])
    print(f"  NY_AM net {am:+.1f}% ({n_am}건, 전체의 {100 * n_am / max(n_all, 1):.0f}%)", flush=True)
    print(f"  → 전체 {grand:+.1f}% 대비 "
          f"{'우월' if am > grand else '열위'} / 건당 "
          f"{am / max(n_am, 1):+.3f} vs 전체 {grand / max(n_all, 1):+.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
