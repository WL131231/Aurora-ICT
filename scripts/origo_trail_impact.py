"""Origo trailing 도입 검증 — FST #1 자율연구.

FST 2026-07-01 라이브 진단: Origo RR 0.40. 손실은 전부 sl_hit(잦고 큼), tp_hit
도달은 드묾(추세 이익 못 먹음), flip_close 는 전승이나 작음(+1.9). 고정 TP 가
추세를 못 잡는 게 병 → trailing 으로 추세 끝까지 따라가면 RR 개선되는지 검증.

trailing 은 이미 replay._simulate_exit 에 구현(trail_trigger/trail_dist, risk0
배수). 진입 timeline 은 동일(청산만 변형)이라 캐시 1회 빌드 후 재사용.

비교(안정 설정 0.707/min_rr2.0, 7페어, cisd+po3, sl x3):
    - 현행: 고정 TP(tp_rr_override=1.0) + EMA flip.
    - trail x: TP 멀리(5R) + trigger/dist 배수 스윕 → 트레일 청산 위주.
    - partial+trail: 1R 절반 익절(본전이동) + 나머지 트레일.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/origo_trail_impact.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
OTE, MIN_RR = 0.707, 2.0  # 안정(1.2 적용안)
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=OTE, min_rr=MIN_RR,
)
# (라벨, 청산 override dict)
VARIANTS = [
    ("현행 고정tp1", dict(tp_rr_override=1.0)),
    ("trail 1.0/1.0", dict(tp_rr_override=5.0, trail_trigger=1.0, trail_dist=1.0)),
    ("trail 1.5/1.0", dict(tp_rr_override=5.0, trail_trigger=1.5, trail_dist=1.0)),
    ("trail 1.0/0.5", dict(tp_rr_override=5.0, trail_trigger=1.0, trail_dist=0.5)),
    ("trail 2.0/1.5", dict(tp_rr_override=5.0, trail_trigger=2.0, trail_dist=1.5)),
    ("partial1R+trail1/1", dict(tp_rr_override=5.0, partial_tp_rr=1.0, partial_be=True,
                                trail_trigger=1.0, trail_dist=1.0)),
]


def _metrics(trades, days):
    ts = [t for t in trades]
    if not ts:
        return None
    cum = peak = mdd = 0.0
    for t in sorted(ts, key=lambda t: t.exit_idx):
        cum += t.net_pnl_pct
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    wins = [t.net_pnl_pct for t in ts if t.net_pnl_pct > 0]
    losses = [t.net_pnl_pct for t in ts if t.net_pnl_pct < 0]
    n = len(ts)
    aw = sum(wins) / len(wins) if wins else 0.0
    al = sum(losses) / len(losses) if losses else 0.0
    rr = aw / -al if al < 0 else 0.0
    return dict(usdt=cum * SEED / 100, mdd=mdd * SEED / 100,
                wr=len(wins) / n * 100, n=n, rr=rr, freq=n / days,
                aw=aw, al=al)


def main() -> int:
    agg = {v[0]: {"usdt": 0.0, "mdd": 0.0, "n": 0, "nwin": 0, "days": 0.0,
                  "gw": 0.0, "gl": 0.0, "nl": 0} for v in VARIANTS}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        days = len(df5) / 288.0
        ttl = 12 if sym == "BTCUSDT" else 6
        # 진입 timeline 은 변형 무관 동일 — 1회 빌드.
        tl = cached_setup_timeline(
            df5, BacktestConfig(**{**BASE, "entry_ttl_bars": ttl}), sym)
        for label, ov in VARIANTS:
            cfg = {**BASE, "entry_ttl_bars": ttl, **ov}
            bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
            m = _metrics(bt.trades, days)
            if not m:
                continue
            a = agg[label]
            a["usdt"] += m["usdt"]; a["mdd"] += m["mdd"]; a["n"] += m["n"]
            a["nwin"] += round(m["wr"] / 100 * m["n"]); a["days"] += days
            a["gw"] += m["aw"] * round(m["wr"] / 100 * m["n"])
            a["gl"] += m["al"] * (m["n"] - round(m["wr"] / 100 * m["n"]))
        print(f"  {sym} done", flush=True)

    lines = [
        "===== Origo trailing 검증 (안정 0.707/rr2, 7페어, 시드1000) =====",
        "진입 동일(cisd+po3 sl x3). 청산만 변형. RR=평균익/평균손.",
        "",
        f"{'방식':<20}{'USDT':>8}{'최대DD':>8}{'승률':>6}{'RR':>6}{'거래':>7}",
    ]
    for label, _ in VARIANTS:
        a = agg[label]
        wr = a["nwin"] / a["n"] * 100 if a["n"] else 0.0
        nloss = a["n"] - a["nwin"]
        aw = a["gw"] / a["nwin"] if a["nwin"] else 0.0
        al = a["gl"] / nloss if nloss else 0.0
        rr = aw / -al if al < 0 else 0.0
        lines.append(
            f"{label:<20}{a['usdt']:>+8.0f}{a['mdd']:>7.0f}↓{wr:>5.0f}%"
            f"{rr:>6.2f}{a['n']:>7d}")

    txt = "\n".join(lines)
    with open("origo_trail_impact_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 origo_trail_impact_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
