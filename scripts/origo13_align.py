"""Origo 1.3 라이브 정합 스윕 — FST #2 자율연구 (2026-07-02).

배포된 Origo 1.3(conf5+sl4)의 백테 검증(+124)은 min_rr 2.0·고정 단일 TP 가정.
그러나 라이브 구독제는 (a) min_rr 2.5 강제, (b) 1.5R 분할익절(partial TP) 사용.
이 두 괴리가 +124 를 지키는지/깎는지 정합 검증 + 후속 축(conf6/sl5/trail) 확장.

min_rr 은 timeline 캐시 키 포함 → 2.5/3.0 은 페어별 재빌드 필요(수십분+).
Pool(4) 페어 병렬로 벽시계 단축, 빌드 결과는 tl_cache 에 남아 재실행 즉시.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/origo13_align.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
BASE13 = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=1.0,
)
# (라벨, override) — rr2.5 계열이 라이브 정합.
VARIANTS = [
    ("rr2.0 고정tp (배포 검증치)", {}),
    ("rr2.5 고정tp (라이브 정합)", dict(min_rr=2.5)),
    ("rr2.5 + partial1.5R+BE (라이브 분할익절)",
     dict(min_rr=2.5, partial_tp_rr=1.5, partial_be=True)),
    ("rr2.5 + trail2.0/1.5",
     dict(min_rr=2.5, tp_rr_override=5.0, trail_trigger=2.0, trail_dist=1.5)),
    ("rr2.5 + conf6", dict(min_rr=2.5, min_confluence=6)),
    ("rr2.5 + sl5", dict(min_rr=2.5, sl_dist_mult=5.0)),
    ("rr2.5 + stale2", dict(min_rr=2.5, setup_stale_bars=2)),
    ("rr3.0 고정tp", dict(min_rr=3.0)),
]


def _pair_worker(sym: str):
    """한 페어: 변형별 (timeline 캐시빌드 → 재생) — Pool top-level."""
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    out = {}
    for label, ov in VARIANTS:
        cfg = {**BASE13, "entry_ttl_bars": ttl, **ov}
        tl = cached_setup_timeline(df5, BacktestConfig(**cfg), sym)
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
        ts = list(bt.trades)
        wins = [t.net_pnl_pct for t in ts if t.net_pnl_pct > 0]
        losses = [t.net_pnl_pct for t in ts if t.net_pnl_pct < 0]
        cum = peak = mdd = 0.0
        for t in sorted(ts, key=lambda t: t.exit_idx):
            cum += t.net_pnl_pct
            peak = max(peak, cum)
            mdd = max(mdd, peak - cum)
        out[label] = dict(cum=cum, mdd=mdd, n=len(ts), nwin=len(wins),
                          gw=sum(wins), gl=sum(losses), days=len(df5) / 288.0)
        print(f"  {sym} {label} done n={len(ts)}", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)

    lines = ["===== Origo 1.3 라이브 정합 스윕 (7페어 5년, 시드1000) =====",
             "BASE13 = conf5 + SLx4 + ote0.707 + stale3 + 킬존ON.",
             "",
             f"{'변형':<38}{'USDT':>8}{'DD':>7}{'승률':>6}{'RR':>6}{'거래':>7}{'빈도':>7}"]
    for label, _ in VARIANTS:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0, "nwin": 0, "gw": 0.0, "gl": 0.0, "days": 0.0}
        for _sym, out in results:
            if out is None or label not in out:
                continue
            for k in tot:
                tot[k] += out[label][k]
        n = tot["n"]
        wr = tot["nwin"] / n * 100 if n else 0.0
        aw = tot["gw"] / tot["nwin"] if tot["nwin"] else 0.0
        al = tot["gl"] / (n - tot["nwin"]) if (n - tot["nwin"]) else 0.0
        rr = aw / -al if al < 0 else 0.0
        freq = n / (tot["days"] / len(PAIRS)) if tot["days"] else 0.0
        lines.append(f"{label:<36}{tot['cum'] * SEED / 100:>+8.0f}{tot['mdd'] * SEED / 100:>7.0f}"
                     f"{wr:>5.0f}%{rr:>6.2f}{n:>7d}{freq:>6.2f}회")
    txt = "\n".join(lines)
    with open("origo13_align_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
