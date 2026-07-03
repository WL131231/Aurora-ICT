"""Origo 1.4 후속 — 트레일 정밀 그리드 + 킬존 선별 (FST #2 자율연구 3라운드).

A. 트레일 그리드: 배포값 2.0/1.5 는 4개 후보 중 최선이었을 뿐 — trigger×dist
   12조합 정밀 스윕으로 더 나은 지점/과최적화 민감도 확인 (전부 재생만, 캐시).
B. 킬존 선별: 라이브 실측 NY_PM(02-05 KST) 승률 0~21% 최악 (6/24 연구 재확인).
   타임라인 setup 의 진입시각(KST)을 버킷별로 제외한 재생으로 백테 교차검증.
   백테도 같은 방향이면 라이브 킬존 선별(NY_PM 제외)이 다음 처방 후보.

BASE = Origo 1.4 정합 (conf5/SLx4/rr2.0 + trail). 7페어 5년.
사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/origo_kz_trail_grid.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import os
import sys
from datetime import timedelta
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
SEED = 1000.0
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=5.0,
    trail_trigger=2.0, trail_dist=1.5,
)
# A. 트레일 그리드 (trigger, dist)
GRID = [(tr, d) for tr in (1.5, 2.0, 2.5, 3.0) for d in (1.0, 1.5, 2.0)]
# B. 킬존 버킷 (KST 시간대) — 하나씩 제외
KZ = {"asia": (8, 12), "london": (16, 20), "ny_am": (21, 24), "ny_pm": (2, 5)}


def _kz_filtered(tl, df, exclude: str):
    """타임라인 복사본 — 진입시각(KST hour)이 exclude 버킷이면 setup 제거."""
    a, b = KZ[exclude]
    hours = (df.index + timedelta(hours=9)).hour
    out = list(tl)
    for i, item in enumerate(out):
        if item is not None and a <= hours[i] < b:
            out[i] = None
    return out


def _metrics(bt):
    ts = list(bt.trades)
    wins = [t.net_pnl_pct for t in ts if t.net_pnl_pct > 0]
    losses = [t.net_pnl_pct for t in ts if t.net_pnl_pct < 0]
    cum = peak = mdd = 0.0
    for t in sorted(ts, key=lambda t: t.exit_idx):
        cum += t.net_pnl_pct
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
    return dict(cum=cum, mdd=mdd, n=len(ts), nwin=len(wins),
                gw=sum(wins), gl=sum(losses))


def _pair_worker(sym: str):
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    ttl = 12 if sym == "BTCUSDT" else 6
    base_cfg = {**BASE, "entry_ttl_bars": ttl}
    tl = cached_setup_timeline(df5, BacktestConfig(**base_cfg), sym)
    out = {}
    # A. 트레일 그리드
    for tr, d in GRID:
        cfg = BacktestConfig(**{**base_cfg, "trail_trigger": tr, "trail_dist": d})
        out[f"trail {tr}/{d}"] = _metrics(run_backtest_from_timeline(df5, tl, cfg))
    # B. 킬존 제외 (배포값 트레일 고정)
    cfg = BacktestConfig(**base_cfg)
    for kz in KZ:
        ftl = _kz_filtered(tl, df5, kz)
        out[f"기준-{kz}제외"] = _metrics(run_backtest_from_timeline(df5, ftl, cfg))
    out["기준(전체킬존)"] = _metrics(run_backtest_from_timeline(df5, tl, cfg))
    out["_days"] = {"cum": len(df5) / 288.0, "mdd": 0, "n": 0, "nwin": 0, "gw": 0, "gl": 0}
    print(f"  {sym} done", flush=True)
    return sym, out


def main() -> int:
    with Pool(4) as p:
        results = p.map(_pair_worker, PAIRS)
    labels = ([f"trail {tr}/{d}" for tr, d in GRID]
              + ["기준(전체킬존)"] + [f"기준-{k}제외" for k in KZ])
    days_tot = sum(out["_days"]["cum"] for _s, out in results)
    lines = ["===== Origo 1.4 트레일 그리드 + 킬존 선별 (7페어 5년) =====",
             "BASE = conf5/SLx4/rr2.0 + trail(기준 2.0/1.5).",
             "",
             f"{'변형':<22}{'USDT':>8}{'DD':>7}{'승률':>6}{'RR':>6}{'거래':>7}{'빈도':>7}"]
    for label in labels:
        tot = {"cum": 0.0, "mdd": 0.0, "n": 0, "nwin": 0, "gw": 0.0, "gl": 0.0}
        for _sym, out in results:
            m = out.get(label)
            if m:
                for k in tot:
                    tot[k] += m[k]
        n = tot["n"]
        wr = tot["nwin"] / n * 100 if n else 0.0
        aw = tot["gw"] / tot["nwin"] if tot["nwin"] else 0.0
        al = tot["gl"] / (n - tot["nwin"]) if (n - tot["nwin"]) else 0.0
        rr = aw / -al if al < 0 else 0.0
        freq = n / (days_tot / len(PAIRS)) if days_tot else 0.0
        lines.append(f"{label:<20}{tot['cum'] * SEED / 100:>+8.0f}{tot['mdd'] * SEED / 100:>7.0f}"
                     f"{wr:>5.0f}%{rr:>6.2f}{n:>7d}{freq:>6.2f}회")
    txt = "\n".join(lines)
    with open("origo_kz_trail_grid_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
