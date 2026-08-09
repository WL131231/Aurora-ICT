"""안정형 평가 재설계 — 시드 1000 USDT + 최대드로다운 + 양수유지율 + 승률 + 빈도.

파트너(6/22): net 후순위, 안정성(최대DD↓)+승률+빈도+꾸준한 양수 우선. 대규모 탐색서
나온 안정형 후보(깊은진입 ote × 짧은TP tp1.0)를 누적손익곡선으로 평가. timeline
캐시 재사용(즉시). 시드 1000 USDT 기준 USDT 환산.

평가 지표:
- USDT(시드1000): net% × 10 (1%=10USDT). 누적·최종.
- 최대DD(%): 누적손익곡선 고점 대비 최대 낙폭 (시드 깎이는 폭 — 안정성 핵심).
- 양수월%: 거래를 30일 버킷으로 묶어 버킷 net>0 비율 (사용자가 "계속 양수" 체감).
- 승률·빈도(1일).

후보: 안정형(0.886/1.0/0.707 × tp1.0 × ttl6/12) + 현행(0.5/swing/ttl6) 대조. BTC.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/eval_stable.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

SYM = "BTCUSDT"
SEED = 1000.0
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)
# (라벨, ote, min_rr, tp_rr, ttl)
CANDS = [
    ("현행(0.5/swing/6)", 0.5, 2.5, 0.0, 6),
    ("안정 0.886/tp1/6", 0.886, 2.0, 1.0, 6),
    ("안정 0.886/tp1/18", 0.886, 2.0, 1.0, 18),
    ("안정 1.0/tp1/18", 1.0, 2.5, 1.0, 18),
    ("안정 0.707/tp1/6", 0.707, 2.0, 1.0, 6),
    ("0.886/tp1.5/6", 0.886, 2.0, 1.5, 6),
    ("0.886/swing/6", 0.886, 2.0, 0.0, 6),
]


def _metrics(trades):
    """체결순(exit_idx) 누적손익곡선 → USDT·최대DD·양수월%·승률·n."""
    ts = sorted(trades, key=lambda t: t.exit_idx)
    if not ts:
        return None
    cum = 0.0
    peak = 0.0
    mdd = 0.0
    eq = []  # (exit_idx, cum%)
    for t in ts:
        cum += t.net_pnl_pct
        peak = max(peak, cum)
        mdd = max(mdd, peak - cum)
        eq.append((t.exit_idx, cum))
    n = len(ts)
    nwin = sum(1 for t in ts if t.net_pnl_pct > 0)
    # 30일(=288*30 봉) 버킷 양수율
    bucket = 288 * 30
    buckets = {}
    for t in ts:
        b = t.exit_idx // bucket
        buckets[b] = buckets.get(b, 0.0) + t.net_pnl_pct
    pos_mon = sum(1 for v in buckets.values() if v > 0)
    mon_pct = (pos_mon / len(buckets) * 100) if buckets else 0.0
    return {
        "net": cum, "usdt": cum * SEED / 100, "mdd": mdd, "mdd_usdt": mdd * SEED / 100,
        "wr": nwin / n * 100, "n": n, "mon": mon_pct, "nmon": len(buckets),
    }


def main() -> int:
    df5 = _resample(_load_full(SYM))
    days = len(df5) / 288.0
    print(f"BTC {len(df5)} ({days:.0f}일), 시드 {SEED:.0f} USDT", flush=True)
    rows = []
    for label, ote, mr, tp, ttl in CANDS:
        detect_cfg = {**BASE, "ote_level": ote, "min_rr": mr, "entry_ttl_bars": 6}
        tl = cached_setup_timeline(df5, BacktestConfig(**detect_cfg), SYM)
        cfg = {**BASE, "ote_level": ote, "min_rr": mr, "entry_ttl_bars": ttl, "tp_rr_override": tp}
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**cfg))
        m = _metrics(bt.trades)
        if m:
            m["label"] = label
            m["freq"] = m["n"] / days
            rows.append(m)
        print(f"  {label} done", flush=True)

    lines = ["===== 안정형 평가 (BTC 5년, 시드 1000 USDT) =====",
             f"{'조합':<18} {'USDT':>8} {'최대DD':>9} {'승률':>6} {'양수월':>7} {'1일빈도':>8} {'거래':>5}"]
    for m in rows:
        lines.append(
            f"{m['label']:<18} {m['usdt']:+8.0f} {m['mdd_usdt']:8.0f}↓ {m['wr']:5.0f}% "
            f"{m['mon']:5.0f}%({m['nmon']}) {m['freq']:7.2f}회 {m['n']:5d}"
        )
    lines.append("\n※ 안정성=최대DD(시드 깎이는 폭)↓, 양수월%=30일 버킷 net>0 비율(사용자 체감).")
    lines.append("  USDT=시드1000 기준 5년 누적 순손익. 승률·빈도 함께.")
    # 안정형 점수(최대DD↓ + 승률↑ + 양수월↑) 랭킹
    lines.append("\n[안정성 랭킹 (최대DD 작은 순)]")
    for m in sorted(rows, key=lambda x: x["mdd"]):
        lines.append(f"  {m['label']}: 최대DD {m['mdd_usdt']:.0f}USDT, 승률{m['wr']:.0f}%, 양수월{m['mon']:.0f}%, USDT{m['usdt']:+.0f}")

    txt = "\n".join(lines)
    with open("eval_stable_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 eval_stable_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
