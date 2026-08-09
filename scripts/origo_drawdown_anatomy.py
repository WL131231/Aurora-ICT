"""#AUTONOMOUS 2026-08-06: 낙폭 구간 해부 — 무엇이 계좌를 녹이는가.

파트너: "그 낙폭 구간을 찾아볼래? 그런 구간은 배율로 잡을 게 아니라 로직으로
잡아야겠는데." — 정확한 판단이다. 배율을 낮추면 낙폭과 수익이 **함께** 줄지만,
원인을 찾아 로직으로 막으면 배율을 유지한 채 낙폭만 깎을 수 있다.

## 방법
7x·동시보유 반영·DD 스로틀 적용(= 배포될 설정)으로 자산 곡선을 만들고,
**상위 낙폭 구간 3개**를 뽑아 그 구간의 거래를 정상 구간과 대조한다.

보는 축 — "로직으로 막을 수 있는가"에 답하는 것들:
    · 연속 손실 길이 (한 방인가, 누적인가)
    · 페어 쏠림 (특정 종목이 원인인가)
    · 방향 쏠림 (롱/숏 어느 쪽인가)
    · 시기·요일·시간대
    · 동시 보유 수 (한꺼번에 여러 개 물렸는가)
    · 손실 크기 분포 (평소보다 큰 손실인가, 평소 손실이 많은 건가)

마지막 축이 특히 중요하다. **"큰 손실 몇 건"이면 손절 로직**, **"평범한 손실
연속"이면 진입 빈도·상관** 문제다. 처방이 완전히 다르다.
"""

from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import LIVE_BASE, PAIRS, run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

SIZE = LIVE_BASE["size_pct"]
LEV = 7.0                      # 파트너 결정 — 배포 예정값
DD_PCT, DD_FACTOR = 0.25, 0.7
TOP_N = 3


def collect():
    rows = []
    for sym in PAIRS:
        df5, kept, _ = run_live_parity(sym)
        idx = df5.index
        for t in kept:
            ent = idx[t.entry_idx]
            ex = idx[min(t.exit_idx, len(idx) - 1)]
            rows.append({
                "sym": sym, "entry": ent, "exit": ex,
                "raw": float(t.raw_pnl_pct),
                "dir": str(getattr(t.direction, "value", t.direction)).lower(),
                "outcome": str(getattr(t, "outcome", "")),
            })
    rows.sort(key=lambda r: r["entry"])
    return rows


def equity_curve(rows):
    """거래별 자산 곡선 — 동시보유로 size 를 나누고 DD 스로틀 적용."""
    n = len(rows)
    conc = np.empty(n)
    for i, r in enumerate(rows):
        conc[i] = 1.0 + sum(
            1 for j, q in enumerate(rows)
            if j != i and q["entry"] <= r["exit"] and q["exit"] >= r["entry"]
        )
    eq, peak = 1.0, 1.0
    curve, steps = [1.0], []
    for i, r in enumerate(rows):
        sz = SIZE / conc[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        step = r["raw"] * sz * LEV - 2.0 * TAKER_FEE_PCT * sz * LEV
        eq *= (1.0 + step)
        peak = max(peak, eq)
        curve.append(eq)
        steps.append(step)
    return np.array(curve), np.array(steps), conc


def top_drawdowns(curve, k=TOP_N):
    """겹치지 않는 상위 낙폭 구간 — (시작idx, 바닥idx, 낙폭%)."""
    out = []
    used = np.zeros(len(curve), dtype=bool)
    for _ in range(k):
        peak_i = trough_i = -1
        best = 0.0
        cur_peak_i = 0
        for i in range(len(curve)):
            if used[i]:
                cur_peak_i = i
                continue
            if curve[i] > curve[cur_peak_i]:
                cur_peak_i = i
            dd = 1.0 - curve[i] / curve[cur_peak_i]
            if dd > best:
                best, peak_i, trough_i = dd, cur_peak_i, i
        if peak_i < 0 or best <= 0:
            break
        out.append((peak_i, trough_i, 100.0 * best))
        used[peak_i:trough_i + 1] = True
    return out


def describe(tag, rows, steps, conc, lo, hi):
    """구간 [lo, hi) 의 거래 특성."""
    sub = rows[lo:hi]
    st = steps[lo:hi]
    if not sub:
        print(f"  {tag}: 거래 없음", flush=True)
        return
    losses = [s for s in st if s < 0]
    # 최대 연속 손실
    run = best_run = 0
    for s in st:
        run = run + 1 if s < 0 else 0
        best_run = max(best_run, run)
    syms = Counter(r["sym"] for r in sub)
    dirs = Counter(r["dir"] for r in sub)
    outs = Counter(r["outcome"] for r in sub)
    top_sym, top_cnt = syms.most_common(1)[0]
    print(f"  {tag}", flush=True)
    print(f"    거래 {len(sub)}건 · 승률 {100 * sum(1 for s in st if s > 0) / len(st):.0f}%"
          f" · 최대 연속손실 {best_run}건 · 동시보유 평균 {conc[lo:hi].mean():.2f}",
          flush=True)
    print(f"    손실 {len(losses)}건 · 평균 {100 * np.mean(losses) if losses else 0:.2f}%"
          f" · 최악 {100 * min(st):.2f}% · 최대이익 {100 * max(st):.2f}%", flush=True)
    print(f"    페어 쏠림: {top_sym} {top_cnt}/{len(sub)}건"
          f" ({100 * top_cnt / len(sub):.0f}%) · 전체 {dict(syms)}", flush=True)
    print(f"    방향 {dict(dirs)} · 청산유형 {dict(outs)}", flush=True)
    print(f"    기간 {sub[0]['entry'].date()} ~ {sub[-1]['exit'].date()}", flush=True)


def main() -> int:
    rows = collect()
    curve, steps, conc = equity_curve(rows)
    print(f"=== Origo 낙폭 해부 (7x · 동시보유 반영 · DD 스로틀) ===", flush=True)
    print(f"  거래 {len(rows)}건 · 최종 {curve[-1]:.2f}배", flush=True)

    dds = top_drawdowns(curve)
    print(f"\n### 상위 낙폭 구간 {len(dds)}개", flush=True)
    all_bad = set()
    for k, (p, t, d) in enumerate(dds, 1):
        span_days = (rows[min(t, len(rows) - 1)]["exit"]
                     - rows[max(p - 1, 0)]["entry"]).days
        print(f"\n[{k}위] 낙폭 {d:.1f}%  ({curve[p]:.2f}배 → {curve[t]:.2f}배)"
              f"  거래 #{p}~#{t}  약 {span_days}일", flush=True)
        describe("구간 내역", rows, steps, conc, p, t)
        all_bad.update(range(p, t))

    good = [i for i in range(len(rows)) if i not in all_bad]
    print(f"\n### 대조 — 낙폭 구간 밖 (정상 {len(good)}건)", flush=True)
    if good:
        gsteps = steps[good]
        gl = [s for s in gsteps if s < 0]
        print(f"    승률 {100 * sum(1 for s in gsteps if s > 0) / len(gsteps):.0f}%"
              f" · 손실 평균 {100 * np.mean(gl) if gl else 0:.2f}%"
              f" · 최악 {100 * min(gsteps):.2f}%", flush=True)
        gs = Counter(rows[i]["sym"] for i in good)
        gd = Counter(rows[i]["dir"] for i in good)
        print(f"    페어 {dict(gs)}", flush=True)
        print(f"    방향 {dict(gd)}", flush=True)

    # 전체에서 손실 상위 10건이 낙폭 구간에 몰려 있나
    order = np.argsort(steps)[:10]
    inbad = sum(1 for i in order if i in all_bad)
    print(f"\n### 최악 손실 10건 중 {inbad}건이 낙폭 구간 안에 있다", flush=True)
    for i in order[:10]:
        mark = "★" if i in all_bad else " "
        print(f"  {mark} #{i:3d} {rows[i]['sym']:<10}{rows[i]['dir']:<6}"
              f"{100 * steps[i]:>7.2f}%  {rows[i]['entry'].date()}"
              f"  동시보유 {conc[i]:.0f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
