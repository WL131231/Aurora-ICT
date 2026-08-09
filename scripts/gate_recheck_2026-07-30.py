"""#AUTONOMOUS 2026-07-30: 진입 게이트 재검증 — flip 수정 후 NY_PM·cond_align 이 여전히 유효한가.

## 왜 재검증인가
게이트 차단분 성적을 재보니(정합 하니스 + flip 1.5R):
    regime_filter  차단 117건 → net  **-91.7%** (승률 33%)  ✅ 손실 제거 = 제 일 함
    cond_align     차단  28건 → net **+269.4%**             ❌ 수익을 자름
    exclude_nypm   차단  83건 → net **+109.6%**             ❌ 수익을 자름
NY_PM 은 7/16 에 "+4.3%→+17.7%, 7페어 전부 개선" 으로 검증해 배포한 게이트다.
그때는 **flip 이 익절을 0.61R 에서 자르던 시절** — NY_PM 거래가 손실로 끝났다.
flip 을 고친 지금(#FLIP-MIN-R 1.5R) 그 거래들이 2R 까지 갈 수 있어 살아난 것으로 보인다.
같은 논리가 cond_align 에도 적용될 수 있다.

## 한계 (결론에 반드시 명시)
replay.py 에 exclude_nypm/cond_align 옵션이 없어 **사후 필터**로만 구현된다.
따라서 "게이트 ON" 이 근사이고 "OFF" 가 원본 타임라인이다. 게이트가 막은 자리에서
라이브가 다른 셋업을 잡았을 가능성(대체 효과)은 반영되지 않는다 —
동시 포지션 1개 제약 때문에 실제 개선폭은 이보다 작을 수 있다.

배터리: 연도별 · 페어별 · 롱숏 · 부트스트랩(쌍 재표본) · MDD.
판정: net 개선 + 연도 다수 + 페어 과반 + 양방향 + 부트스트랩 하한>0.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import cached_setup_timeline  # noqa: E402
from flip_ab_backtest import build_fvg_zones  # noqa: E402
from flip_verdict import apply_flip  # noqa: E402
from live_parity import (  # noqa: E402
    LIVE_BASE, LIVE_FLIP_MIN_R, PAIRS, gate_cond_align, gate_nypm, gate_regime,
    line, stat,
)

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

# 게이트 조합 — (라벨, regime, cond_align, nypm)
COMBOS = [
    ("현행 (전부 ON)", True, True, True),
    ("NY_PM OFF", True, True, False),
    ("cond_align OFF", True, False, True),
    ("둘 다 OFF", True, False, False),
    ("regime 만 ON", True, False, False),      # = 둘 다 OFF (표시용 중복 확인)
    ("전부 OFF", False, False, False),
    ("regime OFF만", False, True, True),
]


def build_cache():
    """페어별 (df5, 전체 trades, zones) — 게이트 적용 전 원본 타임라인."""
    cache = {}
    for sym in PAIRS:
        from bt_par import _load_full, _resample  # noqa: PLC0415
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**LIVE_BASE)
        bt = run_backtest_from_timeline(df5, cached_setup_timeline(df5, cfg, sym), cfg)
        cache[sym] = (df5, bt.trades, build_fvg_zones(df5))
        print(f"  {sym}: 원본 {len(bt.trades)}건", flush=True)
    return cache


def run_combo(cache, use_regime: bool, use_cond: bool, use_nypm: bool,
              side: str | None = None):
    """게이트 조합 적용 + flip 1.5R. 반환 (ts, net, sym) 리스트."""
    out = []
    for sym, (df5, trades, zones) in cache.items():
        kept = []
        for t in trades:
            ts = df5.index[t.entry_idx]
            if use_regime and not gate_regime(t, sym):
                continue
            if use_cond and not gate_cond_align(t, sym):
                continue
            if use_nypm and not gate_nypm(ts):
                continue
            if side:
                v = str(getattr(t.direction, "value", t.direction)).lower()
                if v != side:
                    continue
            kept.append(t)
        if not kept:
            continue
        res, _ = apply_flip(df5, kept, zones, mode="minr", min_w=4,
                            min_r=LIVE_FLIP_MIN_R, sym=sym)
        out.extend(res)
    return out


def main() -> int:
    print("=== 원본 타임라인 수집 (게이트 적용 전) ===", flush=True)
    cache = build_cache()

    print("\n\n===== 게이트 조합 비교 (전부 flip 1.5R 적용) =====", flush=True)
    base = None
    results = {}
    seen = set()
    for label, rg, cd, ny in COMBOS:
        keyt = (rg, cd, ny)
        if keyt in seen:
            continue
        seen.add(keyt)
        rows = run_combo(cache, rg, cd, ny)
        s = stat(rows)
        results[label] = (rows, s)
        if base is None:
            base = s
        d = "" if s is base else f"  현행대비 {s['net'] - base['net']:+.1f}%"
        print(f"  {label:<18} {line(s)}{d}", flush=True)

    # 최선 후보 선정 — net 최대가 아니라 **net/MDD × 연도 일관성**.
    # (net 최대는 위험 증가·방향 비대칭을 가릴 수 있다: "둘 다 OFF" 는 net 1위였으나
    #  롱이 -453% 악화해 적자 전환, MDD 증가, 연도 4/6 이었다.)
    def _score(k: str) -> tuple:
        s = results[k][1]
        return (s["ypos"], s["net"] / max(s["mdd"], 1e-9))
    cand = max((k for k in results if k != "현행 (전부 ON)"), key=_score)
    print(f"\n  → 최선 후보: **{cand}**", flush=True)

    cur_rows = results["현행 (전부 ON)"][0]
    new_rows = results[cand][0]
    cfgmap = {lb: (rg, cd, ny) for lb, rg, cd, ny in COMBOS}
    rg, cd, ny = cfgmap[cand]

    print(f"\n\n===== 배터리: {cand} =====", flush=True)
    print("\n[연도별]", flush=True)
    yc: dict[int, float] = {}
    yn: dict[int, float] = {}
    for ts, p, _ in cur_rows:
        yc[ts.year] = yc.get(ts.year, 0.0) + p * 100
    for ts, p, _ in new_rows:
        yn[ts.year] = yn.get(ts.year, 0.0) + p * 100
    imp_y = 0
    print(f"  {'연도':<6}{'현행':>10}{'후보':>10}{'개선':>10}", flush=True)
    for y in sorted(set(yc) | set(yn)):
        d = yn.get(y, 0.0) - yc.get(y, 0.0)
        imp_y += d > 0
        print(f"  {y:<6}{yc.get(y, 0.0):>+10.1f}{yn.get(y, 0.0):>+10.1f}{d:>+10.1f}",
              flush=True)
    print(f"  → 개선 연도 {imp_y}/{len(set(yc) | set(yn))}", flush=True)

    print("\n[페어별]", flush=True)
    imp_s = 0
    print(f"  {'페어':<8}{'현행':>10}{'후보':>10}{'개선':>10}", flush=True)
    for sym in PAIRS:
        c = sum(p for _, p, s in cur_rows if s == sym) * 100
        n = sum(p for _, p, s in new_rows if s == sym) * 100
        imp_s += (n - c) > 0
        print(f"  {sym.replace('USDT', ''):<8}{c:>+10.1f}{n:>+10.1f}{n - c:>+10.1f}",
              flush=True)
    print(f"  → 개선 페어 {imp_s}/{len(PAIRS)}", flush=True)

    print("\n[롱/숏]", flush=True)
    for sd in ("long", "short"):
        sc = stat(run_combo(cache, True, True, True, sd))
        sn = stat(run_combo(cache, rg, cd, ny, sd))
        if sc and sn:
            print(f"  {sd:<6} {sc['net']:+8.1f}% → {sn['net']:+8.1f}% "
                  f"({sn['net'] - sc['net']:+.1f}%)  RR {sc['rr']:.2f}→{sn['rr']:.2f}",
                  flush=True)

    print("\n[부트스트랩 — 추가된 거래의 기여]", flush=True)
    cur_keys = {(ts, s) for ts, _, s in cur_rows}
    added = np.array([p * 100 for ts, p, s in new_rows if (ts, s) not in cur_keys])
    if added.size:
        rngen = np.random.default_rng(0)
        boots = np.array([rngen.choice(added, size=added.size, replace=True).sum()
                          for _ in range(2000)])
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  추가 거래 {added.size}건 / 합계 {added.sum():+.1f}%", flush=True)
        print(f"  95% CI [{lo:+.1f}%, {hi:+.1f}%] → "
              f"{'✅ 하한>0 (개선 유의)' if lo > 0 else '⚠️ 하한<=0 (우연 배제 못함)'}",
              flush=True)
        print(f"  승률 {100 * (added > 0).mean():.0f}% / 평균 {added.mean():+.2f}%",
              flush=True)

    print("\n[MDD]", flush=True)
    sc, sn = stat(cur_rows), stat(new_rows)
    print(f"  현행 MDD {sc['mdd']:.1f}% (net/MDD {sc['net'] / sc['mdd']:.2f})", flush=True)
    print(f"  후보 MDD {sn['mdd']:.1f}% (net/MDD {sn['net'] / sn['mdd']:.2f})", flush=True)

    print("\n\n⚠️ 한계: 사후 필터라 '게이트가 막은 자리에서 다른 셋업을 잡았을' 대체 효과가",
          flush=True)
    print("   반영되지 않음(동시 포지션 1개). 실제 개선폭은 이보다 작을 수 있다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
