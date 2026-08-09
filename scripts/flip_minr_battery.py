"""#AUTONOMOUS 2026-07-30: flip 최소 R 게이트(B 1.5R) 검증 배터리 — 배포 전 최종.

후보: **flip 발동 시 이익이 1.5R 미만이면 flip 무시하고 원래 청산(TP/트레일)까지 홀드.**
근거 두 갈래(방법·데이터 모두 독립):
  ① 라이브 실측 반사실(n=29): flip 은 평균 0.61R 에서 절단, 홀드 시 2R TP 선착 72%
     → 기대값 0.61R → 1.17R (손익분기 1.17 과 일치)
  ② 라이브 정합 백테(n=126): 8시나리오 중 **현행(F1)이 최하위**, B1.5R 이 +239% 개선
     RR 1.20 → 1.94, 연도 흑자 4/6 → 5/6

배터리(7/29~30 확립 항목 전부):
  ①이웃 민감도  — min_r 1.0/1.2/1.5/1.8/2.0/2.5 (1.5 만 튀면 우연)
  ②연도별       — 6년 각각 (특정 시기 효과 배제)
  ③페어별       — 7페어 (한두 페어 요행 배제)
  ④롱/숏 분리   — 방향 편향 배제
  ⑤부트스트랩   — 거래 재표본 1000회, 개선폭 신뢰구간 (소표본 n=126 방어)
  ⑥MDD 장부     — 개선이 위험 증가와 교환된 것은 아닌지
판정: 이웃 연속 개선 + 연도 다수 + 페어 과반 + 양방향 + 부트스트랩 95% 하한 > 0.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flip_verdict import apply_flip  # noqa: E402
from flip_ab_backtest import build_fvg_zones  # noqa: E402
from live_parity import PAIRS, line, run_live_parity, stat  # noqa: E402

NEIGHBORS = (0.0, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5)   # 0.0 = 현행(무조건 flip)


def collect():
    """페어별 (df5, 정합통과 trades, zones) 캐시."""
    out = {}
    for sym in PAIRS:
        df5, kept, _ = run_live_parity(sym)
        out[sym] = (df5, kept, build_fvg_zones(df5))
        print(f"  {sym}: {len(kept)}건", flush=True)
    return out


def scenario(cache, min_r: float, side: str | None = None):
    """min_r 게이트 적용 결과. side='long'/'short' 면 그 방향만."""
    res = []
    for sym, (df5, kept, zones) in cache.items():
        sub = kept
        if side:
            sub = [t for t in kept
                   if str(getattr(t.direction, "value", t.direction)).lower() == side]
        if not sub:
            continue
        mode = "flip" if min_r <= 0 else "minr"
        r, _ = apply_flip(df5, sub, zones, mode=mode, min_w=4, min_r=min_r, sym=sym)
        res.extend(r)
    return res


def main() -> int:
    print("=== 라이브 정합 trade 수집 ===", flush=True)
    cache = collect()

    print("\n\n===== ① 이웃 민감도 (min_r 그리드) =====", flush=True)
    base = None
    nets = {}
    for mr in NEIGHBORS:
        s = stat(scenario(cache, mr))
        nets[mr] = s["net"] if s else 0.0
        tag = "현행(무조건 flip)" if mr == 0 else f"최소 {mr}R"
        if mr == 0:
            base = s
        d = "" if base is None or mr == 0 else f"  현행대비 {s['net'] - base['net']:+.1f}%"
        star = " ★" if mr == 1.5 else ""
        print(f"  {tag:<18} {line(s)}{d}{star}", flush=True)
    mono = all(nets[a] <= nets[b] for a, b in zip(NEIGHBORS[1:-1], NEIGHBORS[2:]))
    print(f"\n  → 1.5R 단독 스파이크 아님?  이웃 1.2R {nets[1.2]:+.0f} / 1.5R {nets[1.5]:+.0f} "
          f"/ 1.8R {nets[1.8]:+.0f}  ({'연속 개선 경향' if mono else '비단조 — 주의'})",
          flush=True)

    print("\n\n===== ② 연도별 (현행 vs 1.5R) =====", flush=True)
    cur = scenario(cache, 0.0)
    new = scenario(cache, 1.5)
    ys_c: dict[int, float] = {}
    ys_n: dict[int, float] = {}
    for ts, p, _ in cur:
        ys_c[ts.year] = ys_c.get(ts.year, 0.0) + p * 100
    for ts, p, _ in new:
        ys_n[ts.year] = ys_n.get(ts.year, 0.0) + p * 100
    print(f"  {'연도':<6} {'현행':>10} {'1.5R':>10} {'개선':>10}", flush=True)
    imp_y = 0
    for y in sorted(ys_c):
        d = ys_n.get(y, 0.0) - ys_c[y]
        if d > 0:
            imp_y += 1
        print(f"  {y:<6} {ys_c[y]:>+10.1f} {ys_n.get(y, 0.0):>+10.1f} {d:>+10.1f}", flush=True)
    print(f"  → 개선 연도 {imp_y}/{len(ys_c)}", flush=True)

    print("\n\n===== ③ 페어별 =====", flush=True)
    print(f"  {'페어':<10} {'현행':>10} {'1.5R':>10} {'개선':>10}", flush=True)
    imp_s = 0
    for sym in PAIRS:
        c = sum(p for _, p, s in cur if s == sym) * 100
        n = sum(p for _, p, s in new if s == sym) * 100
        d = n - c
        if d > 0:
            imp_s += 1
        print(f"  {sym.replace('USDT', ''):<10} {c:>+10.1f} {n:>+10.1f} {d:>+10.1f}", flush=True)
    print(f"  → 개선 페어 {imp_s}/{len(PAIRS)}", flush=True)

    print("\n\n===== ④ 롱/숏 분리 =====", flush=True)
    for sd in ("long", "short"):
        c = stat(scenario(cache, 0.0, sd))
        n = stat(scenario(cache, 1.5, sd))
        if c and n:
            print(f"  {sd:<6} 현행 {c['net']:+8.1f}% → 1.5R {n['net']:+8.1f}% "
                  f"({n['net'] - c['net']:+.1f}%)  RR {c['rr']:.2f}→{n['rr']:.2f}", flush=True)

    print("\n\n===== ⑤ 부트스트랩 (거래 재표본 1000회) =====", flush=True)
    # 같은 trade 집합의 쌍(현행, 1.5R) 차이를 재표본 — 소표본 n=126 방어.
    key = {}
    for ts, p, s in cur:
        key[(ts, s)] = [p, None]
    for ts, p, s in new:
        if (ts, s) in key:
            key[(ts, s)][1] = p
    diffs = np.array([v[1] - v[0] for v in key.values() if v[1] is not None]) * 100
    rng = np.random.default_rng(0)
    boots = np.array([rng.choice(diffs, size=len(diffs), replace=True).sum()
                      for _ in range(1000)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  쌍 표본 {len(diffs)}건 / 총 개선 {diffs.sum():+.1f}%", flush=True)
    print(f"  95% 신뢰구간 [{lo:+.1f}%, {hi:+.1f}%]  "
          f"→ {'✅ 하한 > 0 (개선 유의)' if lo > 0 else '⚠️ 하한 <= 0 (우연 배제 못함)'}",
          flush=True)
    print(f"  개선된 거래 {100 * (diffs > 0).mean():.0f}% / 악화 {100 * (diffs < 0).mean():.0f}% "
          f"/ 무변화 {100 * (diffs == 0).mean():.0f}%", flush=True)

    print("\n\n===== ⑥ MDD 장부 =====", flush=True)
    sc, sn = stat(cur), stat(new)
    print(f"  현행 MDD {sc['mdd']:.1f}% (net/MDD {sc['net'] / sc['mdd']:.2f})", flush=True)
    print(f"  1.5R MDD {sn['mdd']:.1f}% (net/MDD {sn['net'] / sn['mdd']:.2f})", flush=True)
    print(f"  → {'위험 감소' if sn['mdd'] < sc['mdd'] else '위험 증가 — 교환 확인 필요'}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
