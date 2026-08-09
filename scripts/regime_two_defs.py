"""#AUTONOMOUS 2026-07-30: "횡보" 두 정의의 2×2 분해 — 파트너 질문 정면 확인.

파트너 질문: "횡보에서 돈을 번다고? 우리 횡보 진입 차단 게이트 있었잖아?"
→ 정당한 의문. 다음 두 사실이 동시에 성립하는 것처럼 보인다:
   (A) regime_filter(|entry_trend_pct| < q33) 차단이 net 흑자의 필수조건 (6/23 검증)
   (B) CSI/ADX/BBW 가 "횡보" 라 판정한 거래가 오히려 수익군 (7/27, 오늘 재확인)

가설: **두 게이트가 서로 다른 현상을 재고 있다.**
   q33  = 진입 직전 20봉 **추세 크기**(방향성 부재) → 진입해도 안 움직임 → 손실
   CSI  = 변동성 수축·추세강도 부재(눌린 구간) → Origo(FVG 되돌림)의 최적 진입 환경
   Origo 본질 = 압축 후 킬존 확장 초입을 먹는 것이므로 (B) 가 이치에 맞다.

검증: 두 판정의 **겹침(교차표)** + 4칸 각각의 성과.
   가설이 참이면 ① 겹침이 낮고 ② q33만 O 칸은 손실 ③ CSI만 O 칸은 수익.
   가설이 거짓(같은 것을 잼)이면 겹침이 높고 4칸 성과가 한 방향으로 정렬된다.
추가: 두 재료의 상관, 그리고 각 칸의 연도·페어 분산.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chop_state_index import fit_csi  # noqa: E402
from regime_swap_csi import PAIRS, collect  # noqa: E402


def main() -> int:
    model = fit_csi(PAIRS)
    ds = []
    for sym in PAIRS:
        try:
            ds.append(collect(sym, model))
        except Exception as e:  # noqa: BLE001
            print(f"{sym} 실패: {e}", flush=True)
    D = pd.concat(ds, ignore_index=True).dropna(subset=["csi", "trend"])
    q33 = D.groupby("sym").trend.quantile(1 / 3).to_dict()
    D["q33_chop"] = [r.trend < q33.get(r.sym, 0.0) for r in D.itertuples()]

    for thr in (0.5, 0.6):
        D["csi_chop"] = D.csi >= thr
        print(f"\n\n########## CSI 임계 {thr} ##########", flush=True)
        # 교차표
        ct = pd.crosstab(D.q33_chop, D.csi_chop)
        print("\n[겹침 교차표] 행=q33 횡보, 열=CSI 횡보", flush=True)
        print(ct.to_string(), flush=True)
        both = int(((D.q33_chop) & (D.csi_chop)).sum())
        union = int(((D.q33_chop) | (D.csi_chop)).sum())
        jac = both / max(union, 1)
        # 두 이진 판정의 상관(phi)
        phi = np.corrcoef(D.q33_chop.astype(int), D.csi_chop.astype(int))[0, 1]
        print(f"\n  Jaccard 겹침 = {100 * jac:.1f}%   phi 상관 = {phi:+.3f}", flush=True)
        print("  → 겹침·상관이 낮으면 '두 게이트는 다른 현상을 잰다' 가설 지지", flush=True)

        print("\n[2×2 성과]", flush=True)
        print(f"  {'q33':<6} {'CSI':<6} {'n':>5} {'net':>9} {'건당':>8} {'승률':>6} "
              f"{'연도+':>6} {'페어+':>6}", flush=True)
        for q in (True, False):
            for c in (True, False):
                s = D[(D.q33_chop == q) & (D.csi_chop == c)]
                if s.empty:
                    continue
                ys = s.groupby(s.ts.dt.year).pnl.sum()
                sy = s.groupby("sym").pnl.sum()
                print(f"  {'횡보' if q else '추세':<6} {'횡보' if c else '추세':<6} "
                      f"{len(s):5d} {s.pnl.sum():+9.1f} {s.pnl.mean():+7.3f} "
                      f"{100 * (s.pnl > 0).mean():5.0f}% "
                      f"{int((ys > 0).sum())}/{len(ys):<4} {int((sy > 0).sum())}/{len(sy)}",
                      flush=True)

        print("\n[단독 효과] 각 판정이 '홀로' 잡아낸 거래", flush=True)
        only_q = D[(D.q33_chop) & (~D.csi_chop)]
        only_c = D[(~D.q33_chop) & (D.csi_chop)]
        print(f"  q33 만 횡보: n={len(only_q):4d} net={only_q.pnl.sum():+8.1f} "
              f"건당={only_q.pnl.mean():+.3f}  ← 차단이 이득이면 음수여야", flush=True)
        print(f"  CSI 만 횡보: n={len(only_c):4d} net={only_c.pnl.sum():+8.1f} "
              f"건당={only_c.pnl.mean():+.3f}  ← 차단이 손해면 양수", flush=True)

    # 연속값 상관 — trend 크기 vs CSI 확률
    print("\n\n[연속값 상관] |entry_trend_pct| vs CSI 확률", flush=True)
    r = np.corrcoef(D.trend, D.csi)[0, 1]
    print(f"  Pearson r = {r:+.3f}  (강한 음수면 같은 것을 잼 / 0 근처면 직교)", flush=True)
    # 추세크기 분위별 평균 CSI·성과
    D["tq"] = pd.qcut(D.trend, 5, labels=[f"Q{i}" for i in range(1, 6)])
    print(f"\n  {'추세분위':<8} {'n':>5} {'평균CSI':>9} {'net':>9} {'건당':>8} {'승률':>6}", flush=True)
    for q, g in D.groupby("tq", observed=True):
        print(f"  {str(q):<8} {len(g):5d} {g.csi.mean():9.3f} {g.pnl.sum():+9.1f} "
              f"{g.pnl.mean():+7.3f} {100 * (g.pnl > 0).mean():5.0f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
