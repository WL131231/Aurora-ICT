"""#AUTONOMOUS 2026-08-09: TP·트레일링 축 — 먼저 구조적 상한부터.

## 왜 이 축인가
SB 부검 결과 필요한 건 **RR 0.37** 또는 **승률 2.5%p** 다. 승률 쪽(진입 선별)은
confluence 연구로 여러 차례 실패했다. 남은 건 청산 쪽이고, 8/8 재판정에서
**MFE +1.691R vs 실현 +0.382R (4.4배)** 이라는 미탐색 항목이 있었다.

## 그런데 스윕부터 하면 안 된다 (2026-08-07 피보나치 교훈)
`ote_level` 때 관측 격차 0.142R 을 신호로 착각할 뻔했다. 그 파라미터가 물리적으로
움직일 수 있는 상한이 0.027R 이었기 때문이다. **상한이 노이즈보다 작으면 스윕이
무의미하다.** 그래서 이번에는 순서를 바꾼다:

  1단계 (이 스크립트) — 각 거래의 MFE/MAE 를 실측해 **이론 상한**을 낸다.
     · 완벽 익절(미래를 아는 경우) = 절대 상한, 도달 불가능하나 천장을 알려준다
     · 고정 R 익절이 현실적으로 얼마나 회수하는가
     · 상한이 필요한 0.37R 보다 작으면 이 축도 닫고 다른 데를 본다
  2단계 (상한이 충분할 때만) — 사전등록 변형 스윕 + 홀드아웃 4관문

## 주의 — 사후 최적화 방지
후보 R 값을 **미리 등록**하고(1.0~4.0 고정 격자), 최고값을 고르지 않는다.
본표본에서 형태만 보고, 판정은 홀드아웃에서 한다.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import run_live_parity  # noqa: E402

SYMS = ["BTCUSDT", "ETHUSDT"]
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "data", "axis", "_tp_mfe_rows.pkl")
TP_GRID = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)   # 사전등록 — 사후에 늘리지 않는다


def collect() -> list[dict]:
    """거래별 MFE/MAE 를 R 단위로 실측."""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            rows = pickle.load(f)
        print(f"  (캐시 재사용 — {len(rows)}건)", flush=True)
        return rows

    rows = []
    for sym in SYMS:
        print(f"  {sym} 계산 중 …", flush=True)
        df5, kept, _ = run_live_parity(sym)
        hi = df5["high"].to_numpy(float)
        lo = df5["low"].to_numpy(float)
        for t in kept:
            e = float(t.entry)
            sl = float(getattr(t, "entry_sl", 0.0) or 0.0)
            risk = abs(e - sl)
            if risk <= 0 or e <= 0:
                continue
            i, j = int(t.entry_idx), min(int(t.exit_idx), len(df5) - 1)
            if j <= i:
                continue
            up = float(hi[i:j + 1].max())
            dn = float(lo[i:j + 1].min())
            is_long = str(getattr(t.direction, "value", t.direction)).lower() == "long"
            # MFE = 보유 중 최대 유리 / MAE = 최대 불리 (둘 다 R 단위, 양수)
            mfe = (up - e) / risk if is_long else (e - dn) / risk
            mae = (e - dn) / risk if is_long else (up - e) / risk
            rows.append({
                "sym": sym, "dir": 1 if is_long else -1,
                "r": float(t.raw_pnl_pct) * e / risk,
                "mfe": mfe, "mae": mae, "outcome": t.outcome,
                "bars": j - i,
            })
        print(f"    {sym} {len([x for x in rows if x['sym'] == sym])}건", flush=True)

    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(rows, f)
    return rows


def main() -> int:
    print("=== TP 축 구조적 상한 — 스윕 전에 천장부터", flush=True)
    rows = collect()
    if not rows:
        print("  거래 0건", flush=True)
        return 0

    r = np.array([x["r"] for x in rows])
    mfe = np.array([x["mfe"] for x in rows])
    mae = np.array([x["mae"] for x in rows])
    n = len(r)

    print(f"\n  거래 {n}건 · 현재 건당 {r.mean():+.3f}R", flush=True)
    print(f"  MFE(최대 유리)  평균 {mfe.mean():+.3f}R · 중앙 {np.median(mfe):+.3f}R"
          f" · 75%분위 {np.percentile(mfe, 75):+.3f}R", flush=True)
    print(f"  MAE(최대 불리)  평균 {mae.mean():+.3f}R · 중앙 {np.median(mae):+.3f}R",
          flush=True)

    # ── 절대 상한 — 미래를 알고 MFE 를 그대로 챙겼다면 (도달 불가, 천장 표시용)
    print(f"\n  [절대 상한] 완벽 익절 = 건당 {mfe.mean():+.3f}R"
          f"  → 현재 대비 {mfe.mean() - r.mean():+.3f}R", flush=True)
    print("    ※ 미래를 아는 경우이므로 도달 불가능. 이 값보다 큰 개선은 원리상 없다.",
          flush=True)

    # ── 현실 상한 — 고정 R 익절 격자 (사전등록). 손절은 그대로 −1R.
    print(f"\n  [고정 R 익절 격자]  ※ MFE 가 목표에 닿았으면 그 R, 아니면 실제 결과", flush=True)
    print(f"    {'목표':<8}{'도달률':>8}{'건당R':>10}{'현재대비':>10}", flush=True)
    best = None
    for tp in TP_GRID:
        hit = mfe >= tp
        # 도달했으면 tp 확보, 아니면 실제 실현값(그대로)
        sim = np.where(hit, tp, r)
        d = sim.mean() - r.mean()
        print(f"    {tp:<8.1f}{100 * hit.mean():>7.0f}%{sim.mean():>+10.3f}{d:>+10.3f}",
              flush=True)
        if best is None or sim.mean() > best[1]:
            best = (tp, sim.mean())

    # ── 부분 익절 상한 — 절반을 목표에서, 나머지는 현행
    print(f"\n  [절반 익절 격자]  50% 를 목표에서 청산, 나머지 50% 는 현행 규칙", flush=True)
    print(f"    {'목표':<8}{'건당R':>10}{'현재대비':>10}", flush=True)
    for tp in TP_GRID:
        hit = mfe >= tp
        sim = np.where(hit, 0.5 * tp + 0.5 * r, r)
        print(f"    {tp:<8.1f}{sim.mean():>+10.3f}{sim.mean() - r.mean():>+10.3f}",
              flush=True)

    need = 0.37
    ceil_ = mfe.mean() - r.mean()
    print(f"\n  판정 — SB 손익분기까지 필요한 건 {need:.2f}R.", flush=True)
    print(f"    절대 상한 {ceil_:+.3f}R "
          f"{'≥ 필요치 → 이 축은 열려 있다. 2단계(사전등록 스윕+홀드아웃) 진행.' if ceil_ >= need else '< 필요치 → 이 축만으로는 못 넘는다. 닫는다.'}",
          flush=True)
    if best:
        print(f"    고정 격자 최고는 {best[0]:.1f}R 에서 {best[1]:+.3f}R 이나, "
              f"**사후 argmax 라 이 값 자체는 근거가 아니다** — 홀드아웃 필수.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
