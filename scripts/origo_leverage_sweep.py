"""#AUTONOMOUS 2026-08-06: Origo 레버리지 3~20x — 복리 기준 생존·최적 배율.

파트너 요청: "배율 낮추면 어떨지. x3 ~ x20 까지".

## 왜 단리 백테로는 답이 안 나오나

Origo 는 SL 이 구조 기반(ATR×4), TP 가 2R 이라 **R 배수가 레버리지와 무관**하다.
단리로 합산하면 net 도 MDD 도 배율에 정비례해서 net/MDD 가 불변 — 20x 가 3x 보다
항상 좋아 보이고 비교가 무의미하다.

실제 계좌는 **복리**로 움직인다. -50% 를 맞으면 +100% 를 벌어야 본전이라, 배율이
올라갈수록 한 번의 손실이 이후 모든 거래의 원금을 깎는다. 그래서 어느 지점부터는
배율을 더 올려도 기대 자산이 오히려 줄어든다(켈리 기준의 우변). 이 스크립트는
**실제 자산 곡선을 배율별로 다시 굴려** 그 지점을 찾는다.

## 측정
- 최종 자산 배수(중앙값) · 최대낙폭 · **파산 확률**(자산 20% 이하로 떨어질 확률)
- 부트스트랩 2000회 — 거래 **순서를 섞어** 특정 배열에 의존하지 않는 결과를 본다.
  (실제 순서 한 번만 보면 "운 좋은 배열"을 최적으로 착각한다.)

## 한계 (결론에 명시)
- 여러 페어를 동시에 들고 있어도 순차 체결로 근사한다(자본 분할 미반영).
- 라이브의 일일 손실 한도 15% · DD 스로틀 25% 는 하니스 미구현(live_parity.GAPS).
  둘 다 **고배율에서 더 자주 발동**하므로, 실제 고배율 성적은 여기보다 완만하다.
- 청산은 sl_liq_cap(청산가 캡)까지만 반영 — 갭으로 청산가를 건너뛰는 경우 미반영.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import LIVE_BASE, PAIRS, run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

RNG = np.random.default_rng(20260806)
N_BOOT = 2000
SIZE = LIVE_BASE["size_pct"]
RUIN = 0.20            # 시드의 20% 이하 = 사실상 복구 불가 → 파산으로 본다


def collect_raw() -> list[tuple[int, float]]:
    """라이브 정합 거래를 시간순으로 — (entry_ms, raw_pnl_pct)."""
    rows: list[tuple[int, float]] = []
    for sym in PAIRS:
        df5, kept, _ = run_live_parity(sym)
        for t in kept:
            ts = df5.index[t.entry_idx]
            rows.append((int(ts.value), float(t.raw_pnl_pct)))
    rows.sort(key=lambda x: x[0])
    return rows


def simulate(raws: np.ndarray, lev: float, days: np.ndarray | None = None,
             daily_stop: float = 0.0) -> tuple[float, float, bool]:
    """복리 자산 곡선 — (최종 배수, MDD%, 파산 여부).

    거래당 수익률 = raw × size × lev − 왕복 수수료(notional 기준).
    자산이 RUIN 이하로 내려가면 그 시점에 멈춘다(실제로는 더 못 굴린다).
    """
    fee = 2.0 * TAKER_FEE_PCT * SIZE * lev
    eq = 1.0
    peak = 1.0
    mdd = 0.0
    cur_day = -1
    day_start_eq = 1.0
    for i, r in enumerate(raws):
        # 라이브 #SAFETY-1 근사 — 하루 누적 손실이 한도를 넘으면 그날 신규 진입 중단.
        # 고배율일수록 자주 발동하므로 이걸 빼면 고배율 위험이 과대평가된다.
        if daily_stop > 0 and days is not None:
            d = int(days[i])
            if d != cur_day:
                cur_day, day_start_eq = d, eq
            elif eq <= day_start_eq * (1.0 - daily_stop):
                continue
        step = r * SIZE * lev - fee
        eq *= (1.0 + step)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return eq, 100.0 * mdd, True
    return eq, 100.0 * mdd, False


def main() -> int:
    rows = collect_raw()
    raws = np.array([r for _, r in rows], float)
    days = np.array([ts // 86_400_000_000_000 for ts, _ in rows], dtype=np.int64)
    n = len(raws)
    print("=== Origo 레버리지 스윕 (복리 · 라이브 정합 거래) ===", flush=True)
    print(f"  거래 {n}건 · size {SIZE:.0%} · 수수료 왕복 {2 * TAKER_FEE_PCT:.3%}"
          f"(notional 기준) · 파산 기준 시드 {RUIN:.0%}", flush=True)
    print(f"  raw 평균 {raws.mean():+.4%} · 최악 {raws.min():+.2%} · "
          f"최고 {raws.max():+.2%}", flush=True)

    for stop, label in ((0.0, "일일스탑 없음 (하니스 그대로)"),
                        (0.15, "일일 -15% 스탑 적용 (라이브 정합)")):
        print(f"\n### {label}", flush=True)
        print(f"  {'배율':<5}{'실효노출':>9}{'자산':>10}{'MDD':>8}"
              f"{'부트중앙':>10}{'5%분위':>9}{'파산확률':>10}", flush=True)
        best = None
        for lev in (3, 5, 7, 10, 12, 15, 17, 20):
            eq0, mdd0, _ = simulate(raws, lev, days, stop)
            fin = np.empty(N_BOOT)
            ruins = 0
            for i in range(N_BOOT):
                idx = RNG.permutation(len(raws))   # 순서만 섞는다(구성 동일)
                e, _, r_ = simulate(raws[idx], lev, days[idx], stop)
                fin[i] = e
                ruins += int(r_)
            p50, p5 = np.percentile(fin, [50, 5])
            pr = 100.0 * ruins / N_BOOT
            if best is None or p50 > best[1]:
                best = (lev, p50)
            print(f"  {lev:<5.0f}{lev * SIZE:>8.1f}x{eq0:>9.2f}x{mdd0:>7.1f}%"
                  f"{p50:>9.2f}x{p5:>8.2f}x{pr:>9.1f}%", flush=True)
        print(f"  → 중앙값 최대: {best[0]:.0f}x ({best[1]:.2f}배)", flush=True)

    print("\n  ※ DD 스로틀(-25% 시 리스크 축소)은 여전히 미반영 — 실제는 더 완만하다.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
