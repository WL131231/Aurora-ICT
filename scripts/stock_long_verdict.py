"""#AUTONOMOUS 2026-08-06: 주식 롱 전용 판정 — 전략이 '그냥 사서 들고 있기' 를 이기나.

파트너 지시: 숏 제외, 롱만.

## 왜 이 비교가 필수인가

1차 결과에서 나스닥 롱 +205.6%, 코스닥 롱 +200.2% 가 나왔다. 그런데 **10년간 대형주를
그냥 보유만 해도 크게 올랐다.** 롱 전략이 흑자인 것은 전략의 공로가 아니라 시장의
우상향일 수 있다. 두 대조군을 세운다:

① **Buy & Hold** — 첫 봉에 사서 마지막 봉까지 보유. 전략이 이걸 못 이기면 의미 없다.
   단 전략은 시장에 **일부 시간만** 노출된다(현금 대기). 그래서 총수익뿐 아니라
   **노출 시간당 수익**도 같이 본다 — 노출이 절반인데 수익이 같다면 시간당으론 이긴다.

② **플라시보(무작위 롱)** — 전략과 **같은 거래 수·같은 보유 기간 분포**로, 진입
   시점만 무작위로 잡는다. 이게 통과 못 하면 "신호가 아니라 그냥 롱이라서 벌었다".
   7/29 임펄스 연구에서 플라시보가 흑자를 내며 기각시킨 선례가 있다.

판정: 전략 롱이 **플라시보를 유의하게 이기고**(순열 p<0.05) **B&H 대비 시간당 수익
우위**가 있어야 "주식에서도 통한다"고 말할 수 있다. 하나라도 못 넘기면 기각.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from stock_cursus_bt import COST, simulate  # noqa: E402
from stock_fetch import MARKETS, fetch  # noqa: E402

RNG = np.random.default_rng(20260806)
N_PLACEBO = 400


def buy_hold(df, cost: float) -> tuple[float, int]:
    """첫 봉 매수 → 마지막 봉 매도. 반환 (수익률%, 보유 봉수)."""
    c = df["close"].to_numpy(float)
    return (c[-1] - c[0]) / c[0] * 100.0 - cost * 100.0, len(c)


def placebo(df, cost: float, n_tr: int, holds: np.ndarray) -> np.ndarray:
    """무작위 시점 롱 진입 — 전략과 같은 거래 수·보유 기간 분포.

    Returns:
        N_PLACEBO 회 시행의 총수익률(%) 배열.
    """
    c = df["close"].to_numpy(float)
    n = len(c)
    if n_tr <= 0 or n < 50 or len(holds) == 0:
        return np.zeros(0)
    out = np.empty(N_PLACEBO)
    for k in range(N_PLACEBO):
        hs = RNG.choice(holds, size=n_tr, replace=True)
        starts = RNG.integers(0, n - 2, size=n_tr)
        ends = np.minimum(starts + hs, n - 1)
        out[k] = float(np.sum((c[ends] - c[starts]) / c[starts] - cost)) * 100.0
    return out


def main() -> int:
    print("=== 주식 롱 전용 판정 — 전략 vs 그냥 보유 vs 무작위 롱 ===", flush=True)
    print("  Cursus(DualST) · 레버리지 1x · 갭 시가체결 · 비용 시장별", flush=True)

    for interval, label in (("1d", "일봉 10년"), ("1h", "1시간봉 ~3년")):
        print(f"\n### {label}", flush=True)
        print(f"  {'시장':<10}{'전략net':>10}{'노출%':>7}{'B&H':>10}"
              f"{'시간당 전략':>12}{'시간당 B&H':>11}{'플라시보중앙':>13}{'p값':>7}",
              flush=True)
        for mkt, tickers in MARKETS.items():
            s_net = bh_net = 0.0
            s_bars = bh_bars = 0
            all_holds: list[int] = []
            pl_tot = np.zeros(N_PLACEBO)
            ok = 0
            for t in tickers:
                try:
                    df = fetch(t, interval=interval,
                               period="730d" if interval == "1h" else "10y")
                except Exception:  # noqa: BLE001
                    continue
                if df is None or len(df) < 200:
                    continue
                ok += 1
                tr = simulate(df, COST[mkt], side_only="long")
                holds = np.array([x[3] for x in tr if x[3] > 0], dtype=int)
                s_net += 100.0 * sum(x[0] for x in tr)
                s_bars += int(holds.sum())
                b, nb = buy_hold(df, COST[mkt])
                bh_net += b
                bh_bars += nb
                all_holds.extend(holds.tolist())
                if len(tr) and len(holds):
                    p = placebo(df, COST[mkt], len(tr), holds)
                    if len(p):
                        pl_tot += p
            if ok == 0:
                continue
            expo = 100.0 * s_bars / max(bh_bars, 1)
            per_s = s_net / max(s_bars, 1)          # 봉당 수익률
            per_b = bh_net / max(bh_bars, 1)
            pl_med = float(np.median(pl_tot)) if len(pl_tot) else float("nan")
            pval = float((pl_tot >= s_net).mean()) if len(pl_tot) else float("nan")
            print(f"  {mkt:<10}{s_net:>+9.1f}%{expo:>6.1f}%{bh_net:>+9.1f}%"
                  f"{per_s:>+11.4f}%{per_b:>+10.4f}%{pl_med:>+12.1f}%{pval:>7.3f}",
                  flush=True)

        print("  판정 — p<0.05 (무작위 롱을 유의하게 이김) + 시간당 전략 > 시간당 B&H",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
