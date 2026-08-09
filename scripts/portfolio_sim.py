"""#LIVE-SIZING 2026-08-09: 포트폴리오 복리 시뮬 — 라이브 자금관리 그대로.

## 왜 별도인가
`replay` 는 페어별로 독립 실행되는데 실제 계좌는 **페어가 자산을 공유**한다.
동시보유·일일캡·서킷·낙폭 스로틀은 계좌 단위라 페어별 백테에서는 표현이 안 된다.
그래서 replay 는 거래마다 "이 진입이 무엇을 입력으로 썼나"만 남기고(smart_size_scale·
risk_pct_used·entry_sl), 실제 자산 곡선은 여기서 계산한다.

## 무엇이 바뀌나
지금까지 복리·낙폭·파산은 **"참고치"** 였다. 백테가 `size_pct × leverage = 0.9×7
= 6.3배` 를 고정으로 걸었는데 라이브는 손절거리 역산이라 실측 평균 **3.16배** 였다.
정확히 2배를 과하게 걸고 있었으니 파산확률 같은 숫자가 의미가 없었다.

## 라이브 자금관리 (settings.py + bot_ict_instance.py)
    건당리스크% = min(3.0 + 1.5×점수, 6.0)        문턱 5 라 실질 6.0 상수
    리스크금액  = 자산 × 그% × 낙폭스로틀 × 품질배수
    수량        = 리스크금액 / 손절거리
    명목상한    = 자산 × 레버리지(7) × position_pct_max(80%)  = 5.6배
    · 낙폭 25% 초과 시 리스크 ×0.7 (구독 강제)
    · 계좌 일일 실현손실 15% 도달 시 당일 신규진입 중단 (구독 강제)
    · 페어당 하루 −2R 도달 시 그 페어 당일 중단

## 사용
    PYTHONPATH=src python scripts/portfolio_sim.py                 # 기본(BTC+ETH)
    PYTHONPATH=src python scripts/portfolio_sim.py --pairs BTCUSDT,ETHUSDT,SOLUSDT

안전장치를 하나씩 끈 변형을 같이 출력해 **각 장치의 기여**를 분리해서 본다.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "data", "axis", "_portfolio_rows.pkl")
LEV = 7.0
POS_MAX = 0.80          # position_pct_max
DD_PCT, DD_FACTOR = 0.25, 0.7
DAILY_LOSS_LIMIT = 0.15
PAIR_DAILY_R = 2.0
RUIN = 0.20             # 시드의 20% 밑이면 파산 처리
N_BOOT = 2000


def collect(pairs: list[str]) -> list[dict]:
    key = CACHE.replace(".pkl", f"_{'_'.join(pairs)}.pkl")
    if os.path.exists(key):
        with open(key, "rb") as f:
            rows = pickle.load(f)
        print(f"  (캐시 재사용 — {len(rows)}건)", flush=True)
        return rows
    rows = []
    for sym in pairs:
        print(f"  {sym} …", flush=True)
        df5, kept, _ = run_live_parity(sym)
        for t in kept:
            e, sl = float(t.entry), float(getattr(t, "entry_sl", 0.0) or 0.0)
            if e <= 0 or sl <= 0 or abs(e - sl) <= 0:
                continue
            rows.append({
                "sym": sym,
                "ent": int(df5.index[t.entry_idx].value // 10**6),
                "ex": int(df5.index[min(t.exit_idx, len(df5) - 1)].value // 10**6),
                "day": str(df5.index[t.entry_idx].date()),
                "raw": float(t.raw_pnl_pct),
                "r": float(t.raw_pnl_pct) * e / abs(e - sl),
                "sl_dist": abs(e - sl) / e,
                "ss": float(getattr(t, "smart_size_scale", 1.0)),
                "rp": float(getattr(t, "risk_pct_used", 6.0)),
                "fund": float(getattr(t, "funding_pct", 0.0)),
            })
    rows.sort(key=lambda x: x["ent"])
    os.makedirs(os.path.dirname(key), exist_ok=True)
    with open(key, "wb") as f:
        pickle.dump(rows, f)
    return rows


def simulate(
    rows: list[dict],
    *,
    live_sizing: bool = True,
    dd_throttle: bool = True,
    daily_cap: bool = True,
    pair_cap: bool = True,
    funding: bool = True,
    fixed_size: float = 0.9,
) -> dict:
    """자산 곡선 1회. 반환: 최종배수·최대낙폭%·파산여부·차단건수."""
    eq, peak, mdd = 1.0, 1.0, 0.0
    day_start_eq: dict[str, float] = {}
    day_loss: dict[str, float] = {}      # 계좌 당일 실현손익(비율)
    pair_r: dict[tuple[str, str], float] = {}
    blocked_daily = blocked_pair = 0

    for x in rows:
        d = x["day"]
        if d not in day_start_eq:
            day_start_eq[d] = eq
            day_loss[d] = 0.0

        if daily_cap and day_loss[d] <= -DAILY_LOSS_LIMIT:
            blocked_daily += 1
            continue
        k = (x["sym"], d)
        if pair_cap and pair_r.get(k, 0.0) <= -PAIR_DAILY_R:
            blocked_pair += 1
            continue

        if live_sizing:
            risk = eq * (x["rp"] / 100.0) * x["ss"]
            if dd_throttle and eq < peak * (1.0 - DD_PCT):
                risk *= DD_FACTOR
            notional = min(risk / x["sl_dist"], eq * LEV * POS_MAX)
        else:
            notional = eq * fixed_size * LEV
            if dd_throttle and eq < peak * (1.0 - DD_PCT):
                notional *= DD_FACTOR

        cost = 2.0 * TAKER_FEE_PCT * notional
        if funding:
            cost += x["fund"] * notional
        pnl = x["raw"] * notional - cost

        eq += pnl
        day_loss[d] += pnl / max(day_start_eq[d], 1e-12)
        pair_r[k] = pair_r.get(k, 0.0) + x["r"]

        if eq <= 0:
            return {"eq": 0.0, "mdd": 100.0, "ruin": True,
                    "bd": blocked_daily, "bp": blocked_pair}
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return {"eq": eq, "mdd": 100.0 * mdd, "ruin": True,
                    "bd": blocked_daily, "bp": blocked_pair}

    return {"eq": eq, "mdd": 100.0 * mdd, "ruin": False,
            "bd": blocked_daily, "bp": blocked_pair}


def boot(rows: list[dict], **kw) -> tuple[float, float, float]:
    """부트스트랩 — 거래 순서·구성을 흔들었을 때의 분포."""
    rng = np.random.default_rng(20260809)
    n = len(rows)
    fin, ruin = np.empty(N_BOOT), 0
    for i in range(N_BOOT):
        pick = [rows[j] for j in rng.integers(0, n, size=n)]
        pick.sort(key=lambda x: x["ent"])
        r = simulate(pick, **kw)
        fin[i] = r["eq"]
        ruin += int(r["ruin"])
    p50, p5 = np.percentile(fin, [50, 5])
    return float(p50), float(p5), 100.0 * ruin / N_BOOT


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="BTCUSDT,ETHUSDT")
    a = ap.parse_args()
    pairs = [s.strip() for s in a.pairs.split(",") if s.strip()]

    print(f"=== 포트폴리오 복리 시뮬 — {', '.join(pairs)}", flush=True)
    rows = collect(pairs)
    if not rows:
        print("  거래 0건", flush=True)
        return 0
    months = (rows[-1]["ex"] - rows[0]["ent"]) / 86400000 / 30.4
    print(f"  거래 {len(rows)}건 · {months:.1f}개월", flush=True)

    variants = [
        ("라이브 그대로", dict()),
        ("  − 일일 계좌캡", dict(daily_cap=False)),
        ("  − 페어 일일 -2R", dict(pair_cap=False)),
        ("  − 낙폭 스로틀", dict(dd_throttle=False)),
        ("  − 펀딩", dict(funding=False)),
        ("  안전장치 전부 끔", dict(daily_cap=False, pair_cap=False,
                                dd_throttle=False)),
        ("기존 백테(고정 0.9×7)", dict(live_sizing=False)),
    ]

    print(f"\n  {'구성':<24}{'자산':>10}{'최대낙폭':>10}{'파산':>7}"
          f"{'차단(계좌/페어)':>16}", flush=True)
    for name, kw in variants:
        r = simulate(rows, **kw)
        print(f"  {name:<24}{r['eq']:>9.2f}x{r['mdd']:>9.1f}%"
              f"{'예' if r['ruin'] else '아니오':>7}"
              f"{r['bd']:>9}/{r['bp']:<6}", flush=True)

    print(f"\n  부트스트랩 {N_BOOT}회 (거래 구성·순서를 흔들었을 때)", flush=True)
    print(f"  {'구성':<24}{'중앙':>10}{'최악5%':>10}{'파산확률':>10}", flush=True)
    for name, kw in (("라이브 그대로", dict()),
                     ("기존 백테(고정 0.9×7)", dict(live_sizing=False))):
        p50, p5, pr = boot(rows, **kw)
        print(f"  {name:<24}{p50:>9.2f}x{p5:>9.2f}x{pr:>9.1f}%", flush=True)

    print("\n  ※ '라이브 그대로' 가 이제 판단 근거다. 나머지는 기여도 분해용.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
