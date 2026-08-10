"""#SIZE-SPLIT 2026-08-10: SB 진입 크기를 줄이면 자산이 어떻게 되나 — 파트너 결정용.

## 배경
같은 정합 기준선에서 두 모델의 성적이 갈린다:
    SB    402건 · 건당 −0.067R [−0.192 ~ +0.054]   ← 0 을 포함(어느 쪽인지 모름)
    MMBM  695건 · 건당 +0.130R                      ← 홀드아웃 4관문 통과
SB 는 개선 축을 오늘 전부 확인했고(진입선별·소스제거·익절·트레일링·시간대·빈도)
메울 방법이 없다. 그렇다고 "확실히 손해"라는 증거도 없다 — 구간이 0 을 포함하고,
알트 홀드아웃에서는 +0.012R 로 경계를 넘는다.

그래서 끄는 대신 **크기만 줄이는** 안을 잰다. 되돌리는 비용이 배수 한 줄이고,
데이터 축적도 계속된다.

## 지금은 이걸 숫자로 답할 수 있다
어제까지는 못 했다. ① 진입 크기 계산이 백테에 없었고(백테가 라이브의 2배를 걸었다)
② MMBM 이 백테에 없었다. 둘 다 오늘 이식했다.

## 변형 (사전등록)
    1.00  현행 — 두 모델 같은 크기
    0.75  SB 를 3/4 로
    0.50  SB 를 절반으로          ★ 파트너 제안
    0.25  SB 를 1/4 로
    0.00  SB 중단 (MMBM 단독)     ← 상한 확인용

## 판정
자산·최대낙폭·파산확률을 같이 본다. 오늘 사이징을 이식했으므로 이 셋은
이제 참고치가 아니라 **판단 근거**다. 부트스트랩으로 순서 의존성도 확인한다.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "scripts")
from live_parity import run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT"]
CACHE = "data/axis/_size_split_rows.pkl"
LEV, POS_MAX = 7.0, 0.80
DD_PCT, DD_FACTOR = 0.25, 0.7
DAILY_LOSS_LIMIT, PAIR_DAILY_R = 0.15, 2.0
RUIN = 0.20
N_BOOT = 2000
SCALES = (1.00, 0.75, 0.50, 0.25, 0.00)


def collect() -> list[dict]:
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            rows = pickle.load(f)
        print(f"  (캐시 재사용 — {len(rows)}건)", flush=True)
        return rows
    rows = []
    for sym in PAIRS:
        print(f"  {sym} …", flush=True)
        df5, kept, _ = run_live_parity(sym)
        for t in kept:
            e, sl = float(t.entry), float(getattr(t, "entry_sl", 0.0) or 0.0)
            if e <= 0 or sl <= 0 or abs(e - sl) <= 0:
                continue
            rows.append({
                "sym": sym,
                "ent": int(df5.index[t.entry_idx].value // 10**6),
                "day": str(df5.index[t.entry_idx].date()),
                "raw": float(t.raw_pnl_pct),
                "r": float(t.raw_pnl_pct) * e / abs(e - sl),
                "sl_dist": abs(e - sl) / e,
                "ss": float(getattr(t, "smart_size_scale", 1.0)),
                "rp": float(getattr(t, "risk_pct_used", 6.0)),
                "fund": float(getattr(t, "funding_pct", 0.0)),
                "mmbm": "mmbm" in tuple(t.confluences),
            })
    rows.sort(key=lambda x: x["ent"])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump(rows, f)
    return rows


def simulate(rows: list[dict], sb_scale: float) -> dict:
    """라이브 자금관리 + SB 진입에만 배수 적용."""
    eq, peak, mdd = 1.0, 1.0, 0.0
    day_start: dict[str, float] = {}
    day_loss: dict[str, float] = {}
    pair_r: dict[tuple[str, str], float] = {}
    n_taken = 0
    for x in rows:
        if not x["mmbm"] and sb_scale <= 0.0:
            continue
        d = x["day"]
        if d not in day_start:
            day_start[d], day_loss[d] = eq, 0.0
        if day_loss[d] <= -DAILY_LOSS_LIMIT:
            continue
        k = (x["sym"], d)
        if pair_r.get(k, 0.0) <= -PAIR_DAILY_R:
            continue

        risk = eq * (x["rp"] / 100.0) * x["ss"]
        if eq < peak * (1.0 - DD_PCT):
            risk *= DD_FACTOR
        if not x["mmbm"]:
            risk *= sb_scale
        notional = min(risk / x["sl_dist"], eq * LEV * POS_MAX)
        pnl = x["raw"] * notional - (2.0 * TAKER_FEE_PCT + x["fund"]) * notional

        eq += pnl
        n_taken += 1
        day_loss[d] += pnl / max(day_start[d], 1e-12)
        pair_r[k] = pair_r.get(k, 0.0) + x["r"]
        if eq <= 0:
            return {"eq": 0.0, "mdd": 100.0, "ruin": True, "n": n_taken}
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return {"eq": eq, "mdd": 100.0 * mdd, "ruin": True, "n": n_taken}
    return {"eq": eq, "mdd": 100.0 * mdd, "ruin": False, "n": n_taken}


def boot(rows: list[dict], sb_scale: float):
    rng = np.random.default_rng(20260810)
    n = len(rows)
    fin, ruin = np.empty(N_BOOT), 0
    for i in range(N_BOOT):
        pick = sorted((rows[j] for j in rng.integers(0, n, size=n)),
                      key=lambda x: x["ent"])
        r = simulate(pick, sb_scale)
        fin[i] = r["eq"]
        ruin += int(r["ruin"])
    p50, p5 = np.percentile(fin, [50, 5])
    return float(p50), float(p5), 100.0 * ruin / N_BOOT


def main() -> int:
    print("=== SB 진입 크기 차등 — 자산 영향", flush=True)
    rows = collect()
    sb = [x for x in rows if not x["mmbm"]]
    mm = [x for x in rows if x["mmbm"]]
    months = (rows[-1]["ent"] - rows[0]["ent"]) / 86400000 / 30.4
    print(f"  거래 {len(rows)}건 ({months:.1f}개월) — SB {len(sb)} · MMBM {len(mm)}",
          flush=True)
    print(f"  건당 R — SB {np.mean([x['r'] for x in sb]):+.3f} · "
          f"MMBM {np.mean([x['r'] for x in mm]):+.3f}", flush=True)

    print(f"\n  {'SB 크기':<12}{'거래':>6}{'자산':>10}{'최대낙폭':>10}{'파산':>7}"
          f"   {'부트 중앙':>10}{'최악5%':>9}{'파산확률':>9}", flush=True)
    for s in SCALES:
        r = simulate(rows, s)
        p50, p5, pr = boot(rows, s)
        lab = "현행 1.00" if s == 1.0 else ("SB 중단" if s == 0.0 else f"{s:.2f}배")
        print(f"  {lab:<12}{r['n']:>6}{r['eq']:>9.2f}x{r['mdd']:>9.1f}%"
              f"{'예' if r['ruin'] else '아니오':>7}"
              f"   {p50:>9.2f}x{p5:>8.2f}x{pr:>8.1f}%", flush=True)

    print("\n  ※ 사이징·MMBM 을 오늘 이식했으므로 이 숫자들은 참고치가 아니라 판단 근거다.",
          flush=True)
    print("  ※ 남은 한계 — 증거금 가드(라이브 진입의 5.3% 차단)와 미완성 봉은 미이식.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
