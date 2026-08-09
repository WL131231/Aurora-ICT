"""#AXIS 2026-08-07: Origo min_rr(최소 손익비) 게이트 축 연구.

## 무엇을 재는가

min_rr 은 셋업 검출 단계의 **통과/탈락 문턱**이다(silver_bullet.py 의
``if rr < min_rr: continue``). 여기서 rr = |다음 유동성 목표 - 진입가| /
|진입가 - SL|. 중요한 구조적 사실 두 가지:

1. **min_rr 은 진입가·SL·TP 를 하나도 바꾸지 않는다.** TP 는 "다음 유동성"
   이지 min_rr 배수가 아니다. 따라서 두 설정에서 **살아남는 같은 셋업의 R
   결과는 완전히 동일**하다. 바뀌는 것은 오직 **거래 집합(체결셋)**이다.
   → 직전 피보 연구(ote_level)와 정반대다. ote_level 은 진입가를 0.052% 만
     움직여 R 영향 상한이 건당 0.027R 이었지만, min_rr 은 거래를 통째로
     넣고 뺀다. 지렛대의 크기는 "몇 건이 들고 나는가 × 그 건들의 R" 이다.

2. **집합이 포함관계(nested)가 아니다.** detect 는 (날짜, 킬존윈도우) 조합마다
   **첫 번째 유효 FVG 한 건**만 셋업으로 채택한다(``seen_windows``).
   min_rr 을 낮추면 이전에 탈락하던 **이른 시각의 저RR FVG 가 슬롯을 선점**해,
   원래 채택되던 **뒤쪽 고RR FVG 를 밀어낼 수** 있다. 그래서 min_rr 을 낮췄는데
   기존 거래가 사라지는 일이 실제로 생긴다. 이 치환(substitution)을 별도로 센다.

## 왜 재는가

직전 피보 연구에서 43↔52건 차이의 출처로 min_rr 이 지목됐다. 이 축은 체결셋을
직접 가르므로 "구조적 지렛대"는 확실히 있다. 문제는 **추가되는 거래가 흑자인가**
뿐이다. 2026-08-06 confluence 실험(5→4)에서 추가분 승률 25% 로 기각된 선례가 있어
**증분 분석**을 판정의 중심에 둔다.

## 실행 모드

    python scripts/axis_min-rr_2026-08-07.py ceiling      # 1단계 구조적 상한
    python scripts/axis_min-rr_2026-08-07.py run --min-rr 1.8   # 레벨 1개 백테
    python scripts/axis_min-rr_2026-08-07.py analyze      # 종합 분석 + JSON 저장

## 방법상 규칙 (팀 판정 기준)

  · 비교는 전부 **R 배수**. min_rr 에 따라 SL 폭 분포가 달라지므로 %수익 비교는 왜곡.
  · 평가는 **복리** — 레버리지 7x · size_pct 0.9 · 동시보유 분할 · DD 스로틀(-25%→×0.7)
    · 파산(시드 20% 이하) 판정 · 부트스트랩 복원추출.
  · 순열검정 20000회. 국면×방향 기저 통제(롱/숏 분해). 연도 일관성.
  · 표본 30건 미만 셀은 "표본부족"으로 명시하고 결론에서 제외.
  · 시도한 조합 수를 명시(다중비교).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import LIVE_BASE, run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402

# ── 연구 상수 ────────────────────────────────────────────────────────────
PAIRS = ["BTCUSDT", "ETHUSDT"]          # 현행 라이브 고정 페어
LEVELS = [1.5, 1.8, 2.0, 2.5, 3.0]      # 스윕 대상 (지시받은 5개)
BASELINE = 2.0                          # 현행 라이브
MIN_N = 30                              # 이 미만 = 표본부족
N_PERM = 20_000                         # 순열검정 반복
N_BOOT = 2000                           # 부트스트랩 반복

LEV = 7.0                               # 라이브 레버리지
SIZE = LIVE_BASE["size_pct"]            # 0.9
RUIN = 0.20                             # 시드 20% 이하 = 파산
DD_PCT, DD_FACTOR = 0.25, 0.7           # DD 스로틀

RNG = np.random.default_rng(20260807)
OUT_DIR = Path("data/axis")
RAW_DIR = OUT_DIR / "_minrr_raw"


# ══════════════════════════════════════════════════════════════════════
# 1단계 — 구조적 상한 (스윕 전에 반드시 먼저)
# ══════════════════════════════════════════════════════════════════════
def ceiling() -> None:
    """min_rr 이 물리적으로 몇 건을 넣고 뺄 수 있는지 세는 단계.

    min_rr=1.5 로 빌드된 셋업 타임라인(=1.5 이상 전 셋업의 상한 집합)을 읽어
    RR 구간별 셋업 수를 센다. 스윕 범위(1.5~3.0)에서 움직일 수 있는 거래의
    **물리적 최대치**가 여기서 나온다.

    3개 층으로 센다:
      A층 전체 셋업        — detect 가 만든 모든 셋업
      B층 도달가능 셋업     — stale 게이트(bars_since <= setup_stale_bars) 통과분
      C층 실제 체결        — analyze 단계의 백테 결과 (여기서는 미측정)
    """
    from bt_par import _load_full, _resample, cached_setup_timeline
    from aurora_ict.backtest.replay import BacktestConfig

    bands = [(1.5, 1.8), (1.8, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 1e9)]
    stale = LIVE_BASE["setup_stale_bars"]

    print("=" * 100)
    print("[1단계] min_rr 축의 구조적 상한 — 스윕 전 물리 한계 산출")
    print("=" * 100)
    print("\nRR 정의: |다음 유동성 TP - 진입가| / |진입가 - SL|")
    print("min_rr 은 이 값의 문턱일 뿐 진입가/SL/TP 를 바꾸지 않는다")
    print("→ 살아남는 셋업의 R 결과는 설정과 무관하게 동일. 바뀌는 건 집합뿐.\n")

    total_a = Counter()
    total_b = Counter()
    per_sym = {}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        kw = dict(LIVE_BASE)
        kw["min_rr"] = 1.5          # 스윕 범위 하한 = 상한 집합
        cfg = BacktestConfig(**kw)
        tl = cached_setup_timeline(df5, cfg, sym)

        seen_a: dict[tuple, float] = {}   # 셋업 고유키 → rr
        seen_b: dict[tuple, float] = {}
        for i, cell in enumerate(tl):
            if cell is None:
                continue
            st, bars_since = cell
            key = (int(st.ts_ms), str(getattr(st.direction, "value", st.direction)),
                   round(float(st.entry), 6))
            rr = float(st.risk_reward)
            seen_a.setdefault(key, rr)
            if bars_since <= stale:
                seen_b.setdefault(key, rr)

        ca, cb = Counter(), Counter()
        for d, c in ((seen_a, ca), (seen_b, cb)):
            for rr in d.values():
                for lo, hi in bands:
                    if lo <= rr < hi:
                        c[(lo, hi)] += 1
                        break
        per_sym[sym] = (len(seen_a), len(seen_b), ca, cb)
        total_a.update(ca)
        total_b.update(cb)
        print(f"  {sym}: 전체 셋업 {len(seen_a)}건 / stale({stale}봉) 통과 {len(seen_b)}건")

    print(f"\n[표0] RR 구간별 셋업 수 (BTC+ETH 합, 2021-07~2026-06)")
    print(f"{'RR 구간':>14} | {'A층 전체':>9} | {'B층 도달가능':>12} | 설명")
    print("-" * 100)
    labels = {
        (1.5, 1.8): "min_rr 2.0→1.5 로 낮출 때 새로 열리는 후보",
        (1.8, 2.0): "min_rr 2.0→1.8 로 낮출 때 새로 열리는 후보",
        (2.0, 2.5): "min_rr 2.0→2.5 로 올릴 때 잘려나가는 후보",
        (2.5, 3.0): "min_rr 2.5→3.0 로 올릴 때 잘려나가는 후보",
        (3.0, 1e9): "어떤 설정에서도 통과 (기저 집합)",
    }
    for b in bands:
        hi = "inf" if b[1] > 1e8 else f"{b[1]:.1f}"
        print(f"{b[0]:>6.1f}~{hi:>6} | {total_a[b]:>9} | {total_b[b]:>12} | {labels[b]}")
    print("-" * 100)
    print(f"{'합계':>14} | {sum(total_a.values()):>9} | {sum(total_b.values()):>12} |")

    base_pool = total_b[(2.0, 2.5)] + total_b[(2.5, 3.0)] + total_b[(3.0, 1e9)]
    add15 = total_b[(1.5, 1.8)] + total_b[(1.8, 2.0)]
    cut25 = total_b[(2.0, 2.5)]
    print("\n[판정 0] 구조적 지렛대 크기 (B층 기준, 게이트 이전 상한)")
    print(f"  현행 2.0 통과 후보 : {base_pool}건")
    print(f"  2.0→1.5 최대 추가  : +{add15}건 ({100*add15/max(base_pool,1):.0f}%)")
    print(f"  2.0→2.5 최대 삭감  : -{cut25}건 ({100*cut25/max(base_pool,1):.0f}%)")
    print("\n  ※ 이 수치는 confluence>=5 · HTF align · regime · cond_align · nypm ·")
    print("    ttl 미체결 게이트 **이전**이라 실제 체결은 이보다 훨씬 적다.")
    print("    다만 '이 손잡이가 R 을 움직일 수 있는가'의 답은 여기서 결정된다:")
    print("    체결셋 자체가 바뀌므로 **R 영향 상한은 건당 R 분포 폭 그 자체**이며,")
    print("    ote_level 의 0.027R 같은 기계적 천장이 없다. → 스윕할 가치 있음.")


# ══════════════════════════════════════════════════════════════════════
# 2단계 — 레벨 1개 백테 (타임라인 캐시 키에 min_rr 이 들어가 레벨마다 프로세스 분리)
# ══════════════════════════════════════════════════════════════════════
def run_level(min_rr: float) -> dict:
    """min_rr 값 하나로 라이브 정합 백테를 돌려 거래를 JSON 으로 저장."""
    rows, gates = [], {}
    for sym in PAIRS:
        df5, kept, gate = run_live_parity(sym, {"min_rr": min_rr})
        gates[sym] = {k: int(v) for k, v in gate.items()}
        idx = df5.index
        for t in kept:
            sl = float(getattr(t, "entry_sl", 0.0) or 0.0)
            risk = abs(float(t.entry) - sl)
            rows.append({
                "sym": sym,
                "ts": int(idx[t.entry_idx].value // 10**6),
                "ex": int(idx[min(t.exit_idx, len(idx) - 1)].value // 10**6),
                "raw": float(t.raw_pnl_pct),
                # R 배수 = 원시수익률 × 진입가 / 위험폭
                "r": (float(t.raw_pnl_pct) * float(t.entry) / risk) if risk > 0 else 0.0,
                "dir": str(getattr(t.direction, "value", t.direction)).lower(),
                "outcome": str(getattr(t, "outcome", "")),
                "entry": float(t.entry),
                "tp": float(getattr(t, "entry_tp", 0.0) or 0.0),
                "sl": sl,
                # 셋업 원본 RR (진입가 기준 TP/SL 비) — 증분 거래의 정체 확인용
                "rr": (abs(float(getattr(t, "entry_tp", 0.0) or 0.0) - float(t.entry))
                       / risk) if risk > 0 else 0.0,
                "trend": float(getattr(t, "entry_trend_pct", 0.0) or 0.0),
            })
    rows.sort(key=lambda r: r["ts"])
    return {"min_rr": min_rr, "n": len(rows), "gate": gates, "trades": rows}


# ══════════════════════════════════════════════════════════════════════
# 지표 / 복리 시뮬
# ══════════════════════════════════════════════════════════════════════
def year_of(ms: int) -> int:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).year


def rstats(trades: list[dict]) -> dict | None:
    """R 기준 기본 지표."""
    if not trades:
        return None
    r = np.array([t["r"] for t in trades], float)
    se = r.std(ddof=1) / np.sqrt(len(r)) if len(r) > 1 else float("nan")
    return {
        "n": len(r), "r_mean": float(r.mean()), "r_sum": float(r.sum()),
        "win": float((r > 0).mean() * 100),
        "se": float(se), "t": float(r.mean() / se) if se and se > 0 else float("nan"),
    }


def concurrency(trades: list[dict]) -> np.ndarray:
    """각 거래가 열려 있는 동안의 평균 동시 보유 수(자기 포함) — 라이브 자본 분할 재현."""
    out = np.empty(len(trades))
    for i, a in enumerate(trades):
        ov = sum(1 for j, b in enumerate(trades)
                 if j != i and b["ts"] <= a["ex"] and b["ex"] >= a["ts"])
        out[i] = 1.0 + ov
    return out


def sim(raws: np.ndarray, scale: np.ndarray) -> tuple[float, float, bool]:
    """복리 시뮬 — (최종자산배수, MDD%, 파산여부). 7x · size 0.9 · DD 스로틀 포함."""
    eq = peak = 1.0
    mdd = 0.0
    for i, r in enumerate(raws):
        sz = SIZE * scale[i]
        if eq < peak * (1.0 - DD_PCT):      # DD 스로틀
            sz *= DD_FACTOR
        step = r * sz * LEV - 2.0 * TAKER_FEE_PCT * sz * LEV
        eq *= (1.0 + step)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return float(eq), 100.0 * mdd, True
    return float(eq), 100.0 * mdd, False


def compound_eval(trades: list[dict]) -> dict:
    """복리 평가 — 실적 자산배수 · MDD · 파산확률 · 부트스트랩 중앙값/5%분위."""
    if len(trades) < 2:
        return {"compound": float("nan"), "mdd": float("nan"), "ruin": float("nan"),
                "boot_p50": float("nan"), "boot_p05": float("nan")}
    tr = sorted(trades, key=lambda t: t["ts"])
    raws = np.array([t["raw"] for t in tr], float)
    scale = 1.0 / concurrency(tr)
    e0, m0, _ = sim(raws, scale)
    n = len(raws)
    fin = np.empty(N_BOOT)
    ruin = 0
    for k in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)     # 복원추출 (미래 가정, 보수적)
        e, _, r_ = sim(raws[idx], scale[idx])
        fin[k] = e
        ruin += int(r_)
    p50, p05 = np.percentile(fin, [50, 5])
    return {"compound": e0, "mdd": m0, "ruin": 100.0 * ruin / N_BOOT,
            "boot_p50": float(p50), "boot_p05": float(p05),
            "conc": float(concurrency(tr).mean())}


def perm_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    """두 집단 평균차 순열검정 (라벨 무작위 재배정, 양측)."""
    obs = a.mean() - b.mean()
    pool = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(N_PERM):
        p = RNG.permutation(pool)
        if abs(p[:na].mean() - p[na:].mean()) >= abs(obs) - 1e-12:
            cnt += 1
    return (cnt + 1) / (N_PERM + 1)


def perm_mean_positive(x: np.ndarray) -> float:
    """평균이 0보다 큰가 — 부호섞기 순열검정 (양측)."""
    obs = x.mean()
    signs = RNG.choice([-1.0, 1.0], size=(N_PERM, len(x)))
    null = (signs * x).mean(axis=1)
    return float(((np.abs(null) >= abs(obs) - 1e-12).sum() + 1) / (N_PERM + 1))


# ══════════════════════════════════════════════════════════════════════
# 3단계 — 종합 분석
# ══════════════════════════════════════════════════════════════════════
def key_of(t: dict) -> tuple:
    return (t["sym"], t["ts"])


def analyze() -> None:
    data = {}
    for lv in LEVELS:
        p = RAW_DIR / f"minrr_{lv:.1f}.json"
        if not p.exists():
            print(f"  !! {p} 없음 — run 모드로 먼저 돌릴 것", flush=True)
            continue
        data[lv] = json.load(open(p, encoding="utf-8"))
    if BASELINE not in data:
        raise SystemExit("기준선(2.0) 결과가 없다")

    base = data[BASELINE]["trades"]
    base_keys = {key_of(t) for t in base}
    all_ts = [t["ts"] for d in data.values() for t in d["trades"]]

    print("=" * 100)
    print("[2단계] min_rr 스윕 — Origo / BTC+ETH / 라이브 정합 (7x · size 0.9 · 복리)")
    print(f"구간 {datetime.fromtimestamp(min(all_ts)/1000, tz=timezone.utc).date()} ~ "
          f"{datetime.fromtimestamp(max(all_ts)/1000, tz=timezone.utc).date()}"
          f"  ·  시도 조합 {len(LEVELS)}개 (min_rr 1.5/1.8/2.0/2.5/3.0)")
    print("=" * 100)

    # ---- 표1: 레벨별 기본 지표
    print("\n[표1] 레벨별 전 구간 (R 배수 기준)")
    print(f"{'min_rr':>7} | {'거래수':>5} | {'월빈도':>6} | {'건당R':>7} | {'승률':>6} | "
          f"{'합계R':>8} | {'t값':>5} | 비고")
    print("-" * 100)
    months = (max(all_ts) - min(all_ts)) / 1000 / 86400 / 30.44
    rows_tbl = {}
    for lv in sorted(data):
        s = rstats(data[lv]["trades"])
        rows_tbl[lv] = s
        note = "<<< 현행 라이브" if lv == BASELINE else ""
        if s["n"] < MIN_N:
            note = ("표본부족 " + note).strip()
        print(f"{lv:>7.1f} | {s['n']:>5} | {s['n']/months:>6.2f} | {s['r_mean']:>+7.3f} | "
              f"{s['win']:>5.1f}% | {s['r_sum']:>+8.2f} | {s['t']:>+5.2f} | {note}")

    # ---- 표2: 집합 변화 — 추가 / 삭제 / 치환
    print("\n[표2] 기준선(2.0) 대비 체결셋 변화 — 포함관계가 아님을 확인")
    print(f"{'min_rr':>7} | {'유지':>5} | {'추가':>5} | {'소멸':>5} | 소멸 원인")
    print("-" * 100)
    incr = {}
    for lv in sorted(data):
        ks = {key_of(t) for t in data[lv]["trades"]}
        keep = len(ks & base_keys)
        added = [t for t in data[lv]["trades"] if key_of(t) not in base_keys]
        lost = [t for t in base if key_of(t) not in ks]
        incr[lv] = (added, lost)
        # 소멸 원인: 기준선 거래의 원본 RR 이 lv 미만이면 '문턱컷', 아니면 '슬롯치환'
        cut = sum(1 for t in lost if t["rr"] < lv - 1e-9)
        sub = len(lost) - cut
        cause = f"문턱컷 {cut} / 슬롯치환 {sub}" if lost else "-"
        print(f"{lv:>7.1f} | {keep:>5} | {len(added):>5} | {len(lost):>5} | {cause}")

    # ---- 표3: 증분 분석 (핵심)
    print("\n[표3] ★증분 분석 — 기준선 대비 '새로 들어온 거래'만 따로")
    print(f"{'min_rr':>7} | {'추가n':>5} | {'건당R':>7} | {'승률':>6} | {'합계R':>8} | "
          f"{'p(평균≠0)':>9} | 판정")
    print("-" * 100)
    incr_stats = {}
    for lv in sorted(data):
        added, _ = incr[lv]
        if not added:
            print(f"{lv:>7.1f} |     0 |       - |      - |        - |         - | 추가 없음")
            continue
        r = np.array([t["r"] for t in added], float)
        p = perm_mean_positive(r) if len(r) > 1 else float("nan")
        s = rstats(added)
        incr_stats[lv] = dict(s, p=p)
        if len(r) < MIN_N:
            v = "표본부족"
        elif r.mean() <= 0:
            v = "기각 — 추가분 적자"
        elif p >= 0.05:
            v = "기각 — 추가분 흑자가 무작위와 구분 안 됨"
        else:
            v = "유망"
        print(f"{lv:>7.1f} | {len(r):>5} | {r.mean():>+7.3f} | {(r>0).mean()*100:>5.1f}% | "
              f"{r.sum():>+8.2f} | {p:>9.4f} | {v}")

    # 소멸분(높은 min_rr 로 잘려나간 거래)도 같은 방식으로
    print("\n[표3-b] 소멸 분석 — 문턱을 올려서 '잘려나간 거래'만 따로")
    print(f"{'min_rr':>7} | {'소멸n':>5} | {'건당R':>7} | {'승률':>6} | {'합계R':>8} | 해석")
    print("-" * 100)
    for lv in sorted(data):
        _, lost = incr[lv]
        if not lost:
            print(f"{lv:>7.1f} |     0 |       - |      - |        - | 소멸 없음")
            continue
        r = np.array([t["r"] for t in lost], float)
        tag = "잘라서 이득" if r.mean() < 0 else "좋은 거래를 잘라냄"
        if len(r) < MIN_N:
            tag += " (표본부족)"
        print(f"{lv:>7.1f} | {len(r):>5} | {r.mean():>+7.3f} | {(r>0).mean()*100:>5.1f}% | "
              f"{r.sum():>+8.2f} | {tag}")

    # ---- 표4: 국면×방향 기저 통제
    print("\n[표4] 방향 분해 — '상승장 롱'의 재확인이 아닌지 통제")
    print(f"{'min_rr':>7} | {'롱n':>4} {'롱건당R':>8} {'롱승률':>7} | "
          f"{'숏n':>4} {'숏건당R':>8} {'숏승률':>7}")
    print("-" * 100)
    for lv in sorted(data):
        lo = rstats([t for t in data[lv]["trades"] if t["dir"] == "long"])
        sh = rstats([t for t in data[lv]["trades"] if t["dir"] == "short"])
        f = lambda s: (f"{s['n']:>4} {s['r_mean']:>+8.3f} {s['win']:>6.1f}%"
                       if s else f"{0:>4} {'-':>8} {'-':>7}")
        print(f"{lv:>7.1f} | {f(lo)} | {f(sh)}")

    print("\n[표4-b] 증분 거래의 방향 구성 (기저효과 확인)")
    for lv in sorted(incr_stats):
        added, _ = incr[lv]
        lo = [t for t in added if t["dir"] == "long"]
        sh = [t for t in added if t["dir"] == "short"]
        fl = f"롱 {len(lo)}건 {np.mean([t['r'] for t in lo]):+.3f}R" if lo else "롱 0건"
        fs = f"숏 {len(sh)}건 {np.mean([t['r'] for t in sh]):+.3f}R" if sh else "숏 0건"
        print(f"  {lv:.1f}: {fl} / {fs}")

    # ---- 표5: 연도 일관성
    print("\n[표5] 연도별 합계R (괄호=거래수) — 특정 연도 몰빵 기각용")
    years = sorted({year_of(t["ts"]) for d in data.values() for t in d["trades"]})
    print(f"{'min_rr':>7} | " + " | ".join(f"{y:>13}" for y in years))
    print("-" * 100)
    for lv in sorted(data):
        cells = []
        for y in years:
            sub = [t for t in data[lv]["trades"] if year_of(t["ts"]) == y]
            cells.append(f"{sum(t['r'] for t in sub):>+7.2f}({len(sub):>2})" if sub
                         else "      -      ")
        print(f"{lv:>7.1f} | " + " | ".join(f"{c:>13}" for c in cells))

    print("\n[표5-b] 증분 거래의 연도 분포 (몰빵 확인)")
    for lv in sorted(incr_stats):
        added, _ = incr[lv]
        yy = defaultdict(list)
        for t in added:
            yy[year_of(t["ts"])].append(t["r"])
        tot = sum(t["r"] for t in added)
        s = " ".join(f"{y}:{sum(v):+.2f}({len(v)})" for y, v in sorted(yy.items()))
        top = max(yy.items(), key=lambda kv: abs(sum(kv[1]))) if yy else None
        share = (abs(sum(top[1])) / abs(tot) * 100) if top and abs(tot) > 1e-9 else 0.0
        print(f"  {lv:.1f}: {s}   → 최대 연도 기여 {share:.0f}%")

    # ---- 표6: 복리 평가
    print("\n[표6] 복리 평가 — 7x · size 0.9 · 동시보유 분할 · DD 스로틀 · 파산판정")
    print(f"{'min_rr':>7} | {'거래수':>5} | {'동시보유':>7} | {'최종자산':>9} | {'MDD':>7} | "
          f"{'부트중앙':>9} | {'5%분위':>9} | {'파산확률':>8}")
    print("-" * 100)
    comp = {}
    for lv in sorted(data):
        c = compound_eval(data[lv]["trades"])
        comp[lv] = c
        print(f"{lv:>7.1f} | {len(data[lv]['trades']):>5} | {c['conc']:>7.2f} | "
              f"{c['compound']:>8.2f}x | {c['mdd']:>6.1f}% | {c['boot_p50']:>8.2f}x | "
              f"{c['boot_p05']:>8.2f}x | {c['ruin']:>7.1f}%")

    # ---- 표7: 기준선 대비 순열검정
    print("\n[판정A] 기준선(2.0) 대비 건당R 차이 — 비대응 순열검정 20000회")
    print(f"{'min_rr':>7} | {'ΔR/건':>8} | {'p':>7} | 판정")
    print("-" * 100)
    br = np.array([t["r"] for t in base], float)
    pvals = {}
    for lv in sorted(data):
        if lv == BASELINE:
            continue
        cr = np.array([t["r"] for t in data[lv]["trades"]], float)
        p = perm_two_sample(cr, br)
        pvals[lv] = p
        d = cr.mean() - br.mean()
        v = "기각 — 기준선과 구분 불가" if p >= 0.05 else (
            "유망" if d > 0 else "기각 — 기준선보다 나쁨")
        print(f"{lv:>7.1f} | {d:>+8.3f} | {p:>7.4f} | {v}")

    # ---- 판정 A-2: 증분 거래 vs 기준선 거래 직접 비교
    # "추가분이 기준선만큼은 하는가" — 낮출 가치의 직접 조건.
    print("\n[판정A-2] 증분 거래 vs 기준선 거래 (비대응 순열검정)")
    print(f"{'min_rr':>7} | {'증분n':>5} | {'증분건당R':>9} | {'기준선건당R':>11} | "
          f"{'차이':>8} | {'p':>7} | 판정")
    print("-" * 100)
    for lv in sorted(incr_stats):
        added, _ = incr[lv]
        ar = np.array([t["r"] for t in added], float)
        if len(ar) < 2:
            continue
        p = perm_two_sample(ar, br)
        d = ar.mean() - br.mean()
        v = ("표본부족" if len(ar) < MIN_N else
             ("증분이 기준선보다 유의하게 나쁨" if p < 0.05 and d < 0 else
              "차이 유의하지 않음(그러나 부호는 " + ("+" if d > 0 else "-") + ")"))
        print(f"{lv:>7.1f} | {len(ar):>5} | {ar.mean():>+9.3f} | {br.mean():>+11.3f} | "
              f"{d:>+8.3f} | {p:>7.4f} | {v}")

    # ---- 판정 A-3: 증분 거래의 원본 RR 구성 (증분의 정체 확인)
    print("\n[판정A-3] 증분 거래의 셋업 원본 RR 구성 — 저RR 구간이 맞는지")
    for lv in sorted(incr_stats):
        added, _ = incr[lv]
        rr = np.array([t["rr"] for t in added], float)
        low = rr[(rr >= lv - 1e-9) & (rr < BASELINE)]
        print(f"  {lv:.1f}: 증분 {len(rr)}건 중 원본RR<{BASELINE} 인 것 {len(low)}건 "
              f"(중앙 RR {np.median(rr):.2f}) / 기준선 거래 중앙 RR "
              f"{np.median([t['rr'] for t in base]):.2f}")

    # ---- 판정 B: 최종 후보 요약
    print("\n[판정B] 종합 — 후보별 기각/유망")
    print(f"{'min_rr':>7} | {'n':>4} | {'건당R':>7} | {'복리':>8} | {'p':>7} | 최종 판정")
    print("-" * 100)
    cands = []
    for lv in sorted(data):
        s = rstats(data[lv]["trades"])
        c = comp[lv]
        p = pvals.get(lv, float("nan"))
        if lv == BASELINE:
            verdict = "기준선"
        else:
            reasons = []
            if s["n"] < MIN_N:
                reasons.append("표본부족")
            if not (p < 0.05):
                reasons.append(f"순열검정 p={p:.3f}>=0.05")
            if s["r_mean"] <= rstats(base)["r_mean"]:
                reasons.append("건당R 기준선 이하")
            ist = incr_stats.get(lv)
            if ist and ist["n"] >= MIN_N and ist["r_mean"] <= 0:
                reasons.append("증분 적자")
            verdict = "유망" if not reasons else "기각 — " + " / ".join(reasons)
        print(f"{lv:>7.1f} | {s['n']:>4} | {s['r_mean']:>+7.3f} | {c['compound']:>7.2f}x | "
              f"{p:>7.4f} | {verdict}" if p == p else
              f"{lv:>7.1f} | {s['n']:>4} | {s['r_mean']:>+7.3f} | {c['compound']:>7.2f}x | "
              f"{'-':>7} | {verdict}")
        cands.append({
            "name": f"min_rr={lv:.1f}",
            "params": {"min_rr": lv},
            "n": s["n"], "r_mean": round(s["r_mean"], 4),
            "win": round(s["win"], 2), "r_sum": round(s["r_sum"], 3),
            "compound": round(c["compound"], 4), "mdd": round(c["mdd"], 2),
            "ruin": round(c["ruin"], 2),
            "boot_p50": round(c["boot_p50"], 4), "boot_p05": round(c["boot_p05"], 4),
            "p_perm": (round(p, 5) if p == p else None),
            "incr_n": (incr_stats[lv]["n"] if lv in incr_stats else 0),
            "incr_r_mean": (round(incr_stats[lv]["r_mean"], 4) if lv in incr_stats else None),
            "incr_win": (round(incr_stats[lv]["win"], 2) if lv in incr_stats else None),
            "incr_p": (round(incr_stats[lv]["p"], 5) if lv in incr_stats else None),
            "verdict": verdict,
        })

    # ---- 저장
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bs = rstats(base)
    bc = comp[BASELINE]
    out = {
        "axis": "min_rr (셋업 최소 손익비 게이트)",
        "candidates": cands,
        "baseline": {
            "name": f"min_rr={BASELINE:.1f} (현행 라이브)",
            "params": {"min_rr": BASELINE},
            "n": bs["n"], "r_mean": round(bs["r_mean"], 4),
            "win": round(bs["win"], 2), "r_sum": round(bs["r_sum"], 3),
            "compound": round(bc["compound"], 4), "mdd": round(bc["mdd"], 2),
            "ruin": round(bc["ruin"], 2),
            "boot_p50": round(bc["boot_p50"], 4), "boot_p05": round(bc["boot_p05"], 4),
            "p_perm": None,
        },
        "notes": (
            "min_rr 은 진입가/SL/TP 를 바꾸지 않는 순수 필터 → 살아남는 셋업의 R 은 동일, "
            "체결셋만 바뀐다(ote_level 과 달리 기계적 R 천장이 없음). "
            "다만 detect 의 (날짜,윈도우)당 첫 FVG 채택 규칙 때문에 집합이 포함관계가 아니며, "
            "문턱을 낮추면 이른 저RR FVG 가 슬롯을 선점해 기존 고RR 거래를 밀어내는 "
            "'슬롯치환'이 발생한다. 시도 조합 5개(1.5/1.8/2.0/2.5/3.0), 무보정. "
            f"페어 BTC+ETH, 순열검정 {N_PERM}회, 부트스트랩 {N_BOOT}회 복원추출, "
            "레버 7x·size 0.9·동시보유분할·DD스로틀·파산=시드20%. "
            "한계(live_parity GAPS): ote_up_level 0.786 · sweep_gate · smart_size · "
            "daily_loss_limit 미구현."
        ),
    }
    p = OUT_DIR / "min-rr.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {p.resolve()}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["ceiling", "run", "analyze"])
    ap.add_argument("--min-rr", type=float)
    a = ap.parse_args()
    if a.mode == "ceiling":
        ceiling()
    elif a.mode == "run":
        res = run_level(a.min_rr)
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        p = RAW_DIR / f"minrr_{a.min_rr:.1f}.json"
        p.write_text(json.dumps(res), encoding="utf-8")
        rs = [t["r"] for t in res["trades"]]
        avg = sum(rs) / len(rs) if rs else 0.0
        wr = 100.0 * sum(1 for x in rs if x > 0) / len(rs) if rs else 0.0
        print(f"min_rr={a.min_rr:.1f} n={res['n']} 건당={avg:+.3f}R 승률={wr:.0f}% → {p}",
              flush=True)
    else:
        analyze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
