"""#SRC-ALL 2026-08-10: ICT 매매법 전수 테스트 — 파트너 지시 "전부 다".

## 대상
현재 진입에 쓰는 것 + 정통 PD-array 중 미구현이던 것 전부.

    기존   fvg · turtle_soup · implied_fvg · rejection_block · mitigation_block
    추가   ifvg(뒤집힌 FVG) · breaker(깨진 OB) · unicorn(breaker∩FVG) · bpr
    제외   vacuum — 크립토 6만봉에서 검출 0건(24시간 시장이라 갭이 없다). 종결.

## 방법
각 소스를 **하나씩만** 켜서 그 소스 단독 성적을 낸다. 기존 4소스는 이미 켜져 있으므로
`fvg_only` 기준선과 대조한다. SL/TP 규약은 전부 동일하게 맞췄다(구간 중앙 진입 ·
반대편 + 버퍼 SL · ATR 바닥 · 다음 미스윕 스윙 TP · min_rr 게이트) — 그래야 성적
차이가 **소스 때문**인지 청산 규칙 때문인지 갈린다.

## 판정 — 4관문
    ① 건당 R 95% 구간이 0 초과   ② 심볼 일관성
    ③ 롱/숏 양쪽 생존            ④ 순열검정(부호 셔플) p<0.05
본표본(BTC+ETH) 통과분만 홀드아웃으로 넘긴다.

## 비용
소스마다 타임라인 캐시가 따로라 페어당 재빌드가 필요하다. 1차는 최근 12만 봉
(약 14개월)으로 방향만 본다.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from live_parity import live_cfg  # noqa: E402

from aurora_ict.backtest.replay import run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT"]
BARS = 120_000
OUT = "data/axis/all_sources.json"
NEW = ("ifvg", "breaker", "unicorn", "bpr")
RNG = np.random.default_rng(20260810)
N_BOOT, N_PERM, MIN_N = 20000, 20000, 30


def ci(r):
    m = np.array([r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def sign_perm(r: np.ndarray) -> float:
    """부호 순열 — 방향 정보가 없다는 귀무가설."""
    obs = r.mean()
    d = np.array([(r * RNG.choice([-1.0, 1.0], size=len(r))).mean()
                  for _ in range(N_PERM)])
    return float((d >= obs).mean())


def run(sym: str, extra: dict) -> list[dict]:
    df = _resample(_load_full(sym)).tail(BARS)
    cfg = live_cfg(sym, extra or None)
    tl = cached_setup_timeline(df, cfg, sym)
    bt = run_backtest_from_timeline(df, tl, cfg)
    out = []
    for t in bt.trades:
        risk = abs(float(t.entry) - float(getattr(t, "entry_sl", 0.0) or 0.0))
        if risk <= 0 or t.entry <= 0:
            continue
        conf = tuple(t.confluences)
        src = ("mmbm" if "mmbm" in conf
               else next((k for k in (*NEW, "turtle_soup", "implied_fvg",
                                      "mitigation_block", "rejection_block")
                          if k in conf), "fvg"))
        out.append({
            "sym": sym, "src": src,
            "r": float(t.raw_pnl_pct) * float(t.entry) / risk,
            "dir": str(getattr(t.direction, "value", t.direction)).lower(),
        })
    return out


def collect() -> dict[str, list[dict]]:
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            return json.load(f)
    res: dict[str, list[dict]] = {}
    # 기준선 — 현행(기존 4소스)
    print("  [기준선] 현행 …", flush=True)
    t0 = time.time()
    res["BASE"] = [r for s in PAIRS for r in run(s, {})]
    print(f"    {len(res['BASE'])}건 ({time.time() - t0:.0f}초)", flush=True)
    # 새 소스 하나씩
    for k in NEW:
        t0 = time.time()
        print(f"  [+{k}] …", flush=True)
        res[k] = [r for s in PAIRS for r in run(s, {"research_sources": (k,)})]
        print(f"    {len(res[k])}건 ({time.time() - t0:.0f}초)", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f)
    return res


def report(tag: str, rows: list[dict], syms: list[str]) -> None:
    if len(rows) < 10:
        print(f"  {tag:<18}{len(rows):>6}  표본부족", flush=True)
        return
    r = np.array([x["r"] for x in rows])
    lo, hi = ci(r) if len(r) >= MIN_N else (float("nan"),) * 2
    g1 = lo > 0
    same = jud = 0
    for sy in syms:
        rr = np.array([x["r"] for x in rows if x["sym"] == sy])
        if len(rr) < 10:
            continue
        jud += 1
        same += int((rr.mean() > 0) == (r.mean() > 0))
    g2 = jud > 0 and same == jud
    parts, g3 = [], True
    for d_ in ("long", "short"):
        rr = np.array([x["r"] for x in rows if x["dir"] == d_])
        if len(rr) >= 10:
            parts.append(f"{'롱' if d_ == 'long' else '숏'} {rr.mean():+.2f}")
            g3 &= (rr.mean() > 0) == (r.mean() > 0)
    p = sign_perm(r) if len(r) >= MIN_N else float("nan")
    g4 = p < 0.05
    gates = f"{'①' if g1 else '·'}{'②' if g2 else '·'}{'③' if g3 else '·'}{'④' if g4 else '·'}"
    print(f"  {tag:<18}{len(r):>6}{r.mean():>+9.3f}{100 * (r > 0).mean():>6.0f}%"
          f"   [{lo:+.3f} ~ {hi:+.3f}]{p:>8.4f}  {gates:<6}{' · '.join(parts)}",
          flush=True)


def main() -> int:
    print(f"=== ICT 소스 전수 테스트 — 최근 {BARS:,}봉", flush=True)
    res = collect()
    syms = PAIRS

    print(f"\n  {'소스':<18}{'거래':>6}{'건당R':>9}{'승률':>7}"
          f"   {'95% 구간':<22}{'순열p':>8}  {'관문':<6}{'롱/숏'}", flush=True)
    print("  --- 기존 (현행 진입에 이미 포함) ---", flush=True)
    base = res.get("BASE", [])
    for s in ("fvg", "turtle_soup", "implied_fvg", "rejection_block",
              "mitigation_block", "mmbm"):
        report(s, [x for x in base if x["src"] == s], syms)

    print("  --- 신규 (이번에 구현) ---", flush=True)
    for k in NEW:
        rows = res.get(k, [])
        report(k, [x for x in rows if x["src"] == k], syms)

    print("\n  --- 전체 조합 성적 ---", flush=True)
    b = np.array([x["r"] for x in base])
    print(f"  {'현행 전체':<18}{len(b):>6}{b.mean():>+9.3f}", flush=True)
    for k in NEW:
        rows = res.get(k, [])
        if rows:
            a = np.array([x["r"] for x in rows])
            print(f"  {'현행 + ' + k:<18}{len(a):>6}{a.mean():>+9.3f}"
                  f"   (현행 대비 {a.mean() - b.mean():+.3f}R)", flush=True)

    print("\n  관문 ①구간0초과 ②심볼일관 ③롱숏양쪽 ④순열p<0.05", flush=True)
    print("  넷 다 통과한 것만 홀드아웃으로 넘긴다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
