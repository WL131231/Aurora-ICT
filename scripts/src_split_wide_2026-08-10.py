"""#SRC-SPLIT-WIDE 2026-08-10: 킬존 전면 개방 + 소스별 기여도 — 파트너 지시.

현행 킬존(미장 안)에서 낸 결론:
    fvg          141건 +0.122R
    turtle_soup  234건 **-0.238R [-0.381 ~ -0.087] 적자확정** (심볼 2/2 · 롱숏 양쪽)
    implied_fvg   21건 +0.587R
    → turtle 제거 시 402→168건 +0.171R (본표본 p=0.0010 / 홀드아웃 p=0.0534)

홀드아웃 p 가 0.0534 로 문턱을 간발로 넘는다. 파트너 지시대로 **킬존을 전부 열어**
표본을 늘린 상태에서 다시 잰다(nyse_gate=False — 아시아·런던·런던SB·NY AM 전체).
킬존 확대 자체는 성적에 차이가 없음이 확인됐으므로(8/9, p=0.48) 표본만 늘어난다.

타임라인 캐시가 새로 필요해 페어당 약 2시간이 든다. 본표본(BTC+ETH) 먼저 돌리고,
결과가 유망하면 홀드아웃을 이어서 돌린다.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from live_parity import run_live_parity  # noqa: E402

import os as _os
# [2차] 홀드아웃 — 탐색에 안 쓴 알트. HOLDOUT=1 로 전환한다.
_HO = _os.environ.get("HOLDOUT") == "1"
PAIRS = (["SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
         if _HO else ["BTCUSDT", "ETHUSDT"])
OUT = ("data/axis/src_wide_holdout_ts.json" if _HO
       else "data/axis/src_wide_rows_ts.json")
SOURCES = ("turtle_soup", "implied_fvg", "mitigation_block", "rejection_block")
RNG = np.random.default_rng(20260810)
N_BOOT, N_PERM, MIN_N = 20000, 20000, 30


def ci(r):
    m = np.array([r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def perm(a, b):
    obs = b.mean() - a.mean()
    both = np.concatenate([a, b])
    na = len(a)
    d = np.empty(N_PERM)
    for k in range(N_PERM):
        p = RNG.permutation(both)
        d[k] = p[na:].mean() - p[:na].mean()
    return float((d >= obs).mean())


def collect() -> list[dict]:
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            rows = json.load(f)
        print(f"  (캐시 재사용 — {len(rows)}건)", flush=True)
        return rows
    rows = []
    for sym in PAIRS:
        t0 = time.time()
        print(f"  {sym} 계산 중 (킬존 전면 개방, 타임라인 재빌드) …", flush=True)
        df5, kept, _ = run_live_parity(sym, {"nyse_gate": False})
        for t in kept:
            risk = abs(float(t.entry) - float(getattr(t, "entry_sl", 0.0) or 0.0))
            if risk <= 0 or t.entry <= 0:
                continue
            conf = tuple(t.confluences)
            src = "mmbm" if "mmbm" in conf else next(
                (s for s in SOURCES if s in conf), "fvg")
            # [08-15] 진입 시각 추가 — 연도별 분석에 필요하다. 처음엔 안 넣어서
            # "구간 1/5" 같은 상대 위치로만 볼 수 있었고, 그게 몇 년도인지
            # 답할 수 없었다.
            _ts = df5.index[t.entry_idx] if hasattr(df5, "index") else None
            rows.append({
                "sym": sym, "src": src,
                "r": float(t.raw_pnl_pct) * float(t.entry) / risk,
                "dir": str(getattr(t.direction, "value", t.direction)).lower(),
                "ts": (_ts.isoformat() if _ts is not None else None),
                "year": (int(_ts.year) if _ts is not None else None),
            })
        print(f"    {sym} {len([x for x in rows if x['sym'] == sym])}건"
              f"  ({time.time() - t0:.0f}초)", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(rows, f)
    return rows


def main() -> int:
    print(f"=== 킬존 전면 개방 + 소스별 기여도 "
      f"({'홀드아웃 알트5' if _HO else '본표본 BTC+ETH'})", flush=True)
    rows = collect()
    if not rows:
        print("  거래 0건", flush=True)
        return 0
    syms = sorted({x["sym"] for x in rows})

    print(f"\n  {'소스':<18}{'거래':>6}{'비중':>7}{'건당R':>9}{'승률':>7}"
          f"   {'95% 구간':<22}{'심볼':>6}  {'롱/숏':<20}", flush=True)
    for s in ("fvg", *SOURCES, "mmbm"):
        sub = [x for x in rows if x["src"] == s]
        if len(sub) < 10:
            print(f"  {s:<18}{len(sub):>6}  표본부족", flush=True)
            continue
        r = np.array([x["r"] for x in sub])
        lo, hi = ci(r) if len(r) >= MIN_N else (float("nan"),) * 2
        mark = "★0초과" if lo > 0 else ("적자확정" if hi < 0 else "0포함")
        same = jud = 0
        for sy in syms:
            rr = np.array([x["r"] for x in sub if x["sym"] == sy])
            if len(rr) < 10:
                continue
            jud += 1
            same += int((rr.mean() > 0) == (r.mean() > 0))
        parts = []
        for d_ in ("long", "short"):
            rr = np.array([x["r"] for x in sub if x["dir"] == d_])
            if len(rr) >= 10:
                parts.append(f"{'롱' if d_ == 'long' else '숏'} {rr.mean():+.2f}")
        print(f"  {s:<18}{len(r):>6}{100 * len(r) / len(rows):>6.1f}%"
              f"{r.mean():>+9.3f}{100 * (r > 0).mean():>6.0f}%"
              f"   [{lo:+.3f} ~ {hi:+.3f}] {mark:<8}{same}/{jud:<4}  "
              f"{' · '.join(parts):<20}", flush=True)

    print(f"\n  {'제거하면':<18}{'남는 거래':>9}{'건당R':>9}{'95% 구간':>24}{'순열 p':>9}",
          flush=True)
    for s in SOURCES:
        keep = np.array([x["r"] for x in rows if x["src"] != s])
        drop = np.array([x["r"] for x in rows if x["src"] == s])
        if len(keep) < MIN_N or len(drop) < MIN_N:
            continue
        lo, hi = ci(keep)
        print(f"  −{s:<17}{len(keep):>9}{keep.mean():>+9.3f}"
              f"   [{lo:+.3f} ~ {hi:+.3f}]{perm(drop, keep):>9.4f}", flush=True)
    tot = np.array([x["r"] for x in rows])
    print(f"  {'(제거 없음)':<18}{len(tot):>9}{tot.mean():>+9.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
