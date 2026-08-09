"""#KZ-WIDE 2026-08-09: 아시아·런던 킬존을 다시 열면? — 파트너 요청.

## 왜
2026-05-28 에 "미장(NYSE 09:30~16:00 ET) 안의 킬존/매크로/SB 만" 으로 좁혔다.
계기는 **거래 한 건**이었다 — 뉴욕 03:02(런던 킬존, 미장 밖) 진입이 −283 USDT.
ICT 정통 검토를 붙이긴 했지만 **백테로 검증한 적이 없다.**

지금은 잴 수 있다. 정합도 맞췄고 검증 절차도 있다.

## 변형 (사전등록)
    NYSE     현행 — 미장 안의 킬존/매크로/SB          (nyse_gate=True)
    WIDE     미장 제약 제거 — 킬존/매크로/SB 전부      (nyse_gate=False)
    ALL24    시간 필터 자체 제거 (무료 계정 정책)      (disable_time_filter=True)

WIDE 가 파트너 요청("모든 킬존·실버불릿, NY AM 도 전체")에 해당한다.
아시아(19~24 ET)·런던(02~05 ET)·런던 SB(03~04)·NY AM 전체(07~10)가 열린다.

## 판정 — MMBM 홀드아웃에 쓴 것과 같은 4관문
    ① 건당 R 95% 구간이 0 초과      ② 심볼 일관성
    ③ 롱/숏 양쪽 생존               ④ 순열검정 p<0.05
현행(NYSE) 대비 **개선**이 있어야 의미가 있으므로 페어드 비교도 함께 본다.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from live_parity import run_live_parity  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT"]
HOLDOUT = ["SOLUSDT", "XRPUSDT", "DOGEUSDT"]
RNG = np.random.default_rng(20260809)
N_BOOT = 20000
N_PERM = 20000
MIN_N = 30

# [08-09] ALL24(시간필터 없음)는 캐시가 따로라 페어당 2시간이 더 든다.
# 판정에 필요한 건 "현행 vs 확대" 대조이므로 1차에서는 뺀다. WIDE 가 살아남으면
# 그때 참고군으로 붙인다.
VARIANTS = (
    ("NYSE (현행)", {}),
    ("WIDE (모든 킬존·SB)", {"nyse_gate": False}),
)


def ci(r: np.ndarray) -> tuple[float, float]:
    m = np.array([r[RNG.integers(0, len(r), len(r))].mean() for _ in range(N_BOOT)])
    return tuple(np.percentile(m, [2.5, 97.5]))


def rows_for(syms: list[str], extra: dict) -> list[dict]:
    out = []
    for s in syms:
        df5, kept, _ = run_live_parity(s, extra or None)
        for t in kept:
            risk = abs(t.entry - t.entry_sl)
            if risk <= 0 or t.entry <= 0:
                continue
            out.append({
                "sym": s,
                "r": float(t.raw_pnl_pct) * float(t.entry) / risk,
                "dir": str(getattr(t.direction, "value", t.direction)).lower(),
                "ts": df5.index[t.entry_idx],
                "months": (df5.index[-1] - df5.index[0]).days / 30.4,
            })
    return out


def report(tag: str, rows: list[dict], base: np.ndarray | None) -> np.ndarray | None:
    if len(rows) < MIN_N:
        print(f"  {tag:<24}{len(rows):>6}  표본부족", flush=True)
        return None
    r = np.array([x["r"] for x in rows])
    months = max(x["months"] for x in rows)
    lo, hi = ci(r)
    mark = "★0초과" if lo > 0 else ("적자" if hi < 0 else "0포함")
    print(f"  {tag:<24}{len(r):>6}{len(r) / max(months, 1e-9):>8.2f}"
          f"{r.mean():>+9.3f}{100 * (r > 0).mean():>6.0f}%"
          f"   [{lo:+.3f} ~ {hi:+.3f}]  {mark}", flush=True)

    # 심볼·방향 (관문 ②③)
    syms = sorted({x["sym"] for x in rows})
    pos = sum(1 for s in syms
              if len([x for x in rows if x["sym"] == s]) >= 10
              and np.mean([x["r"] for x in rows if x["sym"] == s]) > 0)
    jud = sum(1 for s in syms if len([x for x in rows if x["sym"] == s]) >= 10)
    parts = [f"심볼 {pos}/{jud}"]
    for d in ("long", "short"):
        sub = np.array([x["r"] for x in rows if x["dir"] == d])
        if len(sub) >= MIN_N:
            parts.append(f"{'롱' if d == 'long' else '숏'} {sub.mean():+.3f}")
    print(f"  {'':<24}{' · '.join(parts)}", flush=True)

    # 현행 대비 (관문 ④ — 두 표본 라벨 순열)
    if base is not None and len(base) >= MIN_N:
        obs = r.mean() - base.mean()
        both = np.concatenate([base, r])
        nb = len(base)
        d_ = np.empty(N_PERM)
        for k in range(N_PERM):
            p = RNG.permutation(both)
            d_[k] = p[nb:].mean() - p[:nb].mean()
        pv = float((d_ >= obs).mean())
        print(f"  {'':<24}현행 대비 {obs:+.3f}R · p={pv:.4f}"
              f"  {'유의' if pv < 0.05 else '유의하지 않음'}", flush=True)
    return r


def block(title: str, syms: list[str]) -> None:
    print(f"\n### {title}", flush=True)
    print(f"  {'변형':<24}{'거래':>6}{'월빈도':>8}{'건당R':>9}{'승률':>7}"
          f"   {'95% 구간':<22}", flush=True)
    base = None
    for name, extra in VARIANTS:
        t0 = time.time()
        try:
            rows = rows_for(syms, extra)
        except Exception as e:  # noqa: BLE001
            print(f"  {name:<24}실패 — {type(e).__name__}: {str(e)[:60]}", flush=True)
            continue
        r = report(name, rows, base)
        if base is None:
            base = r
        print(f"  {'':<24}({time.time() - t0:.0f}초)", flush=True)


def main() -> int:
    print("=== 킬존 확대 검증 — 2026-05-28 '미장 안에서만' 결정 재검토", flush=True)
    print("  계기였던 거래 #6(뉴욕 03:02 런던 킬존, -283 USDT)은 표본 1건이었다.",
          flush=True)
    block("본표본 BTC+ETH", PAIRS)
    # 홀드아웃은 본표본에서 개선이 확인된 뒤에만 돌린다 — 페어당 2시간이라
    # 개선이 없는데 미리 돌리면 10시간을 버린다.
    if "--holdout" in sys.argv:
        block("홀드아웃 SOL+XRP+DOGE", HOLDOUT)
    else:
        print("\n  (홀드아웃은 --holdout 로 별도 실행 — 본표본 결과를 먼저 본다)",
              flush=True)
    print("\n  판정 — WIDE 가 현행보다 유의하게 낫고(p<0.05) 홀드아웃에서도"
          " 재현돼야 되돌린다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
