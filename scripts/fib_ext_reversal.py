"""#AUTONOMOUS 2026-07-29: 피보 확장 = **변곡점 진입 신호** (파트너 실제 방식).

앞선 v1~v3 은 확장 레벨을 TP 로 놓고 임펄스를 **따라갔다** → 기각.
파트너 정정: 확장 레벨은 **파동의 끝(소진 지점)** 을 찾는 도구. 거기가 변곡점이니
**역방향 진입**한다. 차트의 -0.618/-1/-1.618/-2.618 이 그것.

좌표 정리 (임펄스 start→end, leg=|end-start|):
    확장 1.618 = end + 0.618×leg   (차트 -0.618)
    확장 2.0   = end + 1.000×leg   (차트 -1)
    확장 2.236 = end + 1.236×leg
    확장 2.618 = end + 1.618×leg   (차트 -1.618)
    확장 3.0   = end + 2.000×leg
    확장 3.618 = end + 2.618×leg   (차트 -2.618)

[1단계] gross 검정 — 매매 없이 "확장 레벨 도달 후 실제로 꺾이는가".
  각 레벨 도달 시점에서 **역방향** 정규화 수익률 +6/12/24/48봉 평균·승률·t.
  비교군 = 같은 페어 무작위 시점(기저). 레벨이 멀수록 반전력이 커지면 가설 지지.
[2단계] 매매화 — 레벨 터치 역진입, SL=다음 레벨 너머, TP=되돌림 피보.
같은 봉 진입+청산 배제(fill+1) 기본 적용 — 7/29 교훈.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from impulse_fib_v2 import PAIRS, impulses, load  # noqa: E402

LEVELS = (1.618, 2.0, 2.236, 2.618, 3.0, 3.618)
HZ = (6, 12, 24, 48)
WATCH = 240          # 임펄스 후 확장 도달을 기다리는 최대 봉
INVALID_RETR = 0.5   # 끝에서 leg 의 0.5 되돌리면 그 파동은 종료(확장 실패)


def touches(c, h, lo, atr, win, k, mp, gap, n):
    """임펄스별로 각 확장 레벨에 **처음 도달한 시점**을 수집.

    Returns: [(j, level, d, leg, end, start)] — j 는 도달 봉.
    """
    out = []
    for (i, start, end, d) in impulses(c, h, lo, atr, win, k, mp, gap):
        leg = abs(end - start)
        if leg <= 0:
            continue
        invalid = end - d * INVALID_RETR * leg
        pending = {lv: end + d * (lv - 1.0) * leg for lv in LEVELS}
        for j in range(i + 1, min(i + 1 + WATCH, n)):
            # 파동 종료 판정 먼저(보수적)
            if (d == 1 and lo[j] <= invalid) or (d == -1 and h[j] >= invalid):
                break
            hit = [lv for lv, tgt in pending.items()
                   if (d == 1 and h[j] >= tgt) or (d == -1 and lo[j] <= tgt)]
            for lv in hit:
                out.append((j, lv, d, leg, end, start))
                del pending[lv]
            if not pending:
                break
    return out


def main() -> int:
    data = {sym: load(sym) for sym in PAIRS}
    win, k, mp, gap = 12, 3.0, 0.20, 24
    rows = []
    for sym, (df, o, c, h, lo, atr) in data.items():
        n = len(c)
        for (j, lv, d, leg, end, start) in touches(c, h, lo, atr, win, k, mp, gap, n):
            if j + max(HZ) >= n - 1:
                continue
            base = o[j + 1]              # 도달 다음봉 시가 진입(인과)
            if base <= 0:
                continue
            r = dict(sym=sym, lv=lv, ts=df.index[j])
            for hz in HZ:
                # **역방향** 수익률 — 상승 임펄스면 숏
                r[f"r{hz}"] = (c[j + hz] - base) / base * 100 * (-d)
            rows.append(r)
        # 기저 — 무작위 시점 역방향(방향 무작위)
        rng = np.random.default_rng(abs(hash(sym)) % 2**31)
        pick = rng.choice(np.arange(50, n - max(HZ) - 2), size=min(2000, n - 100),
                          replace=False)
        for j in pick:
            base = o[j + 1]
            if base <= 0:
                continue
            dd = 1 if rng.random() > 0.5 else -1
            r = dict(sym=sym, lv=0.0, ts=df.index[j])
            for hz in HZ:
                r[f"r{hz}"] = (c[j + hz] - base) / base * 100 * dd
            rows.append(r)
    d = pd.DataFrame(rows)
    print("=== [1단계] 확장 레벨 도달 후 **역방향** gross (매매 아님) ===", flush=True)
    print(f"임펄스 정의: win{win} ATR×{k} 되돌림≤{mp} · 무효화={INVALID_RETR}되돌림\n", flush=True)
    print(f"{'레벨':<10} {'n':>6} " + "  ".join(f"{'h' + str(z):>24}" for z in HZ), flush=True)
    for lv in (0.0,) + LEVELS:
        sub = d[d.lv == lv]
        if len(sub) < 30:
            continue
        name = "기저(무작위)" if lv == 0 else f"확장 {lv}"
        parts = []
        for hz in HZ:
            col = sub[f"r{hz}"].dropna()
            t = col.mean() / (col.std() / np.sqrt(len(col)) + 1e-12)
            parts.append(f"{col.mean():+.3f}%(승{100 * (col > 0).mean():.0f}% t{t:+.1f})".rjust(24))
        print(f"{name:<10} {len(sub):6,} " + "  ".join(parts), flush=True)
    print("\n→ 기저 대비 평균·t 가 유의하게 크면(양수) 변곡점 가설 지지 → 2단계 매매화",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
