"""#AUTONOMOUS 2026-07-29: 임펄스+피보확장 v3 — 후보 검증 3종.

v2 후보: win12 ATR×4.0 되돌림0.382 → 확장1.0 (n=1760 net+6868% 승률67% 페어6/7).
의심 3가지를 정면으로 친다:

 [V1] **같은 봉 진입+청산 배제** — 1h 봉 안에서 지정가 체결과 TP 도달이 같이
      일어나면 순서를 알 수 없다. 얕은 되돌림(0.382)+가까운 TP(고점) 조합은 이게
      특히 잦다. 청산 스캔을 fill+1 부터 시작해 재측정. 낙관 편향의 크기를 잰다.
 [V2] **플라시보** — 임펄스와 무관한 무작위 시점에 동일 구조(같은 leg 크기·같은
      진입/TP/SL 비율)로 매매. 임펄스라는 조건이 진짜 정보인지, 아니면 그냥
      "얕은 되돌림+가까운 TP = 고승률 구조"라서 아무 데서나 나오는지 판별.
      → 플라시보도 흑자면 임펄스는 무의미(구조 효과일 뿐).
 [V3] **ATR 미세 그리드 3.0~5.0 step 0.25** — v2 에서 3.5 는 붕괴(페어3/7 H2 음수),
      4.0·4.5 는 통과. 절벽인지 연속인지. 우연한 섬이면 기각.
 [V4] 비용 민감도 0.08% → 0.11%/0.15% (지정가 미체결·슬리피지 현실 반영)

판정: V1 통과(감쇠 후에도 흑자·페어 5/7) + V2 플라시보 대비 유의한 우위 +
      V3 연속 흑자 구간 + V4 비용 0.15% 에서도 흑자 → 배포 후보. 하나라도 실패 시 기각.
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts")
from impulse_fib_v2 import PAIRS, impulses, line, load, stat  # noqa: E402

TTL_RETR = 24
HOLD = 240


def sim(data, win, k, mp, gap, f, e, cost=0.0008, skip_fill_bar=False, placebo_rng=None):
    """placebo_rng 가 있으면 임펄스 시점을 무작위 시점으로 대체(leg 크기·방향은 유지)."""
    out = []
    for sym, (df, o, c, h, lo, atr) in data.items():
        n = len(c)
        imps = impulses(c, h, lo, atr, win, k, mp, gap)
        if placebo_rng is not None:
            # 같은 개수·같은 leg 비율·같은 방향이되 시점만 무작위 → 구조 효과 분리
            legs = [(abs(end - start) / max(c[i], 1e-12), d) for (i, start, end, d) in imps]
            picks = placebo_rng.choice(np.arange(win + 20, n - HOLD - 2),
                                       size=min(len(legs), max(n - HOLD - win - 25, 1)),
                                       replace=False)
            imps = []
            for idx, (legr, d) in zip(sorted(picks), legs):
                px = c[idx]
                leg = legr * px
                end = px
                start = px - d * leg
                imps.append((idx, start, end, d))
        busy_until = -1
        for (i, start, end, d) in imps:
            if i <= busy_until:
                continue
            leg = abs(end - start)
            if leg <= 0:
                continue
            entry = end - d * f * leg
            fill = None
            for j in range(i + 1, min(i + 1 + TTL_RETR, n)):
                if lo[j] <= entry <= h[j]:
                    fill = j; break
            if fill is None:
                continue
            sl = start
            risk = abs(entry - sl)
            if risk <= 0 or risk / entry < 0.002:
                continue
            tp = end + d * (e - 1.0) * leg if e > 1.0 else end
            if (d == 1 and tp <= entry) or (d == -1 and tp >= entry):
                continue
            j0 = fill + 1 if skip_fill_bar else fill
            raw = 0.0
            exit_j = min(fill + HOLD, n - 1)
            for j in range(j0, exit_j + 1):
                if d == 1:
                    if lo[j] <= sl:
                        raw = (sl - entry) / entry; exit_j = j; break
                    if h[j] >= tp:
                        raw = (tp - entry) / entry; exit_j = j; break
                else:
                    if h[j] >= sl:
                        raw = (entry - sl) / entry; exit_j = j; break
                    if lo[j] <= tp:
                        raw = (entry - tp) / entry; exit_j = j; break
            else:
                raw = ((c[exit_j] - entry) / entry) * d
            out.append((df.index[fill], (raw - cost) * 18.0 * 100, sym))
            busy_until = exit_j
    return out


def main() -> int:
    data = {sym: load(sym) for sym in PAIRS}
    W, K, MP, GAP, F, E = 12, 4.0, 0.20, 24, 0.382, 1.0

    print("===== [V1] 같은 봉 진입+청산 배제 =====", flush=True)
    base = stat(sim(data, W, K, MP, GAP, F, E))
    strict = stat(sim(data, W, K, MP, GAP, F, E, skip_fill_bar=True))
    print(f"  원본(같은봉 허용): {line(base)}", flush=True)
    print(f"  엄격(fill+1 부터): {line(strict)}", flush=True)
    if base and strict:
        print(f"  → 감쇠 {100 * (1 - strict['net'] / base['net']):.0f}% "
              f"(승률 {base['wr']:.0f}%→{strict['wr']:.0f}%)", flush=True)

    print("\n===== [V2] 플라시보 (임펄스 무관 무작위 시점, 동일 구조) =====", flush=True)
    for seed in (0, 1, 2):
        rng = np.random.default_rng(seed)
        p = stat(sim(data, W, K, MP, GAP, F, E, skip_fill_bar=True, placebo_rng=rng))
        print(f"  seed{seed}: {line(p)}", flush=True)

    print("\n===== [V3] ATR 미세 그리드 (엄격 기준) =====", flush=True)
    for k in (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0):
        s = stat(sim(data, W, k, MP, GAP, F, E, skip_fill_bar=True))
        mark = "★" if s and s["ok"] else " "
        print(f"  {mark}ATR×{k:<5}: {line(s)}", flush=True)

    print("\n===== [V4] 비용 민감도 (엄격 기준) =====", flush=True)
    for cost in (0.0008, 0.0011, 0.0015):
        s = stat(sim(data, W, K, MP, GAP, F, E, cost=cost, skip_fill_bar=True))
        print(f"  비용{100 * cost:.2f}%: {line(s)}", flush=True)

    print("\n===== [참고] 엄격 기준 페어별 =====", flush=True)
    if strict:
        print("  " + " ".join(f"{a.replace('USDT', '')}:{b:+.0f}"
                              for a, b in strict["syms"].items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
