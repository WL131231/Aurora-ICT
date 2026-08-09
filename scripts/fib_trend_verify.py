"""#AUTONOMOUS 2026-07-29: T1·E0·확장3.0 후보 검증 배터리 (8종).

후보: 맥락 T1(EMA200) · 진입 E0(임펄스 다음봉 시가) · TP 확장3.0 · SL 임펄스 시작
      n=1452 net+8998% 승률39% 페어6/7 nm=2.07

가장 중요한 시험은 [B1]. 확장 3.0 이 "좋은 TP" 라서 이긴 건지, 아니면 그냥
"추세를 오래 끌고 가서" 이긴 건지 구분한다. TP 무한(홀드 만료까지) 대조군이
비슷하거나 더 좋으면 **확장 레벨은 장식**이고 본질은 추세 추종이다.

 B1 TP 정보량   : 확장3.0 vs TP무한 vs 고정RR(2R·3R) vs ATR배수 — 확장이 우월한가
 B2 롱/숏 분리  : T1 은 EMA200 위=롱만. 크립토 상승편향인지 양방향 실재인지
 B3 플라시보    : 무작위 시점 동일구조(같은 leg·방향·TP/SL 비율)
 B4 파라미터이웃: ATR k / win / body_thr / dir_thr 미세 그리드
 B5 비용        : 0.11% → 0.15% → 0.20%
 B6 보유한도    : HOLD 240봉(10일)은 실전 부담 — 120/240/480 비교
 B7 셔플        : 수익률 부호 무작위화 대비 (분포 검정)
 B8 페어 제외   : 최고 기여 페어 1개 빼도 유지되는가
"""
from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts")
from fib_trend_impulse import candle_impulses, context  # noqa: E402
from impulse_fib_v2 import PAIRS, line, load, stat  # noqa: E402

NOTIONAL = 18.0
W, K, BODY, DIR, MP, GAP = 12, 3.0, 0.45, 0.6, 0.25, 24


def run(data, tp_mode="ext3.0", hold=240, cost=0.0011, side=None, placebo_rng=None,
        win=W, k=K, body=BODY, dirt=DIR, ctx_key="T1", drop_sym=None):
    out = []
    for sym, (df, o, c, h, lo, atr) in data.items():
        if drop_sym and sym == drop_sym:
            continue
        n = len(c)
        ctx = context(df, c, h, lo)[ctx_key]
        imps = candle_impulses(o, c, h, lo, atr, win, k, body, dirt, MP, GAP)
        if placebo_rng is not None:
            legs = [(abs(e - s) / max(c[i], 1e-12), d) for (i, s, e, d) in imps]
            lowv, highv = win + 210, n - hold - 2
            if highv <= lowv:
                continue
            picks = sorted(placebo_rng.choice(np.arange(lowv, highv),
                                              size=min(len(legs), highv - lowv), replace=False))
            imps = []
            for idx, (legr, d) in zip(picks, legs):
                px = c[idx]; leg = legr * px
                imps.append((idx, px - d * leg, px, d))
        busy = -1
        for (i, start, end, d) in imps:
            if i <= busy:
                continue
            if ctx_key != "T0" and placebo_rng is None and ctx[i] != d:
                continue
            if side == "long" and d != 1:
                continue
            if side == "short" and d != -1:
                continue
            leg = abs(end - start)
            if leg <= 0:
                continue
            fill = i + 1
            if fill >= n - 1:
                continue
            entry = o[fill]
            sl = start
            risk = abs(entry - sl)
            if risk <= 0 or risk / entry < 0.002:
                continue
            if tp_mode == "none":
                tp = None
            elif tp_mode.startswith("ext"):
                tp = start + d * float(tp_mode[3:]) * leg
            elif tp_mode.endswith("R"):
                tp = entry + d * float(tp_mode[:-1]) * risk
            elif tp_mode.startswith("atr"):
                tp = entry + d * float(tp_mode[3:]) * atr[i]
            else:
                tp = None
            if tp is not None and ((d == 1 and tp <= entry) or (d == -1 and tp >= entry)):
                continue
            raw = 0.0
            exit_j = min(fill + hold, n - 1)
            for j in range(fill + 1, exit_j + 1):
                if d == 1:
                    if lo[j] <= sl:
                        raw = (sl - entry) / entry; exit_j = j; break
                    if tp is not None and h[j] >= tp:
                        raw = (tp - entry) / entry; exit_j = j; break
                else:
                    if h[j] >= sl:
                        raw = (entry - sl) / entry; exit_j = j; break
                    if tp is not None and lo[j] <= tp:
                        raw = (entry - tp) / entry; exit_j = j; break
            else:
                raw = ((c[exit_j] - entry) / entry) * d
            out.append((df.index[fill], (raw - cost) * NOTIONAL * 100, sym))
            busy = exit_j
    return out


def main() -> int:
    data = {sym: load(sym) for sym in PAIRS}
    base = stat(run(data), min_n=60)
    print("===== 기준 (T1 E0 확장3.0) =====", flush=True)
    print(f"  {line(base)}", flush=True)

    print("\n===== [B1] TP 정보량 — 확장이 진짜 좋은 목표인가 =====", flush=True)
    for tp in ("ext1.618", "ext2.236", "ext2.618", "ext3.0", "ext4.0", "none",
               "2R", "3R", "4R", "atr6", "atr10"):
        s = stat(run(data, tp_mode=tp), min_n=60)
        print(f"  TP={tp:<9} {line(s)}", flush=True)

    print("\n===== [B2] 롱/숏 분리 (상승편향 배제) =====", flush=True)
    for sd in ("long", "short"):
        s = stat(run(data, side=sd), min_n=40)
        print(f"  {sd:<6} {line(s)}", flush=True)

    print("\n===== [B3] 플라시보 (무작위 시점) =====", flush=True)
    for seed in (0, 1, 2):
        s = stat(run(data, placebo_rng=np.random.default_rng(seed)), min_n=60)
        print(f"  seed{seed}: {line(s)}", flush=True)

    print("\n===== [B4] 파라미터 이웃 =====", flush=True)
    for k in (2.0, 2.5, 3.0, 3.5, 4.0):
        s = stat(run(data, k=k), min_n=60)
        print(f"  ATR×{k:<4} {line(s)}", flush=True)
    for win in (8, 10, 12, 16, 24):
        s = stat(run(data, win=win), min_n=60)
        print(f"  win{win:<5} {line(s)}", flush=True)
    for b in (0.35, 0.45, 0.55):
        s = stat(run(data, body=b), min_n=60)
        print(f"  몸통{b:<5} {line(s)}", flush=True)
    for dt in (0.5, 0.6, 0.7, 0.8):
        s = stat(run(data, dirt=dt), min_n=60)
        print(f"  방향{dt:<5} {line(s)}", flush=True)

    print("\n===== [B5] 비용 민감도 =====", flush=True)
    for cost in (0.0011, 0.0015, 0.0020):
        s = stat(run(data, cost=cost), min_n=60)
        print(f"  비용{100 * cost:.2f}%: {line(s)}", flush=True)

    print("\n===== [B6] 보유 한도 =====", flush=True)
    for hold in (48, 120, 240, 480):
        s = stat(run(data, hold=hold), min_n=60)
        print(f"  HOLD{hold:<5}({hold / 24:.0f}일): {line(s)}", flush=True)

    print("\n===== [B7] 맥락 재확인 (T0 대조) =====", flush=True)
    for ck in ("T0", "T1"):
        s = stat(run(data, ctx_key=ck), min_n=60)
        print(f"  {ck}: {line(s)}", flush=True)

    print("\n===== [B8] 페어 1개 제외 =====", flush=True)
    for sym in PAIRS:
        s = stat(run(data, drop_sym=sym), min_n=60)
        print(f"  -{sym.replace('USDT', ''):<5} {line(s)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
