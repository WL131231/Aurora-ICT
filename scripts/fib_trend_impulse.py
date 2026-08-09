"""#AUTONOMOUS 2026-07-29: 추세맥락 + 캔들형태 임펄스 → 확장 TP (파트너 방식 정본).

파트너 정정(중요): "차트가 **하락 곡선 상태**에서 하락 임펄스가 나오면, 그 임펄스에
확장을 대입하고 확장값을 **목표 TP** 로 잡는다."
→ 앞선 v2/v3 에 없던 결정적 조건 = **상위 추세와 임펄스 방향 일치(맥락 게이트)**.
   v2/v3 은 아무 임펄스나 다 잡았다. 이번엔 추세 순행 임펄스만.
파트너 정정2: 엘리엇 파동 카운팅은 사람마다 달라 백테 불가 → **캔들 형태로만** 임펄스 판별.

임펄스 = 캔들 형태 (파동 카운팅 없음):
  · 크기      : win 봉 순이동 >= ATR14 × k
  · 몸통비율  : 구간 평균 |종가-시가| / (고-저) >= body_thr  (도지 배제)
  · 방향일관성: 구간 봉 중 같은 방향 비율 >= dir_thr
  · 되돌림    : 구간 내 역행 <= max_pull
추세 맥락(3종 비교):
  T0 없음        : 게이트 미적용 (v2/v3 재현 — 대조군)
  T1 EMA200      : 종가가 EMA200 대비 임펄스 방향과 같은 편
  T2 EMA50/200   : 정배열/역배열이 임펄스 방향과 일치
  T3 스윙구조    : 최근 고/저 갱신 패턴(LH·LL = 하락 / HH·HL = 상승)이 일치
진입 2종: E0 임펄스 확정 다음봉 시가(추종) / E618 되돌림 0.618 지정가(TTL 24)
TP: 확장 1.618 / 2.236 / 2.618 / 3.0 (임펄스 시작 기준 leg 배수 = 차트의 -0.618/-1.618 등)
SL: 임펄스 시작점(파동 무효)
같은 봉 진입+청산 배제 기본 적용(7/29 교훈).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from impulse_fib_v2 import PAIRS, line, load, stat  # noqa: E402

TTL_RETR = 24
HOLD = 240
COST = 0.0008
NOTIONAL = 18.0
EXTS = (1.618, 2.236, 2.618, 3.0)


def context(df, c, h, lo):
    """추세 맥락 3종 — 각 봉에서 +1(상승) / -1(하락) / 0(불명). 전부 직전 완결봉 기준."""
    s = pd.Series(c)
    ema50 = s.ewm(span=50, adjust=False).mean().shift(1).to_numpy()
    ema200 = s.ewm(span=200, adjust=False).mean().shift(1).to_numpy()
    cp = np.concatenate([[np.nan], c[:-1]])
    t1 = np.where(cp > ema200, 1, np.where(cp < ema200, -1, 0))
    t2 = np.where((ema50 > ema200) & (cp > ema50), 1,
                  np.where((ema50 < ema200) & (cp < ema50), -1, 0))
    # 스윙 구조 — 최근 24봉 고/저가 그 이전 24봉 대비 갱신 방향
    hh = pd.Series(h).rolling(24).max()
    ll = pd.Series(lo).rolling(24).min()
    t3 = np.where((hh.shift(1) > hh.shift(25)) & (ll.shift(1) > ll.shift(25)), 1,
                  np.where((hh.shift(1) < hh.shift(25)) & (ll.shift(1) < ll.shift(25)), -1, 0))
    return {"T0": np.zeros(len(c)), "T1": t1, "T2": t2, "T3": t3}


def candle_impulses(o, c, h, lo, atr, win, k, body_thr, dir_thr, max_pull, gap):
    """캔들 형태 기반 임펄스 — 파동 카운팅 없음."""
    n = len(c)
    out = []
    last = -10**9
    rng_ = np.maximum(h - lo, 1e-12)
    body = np.abs(c - o) / rng_
    updn = np.sign(c - o)
    i = win + 205
    while i < n - 1:
        if np.isnan(atr[i]) or i - last < gap:
            i += 1; continue
        seg_c = c[i - win:i + 1]
        move = seg_c[-1] - seg_c[0]
        if abs(move) < k * atr[i]:
            i += 1; continue
        d = 1 if move > 0 else -1
        if body[i - win + 1:i + 1].mean() < body_thr:
            i += 1; continue
        if (updn[i - win + 1:i + 1] == d).mean() < dir_thr:
            i += 1; continue
        seg_h = h[i - win:i + 1]; seg_l = lo[i - win:i + 1]
        if d == 1:
            start, end = seg_l.min(), seg_h.max()
            pull = (seg_h.max() - seg_c[-1]) / max(end - start, 1e-12)
        else:
            start, end = seg_h.max(), seg_l.min()
            pull = (seg_c[-1] - seg_l.min()) / max(abs(end - start), 1e-12)
        if pull > max_pull:
            i += 1; continue
        out.append((i, start, end, d))
        last = i
        i += 1
    return out


def run(data, ctx_key, entry_mode, ext, win=12, k=3.0, body_thr=0.45,
        dir_thr=0.6, max_pull=0.25, gap=24):
    out = []
    for sym, (df, o, c, h, lo, atr) in data.items():
        n = len(c)
        ctx = context(df, c, h, lo)[ctx_key]
        busy = -1
        for (i, start, end, d) in candle_impulses(o, c, h, lo, atr, win, k,
                                                  body_thr, dir_thr, max_pull, gap):
            if i <= busy:
                continue
            # 맥락 게이트 — 추세와 임펄스 방향 일치할 때만 (T0 은 전부 통과)
            if ctx_key != "T0" and ctx[i] != d:
                continue
            leg = abs(end - start)
            if leg <= 0:
                continue
            if entry_mode == "E0":
                fill = i + 1
                if fill >= n:
                    continue
                entry = o[fill]
                cost = 0.0011
            else:
                entry = end - d * 0.618 * leg
                fill = None
                for j in range(i + 1, min(i + 1 + TTL_RETR, n)):
                    if lo[j] <= entry <= h[j]:
                        fill = j; break
                if fill is None:
                    continue
                cost = COST
            sl = start
            risk = abs(entry - sl)
            if risk <= 0 or risk / entry < 0.002:
                continue
            tp = start + d * ext * leg          # 확장 = 임펄스 시작 기준 leg 배수
            if (d == 1 and tp <= entry) or (d == -1 and tp >= entry):
                continue
            raw = 0.0
            exit_j = min(fill + HOLD, n - 1)
            for j in range(fill + 1, exit_j + 1):      # 같은 봉 청산 배제
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
            out.append((df.index[fill], (raw - cost) * NOTIONAL * 100, sym))
            busy = exit_j
    return out


def main() -> int:
    data = {sym: load(sym) for sym in PAIRS}
    print("=== 추세맥락 × 진입 × 확장TP (캔들형태 임펄스) ===", flush=True)
    print("맥락: T0 없음(대조) T1 EMA200 T2 EMA50/200 T3 스윙구조\n", flush=True)
    winners = []
    for ctx_key in ("T0", "T1", "T2", "T3"):
        print(f"########## 맥락 {ctx_key} ##########", flush=True)
        for em in ("E0", "E618"):
            for ext in EXTS:
                s = stat(run(data, ctx_key, em, ext), min_n=60)
                mark = "★" if s and s["ok"] else " "
                nm = f"{em:<5} 확장{ext:<6}"
                print(f"  {mark}{nm} {line(s)}", flush=True)
                if mark == "★":
                    winners.append((s["net"], ctx_key, em, ext, s))
        print("", flush=True)
    print("===== ★ 통과 =====", flush=True)
    if not winners:
        print("  없음", flush=True)
    for net, ck, em, ext, s in sorted(winners, reverse=True, key=lambda x: x[0])[:8]:
        print(f"  {ck} {em} 확장{ext}: {line(s)}", flush=True)
        print("     페어별: " + " ".join(f"{a.replace('USDT', '')}:{b:+.0f}"
                                        for a, b in s["syms"].items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
