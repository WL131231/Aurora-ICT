"""#AUTONOMOUS 2026-07-29: 임펄스 + 피보 확장 목표가 연구 (파트너 지시 — "되면 죽이는 것").

파트너 차트(SK하이닉스·SOL·BTC·ETH): 큰 임펄스(급격한 한 방향 다리)를 잡고 피보
**확장** 레벨(1.382 / 1.618 / 2.618 / -0.618 / -1 등)을 목표가로 미리 긋는 방식.
우리 언어로는 "임펄스 이후 되돌림 진입 → 확장 레벨을 TP" = TP 위치 문제의 정면 연구.

[1단계] 확장 레벨의 **도달률·선착순** 통계 (매매 아님, 순수 구조 검정)
  임펄스 정의: N봉(12/24) 안에 ATR14 대비 k배(2.0/3.0) 이상 순방향 이동 + 방향 일관
  (되돌림 30% 미만). 임펄스 끝(극단)에서:
    · 되돌림 레벨 0.382/0.5/0.618/0.707/0.786 중 어디까지 되돌리는가(분포)
    · 되돌림 후 확장 레벨 1.0(임펄스 재돌파)/1.272/1.382/1.618/2.618 도달률
    · **SL(되돌림 100% 붕괴) vs 각 확장 레벨 — 어느 쪽이 먼저 오는가**(선착순)
  → 각 확장 레벨의 "도달 확률 × 배당"으로 기대값 계산. 기대값>0 레벨만 2단계.
[2단계] 매매화: 되돌림 f 진입(지정가·maker) → 확장 e TP → SL=임펄스 시작점.
  f ∈ {0.382,0.5,0.618,0.707,0.786} × e ∈ {1.0,1.272,1.382,1.618,2.618}
  비용 0.08%(maker) / 0.11%(taker). TF 1h(임펄스는 큰 구조라 5m 노이즈 회피).
판정: net>0 + 양반기 + 연도 다수 + 페어 분산 → ★.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
TF = "1h"
FIBS_RETR = (0.382, 0.5, 0.618, 0.707, 0.786)
FIBS_EXT = (1.0, 1.272, 1.382, 1.618, 2.618)
NOTIONAL = 18.0
TTL_RETR = 24      # 되돌림 대기 최대 봉
HOLD = 240         # 진입 후 최대 보유


def load(sym: str):
    df = _resample(_load_full(sym)).resample(TF).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    c = df["close"].to_numpy(); h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    o = df["open"].to_numpy()
    tr = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    return df, o, c, h, lo, atr


def find_impulses(o, c, h, lo, atr, win: int, k: float):
    """임펄스 = win봉 내 ATR×k 이상 순이동 + 되돌림 30% 미만. 반환: (끝idx, 시작가, 끝가, 방향)."""
    n = len(c)
    out = []
    i = win + 15
    while i < n - 1:
        if np.isnan(atr[i]):
            i += 1; continue
        seg_c = c[i - win:i + 1]
        seg_h = h[i - win:i + 1]
        seg_l = lo[i - win:i + 1]
        move = seg_c[-1] - seg_c[0]
        if abs(move) < k * atr[i]:
            i += 1; continue
        d = 1 if move > 0 else -1
        # 되돌림 체크 — 임펄스 도중 역행 폭이 전체의 30% 미만
        if d == 1:
            start, end = seg_l.min(), seg_h.max()
            worst = (seg_c[-1] - seg_h.max()) / max(end - start, 1e-12)
        else:
            start, end = seg_h.max(), seg_l.min()
            worst = (seg_l.min() - seg_c[-1]) / max(abs(end - start), 1e-12)
        if abs(worst) > 0.30:
            i += 1; continue
        out.append((i, start, end, d))
        i += win  # 중복 방지 — 임펄스 길이만큼 건너뜀
    return out


def stage1(data, win: int, k: float) -> None:
    """확장 레벨 도달률·선착순 — 매매 없이 구조만."""
    stats = {e: [0, 0] for e in FIBS_EXT}     # [도달, 전체]
    retr_hist = {f: 0 for f in FIBS_RETR}
    tot = 0
    for sym, (df, o, c, h, lo, atr) in data.items():
        n = len(c)
        for (i, start, end, d) in find_impulses(o, c, h, lo, atr, win, k):
            leg = abs(end - start)
            if leg <= 0:
                continue
            tot += 1
            # 되돌림 깊이 — 이후 TTL 봉 내 최대 되돌림
            j_end = min(i + TTL_RETR, n - 1)
            if d == 1:
                deepest = (end - lo[i + 1:j_end + 1].min()) / leg if j_end > i else 0
            else:
                deepest = (h[i + 1:j_end + 1].max() - end) / leg if j_end > i else 0
            for f in FIBS_RETR:
                if deepest >= f:
                    retr_hist[f] += 1
            # 확장 도달 — 되돌림 100%(임펄스 시작 붕괴) 전에 도달했는가
            invalid = start
            for e in FIBS_EXT:
                target = end + d * (e - 1.0) * leg if e > 1.0 else end
                hit = False
                for j in range(i + 1, min(i + 1 + HOLD, n)):
                    if d == 1:
                        if lo[j] <= invalid:
                            break
                        if h[j] >= target:
                            hit = True; break
                    else:
                        if h[j] >= invalid:
                            break
                        if lo[j] <= target:
                            hit = True; break
                stats[e][1] += 1
                if hit:
                    stats[e][0] += 1
    print(f"\n[1단계] 임펄스 {tot}건 (win={win}, ATR×{k})", flush=True)
    print("  되돌림 도달률: " + " ".join(
        f"{f}={100 * retr_hist[f] / max(tot, 1):.0f}%" for f in FIBS_RETR), flush=True)
    print(f"  {'확장레벨':<8} {'도달률':>7} {'배당(R)':>8} {'기대값':>8}", flush=True)
    for e in FIBS_EXT:
        hit, all_ = stats[e]
        p = hit / max(all_, 1)
        # 배당 근사: 0.618 되돌림 진입 가정 시 R = (e-0.618)/0.618... 단순화해
        # 진입=0.618 되돌림, SL=100% 되돌림 → 위험 0.382leg, 보상 (e-0.618)leg
        rr = (e - 0.618) / 0.382 if e > 0.618 else 0
        ev = p * rr - (1 - p) * 1.0
        print(f"  {e:<8} {100 * p:6.1f}% {rr:7.2f}R {ev:+7.2f}R", flush=True)


def stage2(data, win: int, k: float, f: float, e: float, cost: float):
    out = []
    for sym, (df, o, c, h, lo, atr) in data.items():
        n = len(c)
        for (i, start, end, d) in find_impulses(o, c, h, lo, atr, win, k):
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
            if risk <= 0:
                continue
            tp = end + d * (e - 1.0) * leg if e > 1.0 else end
            if (d == 1 and tp <= entry) or (d == -1 and tp >= entry):
                continue
            raw = 0.0
            for j in range(fill, min(fill + HOLD, n)):
                if d == 1:
                    if lo[j] <= sl:
                        raw = (sl - entry) / entry; break
                    if h[j] >= tp:
                        raw = (tp - entry) / entry; break
                else:
                    if h[j] >= sl:
                        raw = (entry - sl) / entry; break
                    if lo[j] <= tp:
                        raw = (entry - tp) / entry; break
            else:
                raw = ((c[min(fill + HOLD, n - 1)] - entry) / entry) * d
            out.append((df.index[fill], (raw - cost) * NOTIONAL * 100, sym))
    return out


def stat(tr):
    if len(tr) < 30:
        return None
    tr = sorted(tr)
    nets = [p for _, p, _ in tr]
    net = sum(nets)
    w = sum(1 for p in nets if p > 0)
    half = len(tr) // 2
    h1 = sum(p for _, p, _ in tr[:half]); h2 = sum(p for _, p, _ in tr[half:])
    eq = pk = mdd = 0.0
    for _, p, _ in tr:
        eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    ys: dict[int, float] = {}
    for t, p, _ in tr:
        ys[t.year] = ys.get(t.year, 0.0) + p
    ypos = sum(1 for x in ys.values() if x > 0)
    syms = {}
    for _, p, s in tr:
        syms[s] = syms.get(s, 0.0) + p
    spos = sum(1 for v in syms.values() if v > 0)
    return dict(n=len(tr), net=net, wr=100 * w / len(tr), h1=h1, h2=h2, mdd=mdd,
                nm=net / max(mdd, 1e-9), ys=ys, syms=syms,
                ok=net > 0 and h1 > 0 and h2 > 0 and ypos >= len(ys) - 1 and spos >= 4)


def line(s):
    if s is None:
        return "표본부족"
    y = " ".join(f"{k}:{v:+.0f}" for k, v in sorted(s["ys"].items()))
    return (f"n={s['n']:4d} net={s['net']:+8.1f}% 승률={s['wr']:3.0f}% H1={s['h1']:+7.1f} "
            f"H2={s['h2']:+7.1f} MDD={s['mdd']:6.1f} net/MDD={s['nm']:5.2f} [{y}]")


def main() -> int:
    data = {sym: load(sym) for sym in PAIRS}
    for win, k in ((12, 2.0), (24, 3.0)):
        stage1(data, win, k)
    print("\n\n[2단계] 매매화 — 되돌림 진입 × 확장 TP", flush=True)
    winners = []
    for win, k in ((12, 2.0), (24, 3.0)):
        print(f"\n########## 임펄스 win={win} ATR×{k} ##########", flush=True)
        for f in FIBS_RETR:
            for e in FIBS_EXT:
                if e <= f:
                    continue
                s = stat(stage2(data, win, k, f, e, 0.0008))
                mark = "★" if s and s["ok"] else " "
                print(f"  {mark}되돌림{f} → 확장{e:<6} {line(s)}", flush=True)
                if mark == "★":
                    winners.append((s["net"], win, k, f, e, s))
    print("\n\n===== ★ 통과 요약 =====", flush=True)
    if not winners:
        print("  없음", flush=True)
    for net, win, k, f, e, s in sorted(winners, reverse=True, key=lambda x: x[0])[:10]:
        print(f"  win{win} ATR×{k} 되돌림{f}→확장{e}: {line(s)}", flush=True)
        print(f"     페어별: " + " ".join(f"{a.replace('USDT','')}:{b:+.0f}"
                                          for a, b in s["syms"].items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
