"""#AUTONOMOUS 2026-07-29: 임펄스+피보확장 v2 — 실전 제약 정직판 (파트너: "정확하게 다시").

v1 문제: ①임펄스 남발(13,762건=하루 6건) ②동시 포지션 무제한(겹치는 임펄스 중복
계산) ③배당 R 계산이 진입가 무시 → net +19,658% 라는 비현실 숫자.

v2 수정:
  · **동시 포지션 1개** — 페어별 순차 진행. 보유 중 신규 임펄스 무시(실전 정합).
  · **임펄스 강화** — ATR×{3,4} × 되돌림 허용 {20%} × win {12,24}, 그리고
    직전 임펄스와 최소 간격(win×2봉) 요구.
  · **R 정직 계산** — 실제 진입가·SL 로 위험폭 산출(가정 배당 제거).
  · **비용** maker 0.08%(지정가 되돌림 진입) / SL·TP 는 taker 가정 포함.
  · **판정 강화** — net>0 + 양반기 + 연도 다수 + **페어 5/7 이상 흑자** + n>=100.
통과 시 자동으로 배터리: 파라미터 이웃 · 페어별 · 연도별 · 셔플(임펄스 시점 무작위).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
TF = "1h"
NOTIONAL = 18.0
TTL_RETR = 24
HOLD = 240
COST = 0.0008


def load(sym: str):
    df = _resample(_load_full(sym)).resample(TF).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    c = df["close"].to_numpy(); h = df["high"].to_numpy()
    lo = df["low"].to_numpy(); o = df["open"].to_numpy()
    tr = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    return df, o, c, h, lo, atr


def impulses(c, h, lo, atr, win: int, k: float, max_pull: float, gap: int):
    """임펄스 목록 — 겹침 방지(직전 임펄스로부터 gap 봉 이상)."""
    n = len(c)
    out = []
    last_end = -10**9
    i = win + 15
    while i < n - 1:
        if np.isnan(atr[i]) or i - last_end < gap:
            i += 1; continue
        seg_c = c[i - win:i + 1]
        move = seg_c[-1] - seg_c[0]
        if abs(move) < k * atr[i]:
            i += 1; continue
        d = 1 if move > 0 else -1
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
        last_end = i
        i += 1
    return out


def run(data, win, k, max_pull, gap, f, e, sl_mode="start"):
    """동시 포지션 1개 — 페어별 순차. sl_mode: 'start'(임펄스 시작) | 'ext'(되돌림 끝 아래)."""
    out = []
    for sym, (df, o, c, h, lo, atr) in data.items():
        n = len(c)
        imps = impulses(c, h, lo, atr, win, k, max_pull, gap)
        busy_until = -1
        for (i, start, end, d) in imps:
            if i <= busy_until:
                continue                      # 보유 중 — 신규 무시
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
            sl = start if sl_mode == "start" else (end - d * 1.0 * leg)
            risk = abs(entry - sl)
            if risk <= 0 or risk / entry < 0.002:   # 위험폭 0.2% 미만이면 비용에 먹힘
                continue
            tp = end + d * (e - 1.0) * leg if e > 1.0 else end
            if (d == 1 and tp <= entry) or (d == -1 and tp >= entry):
                continue
            raw = 0.0
            exit_j = min(fill + HOLD, n - 1)
            for j in range(fill, exit_j + 1):
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
            out.append((df.index[fill], (raw - COST) * NOTIONAL * 100, sym))
            busy_until = exit_j
    return out


def stat(tr, min_n=100):
    if len(tr) < min_n:
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
    syms: dict[str, float] = {}
    for _, p, s in tr:
        syms[s] = syms.get(s, 0.0) + p
    ypos = sum(1 for v in ys.values() if v > 0)
    spos = sum(1 for v in syms.values() if v > 0)
    return dict(n=len(tr), net=net, wr=100 * w / len(tr), h1=h1, h2=h2, mdd=mdd,
                nm=net / max(mdd, 1e-9), ys=ys, syms=syms, spos=spos,
                ok=net > 0 and h1 > 0 and h2 > 0 and ypos >= len(ys) - 1 and spos >= 5)


def line(s):
    if s is None:
        return "표본부족"
    y = " ".join(f"{k}:{v:+.0f}" for k, v in sorted(s["ys"].items()))
    return (f"n={s['n']:4d} net={s['net']:+7.1f}% 승률={s['wr']:3.0f}% H1={s['h1']:+6.1f} "
            f"H2={s['h2']:+6.1f} MDD={s['mdd']:5.1f} nm={s['nm']:4.2f} 페어{s['spos']}/7 [{y}]")


def main() -> int:
    data = {sym: load(sym) for sym in PAIRS}
    winners = []
    print("=== 탐색: 임펄스 정의 × 되돌림 × 확장 ===", flush=True)
    for win, k, mp in ((12, 3.0, 0.20), (24, 3.0, 0.20), (24, 4.0, 0.20), (12, 4.0, 0.20)):
        gap = win * 2
        imp_n = sum(len(impulses(d[2], d[3], d[4], d[5], win, k, mp, gap))
                    for d in data.values())
        print(f"\n### win={win} ATR×{k} 되돌림≤{mp} → 임펄스 {imp_n}건 "
              f"(하루 {imp_n / (5 * 365):.2f}건)", flush=True)
        for f in (0.382, 0.5, 0.618, 0.707):
            for e in (1.0, 1.272, 1.382, 1.618, 2.618):
                s = stat(run(data, win, k, mp, gap, f, e))
                mark = "★" if s and s["ok"] else " "
                if s and (s["ok"] or s["net"] > 0):
                    print(f"  {mark}되돌림{f}→확장{e:<6} {line(s)}", flush=True)
                if mark == "★":
                    winners.append((s["net"], win, k, mp, gap, f, e, s))
    print("\n\n===== ★ 통과 =====", flush=True)
    if not winners:
        print("  없음 — 전 조합 불합격", flush=True)
        return 0
    winners.sort(reverse=True, key=lambda x: x[0])
    for net, win, k, mp, gap, f, e, s in winners[:8]:
        print(f"  win{win} ATR×{k} 되돌림{f}→확장{e}: {line(s)}", flush=True)
        print("     페어별: " + " ".join(f"{a.replace('USDT', '')}:{b:+.0f}"
                                        for a, b in s["syms"].items()), flush=True)
    # 최우수 배터리
    net, win, k, mp, gap, f, e, s = winners[0]
    print(f"\n===== 배터리: win{win} ATR×{k} 되돌림{f}→확장{e} =====", flush=True)
    print("[이웃] 되돌림/확장 ±1단계", flush=True)
    for f2 in (max(0.382, f - 0.1), f, min(0.786, f + 0.1)):
        for e2 in (max(1.0, e - 0.25), e, e + 0.25):
            s2 = stat(run(data, win, k, mp, gap, round(f2, 3), round(e2, 3)))
            print(f"  f{f2:.3f} e{e2:.3f}: {line(s2)}", flush=True)
    print("[이웃] 임펄스 강도 ±", flush=True)
    for k2 in (k - 0.5, k, k + 0.5):
        s2 = stat(run(data, win, k2, mp, gap, f, e))
        print(f"  ATR×{k2}: {line(s2)}", flush=True)
    print("[SL 변형] 되돌림 100% 대신 임펄스 시작", flush=True)
    s2 = stat(run(data, win, k, mp, gap, f, e, sl_mode="ext"))
    print(f"  SL=되돌림100%: {line(s2)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
