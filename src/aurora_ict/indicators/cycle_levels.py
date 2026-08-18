"""Cycle(순환매) 지지/저항 레벨 — 나뇨띠 자료 기반. 차트 표시용.

2026-08-18 파트너 요청: 순환매가 보는 지지/저항을 차트에서 눈으로 확인.

레벨 소스 3종
  ① 구름대 스팬 — 일목 선행스팬 A/B (9/26/52, offset = displacement-1)
  ② 매물대      — 가격대별 거래량 상위 구간 (시간대별이 아니라 **가격대별**)
  ③ 2468        — 1K 단위 안의 200/400(상방) · 600/800(하방). 방향 필터 적용
                   (2,4,6,8 타점매매 PDF: "추세가 상방이면 2·4, 하방이면 6·8")

터치 판정은 봉이 레벨을 스치고 그 방향으로 버텼는지로 본다.
지지 터치(아래에서 받침) / 저항 터치(위에서 눌림)를 구분해 반환한다.

담당: 연구 공용
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CONV, BASE, SPANB, DISP = 9, 26, 52, 26

# 같은 가격대 재터치를 다시 표시하기까지 기다리는 봉 수 (차트 표시 전용).
COOLDOWN = 20


def cloud_spans(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """차트에 그려져 있는 선행스팬 A/B (이미 시프트 적용된 값)."""
    h, lo = df["high"], df["low"]
    conv = (h.rolling(CONV).max() + lo.rolling(CONV).min()) / 2
    base = (h.rolling(BASE).max() + lo.rolling(BASE).min()) / 2
    a = ((conv + base) / 2).shift(DISP - 1)
    b = ((h.rolling(SPANB).max() + lo.rolling(SPANB).min()) / 2).shift(DISP - 1)
    return a.to_numpy(), b.to_numpy()


def volume_profile_levels(df: pd.DataFrame, lookback: int = 300,
                          bins: int = 40, top: int = 3) -> list[float]:
    """최근 lookback 봉의 가격대별 거래량 상위 top 구간 중심가."""
    if len(df) < 20:
        return []
    w = df.iloc[-min(lookback, len(df)):]
    px = w["close"].to_numpy()
    vol = w["volume"].to_numpy() if "volume" in w else np.ones(len(w))
    lo, hi = float(px.min()), float(px.max())
    if not np.isfinite(lo) or hi <= lo:
        return []
    idx = np.clip(((px - lo) / (hi - lo) * bins).astype(int), 0, bins - 1)
    agg = np.bincount(idx, weights=vol, minlength=bins)
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    return [float(centers[b]) for b in np.argsort(agg)[-top:]]


def levels_2468(price: float, direction: str, span: int = 2) -> list[float]:
    """1K 단위 안의 2·4(상방) / 6·8(하방). PDF 원문 기준."""
    k = int(price // 1000)
    offs = (200, 400) if direction == "up" else (600, 800)
    out: list[float] = []
    for d in range(-span, span + 1):
        b = (k + d) * 1000
        if b > 0:
            out.extend(float(b + o) for o in offs)
    return sorted(out)


def collect_levels(df: pd.DataFrame, i: int, direction: str) -> list[tuple[float, str]]:
    """i 시점에서 유효한 레벨 목록 — (가격, 출처)."""
    out: list[tuple[float, str]] = []
    a, b = cloud_spans(df)
    for arr, nm in ((a, "cloud_a"), (b, "cloud_b")):
        if i < len(arr) and np.isfinite(arr[i]):
            out.append((float(arr[i]), nm))
    for p in volume_profile_levels(df.iloc[: i + 1]):
        out.append((p, "profile"))
    px = float(df["close"].iloc[i])
    for p in levels_2468(px, direction):
        out.append((p, "2468"))
    return out


def touched(prev_close: float, high: float, low: float, close: float,
            level: float, tol: float, direction: str) -> bool:
    """레벨 터치 판정.

    지지(long): 위에서 내려와 레벨에 닿고 그 위에서 버팀
    저항(short): 아래서 올라가 레벨에 닿고 그 아래로 밀림
    """
    band = level * tol
    if direction == "long":
        return prev_close > level and low <= level + band and close >= level - band
    return prev_close < level and high >= level - band and close <= level + band


def find_touches(df: pd.DataFrame, tol: float = 0.002,
                 max_bars: int = 300) -> list[dict]:
    """최근 max_bars 구간에서 지지/저항 터치 지점을 찾아 반환.

    Returns:
        [{"idx": int, "price": float, "kind": "support"|"resistance",
          "source": str, "count": int}] — count 는 그 자리에 겹친 레벨 수.
    """
    n = len(df)
    if n < BASE + DISP + 5:
        return []
    a, b = cloud_spans(df)
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    start = max(BASE + DISP, n - max_bars)
    out: list[dict] = []
    prof: list[float] = []
    recent: list[tuple[int, float]] = []
    for i in range(start, n):
        px = c[i]
        if not np.isfinite(px):
            continue
        # 매물대는 20봉마다만 다시 잡는다. 봉마다 히스토그램을 돌리면 차트 한 번에
        # 수백 ms 가 날아가는데, 창이 300봉이라 20봉 이동으로는 값이 거의 안 변한다.
        if not prof or (i - start) % 20 == 0:
            prof = volume_profile_levels(df.iloc[: i + 1])
        for direction, kind in (("long", "support"), ("short", "resistance")):
            lvls = []
            for arr, nm in ((a, "cloud"), (b, "cloud")):
                if np.isfinite(arr[i]):
                    lvls.append((float(arr[i]), nm))
            for p in prof:
                lvls.append((p, "profile"))
            for p in levels_2468(px, "up" if direction == "long" else "down"):
                lvls.append((p, "2468"))
            hits = [(p, s) for p, s in lvls
                    if touched(c[i - 1], h[i], lo[i], px, p, tol, direction)]
            if not hits:
                continue
            best = min(hits, key=lambda x: abs(x[0] - px))
            # 겹침은 **출처 종류 수**로 센다(±0.4%). 레벨 개수로 세면 매물대가 한
            # 자리에 여럿 몰려 늘 2 이상이 나와 변별력이 없었다. 구름·매물대·2468
            # 중 몇 종이 같은 가격에 모였는가가 연구에서 본 '겹친 자리'의 뜻이다.
            cnt = len({
                src for p, src in lvls
                if abs(p - best[0]) / best[0] <= 0.004
            })
            # 쿨다운 — 같은 자리를 연속으로 스치면 봉마다 별이 찍혀 차트가 뭉갠다.
            # 최근 COOLDOWN 봉 안에 같은 가격대(±0.4%)를 이미 표시했으면 건너뛴다.
            if any(i - j <= COOLDOWN and abs(p - best[0]) / best[0] <= 0.004
                   for j, p in recent):
                break
            recent.append((i, best[0]))
            out.append({"idx": i, "price": best[0], "kind": kind,
                        "source": best[1], "count": cnt})
            break
    return out
