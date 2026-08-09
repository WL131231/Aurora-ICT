"""#AUTONOMOUS 2026-07-27: 2단계 — 레인지 엣지 스윕 페이드(터틀수프형) 백테.

1단계 결론: Origo 는 횡보서 오히려 수익(게이트 추가 역효과) → 2단계는 수비가 아닌
**독립 공격 모델** 검증으로 목적 전환. 박스(딜링레인지) 상/하단의 가짜돌파(스윕 후
복귀)를 반대로 먹는 ICT 터틀수프. 기각된 BB평균회귀와 달리 "정의된 박스 + 스윕
트리거 + 종가 복귀 확인"의 강조건.

로직(숏 기준, 롱 미러):
  - 박스: 직전 288×5m(24h) Donchian 고/저 (진입봉 제외 — 인과).
  - 트리거: high 가 박스고점 돌파했는데 종가는 박스 안으로 복귀(가짜돌파).
  - 진입: 확인봉 종가(시장가). SL: 스윕 극단 + 버퍼(0.05%). TP: 박스 중앙 / 2R.
  - 박스폭 필터: 수수료 대비 의미 있게 (0.5% / 1.0%).
  - 국면 게이트 비교: 없음 vs ADX(1h)<25 vs CHOP(1h)>55 — 페이드가 어느 국면에
    사는지 판정(1단계 지표 재사용).
비용: 왕복 0.10%(시장가 진입 현실) 기본, 승자만 0.06% 민감도.
판정: 연도별 일관 + H1/H2 robust + 국면 게이트 유무 효과.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402
from chop_gate_bakeoff import adx14, chop14  # noqa: E402 — 1단계 지표 재사용

RTCOST = 0.0010
BOX_N = 288          # 24h 딜링레인지
SL_BUF = 0.0005      # 스윕 극단 위 0.05% 버퍼
HOLD_MAX = 288       # 최대 보유 24h


def run(sym: str) -> list[dict]:
    df = _resample(_load_full(sym))
    c = df["close"].to_numpy(); h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    n = len(c)
    dh = pd.Series(h).rolling(BOX_N).max().shift(1).to_numpy()
    dl = pd.Series(lo).rolling(BOX_N).min().shift(1).to_numpy()
    # 1h 국면 지표 → 직전 완결봉 5m 매핑(인과).
    d1h = df.resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
    ind = pd.DataFrame({
        "adx": adx14(*(d1h[k].to_numpy() for k in ("high", "low", "close"))),
        "chop": chop14(*(d1h[k].to_numpy() for k in ("high", "low", "close"))),
    }, index=d1h.index).shift(1).reindex(df.index, method="ffill")
    adx = ind["adx"].to_numpy(); chop = ind["chop"].to_numpy()

    trades = []
    i = BOX_N + 1
    while i < n - 1:
        boxw = (dh[i] - dl[i]) / max(c[i], 1e-12)
        sig = 0
        if h[i] > dh[i] and c[i] < dh[i]:
            sig = -1  # 상단 가짜돌파 → 숏 페이드
        elif lo[i] < dl[i] and c[i] > dl[i]:
            sig = 1   # 하단 가짜돌파 → 롱 페이드
        if sig == 0 or np.isnan(boxw):
            i += 1
            continue
        entry = c[i]
        mid = (dh[i] + dl[i]) / 2
        if sig == -1:
            sl = h[i] * (1 + SL_BUF)
            risk = sl - entry
            tps = {"mid": mid, "2R": entry - 2 * risk}
        else:
            sl = lo[i] * (1 - SL_BUF)
            risk = entry - sl
            tps = {"mid": mid, "2R": entry + 2 * risk}
        if risk <= 0:
            i += 1
            continue
        rec = dict(ts=df.index[i], sym=sym, boxw=boxw * 100,
                   adx=adx[i], chop=chop[i], dir=sig,
                   rr_mid=abs(tps["mid"] - entry) / risk)
        # 두 TP 변형 각각 시뮬.
        for tag, tp in tps.items():
            netv = 0.0
            for j in range(i + 1, min(i + 1 + HOLD_MAX, n)):
                if sig == -1:
                    if h[j] >= sl:
                        netv = (entry - sl) / entry
                        break
                    if lo[j] <= tp:
                        netv = (entry - tp) / entry
                        break
                else:
                    if lo[j] <= sl:
                        netv = (sl - entry) / entry
                        break
                    if h[j] >= tp:
                        netv = (tp - entry) / entry
                        break
            rec[f"net_{tag}"] = (netv - RTCOST) * 100
        trades.append(rec)
        i += 3  # 같은 스윕 연타 방지 — 15분 쿨다운
    return trades


def stat(g: list[dict], key: str) -> str:
    n = len(g)
    if not n:
        return "n=   0"
    net = sum(x[key] for x in g)
    w = sum(1 for x in g if x[key] > 0)
    return f"n={n:5d} net={net:+8.1f} 승률={100 * w / n:3.0f}% avg={net / n:+.4f}"


def yearly(g: list[dict], key: str) -> str:
    ys: dict[int, float] = {}
    for x in g:
        ys[x["ts"].year] = ys.get(x["ts"].year, 0.0) + x[key]
    return " ".join(f"{y}:{v:+.0f}" for y, v in sorted(ys.items()))


def main() -> int:
    pairs = sys.argv[1:] or ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT"]
    allt: list[dict] = []
    for sym in pairs:
        rows = run(sym)
        allt += rows
        print(f"{sym}: 후보 {len(rows)}", flush=True)
    variants = {
        "전체(무게이트)": lambda x: True,
        "박스폭>0.5%": lambda x: x["boxw"] > 0.5,
        "박스폭>1.0%": lambda x: x["boxw"] > 1.0,
        "ADX<25(횡보만)": lambda x: x["adx"] < 25,
        "CHOP>55(횡보만)": lambda x: x["chop"] > 55,
        "박스>1% & ADX<25": lambda x: x["boxw"] > 1.0 and x["adx"] < 25,
        "박스>1% & mid>=1R": lambda x: x["boxw"] > 1.0 and x["rr_mid"] >= 1.0,
    }
    for key, label in (("net_mid", "TP=박스중앙"), ("net_2R", "TP=2R")):
        print(f"\n===== {label} =====", flush=True)
        for name, fn in variants.items():
            g = [x for x in allt if fn(x)]
            print(f"{name:<18} {stat(g, key)}  [{yearly(g, key)}]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
