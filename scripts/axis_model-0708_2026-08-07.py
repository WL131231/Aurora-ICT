"""#AUTONOMOUS 2026-08-07: 축 연구 — ICT 2024 Mentorship **07AM/08AM 진입 모델**.

## 무엇을 왜 재는가

Origo 는 진입 모델이 사실상 Silver Bullet(+8/6 배선된 MMBM) 뿐이다. 정통 ICT 에는
별도 모델이 여럿 있고, 그 중 07:00/08:00/09:00 AM(뉴욕 현지) 멘토십 모델은
2026-07-20 검증에서 "진짜 엣지·빈도 30~40배" 로 기록됐다. 그런데 그 기록을 열어보니
**두 가지 문제**가 있다.

  ① 그때의 "07/08AM" 은 독립 모델이 아니라 **MMBM 반전 로직을 NY 오전에 한정한 것**
     이었다(session_gate_origo.py 2026-07-29 확인). 그리고 MMBM 근사판의 그 성적은
     7/20 밤 완전구현·현실비용 재실험에서 **비용 착시**로 정정됐고, 7/27 에 모체가
     기각됐다. 즉 "07/08AM 엣지" 는 사멸한 모체에 얹힌 숫자다.
  ② 정통 07/08AM 모델의 실제 시퀀스는 MMBM 과 다르다 — **REH/REL(상대적 등가 고·저)
     스윕 → 레인지 복귀 + MSS → 스윕 직전 FVG 가 뒤집힌 IFVG(또는 Breaker) 되돌림
     진입 → 이전 세션 저/고 또는 다음 REL/REH 로 TP**. 이건 한 번도 구현된 적이 없다.

따라서 이 스크립트는 7/20 결과를 재현하는 게 아니라, **정통 시퀀스를 처음 구현해서
현행 기준(BTC+ETH · 7x · size 0.9 · 동시보유 · DD 스로틀 · 복리)으로 재측정**한다.

## 1단계 — 스윕 전에 구조적 상한부터 (직전 피보 연구의 교훈)

피보 연구는 손잡이의 **이론 상한**(건당 0.027R)이 관측 노이즈보다 작다는 걸 사후에야
알았다. 그래서 여기서는 백테를 돌리기 전에 다음을 먼저 산출한다.

  · A. 시간 예산 — 07/08/09AM(NY) 창은 5분봉 전체의 몇 %인가.
  · B. 기회 상한 — 그 창 안에서 REH/REL 스윕이 물리적으로 몇 번 일어나는가.
       (필터를 전부 끈 상한. 실제 진입은 이보다 적을 수밖에 없다)
  · C. 노이즈 바닥 — 상한 N 건에서 건당 R 의 표준오차 = sd/√N. 검출하고 싶은 효과
       (건당 +0.15R 정도)가 이 바닥보다 작으면 스윕은 무의미하다.
  · D. SL 지렛대 — 이 모델의 "정밀 진입"(IFVG CE)이 R 을 실제로 움직이는가.
       라이브 SL 은 1.5×ATR 바닥이 깔린다. 스윕 고점까지의 구조 거리가 그 바닥보다
       늘 작다면 진입가를 아무리 정교하게 잡아도 R 은 안 변한다(= 피보와 같은 함정).

## 2~3단계 — 백테와 판정

2단계: 정통 시퀀스 백테. 청산은 Origo 라이브와 같은 유동성TP + 트레일 2R/1.5R + BE 1R.
3단계: 판정 — 순열검정(무작위 20000회) · 플라시보(같은 시각 무작위 방향) ·
       국면×방향 기저 · 연도 일관성 · 롱숏 분리 · 증분(SB+MMBM 과 시각 중복 제거).

## 하니스 규약
Silver Bullet 기준선은 반드시 `run_live_parity()` 로만 잡는다(2026-07-30 정합 규칙).
MMBM 은 `mmbm_full.backtest(detail=True)` — 8/6 재검증이 쓴 것과 같은 경로.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mmbm_full as M  # noqa: E402
from bt_par import _load_full, _resample  # noqa: E402
from live_parity import LIVE_BASE, run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402
from aurora_ict.indicators.fvg import FVGType, detect_fvgs  # noqa: E402
from aurora_ict.indicators.swing_points import SwingType, detect_swing_points  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 현행 라이브 상수 (2026-08-07)
# ─────────────────────────────────────────────────────────────────────────────
SYMS = ["BTCUSDT", "ETHUSDT"]
SIZE = LIVE_BASE["size_pct"]          # 0.9
LEV = 7.0
DD_PCT, DD_FACTOR = 0.25, 0.7         # DD 스로틀
RUIN = 0.20                           # 시드 20% 이하 = 파산
ATR_SL_MULT = 1.5                     # silver_bullet._ATR_SL_MULT — 라이브 SL 바닥
TRAIL_TRIG, TRAIL_DIST, BE_AT = 2.0, 1.5, 1.0   # 라이브 청산 규약

RNG = np.random.default_rng(20260807)
N_BOOT = 1000
N_PERM = 20000
DEDUP_MS = 60 * 60 * 1000             # 같은 심볼·방향·1시간 내면 같은 기회로 본다

# 모델 기본 파라미터 (정통 시퀀스 근사)
AM_HOURS = (7, 8, 9)                  # 뉴욕 현지 07/08/09시
DEF = dict(
    hours=AM_HOURS,
    skip_first_min=0,     # 정통 노트: 각 시각 첫 30분은 역방향 → 진입 보류 옵션
    swing_lr=2,           # 스윙 pivot 좌우 봉 수 (확정 지연 = right)
    reh_tol=0.0015,       # REH/REL 등가 판정 허용오차 (0.15%)
    reh_lb=72,            # REH 형성이 스윕보다 이 봉 안이어야 (6시간)
    mss_ttl=12,           # 스윕 후 MSS 대기 (1시간)
    ifvg_lb=24,           # 스윕 직전 FVG 탐색 범위 (2시간)
    entry_ttl=12,         # MSS 후 되돌림 체결 대기 (1시간)
    min_rr=2.0,           # 라이브 강제
    atr_floor=True,       # 라이브 SL 바닥 적용 (1.5×ATR14, silver_bullet #LIVE-7)
    sl_mult=1.0,          # #ORIGO-1.3 sl_dist_mult — 라이브 SB 는 4.0 강제
    require_htf=False,    # 1h 20봉 바이어스 정합 요구
)


# ─────────────────────────────────────────────────────────────────────────────
# 데이터·지표 준비 (심볼당 1회)
# ─────────────────────────────────────────────────────────────────────────────
_PREP: dict[str, dict] = {}
_SWEEP_CACHE: dict[tuple, list] = {}


def prep(sym: str) -> dict:
    """5분봉 + 뉴욕 현지시각 + ATR + 스윙 + FVG 를 한 번만 만들어 둔다."""
    if sym in _PREP:
        return _PREP[sym]
    df = _resample(_load_full(sym))
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    o = df["open"].to_numpy()
    n = len(df)
    ts_ms = (df.index.astype("int64") // 10**6).to_numpy()

    # 뉴욕 현지시각 (서머타임 자동 반영) — 정통 모델은 NY local 기준이다
    ny = df.index.tz_convert("America/New_York")
    ny_hour = ny.hour.to_numpy()
    ny_min = ny.minute.to_numpy()

    # ATR(14) on 5m — 라이브 SL 바닥 계산용
    prev_c = np.concatenate(([c[0]], c[:-1]))
    tr = np.maximum(h - lo, np.maximum(np.abs(h - prev_c), np.abs(lo - prev_c)))
    atr = pd.Series(tr).rolling(14).mean().to_numpy()

    # 1h 20봉 바이어스 (MMBM 과 동일 정의) — 옵션 필터용
    c1h = df["close"].resample("1h").last().ffill()
    bias1h = (np.sign(c1h - c1h.shift(20)).reindex(df.index, method="ffill")
              .fillna(0).to_numpy())

    # 국면(추세 세기) — 1d ER 대신 라이브 entry_trend_pct 근사: 1h EMA20 대비 %변화
    trend = pd.Series(c).pct_change(12).rolling(12).mean().to_numpy() * 100.0

    swings = detect_swing_points(df, left=DEF["swing_lr"], right=DEF["swing_lr"])
    fvgs = detect_fvgs(df, min_size_pct=0.0005)

    P = dict(df=df, h=h, lo=lo, c=c, o=o, n=n, ts=ts_ms, atr=atr,
             ny_hour=ny_hour, ny_min=ny_min, bias1h=bias1h, trend=trend,
             swings=swings, fvgs=fvgs)
    _PREP[sym] = P
    return P


def _sweep_events(P: dict, p: dict):
    """REH/REL 스윕 이벤트를 **인과적으로** 검출한다.

    look-ahead 방지: 스윙은 idx+right 봉에서야 확정된 것으로 취급한다.
    REH = 직전 확정 스윙고 2개가 reh_tol 안에 있을 때 그 최대값.
    스윕 = 그 레벨을 wick 으로 넘고 close 는 안으로 되돌아온 봉. 넘고 close 까지
    위에 있으면 돌파로 보고 레벨을 소멸시킨다(정통 sweep 정의).

    반환: [(idx, dir, level, wick, ref_swing_price)] — dir=-1 숏(REH 스윕), +1 롱.
    """
    lr = p["swing_lr"]
    n, h, lo, c = P["n"], P["h"], P["lo"], P["c"]
    # 확정시각 순으로 스윙 정렬
    hi_sw = sorted([(s.idx + lr, s.idx, float(s.price))
                    for s in P["swings"] if s.type is SwingType.HIGH])
    lo_sw = sorted([(s.idx + lr, s.idx, float(s.price))
                    for s in P["swings"] if s.type is SwingType.LOW])
    key = (id(P), lr, p["reh_tol"], p["reh_lb"])
    if key in _SWEEP_CACHE:
        return _SWEEP_CACHE[key]
    ph = pl = 0
    last_hi: list[tuple[int, float]] = []   # (idx, price) 확정 순
    last_lo: list[tuple[int, float]] = []
    reh = rel = None      # (level, formed_idx) — 살아있는 유동성 풀
    out = []
    for i in range(n):
        # ① 이 봉까지 확정된 스윙 반영
        while ph < len(hi_sw) and hi_sw[ph][0] <= i:
            last_hi.append((hi_sw[ph][1], hi_sw[ph][2]))
            ph += 1
            if len(last_hi) >= 2:
                a, b = last_hi[-1][1], last_hi[-2][1]
                if abs(a - b) / max(a, 1e-12) <= p["reh_tol"]:
                    reh = (max(a, b), last_hi[-1][0])   # 새 REH 형성
        while pl < len(lo_sw) and lo_sw[pl][0] <= i:
            last_lo.append((lo_sw[pl][1], lo_sw[pl][2]))
            pl += 1
            if len(last_lo) >= 2:
                a, b = last_lo[-1][1], last_lo[-2][1]
                if abs(a - b) / max(a, 1e-12) <= p["reh_tol"]:
                    rel = (min(a, b), last_lo[-1][0])
        # ② 유효기간 경과 시 소멸
        if reh and i - reh[1] > p["reh_lb"]:
            reh = None
        if rel and i - rel[1] > p["reh_lb"]:
            rel = None
        # ③ 스윕 판정
        if reh and h[i] > reh[0]:
            if c[i] < reh[0] and last_lo:
                out.append((i, -1, reh[0], float(h[i]), last_lo[-1][1]))
            reh = None       # 스윕이든 돌파든 이 풀은 소진
        if rel and lo[i] < rel[0]:
            if c[i] > rel[0] and last_hi:
                out.append((i, +1, rel[0], float(lo[i]), last_hi[-1][1]))
            rel = None
    _SWEEP_CACHE[key] = out
    return out


def _in_window(P: dict, i: int, p: dict) -> bool:
    """뉴욕 현지 07/08/09시 창 안인가 (+ 각 시각 첫 skip_first_min 분 제외)."""
    if P["ny_hour"][i] not in p["hours"]:
        return False
    return P["ny_min"][i] >= p["skip_first_min"]


# ─────────────────────────────────────────────────────────────────────────────
# 1단계 — 구조적 상한
# ─────────────────────────────────────────────────────────────────────────────
def structural_bound() -> dict:
    """스윕을 돌리기 전에 이 모델이 물리적으로 만들 수 있는 최대치를 산출한다."""
    print("=" * 78, flush=True)
    print("1단계 — 구조적 상한 (백테 전에 반드시)", flush=True)
    print("=" * 78, flush=True)
    p = dict(DEF)
    agg = dict(bars=0, am_bars=0, sweeps=0, am_sweeps=0, am_mss=0,
               struct_r=[], atr_r=[], years=0.0)
    for sym in SYMS:
        P = prep(sym)
        n = P["n"]
        am = sum(1 for i in range(n) if _in_window(P, i, p))
        ev = _sweep_events(P, p)
        am_ev = [e for e in ev if _in_window(P, e[0], p)]
        # MSS 까지 간 건수 (레인지 복귀 후 반대 스윙 돌파 종가)
        mss = 0
        for (i, d, lvl, wick, ref) in am_ev:
            for j in range(i + 1, min(i + 1 + p["mss_ttl"], n)):
                if (d == -1 and P["c"][j] < ref) or (d == +1 and P["c"][j] > ref):
                    mss += 1
                    break
        # SL 지렛대 — 구조 거리(스윕 wick ↔ 레벨) vs 라이브 ATR 바닥
        for (i, d, lvl, wick, ref) in am_ev:
            a = P["atr"][i]
            if not np.isfinite(a) or a <= 0:
                continue
            agg["struct_r"].append(abs(wick - lvl))
            agg["atr_r"].append(ATR_SL_MULT * a)
        agg["bars"] += n
        agg["am_bars"] += am
        agg["sweeps"] += len(ev)
        agg["am_sweeps"] += len(am_ev)
        agg["am_mss"] += mss
        span = (P["df"].index[-1] - P["df"].index[0]).days / 365.25
        agg["years"] = max(agg["years"], span)
        print(f"  {sym}  전체 {n:,}봉 · AM창 {am:,}봉({100*am/n:.1f}%) · "
              f"스윕 {len(ev):,} 중 AM {len(am_ev):,} · MSS 도달 {mss:,}", flush=True)

    st = np.array(agg["struct_r"])
    af = np.array(agg["atr_r"])
    dom = float((af > st).mean()) if len(st) else float("nan")
    print(f"\n  A. 시간 예산 — AM 창은 전체 5분봉의 "
          f"{100*agg['am_bars']/agg['bars']:.1f}% ({agg['years']:.1f}년)", flush=True)
    print(f"  B. 기회 상한 — AM 스윕 {agg['am_sweeps']:,}건 → MSS 통과 {agg['am_mss']:,}건 "
          f"(월 {agg['am_mss']/(agg['years']*12):.1f}건). 되돌림 체결·RR 필터는 아직 안 걸었다.",
          flush=True)
    # C. 노이즈 바닥 — 이런 종류 거래의 건당 R 표준편차는 경험적으로 1.3~1.8R
    for sd in (1.5,):
        for N in (agg["am_mss"], max(agg["am_mss"] // 3, 1)):
            se = sd / np.sqrt(max(N, 1))
            print(f"  C. 노이즈 바닥 — N={N:,} 이면 건당 R 표준오차 {se:.3f}R "
                  f"(sd 1.5R 가정) → 95%CI 폭 ±{1.96*se:.3f}R", flush=True)
    print(f"  D. SL 지렛대 — AM 스윕의 구조 거리 중앙 {np.median(st):.2f} vs "
          f"1.5×ATR 바닥 중앙 {np.median(af):.2f}. "
          f"바닥이 지배하는 비율 {100*dom:.0f}%", flush=True)
    if dom > 0.8:
        print("     → 구조 SL 이 거의 항상 ATR 바닥에 덮인다 = 진입 정밀도가 R 을 "
              "못 움직인다(피보와 같은 함정). 단 이 모델은 '새 거래를 더하는' 것이므로 "
              "빈도·방향 엣지는 별도로 남는다.", flush=True)
    else:
        print("     → 구조 SL 이 ATR 바닥보다 넓은 경우가 많다 = SL 폭이 실제로 "
              "모델에 의해 결정된다. R 지렛대 존재.", flush=True)
    return dict(am_bar_pct=100 * agg["am_bars"] / agg["bars"],
                am_sweeps=agg["am_sweeps"], am_mss=agg["am_mss"],
                years=agg["years"],
                se_at_mss=float(1.5 / np.sqrt(max(agg["am_mss"], 1))),
                atr_floor_dominant_pct=100 * dom)


# ─────────────────────────────────────────────────────────────────────────────
# 2단계 — 정통 07/08AM 시퀀스 백테
# ─────────────────────────────────────────────────────────────────────────────
def am_backtest(sym: str, **kw):
    """07/08AM 모델 1페어 백테. 반환: row 리스트 (SB/MMBM 과 같은 스키마)."""
    p = dict(DEF)
    p.update(kw)
    P = prep(sym)
    n, h, lo, c, o = P["n"], P["h"], P["lo"], P["c"], P["o"]
    atr, ts = P["atr"], P["ts"]
    # 전부 정렬 배열 + searchsorted 로 접근한다 (선형 스캔이면 8변형×수천건에서 못 끝난다)
    bull_fvg = sorted([f for f in P["fvgs"] if f.type is FVGType.BULLISH],
                      key=lambda f: f.idx)
    bear_fvg = sorted([f for f in P["fvgs"] if f.type is FVGType.BEARISH],
                      key=lambda f: f.idx)
    bull_idx = np.array([f.idx for f in bull_fvg], dtype=np.int64)
    bear_idx = np.array([f.idx for f in bear_fvg], dtype=np.int64)
    sw_hi_i = np.array([s.idx for s in P["swings"] if s.type is SwingType.HIGH],
                       dtype=np.int64)
    sw_hi_p = np.array([float(s.price) for s in P["swings"]
                        if s.type is SwingType.HIGH])
    sw_lo_i = np.array([s.idx for s in P["swings"] if s.type is SwingType.LOW],
                       dtype=np.int64)
    sw_lo_p = np.array([float(s.price) for s in P["swings"]
                        if s.type is SwingType.LOW])
    LIQ_LB = 576          # 유동성 타깃 탐색 범위 (2일)
    lr = p["swing_lr"]

    def last_fvg_before(fl, arr, i, lb):
        """i 직전(최대 lb 봉) 마지막 FVG — '스윕 직전 첫 FVG'(정통)."""
        a = int(np.searchsorted(arr, i - lb, side="left"))
        b = int(np.searchsorted(arr, i, side="right"))
        return fl[b - 1] if b > a else None

    def liq_target(i, price, risk, d):
        """다음 유동성 = 확정 스윙 중 min_rr 이상 떨어진 가장 가까운 것. 없으면 min_rr."""
        floor_ = price + d * p["min_rr"] * risk   # d=+1 롱이면 위, d=-1 숏이면 아래
        si, sp = (sw_hi_i, sw_hi_p) if d == 1 else (sw_lo_i, sw_lo_p)
        a = int(np.searchsorted(si, i - LIQ_LB, side="left"))
        b = int(np.searchsorted(si, i - lr, side="right"))
        if b <= a:
            return floor_
        seg = sp[a:b]
        cand = seg[seg >= floor_] if d == 1 else seg[seg <= floor_]
        if cand.size == 0:
            return floor_
        return float(cand.min() if d == 1 else cand.max())

    rows = []
    for (i, d, lvl, wick, ref) in _sweep_events(P, p):
        if not _in_window(P, i, p):
            continue
        if p["require_htf"] and P["bias1h"][i] * d < 0:
            continue
        # ① 레인지 복귀 + MSS (반대 스윙 종가 돌파)
        mss = None
        for j in range(i + 1, min(i + 1 + p["mss_ttl"], n)):
            if (d == -1 and c[j] < ref) or (d == +1 and c[j] > ref):
                mss = j
                break
        if mss is None:
            continue
        # ② 진입 존 — 스윕 직전 FVG 가 MSS 변위로 뒤집힌 IFVG. 없으면 Breaker.
        zone = None
        src = "breaker"
        f = (last_fvg_before(bull_fvg, bull_idx, i, p["ifvg_lb"]) if d == -1
             else last_fvg_before(bear_fvg, bear_idx, i, p["ifvg_lb"]))
        if f is not None:
            inverted = (c[mss] < f.low) if d == -1 else (c[mss] > f.high)
            if inverted:
                zone = (float(f.low), float(f.high))
                src = "ifvg"
        if zone is None:
            # Breaker — MSS 직전 구간에서 반대색 봉 하나 (숏이면 마지막 양봉)
            seg = range(i, mss + 1)
            cand = [k for k in seg if (c[k] > o[k]) == (d == -1)]
            if not cand:
                continue
            k = max(cand, key=lambda x: h[x]) if d == -1 else min(cand, key=lambda x: lo[x])
            zone = (float(lo[k]), float(h[k]))
        entry = (zone[0] + zone[1]) / 2.0     # CE (50%)
        # ③ 되돌림 체결 (지정가 — MSS 이후 창 안에서 터치)
        fill = None
        for k in range(mss + 1, min(mss + 1 + p["entry_ttl"], n)):
            if lo[k] <= entry <= h[k]:
                fill = k
                break
        if fill is None:
            continue
        # ④ SL — 스윕 wick 바깥. 라이브 ATR 바닥(1.5×ATR) 적용
        if d == -1:
            sl = max(float(h[i:fill + 1].max()), zone[1])
        else:
            sl = min(float(lo[i:fill + 1].min()), zone[0])
        a = atr[fill]
        if p["atr_floor"] and np.isfinite(a) and a > 0:
            need = ATR_SL_MULT * a
            if d == -1:
                sl = max(sl, entry + need)
            else:
                sl = min(sl, entry - need)
        # #ORIGO-1.3 — 라이브 SB 는 구조 SL 거리를 sl_dist_mult(=4.0) 배로 넓힌다.
        if p["sl_mult"] != 1.0:
            sl = entry + (sl - entry) * p["sl_mult"]
        risk = abs(entry - sl)
        if risk <= 0:
            continue
        # ⑤ TP — 다음 유동성(REL/REH), 최소 min_rr
        tp = liq_target(fill, entry, risk, d)
        # ⑥ 청산 — Origo 라이브 규약 (유동성TP + 트레일 2R/1.5R + BE 1R)
        gross, ex = M.simulate_exit(h, lo, fill, entry, sl, tp, d, n, detail=True)
        rows.append({
            "src": "AM", "sym": sym, "zone": src,
            "ent": int(ts[fill]), "ex": int(ts[min(ex, n - 1)]),
            "raw": float(gross), "r": float(gross * entry / risk),
            "dir": int(d), "trend": float(P["trend"][fill]) if np.isfinite(P["trend"][fill]) else 0.0,
            "risk_pct": float(risk / entry),
        })
    return rows


def placebo_rows(rows: list[dict]) -> list[dict]:
    """플라시보 — 같은 시각·같은 리스크 폭에서 **무작위 방향** 진입.

    "AM 시간대 자체가 좋은 것" 과 "이 시퀀스가 좋은 것" 을 가른다.
    """
    out = []
    for sym in SYMS:
        P = prep(sym)
        h, lo, ts, n = P["h"], P["lo"], P["ts"], P["n"]
        pos = {int(t): k for k, t in enumerate(ts)}
        for r in rows:
            if r["sym"] != sym:
                continue
            k = pos.get(r["ent"])
            if k is None:
                continue
            d = int(RNG.choice([-1, 1]))
            entry = float(P["c"][k])
            risk = r["risk_pct"] * entry
            sl = entry + d * -risk
            tp = entry + d * DEF["min_rr"] * risk
            g, ex = M.simulate_exit(h, lo, k, entry, sl, tp, d, n, detail=True)
            out.append({"src": "PLACEBO", "sym": sym, "ent": r["ent"],
                        "ex": int(ts[min(ex, n - 1)]), "raw": float(g),
                        "r": float(g * entry / risk), "dir": d,
                        "trend": r["trend"], "risk_pct": r["risk_pct"]})
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3단계 — 복리 평가 (mmbm_recheck_2026-08-06 과 동일 규약)
# ─────────────────────────────────────────────────────────────────────────────
def sim(rows):
    """복리 — 동시보유 size 분할 + DD 스로틀. 반환 (최종자산, 낙폭%, 파산)."""
    n = len(rows)
    if n < 5:
        return float("nan"), float("nan"), False
    rows = sorted(rows, key=lambda r: r["ent"])
    ent = np.fromiter((r["ent"] for r in rows), dtype=np.int64, count=n)
    ex = np.fromiter((r["ex"] for r in rows), dtype=np.int64, count=n)
    # 동시보유 수 = #{j: ent_j <= ex_i} - #{j: ex_j < ent_i}. (후자는 전자의 부분집합)
    # O(n^2) 이중루프를 searchsorted 로 대체 — 부트스트랩 1000회를 돌려야 하므로 필수.
    ent_s = np.sort(ent)
    ex_s = np.sort(ex)
    conc = (np.searchsorted(ent_s, ex, side="right")
            - np.searchsorted(ex_s, ent, side="left")).astype(float)
    conc = np.maximum(conc, 1.0)
    eq, peak, mdd = 1.0, 1.0, 0.0
    for i, r in enumerate(rows):
        sz = SIZE / conc[i]
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        eq *= (1.0 + r["raw"] * sz * LEV - 2.0 * TAKER_FEE_PCT * sz * LEV)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return eq, 100.0 * mdd, True
    return eq, 100.0 * mdd, False


def boot(rows):
    n = len(rows)
    if n < 5:
        return float("nan"), float("nan"), float("nan")
    fin, ruin = np.empty(N_BOOT), 0
    for k in range(N_BOOT):
        pick = [rows[i] for i in RNG.integers(0, n, size=n)]
        e, _, r_ = sim(pick)
        fin[k] = e
        ruin += int(r_)
    p50, p5 = np.percentile(fin, [50, 5])
    return float(p50), float(p5), 100.0 * ruin / N_BOOT


def describe(tag, rows, store=None):
    if len(rows) < 5:
        print(f"  {tag:<22}{len(rows):>5}  표본부족", flush=True)
        if store is not None:
            store.update(n=len(rows))
        return
    r = np.array([x["r"] for x in rows])
    e0, m0, _ = sim(rows)
    p50, p5, pr = boot(rows)
    span = (max(x["ex"] for x in rows) - min(x["ent"] for x in rows)) / 86400000 / 30.4
    print(f"  {tag:<22}{len(rows):>5}{len(rows) / max(span, 1e-9):>8.2f}"
          f"{r.mean():>8.3f}{100 * (r > 0).mean():>5.0f}%"
          f"{e0:>10.2f}x{m0:>7.1f}%{p50:>9.2f}x{p5:>8.3f}x{pr:>7.1f}%", flush=True)
    if store is not None:
        store.update(n=len(rows), r_mean=float(r.mean()),
                     wr=float(100 * (r > 0).mean()), compound=float(e0),
                     mdd=float(m0), boot_p50=float(p50), boot_p5=float(p5),
                     ruin=float(pr), per_month=float(len(rows) / max(span, 1e-9)))


def perm_label(a: np.ndarray, b: np.ndarray) -> float:
    """라벨 무작위 재배정 순열검정 — a(모델)와 b(플라시보)의 평균 차이가 우연인가.

    두 표본을 합쳐 라벨을 20000번 섞고, 관측 차이 이상이 나오는 비율 = p.
    """
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    pool = np.concatenate([a, b])
    na, N = len(a), len(pool)
    obs = float(a.mean() - b.mean())
    hits = 0
    for _ in range(N_PERM):
        idx = RNG.permutation(N)
        d = pool[idx[:na]].mean() - pool[idx[na:]].mean()
        hits += int(d >= obs)
    return hits / N_PERM


def perm_vs_zero(r: np.ndarray) -> float:
    """부호 무작위화 순열검정 — 건당 R 평균이 0 보다 유의하게 큰가."""
    if len(r) < 5:
        return float("nan")
    obs = float(r.mean())
    ar = np.abs(r)
    hits, done = 0, 0
    while done < N_PERM:                      # 메모리 안전 위해 청크 분할
        b = min(2000, N_PERM - done)
        sgn = RNG.choice(np.array([-1.0, 1.0]), size=(b, len(r)))
        hits += int(((sgn * ar).mean(axis=1) >= obs).sum())
        done += b
    return hits / N_PERM


def header():
    print(f"  {'구성':<22}{'거래':>5}{'월빈도':>8}{'건당R':>8}{'승률':>6}"
          f"{'자산':>10}{'낙폭':>8}{'부트중앙':>9}{'5%':>8}{'파산':>7}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
def sb_rows():
    """Silver Bullet — 라이브 정합(run_live_parity) 그대로."""
    out = []
    for sym in SYMS:
        df5, kept, st = run_live_parity(sym)
        print(f"  SB {sym} 통과 {len(kept)}건 (게이트 {st})", flush=True)
        idx = df5.index
        for t in kept:
            risk = abs(float(t.entry) - float(getattr(t, "entry_sl", 0.0) or 0.0))
            out.append({
                "src": "SB", "sym": sym,
                "ent": int(idx[t.entry_idx].value // 10**6),
                "ex": int(idx[min(t.exit_idx, len(idx) - 1)].value // 10**6),
                "raw": float(t.raw_pnl_pct),
                "r": (float(t.raw_pnl_pct) * float(t.entry) / risk) if risk > 0 else 0.0,
                "dir": 1 if str(getattr(t.direction, "value", t.direction)).lower() == "long" else -1,
                "trend": float(t.entry_trend_pct or 0.0),
                "risk_pct": (risk / float(t.entry)) if t.entry else 0.0,
            })
    return out


def _smt_fast(swings, cdf):
    """`detect_smt_divergence` 와 판정 동일한 고속 구현 (bull_idx, bear_idx 집합).

    원본 `_corr_price_at` 은 상관자산 봉 **전체를 선형 스캔**해 가장 가까운 ts 를
    찾는다. 스윙 9.7만 × 봉 52만 = 5×10^10 회 — 5년 규모에서는 끝나지 않는다.
    (7/20 에 이 함수가 시그니처 오호출로 조용히 죽어 있었던 것도 같은 자리다.)
    여기서는 정렬된 ts 배열에 이분탐색으로 최근접 봉을 찾는다. 동률이면 원본의
    strict `<` 과 같게 **앞선 봉**을 택한다. 허용 드리프트 1시간도 동일.
    """
    from aurora_ict.indicators.smt import _MAX_TS_DRIFT_MS

    cts = (cdf.index.astype("int64") // 10**6).to_numpy()
    chi = cdf["high"].to_numpy()
    clo = cdf["low"].to_numpy()

    def near(ts_arr):
        pos = np.searchsorted(cts, ts_arr)
        lo_i = np.clip(pos - 1, 0, len(cts) - 1)
        hi_i = np.clip(pos, 0, len(cts) - 1)
        dl = np.abs(cts[lo_i] - ts_arr)
        dh = np.abs(cts[hi_i] - ts_arr)
        pick = np.where(dl <= dh, lo_i, hi_i)      # 동률 → 앞선 봉
        return pick, np.minimum(dl, dh)

    out_bull: set[int] = set()
    out_bear: set[int] = set()
    for stype, is_high, bucket in ((SwingType.HIGH, True, out_bear),
                                   (SwingType.LOW, False, out_bull)):
        seq = [s for s in swings if s.type is stype]
        if len(seq) < 2:
            continue
        ts = np.array([s.ts_ms for s in seq], dtype=np.int64)
        pr = np.array([float(s.price) for s in seq])
        ids = np.array([s.idx for s in seq], dtype=np.int64)
        pick, drift = near(ts)
        cp = (chi if is_high else clo)[pick]
        ok = drift <= _MAX_TS_DRIFT_MS
        prev_ok, curr_ok = ok[:-1], ok[1:]
        both = prev_ok & curr_ok
        if is_high:
            main_d = pr[1:] > pr[:-1]
            corr_d = cp[1:] > cp[:-1]
        else:
            main_d = pr[1:] < pr[:-1]
            corr_d = cp[1:] < cp[:-1]
        hit = both & (main_d != corr_d)
        bucket.update(int(x) for x in ids[1:][hit])
    return out_bull, out_bear


def _mmbm_backtest_fast(sym: str):
    """`mmbm_full.backtest(sym, detail=True)` 의 **판정 동일·고속** 재구현.

    원본은 구조이벤트(9.7만) 마다 스윕 인덱스 리스트(4.2만)를 선형 스캔하는
    `had()` 때문에 5년 2페어에 수 시간이 걸린다. 여기서는 같은 판정을 정렬배열
    이분탐색으로 한다. 로직·상수는 원본 그대로(FEE·SLIP·RANGE_N·SWEEP_LB·
    RETRACE_TTL·HTF_LB·SMT·구조SL·유동성TP·simulate_exit) 이며, 동일성은
    `verify_fast_mmbm()` 로 앞 12만봉에서 원본과 대조 확인한다.
    """
    import bisect

    from aurora_ict.indicators.structure import StructureType, detect_structure_events

    df = _resample(_load_full(sym))
    c, h, lo = df["close"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy()
    n = len(c)
    ts_ms = (df.index.astype("int64") // 10**6).to_numpy()
    c1h = df["close"].resample("1h").last().ffill()
    bias1h = (np.sign(c1h - c1h.shift(M.HTF_LB)).reindex(df.index, method="ffill")
              .fillna(0).to_numpy())
    swings = detect_swing_points(df, left=3, right=3)
    events = detect_structure_events(df, swings)
    fvgs = detect_fvgs(df, min_size_pct=0.001)
    sweeps = M.detect_liquidity_sweeps(df, swings)
    bull_fvg = sorted([f for f in fvgs if f.type is FVGType.BULLISH], key=lambda f: f.idx)
    bear_fvg = sorted([f for f in fvgs if f.type is FVGType.BEARISH], key=lambda f: f.idx)
    bull_i = np.array([f.idx for f in bull_fvg], dtype=np.int64)
    bear_i = np.array([f.idx for f in bear_fvg], dtype=np.int64)
    ssl = np.array(sorted(s.idx for s in sweeps if s.type is M.SweepType.BULLISH),
                   dtype=np.int64)
    bsl = np.array(sorted(s.idx for s in sweeps if s.type is M.SweepType.BEARISH),
                   dtype=np.int64)
    sh = sorted([(s.idx, float(s.price)) for s in swings if s.type is SwingType.HIGH])
    sl_ = sorted([(s.idx, float(s.price)) for s in swings if s.type is SwingType.LOW])

    smt_bull_s: set[int] = set()
    smt_bear_s: set[int] = set()
    corr = M.CORR.get(sym, "BTCUSDT")
    if corr != sym:
        cdf = _resample(_load_full(corr))
        smt_bull_s, smt_bear_s = _smt_fast(swings, cdf)
    smt_bull_a = np.array(sorted(smt_bull_s), dtype=np.int64)
    smt_bear_a = np.array(sorted(smt_bear_s), dtype=np.int64)
    use_smt = True

    def had(arr, i, lb):
        if arr.size == 0:
            return False
        a = int(np.searchsorted(arr, i - lb, side="left"))
        b = int(np.searchsorted(arr, i, side="right"))
        return b > a

    def recent_fvg(fl, arr, i):
        a = int(np.searchsorted(arr, i - 10, side="left"))
        b = int(np.searchsorted(arr, i + 2, side="right"))
        return fl[b - 1] if b > a else None

    # 288봉 롤링 고·저 (원본의 h[i-288:i].max() / lo[i-288:i].min() 과 동일 구간)
    rhi_a = pd.Series(h).rolling(M.RANGE_N).max().shift(1).to_numpy()
    rlo_a = pd.Series(lo).rolling(M.RANGE_N).min().shift(1).to_numpy()

    # next_bsl / next_ssl 을 위한 증분 정렬 리스트 (이벤트를 idx 오름차순으로 처리)
    ev_sorted = sorted(events, key=lambda e: e.idx)
    hp: list[float] = []      # si <= i 인 swing high 가격 (정렬)
    lp: list[float] = []
    ph = pl = 0
    trades = []
    for ev in ev_sorted:
        i = ev.idx
        while ph < len(sh) and sh[ph][0] <= i:
            bisect.insort(hp, sh[ph][1]); ph += 1
        while pl < len(sl_) and sl_[pl][0] <= i:
            bisect.insort(lp, sl_[pl][1]); pl += 1
        if i < M.RANGE_N or i >= n - 1:
            continue
        rhi, rlo = rhi_a[i], rlo_a[i]
        if not np.isfinite(rhi) or rhi <= rlo:
            continue
        pos = (c[i] - rlo) / (rhi - rlo)
        bull = ev.type is StructureType.CHOCH_BULLISH
        bear = ev.type is StructureType.CHOCH_BEARISH
        if bull and pos < 0.5 and bias1h[i] >= 0:
            if not had(ssl, i, M.SWEEP_LB):
                continue
            if use_smt and i not in smt_bull_s and not had(smt_bull_a, i, M.SWEEP_LB):
                continue
            fvg = recent_fvg(bull_fvg, bull_i, i)
            if fvg is None:
                continue
            entry = fvg.mean_threshold
            sl = min(float(lo[max(0, i - M.SWEEP_LB):i + 1].min()), fvg.low)
            if entry - sl <= 0:
                continue
            risk = entry - sl
            k = bisect.bisect_right(hp, entry)
            liq = hp[k] if k < len(hp) else None
            tp = max(liq if liq else entry + 2 * risk, entry + 2 * risk)
            d = 1
        elif bear and pos > 0.5 and bias1h[i] <= 0:
            if not had(bsl, i, M.SWEEP_LB):
                continue
            if use_smt and i not in smt_bear_s and not had(smt_bear_a, i, M.SWEEP_LB):
                continue
            fvg = recent_fvg(bear_fvg, bear_i, i)
            if fvg is None:
                continue
            entry = fvg.mean_threshold
            sl = max(float(h[max(0, i - M.SWEEP_LB):i + 1].max()), fvg.high)
            if sl - entry <= 0:
                continue
            risk = sl - entry
            k = bisect.bisect_left(lp, entry) - 1
            liq = lp[k] if k >= 0 else None
            tp = min(liq if liq else entry - 2 * risk, entry - 2 * risk)
            d = -1
        else:
            continue
        fill = None
        for j in range(i + 1, min(i + 1 + M.RETRACE_TTL, n)):
            if lo[j] <= entry <= h[j]:
                fill = j
                break
        if fill is None:
            continue
        gross, ex_j = M.simulate_exit(h, lo, fill, entry, sl, tp, d, n, detail=True)
        trades.append((int(ts_ms[fill]), int(ts_ms[ex_j]), d,
                       float(gross), float(gross * entry / risk)))
    return trades


def mmbm_rows():
    """MMBM — 8/6 재검증과 같은 구성 (현재 라이브 ON). 결과는 pickle 캐시."""
    import pickle
    cache = "data/axis/_mmbm_rows.pkl"
    if os.path.exists(cache):
        with open(cache, "rb") as fh:
            return pickle.load(fh)
    out = []
    for sym in SYMS:
        tr = _mmbm_backtest_fast(sym)
        print(f"  MMBM {sym} {len(tr)}건", flush=True)
        for ent_ms, ex_ms, d, gross, r_mult in tr:
            out.append({"src": "MMBM", "sym": sym, "ent": ent_ms, "ex": ex_ms,
                        "raw": gross, "r": r_mult, "dir": int(d),
                        "trend": 0.0, "risk_pct": 0.0})
    os.makedirs("data/axis", exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump(out, fh)
    return out


def verify_fast_mmbm(bars: int = 120000) -> None:
    """고속 재구현이 원본 `mmbm_full.backtest` 와 같은 거래를 내는지 앞 구간에서 대조."""
    import types
    full = _load_full("BTCUSDT")
    sub = full.iloc[: bars * 5]          # 1분봉 → 5분봉 bars 개
    orig_loader = M._load_full
    M._load_full = lambda s: (sub if s == "BTCUSDT" else orig_loader(s).iloc[: bars * 5])
    g = globals()
    old = g["_load_full"]
    g["_load_full"] = M._load_full
    try:
        _df, a = M.backtest("BTCUSDT", detail=True)
        b = _mmbm_backtest_fast("BTCUSDT")
        sa = {(t[0], t[2]) for t in a}
        sb_ = {(t[0], t[2]) for t in b}
        print(f"  동일성 검증 — 원본 {len(a)}건 · 고속 {len(b)}건 · "
              f"교집합 {len(sa & sb_)} · 원본만 {len(sa - sb_)} · 고속만 {len(sb_ - sa)}",
              flush=True)
    finally:
        M._load_full = orig_loader
        g["_load_full"] = old


def dedup(new, base_rows):
    """base 와 같은 심볼·방향·1시간 버킷이면 같은 기회로 보고 제외."""
    keys = {(x["sym"], x["dir"], x["ent"] // DEDUP_MS) for x in base_rows}
    return [x for x in new if (x["sym"], x["dir"], x["ent"] // DEDUP_MS) not in keys]


def cost_bound() -> dict:
    """구조적 상한 보론 — **왕복 수수료가 1R 의 몇 %인가**.

    이 모델의 SL 은 스윕 wick 기준이라 매우 타이트하다. 타이트한 SL 은 R 배수로는
    유리해 보여도, 고정 size_pct 방식에서는 **수수료가 R 로 환산될 때 폭증**한다.
    왕복 taker 0.08% ÷ 리스크폭 = 손익분기 R. 이 값이 모델이 실제로 낼 수 있는
    건당 R 보다 크면 게임 자체가 성립하지 않는다 — 피보 연구의 '이론 상한' 과 같은
    자리의 검사다.
    """
    print("\n" + "=" * 78, flush=True)
    print("1단계 보론 — 수수료의 R 환산 (손익분기 R)", flush=True)
    print("=" * 78, flush=True)
    rt = 2.0 * TAKER_FEE_PCT
    out = {}
    print(f"  {'구성':<24}{'거래':>6}{'SL폭중앙':>10}{'손익분기R':>11}{'총R평균':>10}{'수수료후':>10}",
          flush=True)
    sb = sb_rows()
    for tag, rows in [("SB(현행·비교용)", sb)] + [
            (nm, sum((am_backtest(s, **kw) for s in SYMS), []))
            for nm, kw in [("V1 기본", {}), ("V9 라이브SL규약(×4)", dict(sl_mult=4.0)),
                           ("V6 ATR바닥 해제", dict(atr_floor=False))]]:
        if not rows:
            continue
        rp = float(np.median([x["risk_pct"] for x in rows if x["risk_pct"] > 0]))
        be = rt / rp
        rm = float(np.mean([x["r"] for x in rows]))
        print(f"  {tag:<24}{len(rows):>6}{100*rp:>9.3f}%{be:>11.3f}{rm:>10.3f}"
              f"{rm - be:>10.3f}", flush=True)
        out[tag] = dict(n=len(rows), sl_pct=100 * rp, breakeven_r=be,
                        r_gross=rm, r_net=rm - be)
    return out


def main() -> int:
    bound = structural_bound()

    print("\n" + "=" * 78, flush=True)
    print("2단계 — 기준선 (현행 SB + MMBM)", flush=True)
    print("=" * 78, flush=True)
    sb = sb_rows()
    mm = mmbm_rows()
    mm_add = dedup(mm, sb)
    base_rows = sb + mm_add
    header()
    b_sb, b_cur = {}, {}
    describe("SB 단독", sb, b_sb)
    describe("MMBM 순수추가분", mm_add)
    describe("현행 SB+MMBM(기준선)", base_rows, b_cur)

    print("\n" + "=" * 78, flush=True)
    print("2단계 — 07/08AM 모델 변형 (시도 조합 수는 아래 명시)", flush=True)
    print("=" * 78, flush=True)
    # ── 시도하는 조합 (다중비교 명시용) ────────────────────────────────────
    VARIANTS = [
        ("V1 기본(07/08/09AM)", {}),
        ("V2 07AM 단독", dict(hours=(7,))),
        ("V3 08AM 단독", dict(hours=(8,))),
        ("V4 첫30분 제외", dict(skip_first_min=30)),
        ("V5 HTF바이어스 정합", dict(require_htf=True)),
        ("V6 ATR바닥 해제", dict(atr_floor=False)),
        ("V7 IFVG만(체결창 확대)", dict(entry_ttl=24)),
        ("V8 MSS대기 축소", dict(mss_ttl=6)),
        ("V9 라이브SL규약(×4)", dict(sl_mult=4.0)),
        ("V10 V9+첫30분제외", dict(sl_mult=4.0, skip_first_min=30)),
    ]
    header()
    results = {}
    for name, kw in VARIANTS:
        rows = []
        for sym in SYMS:
            rows += am_backtest(sym, **kw)
        st = {}
        describe(name, rows, st)
        results[name] = (rows, kw, st)

    print(f"\n  ※ 시도한 조합 = {len(VARIANTS)}개 (무보정 argmax 방지 위해 전부 표기)",
          flush=True)

    print("\n" + "=" * 78, flush=True)
    print("3단계 — 판정 (기본 변형 V1 기준 + 최상위 변형)", flush=True)
    print("=" * 78, flush=True)

    cands = []
    # 표본 30건 미만은 결론에서 제외한다는 규칙에 따라 분리
    ranked = sorted(
        [(nm, v) for nm, v in results.items() if v[2].get("n", 0) >= 30],
        key=lambda kv: kv[1][2].get("r_mean", -9), reverse=True)
    check = ["V1 기본(07/08/09AM)"] + [nm for nm, _ in ranked[:2]]
    seen = set()
    check = [x for x in check if not (x in seen or seen.add(x))]

    for name in check:
        rows, kw, st = results[name]
        entry = dict(name=name, params={k: (list(v) if isinstance(v, tuple) else v)
                                        for k, v in kw.items()})
        entry.update(st)
        print(f"\n  ── {name}  (n={len(rows)}) " + "─" * 30, flush=True)
        if len(rows) < 30:
            print("     표본부족(<30) — 결론에서 제외", flush=True)
            entry["verdict"] = "표본부족"
            cands.append(entry)
            continue
        r = np.array([x["r"] for x in rows])

        # ① 순열검정 — 건당 R 이 0 보다 유의한가 (부호 무작위화 20000회)
        p0 = perm_vs_zero(r)
        entry["p_perm"] = p0
        print(f"  ① 순열검정 건당R vs 0 : 관측 {r.mean():+.3f}R  p={p0:.4f} "
              f"→ {'유의' if p0 < 0.05 else '기각(p>=0.05)'}", flush=True)

        # ② 플라시보 — 같은 시각·같은 리스크, 무작위 방향
        pl = placebo_rows(rows)
        pr_ = np.array([x["r"] for x in pl])
        pv = perm_label(r, pr_) if len(pr_) >= 5 else float("nan")
        entry["p_placebo"] = float(pv)
        entry["placebo_r"] = float(pr_.mean()) if len(pr_) else float("nan")
        print(f"  ② 플라시보(무작위 방향) 건당 {pr_.mean():+.3f}R  vs 모델 {r.mean():+.3f}R"
              f"  p={pv:.4f}", flush=True)

        # ③ 국면×방향 기저 — "상승장 롱" 재확인인지
        print("  ③ 국면×방향 기저:", flush=True)
        tr = np.array([x["trend"] for x in rows])
        dd_ = np.array([x["dir"] for x in rows])
        q = np.nanpercentile(tr, [33, 67])
        for lab, m in (("하락국면", tr < q[0]), ("중립국면", (tr >= q[0]) & (tr < q[1])),
                       ("상승국면", tr >= q[1])):
            for dlab, dm in (("롱", dd_ == 1), ("숏", dd_ == -1)):
                sel = m & dm
                if sel.sum() < 30:
                    print(f"     {lab} {dlab}: n={sel.sum()} 표본부족", flush=True)
                    continue
                print(f"     {lab} {dlab}: n={sel.sum():4d} 건당 {r[sel].mean():+.3f}R "
                      f"승률 {100*(r[sel]>0).mean():.0f}%", flush=True)

        # ④ 연도 일관성 — 특정 연도 몰빵이면 기각
        print("  ④ 연도 일관성:", flush=True)
        ys = np.array([dt.datetime.utcfromtimestamp(x["ent"] / 1000).year for x in rows])
        tot = r.sum()
        ycon = []
        for y in sorted(set(ys.tolist())):
            m = ys == y
            share = 100 * r[m].sum() / tot if tot != 0 else float("nan")
            ycon.append((int(y), int(m.sum()), float(r[m].mean()), float(share)))
            print(f"     {y}  n={m.sum():4d}  건당 {r[m].mean():+.3f}R  "
                  f"총R기여 {r[m].sum():+7.1f} ({share:+.0f}%)", flush=True)
        entry["by_year"] = ycon
        pos_years = sum(1 for _, n_, rm, _ in ycon if n_ >= 30 and rm > 0)
        tot_years = sum(1 for _, n_, _, _ in ycon if n_ >= 30)
        max_share = max((abs(s) for _, n_, _, s in ycon if n_ >= 5), default=0.0)
        entry["years_pos"] = f"{pos_years}/{tot_years}"
        entry["max_year_share"] = float(max_share)

        # ⑤ 롱/숏 분리
        print("  ⑤ 롱/숏:", flush=True)
        for d_, lab in ((1, "롱"), (-1, "숏")):
            sel = dd_ == d_
            if sel.sum() < 30:
                print(f"     {lab}: n={sel.sum()} 표본부족", flush=True)
                continue
            print(f"     {lab}: n={sel.sum():4d} 건당 {r[sel].mean():+.3f}R "
                  f"승률 {100*(r[sel]>0).mean():.0f}%", flush=True)

        # ⑥ 증분 — SB+MMBM 과 시각 중복 제거 후 순수 추가분
        add = dedup(rows, base_rows)
        ar = np.array([x["r"] for x in add]) if add else np.array([])
        print(f"  ⑥ 증분 — 중복 {len(rows)-len(add)}건 제거, 순수 추가 {len(add)}건",
              flush=True)
        header()
        st_add, st_comb = {}, {}
        describe("  순수 추가분", add, st_add)
        describe("  기준선+이 모델", base_rows + add, st_comb)
        entry["incremental"] = st_add
        entry["combined"] = st_comb
        if len(ar) >= 5:
            entry["p_incr"] = perm_vs_zero(ar)
            print(f"     추가분 건당 {ar.mean():+.3f}R  순열 p={entry['p_incr']:.4f}",
                  flush=True)

        # 판정
        ok = (p0 < 0.05
              and (np.isnan(pv) or pv < 0.05)
              and tot_years >= 3 and pos_years >= tot_years - 1
              and max_share <= 60
              and st_comb.get("compound", 0) > b_cur.get("compound", 0)
              and st_comb.get("ruin", 100) <= b_cur.get("ruin", 0) + 1)
        entry["verdict"] = "유망" if ok else "기각"
        print(f"  ▶ 판정: {entry['verdict']}", flush=True)
        cands.append(entry)

    out = dict(
        axis="07AM/08AM 진입 모델 (ICT 2024 Mentorship) — 정통 시퀀스 최초 구현·재측정",
        structural_bound=bound,
        n_variants_tried=len(VARIANTS),
        variants_all={nm: dict(params={k: (list(v) if isinstance(v, tuple) else v)
                                      for k, v in v_[1].items()}, **v_[2])
                      for nm, v_ in results.items()},
        candidates=cands,
        baseline=dict(name="현행 SB+MMBM", **b_cur, sb_only=b_sb),
        notes="",
    )
    os.makedirs("data/axis", exist_ok=True)
    with open("data/axis/model-0708.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2, default=float)
    print("\n저장: data/axis/model-0708.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
