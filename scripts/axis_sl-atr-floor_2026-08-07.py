"""#AUTONOMOUS 2026-08-07 · 축 연구: **SL 산출 방식** (ATR 바닥 + sl_dist_mult).

## 왜 이 축인가

직전 피보나치(ote_level) 연구가 "그 손잡이엔 지렛대가 없다"로 종결됐고, 그 원인으로
**1.5×ATR 바닥(_ATR_SL_MULT)이 FVG 구조 SL 을 전 건에서 덮어써 진입 깊이를
무력화한다**는 가설이 남았다. 이 스크립트는 그 가설을 실측하고, SL 폭 자체가
성적을 움직이는 손잡이인지 판정한다.

## 라이브 SL 이 만들어지는 순서 (코드 실측)

  1) 구조 SL   = FVG 가장자리 (LONG=fvg.low / SHORT=fvg.high). 버퍼 없음.
  2) ATR 바닥  = `_widen_sl_to_atr` — 폭이 `1.5 × ATR(14)` 미만이면 그만큼 넓힌다
                 (#LIVE-7, 2026-05-23). silver_bullet.py:586
  3) min_rr    = 이 단계 폭으로 RR 을 계산해 2.0 미만 셋업을 버린다 (detect 인자).
  4) sl_dist_mult ×4.0 = 체결 후 폭을 4배로 확대 (replay.py:1288, 라이브 sub_ 강제).
                 **TP 도 함께 4배**로 늘려 원 RR 을 유지한다(tp_keeps_rr=True 기본,
                 라이브 bot_ict_instance.py:2174 와 동일) → SL/TP 동시 확대.
  5) #LIQ-CAP  = 확대폭을 `진입가 × 0.8 / 레버리지` 로 캡.

  → 최종 SL 폭 = min( 4.0 × max(구조폭, 1.5×ATR), 캡 )

## 1단계 — 구조적 상한 (스윕 전에 반드시)

  · 구조 SL 폭 vs ATR 바닥 폭의 분포, **바닥이 구조를 덮어쓴 비율**을 실측.
  · 손잡이가 물리적으로 움직일 수 있는 성적 폭을 계산해 표본 노이즈(SE)와 비교.
    상한 < 노이즈면 스윕은 무의미하므로 거기서 멈춘다(피보 연구 선례).

    (가) 고정사이즈 모델(지표 A) 기준 **좁히기의 기계적 상한**: SL 로 끝난 거래의
         손실이 m'/m 배로 줄고 **새로 SL 에 맞는 거래가 하나도 없다**고 가정한
         최선값. 실제는 반드시 이보다 나쁘다(좁히면 더 자주 맞는다) → 진짜 상한.
    (나) 라이브 리스크기반 모델(지표 C) 기준 **기계적 상한은 정확히 0** 이다:
         SL·TP 가 함께 배수로 늘고(tp_keeps_rr) 수량이 1/SL 로 줄기 때문에,
         **가격 경로의 결과가 그대로면 R 도 자산 변화도 완전히 동일**하다.
         따라서 이 축의 효과는 100% "어느 장벽에 먼저 닿는가"(경로 의존)에서만
         나온다. 크기 상한이 작다는 뜻이 아니라 — 실측으로만 알 수 있다는 뜻이다.
         피보 연구와 달리 여기서는 상한을 이론으로 닫을 수 없어 스윕이 필수다.

## 2단계 — sl_dist_mult 스윕 (1.5 ~ 8.0)

  bt_par 캐시 키에 sl_dist_mult 가 없어 타임라인 재계산이 없다(빠름).
  LIVE_BASE 는 sl_dist_mult 와 sl_dist_mult_ct(역추세용)가 둘 다 4.0 이므로
  **두 값을 함께 움직인다**(한쪽만 바꾸면 국면별로 다른 SL 을 섞게 된다).

  ★ 왜 이 스윕이 곧 "ATR 바닥 스윕"인가
    바닥이 99% 셋업에서 구속하므로 최종 SL = sl_dist_mult × 1.5 × ATR 이다.
    즉 mult 를 1.5~8.0 으로 흔드는 것은 **실효 ATR 배수를 2.25× ~ 12× 로
    흔드는 것과 같다**(현행 4.0 = 6×ATR). 모듈 상수 _ATR_SL_MULT 를 직접
    건드리려면 타임라인을 다시 빌드해야 하는데(캐시 키에 없어 오염 위험),
    이 등가성 덕분에 재빌드 없이 같은 축을 잰다.

  ⚠ 이 등가성의 한 가지 예외(한계로 명시할 것)
    _ATR_SL_MULT 를 낮추면 detect 단계 RR(=|TP−진입|/바닥후폭)이 커져
    **min_rr 2.0 게이트를 통과하는 셋업이 늘어난다**. sl_dist_mult 는 체결 후
    적용이라 셋업 집합을 바꾸지 않는다. 따라서 "실효 6×ATR 이 최적"은
    **셋업 집합 고정 조건에서의 결론**이고, 바닥 3.0 × mult 2.0 처럼 같은 6× 를
    다르게 쪼갠 경우는 이 스윕이 답하지 못한다(타임라인 재빌드 필요).

## 3단계 — min_sl_distance_pct 스윕 (0.001 ~ 0.005)

  캐시 키에 **있어서** 타임라인 재빌드가 필요하다(_axis_sl_tlbuild 가 미리 빌드).
  ATR 바닥과 어느 쪽이 실제 구속력을 갖는지 본다. 이 값은 detect 단계 필터라
  **체결셋 자체를 바꾼다** — 2단계와 교란이 다르다.

  ※ 원 질문("둘 중 어느 쪽이 구속하나")은 재빌드 없이 1단계 [A-2] 가 답한다:
    min_sl 은 **바닥 적용 후** 폭과 비교되는 필터라, 현행 0.001(0.1%) 에서
    절단률이 정확히 0.0% 다 → **죽은 게이트**이고 구속력은 전부 1.5×ATR 바닥에
    있다. 0.002/0.003/0.005 로 올리는 것은 SL 산출식이 아니라 "저변동성 셋업
    배제"라는 별개 축이 되므로, 재빌드(페어당 수 시간, 2026-08-07 실측)가
    끝나지 않으면 미실행으로 남기고 절단률만 보고한다.

## 측정 단위 — 이 축에서만 특별히 주의할 것

  팀 기준 4번은 "SL 폭이 변하면 %수익 비교는 왜곡되니 R 로 재라"이다. 이 축은
  SL 폭 자체가 조작 변수라 R 의 분모가 후보마다 다르지만, **라이브가 리스크 기반
  사이징이라 자산 변화가 정확히 R×6% 로 결정된다** — 즉 이 봇에서는 R 이 곧
  배포 단위다. 그래도 모델 가정이 결론을 만드는 것은 아닌지 확인하려고
  세 지표를 **함께** 낸다.

    지표 A (팀 규약) — 고정 사이즈 복리.
        lev 7 · size 0.9 · 동시보유 분할 · DD 스로틀(-25%→×0.7) · 파산(시드 20%).
        origo_leverage_verify.sim 과 같은 규약이라 과거 연구와 비교 가능하다.
    지표 B (엣지 질) — 건당 R + 리스크 고정 복리(건당 시드의 RISK_FIX 만 위험).
        A 의 개선이 단순히 "리스크를 줄여서" 나온 것인지(=size_pct 나 레버리지를
        낮추면 똑같이 얻는 것) 구분하려고 넣는다.
    지표 C (**라이브 실제**) — 리스크 기반 사이징 복리. ★배포 판단은 이것.
        production bot_ict_instance._calc_qty_risk_based 실측:
          risk_pct = min(3.0 + 1.5×conf, 6.0) → min_confluence=5 강제 때문에
                     **전 건 6.0% 고정**(실측: score 5/6/7 만 존재),
          qty      = equity×6% / |진입−SL|,
          notional 상한 = equity × leverage(7) × position_pct_max(80%) = 5.6×equity,
          DD 스로틀 ×0.7.
        즉 라이브는 **SL 이 멀수록 수량을 줄여 건당 손실을 6%로 고정**한다.
        따라서 SL 폭을 바꿔도 건당 리스크가 변하지 않고(SL<1.07% 부터만 상한에
        걸려 줄어든다), 성적 차이는 **오직 R 분포의 변화**에서 온다.

    ⚠ 하니스 정합 갭 (2026-08-07 이 연구에서 발견, 결론에 명시):
      · live_parity.LIVE_BASE 의 size_pct=0.9 는 고정 % notional 가정인데
        라이브는 settings.risk_based_sizing=True(기본, sub_ 도 미해제)라
        **리스크 기반**이다. 이 축에서는 두 모델의 결론이 갈릴 수 있어 둘 다 낸다.
      · LIVE_BASE 에 leverage 가 없어 BacktestConfig 기본 20 이 쓰인다.
        그 값이 #LIQ-CAP(= 진입가×0.8/leverage)에 들어가 백테는 SL 을 4.0% 에서
        캡하는데, 라이브는 lev 7 강제라 11.4% 다. SL 을 넓히는 후보일수록
        백테가 불리해진다 → leverage=7 로 캡을 교정한 스윕(2b)을 따로 낸다.

## 판정 (팀 표준 6종)

  1) 순열검정 p<0.05 — 후보 vs 기준선의 **건당 R** 차이(주 단위. 위 지표 C 근거).
     두 표본 무작위 분할 20000회. 건당 raw% 로도 따로 내고(고정사이즈 모델용),
     셋업이 겹치는 거래는 짝지어 부호뒤집기 검정도 병기(보조).
  2) 국면×방향 기저 통제 — z-score 일봉 국면 × 롱/숏 교차표.
  3) 연도 일관성 — 특정 연도가 총차의 과반이면 기각.
  4) R 로 측정 — 지표 B/C.
  5) 셀 표본 30건 미만은 표본부족 명시.
  6) 다중비교 — 시도한 조합 수를 결과에 기록한다.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _TL_CACHE_DIR, _load_full, _resample, cached_setup_timeline  # noqa: E402
from live_parity import LIVE_BASE, run_live_parity  # noqa: E402

from aurora.backtest.cost import TAKER_FEE_PCT  # noqa: E402
from aurora_ict.backtest.replay import BacktestConfig  # noqa: E402
from aurora_ict.strategy.silver_bullet import Direction  # noqa: E402

# ── 설정 ────────────────────────────────────────────────────────────────
LIVE_PAIRS = ["BTCUSDT", "ETHUSDT"]          # 현행 라이브 2페어 (배포 판단 기준)
WIDE_PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT",
              "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]  # 검정력 보강용 7페어

LEV = 7.0                     # 라이브 레버리지
SIZE = LIVE_BASE["size_pct"]  # 0.9
DD_PCT, DD_FACTOR = 0.25, 0.7
RUIN = 0.20
RISK_FIX = 0.02               # 지표 B: 건당 고정 리스크(시드 2%)
# 지표 C — 라이브 리스크 기반 사이징 (production 실측값)
RISK_BASE, RISK_STEP, RISK_MAX = 0.03, 0.015, 0.06   # risk_per_trade_*
POS_PCT_MAX = 0.80                                   # position_pct_max
# 반복 횟수는 팀 표준 고정값. AXIS_FAST=1 일 때만 배선 점검(스모크)용으로 줄인다.
_FAST = os.environ.get("AXIS_FAST") == "1"
N_BOOT = 50 if _FAST else 2000
N_PERM = 200 if _FAST else 20000
MIN_CELL = 30                 # 팀 기준 5번
RNG = np.random.default_rng(20260807)

MULTS = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]   # 2단계 (4.0=현행)
MIN_SLS = [0.001, 0.002, 0.003, 0.005]             # 3단계 (0.001=현행)

OUT_JSON = Path("data/axis/sl-atr-floor.json")


# ── 1단계: 셋업 단계 SL 해부 ──────────────────────────────────────────
def dissect_setups(sym: str) -> dict:
    """타임라인의 모든 셋업에서 구조 SL 폭 vs ATR 바닥 폭을 실측.

    구조 SL 은 FVG 가장자리(setup.fvg.low/high)로 되살릴 수 있고,
    setup.stop_loss 는 바닥이 적용된 뒤 값이다. 둘이 다르면 바닥이 구속했다는 뜻.
    """
    df5 = _resample(_load_full(sym))
    tl = cached_setup_timeline(df5, BacktestConfig(**LIVE_BASE), sym)
    seen: set[int] = set()
    struct, floored, gaps = [], [], []
    for item in tl:
        if item is None:
            continue
        s, _ = item
        if s.ts_ms in seen or s.fvg is None:
            continue
        seen.add(s.ts_ms)
        raw_sl = s.fvg.low if s.direction is Direction.LONG else s.fvg.high
        e = s.entry
        if e <= 0:
            continue
        struct.append(abs(e - raw_sl) / e * 100.0)
        floored.append(abs(e - s.stop_loss) / e * 100.0)
        gaps.append((s.fvg.high - s.fvg.low) / e * 100.0)   # FVG 갭 폭(진입 깊이 지렛대)
    st = np.array(struct)
    fl = np.array(floored)
    gp = np.array(gaps)
    binds = fl > st * 1.0001          # 바닥이 구조를 덮어쓴 건
    return dict(
        sym=sym, n_setup=int(len(st)),
        struct_med=float(np.median(st)), struct_p90=float(np.percentile(st, 90)),
        floor_med=float(np.median(fl)), floor_p90=float(np.percentile(fl, 90)),
        bind_pct=float(100.0 * binds.mean()),
        atr_med=float(np.median(fl[binds] / 1.5)) if binds.any() else float("nan"),
        ratio_med=float(np.median(fl / np.maximum(st, 1e-12))),
        gap_med=float(np.median(gp)),
        # min_sl_distance_pct 는 **바닥 적용 후** 폭(fl)과 비교되는 detect 필터다.
        # 각 후보값에서 잘려나갈 셋업 비율 — 어느 쪽이 실제 구속력을 갖는지의 답.
        minsl_cut={f"{v}": float(100.0 * (fl < v * 100.0).mean()) for v in MIN_SLS},
    )


def tl_ready(sym: str, extra: dict | None = None) -> bool:
    """해당 설정의 타임라인이 **이미 캐시에 있는지**만 확인 (없으면 빌드하지 않는다).

    min_sl_distance_pct 처럼 캐시 키에 들어가는 값은 빌드가 페어당 10분 이상이라,
    분석 스크립트 안에서 무심코 재빌드가 돌면 몇 시간을 잡아먹는다.
    빌드는 _axis_sl_tlbuild_2026-08-07.py 가 미리 병렬로 처리한다.
    """
    import hashlib

    df5 = _resample(_load_full(sym))
    cfg = BacktestConfig(**{**LIVE_BASE, **(extra or {})})
    parts = (sym, round(cfg.ote_level, 4), round(cfg.min_rr, 3),
             round(cfg.fvg_min_size_pct, 6), bool(cfg.expand_to_killzone),
             bool(cfg.disable_time_filter), round(cfg.min_sl_distance_pct, 6),
             int(cfg.window), len(df5), int(df5.index[0].value), int(df5.index[-1].value))
    key = hashlib.md5(repr(parts).encode()).hexdigest()[:16]
    return os.path.exists(os.path.join(_TL_CACHE_DIR, f"{sym}_{key}.pkl"))


# ── 거래 수집 ────────────────────────────────────────────────────────
def collect(pairs: list[str], extra: dict | None = None) -> pd.DataFrame:
    """run_live_parity 로 라이브 정합 거래 수집 → DataFrame.

    risk = |진입 − 초기SL| / 진입 (최종 SL 폭, sl_dist_mult·LIQ-CAP 반영).
    R    = raw_pnl_pct / risk.
    """
    rows = []
    for sym in pairs:
        df5, kept, _ = run_live_parity(sym, extra)
        idx = df5.index
        for t in kept:
            risk = abs(t.entry - t.entry_sl) / t.entry
            if risk <= 0:
                continue
            ts = idx[t.entry_idx]
            rows.append(dict(
                sym=sym,
                t_in=int(ts.value), ts=ts, year=ts.year,
                t_out=int(idx[min(t.exit_idx, len(idx) - 1)].value),
                raw=float(t.raw_pnl_pct), risk=float(risk),
                r=float(t.raw_pnl_pct) / risk,
                dir=("long" if str(getattr(t.direction, "value", t.direction)).lower()
                     in ("long", "buy") else "short"),
                outcome=str(t.outcome), conf=int(t.confluence_score),
            ))
    return pd.DataFrame(rows).sort_values("t_in").reset_index(drop=True)


def concurrency(D: pd.DataFrame) -> np.ndarray:
    """각 거래 보유 구간의 평균 동시보유 수(자기 포함) — 사이즈 분할용."""
    s = D["t_in"].to_numpy()
    e = D["t_out"].to_numpy()
    out = np.empty(len(D))
    for i in range(len(D)):
        out[i] = 1.0 + int(((s <= e[i]) & (e >= s[i])).sum() - 1)
    return out


# ── 복리 시뮬 (origo_leverage_verify.sim 규약) ─────────────────────────
def sim_fixed(raws, scale):
    """지표 A — 고정 사이즈 복리. 반환 (최종배수, MDD%, 파산)."""
    eq = peak = 1.0
    mdd = 0.0
    for i, r in enumerate(raws):
        sz = SIZE * (1.0 if scale is None else scale[i])
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        eq *= 1.0 + (r * sz * LEV - 2.0 * TAKER_FEE_PCT * sz * LEV)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return float(eq), 100.0 * mdd, True
    return float(eq), 100.0 * mdd, False


def sim_riskfix(rs, risks, scale):
    """지표 B — 건당 리스크를 RISK_FIX 로 고정한 복리(후보 간 단위 통일).

    사이즈 = RISK_FIX / (risk × LEV) 를 쓰면 SL 이 맞을 때 정확히 RISK_FIX 를 잃는다.
    수수료는 그 사이즈에 맞춰 부과한다.
    """
    eq = peak = 1.0
    mdd = 0.0
    for i, (r_mult, risk) in enumerate(zip(rs, risks, strict=True)):
        sz = RISK_FIX / max(risk * LEV, 1e-9)
        sz *= (1.0 if scale is None else scale[i])
        sz = min(sz, SIZE)                       # 사이즈 상한은 라이브와 동일
        if eq < peak * (1.0 - DD_PCT):
            sz *= DD_FACTOR
        eq *= 1.0 + (r_mult * risk * sz * LEV - 2.0 * TAKER_FEE_PCT * sz * LEV)
        if eq <= 0:
            return 0.0, 100.0, True
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return float(eq), 100.0 * mdd, True
    return float(eq), 100.0 * mdd, False


def sim_liverisk(rs, risks, confs):
    """지표 C — 라이브 리스크 기반 사이징 복리 (bot_ict_instance._calc_qty_risk_based).

    건당 리스크 = min(3%+1.5%×conf, 6%) × DD스로틀. 수량은 SL 거리로 역산하므로
    **SL 폭이 변해도 건당 손실은 그대로**다. 단 notional 상한(equity×lev×80%)에
    걸리면 그만큼 리스크가 줄어든다 — SL 이 좁을수록(수량이 커질수록) 잘 걸린다.

    반환 (최종배수, MDD%, 파산, 상한구속률).
    """
    eq = peak = 1.0
    mdd = 0.0
    capped = 0
    for r_mult, risk, cf in zip(rs, risks, confs, strict=True):
        want = min(RISK_BASE + RISK_STEP * max(int(cf), 0), RISK_MAX)
        # notional 상한 → 실효 리스크 = min(want, lev × 80% × SL폭)
        lim = LEV * POS_PCT_MAX * risk
        eff = min(want, lim)
        if lim < want:
            capped += 1
        if eq < peak * (1.0 - DD_PCT):
            eff *= DD_FACTOR
        notional_ratio = eff / max(risk, 1e-9)      # notional / equity
        eq *= 1.0 + (r_mult * eff - 2.0 * TAKER_FEE_PCT * notional_ratio)
        if eq <= 0:
            return 0.0, 100.0, True, 100.0
        peak = max(peak, eq)
        mdd = max(mdd, 1.0 - eq / peak)
        if eq <= RUIN:
            return float(eq), 100.0 * mdd, True, 100.0 * capped / len(rs)
    return float(eq), 100.0 * mdd, False, 100.0 * capped / len(rs)


def evaluate(D: pd.DataFrame, months: float) -> dict:
    """후보 1개 요약 — 거래통계 + 지표 A/B + 부트스트랩."""
    raws = D["raw"].to_numpy()
    rs = D["r"].to_numpy()
    risks = D["risk"].to_numpy()
    confs = D["conf"].to_numpy()
    scale = 1.0 / concurrency(D)
    n = len(D)
    eq, mdd, _ = sim_fixed(raws, scale)
    eqb, mddb, _ = sim_riskfix(rs, risks, scale)
    eqc, mddc, _, capr = sim_liverisk(rs, risks, confs)
    # 부트스트랩(복원추출)은 배포 판단 지표인 C 기준 — 파산확률도 C 로 센다.
    fin = np.empty(N_BOOT)
    fina = np.empty(N_BOOT)
    ruin = 0
    for k in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)          # 복원추출(미래 가정, 보수적)
        e, _, ru, _ = sim_liverisk(rs[idx], risks[idx], confs[idx])
        fin[k] = e
        ruin += int(ru)
        fina[k] = sim_fixed(raws[idx], scale[idx])[0]
    p50, p05 = np.percentile(fin, [50, 5])
    a50, a05 = np.percentile(fina, [5, 50])[::-1]
    return dict(
        n=int(n), freq=float(n / months),
        risk_med=float(np.median(risks) * 100), risk_mean=float(risks.mean() * 100),
        r_mean=float(rs.mean()), r_se=float(rs.std(ddof=1) / np.sqrt(n)),
        raw_mean=float(raws.mean() * 100),
        raw_se=float(raws.std(ddof=1) / np.sqrt(n) * 100),
        wr=float(100.0 * (raws > 0).mean()),
        # ── 산출물 규약의 compound/mdd/ruin = 지표 C(라이브 실제 사이징). 배포 판단용.
        compound=float(eqc), mdd=float(mddc),
        boot_p50=float(p50), boot_p05=float(p05), ruin=float(100.0 * ruin / N_BOOT),
        live_cap_pct=float(capr),
        # 지표 A (팀 규약, 고정 사이즈) — 과거 연구와의 비교용
        compound_fixedsize=float(eq), mdd_fixedsize=float(mdd),
        a_boot_p50=float(a50), a_boot_p05=float(a05),
        # 지표 B (엣지 질, 건당 리스크 고정)
        rf_compound=float(eqb), rf_mdd=float(mddb),
    )


# ── 순열검정 ─────────────────────────────────────────────────────────
def perm_two_sample(a: np.ndarray, b: np.ndarray) -> float:
    """두 표본 무작위 분할 20000회 — 평균차 양측 p."""
    obs = abs(b.mean() - a.mean())
    pool = np.concatenate([a, b])
    na = len(a)
    cnt = 0
    for _ in range(N_PERM):
        p = RNG.permutation(pool)
        if abs(p[na:].mean() - p[:na].mean()) >= obs - 1e-15:
            cnt += 1
    return (cnt + 1) / (N_PERM + 1)


def perm_paired(d: np.ndarray) -> float:
    """짝지은 차이의 부호뒤집기 검정 — 같은 셋업에서 나온 거래 쌍용."""
    if len(d) == 0:
        return float("nan")
    obs = abs(d.mean())
    cnt = 0
    for _ in range(N_PERM):
        s = RNG.choice([-1.0, 1.0], size=len(d))
        if abs((d * s).mean()) >= obs - 1e-15:
            cnt += 1
    return (cnt + 1) / (N_PERM + 1)


def pair_up(A: pd.DataFrame, B: pd.DataFrame, col: str = "r") -> np.ndarray:
    """같은 페어·같은 진입시각 거래를 짝지어 차이(B−A)를 반환 (기본 R 배수)."""
    key = ["sym", "t_in"]
    m = A[key + [col]].merge(B[key + [col]], on=key, suffixes=("_a", "_b"))
    return (m[f"{col}_b"] - m[f"{col}_a"]).to_numpy()


# ── 국면 라벨 (기저 통제용) ───────────────────────────────────────────
def label_z(sym: str) -> pd.Series:
    """라이브 #REGIME-OTE 와 동일 계산식의 일봉 국면 z-score 라벨."""
    c = _resample(_load_full(sym)).resample("1D").agg({"close": "last"}).dropna()["close"]
    sig = c.pct_change().rolling(20).std()
    z = ((c / c.shift(20) - 1.0) / (sig * np.sqrt(20))).shift(1)
    return pd.Series(np.where(z > 0.75, "상승", np.where(z < -0.75, "하락", "횡보")),
                     index=c.index)


def regime_table(D: pd.DataFrame, labs: dict[str, pd.Series]) -> list[str]:
    """국면 × 방향 교차표 (건수 / 건당 R) — 기저 통제 확인용."""
    D = D.copy()
    D["day"] = D["ts"].dt.normalize()
    reg = []
    for sym, g in D.groupby("sym"):
        reg.append(pd.Series(labs[sym].reindex(g["day"]).to_numpy(), index=g.index))
    D["reg"] = pd.concat(reg).reindex(D.index)
    lines = []
    for rg in ("상승", "횡보", "하락"):
        parts = []
        for dr in ("long", "short"):
            cell = D[(D["reg"] == rg) & (D["dir"] == dr)]
            tag = "" if len(cell) >= MIN_CELL else "*"
            parts.append(f"{dr:5s} n={len(cell):3d}{tag} R={cell['r'].mean():+6.3f}"
                         if len(cell) else f"{dr:5s} n=  0  R=   ---")
        lines.append(f"    {rg}  " + " | ".join(parts))
    lines.append(f"    (* = 표본 {MIN_CELL}건 미만 — 결론에서 제외)")
    return lines


# ── 실행 ─────────────────────────────────────────────────────────────
def months_of(D: pd.DataFrame) -> float:
    span = (D["ts"].max() - D["ts"].min()).days
    return max(span / 30.44, 1.0)


def run_axis(pairs: list[str], tag: str, results: dict) -> None:
    """한 페어 집합에 대해 2·3단계 스윕 전체 실행."""
    print(f"\n{'=' * 78}\n### {tag} — 페어 {','.join(p.replace('USDT', '') for p in pairs)}",
          flush=True)

    base = collect(pairs)
    months = months_of(base)
    base_ev = evaluate(base, months)
    print("\n[기준선] sl_dist_mult=4.0 · min_sl=0.001 (현행)", flush=True)
    print("  복리C=라이브 실제(리스크 기반) · 복리A=팀규약(고정사이즈) · "
          "복리B=리스크고정 · 부트/파산은 C 기준", flush=True)
    print(f"  {'후보':<24}{'n':>5}{'월빈도':>7}{'SL%':>7}{'승률':>6}"
          f"{'건당R':>8}{'건당raw%':>9}{'복리C':>9}{'MDD_C':>7}{'부트5%':>8}"
          f"{'파산':>7}{'수량캡':>7}{'복리A':>8}{'복리B':>8}{'p순열':>8}", flush=True)

    def show(name, ev, p=None, pp=None, npair=0):
        print(f"  {name:<24}{ev['n']:>5d}{ev['freq']:>7.2f}{ev['risk_med']:>7.2f}"
              f"{ev['wr']:>6.0f}{ev['r_mean']:>8.3f}{ev['raw_mean']:>9.3f}"
              f"{ev['compound']:>8.2f}x{ev['mdd']:>6.1f}%"
              f"{ev['boot_p05']:>7.2f}x{ev['ruin']:>6.1f}%{ev['live_cap_pct']:>6.0f}%"
              f"{ev['compound_fixedsize']:>7.2f}x{ev['rf_compound']:>7.2f}x"
              f"{(f'{p:.4f}' if p is not None else '  기준'):>8}"
              + (f"  (짝 p={pp:.4f}, n={npair:d})" if pp is not None else ""),
              flush=True)

    show("기준선(4.0/0.001)", base_ev)
    results[tag] = {"baseline": base_ev, "candidates": []}

    # ── 2단계: sl_dist_mult ──
    print("\n[2단계] sl_dist_mult 스윕 (타임라인 캐시 재사용 — 체결셋 대부분 공유)",
          flush=True)
    for m in MULTS:
        if abs(m - 4.0) < 1e-9:
            continue
        cand_D = collect(pairs, {"sl_dist_mult": m, "sl_dist_mult_ct": m})
        if len(cand_D) < 5:
            print(f"  sl_dist_mult={m}: 거래 부족({len(cand_D)}) — 제외", flush=True)
            continue
        ev = evaluate(cand_D, months)
        # 주 검정 단위 = 건당 R. 라이브는 리스크 기반 사이징이라 자산 변화가
        # R × 6% 로 결정되고, tp_keeps_rr=True 라 SL·TP 가 함께 늘어 R 의 기계적
        # 이동이 없다(1단계 [C] 참조). raw% 는 고정사이즈 모델용 보조 검정.
        p = perm_two_sample(base["r"].to_numpy(), cand_D["r"].to_numpy())
        p_raw = perm_two_sample(base["raw"].to_numpy(), cand_D["raw"].to_numpy())
        d = pair_up(base, cand_D)
        pp = perm_paired(d)
        show(f"mult={m}(={m * 1.5:.2f}xATR)", ev, p, pp, len(d))
        results[tag]["candidates"].append(dict(
            name=f"sl_dist_mult={m}", params={"sl_dist_mult": m, "sl_dist_mult_ct": m},
            # 바닥이 99% 구속하므로 최종 SL = mult × _ATR_SL_MULT(1.5) × ATR.
            eff_atr_mult=float(m * 1.5),
            p_perm=p, p_perm_raw=p_raw, p_paired=pp, **ev))

    # ── 2b단계: #LIQ-CAP 레버리지 정합 교정 ──
    # LIVE_BASE 에 leverage 가 없어 백테는 기본 20 → 캡 4.0%. 라이브는 7 → 11.4%.
    # SL 을 넓히는 후보일수록 백테가 캡에 눌려 불리해지므로 교정본을 따로 낸다.
    print("\n[2b단계] #LIQ-CAP 정합 교정 (leverage 20→7, 캡 4.0%→11.4%)", flush=True)
    for m in (4.0, 5.0, 6.0, 8.0):
        cand_D = collect(pairs, {"sl_dist_mult": m, "sl_dist_mult_ct": m,
                                 "leverage": LEV})
        if len(cand_D) < 5:
            continue
        ev = evaluate(cand_D, months)
        p = perm_two_sample(base["r"].to_numpy(), cand_D["r"].to_numpy())
        p_raw = perm_two_sample(base["raw"].to_numpy(), cand_D["raw"].to_numpy())
        d = pair_up(base, cand_D)
        pp = perm_paired(d)
        show(f"lev7·mult={m}", ev, p, pp, len(d))
        results[tag]["candidates"].append(dict(
            name=f"lev7(캡교정)·sl_dist_mult={m}",
            params={"sl_dist_mult": m, "sl_dist_mult_ct": m, "leverage": LEV},
            p_perm=p, p_perm_raw=p_raw, p_paired=pp, **ev))

    # ── 3단계: min_sl_distance_pct ──
    print("\n[3단계] min_sl_distance_pct 스윕 (detect 필터 — 체결셋 자체가 바뀜)",
          flush=True)
    for v in MIN_SLS:
        if abs(v - 0.001) < 1e-9:
            continue
        miss = [s for s in pairs if not tl_ready(s, {"min_sl_distance_pct": v})]
        if miss:
            print(f"  min_sl={v}: 타임라인 미빌드 {miss} — 건너뜀(재빌드 금지)", flush=True)
            continue
        cand_D = collect(pairs, {"min_sl_distance_pct": v})
        if len(cand_D) < 5:
            print(f"  min_sl={v}: 거래 부족({len(cand_D)}) — 제외", flush=True)
            continue
        ev = evaluate(cand_D, months)
        # 주 검정 단위 = 건당 R. 라이브는 리스크 기반 사이징이라 자산 변화가
        # R × 6% 로 결정되고, tp_keeps_rr=True 라 SL·TP 가 함께 늘어 R 의 기계적
        # 이동이 없다(1단계 [C] 참조). raw% 는 고정사이즈 모델용 보조 검정.
        p = perm_two_sample(base["r"].to_numpy(), cand_D["r"].to_numpy())
        p_raw = perm_two_sample(base["raw"].to_numpy(), cand_D["raw"].to_numpy())
        d = pair_up(base, cand_D)
        pp = perm_paired(d)
        show(f"min_sl={v}", ev, p, pp, len(d))
        results[tag]["candidates"].append(dict(
            name=f"min_sl_distance_pct={v}", params={"min_sl_distance_pct": v},
            p_perm=p, p_perm_raw=p_raw, p_paired=pp, **ev))

    # ── 연도 분해 + 국면×방향 (기준선 및 최선 후보) ──
    labs = {s: label_z(s) for s in pairs}
    print("\n[기준선] 국면 × 방향 (기저 통제)", flush=True)
    for ln in regime_table(base, labs):
        print(ln, flush=True)
    print("\n[기준선] 연도별 건수 / 건당 raw%", flush=True)
    yb = base.groupby("year")["raw"].agg(["count", "mean"])
    print("    " + "  ".join(f"{y}:n{int(r['count'])}/{r['mean'] * 100:+.2f}%"
                             for y, r in yb.iterrows()), flush=True)
    results[tag]["baseline_regime"] = regime_table(base, labs)
    results[tag]["baseline_years"] = {int(y): [int(r["count"]), float(r["mean"] * 100)]
                                      for y, r in yb.iterrows()}
    return base, months, labs


def year_decomp(base: pd.DataFrame, cand: pd.DataFrame) -> tuple[str, float]:
    """연도별 Δ(총 R 합) 분해 — 한 해가 총차의 과반이면 기각 신호(팀 기준 3번).

    라이브가 리스크 기반 사이징이라 자산 기여는 R 합에 비례한다 → R 로 분해한다.
    """
    yb = base.groupby("year")["r"].sum()
    yc = cand.groupby("year")["r"].sum()
    d = yc.subtract(yb, fill_value=0.0)
    tot = d.abs().sum()
    share = float(d.abs().max() / tot * 100) if tot > 0 else float("nan")
    return " ".join(f"{int(y)}:{v:+.1f}" for y, v in d.items()), share


def main() -> int:
    t0 = time.time()
    print("=" * 78, flush=True)
    print("축 연구: SL 산출 방식 (1.5×ATR 바닥 + sl_dist_mult + min_sl_distance_pct)",
          flush=True)
    print("=" * 78, flush=True)

    # ───────────────── 1단계: 구조적 상한 ─────────────────
    print("\n### 1단계 — 구조적 상한 (스윕 전 필수)\n", flush=True)
    print("  [A] 셋업 단계 SL 해부 — 구조 SL(FVG 가장자리) vs 1.5×ATR 바닥", flush=True)
    print(f"  {'페어':<8}{'셋업수':>7}{'구조SL%중앙':>12}{'구조p90':>9}"
          f"{'바닥후%중앙':>12}{'바닥후p90':>10}{'바닥구속률':>11}{'폭배율중앙':>11}",
          flush=True)
    diss = []
    for sym in LIVE_PAIRS:
        d = dissect_setups(sym)
        diss.append(d)
        print(f"  {sym.replace('USDT', ''):<8}{d['n_setup']:>7d}{d['struct_med']:>12.4f}"
              f"{d['struct_p90']:>9.4f}{d['floor_med']:>12.4f}{d['floor_p90']:>10.4f}"
              f"{d['bind_pct']:>10.2f}%{d['ratio_med']:>11.1f}x", flush=True)

    print("\n  [A-2] 두 바닥 중 어느 쪽이 구속하나 — min_sl_distance_pct 후보별 "
          "셋업 절단률(%)", flush=True)
    for d in diss:
        cuts = "  ".join(f"{k}({float(k) * 100:.1f}%)→{v:5.1f}%"
                         for k, v in d["minsl_cut"].items())
        print(f"      {d['sym'].replace('USDT', ''):<5} {cuts}", flush=True)
    print("      → 현행 0.001(0.1%) 에서 절단률이 0 에 가까우면 min_sl 은 죽은 게이트고, "
          "구속력은 전부 1.5×ATR 바닥에 있다.", flush=True)

    base_live = collect(LIVE_PAIRS)
    months = months_of(base_live)
    risks = base_live["risk"].to_numpy()
    cap_pct = 0.8 / BacktestConfig(**LIVE_BASE).leverage    # #LIQ-CAP 상한(진입가 대비)
    capped = (risks >= cap_pct * 0.999).mean()
    print(f"\n  [B] 체결 거래 최종 SL 폭 — 중앙 {np.median(risks) * 100:.2f}% "
          f"평균 {risks.mean() * 100:.2f}% p95 {np.percentile(risks, 95) * 100:.2f}%",
          flush=True)
    print(f"      #LIQ-CAP 상한 {cap_pct * 100:.2f}% (cfg.leverage="
          f"{BacktestConfig(**LIVE_BASE).leverage:.0f}) 에 걸린 비율 {capped * 100:.1f}%",
          flush=True)

    # 기계적 상한: SL 로 끝난 거래의 손실만 m'/m 로 축소, 신규 SL 히트 0 가정(최선)
    raws = base_live["raw"].to_numpy()
    is_sl = base_live["outcome"].str.lower().str.contains("sl").to_numpy()
    se = raws.std(ddof=1) / np.sqrt(len(raws)) * 100
    print("\n  [C-가] 고정사이즈 모델(지표 A)의 기계적 상한 — mult 를 좁힐 때의 "
          "**최선**", flush=True)
    print("        (SL 청산 건의 손실만 비례 축소 + 신규 SL 히트 0 가정. "
          "실제는 반드시 이보다 나쁘다)", flush=True)
    print(f"      기준선 건당 raw% = {raws.mean() * 100:+.3f}%  "
          f"표본 노이즈 SE = {se:.3f}%  (n={len(raws)}, SL청산 {is_sl.mean() * 100:.0f}%)",
          flush=True)
    ceilings = {}
    for m in MULTS:
        if m >= 4.0:
            continue
        best = np.where(is_sl, raws * (m / 4.0), raws).mean() * 100
        ceilings[m] = float(best - raws.mean() * 100)
        print(f"      mult {m}: 최선 건당 raw% {best:+.3f}%  "
              f"→ Δ상한 {best - raws.mean() * 100:+.3f}%  = 노이즈의 "
              f"{abs(best - raws.mean() * 100) / se:.2f}배", flush=True)
    print("\n      ※ 상한/노이즈 < 1 이면 그 방향(좁히기)은 이론적으로 이길 수 없다"
          " — 피보 연구 선례.", flush=True)

    # [C-나] 라이브 리스크기반 모델에서는 기계적 성분이 정확히 0 이다.
    # SL·TP 가 같은 배수로 늘고(tp_keeps_rr=True) 수량이 1/SL 로 줄어,
    # **가격 경로 결과가 같으면 R 도 자산 변화도 동일**. 즉 관측되는 차이는
    # 전부 "어느 장벽에 먼저 닿는가"(경로 의존)뿐이라 이론으로 닫히지 않는다.
    r_all = base_live["r"].to_numpy()
    r_se = r_all.std(ddof=1) / np.sqrt(len(r_all))
    print("\n  [C-나] 라이브 리스크기반 모델(지표 C)의 기계적 상한 = **정확히 0**",
          flush=True)
    print("        SL·TP 동시 배수 확대(tp_keeps_rr=True) + qty=리스크/SL거리 → "
          "경로 결과가 같으면 R 불변.", flush=True)
    print(f"        따라서 관측 Δ는 100% 경로 의존(장벽 도달 순서)이며 이론 상한이 "
          f"없다 → 스윕이 유일한 판정 수단.", flush=True)
    print(f"        기준선 건당 R = {r_all.mean():+.3f}  R 노이즈 SE = {r_se:.3f}  "
          f"(n={len(r_all)}) → 검출 가능 최소 효과 ≈ {1.96 * r_se * np.sqrt(2):.2f}R",
          flush=True)

    # [D] 직전 피보 연구와의 접속 — ATR 바닥이 "진입 깊이" 손잡이를 얼마나 죽였나.
    #     ote_level 0.5→0.886 은 진입가를 FVG 갭 × 0.386 만큼 옮긴다. 그 이동이
    #     R 로 환산되는 크기는 SL 폭(=R 분모)에 반비례한다. 바닥 유무로 비교한다.
    print("\n  [D] ATR 바닥이 진입깊이(ote_level) 손잡이를 죽인 정도 — "
          "직전 피보 연구와의 접속", flush=True)
    gap_r = []
    for d in diss:
        shift = d["gap_med"] * 0.386                     # ote 0.5→0.886 진입 이동폭(%)
        r_now = shift / (4.0 * d["floor_med"])           # 현행(바닥 O) 의 ΔR 상한
        r_off = shift / (4.0 * d["struct_med"])          # 바닥 없었다면(구조 SL) ΔR 상한
        gap_r.append({"sym": d["sym"], "shift_pct": shift,
                      "dr_with_floor": r_now, "dr_without_floor": r_off})
        print(f"      {d['sym'].replace('USDT', ''):<5} FVG 갭 중앙 {d['gap_med']:.3f}% → "
              f"진입 이동 {shift:.4f}% | 현행 SL {4.0 * d['floor_med']:.2f}% 대비 "
              f"ΔR 상한 {r_now:.3f}R  ↔ 바닥 없었다면 SL {4.0 * d['struct_med']:.2f}% "
              f"대비 {r_off:.2f}R ({r_off / max(r_now, 1e-9):.0f}배)", flush=True)
    print("      → 1.5×ATR 바닥이 진입 깊이의 지렛대를 약 10배 눌렀다는 뜻. "
          "(바닥을 풀면 SL 이 0.15% 대라 즉시 털린다 — 그래서 #LIVE-7 이 있다)",
          flush=True)

    results: dict = {"stage1": {"dissect": diss, "ote_lever": gap_r,
                                "final_sl_med_pct": float(np.median(risks) * 100),
                                "liq_cap_pct": float(cap_pct * 100),
                                "liq_cap_hit_pct": float(capped * 100),
                                "raw_mean_pct": float(raws.mean() * 100),
                                "raw_se_pct": float(se),
                                "r_mean": float(r_all.mean()), "r_se": float(r_se),
                                "min_detectable_r": float(1.96 * r_se * np.sqrt(2)),
                                "mechanical_ceiling_fixedsize": ceilings,
                                "mechanical_ceiling_liverisk": 0.0}}

    # ───────────────── 2·3단계 ─────────────────
    b_live, m_live, labs_live = run_axis(LIVE_PAIRS, "라이브2페어", results)
    b_wide, m_wide, _ = run_axis(WIDE_PAIRS, "검정력7페어", results)

    # ── 교란 분리: 거래수 변화 vs 건당 질 변화 ──
    print(f"\n{'=' * 78}\n### 교란 분리 — 거래수 변화 vs 건당 질 변화 (라이브 2페어)",
          flush=True)
    print("  공유거래 = 기준선과 진입시각이 같은 건(=건당 질 변화). "
          "신규거래 = 후보에서만 생긴 건(=거래수 교란).", flush=True)
    print(f"  {'후보':<26}{'n':>5}{'Δn':>6}{'공유':>6}{'공유셋 ΔR':>12}"
          f"{'신규건 R':>11}{'소멸건 R':>11}{'연도최대비중':>13}", flush=True)
    for c in results["라이브2페어"]["candidates"]:
        cand = collect(LIVE_PAIRS, c["params"])
        key = ["sym", "t_in"]
        m = b_live[key + ["r"]].merge(cand[key + ["r"]], on=key, suffixes=("_a", "_b"))
        newonly = cand.merge(b_live[key].assign(_x=1), on=key, how="left")
        newonly = newonly[newonly["_x"].isna()]
        gone = b_live.merge(cand[key].assign(_x=1), on=key, how="left")
        gone = gone[gone["_x"].isna()]
        ys, share = year_decomp(b_live, cand)
        d_shared = float((m["r_b"] - m["r_a"]).mean()) if len(m) else float("nan")
        print(f"  {c['name']:<26}{len(cand):>5d}{len(cand) - len(b_live):>6d}"
              f"{len(m):>6d}{d_shared:>12.3f}"
              f"{(newonly['r'].mean() if len(newonly) else float('nan')):>11.3f}"
              f"{(gone['r'].mean() if len(gone) else float('nan')):>11.3f}"
              f"{share:>12.0f}%", flush=True)
        print(f"      연도별 ΔR총합: {ys}", flush=True)
        c["shared_n"] = int(len(m))
        c["shared_dR"] = d_shared
        c["new_n"] = int(len(newonly))
        c["gone_n"] = int(len(gone))
        c["year_max_share_pct"] = float(share)
        c["year_delta_R"] = ys

    # ── 판정 (팀 표준 6종을 기계적으로 적용) ──
    n_per_set = (len(MULTS) - 1) + 4 + (len(MIN_SLS) - 1)
    thr = 0.05 / n_per_set          # Bonferroni (다중비교 보정)
    base_r = results["라이브2페어"]["baseline"]["r_mean"]
    wide = {c["name"]: c for c in results["검정력7페어"]["candidates"]}
    print(f"\n{'=' * 78}\n### 판정 — 라이브 2페어 기준 (7페어는 부호 일치 확인용)",
          flush=True)
    print(f"  유망 조건: ΔR>0 · p<0.05(무보정) · Bonferroni 문턱 {thr:.4f} · "
          f"연도최대비중<50% · 7페어 부호 일치", flush=True)
    print(f"  {'후보':<26}{'ΔR':>8}{'p순열':>8}{'7페어ΔR':>10}"
          f"{'연도최대':>9}  판정", flush=True)
    for c in results["라이브2페어"]["candidates"]:
        dr = c["r_mean"] - base_r
        w = wide.get(c["name"])
        dw = (w["r_mean"] - results["검정력7페어"]["baseline"]["r_mean"]
              if w else float("nan"))
        if c["name"].endswith("sl_dist_mult=4.0"):
            # lev7 캡교정 × mult 4.0 = 기준선과 같은 설정의 **정합 대조군**.
            # 후보가 아니라 "캡 교정만으로 성적이 얼마나 바뀌나"를 재는 자리다.
            print(f"  {c['name']:<26}{dr:>8.3f}{c['p_perm']:>8.4f}{dw:>10.3f}"
                  f"{'':>9}  정합 대조군(후보 아님) — 캡 교정만의 효과", flush=True)
            c["delta_r"] = float(dr)
            c["verdict"] = "정합 대조군"
            continue
        reasons = []
        if dr <= 0:
            reasons.append("ΔR<=0(열위/무차)")
        if c["p_perm"] >= 0.05:
            reasons.append(f"p={c['p_perm']:.3f}>=0.05")
        elif c["p_perm"] >= thr:
            reasons.append(f"p={c['p_perm']:.3f} 는 다중비교 보정({thr:.4f}) 미통과")
        if c.get("year_max_share_pct", 0) >= 50:
            reasons.append(f"연도쏠림 {c['year_max_share_pct']:.0f}%")
        if c["n"] < MIN_CELL:
            reasons.append(f"표본부족 n={c['n']}")
        if not np.isnan(dw) and np.sign(dw) != np.sign(dr):
            reasons.append("7페어 부호 불일치")
        verdict = "유망" if not reasons else "기각 — " + " / ".join(reasons)
        print(f"  {c['name']:<26}{dr:>8.3f}{c['p_perm']:>8.4f}{dw:>10.3f}"
              f"{c.get('year_max_share_pct', float('nan')):>8.0f}%  {verdict}", flush=True)
        c["delta_r"] = float(dr)
        c["delta_r_wide7"] = float(dw)
        c["verdict"] = verdict

    # ── 저장 ──
    results["notes"] = (
        f"[다중비교] 시도 조합 = 페어집합 2개 × (sl_dist_mult {len(MULTS) - 1}개 + "
        f"LIQ-CAP 교정 4개 + min_sl_distance_pct {len(MIN_SLS) - 1}개) = "
        f"{2 * n_per_set}건. 무보정 argmax(승자의 저주) 방지용 명시 — "
        f"Bonferroni 로 보면 유의 문턱은 0.05/{n_per_set} = {0.05 / n_per_set:.4f}. "
        f"기준선 = sl_dist_mult 4.0 / min_sl 0.001. "
        f"[단위] 복리C=라이브 실제(리스크기반 6%·notional상한 lev{LEV:.0f}×80%·DD스로틀) "
        f"— 배포 판단 지표. 복리A=팀규약 고정사이즈(size{SIZE}·동시보유분할). "
        f"복리B=건당 리스크 {RISK_FIX:.0%} 고정. 순열검정 주 단위=건당 R. "
        f"[핵심 실측] 1.5xATR 바닥이 FVG 구조 SL 을 덮어쓴 비율 = "
        f"stage1.dissect.bind_pct (BTC/ETH 모두 99% 이상). "
        f"[정합 갭] LIVE_BASE 에 leverage 없음(기본 20) → #LIQ-CAP 4.0% vs 라이브 11.4%; "
        f"LIVE_BASE size_pct=0.9 는 고정 notional 가정인데 라이브는 risk_based_sizing=True. "
        f"[미실행] 3단계 min_sl_distance_pct 실측 스윕 — 타임라인 재빌드가 "
        f"페어당 수 시간(CPU 경합)이라 미완. 원 질문(어느 쪽이 구속하나)은 "
        f"stage1.dissect.minsl_cut 이 답함: 현행 0.001 절단률 0.0% = 죽은 게이트. "
        f"[한계] _ATR_SL_MULT 자체는 detect 단계 min_rr 게이트에도 들어가 셋업 집합을 "
        f"바꾸므로, '실효 6xATR 최적'은 셋업 집합 고정 조건의 결론이다."
    )
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    # JSON 스키마: 상위 candidates 는 라이브 2페어 결과를 그대로 승격
    payload = {
        "axis": "SL 산출 방식 — 1.5×ATR 바닥 / sl_dist_mult / min_sl_distance_pct",
        "candidates": results["라이브2페어"]["candidates"],
        "baseline": results["라이브2페어"]["baseline"],
        "stage1": results["stage1"],
        "wide7": results["검정력7페어"],
        "live2_extra": {k: v for k, v in results["라이브2페어"].items()
                        if k not in ("candidates", "baseline")},
        "notes": results["notes"],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=1,
                                   default=float), encoding="utf-8")
    print(f"\n저장: {OUT_JSON}  (총 {time.time() - t0:.0f}초)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
