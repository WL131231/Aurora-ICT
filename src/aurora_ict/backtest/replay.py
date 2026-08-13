"""ICT 봇 백테스트 — 슬라이딩 윈도우 진입 시뮬 + PnL 집계.

1차 범위 (#BACKTEST 2026-06-06):
    detect_silver_bullet_setups 를 과거 1m OHLCV 에 슬라이딩 적용 → 가장 최근
    setup 을 stale/게이트(min_rr·min_confluence·high_rr_bypass) 통과 시 진입 →
    다음 봉들에서 SL/TP 도달 시뮬 → 슬리피지·수수료 반영 net PnL 집계.

    confluence_score 는 silver_bullet 내부(OB/macro/sweep/bias)만 반영. bot 레벨
    boost (HTF FVG / SMT / CISD) 는 2차에서 합산 예정 — 1차는 baseline.

가정/단순화:
    - 동시 포지션 1개 (청산 후 다음 진입).
    - 진입 = setup.entry(계획 limit) 즉시 체결 가정 + 진입 슬리피지 (불리 방향).
    - SL/TP = 진입 후 봉의 high/low 도달 판정. 같은 봉 동시 도달 시 SL 우선(보수).
    - 미청산 시 마지막 봉 close 강제 청산("eod").
    - size_pct 고정 (상대 비교용). risk-based sizing 은 2차.

담당: 지영민 (#BACKTEST 2026-06-06)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct
from aurora_ict.indicators.cbdr import (
    CBDRBiasState,
    classify_price_vs_cbdr,
    detect_cbdr_boxes,
)
from aurora_ict.indicators.cisd import CisdType, detect_cisd
from aurora_ict.indicators.daily_bias import compute_daily_bias
from aurora_ict.indicators.dol import compute_dol

# 2026-08-08 라이브 정합 이식용 — _htf_fvg_records / _dol_counter_delta 가 쓴다.
# mark_filled_and_invalidated 는 전체 히스토리 1회 스캔으로 등가 재현(_htf_fvg_records).
from aurora_ict.indicators.fvg import FVGType, detect_fvgs
from aurora_ict.indicators.liquidity import detect_liquidity_sweeps
from aurora_ict.indicators.smt import SmtType, detect_smt_divergence
from aurora_ict.indicators.structure import TrendDirection
from aurora_ict.indicators.swing_points import SwingType, detect_swing_points

# htf_fvg_map: TF 가중치(5m=1 … 1w=40). _apply_htf_supporting_boost 와 동일 상수.
from aurora_ict.strategy.htf_fvg_map import TF_WEIGHT
from aurora_ict.strategy.silver_bullet import (
    Direction,
    build_extra_source_setups,
    detect_silver_bullet_setups,
)
from aurora_ict.timing.power_of_3 import AmdPhase, amd_phase


@dataclass(slots=True)
class BacktestConfig:
    """백테스트 파라미터 — sweep 대상 + 시뮬/비용 설정.

    게이트/검출 파라미터는 봇 settings 와 같은 의미. 시뮬 파라미터는 백테스트 전용.
    """

    # --- 검출/게이트 (sweep 대상) ---
    min_rr: float = 2.0
    min_confluence: int = 2
    high_rr_bypass_min_rr: float = 0.0  # 라이브 settings 일치 (2026-06-06 3.0→0.0 비활성)
    setup_stale_bars: int = 120
    fvg_min_size_pct: float = 0.0006
    min_sl_distance_pct: float = 0.001
    disable_time_filter: bool = True  # 백테스트 기본 24h (시간 필터 영향 분리)
    expand_to_killzone: bool = False
    # --- bot 레벨 boost 반영 (2차-②) — confluence_score 에 가산 후 게이트 ---
    apply_cisd: bool = False  # CISD 방향 일치 시 +1 (단일 심볼)
    apply_smt: bool = False   # SMT divergence 방향 일치 시 +1 (corr_df 필요)
    # --- 시뮬/비용 ---
    window: int = 500  # 슬라이딩 lookback 봉 수 (detect 입력 길이)
    entry_ttl_bars: int = 5  # limit 체결 대기 봉 수 (entry_limit_ttl_sec 300s / 60s @1m)
    leverage: float = 20.0
    size_pct: float = 0.3  # 시드 대비 포지션 (고정 — 상대 비교용)
    # --- #LIVE-SIZING 2026-08-09: 라이브 진입 크기 계산 재현 (기록 전용) ---
    # 손익 계산은 바꾸지 않는다. Trade 에 입력값만 남기고 실제 복리는 포트폴리오
    # 시뮬이 계산한다(페어가 자산을 공유하므로 페어별 replay 에서는 표현 불가).
    # 라이브 기본값 출처: settings.py risk_per_trade_* / position_pct_max,
    # smart_size 는 bot_ict_instance._set_smart_size.
    # #KZ-WIDE 2026-08-09: False 면 미장(NYSE) 제약 없이 킬존/매크로/SB 전체.
    # 2026-05-28 에 거래 한 건(#6, 뉴욕 03:02 런던 킬존, -283)을 보고 미장 밖을
    # 막은 결정을 백테로 처음 검증하기 위한 스위치. 셋업 집합이 바뀌므로
    # 타임라인 캐시 키에도 반드시 포함해야 한다(bt_par.cached_setup_timeline).
    nyse_gate: bool = True
    # #MMBM-BT 2026-08-10: 2번째 진입모델. 라이브는 SB 셋업이 없거나 게이트를 못
    # 넘은 봉에서만 시도하고, **confluence 게이트를 전면 우회**한다(자체 조건으로
    # 검증됨). 지금까지 백테는 이 경로가 없어 라이브의 일부만 보고 있었다.
    # 라이브 출처: bot_ict_instance.py step() 1264-1295.
    # #FVG-REUSE 2026-08-10: 창당 1회 제한(2026-05-12 첫 커밋, 근거 기록 없음)을
    # 끄고 대신 같은 FVG 재사용 횟수를 제한한다. 빈도가 최대 약점인데 검증되지
    # 않은 상한이 걸려 있었다. max_per_fvg=0 이면 무제한.
    window_once: bool = True
    max_per_fvg: int = 0
    # [08-10 연구] 정통 PD-array 추가 소스 (ifvg/breaker/unicorn/bpr/vacuum).
    # 기본 빈 튜플 = 기존 동작 불변. 켜면 셋업 집합이 달라져 캐시도 분리된다.
    # #DROP-TURTLE 2026-08-11: 프로덕션 배포와 동기화. 라이브 기본 False.
    turtle_soup_enabled: bool = True
    research_sources: tuple[str, ...] = ()
    mmbm_enabled: bool = False
    smart_size_enabled: bool = True
    risk_per_trade_base: float = 3.0
    risk_per_trade_step: float = 1.5
    risk_per_trade_max: float = 6.0
    position_pct_max: float = 80.0
    # 같은 봉 SL/TP 동시 도달 처리:
    #   False(기본) = 봉 경로 휴리스틱(bullish 봉 저점 먼저 / bearish 봉 고점 먼저)
    #   True = 무조건 SL 우선(worst-case 보수) — win 과소평가
    sl_priority: bool = False
    # --- 2026-06-10 #SHORT-BIAS: 실시간 봇 HTF EMA bias 게이트 재현 ---
    # 실거래는 _passes_htf_ema_bias(1h EMA[period] vs 현재가)로 방향을 거른다.
    # 백테스트엔 그게 없어 양방향 진입 → 실거래(숏 91%)와 동작 불일치였다.
    #   "off"    = 게이트 없음(양방향, 기존 백테스트 동작)
    #   "strict" = 1h close>EMA→롱만 / close<EMA→숏만 (실시간 봇 현행)
    #   "band"   = EMA ±band_pct 완충대 안이면 양방향 허용(되돌림 whipsaw 회피)
    #   "align"  = 다중 EMA 정배열/역배열 점수 게이트 (조윤 EMA 가중치 아이디어)
    htf_ema_bias: str = "off"
    htf_ema_period: int = 20
    htf_ema_band_pct: float = 0.0  # "band" 모드 완충 폭 (예: 0.003 = 0.3%)
    # "align" 모드 — 인접 EMA 쌍 정렬 점수. 60>120>200>... 정배열이면 +1씩(롱),
    # 역배열이면 -1씩(숏). 범위 -(N-1)~+(N-1). |score|>=threshold 면 그 방향만
    # 진입, 미만이면 추세 불명확 → 진입 자제(되돌림/횡보 whipsaw 회피).
    htf_align_periods: tuple[int, ...] = (60, 120, 200, 350, 480, 620)
    htf_align_threshold: int = 1
    # 2026-06-10 조윤 동적 전환: 보유 중 EMA 정렬 점수가 보유방향과 반대로
    # |score|>=flip_threshold 강반전하면 SL/TP 기다리지 않고 그 봉 close 에서
    # 즉시 청산(outcome="flip"). 반등 조기 포착 — 다음 봉부터 재진입(게이트 적용).
    htf_align_flip: bool = False
    htf_align_flip_threshold: int = 3
    # #MSS-FLIP (2026-06-15): 보유 중 보유방향 반대 CHoCH(구조전환) 확정 봉의 close
    # 에서 즉시 청산(outcome="mss_flip"). EMA flip(후행)보다 빠른 구조 기반 반등 컷.
    mss_flip: bool = False
    mss_swing_left: int = 2   # swing 감도(클수록 둔감 — 5m 노이즈 회피)
    mss_swing_right: int = 2
    # #OTE (2026-06-15): FVG/IFVG 진입 되돌림 깊이. 0.5=mean(CE,현행), 0.62~0.79=
    # ICT OTE 더 깊은 진입(RR↑·체결률↓ 트레이드오프). silver_bullet entry 에 전달.
    ote_level: float = 0.5
    # #MSS-BIAS-GATE (2026-06-15): 진입 방향을 마지막 CHoCH(구조전환) 방향으로 제한.
    # EMA align 게이트의 구조 기반 대체/병행 검증용(연구5: EMA 둘지 뺄지).
    mss_bias_gate: bool = False
    # #ALIGN-MSS-FILL (2026-06-15): 검증된 EMA align 은 그대로 두되, align 이
    # 침묵하는 애매구간(|score|<T, 기존 진입자제)만 MSS 구조 방향으로 채움.
    # EMA 유지 + MSS 보완 결합(파트너 의도: EMA 수치는 검증값이라 버리지 않음).
    align_mss_fill: bool = False
    # #UNUSED-DETECTOR (2026-06-15): 구현됐지만 진입에 미활용이던 ICT detector 를
    # confluence 가점(+1)으로 통합 — 방향 일치 시. 각 단독 기여도 검증용.
    apply_cbdr: bool = False
    apply_dol: bool = False
    apply_po3: bool = False
    apply_dailybias: bool = False
    # #OTE-FIB (2026-06-18): 직전 임펄스 swing leg 의 피보나치 0.618~0.786 되돌림
    # (ICT Optimal Trade Entry) 구간에 진입가가 있으면 confluence +1. 진입 방향과
    # 임펄스 방향 정합 필요(LONG=상승 leg 되돌림). ICT 정통 핵심 — 파트너 6/18.
    apply_ote: bool = False
    # #PD-FILTER (2026-06-23): ICT 프리미엄/디스카운트 게이트. dealing range(최근 swing
    # high/low)의 equilibrium(50%) 기준 — 진입가가 LONG=디스카운트(eq 아래)/SHORT=프리미엄
    # (eq 위)일 때만 진입. ICT 정통 독립 가드(align/prefer_direction 과 별개)로 숏 한쪽
    # 쏠림(숏91% 고착)을 구조적으로 차단. 판정 불가(swing 부족) 시 통과. True=활성.
    apply_pd_filter: bool = False
    # #REGIME (2026-06-15): 국면 적응 — 1h EMA 간격(스프레드)이 임계 미만이면
    # (반등·횡보 의심) align 방향게이트를 풀어 양방향 허용. 추세장(스프레드 큼)은
    # 현행 align 유지. 분리도 d=0.41 (눈 후보 단독 1위, 구조·ATR·결합 모두 열위).
    regime_adaptive: bool = False
    regime_spread_thr: float = 0.04
    # #TP-RR (2026-06-16): TP 를 risk 의 고정 배수로 강제(>0 일 때). target swing TP
    # 대신 entry±risk*tp_rr_override. 낮추면 TP 가까워 승률↑·RR↓ — 손익분기 대비
    # 순효과를 5년으로 검증(파트너: TP 낮춰 승률 높이기).
    tp_rr_override: float = 0.0
    # #BREAKEVEN (2026-06-17): 이익이 be_trigger*risk 도달 시 SL 을 entry(본전)로 이동.
    # "방향 맞는데 TP 전 되돌림→SL"(이익 반납) 손절 방지 — 파트너 실거래 관찰. 0=off.
    # be_lock>0 이면 본전 대신 entry±be_lock*risk(약간의 이익)로 잠금.
    be_trigger: float = 0.0
    be_lock: float = 0.0
    # #PARTIAL-TP (2026-06-17): partial_tp_rr*risk 도달 시 포지션 절반 익절, 나머지
    # 절반은 원 SL/TP 까지(partial_be 면 나머지 SL 을 본전으로). "10%에서 반익" 파트너
    # 철학. breakeven(전량 본전)과 달리 절반은 추세 끝까지 → net 보존 + 승률↑ 절충.
    # 0=off. 반환 exit_price=0.5*tp1+0.5*최종 가중(비용은 근사).
    partial_tp_rr: float = 0.0
    partial_be: bool = False
    # #LADDER-TP (2026-06-18): 손익률%(레버리지 적용) 기준 다단 분할 익절. 파트너 안 —
    # "10%/20%/원목표 3단 + 20% 도달 시 SL 본전+4%". ladder_levels_pnl 각 지점에서
    # ladder_alloc 비율 익절, 나머지는 원 TP 까지. ladder_be_after 번째 레벨 도달 시
    # 남은 SL 을 entry±ladder_be_pnl% 손익선으로 이동. 가격환산: Δprice = pnl%/100/lev.
    # ladder_tp=False 면 완전 비활성(회귀 없음). 승률·확정수익 vs net 트레이드오프 검증용.
    ladder_tp: bool = False
    ladder_levels_pnl: tuple = (10.0, 20.0)
    ladder_alloc: tuple = (0.34, 0.33)
    ladder_be_pnl: float = 4.0
    ladder_be_after: int = 2
    # ladder_mode: "pnl"=절대 손익%(ladder_levels_pnl) 지점 / "tpfrac"=원 TP 까지 거리를
    # ladder_tp_fracs 비율로 분할(예 1/3·2/3·원목표). tpfrac 은 RR 비례라 익절 지점이
    # 멀어 대박 보존↑·net 덜 깎임·승률 덜 오름(파트너 6/18 X vs Y 비교).
    ladder_mode: str = "pnl"
    ladder_tp_fracs: tuple = (0.3333, 0.6667)
    # #TRAIL (2026-06-17): trail_trigger*risk 이익 도달 후 SL 을 (최고가 − trail_dist*risk)
    # 로 따라 올림(LONG; SHORT 대칭). breakeven(본전 고정, 대박을 본전서 잘라먹음)과 달리
    # 대박을 따라가며 보존 → 승률↑ + net 유지 절충. 0=off.
    trail_trigger: float = 0.0
    trail_dist: float = 1.0
    # 2026-06-11 TF 플립: run_backtest_multitf 전용. 보유 중(진입~청산 사이 매 1m
    # 봉)에 현재 보유 setup 의 TF 보다 *더 높은 TF* 에서 새 유효 setup(stale 아님 +
    # conf 게이트 + EMA/align 방향 게이트 통과)이 뜨면, 그 봉 close 에서 기존 포지션을
    # 즉시 청산(outcome="tf_flip")하고 그 HTF setup 으로 전환 진입을 시도한다(체결되면
    # 그 SL/TP 로 재보유, 더 높은 TF 뜨면 또 플립 — 반복). 방향 무관(HTF 가 더 중요).
    # False 면 기존 정적 우선 진입과 100% 동일(회귀 없음). run_backtest 엔 영향 없음.
    tf_flip: bool = False
    # 2026-06-10 흑자 탐색: SL 거리(entry~setup.stop_loss)를 mult 배 (1.0=원본,
    # >1 넓힘=stop-hunt 회피·손실 큼, <1 좁힘). tp_keeps_rr=True 면 TP 를 원 RR
    # 유지하게 재계산(SL 비례), False 면 TP 고정(RR 변동).
    sl_dist_mult: float = 1.0
    tp_keeps_rr: bool = True
    # 2026-06-18 #CT-SL 국면별 동적 SL: 진입 시점 "방향 정합 추세"
    # (signed_trend = entry_trend_pct × 방향부호)가 ct_trend_threshold 미만
    # =역추세(되돌림 진입)면 sl_dist_mult 대신 sl_dist_mult_ct 사용. 0=비활성.
    # 근거: 역추세 분위에서 x4 가 전·후반 robust(되돌림 터지면 큰 폭 → 먼 TP 가 먹음),
    # 순추세 x3 유지. 트렌드 계산은 _entry_trend_pct(closes, fill_idx).
    sl_dist_mult_ct: float = 0.0
    ct_trend_threshold: float = 0.0
    # 2026-06-12 #LIQ-CAP 재현: 라이브 hotfix 와 동일 — 확장 SL 거리를
    # 청산 거리(entry/leverage)의 80% 로 캡, 캡<원본이면 확장 포기(원본 유지).
    sl_liq_cap: bool = False

    # ================================================================
    # 2026-08-08 라이브 정합 이식 (전수 감사 data/parity/audit.json)
    # ----------------------------------------------------------------
    # 배경: 라이브 45경로 중 40개가 백테와 달랐다. 아래 필드들은 전부
    # "프로덕션 BotIctInstance 에는 있는데 백테엔 없던" 경로를 이식한 것이며,
    # 각 필드 주석에 **라이브 어디서 옮겼는지** 출처를 박아 둔다.
    # 프로덕션 경로: C:/Users/지영민/Desktop/Aurora-ICT/src/aurora_ict/...
    #
    # ⚠️ 이 플래그들은 build_setup_timeline + run_backtest_from_timeline
    #    (= scripts/live_parity.py 가 쓰는 경로) 에서만 동작한다.
    #    구식 run_backtest() 는 정합 대상이 아니며 플래그가 켜져 있으면 거부한다.
    # ================================================================

    # [#1 impact:high] Phase B 4소스 (turtle_soup / mitigation_block /
    # implied_fvg / rejection_block). 출처: signal/ict_signal.py:113-121
    # generate_ict_signal → strategy/silver_bullet.py:843 build_extra_source_setups.
    # 라이브 진입 56건 중 38건(68%)이 이 경로로만 생성되는 셋업이었다.
    phase_b_sources: bool = False
    # [#2 impact:high] prefer_direction 셋업 선택. 출처: bot_ict_instance.py:1243
    # (prefer_direction=ema_dir) → signal/ict_signal.py:163-175.
    # 라이브는 "align 방향과 같은 셋업"만 남기고 그중 최신을 고른다.
    # 백테는 방향 무관 setups[-1] 을 고른 뒤 align 게이트로 차단 → 반대방향
    # 최신 셋업 하나가 그 봉의 거래를 통째로 없앤다. 반드시 timeline 빌드 시점에
    # 방향별 최신 셋업을 따로 보관해야 재현된다.
    prefer_direction_select: bool = False
    # [#3 impact:high] align 점수 계산식을 라이브 방식으로.
    # 출처: bot_ict_instance.py:1846-1898 (_ema_last / _compute_ema_align_score).
    # 라이브는 1h 봉 (pmax+50) 개만 받아 SMA(period) 시드 + 나머지 갱신으로 EMA 를
    # 만들고, **미완성 1h 봉**이 마지막 종가로 들어간다. 백테 기본
    # (_precompute_align_score) 은 전체 히스토리 ewm+shift(1) 이라 값이 다르다.
    align_live_formula: bool = False
    # [#5 impact:high] HTF FVG supporting boost (+1~+3).
    # 출처: bot_ict_instance.py:4502-4561 _apply_htf_supporting_boost
    #       + strategy/htf_fvg_map.py find_supporting_htf_fvg / TF_WEIGHT.
    # 라이브 진입의 93% 가 이 boost 를 받았다 → 실효 문턱이 5가 아니라 2~4였다.
    htf_fvg_support: bool = False
    # [mid] HTF/LTF 방향 충돌 게이트. 출처: bot_ict_instance.py:1483-1511
    # (htf_ltf_conflict_guard_ratio, 라이브 기본 1.10). 0=비활성.
    htf_ltf_conflict_guard_ratio: float = 0.0
    # [#7 impact:high] HTF FVG flip 청산 + #FLIP-MIN-R.
    # 출처: bot_ict_instance.py:4459-4500 _evaluate_htf_override (진입 시 flip target
    # 결정) + 4563-4589 _maybe_flip (봉 close 검사) + 4614-4640 handle_htf_flip
    # (#FLIP-MIN-R 게이트). htf_override_mode="C" 라이브 기본.
    htf_fvg_flip: bool = False
    flip_min_r: float = 0.0
    # HTF FVG map 빌드 파라미터. 출처: bot_ict_instance.py:4438-4443
    # (build_htf_fvg_map(tfs=htf_fvg_tfs, fvg_min_size_pct, limit=200))
    # + config/settings.py:423 htf_fvg_tfs 기본값.
    htf_fvg_tfs: tuple[str, ...] = ("15m", "1h", "2h", "4h", "1d", "1w")
    htf_fvg_limit: int = 200
    # [#8 impact:high] DOL 역방향 **감점**. 출처: bot_ict_instance.py:3104-3151
    # _apply_dol_bias + 모듈 상수 _DOL_COUNTER_PENALTY = 2 (line 280).
    # 기존 cfg.apply_dol 은 정방향 **+1 보너스**라 부호도 크기도 반대였다.
    # 라이브에선 -2 를 맞은 셋업이 문턱 5에서 전멸 → 사실상 강력한 컷 게이트.
    apply_dol_counter: bool = False
    dol_counter_penalty: int = 2
    # [#6 impact:high] 트레일 무장 시 분할익절 무효화.
    # 출처: bot_ict_instance.py:2365 (_arm_trailing 이 진입 직후 호출) →
    #       2646 pos.trail_armed=True → 3345 _maybe_partial_exit 즉시 return.
    # 즉 라이브에서 trail_trigger_r>0 이면 1.5R 분할익절은 **항상 비활성**인데
    # _simulate_exit 는 둘을 동시 적용해 승률·RR·건당R 이 전부 왜곡됐다.
    trail_supersedes_partial: bool = False
    # [#9 impact:high] 진입 '시점' 킬존 게이트 (#KZ-ENTRY).
    # 출처: bot_ict_instance.py:1337-1346 (not disable_time_filter 일 때
    # in_trade_window_sub(마지막 봉 ts) 재확인).
    entry_killzone_gate: bool = False
    # [mid] NY_PM 진입 차단 (#NYPM-GATE). 출처: bot_ict_instance.py:1320-1329
    # classify_killzone(ts) is KillzoneName.PM → NY local 13:30-16:00 (DST 반영).
    # live_parity 의 고정 UTC 17-21 근사는 최대 1.5시간 어긋난다.
    exclude_nypm: bool = False
    # [mid] setup.entry 가 현재가에서 멀면 entry/SL/TP 평행 이동.
    # 출처: bot_ict_instance.py:2193-2217 (#ENTRY-ADJ-RR). 라이브 기본 0.005.
    max_entry_distance_pct: float = 0.0
    # [mid] 일봉 스윕-반전 후 K일 역방향 차단 (#SWEEP-GATE).
    # 출처: bot_ict_instance.py:2762-2804 _sweep_gate_blocked. 구독제 강제 2.
    sweep_gate_days: int = 0
    # [mid] 상승 국면 전용 깊은 OTE (#REGIME-OTE, Origo 1.7).
    # 출처: bot_ict_instance.py:2726-2760 _regime_is_up / _effective_ote.
    # 0=비활성. 구독제 강제 0.786. 진입가가 바뀌므로 **detect 파라미터**다.
    ote_up_level: float = 0.0
    # [mid] #REGIME / #COND-ALIGN 를 **재생 루프 안**으로 이식.
    # 출처: bot_ict_instance.py:1407-1442 (_regime_floor / _strong_trend_floor).
    # 그동안 live_parity 는 이 둘을 '거래 사후 필터'로 걸었는데, 그러면 라이브가
    # 거르는 셋업이 백테에선 포지션 슬롯을 먹어 뒤 셋업을 가리는 인공물이 생긴다.
    # regime_rolling=True 면 라이브처럼 최근 150표본(>=20) 롤링 33/70분위를 쓰고,
    # 표본 부족이면 아래 하드코딩 floor 로 fallback (bot_ict_instance.py:3153-3175).
    regime_filter: bool = False
    cond_align: bool = False
    regime_rolling: bool = False
    regime_floor: float = 0.0          # 페어별 q33 fallback
    strong_trend_floor: float = 0.0    # 페어별 q70 fallback
    # [연구 전용, 2026-08-08] confluence **항목 기반** 진입 조건 (AND).
    # 왜 — 후보 룰(`macro_high AND bias` 등)을 거래 JSON 에 사후 필터로 걸면
    # 진짜 재실행과 다르다. replay 는 동시 포지션 1개라 걸러진 셋업이 백테에선
    # 슬롯을 먹어 뒤 셋업을 가리기 때문이다(conf_extract L1). 그래서 게이트를
    # 재생 루프 안, min_confluence 와 **같은 자리**에 둔다.
    # 토큰: "ob"/"sweep"/"bias"/"macro_high"/"macro_normal"/"macro_low"/"macro_any"
    #       /"turtle_soup"/"implied_fvg"/"mitigation_block"/"rejection_block"
    #       /"cisd"/"po3"/"ote"/"smt"/"dol_counter"/"htf>=N"(N=1~3)/"phase_a"/"phase_b"
    # 앞에 "!" 를 붙이면 부정(NOT). 빈 튜플이면 미사용.
    require_items: tuple[str, ...] = ()


@dataclass(slots=True)
class Trade:
    """체결 1건 기록."""

    entry_idx: int
    exit_idx: int
    direction: str
    entry: float
    exit_price: float
    outcome: str  # "tp" / "sl" / "eod"
    raw_pnl_pct: float
    net_pnl_pct: float
    confluence_score: int
    # 멀티 TF 백테스트(run_backtest_multitf)에서 이 setup 이 나온 TF 라벨("5m"/"15m"/"1h").
    # 단일 TF run_backtest 는 비워 둠(None) — 기존 동작/시그니처 불변.
    source_tf: str | None = None
    # 2026-06-17 유동적 ttl 연구: 진입 시점 변동성 (직전 14봉 ATR / 진입가, %).
    # 변동성 구간별 최적 ttl 분석용 — 0.0 이면 미측정(기존 호출 호환).
    entry_atr_pct: float = 0.0
    # 2026-06-18 국면 기반 ttl: 진입 직전 20봉 close 변화율 (%, 추세 방향+강도).
    # +상승추세 / -하락 / ~0 횡보. 국면별 최적 ttl 분석용.
    entry_trend_pct: float = 0.0
    # 2026-07-30 추가: **진입 시점 초기 SL / TP 가격** (sl_dist_mult·sl_liq_cap·
    # tp_rr_override 전부 적용된 최종값. 이후 BE/트레일 이동 전 값).
    # 필요 이유 — 이 값이 없어 R(위험 단위) 을 ATR 로 추정해야 했고, 그 때문에
    #   ① origo_tp_expand.py(7/29 TP 확대 연구) 전 변형이 동일 결과 → **무효 처리**
    #   ② flip_ab_backtest.py(7/30) flip 발동 R 이 라이브 0.61R vs 백테 1.44R 로 어긋남
    # R 기반 처방(TP 배수·flip 최소 R·부분청산 위치·SL 폭)은 전부 이 필드가 전제다.
    # 0.0 이면 미기록(구 호출 호환) — 사용측에서 0 체크 필요.
    entry_sl: float = 0.0
    entry_tp: float = 0.0
    # 2026-08-07 추가(연구 사본 전용): confluence **항목별** 내역.
    # 왜 — Trade 에 합산 점수(confluence_score)만 있어서 "1+2+4+5 식 조합" 실험을
    # 하려면 어떤 항목이 켜졌는지 알아야 하는데 그 정보가 전부 버려지고 있었다.
    #   confluences : silver_bullet 내부 항목 문자열 리스트
    #                 ("ob=...", "macro=...", "macro_high=...", "macro_low=...",
    #                  "sweep", "bias=...")
    #   base_score  : silver_bullet 이 매긴 점수 (boost 가산 전)
    #   boosts      : replay 의 bot 레벨 boost 중 **실제 발동한** 것들의 이름
    #                 ("cisd","smt","cbdr","dol","po3","ote","dailybias")
    # 항등식: confluence_score == base_score + len(boosts)  (각 boost 는 +1)
    confluences: tuple[str, ...] = ()
    base_score: int = 0
    boosts: tuple[str, ...] = ()
    # 2026-08-09 #FUNDING: 보유 구간 펀딩 정산 합계 (명목 대비 비율, **비용이 양수**).
    # 롱은 양(+)의 요율을 지불하고 숏은 수취하므로 방향 부호가 이미 반영돼 있다.
    # 왜 필요한가 — 백테가 수수료·슬리피지는 반영하면서 펀딩은 아예 없었다.
    # SB 보유는 중앙 13.3시간·평균 30.4시간으로 **62% 가 8시간(정산 주기)을 넘고**
    # 거래당 평균 3.8회 정산을 겪는다. 실측 결과 건당 +0.0096R(BTC+ETH) /
    # +0.0037R(알트5) 로 크지 않지만, 손익분기 경계 전략에서는 알트 여유의 30% 를
    # 먹고 **롱만 보면 +0.019R** 이라 롱숏 비교를 왜곡한다.
    # raw_pnl_pct 는 **건드리지 않는다** — 기존 연구 결과와 직접 비교가 깨지지 않게.
    # 반영은 net_pnl_pct 와 이 필드로만 하고, R 환산은 사용측이 선택한다.
    funding_pct: float = 0.0
    # 2026-08-09 #LIVE-SIZING: 라이브 진입 크기 계산 입력값.
    # 라이브는 `qty = equity × risk_pct% × dd스로틀 × 품질배수 / 손절거리` 이고
    # 명목이 `equity × leverage × position_pct_max%`(=5.6배) 를 넘지 않게 상한을 건다.
    # 백테는 `notional = size_pct × leverage` 고정(0.9×7=6.3배)이라 **라이브 평균
    # 2.46배의 2.6배**를 걸고 있었다. 복리·낙폭·파산이 못 믿을 숫자였던 실체다.
    #
    # 복리 자체는 여기서 계산하지 않는다 — replay 는 페어별로 독립 실행되는데
    # 실제 계좌는 페어가 자산을 공유하므로, 동시보유·일일캡·서킷은 **포트폴리오
    # 레벨**에서만 옳게 표현된다. 여기서는 그 시뮬이 필요로 하는 입력만 남긴다.
    #   smart_size_scale : 품질 배수 (볼륨·NW중심선·RSI 정합 개수 → 0.7~1.3)
    #   risk_pct_used    : 건당 리스크 % (문턱 5 때문에 실질 6.0 상수)
    smart_size_scale: float = 1.0
    risk_pct_used: float = 0.0

    @property
    def risk_pct(self) -> float:
        """진입가 대비 초기 위험폭(%). entry_sl 미기록이면 0.0."""
        if self.entry_sl <= 0 or self.entry <= 0:
            return 0.0
        return abs(self.entry - self.entry_sl) / self.entry * 100.0

    def r_multiple(self, price: float) -> float:
        """임의 가격이 진입 대비 몇 R 인지. entry_sl 미기록이면 0.0."""
        risk = abs(self.entry - self.entry_sl)
        if risk <= 0:
            return 0.0
        sign = 1.0 if self.direction == "long" else -1.0
        return (price - self.entry) * sign / risk


@dataclass(slots=True)
class BacktestResult:
    """집계 결과."""

    config: BacktestConfig
    trades: list[Trade] = field(default_factory=list)
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    total_net_pnl_pct: float = 0.0
    avg_net_pnl_pct: float = 0.0
    long_count: int = 0
    short_count: int = 0

    def summary(self) -> str:
        """한 줄 요약 (sweep 표 출력용)."""
        return (
            f"trades={self.n_trades} win={self.win_rate:.1%} "
            f"net={self.total_net_pnl_pct:+.2%} avg={self.avg_net_pnl_pct:+.3%} "
            f"L/S={self.long_count}/{self.short_count}"
        )


def load_ohlcv_parquet(path: str | Path) -> pd.DataFrame:
    """1m OHLCV parquet → DatetimeIndex(UTC) DataFrame.

    detect_silver_bullet_setups 가 fvg.ts_ms (macro confluence 등)에 index 를
    쓰므로 ts 기반 index 가 필요. columns = open/high/low/close/volume.
    """
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    cols = ["open", "high", "low", "close", "volume"]
    return df[[c for c in cols if c in df.columns]]


def _ote_confluence(window: pd.DataFrame, direction: Direction) -> bool:
    """직전 임펄스 swing leg 의 fib 0.618~0.786(ICT OTE)에 현재가가 있으면 True.

    ZigZag auto-fib 의 핵심 로직(마지막 pivot leg → retracement)을 우리 swing
    detector 로 재현. 진입 방향과 임펄스 방향이 정합해야 함:
    LONG = 상승 leg(저점→고점, 마지막 swing=HIGH) 의 되돌림 매수,
    SHORT = 하락 leg(고점→저점, 마지막 swing=LOW) 의 되돌림 매도.
    retracement = LONG:(hi-close)/(hi-lo), SHORT:(close-lo)/(hi-lo).
    """
    swings = detect_swing_points(window)
    if len(swings) < 2:
        return False
    a, b = swings[-2], swings[-1]  # 직전 leg (a→b)
    if direction is Direction.LONG and b.type is not SwingType.HIGH:
        return False
    if direction is Direction.SHORT and b.type is not SwingType.LOW:
        return False
    hi, lo = max(a.price, b.price), min(a.price, b.price)
    if hi <= lo:
        return False
    cl = float(window["close"].iloc[-1])
    retr = (hi - cl) / (hi - lo) if direction is Direction.LONG else (cl - lo) / (hi - lo)
    return 0.618 <= retr <= 0.786  # ICT OTE 구간(sweet spot 0.705)


def _pd_pass(window: pd.DataFrame, direction: Direction, entry: float) -> bool:
    """ICT 프리미엄/디스카운트 게이트 (#PD-FILTER).

    dealing range(최근 swing high/low)의 equilibrium(50%) 기준 — LONG 은 진입가가
    eq 아래(디스카운트), SHORT 은 eq 위(프리미엄)일 때만 True. swing 부족/range≤0
    이면 판정 불가로 True(통과). align/prefer_direction 과 별개 독립 방향 가드 —
    숏 한쪽 쏠림(숏91% 고착)을 구조적으로 차단.
    """
    swings = detect_swing_points(window)
    highs = [s.price for s in swings if s.type is SwingType.HIGH]
    lows = [s.price for s in swings if s.type is SwingType.LOW]
    if not highs or not lows:
        return True
    rh, rl = highs[-1], lows[-1]  # 최근 swing high/low = dealing range
    if rh <= rl:
        return True
    eq = (rh + rl) / 2.0
    if direction is Direction.LONG:
        return entry <= eq  # 디스카운트만
    return entry >= eq  # 프리미엄만 (SHORT)


def _boost_score(
    base_score: int, direction: Direction, window: pd.DataFrame,
    corr_window: pd.DataFrame | None, cfg: BacktestConfig,
    detail: list[str] | None = None,
) -> int:
    """bot 레벨 boost(CISD/SMT)를 base confluence_score 에 가산 — bot step() 재현.

    HTF FVG boost 는 멀티 TF map 필요라 1차 boost 범위에서 제외 (별도 단계).

    Args:
        base_score: silver_bullet 이 매긴 점수.
        direction: setup 방향.
        window: 진입 시점까지의 LTF 슬라이스.
        corr_window: SMT 용 상관 심볼 슬라이스 (없으면 None).
        cfg: 백테스트 파라미터.
        detail: 주어지면 **발동한 boost 이름**을 이 리스트에 append (2026-08-07,
            조합 실험용. 반환값·가산 로직은 전혀 바뀌지 않는다 — 기록만 추가).

    Returns:
        boost 가산 후 점수.
    """
    def _hit(name: str) -> None:
        """boost 1건 발동 기록 — detail 이 None 이면 no-op."""
        if detail is not None:
            detail.append(name)

    score = base_score
    if cfg.apply_cisd:
        cisd = detect_cisd(window)
        if cisd is not None:
            want = CisdType.BULLISH if direction is Direction.LONG else CisdType.BEARISH
            if cisd is want:
                score += 1
                _hit("cisd")
    if cfg.apply_smt and corr_window is not None and len(corr_window) > 0:
        swings = detect_swing_points(window)
        if len(swings) >= 2:
            events = detect_smt_divergence(swings, corr_window)
            if events:
                want = SmtType.BULLISH if direction is Direction.LONG else SmtType.BEARISH
                if events[-1].type is want:
                    score += 1
                    _hit("smt")
    # #UNUSED-DETECTOR: 구현됐지만 미활용이던 detector 를 방향 일치 시 confluence +1.
    is_long = direction is Direction.LONG
    cl = float(window["close"].iloc[-1])
    # 일부 detector 가 df["timestamp"](ms) 컬럼을 요구 → index 에서 보강.
    wdf = window
    if (cfg.apply_cbdr or cfg.apply_dol or cfg.apply_dailybias) and (
        "timestamp" not in window.columns
    ):
        wdf = window.assign(timestamp=window.index.astype("int64") // 1_000_000)
    if cfg.apply_cbdr:
        boxes = detect_cbdr_boxes(wdf)
        if boxes:
            st = classify_price_vs_cbdr(cl, boxes[-1])
            bull = st in (
                CBDRBiasState.ABOVE_1STD, CBDRBiasState.ABOVE_2STD, CBDRBiasState.ABOVE_3STD,
            )
            bear = st in (
                CBDRBiasState.BELOW_1STD, CBDRBiasState.BELOW_2STD, CBDRBiasState.BELOW_3STD,
            )
            if (is_long and bull) or (not is_long and bear):
                score += 1
                _hit("cbdr")
    if cfg.apply_dol:
        dols = compute_dol(wdf, detect_swing_points(wdf))
        bull_d = [d.distance for d in dols if d.type == "bullish"]
        bear_d = [d.distance for d in dols if d.type == "bearish"]
        if bull_d and bear_d:
            nb, nr = min(bull_d), min(bear_d)
            if (is_long and nb < nr) or (not is_long and nr < nb):
                score += 1
                _hit("dol")
    if cfg.apply_po3:
        ts_ms = int(window.index[-1].value // 1_000_000)
        if amd_phase(ts_ms) is AmdPhase.DISTRIBUTION:
            score += 1
            _hit("po3")
    if cfg.apply_ote and _ote_confluence(window, direction):
        score += 1
        _hit("ote")
    if cfg.apply_dailybias:
        daily = wdf.resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last",
             "timestamp": "first"},
        ).dropna()
        if len(daily) >= 2:
            db = compute_daily_bias(daily, cl)
            if (is_long and db is TrendDirection.UP) or (
                not is_long and db is TrendDirection.DOWN
            ):
                score += 1
                _hit("dailybias")
    return score


def resample_ohlcv(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """1m OHLCV → 상위 TF 집계 (라이브 trade TF 5m 일치용).

    DatetimeIndex 기준 OHLC 표준 집계 (open=first, high=max, low=min, close=last).
    봉이 안 채워진 구간(NaN)은 제거.

    Args:
        df: DatetimeIndex OHLCV.
        rule: pandas resample rule (예 "5min", "15min").
    """
    agg = {
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low": df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
    }
    if "volume" in df.columns:
        agg["volume"] = df["volume"].resample(rule).sum()
    return pd.DataFrame(agg).dropna()


def _gate_pass(score: int, rr: float, cfg: BacktestConfig) -> bool:
    """봇 step() 의 confluence 게이트 재현 — min_confluence 또는 고RR 예외."""
    if score >= cfg.min_confluence:
        return True
    return (
        cfg.high_rr_bypass_min_rr > 0
        and score >= 1
        and rr >= cfg.high_rr_bypass_min_rr
    )


def _item_tokens(confluences, boosts) -> set[str]:
    """셋업의 confluences/boosts → 항목 토큰 집합 (require_items 판정용).

    라이브 태그 이름 기준으로 정규화한다. 라이브 매매기록의 항목명
    (htf_support_weight / po3_distribution / turtle_soup …)과 1:1 대응시켜
    조건을 사람이 읽는 그대로 쓸 수 있게 하는 게 목적.

    Args:
        confluences: silver_bullet 항목 문자열 목록 ("ob=..","macro_high=..").
        boosts: replay bot 레벨 boost 이름 목록 ("cisd","po3","htf_support_weight=..").
    Returns:
        토큰 집합 (예 {"ob","macro_high","htf>=2","htf>=1","phase_a"}).
    """
    t: set[str] = set()
    for c in confluences:
        if c.startswith("ob="):
            t.add("ob")
        elif c.startswith("macro_high="):
            t.update(("macro_high", "macro_any"))
        elif c.startswith("macro_low="):
            t.update(("macro_low", "macro_any"))
        elif c.startswith("macro="):
            t.update(("macro_normal", "macro_any"))
        elif c.startswith("bias="):
            t.add("bias")
        elif c in ("sweep", "turtle_soup", "implied_fvg",
                   "mitigation_block", "rejection_block"):
            t.add(c)
    for b in boosts:
        if b.startswith("htf_support_weight="):
            # "..._boost+2" → htf>=1, htf>=2
            n = int(b.rsplit("+", 1)[1])
            for k in range(1, n + 1):
                t.add(f"htf>={k}")
        elif b.startswith("dol_counter"):
            t.add("dol_counter")
        elif b in ("cisd", "po3", "ote", "smt", "cbdr", "dol", "dailybias"):
            t.add(b)
    t.add("phase_b" if (t & {"turtle_soup", "implied_fvg",
                             "mitigation_block", "rejection_block"}) else "phase_a")
    return t


def _require_items_pass(confluences, boosts, req) -> bool:
    """require_items(AND, "!" 로 부정) 판정. req 가 비면 항상 True."""
    if not req:
        return True
    toks = _item_tokens(confluences, boosts)
    for r in req:
        if r.startswith("!"):
            if r[1:] in toks:
                return False
        elif r not in toks:
            return False
    return True


def _entry_atr_pct(
    highs, lows, closes, idx: int, entry: float, period: int = 14,
) -> float:
    """진입 직전 period 봉 ATR / 진입가 (%) — 유동적 ttl 연구용 변동성 측정.

    Args:
        highs/lows/closes: OHLC 배열 (numpy).
        idx: 진입(체결) 봉 인덱스.
        entry: 진입가.
        period: ATR 봉 수 (기본 14).
    Returns:
        ATR / entry × 100 (%). 측정 불가(초반 봉 등) 시 0.0.
    """
    if entry <= 0 or idx < 1:
        return 0.0
    start = max(1, idx - period + 1)
    total = 0.0
    cnt = 0
    for j in range(start, idx + 1):
        tr = max(
            highs[j] - lows[j],
            abs(highs[j] - closes[j - 1]),
            abs(lows[j] - closes[j - 1]),
        )
        total += tr
        cnt += 1
    if cnt == 0:
        return 0.0
    return (total / cnt) / entry * 100.0


def _entry_trend_pct(closes, idx: int, lookback: int = 20) -> float:
    """진입 직전 lookback 봉 close 변화율 (%) — 국면(추세) 측정.

    +면 상승추세, -면 하락추세, ~0 횡보. 국면별 최적 ttl 분석용.

    Args:
        closes: close 배열.
        idx: 진입(체결) 봉 인덱스.
        lookback: 추세 측정 봉 수 (기본 20).
    Returns:
        (close[idx] - close[idx-lookback]) / close[idx-lookback] × 100 (%).
    """
    if idx < lookback or idx >= len(closes):
        return 0.0
    past = closes[idx - lookback]
    if past <= 0:
        return 0.0
    return (closes[idx] - past) / past * 100.0


def _effective_sl_mult(cfg: "BacktestConfig", closes, fill_idx: int, is_long: bool) -> float:
    """국면별 동적 SL mult (#CT-SL).

    역추세 진입(방향 정합 추세 signed_trend < ct_trend_threshold)이면
    sl_dist_mult_ct, 아니면 기본 sl_dist_mult. sl_dist_mult_ct<=0 이면 비활성
    (항상 기본). signed_trend = entry_trend_pct × 방향부호(LONG +, SHORT -).
    """
    if cfg.sl_dist_mult_ct <= 0.0:
        return cfg.sl_dist_mult
    signed = _entry_trend_pct(closes, fill_idx) * (1.0 if is_long else -1.0)
    return cfg.sl_dist_mult_ct if signed < cfg.ct_trend_threshold else cfg.sl_dist_mult


def _simulate_fill(
    highs, lows, signal_idx: int, direction: Direction,
    limit: float, ttl_bars: int,
) -> int | None:
    """limit(setup.entry) 가격이 ttl_bars 봉 내 닿으면 체결 봉 idx, 안 닿으면 None.

    ICT 는 retrace limit 진입 — 신호 후 가격이 FVG mean(entry) 까지 되돌아와야
    체결. TTL 안에 안 닿으면 타점 포기(미체결). bot 의 marketable limit + TTL 재현.

    - LONG: 가격이 내려와 low <= entry 면 체결
    - SHORT: 가격이 올라와 high >= entry 면 체결
    """
    n = len(highs)
    end = min(signal_idx + 1 + ttl_bars, n)
    for j in range(signal_idx + 1, end):
        if direction is Direction.LONG and float(lows[j]) <= limit:
            return j
        if direction is Direction.SHORT and float(highs[j]) >= limit:
            return j
    return None


def _simulate_exit(
    opens, highs, lows, closes, entry_idx: int, direction: Direction,
    sl: float, tp: float, cfg: BacktestConfig,
    align_score: Any = None,
    flip_check: Any = None,
    mss_signal: Any = None,
    entry: float | None = None,
    htf_flip_zone: tuple[float, float] | None = None,
) -> tuple[int, float, str]:
    """진입 후 봉들에서 SL/TP 먼저 닿는 지점 → (exit_idx, exit_price, outcome).

    같은 봉에 SL·TP 둘 다 도달 시 봉 내 경로를 추정:
    bullish 봉(close>=open)은 보통 open→저점→고점→close 경로라 **저점 먼저**,
    bearish 봉은 open→고점→저점→close 라 **고점 먼저** 형성됐다고 가정한다.
    sl_priority=True 면 무조건 SL(worst-case 보수).

    align_score 가 주어지고 cfg.htf_align_flip=True 면, 보유 중 EMA 정렬 점수가
    보유방향과 반대로 |score|>=flip_threshold 강반전한 봉의 close 에서 즉시 청산
    (outcome="flip") — 조윤 동적 전환. SL/TP 우선, 미도달 시 flip 판정.

    flip_check 가 주어지면(run_backtest_multitf 의 TF 플립 전용), 매 봉 SL/TP·align
    flip 미도달 시 ``flip_check(j)`` 를 호출한다. 그게 truthy(=더 높은 TF setup 발생)
    면 그 봉 close 에서 즉시 청산(outcome="tf_flip"). SL/TP 가 항상 우선. flip_check 는
    j 시점까지 닫힌 HTF 봉만 보므로 look-ahead 없음(호출부 _tf_setup_at 책임).
    run_backtest 는 flip_check 를 넘기지 않으므로 동작 불변.

    2026-08-08 정합 이식 2건:
    - ``htf_flip_zone`` — 진입 시점에 확정된 HTF FVG flip target zone(low, high).
      봉 close 가 zone 안이면 그 close 에서 청산(outcome="htf_flip").
      출처: bot_ict_instance.py:4563-4589 ``_maybe_flip``(봉 close 검사) +
      4614-4640 ``handle_htf_flip`` 의 #FLIP-MIN-R 게이트(cfg.flip_min_r).
      R 은 **진입 시점 risk** 기준(_flip_profit_r:4591-4613 과 동일).
    - ``cfg.trail_supersedes_partial`` — 라이브는 진입 직후 _arm_trailing 이
      성공하면 pos.trail_armed=True 가 되고 _maybe_partial_exit(3345)가 즉시
      return 한다. 즉 트레일이 켜져 있으면 분할익절은 **항상 비활성**이다.
      기존 백테는 둘을 동시 적용해 승률·RR·건당R 이 전부 왜곡됐다.
    """
    n = len(highs)
    use_flip = cfg.htf_align_flip and align_score is not None
    flip_t = cfg.htf_align_flip_threshold
    use_mss = cfg.mss_flip and mss_signal is not None
    use_be = cfg.be_trigger > 0 and entry is not None
    risk0 = abs(entry - sl) if entry is not None else 0.0
    be_done = False
    dir_sign = 1.0 if direction is Direction.LONG else -1.0
    # #6: 트레일 무장 시 분할익절 무효 (라이브 _maybe_partial_exit 3343-3346).
    trail_kills_partial = (
        cfg.trail_supersedes_partial and cfg.trail_trigger > 0 and cfg.trail_dist > 0
    )
    use_partial = (
        cfg.partial_tp_rr > 0 and entry is not None and risk0 > 0
        and not trail_kills_partial
    )
    tp1_price = (
        ((entry + cfg.partial_tp_rr * risk0) if direction is Direction.LONG
         else (entry - cfg.partial_tp_rr * risk0))
        if use_partial else 0.0
    )
    partial_done = False
    use_trail = cfg.trail_trigger > 0 and entry is not None and risk0 > 0
    peak = entry if entry is not None else 0.0
    trail_on = False
    # #LADDER-TP: 손익률% → 가격 환산 후 다단 익절 가격 산출. is_long 따라 ± 방향.
    use_ladder = cfg.ladder_tp and entry is not None and cfg.leverage > 0
    is_long_dir = direction is Direction.LONG
    ladder_prices: list[float] = []
    ladder_alloc: list[float] = []
    ladder_hit: list[bool] = []
    ladder_realized = 0.0  # 누적 청산 비율
    ladder_value = 0.0     # 누적 alloc×price (가중 청산가 분자)
    be_price_ladder = 0.0
    if use_ladder:
        if cfg.ladder_mode == "tpfrac":
            # 원 TP 까지 거리를 비율로 분할 (RR 비례). tp 방향이 ±를 자동 결정.
            for fr, al in zip(cfg.ladder_tp_fracs, cfg.ladder_alloc):
                ladder_prices.append(entry + (tp - entry) * fr)
                ladder_alloc.append(al)
        else:
            for lv, al in zip(cfg.ladder_levels_pnl, cfg.ladder_alloc):
                dp = (lv / 100.0) / cfg.leverage  # 손익률%→가격변동률
                ladder_prices.append(entry * (1 + dp) if is_long_dir else entry * (1 - dp))
                ladder_alloc.append(al)
        ladder_hit = [False] * len(ladder_prices)
        dpbe = (cfg.ladder_be_pnl / 100.0) / cfg.leverage
        be_price_ladder = entry * (1 + dpbe) if is_long_dir else entry * (1 - dpbe)

    def _exit(idx: int, price: float, outcome: str) -> tuple[int, float, str]:
        # #LADDER-TP: 일부 익절됐으면 (누적 alloc×익절가) + 잔여비율×최종가 가중.
        if use_ladder and ladder_realized > 0:
            rem = 1.0 - ladder_realized
            return idx, ladder_value + rem * price, "L_" + outcome
        # 부분익절 됐으면 0.5*tp1 + 0.5*최종 가중(절반씩 청산, 비용 근사).
        if partial_done:
            return idx, 0.5 * tp1_price + 0.5 * price, "p_" + outcome
        return idx, price, outcome

    for j in range(entry_idx + 1, n):
        hi, lo = float(highs[j]), float(lows[j])
        # #LADDER-TP: 각 미달성 레벨 도달 시 분할 익절(가중 누적) + ladder_be_after 번째
        # 레벨 도달 시 잔여분 SL 을 본전+α 손익선으로 이동(확정수익 보호).
        if use_ladder:
            for k in range(len(ladder_prices)):
                if not ladder_hit[k]:
                    hit_k = (hi >= ladder_prices[k]) if is_long_dir else (lo <= ladder_prices[k])
                    if hit_k:
                        ladder_hit[k] = True
                        ladder_realized += ladder_alloc[k]
                        ladder_value += ladder_alloc[k] * ladder_prices[k]
                        if (k + 1) >= cfg.ladder_be_after and cfg.ladder_be_pnl != 0.0:
                            if is_long_dir and be_price_ladder > sl:
                                sl = be_price_ladder
                            elif not is_long_dir and be_price_ladder < sl:
                                sl = be_price_ladder
        # #PARTIAL-TP: TP1(이익) 먼저 — 절반 익절, 나머지 계속(partial_be 면 본전 보호).
        if use_partial and not partial_done:
            hit_tp1 = (hi >= tp1_price) if direction is Direction.LONG else (lo <= tp1_price)
            if hit_tp1:
                partial_done = True
                if cfg.partial_be:
                    sl = entry
        # #TRAIL: 최고가 추적 + trigger 도달 후 SL 을 (최고가 ∓ trail_dist*risk)로 올림.
        if use_trail:
            if direction is Direction.LONG:
                if hi > peak:
                    peak = hi
                if not trail_on and (peak - entry) >= cfg.trail_trigger * risk0:
                    trail_on = True
                if trail_on:
                    new_sl = peak - cfg.trail_dist * risk0
                    if new_sl > sl:
                        sl = new_sl
            else:
                if lo < peak:
                    peak = lo
                if not trail_on and (entry - peak) >= cfg.trail_trigger * risk0:
                    trail_on = True
                if trail_on:
                    new_sl = peak + cfg.trail_dist * risk0
                    if new_sl < sl:
                        sl = new_sl
        if direction is Direction.LONG:
            hit_sl, hit_tp = lo <= sl, hi >= tp
        else:
            hit_sl, hit_tp = hi >= sl, lo <= tp
        if hit_sl and hit_tp:
            if cfg.sl_priority:
                return _exit(j, sl, "sl")
            bar_up = float(closes[j]) >= float(opens[j])
            # LONG: SL=저점쪽 → bullish 봉이면 SL 먼저. SHORT: SL=고점쪽 → bearish 봉이면 SL 먼저.
            sl_first = bar_up if direction is Direction.LONG else not bar_up
            return _exit(j, sl, "sl") if sl_first else _exit(j, tp, "tp")
        if hit_sl:
            return _exit(j, sl, "sl")
        if hit_tp:
            return _exit(j, tp, "tp")
        # #HTF-FLIP: 봉 close 가 진입 시 확정된 반대 HTF FVG zone 안이면 즉시 청산.
        # 라이브 _maybe_flip 은 봉 close 에서만 검사(SaaS 는 flip_watch_enabled=False
        # 라 WS tick 경로가 없다) → 여기 위치가 정확한 대응.
        if htf_flip_zone is not None and entry is not None and risk0 > 0:
            cl_j = float(closes[j])
            if htf_flip_zone[0] <= cl_j <= htf_flip_zone[1]:
                # #FLIP-MIN-R: 이익이 최소 R 미만이면 flip 무시하고 홀드 계속.
                r_now = (cl_j - entry) * dir_sign / risk0
                if cfg.flip_min_r <= 0 or r_now >= cfg.flip_min_r:
                    return _exit(j, cl_j, "htf_flip")
        # SL/TP 미도달 — EMA 정렬 점수 강반전 시 flip 청산 (조윤 동적 전환).
        if use_flip:
            sc = align_score[j]
            if sc == sc:  # NaN 아니면
                flipped = (
                    (direction is Direction.LONG and sc <= -flip_t)
                    or (direction is Direction.SHORT and sc >= flip_t)
                )
                if flipped:
                    return _exit(j, float(closes[j]), "flip")
        # SL/TP·EMA flip 미도달 — 보유방향 반대 CHoCH 확정 봉이면 MSS flip 청산.
        if use_mss:
            ms = mss_signal[j]
            if (direction is Direction.SHORT and ms == 1) or (
                direction is Direction.LONG and ms == -1
            ):
                return _exit(j, float(closes[j]), "mss_flip")
        # SL/TP·align flip 미도달 — 더 높은 TF setup 발생 시 TF 플립 청산.
        if flip_check is not None and flip_check(j):
            return _exit(j, float(closes[j]), "tf_flip")
        # #BREAKEVEN: 이익 be_trigger*risk 도달 시 SL 을 본전(±be_lock*risk)으로 이동.
        if use_be and not be_done and risk0 > 0:
            prof = (hi - entry) if direction is Direction.LONG else (entry - lo)
            if prof >= cfg.be_trigger * risk0:
                lock = cfg.be_lock * risk0
                sl = (entry + lock) if direction is Direction.LONG else (entry - lock)
                be_done = True
    return _exit(n - 1, float(closes[-1]), "eod")


def _precompute_htf_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """각 1m 봉 시점의 '직전까지 확정된 1h 봉들'로 계산한 EMA 시리즈.

    look-ahead 방지: 진입 시점 i 의 1h EMA 는 그 시점에 이미 '닫힌' 1h 봉들만
    사용(진행 중 1h 봉 제외). 호출부에서 현재 1m close 와 비교 → 실시간
    ``_passes_htf_ema_bias``(닫힌 EMA vs 현재가)와 같은 의미.

    Args:
        df: 1m OHLCV (DatetimeIndex UTC).
        period: EMA 기간 (1h 봉 기준).

    Returns:
        df.index 에 정렬된 EMA 시리즈 (초기 구간은 NaN — 게이트 skip).
    """
    h1 = df["close"].resample("1h").last()
    ema_h1 = h1.ewm(span=period, adjust=False).mean()
    # 직전 확정봉 EMA (현재 진행 1h 봉은 아직 안 닫혔으니 shift 로 제외).
    ema_valid = ema_h1.shift(1)
    return ema_valid.reindex(df.index, method="ffill")


def _precompute_mss_signal(
    df: pd.DataFrame, left: int, right: int,
) -> np.ndarray:
    """봉별 CHoCH(구조전환) 신호 배열: +1=bullish CHoCH(상승전환), -1=bearish, 0=없음.

    구조전환(CHoCH)만 신호화하고 BOS(추세지속)는 제외. 각 신호를 돌파 확정 봉
    (event.idx)에 박는다 — 보유 중 그 봉 close 청산에 쓰며 확정 봉 정보만 보므로
    look-ahead 없음(EMA flip 과 동일 성격).

    Args:
        df: OHLCV (백테스트 입력 TF 그대로).
        left/right: swing 감도 (detect_swing_points; 클수록 둔감).

    Returns:
        len(df) 길이 int8 배열 (+1 / -1 / 0).
    """
    from aurora_ict.indicators.structure import (
        StructureType,
        detect_structure_events,
    )

    swings = detect_swing_points(df, left=left, right=right)
    events = detect_structure_events(df, swings)
    sig = np.zeros(len(df), dtype=np.int8)
    for ev in events:
        if ev.type is StructureType.CHOCH_BULLISH:
            sig[ev.idx] = 1
        elif ev.type is StructureType.CHOCH_BEARISH:
            sig[ev.idx] = -1
    return sig


def _precompute_mss_bias(
    df: pd.DataFrame, left: int, right: int,
) -> np.ndarray:
    """봉별 현재 구조 방향(마지막 CHoCH 를 ffill): +1=상승전환 우위(롱),
    -1=하락전환 우위(숏), 0=아직 CHoCH 없음(미정).

    EMA align 게이트의 구조 기반 대체용(연구5). 진입 시점 i 의 값은 i 이하에서
    확정된 마지막 CHoCH 방향이므로 look-ahead 없음.

    Args:
        df: OHLCV.
        left/right: swing 감도.

    Returns:
        len(df) 길이 int8 배열 (+1 / -1 / 0).
    """
    from aurora_ict.indicators.structure import (
        StructureType,
        detect_structure_events,
    )

    swings = detect_swing_points(df, left=left, right=right)
    events = sorted(detect_structure_events(df, swings), key=lambda e: e.idx)
    bias = np.zeros(len(df), dtype=np.int8)
    cur = 0
    ei = 0
    for i in range(len(df)):
        while ei < len(events) and events[ei].idx <= i:
            t = events[ei].type
            if t is StructureType.CHOCH_BULLISH:
                cur = 1
            elif t is StructureType.CHOCH_BEARISH:
                cur = -1
            ei += 1
        bias[i] = cur
    return bias


def _precompute_ema_spread(
    df: pd.DataFrame, periods: tuple[int, ...],
) -> np.ndarray:
    """봉별 1h EMA(periods) 간격/close. 추세=발산(큼), 반등·횡보=수렴(작음).

    국면 적응 게이트용 '눈'(분리도 d=0.41, 단독 1위). look-ahead 방지:
    _precompute_align_score 와 동일하게 직전 확정 1h 봉 기준(shift).

    Args:
        df: 1m(또는 리샘플) OHLCV.
        periods: align EMA 기간들(1h 봉 기준).

    Returns:
        df.index 정렬 spread 배열 (초기 NaN).
    """
    h1 = df["close"].resample("1h").last()
    em = pd.concat(
        [h1.ewm(span=p, adjust=False).mean() for p in periods], axis=1,
    )
    spread = (em.max(axis=1) - em.min(axis=1)) / h1
    return spread.shift(1).reindex(df.index, method="ffill").to_numpy()


def _precompute_align_score(
    df: pd.DataFrame, periods: tuple[int, ...],
) -> pd.Series:
    """각 1m 봉 시점의 다중 EMA 정렬 점수 (조윤 EMA 가중치 아이디어).

    1h 봉 기준 EMA[periods] 를 계산하고, 인접 쌍이 정배열(짧은>긴)이면 +1,
    역배열이면 -1 누적. 전부 정배열이면 +(N-1)(강한 상승추세), 전부 역배열이면
    -(N-1)(강한 하락). 0 근처면 추세 불명확(되돌림/횡보).

    look-ahead 방지: _precompute_htf_ema 와 동일하게 직전 확정 1h 봉 기준(shift).

    Args:
        df: 1m OHLCV (DatetimeIndex UTC).
        periods: EMA 기간들 (짧은→긴 순서, 1h 봉 기준).

    Returns:
        df.index 에 정렬된 점수 시리즈 (초기 구간 NaN — 게이트 skip).
    """
    h1 = df["close"].resample("1h").last()
    emas = [h1.ewm(span=p, adjust=False).mean() for p in periods]
    score = pd.Series(0.0, index=h1.index)
    for a, b in zip(emas[:-1], emas[1:], strict=False):
        score = score + (a > b).astype(int) - (a < b).astype(int)
    # EMA 미성숙 구간 무효 처리 — ewm(adjust=False) 는 첫 봉부터 비-NaN 이라
    # notna() 마스크는 무효였다(2026-06-11 리뷰). 가장 긴 period 만큼 1h 봉이
    # 쌓이기 전엔 NaN → 게이트 skip(방향 강제 안 함). 라이브는 거래소에서
    # pmax+50 봉을 받아 항상 성숙 상태이므로, 슬라이스 초기 구간은 보수적으로
    # 게이트를 끄는 쪽이 잘못된 방향 강제보다 낫다.
    pmax = max(periods)
    mature = pd.Series(range(len(h1)), index=h1.index) >= pmax
    score = score.where(mature)
    return score.shift(1).reindex(df.index, method="ffill")


# ======================================================================
# 2026-08-08 라이브 정합 이식 블록
# ----------------------------------------------------------------------
# 아래 함수들은 전부 프로덕션 BotIctInstance(및 그것이 부르는 모듈)에서
# **로직을 그대로 옮긴 것**이다. 단순화한 곳은 없고, 옮기지 못한 것은
# 근사하지 않고 scripts/live_parity.py 의 GAPS 에 명시했다.
# 출처 파일: C:/Users/지영민/Desktop/Aurora-ICT/src/aurora_ict/
# ======================================================================

# 라이브가 in_trade_window_sub / classify_killzone 로 쓰는 NY local 정의를
# 벡터화용 (시작초, 끝초) 로 옮긴 것. 출처: timing/killzone.py:68-98.
# _within 은 start <= t < end 라 끝은 배타적. ASIAN 은 23:59:59 배타.
_KZ_WINDOWS_SEC: tuple[tuple[int, int], ...] = (
    (19 * 3600, 23 * 3600 + 59 * 60 + 59),   # ASIAN
    (2 * 3600, 5 * 3600),                     # LONDON
    (7 * 3600, 10 * 3600),                    # NY_AM
    (10 * 3600, 12 * 3600),                   # LONDON_CLOSE
    (13 * 3600 + 30 * 60, 16 * 3600),         # PM
)
_SB_WINDOWS_SEC: tuple[tuple[int, int], ...] = (
    (3 * 3600, 4 * 3600), (10 * 3600, 11 * 3600), (14 * 3600, 15 * 3600),
)
_MACRO_WINDOWS_SEC: tuple[tuple[int, int], ...] = (
    (2 * 3600 + 33 * 60, 3 * 3600), (4 * 3600 + 3 * 60, 4 * 3600 + 30 * 60),
    (8 * 3600 + 50 * 60, 9 * 3600 + 10 * 60),
    (9 * 3600 + 50 * 60, 10 * 3600 + 10 * 60),
    (10 * 3600 + 50 * 60, 11 * 3600 + 10 * 60),
    (11 * 3600 + 50 * 60, 12 * 3600 + 10 * 60),
    (13 * 3600 + 10 * 60, 13 * 3600 + 40 * 60),
    (15 * 3600 + 15 * 60, 15 * 3600 + 45 * 60),
)
# PM killzone (NY local 13:30-16:00) — #NYPM-GATE 용.
# classify_killzone 은 STANDARD_KILLZONES 순서로 첫 매칭을 돌려주는데 PM 구간과
# 겹치는 앞 killzone 이 없으므로 "PM 창 안 = classify 결과 PM" 이 정확히 성립.
_PM_START_SEC = 13 * 3600 + 30 * 60
_PM_END_SEC = 16 * 3600
_NYSE_OPEN_SEC = 9 * 3600 + 30 * 60   # timing/killzone.py:164 _NYSE_OPEN
_NYSE_CLOSE_SEC = 16 * 3600           # timing/killzone.py:165 _NYSE_CLOSE


def _ny_local_seconds(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """봉 시각을 NY local(DST 자동) 의 (자정 이후 초, 요일) 로 변환.

    출처: timing/killzone.py:101-103 ``_to_ny_time`` — 라이브도 zoneinfo
    America/New_York 으로 변환하므로 서머타임 전환이 그대로 반영된다.
    (live_parity 의 고정 UTC 17-21 근사는 최대 1.5시간 어긋났다.)

    Args:
        df: DatetimeIndex(UTC) OHLCV.
    Returns:
        (초 배열, 요일 배열) — 둘 다 len(df).
    """
    ny = df.index.tz_convert("America/New_York")
    secs = (ny.hour * 3600 + ny.minute * 60 + ny.second).to_numpy()
    return secs.astype(np.int32), ny.weekday.to_numpy().astype(np.int8)


def _in_any_window(secs: np.ndarray, windows) -> np.ndarray:
    """start <= t < end 창 중 하나라도 걸리면 True (killzone._within 동일)."""
    out = np.zeros(len(secs), dtype=bool)
    for s, e in windows:
        out |= (secs >= s) & (secs < e)
    return out


def _precompute_ny_gates(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(#KZ-ENTRY 통과 마스크, NY_PM 마스크) 를 봉별로 미리 계산.

    - 통과 마스크 = ``in_trade_window_sub`` 재현
      (timing/killzone.py:180-207: NYSE 09:30~16:00 평일 **AND**
       (Killzone OR Macro OR Silver Bullet)).
      라이브 호출부: bot_ict_instance.py:1337-1346 (#KZ-ENTRY, 진입 '시점' 검사).
    - NY_PM 마스크 = ``classify_killzone(ts) is KillzoneName.PM``
      라이브 호출부: bot_ict_instance.py:1320-1329 (#NYPM-GATE).

    Args:
        df: DatetimeIndex(UTC) OHLCV.
    Returns:
        (sub_ok, is_nypm) — 둘 다 bool 배열 len(df).
    """
    secs, wd = _ny_local_seconds(df)
    nyse = (wd < 5) & (secs >= _NYSE_OPEN_SEC) & (secs <= _NYSE_CLOSE_SEC)
    precise = (
        _in_any_window(secs, _KZ_WINDOWS_SEC)
        | _in_any_window(secs, _SB_WINDOWS_SEC)
        | _in_any_window(secs, _MACRO_WINDOWS_SEC)
    )
    is_pm = (secs >= _PM_START_SEC) & (secs < _PM_END_SEC)
    return (nyse & precise), is_pm


def _precompute_regime_up(df: pd.DataFrame) -> np.ndarray:
    """#REGIME-OTE 상승 국면 판정 — 봉별 bool.

    출처: bot_ict_instance.py:2726-2754 ``_regime_is_up``.
        1d 30봉 fetch → ``closes = d["close"].iloc[:-1]`` (미완성 오늘 제외)
        → ``sig = ret.iloc[-20:].std()`` / ``r20 = closes[-1]/closes[-21]-1``
        → ``is_up = r20 / (sig*sqrt(20)) > 0.75``. 결과는 UTC 날짜 단위 캐시.
    즉 D일의 판정은 **D-1 일까지의 일봉**만 쓴다 → shift(1) 로 재현.

    Args:
        df: DatetimeIndex(UTC) OHLCV.
    Returns:
        len(df) bool 배열. 판정 불가(초기)면 False (라이브 기본 OTE 유지와 동일).
    """
    daily = df["close"].resample("1D").last().dropna()
    ret = daily.pct_change()
    sig = ret.rolling(20).std()                      # pandas ddof=1 — 라이브 동일
    r20 = daily / daily.shift(20) - 1.0
    z = r20 / (sig * (20 ** 0.5))
    is_up_day = (z > 0.75).astype(bool)
    # D일 판정은 D-1 까지의 데이터 → 하루 밀어서 사용.
    is_up_day = is_up_day.shift(1, fill_value=False).astype(bool)
    idx = is_up_day.index.get_indexer(df.index.floor("1D"), method="ffill")
    out = np.zeros(len(df), dtype=bool)
    ok = idx >= 0
    out[ok] = is_up_day.to_numpy()[idx[ok]]
    return out


def _precompute_align_score_live(
    df: pd.DataFrame, periods: tuple[int, ...],
) -> np.ndarray:
    """라이브 계산식 그대로의 다중 EMA 정렬 점수 (#3, impact:high).

    출처: bot_ict_instance.py:1846-1898.
        ``_compute_ema_align_score`` 는 htf_ema_bias_tf(1h) 봉을 **pmax+50 개만**
        fetch 한 뒤 각 period 마다 ``_ema_last`` 로 EMA 를 계산한다:
            k = 2/(p+1); ema = mean(closes[:p]);  # SMA 시드
            for px in closes[p:]: ema = px*k + ema*(1-k)
        그리고 **마지막 봉은 미완성 1h 봉**(현재가)이다.
        기본 백테(_precompute_align_score)는 전체 히스토리 ewm(span,adjust=False)
        + shift(1) 이라 같은 시점에 다른 값이 나온다.

    구현 메모(수학적으로 동일, 벡터화만):
        길이 N=pmax+50 의 창에서 갱신 횟수 u=N-p 라 하면
            ema = k*마지막종가 + (1-k)*E,
            E   = seed*(1-k)^(u-1) + Σ_{d=0}^{u-2} k(1-k)^d * c1h[g-1-d]
            seed= mean(c1h[g-N+1 .. g-N+p])   (= rolling(p).mean()[g-u])
        seed 항은 rolling mean, Σ 항은 지수커널 convolution 으로 한 번에 뽑는다.
        루프 재귀와 부동소수 오차 수준까지 같은 값이다.

    Args:
        df: 5m(또는 매매 TF) OHLCV, DatetimeIndex UTC.
        periods: EMA 기간들 (짧은→긴, 1h 봉 기준).
    Returns:
        len(df) float 배열. 창이 안 차는 초반 구간은 NaN(게이트 skip).
    """
    ps = [max(2, int(p)) for p in periods]
    pmax = max(ps)
    n_win = pmax + 50  # 라이브 fetch limit — bot_ict_instance.py:1870
    h1 = df["close"].resample("1h").last().dropna()
    c1h = h1.to_numpy(dtype=float)
    m = len(c1h)
    close5 = df["close"].to_numpy(dtype=float)
    g = h1.index.get_indexer(df.index.floor("1h"), method="ffill")
    valid = g >= (n_win - 1)
    emas: list[np.ndarray] = []
    for p in ps:
        k = 2.0 / (p + 1.0)
        u = n_win - p                  # 창 안 갱신 횟수
        roll = pd.Series(c1h).rolling(p).mean().to_numpy()
        seed = np.full(m, np.nan)
        if u < m:
            seed[u:] = roll[: m - u]   # seed[g] = mean(c1h[g-N+1 .. g-N+p])
        tail_len = u - 1
        if tail_len > 0:
            w = k * (1.0 - k) ** np.arange(tail_len)
            conv = np.convolve(c1h, w)[:m]     # conv[t] = Σ_d w[d]*c1h[t-d]
            tail = np.full(m, np.nan)
            tail[1:] = conv[: m - 1]           # tail[g] = conv[g-1]
        else:
            tail = np.zeros(m)
        e_closed = seed * (1.0 - k) ** tail_len + tail
        cur = np.full(len(df), np.nan)
        cur[valid] = k * close5[valid] + (1.0 - k) * e_closed[g[valid]]
        emas.append(cur)
    score = np.zeros(len(df))
    for a, b in zip(emas[:-1], emas[1:], strict=False):
        score += np.where(a > b, 1.0, 0.0) - np.where(a < b, 1.0, 0.0)
    score[~valid] = np.nan
    return score


def _precompute_sweep_gate(
    df: pd.DataFrame, days: int,
) -> tuple[np.ndarray, np.ndarray]:
    """#SWEEP-GATE — 봉별 (숏 차단, 롱 차단) 마스크.

    출처: bot_ict_instance.py:2762-2804 ``_sweep_gate_blocked``.
        1d 봉에서 마지막(오늘 미완성) 제외 → 최근 K 일 각각에 대해
        low < 직전10일 최저  &&  close > (low+high)/2  → SHORT 차단
        high > 직전10일 최고 &&  close < (low+high)/2  → LONG  차단
        (판정은 UTC 날짜 단위 캐시.)

    Args:
        df: DatetimeIndex(UTC) OHLCV.
        days: sweep_gate_days (K). 0 이면 전부 False.
    Returns:
        (block_short, block_long) — bool 배열 len(df).
    """
    n = len(df)
    if days <= 0:
        z = np.zeros(n, dtype=bool)
        return z, z
    d = pd.DataFrame({
        "high": df["high"].resample("1D").max(),
        "low": df["low"].resample("1D").min(),
        "close": df["close"].resample("1D").last(),
    }).dropna()
    lows, highs, closes = (
        d["low"].to_numpy(), d["high"].to_numpy(), d["close"].to_numpy(),
    )
    nd = len(d)
    bs_day = np.zeros(nd, dtype=bool)
    bl_day = np.zeros(nd, dtype=bool)
    for t in range(nd):  # t = "오늘"(미완성) — 판정은 t-1 .. t-days 마감일
        for j in range(1, days + 1):
            i = t - j
            if i < 10:
                break
            mid = (lows[i] + highs[i]) / 2.0
            if lows[i] < lows[i - 10:i].min() and closes[i] > mid:
                bs_day[t] = True
            if highs[i] > highs[i - 10:i].max() and closes[i] < mid:
                bl_day[t] = True
    idx = d.index.get_indexer(df.index.floor("1D"), method="ffill")
    bs = np.zeros(n, dtype=bool)
    bl = np.zeros(n, dtype=bool)
    ok = idx >= 0
    bs[ok] = bs_day[idx[ok]]
    bl[ok] = bl_day[idx[ok]]
    return bs, bl


# TF 라벨 → pandas resample rule. 1d 는 UTC 자정, 1w 는 월요일 시작(거래소 정합).
_TF_RESAMPLE_RULE: dict[str, str] = {
    "5m": "5min", "15m": "15min", "1h": "1h", "2h": "2h",
    "4h": "4h", "1d": "1D", "1w": "W-MON",
}


def _resample_tf(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """HTF FVG map 용 TF 리샘플 — 주봉만 월요일 시작(closed/label='left')."""
    rule = _TF_RESAMPLE_RULE[tf]
    kw = {"closed": "left", "label": "left"} if tf == "1w" else {}
    r = df.resample(rule, **kw)
    return pd.DataFrame({
        "open": r["open"].first(), "high": r["high"].max(),
        "low": r["low"].min(), "close": r["close"].last(),
    }).dropna()


@dataclass(slots=True)
class _HtfFvgRec:
    """HTF FVG 1건 + 활성 구간. build_htf_fvg_map 결과 1건에 대응."""

    weight: int
    is_bull: bool
    low: float
    high: float
    start_e: int   # 이 TF 의 닫힌봉 index 가 이 값 이상이면 map 에 등장
    end_e: int     # 이 값 **이하**까지 활성 (체결/무효화/200봉 창 이탈)


def _htf_fvg_records(df: pd.DataFrame, cfg: BacktestConfig) -> dict[str, list]:
    """TF 별 HTF FVG 활성 구간 레코드.

    출처: strategy/htf_fvg_map.py:73-115 ``build_htf_fvg_map``
        (TF 마다 limit=200 봉 fetch → detect_fvgs(min_size_pct)
         → mark_filled_and_invalidated → filled/invalidated 아닌 것만 채택,
         TF_WEIGHT 가중치 부여).
    라이브는 5분마다 재빌드하지만 200봉 창 안에서만 보므로, FVG 1건의 활성
    구간은 [중간봉+1, min(중간봉+198, 체결봉-1, 무효화봉-1)] 로 **정확히**
    닫힌 구간이 된다(창 시작 = e-199 ≤ 중간봉).

    ⚠️ 미완성 TF 봉이 FVG 3번째 봉이 되거나 체결 판정에 쓰이는 경우는
       재현하지 않는다 — 이는 이미 선언된 GAP "미완성 봉(60초 폴링)" 의
       부분집합이며, 근사하지 않고 닫힌 봉만 쓴다.

    Args:
        df: 매매 TF OHLCV (DatetimeIndex UTC).
        cfg: htf_fvg_tfs / htf_fvg_limit / fvg_min_size_pct 사용.
    Returns:
        {tf: (tf_index, [_HtfFvgRec, ...])} — tf_index 는 그 TF 봉의 DatetimeIndex.
    """
    out: dict[str, list] = {}
    lim = cfg.htf_fvg_limit
    for tf in cfg.htf_fvg_tfs:
        if tf not in _TF_RESAMPLE_RULE:
            continue
        weight = TF_WEIGHT.get(tf, 1)
        tdf = _resample_tf(df, tf)
        if len(tdf) < 5:
            continue
        fvgs = detect_fvgs(tdf, min_size_pct=cfg.fvg_min_size_pct)
        if not fvgs:
            out[tf] = (tdf.index, [])
            continue
        hs = tdf["high"].to_numpy()
        ls = tdf["low"].to_numpy()
        cs = tdf["close"].to_numpy()
        recs: list[_HtfFvgRec] = []
        nb = len(tdf)
        for fv in fvgs:
            # mark_filled_and_invalidated(indicators/fvg.py:203-240) 재현 —
            # 중간봉+2 부터 스캔, 체결(mean_threshold 터치) / 무효화(종가 이탈).
            mt = fv.mean_threshold
            fill_j = inv_j = nb  # nb = "끝까지 없음"
            is_bull = fv.type is FVGType.BULLISH
            for j in range(fv.idx + 2, nb):
                if is_bull:
                    if fill_j == nb and ls[j] <= mt:
                        fill_j = j
                    if cs[j] < fv.low:
                        inv_j = j
                        break
                else:
                    if fill_j == nb and hs[j] >= mt:
                        fill_j = j
                    if cs[j] > fv.high:
                        inv_j = j
                        break
            end_e = min(fv.idx + lim - 2, fill_j - 1, inv_j - 1, nb - 1)
            start_e = fv.idx + 1
            if end_e < start_e:
                continue
            recs.append(_HtfFvgRec(
                weight=weight, is_bull=is_bull,
                low=float(fv.low), high=float(fv.high),
                start_e=start_e, end_e=end_e,
            ))
        out[tf] = (tdf.index, recs)
    return out


def _precompute_htf_fvg(df: pd.DataFrame, cfg: BacktestConfig) -> dict[str, np.ndarray]:
    """HTF FVG map 을 봉별 집계 배열로 — supporting boost / 충돌 게이트 / flip 공용.

    라이브 대응:
      - ``sup_long``  = LONG 셋업의 supporting 가중치 합
        (find_supporting_htf_fvg(buy): bullish & ``e.high < price``)
      - ``sup_short`` = SHORT 셋업의 supporting 가중치 합
        (bearish & ``e.low > price``)
      - **같은 집합이 반대 방향의 opposite 집합**이다
        (find_opposite_htf_fvg(buy) = bearish & low>price = sup_short 집합).
        htf_fvg_map.py:143-162 / 192-210 을 대조하면 완전 동일 조건.
      - ``tot_bull`` / ``tot_bear`` = 가격 무관 전체 가중치 합
        (bot_ict_instance.py:1485-1492 #HTF-LTF-CONFLICT 용)
      - ``flip_lo_long``/``flip_hi_long`` = LONG 포지션의 flip target zone
        (_evaluate_htf_override: 합산>max(5m가중치*3,6)=6 통과 후
         weight>=_FLIP_TARGET_MIN_WEIGHT(4) 인 것 중 |mid-price| 최소)

    touch_count 는 flip_watcher(WS) 만 올리는데 SaaS 는
    ``flip_watch_enabled=False``(multi_user_manager.py) 라 라이브에서 항상 0 →
    max_touch_count 필터가 발동하지 않는다. 그래서 여기서도 0 으로 둔다(근사 아님).

    Args:
        df: 매매 TF OHLCV.
        cfg: htf_fvg_* 필드 사용.
    Returns:
        dict of len(df) 배열.
    """
    n = len(df)
    price = df["close"].to_numpy(dtype=float)
    per_tf = _htf_fvg_records(df, cfg)
    # 각 TF 의 "지금까지 닫힌 봉 index" 를 매매 TF 봉마다 매핑.
    tf_e: dict[str, np.ndarray] = {}
    tf_recs: dict[str, list] = {}
    for tf, (tidx, recs) in per_tf.items():
        # searchsorted(right) - 1 = 이 시각까지 **시작된** TF 봉. 그 봉은 아직
        # 미완성이므로 닫힌 마지막 봉은 그보다 하나 앞.
        pos = np.searchsorted(tidx.to_numpy(), df.index.to_numpy(), side="right") - 1
        tf_e[tf] = pos - 1
        tf_recs[tf] = recs
    # 활성 레코드 집합이 바뀌는 지점만 재구성 (TF 봉이 닫힐 때만 바뀜).
    keys = list(tf_e.keys())
    if not keys:
        z = np.zeros(n, dtype=np.float32)
        nanarr = np.full(n, np.nan)
        return {"sup_long": z, "sup_short": z.copy(),
                "tot_bull": z.copy(), "tot_bear": z.copy(),
                "flip_lo_long": nanarr, "flip_hi_long": nanarr.copy(),
                "flip_lo_short": nanarr.copy(), "flip_hi_short": nanarr.copy()}
    sup_long = np.zeros(n, dtype=np.float32)
    sup_short = np.zeros(n, dtype=np.float32)
    tot_bull = np.zeros(n, dtype=np.float32)
    tot_bear = np.zeros(n, dtype=np.float32)
    flip_lo_l = np.full(n, np.nan)
    flip_hi_l = np.full(n, np.nan)
    flip_lo_s = np.full(n, np.nan)
    flip_hi_s = np.full(n, np.nan)
    # 5m LTF 가중치 기준 threshold — _evaluate_htf_override:4484
    ltf_w = TF_WEIGHT.get("5m", 1)
    thr = max(ltf_w * 3, 6)
    flip_min_w = 4  # bot_ict_instance.py:294 _FLIP_TARGET_MIN_WEIGHT

    state = None
    cur_w = cur_lo = cur_hi = cur_mid = None
    cur_bull = None
    i = 0
    while i < n:
        st = tuple(int(tf_e[k][i]) for k in keys)
        if st != state:
            state = st
            ws: list[int] = []
            los: list[float] = []
            his: list[float] = []
            bulls: list[bool] = []
            for k, e in zip(keys, st, strict=False):
                if e < 0:
                    continue
                for r in tf_recs[k]:
                    if r.start_e <= e <= r.end_e:
                        ws.append(r.weight)
                        los.append(r.low)
                        his.append(r.high)
                        bulls.append(r.is_bull)
            cur_w = np.asarray(ws, dtype=np.float64)
            cur_lo = np.asarray(los, dtype=np.float64)
            cur_hi = np.asarray(his, dtype=np.float64)
            cur_mid = (cur_lo + cur_hi) / 2.0 if len(ws) else cur_lo
            cur_bull = np.asarray(bulls, dtype=bool)
        # 같은 상태가 유지되는 구간 끝 찾기
        j = i + 1
        while j < n and tuple(int(tf_e[k][j]) for k in keys) == state:
            j += 1
        if len(cur_w):
            tb = float(cur_w[cur_bull].sum())
            tr = float(cur_w[~cur_bull].sum())
            tot_bull[i:j] = tb
            tot_bear[i:j] = tr
            for b in range(i, j):
                px = price[b]
                m_bb = cur_bull & (cur_hi < px)    # bullish, 가격 아래 = LONG 지지
                m_ba = (~cur_bull) & (cur_lo > px)  # bearish, 가격 위 = SHORT 지지
                w_bb = float(cur_w[m_bb].sum())
                w_ba = float(cur_w[m_ba].sum())
                sup_long[b] = w_bb
                sup_short[b] = w_ba
                # LONG 포지션의 flip target = 반대(bearish above)
                if w_ba > thr:
                    sel = m_ba & (cur_w >= flip_min_w)
                    if sel.any():
                        idx = np.argmin(np.abs(cur_mid[sel] - px))
                        flip_lo_l[b] = cur_lo[sel][idx]
                        flip_hi_l[b] = cur_hi[sel][idx]
                if w_bb > thr:
                    sel = m_bb & (cur_w >= flip_min_w)
                    if sel.any():
                        idx = np.argmin(np.abs(cur_mid[sel] - px))
                        flip_lo_s[b] = cur_lo[sel][idx]
                        flip_hi_s[b] = cur_hi[sel][idx]
        i = j
    return {
        "sup_long": sup_long, "sup_short": sup_short,
        "tot_bull": tot_bull, "tot_bear": tot_bear,
        "flip_lo_long": flip_lo_l, "flip_hi_long": flip_hi_l,
        "flip_lo_short": flip_lo_s, "flip_hi_short": flip_hi_s,
    }


def _htf_support_boost(total_weight: float) -> int:
    """supporting 가중치 합 → 계단식 boost.

    출처: bot_ict_instance.py:4539-4551 (_apply_htf_supporting_boost).
        2026-06-04 최저 임계 4→2 완화가 **실제 코드값**이라 그대로 따른다:
            합산 >= 20 → +3 / >= 10 → +2 / >= 2 → +1 / 그 외 0
    """
    if total_weight >= 20:
        return 3
    if total_weight >= 10:
        return 2
    if total_weight >= 2:
        return 1
    return 0


def _dol_counter_delta(
    window: pd.DataFrame, direction: Direction, penalty: int,
) -> tuple[int, str | None]:
    """#3 보완 DOL 역방향 **감점** — 라이브 그대로.

    출처: bot_ict_instance.py:3104-3151 ``_apply_dol_bias`` + 상수
        ``_DOL_COUNTER_PENALTY = 2`` (line 280).
        지배적 draw = compute_dol 이 돌려주는 bullish/bearish 중 거리가 가까운 쪽
        (compute_dol 은 방향별 최대 1건이라 ``next(...)`` == 최근접).
        **detect_liquidity_sweeps 를 먼저 불러 swept 를 마킹**한다 — 기존 백테
        cfg.apply_dol 은 이 호출이 없어 이미 먹힌 유동성까지 DOL 로 봤다.

    Args:
        window: 진입 시점까지의 매매 TF 슬라이스.
        direction: 셋업 방향.
        penalty: 감점 폭(라이브 2).
    Returns:
        (점수 delta(0 또는 -penalty), confluences 문자열 또는 None)
    """
    if window is None or len(window) < 5:
        return 0, None
    swings = detect_swing_points(window)
    if not swings:
        return 0, None
    detect_liquidity_sweeps(window, swings)   # swept 마킹 (라이브 3117 동일)
    dols = compute_dol(window, swings)
    if not dols:
        return 0, None
    bull = next((d for d in dols if d.type == "bullish"), None)
    bear = next((d for d in dols if d.type == "bearish"), None)
    if bull is not None and bear is not None:
        draw = Direction.LONG if bull.distance < bear.distance else Direction.SHORT
    elif bull is not None:
        draw = Direction.LONG
    elif bear is not None:
        draw = Direction.SHORT
    else:
        return 0, None
    if direction is not draw:
        return -penalty, f"dol_counter_{draw.value}_-{penalty}"
    return 0, None


def _live_candidate_setups(
    window: pd.DataFrame, cfg: BacktestConfig, ote_level: float,
) -> list:
    """라이브 ``generate_ict_signal`` 의 **후보 셋업 생성**을 그대로 재현.

    출처: signal/ict_signal.py:103-148.
        1) detect_silver_bullet_setups (Phase A — FVG/IFVG)
        2) build_extra_source_setups (Phase B — turtle/mitigation/implied/rejection)
           → 합친 뒤 anchor_idx 로 정렬
        3) #MIN-SL-EXTRA: 합친 **후** min_sl_distance_pct 가드 일괄 적용
           (Phase B 가 이 인자를 안 받아 SL 가드를 우회하던 2026-05-29 버그 fix)

    prefer_direction 필터와 stale 검사는 여기서 하지 않는다 — 재생 단계에서
    align 점수(스윕 대상)와 함께 적용해야 타임라인을 재사용할 수 있기 때문.

    Args:
        window: 슬라이딩 윈도우 슬라이스.
        cfg: detect 파라미터.
        ote_level: 이 시점의 OTE 깊이 (#REGIME-OTE 로 봉마다 다를 수 있음).
    Returns:
        anchor_idx 오름차순 후보 셋업 리스트.
    """
    setups = detect_silver_bullet_setups(
        window,
        min_rr=cfg.min_rr,
        fvg_min_size_pct=cfg.fvg_min_size_pct,
        min_confluence=0,   # 게이트는 재생 단계에서 — 라이브도 bot 이 판정
        expand_to_killzone=cfg.expand_to_killzone,
        disable_time_filter=cfg.disable_time_filter,
        min_sl_distance_pct=cfg.min_sl_distance_pct,
        ote_level=ote_level,
        nyse_gate=cfg.nyse_gate,
        window_once=cfg.window_once,
        max_per_fvg=cfg.max_per_fvg,
    )
    if not cfg.phase_b_sources:
        return list(setups)
    extra = build_extra_source_setups(
        window,
        min_rr=cfg.min_rr,
        bias=None,   # 라이브는 HTF bias 를 넘기지만 #BIAS-DIRECTION 이후 방향
                     # 강제에 쓰이지 않는다(GAPS "bias 주입" 참고).
        disable_time_filter=cfg.disable_time_filter,
        research_sources=cfg.research_sources,
            enable_turtle_soup=cfg.turtle_soup_enabled,
            nyse_gate=cfg.nyse_gate,
    )
    if extra:
        setups = list(setups) + list(extra)
        setups.sort(key=lambda s: s.anchor_idx)
    if cfg.min_sl_distance_pct > 0 and setups:
        kept = []
        for s in setups:
            entry_v = float(getattr(s, "entry", 0.0) or 0.0)
            sl_v = float(getattr(s, "stop_loss", 0.0) or 0.0)
            if entry_v <= 0 or sl_v <= 0:
                continue
            if (abs(entry_v - sl_v) / entry_v) >= cfg.min_sl_distance_pct:
                kept.append(s)
        setups = kept
    return list(setups)


def _parity_flags_on(cfg: BacktestConfig) -> bool:
    """2026-08-08 정합 이식 플래그가 하나라도 켜져 있는지."""
    return bool(
        cfg.phase_b_sources or cfg.prefer_direction_select or cfg.align_live_formula
        or cfg.htf_fvg_support or cfg.htf_ltf_conflict_guard_ratio > 0
        or cfg.htf_fvg_flip or cfg.apply_dol_counter or cfg.trail_supersedes_partial
        or cfg.entry_killzone_gate or cfg.exclude_nypm
        or cfg.max_entry_distance_pct > 0 or cfg.sweep_gate_days > 0
        or cfg.ote_up_level > 0,
    )


def run_backtest(
    df: pd.DataFrame, cfg: BacktestConfig, corr_df: pd.DataFrame | None = None,
) -> BacktestResult:
    """슬라이딩 윈도우 백테스트 실행.

    Args:
        df: 1m OHLCV (DatetimeIndex, columns open/high/low/close). load_ohlcv_parquet 산출.
        cfg: 파라미터.
        corr_df: SMT boost 용 상관 심볼 OHLCV (apply_smt=True 시 필요, 같은 기간 정렬).

    Returns:
        BacktestResult — per-trade 기록 + 집계.

    Raises:
        ValueError: 2026-08-08 라이브 정합 플래그가 켜진 채 호출될 때. 정합 경로는
            build_setup_timeline + run_backtest_from_timeline 에만 이식했다
            (라이브 셋업 선택이 방향별 타임라인을 필요로 해 구조가 다르다).
            Origo 정합 연구는 scripts/live_parity.py 의 run_live_parity 를 쓸 것.
    """
    if _parity_flags_on(cfg):
        raise ValueError(
            "정합 이식 플래그(phase_b_sources/prefer_direction_select/... )는 "
            "run_backtest 에 미구현. run_backtest_from_timeline(또는 "
            "scripts/live_parity.run_live_parity)을 쓸 것 — 조용히 다른 봇이 "
            "되는 것을 막으려고 일부러 막아 둔다.",
        )
    n = len(df)
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    # #SHORT-BIAS: HTF EMA bias 게이트용 1h EMA (off 면 계산 skip).
    htf_ema = (
        _precompute_htf_ema(df, cfg.htf_ema_period).to_numpy()
        if cfg.htf_ema_bias in ("strict", "band") else None
    )
    # align 점수 — 진입 게이트("align") 또는 동적 flip 청산 중 하나라도 쓰면 계산.
    htf_align = (
        _precompute_align_score(df, cfg.htf_align_periods).to_numpy()
        if (cfg.htf_ema_bias == "align" or cfg.htf_align_flip) else None
    )
    # #MSS-FLIP: 보유 중 반대 CHoCH 조기 청산용 봉별 신호 (mss_flip 일 때만).
    mss_sig = (
        _precompute_mss_signal(df, cfg.mss_swing_left, cfg.mss_swing_right)
        if cfg.mss_flip else None
    )
    # #MSS-BIAS-GATE: 진입 방향 제한용 구조 방향 (mss_bias_gate 일 때만).
    mss_bias = (
        _precompute_mss_bias(df, cfg.mss_swing_left, cfg.mss_swing_right)
        if (cfg.mss_bias_gate or cfg.align_mss_fill) else None
    )
    # #REGIME: 국면 적응 게이트용 EMA 스프레드 (regime_adaptive 일 때만).
    ema_spread_arr = (
        _precompute_ema_spread(df, cfg.htf_align_periods)
        if cfg.regime_adaptive else None
    )
    trades: list[Trade] = []
    i = cfg.window
    while i < n - 1:
        window = df.iloc[i - cfg.window : i + 1]
        setups = detect_silver_bullet_setups(
            window,
            min_rr=cfg.min_rr,
            fvg_min_size_pct=cfg.fvg_min_size_pct,
            min_confluence=0,  # 게이트는 하니스(_gate_pass)에서 — bot step() 재현
            expand_to_killzone=cfg.expand_to_killzone,
            disable_time_filter=cfg.disable_time_filter,
            min_sl_distance_pct=cfg.min_sl_distance_pct,
            ote_level=cfg.ote_level,
            nyse_gate=cfg.nyse_gate,
            window_once=cfg.window_once,
            max_per_fvg=cfg.max_per_fvg,
        )
        if not setups:
            i += 1
            continue
        setup = setups[-1]
        bars_since = len(window) - 1 - setup.anchor_idx
        if bars_since > cfg.setup_stale_bars:
            i += 1
            continue
        # #PD-FILTER: ICT 프리미엄/디스카운트 — 진입가가 방향에 맞는 존(LONG=디스카운트/
        # SHORT=프리미엄)일 때만 진입. 숏 한쪽 쏠림 구조 차단. swing 부족 시 통과.
        if cfg.apply_pd_filter and not _pd_pass(window, setup.direction, setup.entry):
            i += 1
            continue
        # bot 레벨 boost(CISD/SMT) 반영 — confluence_score 가산 후 게이트 (bot step 재현).
        corr_window = (
            corr_df.iloc[i - cfg.window : i + 1] if corr_df is not None else None
        )
        boost_names: list[str] = []   # 2026-08-07 조합 실험용 — 발동 boost 기록
        score = _boost_score(
            setup.confluence_score, setup.direction, window, corr_window, cfg,
            detail=boost_names,
        )
        if not _gate_pass(score, setup.risk_reward, cfg):
            i += 1
            continue
        # #SHORT-BIAS: HTF EMA bias 방향 게이트 (실시간 _passes_htf_ema_bias 재현).
        if htf_ema is not None:
            ema_v = htf_ema[i]
            if ema_v == ema_v:  # NaN 아니면 (초기 구간 NaN 은 게이트 skip = 허용)
                cl = float(closes[i])
                band = ema_v * cfg.htf_ema_band_pct
                is_long = setup.direction is Direction.LONG
                if cfg.htf_ema_bias == "strict":
                    blocked = (cl > ema_v and not is_long) or (cl < ema_v and is_long)
                else:  # "band" — 완충대 밖에서만 방향 강제, 안이면 양방향 허용
                    blocked = (
                        (cl > ema_v + band and not is_long)
                        or (cl < ema_v - band and is_long)
                    )
                if blocked:
                    i += 1
                    continue
        # "align" — 다중 EMA 정렬 점수 게이트 (조윤 EMA 가중치).
        # #REGIME: 반등 의심(EMA 스프레드 < 임계)이면 align 게이트 풀기(양방향 허용).
        regime_off = (
            cfg.regime_adaptive and ema_spread_arr is not None
            and ema_spread_arr[i] < cfg.regime_spread_thr
        )
        if htf_align is not None and cfg.htf_ema_bias == "align" and not regime_off:
            sc = htf_align[i]
            if sc == sc:  # NaN 아니면
                is_long = setup.direction is Direction.LONG
                t = cfg.htf_align_threshold
                if sc >= t:
                    blocked = not is_long       # 상승추세 → 롱만
                elif sc <= -t:
                    blocked = is_long           # 하락추세 → 숏만
                else:
                    # 애매구간(되돌림/횡보): 기본 진입자제. align_mss_fill 이면
                    # 검증된 EMA 와 MSS 구조 결합 — MSS 방향으로만 진입 허용.
                    if cfg.align_mss_fill and mss_bias is not None and mss_bias[i] != 0:
                        mb = mss_bias[i]
                        blocked = (mb == 1 and not is_long) or (mb == -1 and is_long)
                    else:
                        blocked = True              # 추세 불명확 → 진입 자제
                if blocked:
                    i += 1
                    continue
        # #MSS-BIAS-GATE: 마지막 CHoCH 방향과 반대 진입 차단 (EMA 대체/병행).
        if mss_bias is not None:
            mb = mss_bias[i]
            if mb != 0:
                is_long = setup.direction is Direction.LONG
                if (mb == 1 and not is_long) or (mb == -1 and is_long):
                    i += 1
                    continue
        # 체결 시뮬 — limit(setup.entry)에 ttl 봉 내 가격이 닿아야 체결. 미체결이면 skip.
        fill_idx = _simulate_fill(
            highs, lows, i, setup.direction, setup.entry, cfg.entry_ttl_bars,
        )
        if fill_idx is None:
            i += 1
            continue  # 타점 미도달 — 미체결(타점 포기)
        d_val = setup.direction.value
        # limit 체결이라 entry 슬리피지 0 (계획가 그대로). 청산만 슬리피지(시장가 발동).
        entry = setup.entry
        # 흑자 탐색: SL 거리 변형 (stop-hunt 회피 vs 손실크기 트레이드오프).
        sl, tp = setup.stop_loss, setup.take_profit
        eff_mult = _effective_sl_mult(cfg, closes, fill_idx, setup.direction is Direction.LONG)
        if eff_mult != 1.0:
            risk = abs(entry - sl)
            if risk > 0:
                rr0 = abs(tp - entry) / risk
                new_risk = risk * eff_mult
                if cfg.sl_liq_cap and cfg.leverage > 0:
                    cap = entry * 0.8 / cfg.leverage
                    if new_risk > cap:
                        new_risk = max(risk, cap)  # 라이브 #LIQ-CAP 동일
                if setup.direction is Direction.LONG:
                    sl = entry - new_risk
                    tp = entry + new_risk * rr0 if cfg.tp_keeps_rr else tp
                else:
                    sl = entry + new_risk
                    tp = entry - new_risk * rr0 if cfg.tp_keeps_rr else tp
        # #TP-RR: TP 를 risk 의 고정 배수로 강제 (승률↑·RR↓ 트레이드오프 연구).
        if cfg.tp_rr_override > 0:
            risk2 = abs(entry - sl)
            tp = (
                entry + risk2 * cfg.tp_rr_override
                if setup.direction is Direction.LONG
                else entry - risk2 * cfg.tp_rr_override
            )
        exit_idx, exit_raw, outcome = _simulate_exit(
            opens, highs, lows, closes, fill_idx, setup.direction,
            sl, tp, cfg,
            align_score=htf_align if cfg.htf_align_flip else None,
            mss_signal=mss_sig,
            entry=entry,
        )
        exit_slip = slip_pct(
            float(highs[exit_idx]), float(lows[exit_idx]), float(closes[exit_idx]),
        )
        exit_price = apply_slippage(exit_raw, d_val, "exit", exit_slip)
        sign = 1.0 if setup.direction is Direction.LONG else -1.0
        raw_pnl_pct = (exit_price - entry) / entry * sign
        net_pnl_pct, _ = apply_costs(raw_pnl_pct, cfg.size_pct, cfg.leverage)
        trades.append(Trade(
            entry_idx=fill_idx, exit_idx=exit_idx, direction=d_val,
            entry=entry, exit_price=exit_price, outcome=outcome,
            raw_pnl_pct=raw_pnl_pct, net_pnl_pct=net_pnl_pct,
            confluence_score=score,
            entry_atr_pct=_entry_atr_pct(highs, lows, closes, fill_idx, entry),
            entry_trend_pct=_entry_trend_pct(closes, fill_idx),
            entry_sl=float(sl), entry_tp=float(tp),
            confluences=tuple(setup.confluences),
            base_score=int(setup.confluence_score),
            boosts=tuple(boost_names),
        ))
        i = exit_idx + 1  # 청산 후 다음 봉부터 재탐색 (동시 포지션 1개)
    return _aggregate(cfg, trades)


def _aggregate(cfg: BacktestConfig, trades: list[Trade]) -> BacktestResult:
    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t.net_pnl_pct > 0)
    total = sum(t.net_pnl_pct for t in trades)
    longs = sum(1 for t in trades if t.direction == "long")
    return BacktestResult(
        config=cfg,
        trades=trades,
        n_trades=n_trades,
        n_wins=n_wins,
        win_rate=(n_wins / n_trades) if n_trades else 0.0,
        total_net_pnl_pct=total,
        avg_net_pnl_pct=(total / n_trades) if n_trades else 0.0,
        long_count=longs,
        short_count=n_trades - longs,
    )


def _detect_params(cfg: BacktestConfig) -> dict[str, Any]:
    """detect_silver_bullet_setups 결과에 영향 주는 cfg 필드만 추출.

    타임라인 캐시(build_setup_timeline)의 유효 조건 = 이 필드들이 전부 같을 것.
    conf 게이트(min_confluence)·sl_dist_mult·entry_ttl_bars·align threshold 등은
    detect 와 무관하므로 여기 포함하지 않는다 (그래서 조합 간 타임라인 공유 가능).

    Args:
        cfg: 백테스트 파라미터.

    Returns:
        detect 입력 인자에 대응하는 cfg 필드 dict (검증/기록용).
    """
    return {
        "min_rr": cfg.min_rr,
        "fvg_min_size_pct": cfg.fvg_min_size_pct,
        "expand_to_killzone": cfg.expand_to_killzone,
        "disable_time_filter": cfg.disable_time_filter,
        "min_sl_distance_pct": cfg.min_sl_distance_pct,
        "window": cfg.window,
        # 2026-08-08 정합 이식 — 셋업 **생성**을 바꾸는 것들이라 detect 인자와 동급.
        # phase_b_sources: 후보 셋업 집합 자체가 달라짐
        # ote_up_level: 상승 국면 봉에서 진입가(OTE 깊이)가 달라짐
        # prefer_direction_select: 방향별 최신 셋업(dir_items)이 필요 → 타임라인 형태 변경
        "phase_b_sources": cfg.phase_b_sources,
        "ote_up_level": cfg.ote_up_level,
        "prefer_direction_select": cfg.prefer_direction_select,
    }


class SetupTimeline(list):
    """build_setup_timeline 산출 컨테이너 — 평범한 list + detect 파라미터 메타.

    ``timeline[i]`` = i 번째 1m 봉 시점의 detect 결과:
        - ``None``  — setup 없음 (또는 i 가 슬라이딩 범위 밖),
        - ``(setup, bars_since)`` — 가장 최근 setup + 경과 봉 수(stale 판정은
          재생 시 cfg.setup_stale_bars 와 비교해야 하므로 숫자로 보관).

    ``detect_params`` 속성에 빌드 당시 detect 인자(cfg 필드)를 기록해 둬서
    run_backtest_from_timeline 이 cfg 불일치(특히 min_rr)를 잡아낼 수 있다.

    2026-08-08 정합 이식 — ``dir_items`` 속성(옵션) 추가.
        ``dir_items[i] = (setup_long, bars_long, setup_short, bars_short)``.
        라이브는 prefer_direction(EMA align 방향)과 **같은 방향 셋업만** 남기고
        그중 최신을 고르므로(signal/ict_signal.py:163-175), 재생 단계에서
        align 점수에 따라 방향별 최신 셋업을 꺼내 쓸 수 있어야 한다.
        ``cfg.prefer_direction_select=False`` 면 None (구식 타임라인 호환).
    """

    detect_params: dict[str, Any]
    dir_items: list | None = None


def build_setup_timeline(df: pd.DataFrame, cfg: BacktestConfig) -> list:
    """파라미터 그리드 가속용 setup 타임라인 빌드 — detect 결과 캐시.

    run_backtest 와 동일한 슬라이딩(각 i, cfg.window <= i < n-1)으로
    detect_silver_bullet_setups 를 1회씩 돌려 그 결과만 저장한다. 백테스트
    비용의 거의 전부가 이 detect 라서, conf 게이트·SL 배수·entry ttl·align
    threshold 처럼 detect 와 무관한 파라미터만 다른 조합들은 이 타임라인
    하나를 공유해 run_backtest_from_timeline 으로 고속 재생할 수 있다.

    주의: min_rr / fvg_min_size_pct / expand_to_killzone / disable_time_filter /
    min_sl_distance_pct / window 는 detect 인자라서 **이 중 하나라도 다르면
    타임라인을 따로 만들어야 한다** (예: rr sweep 은 rr 별로 빌드).

    Args:
        df: 1m OHLCV (DatetimeIndex UTC). load_ohlcv_parquet 산출.
        cfg: 백테스트 파라미터 — detect 관련 필드(min_rr 등)만 사용.

    Returns:
        길이 len(df) 의 SetupTimeline(list 서브클래스). timeline[i] = None 또는
        (setup, bars_since). detect_params 속성에 빌드 cfg 의 detect 필드 기록.
    """
    n = len(df)
    timeline = SetupTimeline([None] * n)
    timeline.detect_params = _detect_params(cfg)
    # 2026-08-08 #REGIME-OTE: 상승 국면이면 OTE 깊이를 ote_up_level 로.
    # 진입가가 바뀌므로 detect 단계에서 봉마다 적용해야 한다(사후 필터 불가).
    # 출처: bot_ict_instance.py:1244 ote_level=await self._effective_ote().
    regime_up = (
        _precompute_regime_up(df) if cfg.ote_up_level > 0 else None
    )
    if cfg.prefer_direction_select:
        timeline.dir_items = [None] * n
    for i in range(cfg.window, n - 1):
        window = df.iloc[i - cfg.window : i + 1]
        ote = cfg.ote_level
        if regime_up is not None and regime_up[i]:
            ote = cfg.ote_up_level
        setups = _live_candidate_setups(window, cfg, ote)
        if not setups:
            continue
        last = len(window) - 1
        setup = setups[-1]
        timeline[i] = (setup, last - setup.anchor_idx)
        if timeline.dir_items is not None:
            # 라이브 prefer_direction 재현용 — 방향별 **최신** 셋업만 보관.
            s_l = s_s = None
            for s in setups:            # anchor_idx 오름차순 → 마지막이 최신
                if s.direction is Direction.LONG:
                    s_l = s
                else:
                    s_s = s
            timeline.dir_items[i] = (
                s_l, (last - s_l.anchor_idx) if s_l is not None else -1,
                s_s, (last - s_s.anchor_idx) if s_s is not None else -1,
            )
    return timeline


def _mmbm_bias_signs(df: pd.DataFrame, lookback: int = 20) -> np.ndarray:
    """봉별 HTF 추세 부호 — 라이브 `_mmbm_htf_bias_sign` 과 같은 식.

    라이브는 매 step 에서 5m df 를 1h 로 리샘플해 최근 20봉 종가 변화의 부호를 낸다.
    백테는 그걸 봉마다 다시 계산하면 O(n²) 이 되므로, 1h 시리즈를 한 번 만들고
    각 5m 봉을 **그 시점까지 닫힌 1h 봉**에 매핑해 미리 구한다(look-ahead 차단).

    Returns:
        길이 len(df) 의 +1/-1/0 배열.
    """
    out = np.zeros(len(df), dtype=float)
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) == 0:
        return out
    try:
        h1 = df["close"].resample("1h").last().dropna()
    except (TypeError, ValueError):
        return out
    if len(h1) < lookback + 1:
        return out
    delta = h1.to_numpy(float)[lookback:] - h1.to_numpy(float)[:-lookback]
    sign = np.sign(delta)                       # +1 / -1 / 0
    valid_ts = h1.index[lookback:]
    # 각 5m 봉 → 그 시점 이하의 마지막 1h 봉 (searchsorted, 우측 배타)
    pos = np.searchsorted(valid_ts.values, df.index.values, side="right") - 1
    ok = pos >= 0
    out[ok] = sign[pos[ok]]
    return out


def _precompute_mmbm(df: pd.DataFrame, cfg: BacktestConfig) -> dict[int, object]:
    """#MMBM-BT 2026-08-10: 봉별 MMBM 셋업 — 라이브와 같은 진입점을 쓴다.

    라이브는 매 봉 `detect_mmbm_setup(df, bias_sign, ...)` 을 부르는데, 그 함수는
    **최근 `_FRESH`(2)봉 안에 형성된 CHoCH** 만 본다. 그래서 구조 이벤트 근처가
    아닌 봉은 항상 None 이다. 52만 봉 전수 호출 대신 **CHoCH 이벤트 idx ~ idx+2**
    만 평가해 같은 결과를 훨씬 싸게 얻는다.

    Returns:
        {봉 인덱스: SilverBulletSetup} — 그 봉에서 MMBM 이 잡은 셋업.
    """
    if not cfg.mmbm_enabled or len(df) < 60:
        return {}
    from aurora_ict.indicators.structure import StructureType, detect_structure_events
    from aurora_ict.indicators.swing_points import detect_swing_points
    from aurora_ict.strategy.mmbm import _FRESH, detect_mmbm_setup

    signs = _mmbm_bias_signs(df)
    swings = detect_swing_points(df, left=3, right=3)
    events = detect_structure_events(df, swings)
    cand: set[int] = set()
    for ev in events:
        if ev.type not in (StructureType.CHOCH_BULLISH, StructureType.CHOCH_BEARISH):
            continue
        for k in range(int(ev.idx), int(ev.idx) + _FRESH + 1):
            if 0 <= k < len(df):
                cand.add(k)

    out: dict[int, object] = {}
    for i in sorted(cand):
        # 그 봉까지만 잘라 넘긴다 — 라이브가 보는 것과 같은 창(look-ahead 차단)
        lo = max(0, i + 1 - cfg.window)
        # 예외를 삼키지 않는다. 처음엔 `except Exception: continue` 로 감쌌다가
        # `SetupSource.MMBM` 미정의(연구 사본이 구버전)로 **전 구간 0건**이 나왔고,
        # 조용히 "MMBM 은 진입이 없다"는 결론이 될 뻔했다. 오늘 라이브 코드에서
        # 같은 패턴(mmbm_full 의 SMT except)을 찾아낸 직후에 똑같이 반복한 것이다.
        # 게이트를 끄는 except 는 로그가 아니라 **실패**여야 한다.
        st = detect_mmbm_setup(
            df.iloc[lo : i + 1],
            float(signs[i]),
            min_rr=cfg.min_rr,
            fvg_min_size_pct=cfg.fvg_min_size_pct,
        )
        if st is not None:
            out[i] = st
    return out


def _mmbm_fill_gaps(
    df: pd.DataFrame,
    sb_trades: list[Trade],
    mmbm_map: dict[int, object],
    cfg: BacktestConfig,
    funding: pd.DataFrame | None = None,
) -> list[Trade]:
    """SB 가 비운 구간에 MMBM 진입을 채워 넣는다 — 라이브 순서 재현.

    라이브는 봉마다 ① SB 셋업을 보고 ② 없거나 게이트를 못 넘으면 MMBM 을 시도하며,
    **포지션이 있으면 어느 모델이든 신규 진입을 하지 않는다**. 그래서 SB 백테를
    그대로 두고 그 보유 구간 밖에서만 MMBM 을 채우면 같은 결과가 된다.
    재생 루프 안에 끼워 넣지 않는 이유: 루프에 조기 이탈 지점이 8곳이라 전부
    고쳐야 하고, 하나라도 놓치면 조용히 어긋난다.

    MMBM 은 confluence 게이트를 우회하지만(자체 조건으로 검증) 청산 규칙·비용·
    사이징 기록은 SB 와 공유한다 — 라이브 `_execute_setup` 이 공통 경로인 것과 같다.

    Args:
        df: 5m OHLCV.
        sb_trades: SB 경로 체결 목록 (시간순).
        mmbm_map: `_precompute_mmbm` 결과 {봉 인덱스: 셋업}.
        cfg: 백테 설정.
        funding: 펀딩 요율 (있으면 net 에서 차감).

    Returns:
        MMBM 체결 목록. SB 것과 합쳐 시간순 정렬해 쓴다.
    """
    if not mmbm_map:
        return []
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    volumes = (
        df["volume"].to_numpy() if "volume" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    busy = sorted((int(t.entry_idx), int(t.exit_idx)) for t in sb_trades)

    def _blocked(i: int) -> bool:
        """i 봉에 SB 포지션이 열려 있나 (진입봉~청산봉 포함)."""
        for a, b in busy:
            if a <= i <= b:
                return True
            if a > i:
                break
        return False

    out: list[Trade] = []
    free_from = 0        # 직전 MMBM 청산 다음 봉부터 재진입 가능
    for i in sorted(mmbm_map):
        if i < free_from or i >= len(df) - 1 or _blocked(i):
            continue
        st = mmbm_map[i]
        d_val = st.direction
        is_long = d_val is Direction.LONG
        entry = float(st.entry)
        sl = float(st.stop_loss)
        tp = float(st.take_profit)
        if entry <= 0 or sl <= 0 or abs(entry - sl) <= 0:
            continue
        # ★ 체결 시뮬 — MMBM 진입가는 FVG mean(할인/프리미엄 자리)이라 **가격이
        # 되돌아와야 체결된다**. 라이브도 지정가를 걸고 TTL 안에 안 닿으면 진입이
        # 없다. 이걸 빼고 무조건 체결로 두면 유리한 자리만 골라 잡는 셈이 되어
        # 건당 R 이 +0.87 까지 부풀었다(실제 mmbm_full 실측 +0.199R).
        fill_idx = _simulate_fill(highs, lows, i, d_val, entry, cfg.entry_ttl_bars)
        if fill_idx is None:
            continue
        if _blocked(fill_idx):
            continue
        exit_idx, exit_raw, outcome = _simulate_exit(
            opens, highs, lows, closes, fill_idx, d_val, sl, tp, cfg, entry=entry,
        )
        # SB 가 잡고 있는 구간으로 넘어가면 그 앞에서 끊는다(동시보유 금지)
        for a, b in busy:
            if fill_idx < a <= exit_idx:
                exit_idx = a - 1
                exit_raw = float(closes[exit_idx])
                outcome = "cut_sb"
                break
        if exit_idx <= fill_idx:
            continue
        exit_slip = slip_pct(
            float(highs[exit_idx]), float(lows[exit_idx]), float(closes[exit_idx]),
        )
        exit_price = apply_slippage(exit_raw, d_val, "exit", exit_slip)
        sign = 1.0 if is_long else -1.0
        raw_pnl_pct = (exit_price - entry) / entry * sign
        net_pnl_pct, _ = apply_costs(raw_pnl_pct, cfg.size_pct, cfg.leverage)
        fund_pct = _funding_cost(funding, df.index, fill_idx, exit_idx, is_long)
        if fund_pct:
            net_pnl_pct -= fund_pct * cfg.size_pct * cfg.leverage
        out.append(Trade(
            entry_idx=fill_idx, exit_idx=exit_idx, direction=d_val,
            entry=entry, exit_price=exit_price, outcome=outcome,
            raw_pnl_pct=raw_pnl_pct, net_pnl_pct=net_pnl_pct,
            funding_pct=fund_pct,
            smart_size_scale=(
                _smart_size_scale(closes, volumes, fill_idx, is_long)
                if cfg.smart_size_enabled else 1.0
            ),
            # MMBM 은 confluence 게이트를 우회하므로 점수가 없다. 라이브
            # `_calc_qty_risk_based` 는 setup.confluence_score 를 그대로 쓰는데
            # MMBM 셋업의 기본값이 0 이라 리스크%가 base(3.0)로 들어간다.
            risk_pct_used=_risk_pct_for(int(getattr(st, "confluence_score", 0)), cfg),
            entry_sl=sl, entry_tp=tp,
            # MMBM 은 confluence 게이트를 우회하므로 점수 개념이 없다. 0 으로 두면
            # 분석 스크립트가 "저점수 진입"으로 오해할 수 있어 source 를 남긴다.
            confluence_score=int(getattr(st, "confluence_score", 0)),
            confluences=("mmbm",), base_score=0, boosts=(),
        ))
        free_from = exit_idx + 1
    return out


def _smart_size_scale(
    closes: np.ndarray,
    volumes: np.ndarray,
    idx: int,
    is_long: bool,
) -> float:
    """#SMART-SIZE 품질 배수 — 라이브 `_set_smart_size` 와 같은 식.

    거래를 거르는 게 아니라(빈도 불변) 좋은 진입에 자금을 더 배분한다.
    진입 방향과 정합하는 신호 개수 q(0~3) 를 세어 `clip(0.7 + q×0.2, 0.4, 1.4)`.

    신호 셋 (출처: bot_ict_instance.py `_set_smart_size`):
        · 볼륨    — 진입봉 거래량 >= 최근 20봉 평균
        · NW 중심 — 가우시안 커널(bw=8, 최근 50봉) 중심선 대비 위치가 방향과 정합
        · RSI(14) — 롱이면 >50, 숏이면 <50

    Args:
        closes: 종가 배열.
        volumes: 거래량 배열.
        idx: 진입 봉 인덱스 (이 봉까지만 본다 — 미래 참조 금지).
        is_long: 롱 여부.

    Returns:
        0.4~1.4 배수. 데이터가 50봉 미만이면 1.0(중립).
    """
    if idx < 49:
        return 1.0
    c = closes[: idx + 1]
    v = volumes[: idx + 1]

    vol_ma = float(v[-20:].mean())
    vol_ok = vol_ma > 0 and float(v[-1]) >= vol_ma

    win = min(50, len(c))
    seg = c[-win:]
    ar = np.arange(win)
    w = np.exp(-((win - 1 - ar) ** 2) / (2 * 8.0 ** 2))
    nw_center = float(np.sum(seg * w) / np.sum(w))
    nw_ok = (float(c[-1]) > nw_center) == is_long

    d = np.diff(c[-15:])
    up = float(np.sum(np.where(d > 0, d, 0.0)))
    dn = float(np.sum(np.where(d < 0, -d, 0.0)))
    rsi = 100.0 - 100.0 / (1.0 + up / (dn + 1e-9))
    rsi_ok = (rsi > 50) == is_long

    q = int(vol_ok) + int(nw_ok) + int(rsi_ok)
    return float(np.clip(0.7 + q * 0.2, 0.4, 1.4))


def _risk_pct_for(score: int, cfg: BacktestConfig) -> float:
    """건당 리스크 % — 라이브 `_calc_qty_risk_based` 와 같은 식.

    `min(base + step × score, max)`. 진입 문턱이 5 라 score>=5 이므로
    3.0 + 1.5×5 = 10.5 → 상한 6.0 에 걸린다. **실질 상수**라는 뜻이고,
    htf_supporting boost 가 상수 +3 이던 것과 같은 패턴이다.
    """
    return min(
        cfg.risk_per_trade_base + cfg.risk_per_trade_step * max(0, score),
        cfg.risk_per_trade_max,
    )


def _funding_cost(
    funding: pd.DataFrame | None,
    index: pd.Index,
    fill_idx: int,
    exit_idx: int,
    is_long: bool,
) -> float:
    """보유 구간 펀딩 정산 합계 — 명목 대비 비율, **비용이 양수**.

    무기한 선물은 8시간마다 정산한다. 진입 **후**부터 청산 시각 **까지**의 정산만
    센다(진입 봉의 이미 지나간 정산은 부담하지 않는다).

    Args:
        funding: index=DatetimeIndex(UTC), column="rate" 인 펀딩 요율. None 이면 0.
        index: 가격 df 의 인덱스 (DatetimeIndex 여야 계산 가능).
        fill_idx: 체결 봉 인덱스.
        exit_idx: 청산 봉 인덱스.
        is_long: 롱이면 양(+) 요율을 **지불**, 숏이면 **수취**.

    Returns:
        비용 비율(양수=비용). 데이터가 없거나 구간에 정산이 없으면 0.0.
    """
    if funding is None or len(funding) == 0:
        return 0.0
    if not isinstance(index, pd.DatetimeIndex):
        return 0.0
    a, b = index[fill_idx], index[min(exit_idx, len(index) - 1)]
    seg = funding.loc[(funding.index > a) & (funding.index <= b), "rate"]
    if seg.empty:
        return 0.0
    return float(seg.sum()) * (1.0 if is_long else -1.0)


def run_backtest_from_timeline(
    df: pd.DataFrame,
    timeline: list,
    cfg: BacktestConfig,
    corr_df: pd.DataFrame | None = None,
    funding: pd.DataFrame | None = None,
) -> BacktestResult:
    """타임라인 캐시 재생 백테스트 — **Origo 라이브 정합 경로**.

    run_backtest 의 detect 호출을 timeline[i] 조회로 대체하고 나머지(stale 검사 →
    boost → _gate_pass → HTF EMA/align 게이트 → _simulate_fill → sl_dist_mult →
    _simulate_exit → 비용 → i=exit_idx+1 점프)는 한 줄 한 줄 동일 로직이다.

    2026-08-08 전수 감사(data/parity/audit.json) 이후, **라이브 봇 step() 의 진입
    파이프라인 순서 그대로** 정합 경로를 이식했다. 라이브 순서(bot_ict_instance.py
    step, 1264~1525)와 대응:

        1264  셋업 선택(prefer_direction → 방향 일치 최신 → stale)   [#1 #2]
        1320  #NYPM-GATE (NY local 13:30-16:00, DST)                  [mid]
        1337  #KZ-ENTRY  (in_trade_window_sub 재확인)                 [#9]
        1370  _passes_htf_ema_bias → align 게이트                     [#3]
        1381  _evaluate_htf_override → flip target 확정               [#7]
        1385  _apply_htf_supporting_boost  (+1~+3)                    [#5]
        1387  _apply_dol_bias              (-2)                       [#8]
        1390~ cisd / po3 / smt / ote boost (+1 각)                    [기존]
        1401  _set_entry_trend  (신호 봉 20봉 추세 → #CT-SL 배수 결정)
        1407  #REGIME  (롤링 q33 또는 하드코딩 floor)                 [mid]
        1427  #COND-ALIGN (롤링 q70)                                  [mid]
        1445  min_confluence + high_rr_bypass
        1483  #HTF-LTF-CONFLICT (bull/bear 가중치 비)                 [mid]
        1515  #SWEEP-GATE (일봉 스윕-반전 K일)                        [mid]
        2193  entry/SL/TP 평행이동 (max_entry_distance_pct)           [mid]

    _boost_score 는 window 슬라이스가 필요해서 켜진 경우에만 슬라이스해 호출한다
    (결과 동일 — 최적화일 뿐).

    Args:
        df: 타임라인 빌드에 쓴 것과 **같은** 매매 TF OHLCV (DatetimeIndex UTC).
        timeline: build_setup_timeline(df, build_cfg) 산출. build_cfg 의 detect
            필드(min_rr·phase_b_sources·ote_up_level 등)가 cfg 와 일치해야 함.
        cfg: 재생 파라미터. detect 무관 필드는 자유롭게 바꿔도 됨.
        corr_df: SMT boost 용 상관 심볼 OHLCV (apply_smt=True 시 필요).
        funding: 펀딩 요율 (index=DatetimeIndex, column="rate"). 주면 보유 구간
            정산을 net_pnl_pct 에서 차감하고 Trade.funding_pct 에 기록한다.
            None 이면 기존 동작(펀딩 미반영) 그대로.

    Returns:
        BacktestResult — run_backtest 와 동일 집계.

    Raises:
        ValueError: 타임라인 길이/detect_params 불일치, 또는
            prefer_direction_select 인데 타임라인에 dir_items 가 없을 때.
    """
    n = len(df)
    if len(timeline) != n:
        raise ValueError(
            f"timeline 길이({len(timeline)}) != df 길이({n}) — 같은 df 로 빌드한 "
            "타임라인을 쓸 것",
        )
    built = getattr(timeline, "detect_params", None)
    if built is not None:
        want = _detect_params(cfg)
        if built != want:
            raise ValueError(
                f"detect 파라미터 불일치 — 타임라인 빌드 {built} vs cfg {want}. "
                "min_rr 등 detect 인자가 다르면 타임라인을 따로 빌드해야 함",
            )
    dir_items = getattr(timeline, "dir_items", None)
    if cfg.prefer_direction_select and dir_items is None:
        raise ValueError(
            "prefer_direction_select=True 인데 타임라인에 dir_items 가 없다 — "
            "같은 cfg 로 build_setup_timeline 을 다시 돌릴 것(캐시 무효).",
        )
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    # #LIVE-SIZING: 품질 배수의 볼륨 신호용. 컬럼이 없으면 0 배열
    # (볼륨 신호가 항상 False → 배수가 최대 0.2 낮게 나온다).
    volumes = (
        df["volume"].to_numpy() if "volume" in df.columns
        else np.zeros(len(df), dtype=float)
    )
    # #SHORT-BIAS: HTF EMA bias 게이트용 1h EMA (off 면 계산 skip) — run_backtest 동일.
    htf_ema = (
        _precompute_htf_ema(df, cfg.htf_ema_period).to_numpy()
        if cfg.htf_ema_bias in ("strict", "band") else None
    )
    # align 점수 — #3 정합: align_live_formula 면 라이브 계산식(SMA시드·670봉·미완성봉),
    # 아니면 기존 백테식(전체 ewm + shift). 둘은 같은 시점에 다른 값을 낸다.
    htf_align = None
    if cfg.htf_ema_bias == "align" or cfg.htf_align_flip:
        htf_align = (
            _precompute_align_score_live(df, cfg.htf_align_periods)
            if cfg.align_live_formula
            else _precompute_align_score(df, cfg.htf_align_periods).to_numpy()
        )
    # #MSS-FLIP: 보유 중 반대 CHoCH 조기 청산용 봉별 신호 (mss_flip 일 때만).
    mss_sig = (
        _precompute_mss_signal(df, cfg.mss_swing_left, cfg.mss_swing_right)
        if cfg.mss_flip else None
    )
    # #MSS-BIAS-GATE: 진입 방향 제한용 구조 방향 (mss_bias_gate 일 때만).
    mss_bias = (
        _precompute_mss_bias(df, cfg.mss_swing_left, cfg.mss_swing_right)
        if (cfg.mss_bias_gate or cfg.align_mss_fill) else None
    )
    # #REGIME: 국면 적응 게이트용 EMA 스프레드 (regime_adaptive 일 때만).
    ema_spread_arr = (
        _precompute_ema_spread(df, cfg.htf_align_periods)
        if cfg.regime_adaptive else None
    )
    # ---- 2026-08-08 정합 이식 전처리 ----
    need_htf_map = (
        cfg.htf_fvg_support or cfg.htf_ltf_conflict_guard_ratio > 0 or cfg.htf_fvg_flip
    )
    htf_map = _precompute_htf_fvg(df, cfg) if need_htf_map else None
    kz_ok = is_nypm = None
    if cfg.entry_killzone_gate or cfg.exclude_nypm:
        kz_ok, is_nypm = _precompute_ny_gates(df)
    sweep_bs, sweep_bl = _precompute_sweep_gate(df, cfg.sweep_gate_days)
    # 라이브 _trend_history — 최근 150표본 롤링(REGIME_ROLLING_WINDOW), 최소 20.
    trend_hist: deque[float] = deque(maxlen=150)
    live_order = _parity_flags_on(cfg)

    need_boost = (  # 꺼져 있으면 _boost_score no-op → skip. apply_pd_filter 도 window 필요.
        cfg.apply_cisd or cfg.apply_smt
        or cfg.apply_cbdr or cfg.apply_dol or cfg.apply_po3 or cfg.apply_dailybias
        or cfg.apply_ote or cfg.apply_pd_filter
    )
    need_window = need_boost or cfg.apply_dol_counter
    trades: list[Trade] = []
    i = cfg.window
    while i < n - 1:
        item = timeline[i]  # run_backtest 의 detect 호출 대체 (캐시 조회)
        if item is None:
            i += 1
            continue
        # ---------------- [#2] 라이브 셋업 선택 (prefer_direction) ----------------
        # 출처: bot_ict_instance.py:1218 ema_dir=_compute_htf_ema_direction()
        #      → 1243 prefer_direction=ema_dir → signal/ict_signal.py:163-175.
        # |score|>=T 면 그 방향 셋업만 후보(그중 최신), 미만이면 방향 강제 없음
        # (대신 뒤 align 게이트가 전부 차단). score 없음(NaN)이면 게이트도 통과.
        pref: Direction | None = None
        if cfg.prefer_direction_select and htf_align is not None:
            sc0 = htf_align[i]
            if sc0 == sc0:
                t0 = max(1, int(cfg.htf_align_threshold))
                if sc0 >= t0:
                    pref = Direction.LONG
                elif sc0 <= -t0:
                    pref = Direction.SHORT
        if pref is not None:
            di = dir_items[i]
            if di is None:
                i += 1
                continue
            s_l, b_l, s_s, b_s = di
            setup, bars_since = (s_l, b_l) if pref is Direction.LONG else (s_s, b_s)
            if setup is None:
                i += 1
                continue   # 라이브: "no setup matching trend" → 그 봉 진입 없음
        else:
            setup, bars_since = item
        if bars_since > cfg.setup_stale_bars:
            i += 1
            continue
        is_long = setup.direction is Direction.LONG
        # ---------------- [mid] #NYPM-GATE — 진입 '시점' NY_PM 차단 ----------------
        if cfg.exclude_nypm and is_nypm is not None and is_nypm[i]:
            i += 1
            continue
        # ---------------- [#9] #KZ-ENTRY — 진입 '시점' 킬존 재확인 ----------------
        if (
            cfg.entry_killzone_gate and not cfg.disable_time_filter
            and kz_ok is not None and not kz_ok[i]
        ):
            i += 1
            continue
        # bot 레벨 boost — 켜진 경우만 window 슬라이스 (run_backtest 동일 의미).
        boost_names: list[str] = []   # 2026-08-07 조합 실험용 — 발동 boost 기록
        window = None
        corr_window = None
        if need_window:
            window = df.iloc[i - cfg.window : i + 1]
            corr_window = (
                corr_df.iloc[i - cfg.window : i + 1] if corr_df is not None else None
            )
        # #PD-FILTER: ICT 프리미엄/디스카운트 — LONG=디스카운트/SHORT=프리미엄일 때만.
        if cfg.apply_pd_filter and not _pd_pass(window, setup.direction, setup.entry):
            i += 1
            continue
        # ------- [#3] "align" 다중 EMA 정렬 게이트 (라이브 _passes_ema_align_gate) -------
        # #REGIME: 반등 의심(EMA 스프레드 < 임계)이면 align 게이트 풀기(양방향 허용).
        regime_off = (
            cfg.regime_adaptive and ema_spread_arr is not None
            and ema_spread_arr[i] < cfg.regime_spread_thr
        )
        if htf_align is not None and cfg.htf_ema_bias == "align" and not regime_off:
            sc = htf_align[i]
            if sc == sc:  # NaN 아니면 (라이브도 score None 이면 게이트 통과)
                t = cfg.htf_align_threshold
                if sc >= t:
                    blocked = not is_long       # 상승추세 → 롱만
                elif sc <= -t:
                    blocked = is_long           # 하락추세 → 숏만
                else:
                    # 애매구간(되돌림/횡보): 기본 진입자제. align_mss_fill 이면
                    # 검증된 EMA 와 MSS 구조 결합 — MSS 방향으로만 진입 허용.
                    if cfg.align_mss_fill and mss_bias is not None and mss_bias[i] != 0:
                        mb = mss_bias[i]
                        blocked = (mb == 1 and not is_long) or (mb == -1 and is_long)
                    else:
                        blocked = True              # 추세 불명확 → 진입 자제
                if blocked:
                    i += 1
                    continue
        # #SHORT-BIAS: 단일 EMA bias 게이트 (strict/band 모드 — align 과 배타).
        if htf_ema is not None:
            ema_v = htf_ema[i]
            if ema_v == ema_v:  # NaN 아니면 (초기 구간 NaN 은 게이트 skip = 허용)
                cl = float(closes[i])
                band = ema_v * cfg.htf_ema_band_pct
                if cfg.htf_ema_bias == "strict":
                    blocked = (cl > ema_v and not is_long) or (cl < ema_v and is_long)
                else:  # "band" — 완충대 밖에서만 방향 강제, 안이면 양방향 허용
                    blocked = (
                        (cl > ema_v + band and not is_long)
                        or (cl < ema_v - band and is_long)
                    )
                if blocked:
                    i += 1
                    continue
        # #MSS-BIAS-GATE: 마지막 CHoCH 방향과 반대 진입 차단 (EMA 대체/병행).
        if mss_bias is not None:
            mb = mss_bias[i]
            if mb != 0:
                if (mb == 1 and not is_long) or (mb == -1 and is_long):
                    i += 1
                    continue
        # ---------------- 점수 보정 (라이브 1385~1398 순서 그대로) ----------------
        score = int(setup.confluence_score)
        # [#5] HTF FVG supporting boost (+1~+3) — 라이브 진입의 93% 가 받던 가점.
        if cfg.htf_fvg_support and htf_map is not None:
            tw = float(
                htf_map["sup_long"][i] if is_long else htf_map["sup_short"][i],
            )
            hb = _htf_support_boost(tw)
            if hb > 0:
                score += hb
                boost_names.append(f"htf_support_weight={tw:.0f}_boost+{hb}")
        # [#8] DOL 역방향 감점 (-2) — 부호·크기 모두 기존 apply_dol(+1) 과 반대.
        if cfg.apply_dol_counter and window is not None:
            d_delta, d_tag = _dol_counter_delta(
                window, setup.direction, cfg.dol_counter_penalty,
            )
            if d_delta != 0:
                score += d_delta
                boost_names.append(d_tag)
        # cisd / smt / po3 / ote / cbdr / dailybias (+1 각) — 기존 경로.
        if need_boost:
            score = _boost_score(
                score, setup.direction, window, corr_window, cfg,
                detail=boost_names,
            )
        # ------------- 진입 직전 추세 (#CT-SL / #REGIME / #COND-ALIGN) -------------
        # 라이브 _set_entry_trend(3415-3426)는 **신호 봉** 기준 20봉 변화율을 쓴다.
        # 기존 백테는 체결 봉(fill_idx) 기준이라 값이 달랐다 — 정합 모드에서만 교체
        # (레거시 연구 결과 재현성 보존).
        trend_sig = _entry_trend_pct(closes, i)
        if cfg.regime_filter:
            floor = cfg.regime_floor
            if cfg.regime_rolling and len(trend_hist) >= 20:
                vals = sorted(trend_hist)
                floor = vals[len(vals) // 3]          # 하위 33분위 (라이브 3162)
            cur_t = abs(trend_sig)
            trend_hist.append(cur_t)   # 라이브 1410 — floor 계산 **후** 누적
            if floor > 0 and cur_t < floor:
                i += 1
                continue
        if cfg.cond_align:
            strong = cfg.strong_trend_floor
            if cfg.regime_rolling and len(trend_hist) >= 20:
                vals = sorted(trend_hist)
                strong = vals[int(len(vals) * 0.7)]   # 상위 30% 경계 (라이브 3174)
            signed_t = trend_sig * (1.0 if is_long else -1.0)
            if strong > 0 and abs(trend_sig) < strong and signed_t < 0:
                i += 1
                continue
        # ---------------- min_confluence + 고RR 예외 ----------------
        if not _gate_pass(score, setup.risk_reward, cfg):
            i += 1
            continue
        # ---------------- [연구] 항목 기반 조건 (require_items) ----------------
        if cfg.require_items and not _require_items_pass(
            setup.confluences, boost_names, cfg.require_items,
        ):
            i += 1
            continue
        # ---------------- [mid] #HTF-LTF-CONFLICT ----------------
        if cfg.htf_ltf_conflict_guard_ratio > 0 and htf_map is not None:
            bull_w = float(htf_map["tot_bull"][i])
            bear_w = float(htf_map["tot_bear"][i])
            if bull_w > 0 and bear_w > 0:
                r_thr = cfg.htf_ltf_conflict_guard_ratio
                if is_long and (bear_w / bull_w) >= r_thr:
                    i += 1
                    continue
                if (not is_long) and (bull_w / bear_w) >= r_thr:
                    i += 1
                    continue
        # ---------------- [mid] #SWEEP-GATE ----------------
        if cfg.sweep_gate_days > 0 and (
            (is_long and sweep_bl[i]) or ((not is_long) and sweep_bs[i])
        ):
            i += 1
            continue
        # ---------------- [#7] flip target 확정 (진입 시점 고정) ----------------
        flip_zone: tuple[float, float] | None = None
        if cfg.htf_fvg_flip and htf_map is not None:
            f_lo = htf_map["flip_lo_long"][i] if is_long else htf_map["flip_lo_short"][i]
            f_hi = htf_map["flip_hi_long"][i] if is_long else htf_map["flip_hi_short"][i]
            if f_lo == f_lo and f_hi == f_hi:   # NaN 아니면
                flip_zone = (float(f_lo), float(f_hi))
        # ------------ [mid] entry/SL/TP 평행이동 (max_entry_distance_pct) ------------
        entry = setup.entry
        sl, tp = setup.stop_loss, setup.take_profit
        if cfg.max_entry_distance_pct > 0 and entry > 0:
            cur_px = float(closes[i])
            max_dist = cur_px * cfg.max_entry_distance_pct
            if abs(entry - cur_px) > max_dist:
                adj = (cur_px - max_dist) if entry < cur_px else (cur_px + max_dist)
                delta = adj - entry
                entry = adj
                sl += delta      # RR 보존 평행이동 (#ENTRY-ADJ-RR)
                tp += delta
        # 체결 시뮬 — limit(entry)에 ttl 봉 내 가격이 닿아야 체결. 미체결이면 skip.
        fill_idx = _simulate_fill(
            highs, lows, i, setup.direction, entry, cfg.entry_ttl_bars,
        )
        if fill_idx is None:
            i += 1
            continue  # 타점 미도달 — 미체결(타점 포기)
        d_val = setup.direction.value
        # 흑자 탐색: SL 거리 변형 (stop-hunt 회피 vs 손실크기 트레이드오프).
        # #CT-SL 판정 기준 봉: 정합 모드면 신호 봉(라이브 동일), 아니면 기존 체결 봉.
        if cfg.sl_dist_mult_ct > 0.0:
            _sg = trend_sig if live_order else _entry_trend_pct(closes, fill_idx)
            _sg *= (1.0 if is_long else -1.0)
            eff_mult = (
                cfg.sl_dist_mult_ct if _sg < cfg.ct_trend_threshold else cfg.sl_dist_mult
            )
        else:
            eff_mult = cfg.sl_dist_mult
        if eff_mult != 1.0:
            risk = abs(entry - sl)
            if risk > 0:
                rr0 = abs(tp - entry) / risk
                new_risk = risk * eff_mult
                if cfg.sl_liq_cap and cfg.leverage > 0:
                    cap = entry * 0.8 / cfg.leverage
                    if new_risk > cap:
                        new_risk = max(risk, cap)  # 라이브 #LIQ-CAP 동일
                if is_long:
                    sl = entry - new_risk
                    tp = entry + new_risk * rr0 if cfg.tp_keeps_rr else tp
                else:
                    sl = entry + new_risk
                    tp = entry - new_risk * rr0 if cfg.tp_keeps_rr else tp
        # #TP-RR: TP 를 risk 의 고정 배수로 강제 (승률↑·RR↓ 트레이드오프 연구).
        if cfg.tp_rr_override > 0:
            risk2 = abs(entry - sl)
            tp = (
                entry + risk2 * cfg.tp_rr_override if is_long
                else entry - risk2 * cfg.tp_rr_override
            )
        exit_idx, exit_raw, outcome = _simulate_exit(
            opens, highs, lows, closes, fill_idx, setup.direction,
            sl, tp, cfg,
            align_score=htf_align if cfg.htf_align_flip else None,
            mss_signal=mss_sig,
            entry=entry,
            htf_flip_zone=flip_zone,
        )
        exit_slip = slip_pct(
            float(highs[exit_idx]), float(lows[exit_idx]), float(closes[exit_idx]),
        )
        exit_price = apply_slippage(exit_raw, d_val, "exit", exit_slip)
        sign = 1.0 if is_long else -1.0
        raw_pnl_pct = (exit_price - entry) / entry * sign
        net_pnl_pct, _ = apply_costs(raw_pnl_pct, cfg.size_pct, cfg.leverage)
        # #FUNDING 2026-08-09: 보유 구간 정산분. 롱은 지불(+비용) · 숏은 수취(−).
        # raw 는 손대지 않고 net 에서만 뺀다(기존 연구와의 비교 보존).
        fund_pct = _funding_cost(funding, df.index, fill_idx, exit_idx, is_long)
        if fund_pct:
            net_pnl_pct -= fund_pct * cfg.size_pct * cfg.leverage
        # #LIVE-SIZING: 크기 계산 입력값 기록. 손익은 바꾸지 않는다 —
        # 실제 복리는 이 값들을 받아 포트폴리오 시뮬이 계산한다.
        ss = (
            _smart_size_scale(closes, volumes, fill_idx, is_long)
            if cfg.smart_size_enabled else 1.0
        )
        trades.append(Trade(
            entry_idx=fill_idx, exit_idx=exit_idx, direction=d_val,
            entry=entry, exit_price=exit_price, outcome=outcome,
            raw_pnl_pct=raw_pnl_pct, net_pnl_pct=net_pnl_pct,
            funding_pct=fund_pct,
            smart_size_scale=ss,
            risk_pct_used=_risk_pct_for(int(score), cfg),
            confluence_score=score,
            entry_atr_pct=_entry_atr_pct(highs, lows, closes, fill_idx, entry),
            entry_trend_pct=(
                trend_sig if live_order else _entry_trend_pct(closes, fill_idx)
            ),
            entry_sl=float(sl), entry_tp=float(tp),
            confluences=tuple(setup.confluences),
            base_score=int(setup.confluence_score),
            boosts=tuple(boost_names),
        ))
        i = exit_idx + 1  # 청산 후 다음 봉부터 재탐색 (동시 포지션 1개)

    # ---------------- [#MMBM-BT] 2번째 진입모델 ----------------
    # 라이브는 SB 셋업이 없거나 게이트를 못 넘은 봉에서 MMBM 을 시도한다
    # (bot_ict_instance.py step() 1264-1295). SB 가 포지션을 들고 있는 동안에는
    # 어느 모델도 신규 진입을 하지 않으므로, SB 결과를 그대로 두고 그 **보유 구간
    # 밖만** 채우면 같은 결과가 된다.
    if cfg.mmbm_enabled:
        mm = _mmbm_fill_gaps(df, trades, _precompute_mmbm(df, cfg), cfg, funding)
        if mm:
            trades = sorted(trades + mm, key=lambda t: t.entry_idx)
    return _aggregate(cfg, trades)


# 멀티 TF 기본 스펙: (TF 라벨, resample rule, ttl_bars=체결 대기 1m 봉 수).
# ttl_bars 를 TF 봉 길이(분)와 같게 둬서 "그 TF 한 봉 동안 retrace 체결을 기다린다".
# (파트너 핵심 아이디어 — 높은 TF setup 은 더 오래 기다려 준다.)
_DEFAULT_TF_SPECS: tuple[tuple[str, str, int], ...] = (
    ("5m", "5min", 5),
    ("15m", "15min", 15),
    ("1h", "1h", 60),
)


def _latest_closed_tf_idx(
    tf_index: pd.DatetimeIndex, tf_delta: pd.Timedelta, now_ts: pd.Timestamp,
) -> int:
    """now_ts(현재 1m 봉 시각) 시점에 '이미 닫힌' 마지막 TF 봉의 위치(없으면 -1).

    look-ahead 방지의 핵심: TF 봉 라벨(open time) L 은 [L, L+tf_delta) 구간을 덮고
    L+tf_delta 에 비로소 '확정(닫힘)'된다. 따라서 현재 1m 시각 now_ts 에서 사용
    가능한 마지막 TF 봉은 ``L + tf_delta <= now_ts`` 를 만족하는 마지막 봉이다.
    진행 중(아직 안 닫힌) TF 봉은 절대 보지 않는다 → 미래 봉 참조 없음.

    Args:
        tf_index: 해당 TF DataFrame 의 DatetimeIndex (봉 open time, 오름차순).
        tf_delta: TF 한 봉 길이 (예 5min → Timedelta("5min")).
        now_ts: 현재 1m 봉의 timestamp.

    Returns:
        닫힌 마지막 TF 봉의 정수 위치. 아직 닫힌 봉이 없으면 -1.
    """
    # 닫힘 시각 = open_time + tf_delta. 그 값이 now_ts 이하인 마지막 봉을 이진탐색.
    # searchsorted(now_ts - tf_delta, "right") - 1 == (open_time <= now_ts - tf_delta)
    # 인 마지막 봉 == (open_time + tf_delta <= now_ts) 인 마지막 봉.
    cutoff = now_ts - tf_delta
    return int(tf_index.searchsorted(cutoff, side="right")) - 1


def run_backtest_multitf(
    df_1m: pd.DataFrame,
    cfg: BacktestConfig,
    tf_specs: list[tuple[str, str, int]] | None = None,
    corr_df: pd.DataFrame | None = None,
) -> BacktestResult:
    """멀티 TF setup 백테스트 — 높은 TF 우선 진입 + TF별 체결 대기.

    단일 TF(5m) setup 만 잡는 ``run_backtest`` 와 비교용. 여러 TF(기본 5m·15m·1h)
    에서 각각 ``detect_silver_bullet_setups`` 를 돌리고, 매 1m 시점마다 **높은
    TF(1h>15m>5m) 우선**으로 유효(stale 아님 + 게이트 통과) setup 을 찾아 진입한다.
    1h setup 이 있으면 그걸, 없으면 15m, 그것도 없으면 5m 을 쓴다.

    게이트/비용/시뮬은 ``run_backtest`` 와 **완전히 동일**:
        - confluence 게이트 ``_gate_pass`` (boost 는 1m 윈도우 기준 _boost_score),
        - HTF EMA bias 게이트 (``htf_ema_bias`` strict/band/align, 1m 기준 precompute),
        - SL 거리 변형 ``sl_dist_mult`` / ``tp_keeps_rr``,
        - 체결 ``_simulate_fill`` / 청산 ``_simulate_exit`` (둘 다 1m 봉 경로),
        - 슬리피지/수수료 (apply_slippage / apply_costs).
    차이는 (1) setup 을 어느 TF 에서 잡는가, (2) **진입 시 그 setup 이 나온 TF 의
    ttl_bars 로 체결 대기**한다는 점뿐.

    TF 플립 (``cfg.tf_flip=True``):
        위 정적 우선 진입은 그대로 두고, **보유 중**(진입~청산 사이 매 1m 봉) 추가로
        '현재 보유 setup 의 TF 보다 더 높은 TF' 에서 새 유효 setup(stale 아님 + conf
        게이트 + EMA/align 방향 게이트 통과)이 뜨는지 본다. 뜨면 그 봉 close 에서 기존
        포지션을 즉시 청산(``outcome="tf_flip"``)하고 그 HTF setup 으로 **전환 진입**한다
        (그 HTF ttl_bars 안에 ``_simulate_fill`` 로 체결되면 그 SL/TP 로 재보유 → 또 더
        높은 TF 뜨면 또 플립, 반복; 타점 미도달이면 청산만 하고 다음 1m 부터 재탐색).
        방향 무관(HTF 가 더 중요 — 같든 다르든 전환). 같은 HTF setup 으로의 중복 플립은
        ``entered_ts``(ts_ms) 로 막는다. 5m 보유→15m·1h 신호면 플립, 15m 보유→1h 신호면
        플립, 1h 보유 중엔 더 높은 TF 가 없어 플립 없음. ``tf_flip=False`` 면 기존 정적
        진입과 **100% 동일**(회귀 없음). ``run_backtest`` 는 영향받지 않는다.

    look-ahead 방지:
        - 각 TF setup 은 그 시점까지 **닫힌** TF 봉들로만 검출한다
          (``_latest_closed_tf_idx``: open_time+tf_delta <= 현재 1m 시각). 진행 중
          TF 봉은 절대 포함하지 않는다.
        - 체결/청산은 1m 배열에서 진입 시점 **이후** 봉만 전방 탐색.
        - HTF EMA / align 점수도 기존 함수가 직전 확정 1h 봉(shift)만 쓴다.
        - TF 플립도 동일: 보유 중 j 봉의 HTF 판정은 ``_tf_setup_at``(닫힌 HTF 봉만)을
          j 시점으로 재사용하고, 전환 체결은 j(=청산 봉) **다음** 봉부터 전방 탐색한다.

    성능:
        매 1m 마다 3개 TF 를 전부 re-detect 하면 5년치를 못 돌린다. TF별로 **그 TF
        봉이 새로 닫혔을 때만** detect 하고 그 결과를 캐시한다(``_tf_cache``). 1m 시각이
        같은 TF 봉 구간 안이면 캐시 재사용 → detect 호출 수가 1m 수가 아니라 TF 봉
        수로 줄어든다.

    Args:
        df_1m: 1m OHLCV (DatetimeIndex UTC). ``load_ohlcv_parquet`` 산출.
        cfg: 파라미터 (``run_backtest`` 와 공유). ``cfg.entry_ttl_bars`` 는 무시되고
            TF별 ``tf_specs`` 의 ttl_bars 가 쓰인다. ``cfg.window`` 는 각 TF detect 의
            lookback 봉 수로 그대로 쓴다.
        tf_specs: ``[(라벨, resample_rule, ttl_bars=분), ...]``. 기본 5m/15m/1h.
            **낮은→높은 TF 순으로 줄 것** (내부에서 진입은 높은 TF 부터 시도).
        corr_df: SMT boost 용 상관 심볼 1m OHLCV (apply_smt=True 시).

    Returns:
        BacktestResult — ``run_backtest`` 와 동일 집계. 각 Trade.source_tf 에 진입
        TF 라벨이 기록된다.
    """
    if tf_specs is None:
        tf_specs = list(_DEFAULT_TF_SPECS)
    # 진입은 높은 TF(긴 봉) 우선 — ttl_bars(분) 큰 순. 입력 순서 무관하게 정렬.
    specs_hi_to_lo = sorted(tf_specs, key=lambda s: s[2], reverse=True)

    n = len(df_1m)
    opens = df_1m["open"].to_numpy()
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    closes = df_1m["close"].to_numpy()
    index_1m = df_1m.index

    # #SHORT-BIAS 게이트용 1m 기준 precompute (run_backtest 와 동일 — look-ahead 안전).
    htf_ema = (
        _precompute_htf_ema(df_1m, cfg.htf_ema_period).to_numpy()
        if cfg.htf_ema_bias in ("strict", "band") else None
    )
    htf_align = (
        _precompute_align_score(df_1m, cfg.htf_align_periods).to_numpy()
        if (cfg.htf_ema_bias == "align" or cfg.htf_align_flip) else None
    )

    # TF별 리샘플 + 봉 길이 Timedelta + detect 결과 캐시 준비.
    tf_frames: dict[str, dict[str, Any]] = {}
    for label, rule, ttl in specs_hi_to_lo:
        tf_df = resample_ohlcv(df_1m, rule)
        tf_frames[label] = {
            "df": tf_df,
            "delta": pd.Timedelta(rule),
            "ttl": ttl,
            "last_detect_pos": -2,        # 마지막으로 detect 한 닫힌 TF 봉 위치
            "cached_setup": None,         # 캐시된 (setup, score) — 게이트 전 raw
        }

    def info_rule(label: str) -> str:
        """label → resample rule (corr_df TF 정렬용)."""
        for lb, rule, _ in specs_hi_to_lo:
            if lb == label:
                return rule
        return "5min"

    def _tf_setup_at(label: str, now_ts: pd.Timestamp, i_1m: int):
        """해당 TF 에서 now_ts 시점에 유효한 setup (stale·게이트 통과) 또는 None.

        그 TF 봉이 새로 닫혔을 때만 detect 하고 캐시. 캐시된 setup 에 stale 검사 +
        boost + _gate_pass + EMA/align 방향 게이트를 (현재 1m 시점 기준) 적용한다.
        """
        info = tf_frames[label]
        tf_df: pd.DataFrame = info["df"]
        tf_index: pd.DatetimeIndex = tf_df.index
        closed_pos = _latest_closed_tf_idx(tf_index, info["delta"], now_ts)
        if closed_pos < 0:
            return None
        # 새 TF 봉이 닫혔을 때만 detect — 같은 봉 구간이면 캐시 재사용 (성능).
        if closed_pos != info["last_detect_pos"]:
            info["last_detect_pos"] = closed_pos
            lo = max(0, closed_pos + 1 - cfg.window)
            window = tf_df.iloc[lo : closed_pos + 1]  # 닫힌 봉까지만 (look-ahead 차단)
            setups = detect_silver_bullet_setups(
                window,
                min_rr=cfg.min_rr,
                fvg_min_size_pct=cfg.fvg_min_size_pct,
                min_confluence=0,
                expand_to_killzone=cfg.expand_to_killzone,
                disable_time_filter=cfg.disable_time_filter,
                min_sl_distance_pct=cfg.min_sl_distance_pct,
                nyse_gate=cfg.nyse_gate,
            window_once=cfg.window_once,
            max_per_fvg=cfg.max_per_fvg,
            )
            if not setups:
                info["cached_setup"] = None
            else:
                setup = setups[-1]
                bars_since = len(window) - 1 - setup.anchor_idx
                # 2026-06-11 리뷰 수정: stale 단위를 1m 환산 — bars_since 는 TF 봉
                # 수라 1h TF 면 120봉=120시간(5일)이나 유효해 HTF 쏠림 왜곡.
                # run_backtest(1m 기준 120봉=2시간)와 같은 시간 의미로 비교.
                tf_min = max(1, int(info["delta"].total_seconds() // 60))
                if bars_since * tf_min > cfg.setup_stale_bars:
                    info["cached_setup"] = None
                else:
                    # boost 는 그 TF 윈도우 기준 (run_backtest 와 동일 의미).
                    corr_win = None
                    if corr_df is not None:
                        corr_tf = resample_ohlcv(corr_df, info_rule(label))
                        corr_win = corr_tf.reindex(window.index)
                    score = _boost_score(
                        setup.confluence_score, setup.direction, window, corr_win, cfg,
                    )
                    info["cached_setup"] = (setup, score)
        cached = info["cached_setup"]
        if cached is None:
            return None
        setup, score = cached
        if not _gate_pass(score, setup.risk_reward, cfg):
            return None
        # HTF EMA bias 방향 게이트 (현재 1m 시점 — run_backtest 와 동일).
        is_long = setup.direction is Direction.LONG
        if htf_ema is not None:
            ema_v = htf_ema[i_1m]
            if ema_v == ema_v:
                cl = float(closes[i_1m])
                band = ema_v * cfg.htf_ema_band_pct
                if cfg.htf_ema_bias == "strict":
                    blocked = (cl > ema_v and not is_long) or (cl < ema_v and is_long)
                else:  # band
                    blocked = (
                        (cl > ema_v + band and not is_long)
                        or (cl < ema_v - band and is_long)
                    )
                if blocked:
                    return None
        if htf_align is not None and cfg.htf_ema_bias == "align":
            sc = htf_align[i_1m]
            if sc == sc:
                t = cfg.htf_align_threshold
                if sc >= t:
                    blocked = not is_long
                elif sc <= -t:
                    blocked = is_long
                else:
                    blocked = True
                if blocked:
                    return None
        return setup, score

    # 진입 확정 setup 의 재진입 방지(ts_ms) — 정적 진입·플립 전환 공통.
    # (TF 우선순위는 ttl_bars 크기로 직접 비교 — 큰 ttl = 더 높은 TF.)
    entered_ts: dict[str, int] = {}

    def _flip_target_at(holding_ttl: int, j: int):
        """j 봉 시점에 '현재 보유 TF 보다 더 높은 TF' 의 유효 setup → (label, ttl, res).

        TF 플립용. holding_ttl(보유 TF 의 ttl_bars)보다 큰 TF 만, 높은 TF 우선으로
        스캔해 첫 유효 setup 을 반환(없으면 None). look-ahead 방지: _tf_setup_at 이
        j 시점까지 '닫힌' HTF 봉만 보고(_latest_closed_tf_idx), EMA/align 게이트도
        j 시점 1m 값으로 평가한다. 방향 무관 — HTF 면 같은 방향이든 반대든 전환.
        같은 HTF setup(ts_ms)으로의 중복 플립은 entered_ts 로 막는다.
        """
        now_j = index_1m[j]
        for label, _rule, ttl in specs_hi_to_lo:
            if ttl <= holding_ttl:  # 더 높은 TF 만 (같거나 낮은 TF 는 플립 대상 아님)
                continue
            res = _tf_setup_at(label, now_j, j)
            if res is None:
                continue
            setup, _ = res
            sid = getattr(setup, "ts_ms", None)
            if sid is not None and entered_ts.get(label) == sid:
                continue  # 이미 그 setup 으로 진입/플립함 — 중복 방지
            return label, ttl, res
        return None

    def _exec_entry(setup, score: int, ttl: int, label: str, i_1m: int):
        """i_1m 시점 setup 을 체결·청산까지 시뮬해 Trade 1건 + 다음 탐색 위치 반환.

        반환: (trade 또는 None, next_i, flip_to). 체결 실패 시 (None, i_1m+1, None).
        cfg.tf_flip=True 면 보유 중 더 높은 TF setup 발생 시 outcome="tf_flip" 으로
        조기 청산하고, 그 HTF 전환 정보를 flip_to=(label,ttl,res) 로 돌려준다(호출부가
        전환 진입을 이어서 시도). flip_to 가 있으면 next_i 는 의미 없음(전환 우선).
        """
        sid = getattr(setup, "ts_ms", None)
        fill_idx = _simulate_fill(
            highs, lows, i_1m, setup.direction, setup.entry, ttl,
        )
        if fill_idx is None:
            return None, i_1m + 1, None
        if sid is not None:
            entered_ts[label] = sid  # 진입 확정 — 이 setup 재진입/중복플립 차단
        d_val = setup.direction.value
        entry = setup.entry
        sl, tp = setup.stop_loss, setup.take_profit
        eff_mult = _effective_sl_mult(cfg, closes, fill_idx, setup.direction is Direction.LONG)
        if eff_mult != 1.0:
            risk = abs(entry - sl)
            if risk > 0:
                rr0 = abs(tp - entry) / risk
                new_risk = risk * eff_mult
                if cfg.sl_liq_cap and cfg.leverage > 0:
                    cap = entry * 0.8 / cfg.leverage
                    if new_risk > cap:
                        new_risk = max(risk, cap)  # 라이브 #LIQ-CAP 동일
                if setup.direction is Direction.LONG:
                    sl = entry - new_risk
                    tp = entry + new_risk * rr0 if cfg.tp_keeps_rr else tp
                else:
                    sl = entry + new_risk
                    tp = entry - new_risk * rr0 if cfg.tp_keeps_rr else tp
        # TF 플립 활성 시, 보유 중 매 봉 더 높은 TF setup 을 탐지하는 flip_check 구성.
        flip_box: dict[str, Any] = {}

        def _flip_check(jbar: int) -> bool:
            ft = _flip_target_at(ttl, jbar)
            if ft is None:
                return False
            flip_box["target"] = ft  # 전환 정보 저장 — _simulate_exit 가 이 봉서 청산
            return True

        exit_idx, exit_raw, outcome = _simulate_exit(
            opens, highs, lows, closes, fill_idx, setup.direction,
            sl, tp, cfg,
            align_score=htf_align if cfg.htf_align_flip else None,
            flip_check=_flip_check if cfg.tf_flip else None,
        )
        exit_slip = slip_pct(
            float(highs[exit_idx]), float(lows[exit_idx]), float(closes[exit_idx]),
        )
        exit_price = apply_slippage(exit_raw, d_val, "exit", exit_slip)
        sign = 1.0 if setup.direction is Direction.LONG else -1.0
        raw_pnl_pct = (exit_price - entry) / entry * sign
        net_pnl_pct, _ = apply_costs(raw_pnl_pct, cfg.size_pct, cfg.leverage)
        trade = Trade(
            entry_idx=fill_idx, exit_idx=exit_idx, direction=d_val,
            entry=entry, exit_price=exit_price, outcome=outcome,
            raw_pnl_pct=raw_pnl_pct, net_pnl_pct=net_pnl_pct,
            confluence_score=score, source_tf=label,
            entry_sl=float(sl), entry_tp=float(tp),
        )
        flip_to = flip_box.get("target") if outcome == "tf_flip" else None
        return trade, exit_idx + 1, flip_to

    trades: list[Trade] = []
    i = cfg.window
    while i < n - 1:
        now_ts = index_1m[i]
        chosen = None
        chosen_label = None
        chosen_ttl = 0
        # 높은 TF 우선 — 첫 유효 setup 에서 멈춤 (정적 우선 진입, flip on/off 공통).
        # 2026-06-11 리뷰 수정: 이미 진입한 setup(dedup)은 *그 TF 만* 건너뛰고
        # 하위 TF 로 폴백 — 기존엔 dedup 이 루프 밖이라 1h setup 하나가 캐시에
        # 살아있는 동안 하위 TF 신규 setup 까지 전부 차단됐다(거래 가뭄 왜곡).
        for label, _rule, ttl in specs_hi_to_lo:
            res = _tf_setup_at(label, now_ts, i)
            if res is None:
                continue
            sid_cand = getattr(res[0], "ts_ms", None)
            if sid_cand is not None and entered_ts.get(label) == sid_cand:
                continue  # 이 TF 의 같은 setup 재진입 방지 — 다음(하위) TF 폴백
            chosen, _score = res
            chosen_label = label
            chosen_ttl = ttl
            break
        if chosen is None:
            i += 1
            continue
        # 진입 → 보유 → 청산. cfg.tf_flip=True 면 보유 중 HTF setup 발생 시
        # outcome="tf_flip" 으로 조기청산하고 그 HTF setup 으로 전환 진입을 이어간다
        # (체결되면 또 보유 — 더 높은 TF 뜨면 또 플립, 반복 가능). 미체결이면 그냥
        # 정적 청산만 하고 다음 1m 부터 재탐색.
        cur_setup, cur_score, cur_ttl, cur_label = chosen, _score, chosen_ttl, chosen_label
        cur_i = i
        while True:
            trade, next_i, flip_to = _exec_entry(
                cur_setup, cur_score, cur_ttl, cur_label, cur_i,
            )
            if trade is not None:
                trades.append(trade)
            if flip_to is None:
                i = next_i  # 정적 청산(tp/sl/eod) 또는 미체결 — 다음 봉부터 재탐색
                break
            # tf_flip — 전환 진입을 그 청산 봉(exit_idx)에서 시도. 체결 안 되면
            # _exec_entry 가 미체결 처리 → flip_to=None 으로 루프 종료.
            flip_label, flip_ttl, (flip_setup, flip_score) = flip_to
            cur_setup, cur_score, cur_ttl, cur_label = (
                flip_setup, flip_score, flip_ttl, flip_label,
            )
            cur_i = trade.exit_idx  # 전환 setup 의 체결 대기는 청산 봉 다음부터
    return _aggregate(cfg, trades)
