"""Aurora-ICT bot instance — Bybit 대상 실매매 실행기.

처리 흐름:
1. 매 분 OHLCV (BTCUSDT 1m) fetch — Bybit
2. ``generate_ict_signal()`` 호출
3. ENTER_LONG / ENTER_SHORT 신호 시 → limit order (entry) + SL + TP placement
4. 진입 후 position 추적, SL/TP 트리거는 거래소가 처리 (Bybit conditional)
5. 청산이 감지되면 내부 position state 리셋

Exchange client는 외부 주입 — ``ExchangeClientProtocol``을 만족하면 됨.
Aurora 측 ``CCXTClient`` (Bybit)를 어댑터로 감싸 사용 (duck typing).
테스트에서는 mock으로 교체.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Protocol
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from ccxt.base.errors import AuthenticationError

from aurora_ict.bot.shared_ohlcv_cache import GLOBAL_OHLCV_CACHE, SharedOhlcvCache
from aurora_ict.bot.structure_trail import compute_structure_trail
from aurora_ict.config.settings import ORIGO_MODEL_NAME
from aurora_ict.indicators.cisd import CisdType, detect_cisd
from aurora_ict.indicators.daily_bias import compute_daily_bias
from aurora_ict.indicators.dol import compute_dol
from aurora_ict.indicators.liquidity import detect_liquidity_sweeps
from aurora_ict.indicators.smt import SmtType, detect_smt_divergence
from aurora_ict.indicators.structure import TrendDirection
from aurora_ict.indicators.swing_points import detect_swing_points
from aurora_ict.interfaces.trades_store import (
    TradeEvent,
    TradeEventType,
    TradesStore,
)
from aurora_ict.paths import data_dir as _ict_data_dir
from aurora_ict.signal.ict_signal import (
    ICTSignal,
    SignalAction,
    generate_ict_signal,
)
from aurora_ict.strategy.htf_fvg_map import (
    TF_WEIGHT,
    HtfFvgEntry,
    build_htf_fvg_map,
    find_opposite_htf_fvg,
    find_supporting_htf_fvg,
)
from aurora_ict.strategy.htf_setup_tracker import HtfSetupTracker
from aurora_ict.strategy.ltf_entry_confirmer import (
    ConfirmedEntry,
    confirm_ltf_entry,
)
from aurora_ict.strategy.mmbm import detect_mmbm_setup
from aurora_ict.strategy.multi_tf_bias import (
    combine_bias,
    compute_bias_from_df,
    htf_pair,
)
from aurora_ict.strategy.silver_bullet import (
    Direction,
    SetupSource,
    SilverBulletSetup,
)
from aurora_ict.strategy.trend_state import TrendState, evaluate_trend
from aurora_ict.timing.killzone import (
    KillzoneName,
    classify_killzone,
    in_trade_window_sub,
)
from aurora_ict.timing.power_of_3 import AmdPhase, amd_phase

logger = logging.getLogger(__name__)


def _log_alert_task_exc(task: asyncio.Task) -> None:
    """매매 알림 fire-and-forget task 의 done callback — 예외를 로그로 남긴다.

    2026-06-10: create_task 로 던진 알림 task 가 실패(format/네트워크)해도
    아무도 결과를 안 봐서 silent 로 묻혔다. 이 콜백으로 예외를 가시화한다.
    """
    try:
        exc = task.exception()
    except asyncio.CancelledError:
        return
    if exc is not None:
        logger.warning("매매 알림 발송 task 실패(silent 방지): %s", exc)

# 키 무효(retCode 10003)가 step 에서 이 횟수만큼 연속되면 봇 자동 정지 —
# 무한 재시도로 인한 로그 폭증·502 차단. 사용자는 거래소 키 재등록 후 재가동.
_AUTH_FAIL_STOP_THRESHOLD = 3

_NY_TZ = ZoneInfo("America/New_York")


class ExchangeClientProtocol(Protocol):
    """Aurora 측 ``CCXTClient``와 호환되는 duck-typed 인터페이스."""

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int,
    ) -> list[list[Any]]: ...

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float | None = None,
        reduce_only: bool = False,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]: ...

    async def fetch_position(self, symbol: str) -> dict[str, Any] | None: ...

    async def fetch_balance(self) -> dict[str, Any]: ...

    async def fetch_ticker(self, symbol: str) -> float | None: ...

    async def cancel_all_orders(self, symbol: str) -> None: ...

    # 2026-07-22: 봇 주문 태그 기반 — 선별취소 + 고아 포지션 소유권 판정.
    async def cancel_bot_orders(self, symbol: str) -> int: ...

    async def position_opened_by_bot(
        self, symbol: str, side: str, entry_price: float, qty: float = 0.0,
    ) -> bool: ...

    async def fetch_closed_positions(
        self, since_ms: int | None = None, limit: int = 200,
    ) -> list[Any]: ...

    async def modify_stop_loss(
        self, symbol: str, new_stop_loss: float,
    ) -> dict[str, Any]: ...

    async def set_position_tpsl(
        self,
        symbol: str,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        tp_size: float | None = None,
        tpsl_mode: str = "Full",
        trailing_stop: float | None = None,
        active_price: float | None = None,
    ) -> dict[str, Any]: ...

    async def set_leverage(
        self, symbol: str, leverage: int,
    ) -> dict[str, Any]: ...

    async def fetch_actual_leverage(
        self, symbol: str,
    ) -> int | None: ...


class BotState(StrEnum):
    """봇 상태."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


# 보호 SL fallback 폭 — 복구 등으로 SL 거리를 모를 때 entry 대비 이 비율로 임시 SL.
# 0.5% (20x 면 ~10% 위험) — 무SL 방치보단 안전. 추후 ATR 기반으로 정교화 가능.
_FALLBACK_SL_PCT = 0.005

# 횡보 게이트 롤링 분위(#REGIME-ROLLING 2026-06-23) — 페어별 q33 하드코딩 대신
# 최근 N개 setup 의 |진입추세%| 33분위를 실시간 floor 로(페어 변동성 자동 적응).
REGIME_ROLLING_WINDOW = 150  # 분위 표본 윈도우 (deque maxlen)
# 최소 표본 — 미만이면 q33 하드코딩 fallback. 2026-06-23 CSV 정합비교: 라이브
# 변동성이 백테 2.4배라 하드코딩 fallback 이 실제보다 낮음 → 롤링 인계를 빠르게
# (30→20) 해 배포 직후 부정확 구간 단축. 후보전체 정밀보정은 shadow 데이터 후.
REGIME_ROLLING_MIN = 20

# DOL 역방향 진입 감점 (#3 보완). 지배적 draw 와 반대인 setup 의 confluence_score 를
# 이만큼 깎아 B+ 게이트(min_confluence)에서 걸러지게 함. 2 = 보통 setup 은 컷, A급만 통과.
_DOL_COUNTER_PENALTY = 2

# SMT divergence 상관 페어 (#SMT 2026-06-06). BTC↔ETH 만 — 상관도 높아 다이버전스
# 신뢰도 유효. 짝 없는 알트 심볼은 SMT 평가 skip (매핑에 없으면 None).
_SMT_CORR_PAIRS: dict[str, str] = {
    "BTCUSDT": "ETHUSDT",
    "ETHUSDT": "BTCUSDT",
}

# 2026-07-02 #FLIP-REFINE (Origo 1.3, FST #2 실거래 반사실 검증):
# flip target 최소 가중치 — 15m(weight=2) 존은 flip 청산 트리거에서 제외, 1h(4)+ 만.
# 근거: flip 절단 122건을 실시세 72h 로 반사실 추적 — @15m 87건은 보유가 우월
# (Δ+46R 승자 절단), @1h~1d 는 flip 이 우월(진짜 반전 방어). 진입 override 평가
# (가중치 합 threshold)는 그대로 — target "선정"만 1h+ 존으로 제한.
_FLIP_TARGET_MIN_WEIGHT = 4
# flip 역진입(청산 후 반대 방향 신규 진입) 스위치 — 실측 113건 net -301 USDT,
# 승률 19%, 전 TF·전 모델 적자(robust) → 기본 OFF. 청산(방어)만 하고 역진입 안 함.
# 되돌리려면 True (레거시 경로 보존).
_FLIP_REVERSE_ENABLED = False


@dataclass(slots=True)
class _ActivePosition:
    """현재 진입 중인 position state."""

    direction: Direction
    entry: float
    stop_loss: float
    take_profit: float
    qty: float
    setup_ts_ms: int
    # 변경 3: HTF FVG override — 봉 close 가 이 zone 안 들어오면 flip.
    htf_flip_target: HtfFvgEntry | None = None
    # LTF setup 가중치 (5m=1 등). flip 판정 시 가중치 비교에 사용.
    ltf_weight: int = 1
    # 2026-05-28: 학습/복기용 trade dataset 위해 진입 컨텍스트 보존
    # (_PendingEntry → _ActivePosition 승격 시 같이 박힘).
    entry_ts_ms: int = 0           # 체결 시각 (close 시점에 duration 계산)
    context_json: str | None = None  # 진입 setup confluences/source/window/HTF FVG snapshot
    equity_at_entry: float = 0.0   # 진입 시점 equity (참고용; close 시 dataset 에 박음)
    # 분할익절(#PARTIAL-TP 2026-06-23) — 진입 시 계산한 TP1(partial_tp_rr×R) 가격.
    # 도달 시 50% reduce_only 청산 + (partial_be 면) 나머지 SL 본전. 0=분할 비대상.
    tp1_price: float = 0.0
    partial_done: bool = False  # TP1 부분익절 1회 실행 여부 (중복 청산 방지)
    # partial_tp_exchange 시 거래소 Partial TP 등록 성공 여부 (#PARTIAL-TP-FALLBACK
    # 2026-06-25). False(기본)면 폴링(_maybe_partial_exit)이 1차 익절 담당 — 진입수량
    # 50%가 거래소 최소주문 미만이면 Partial TP 가 거부되므로, 등록 성공 시에만 True
    # 로 올려 폴링을 끈다(이중청산 방지). 소액 포지션 1차익절 구멍 방지.
    partial_on_exchange: bool = False
    # #TRAIL-EXCHANGE (Origo 1.4): 거래소 트레일링 무장 성공 여부. True 면 TP=5R
    # 원거리 + 분할익절 skip(폴링·거래소 둘 다), 미구분 거래소 청산은 trail_stop 분류.
    trail_armed: bool = False
    # #BE-LOCK (Origo 1.5): 본전 잠금 1회 실행 여부 (중복 이동 방지).
    be_moved: bool = False


@dataclass(slots=True)
class _PendingEntry:
    """marketable limit entry 미체결 대기 상태 (#LIVE-1 fix).

    시장가 슬리피지 회피용 — 현재가 바로 앞 (캔들 앞) 에 limit 을 박고 체결 대기.
    SL/TP 는 entry 주문에 동봉되어 체결 시 거래소가 포지션에 conditional 적용.
    ``ttl_ms`` 안에 체결 안 되면 취소 (그 타점 포기, 다음 신호 대기).
    """

    direction: Direction
    entry: float            # limit 가격 (체결 시 active_position.entry)
    stop_loss: float
    take_profit: float
    qty: float
    setup_ts_ms: int
    placed_ts_ms: int       # 주문 전송 시각 (TTL 만료 판정 기준)
    htf_flip_target: HtfFvgEntry | None = None
    ltf_weight: int = 1
    # 등록 시점에 미리 빌드한 ENTRY 컨텍스트 JSON (confluences/source/window 등).
    # 체결 시 _record_trade(context_json=...) 로 넘겨 거래 저널에 풍부히 기록.
    context_json: str | None = None


@dataclass(slots=True)
class BotIctInstance:
    """ICT Silver Bullet 봇 instance.

    Attributes:
        client: Bybit client (Aurora 측 ``CCXTClient``를 어댑터로 감싼 것).
        symbol: 거래 symbol (e.g. "BTCUSDT").
        timeframe: OHLCV timeframe (표준 "1m").
        leverage: 사용 leverage.
        position_pct_base / _max / _step: confluence-based notional sizing.
        min_rr: 최소 RR (표준 2.0).
        step_interval_sec: step 호출 간격 (표준 60s).
        ohlcv_limit: fetch 봉 수 (표준 200).
    """

    client: ExchangeClientProtocol
    symbol: str = "BTCUSDT"
    timeframe: str = "1m"
    # Notional-based sizing — confluence_score 단계별 % (40 → 55 → 70 → 90, step=15).
    # margin = equity * pct / 100, notional = margin * leverage, qty = notional / entry.
    leverage: int = 20
    position_pct_base: float = 40.0
    position_pct_max: float = 90.0
    position_pct_step: float = 15.0
    # 2026-06-06 리스크 기반 sizing — True 면 qty = risk금액 / SL거리 (손실 고정).
    risk_based_sizing: bool = False
    risk_per_trade_base: float = 1.0
    risk_per_trade_step: float = 0.5
    risk_per_trade_max: float = 2.0
    min_rr: float = 2.0
    # B+ 등급 게이트 (#1/#8): HTF boost 까지 반영된 최종 confluence_score 가 이 값 미만이면
    # 진입 skip. 0=비활성(기존 동작). 등급 C0~1/B2~3/B+4~5/A6+ → 4 면 B+ 이상만.
    min_confluence: int = 0
    # 고RR 예외 — confluence 미달이어도 rr 이 이 값 이상이고 score>=1 이면 통과.
    # 0=비활성. 파트너 결정 2026-06-04 (손익비 좋은 1점 단일신호 셋업 진입 허용).
    high_rr_bypass_min_rr: float = 0.0
    step_interval_sec: int = 60
    ohlcv_limit: int = 200
    fvg_min_size_pct: float = 0.0005
    # SMT divergence (BTC↔ETH 상관) confluence 가점 활성 — #SMT 2026-06-06.
    smt_enabled: bool = True
    # FVG 이후 N 봉 안에 retest 없으면 진입 skip. 1h → 10시간.
    setup_stale_bars: int = 10
    # LuxAlgo SB Strict mode — True 면 FVG mean threshold 까지 retrace 된 setup 만 진입.
    require_retrace: bool = False
    # Setup 시간 윈도우 확장 — True 면 Killzone 전체 (Asian/London/NY_AM/Close/PM),
    # False 면 Silver Bullet 1시간 윈도우만 (NY 3-4am/10-11am/2-3pm).
    expand_to_killzone: bool = True
    # 24h 매매 — True 면 SB / Killzone 시간 필터 완전 skip. expand_to_killzone 보다 우선.
    disable_time_filter: bool = True
    # #NYPM-GATE 2026-07-16 (FST#5): NY_PM(NY 13:30-16:00 = 02-05 KST) 진입 차단.
    # 근거=삼중검증: 라이브 진입기준 승률 10%/-29(1.8 손실 81%), 5년 백테 7/7페어
    # 음수(제외 시 +4.3%→+17.7%), 6/24 킬존연구 NY_PM 최악. NY_PM 은 정통 ICT 상
    # reversal 구간이라 추세추종형 Origo 와 상충. disable_time_filter(24h)·구독
    # 두 티어 모두 적용 (24h 라도 NY_PM 만은 예외 차단).
    exclude_nypm: bool = True
    # #SMART-SIZE 2026-07-20 (FST#7): 품질 기반 자금배분. 진입 시 볼륨(20봉평균↑)·
    # Nadaraya-Watson 중심선 정합·RSI 방향정합 3신호로 품질점수(0~3) → 사이즈 배수
    # clip(0.7+q*0.2, 0.4, 1.4). 좋은 품질=자금↑, 나쁨=자금↓. LuxAlgo 신호계열 대입
    # 결과 유일 walk-forward robust(net/MDD 4.24→4.68, 양반기 개선). 변동성타겟팅은
    # 인과조건서 비robust라 제외. 거래 필터 아닌 배분이라 빈도 불변.
    smart_size_enabled: bool = True
    # #MMBM 2026-07-21 (FST#7): 마켓메이커 반전 모델을 2번째 진입으로 병렬 가동.
    # SB(Silver Bullet) 셋업 없을 때만 시도 — HTF정합 방향 discount/premium 반전
    # (CHoCH). 자체 조건으로 검증돼 SB 게이트는 우회하되 _execute_setup 의 리스크
    # 레이어(서킷브레이커·일일한도·사이징·DD스로틀·maker 지정가) 는 공유. 매매기록
    # model 태그로 SB 와 분리 실측. ⚠️ maker(지정가) 전제·횡보장 약세 — 실측 관찰용.
    mmbm_enabled: bool = True

    # Multi-TF 모드 — True 면 HTF (Trade TF 위 모든 단계) setup 추적 + LTF (Trade TF)
    # 에서 retrace + structure shift + FVG confirm 시 진입. ICT 정통 multi-TF framework.
    # False 면 기존 단일 TF 방식.
    multi_tf: bool = False
    # Multi-TF LTF confirmer lookback (LTF 봉 단위).
    multi_tf_ltf_lookback: int = 30

    # Structure-based trailing stop (ICT 정통) — True 면 진입 후 새 swing 형성 시 SL 이동.
    enable_trail: bool = False
    # Trail SL buffer ratio (swing 가격 × ratio 만큼 buffer).
    trail_buffer_ratio: float = 0.001

    # use_market_entry (#LIVE-1 fix 후 의미 변경):
    # - False (기본/권장): marketable limit entry — 현재가 바로 앞 (캔들 앞) 에 지정가 +
    #   SL/TP 동봉. 슬리피지 0 (체결가 보장). entry_limit_ttl_sec 안에 미체결이면 취소.
    # - True (레거시, 비권장): 즉시 시장가 진입 — slippage 발생 (TP 가 fill 만큼 밀려
    #   목표 liquidity 못 먹던 #LIVE-1 원인).
    use_market_entry: bool = False
    # marketable limit 미체결 TTL — 이 시간 지나면 pending 취소 (그 타점 포기).
    # 사용자 결정 2026-05-22: 10분 (5m 2봉). 시장가처럼 거의 즉시 체결되되 슬리피지 0.
    entry_limit_ttl_sec: int = 600
    # min_sl_distance_pct: SL 거리 / entry 가 이 비율 미만이면 setup skip (noise-trap 회피).
    min_sl_distance_pct: float = 0.0
    # max_sl_distance_pct: SL 거리 / entry 가 이 비율 초과면 setup skip (비정상 큰 SL 차단).
    # 0 = 비활성.
    max_sl_distance_pct: float = 0.0
    # ote_level: FVG 되돌림 진입 깊이. 0.5=CE(기존), 0.707=깊은 OTE 진입.
    # 2026-06-23 안정형 하이브리드 연구: 0.707+swing+횡보게이트+분할익절 = 시드방어형
    # 최선점(net 흑자 유지·최대DD↓·체감승률↑). 0.707 ≈ ICT OTE sweet spot.
    ote_level: float = 0.707
    # regime_filter_enabled: 횡보 국면(|진입추세%| < 페어별 floor) 진입 회피 게이트.
    # 2026-06-23 연구: 횡보 국면은 모든 TP 적자라 회피가 net 흑자의 필수조건
    # (게이트 생략·고정공통 임계는 7페어 net 적자, 페어별 q33 만 흑자 +18).
    regime_filter_enabled: bool = True
    # regime_rolling_enabled: 횡보 floor 를 롤링 33분위로(표본>=REGIME_ROLLING_MIN).
    # False 면 페어별 q33 하드코딩 고정. 표본 부족(초기)이면 자동 하드코딩 fallback.
    regime_rolling_enabled: bool = True
    # #COND-ALIGN 2026-07-17 (Origo 2.0): 조건부 방향정합 게이트. 약/중추세에선
    # 진입방향이 20봉 추세와 정합일 때만, 강추세(|trend|>=q70)면 반전도 허용.
    # regime_filter(q33 크기 회피)와 직교(방향 규율). 극톱질 역추세진입(-1.0) 제거,
    # 5년 net +17.7→+21.3, walk-forward robust. STRONG_TREND_FLOOR fallback.
    cond_align_enabled: bool = True
    # partial_tp_exchange: 진입 시 거래소에 TP 2개(1.5R 50%+swing 50%) Partial 등록 →
    # 봇 폴링(_maybe_partial_exit) 대체. ⚠️ Bybit Partial SL/TP 모드·tpSize 동작 소액
    # 실측 전엔 OFF(미배포). True 면 _setup_partial_tps 사용 + 폴링 분할 skip.
    partial_tp_exchange: bool = False
    # REGIME_TREND_FLOOR: 페어별 |진입추세%| 하위33%(q33) floor — 미만이면 횡보 skip.
    # 2026-06-23 백테 7페어 q33 값(추후 롤링 분위로 전환 예정). 미등록 페어는 0(게이트 off).
    REGIME_TREND_FLOOR: ClassVar[dict[str, float]] = {
        "BTCUSDT": 0.230, "ETHUSDT": 0.268, "SOLUSDT": 0.396, "XRPUSDT": 0.271,
        "DOGEUSDT": 0.275, "LINKUSDT": 0.315, "HYPEUSDT": 0.527,
    }
    # STRONG_TREND_FLOOR: 페어별 |진입추세%| 상위30%(q70) — 이상이면 강추세로 보고
    # cond_align 에서 반전(역추세) 진입 허용. 2026-07-17 백테 7페어 q70 값(NY_PM 제외).
    STRONG_TREND_FLOOR: ClassVar[dict[str, float]] = {
        "BTCUSDT": 0.401, "ETHUSDT": 0.365, "SOLUSDT": 0.598, "XRPUSDT": 0.381,
        "DOGEUSDT": 0.721, "LINKUSDT": 1.036, "HYPEUSDT": 0.924,
    }
    # partial_tp_rr: 분할익절 — TP1(이익 partial_tp_rr×R) 도달 시 50% reduce_only 청산.
    # 0=off. partial_be=True 면 부분익절 후 나머지 50% SL 을 본전으로(무손실 보호).
    # 2026-06-23 연구: 분할 + 본전이동 = 체감승률↑·연속손절↓, 단타/스윙 트레이드오프
    # 우회. RR 점검(고래 RR사수 교훈): 1.0R 은 RR 0.92로 비대칭 역전(net 약함) →
    # 1.5R 채택 = RR 1.47·net+69(횡보회피포함)·체감승률 43%·연속손절 6.1. RR 더 보존.
    partial_tp_rr: float = 1.5
    partial_be: bool = True
    # #TRAIL-EXCHANGE 2026-07-02 (Origo 1.4): 거래소 네이티브 트레일링.
    # trigger_r×R 이익 도달 시 활성(activePrice), dist_r×R 거리로 tick 추적
    # (Bybit trading-stop — 봇 사망/재시작 무관). 0 = off(고정 TP 모드).
    # 정합 스윕(7페어 5년): 고정tp +124 → trail 2.0/1.5 +240 (RR 0.89→1.82).
    # 무장 성공 시 TP 5R 확장 + 분할익절 skip. 무장 실패 시 고정 TP 모드 유지(무해).
    trail_trigger_r: float = 0.0
    trail_dist_r: float = 0.0
    # #BE-LOCK 2026-07-07 (Origo 1.5): 이익 r×R 도달 시 SL 본전 이동. 0=off.
    # MFE 실측(1.2 손절 23%가 +20% ROI 이상 간 뒤 풀손절) 처방 — 트레일 활성(2R)
    # 전 1R~2R 구간 보호. 백테 BE@1R+trail +278 (기준 +240)·DD 265→228·강건 확인.
    be_trigger_r: float = 0.0
    # #SWEEP-GATE 2026-07-07 (Origo 1.5): 일봉 스윕-반전 후 K일 역방향 진입 차단.
    # 0=off. ICT 정통 "유동성 사건 = bias 즉시 전환"의 기계화 — EMA align 이
    # 3~5일 지연해 7/4 숏 전멸(-165) 낸 패턴 방어. 백테 K=2: +253·DD -25%.
    sweep_gate_days: int = 0
    # 스윕 게이트 일일 캐시 (day_key, block_short, block_long) — 일봉은 하루 단위.
    _sweep_gate_state: tuple | None = None
    # #REGIME-OTE 2026-07-10 (Origo 1.7): 상승 국면(일봉 20일 z>0.75) 전용 OTE.
    # 0=off. 국면 랩 검증 — 상승장 얕은 되돌림 롱은 역선택(-96%), 0.786 깊이만
    # 전/후반 동시 개선(+94%). 조건부 적용 시 타국면 불변, 합계 +6.7%.
    ote_up_level: float = 0.0
    # 국면 일일 캐시 (day_key, is_up) — 일봉은 하루 단위로만 바뀜.
    _regime_state: tuple | None = None
    # #DD-THROTTLE 2026-07-10 (Origo 1.8): 계좌 낙폭 > pct% 면 신규 진입 리스크에
    # factor 곱(anti-martingale). 0=off. 포트폴리오 복리 시뮬 — 일일스탑15%와
    # 조합 시 29.2x/MDD 80%/최악일 -30% (기준 23.9x/90%/-46%). peak equity 는
    # 사용자 데이터 폴더 json 으로 영속(재시작 생존). 입출금 시 왜곡 가능(보수 측).
    dd_throttle_pct: float = 0.0
    dd_throttle_factor: float = 0.7
    _peak_equity: float = 0.0
    # 2026-06-11 #EDGE-V2: SL 거리 배수 (1.0=원본). 백테스트 10국면 검증 —
    # 넓힐수록 스탑헌트 생존으로 단조 개선, x3.0 에서 BTC IN/OUT 흑자.
    # TP 는 원 RR 유지 비례 확장. risk sizing ON 이면 건당 손실(R) 불변.
    sl_dist_mult: float = 1.0
    # 2026-06-18 #CT-SL: 역추세 진입(signed_trend < ct_trend_threshold)이면 SL 배수를
    # sl_dist_mult 대신 이 값으로. 0=비활성. Origo 구독제는 settings validator 가 4.0
    # 강제(순추세/횡보 x3, 역추세 x4 — 7페어 5년 robust, net +4.0%p).
    sl_dist_mult_ct: float = 0.0
    ct_trend_threshold: float = 0.0
    # 2026-06-11 #SHADOW: FSD-style 데이터 플라이휠 — 게이트에 걸려 *거른* setup
    # 도 특징과 함께 JSONL 기록(행동 영향 0). 진입한 것만 기록하면 학습 데이터가
    # 편향·희소해지므로, 거른 자리의 사후 결과까지 모아 오프라인 학습 재료로.
    shadow_log_enabled: bool = True
    # max_entry_distance_pct: setup.entry 와 현재가 차이가 이 비율 초과면 setup skip.
    # 0 = 비활성. 너무 멀리 박힌 limit 의 미체결 대기 시간 회피.
    max_entry_distance_pct: float = 0.0
    # heartbeat_interval_sec: step loop 살아있음 INFO 로그 주기. 0 = 비활성.
    heartbeat_interval_sec: int = 0
    # #SAFETY-1: 자본 대비 % 일일 손실(SL) 한도. 0 = 비활성. NY local 자정 reset.
    daily_loss_limit_pct: float = 0.0
    # 2026-06-10 조윤 건의: 자본 대비 % 일일 수익(TP) 한도. 도달 시 그날 신규
    # 진입 중단(active position 은 유지). 0 = 비활성. NY local 자정 reset.
    daily_profit_limit_pct: float = 0.0
    # 2026-06-12 파트너: 페어별 일일 손실 한도 — R(리스크%) 배수 단위. 이 페어의
    # 오늘 누적 손실이 R×배수에 닿으면 *이 페어만* 당일 진입 중단 (다른 페어
    # 계속). 단일 페어 폭주(6/6: 한 페어 19연속 -33%) 차단용. 0 = 비활성.
    daily_pair_loss_limit_r: float = 2.0

    # HTF EMA bias 필터 — multi_tf 와 별개. 진입 직전 htf_ema_bias_tf EMA20 vs 가격
    # 비교 → setup.direction 이 bias 와 반대면 진입 skip.
    htf_ema_bias_enabled: bool = False
    htf_ema_bias_tf: str = "1h"
    htf_ema_bias_period: int = 20
    # 2026-06-10 #ALIGN: 다중 EMA 정렬 게이트 (백테스트 검증 — 단일 EMA20 strict
    # 보다 방향 정확도↑). htf_ema_bias_enabled 이고 이게 True 면 EMA20 단일 대신
    # 인접 EMA 쌍(periods) 정배열/역배열 점수로 방향 게이트. |점수|>=threshold 면
    # 그 방향만 진입, 미만이면 추세 불명확 → 진입 자제(되돌림/횡보 whipsaw 회피).
    htf_ema_align_enabled: bool = False
    htf_ema_align_periods: tuple[int, ...] = (60, 120, 200, 350, 480, 620)
    htf_ema_align_threshold: int = 2

    # 변경 3: HTF FVG override 모드 — "off"/"A"/"C".
    # A = 진입 직전 차단만, C = 진입 + 봉 close 기준 flip + re-entry.
    htf_override_mode: str = "C"
    htf_fvg_tfs: tuple[str, ...] = ("15m", "1h", "2h", "4h", "1d", "1w")

    # 변경 7: 실시간 flip watcher (WS primary + REST polling fallback).
    flip_watch_enabled: bool = True
    flip_watch_ws_url: str = "wss://stream.bybit.com/v5/public/linear"
    flip_watch_polling_interval_sec: float = 0.2
    flip_watch_ws_reconnect_max: int = 5
    # FVG 약화 임계치 — touch 누적 도달 시 mitigation 후보 / flip target 제외.
    htf_fvg_max_touch_count: int = 3

    # 2026-05-29 #HTF-LTF-CONFLICT: HTF FVG bull/bear 우세 + LTF setup 반대 방향 차단.
    # 0 = 비활성, 1.10 = bull 이 bear 의 110% 이상이면 short setup 차단 (역도 마찬가지).
    htf_ltf_conflict_guard_ratio: float = 1.10

    # 2026-05-29 SaaS 매매 로그 격리 — 사용자별 trades.* 디렉토리 인자.
    # None 이면 paths.data_dir() (단일 사용자 / .exe 흐름) — backward compat.
    # MultiUserBotManager 가 사용자별 dir (<data_dir>/users/<code>/) 을 주입해
    # 다른 사용자 거래와 섞이지 않게 격리. JSONL/SQLite write 충돌도 자연 해소.
    trades_data_dir: Path | None = None

    # 2026-05-29 PR 매매기록 mode (파트너 요청): 이 봇 인스턴스가 어느 run_mode
    # 로 가동되는지 — TradeEvent.mode 에 그대로 박혀 UI 가 DEMO/LIVE 표시.
    # None 이면 record 시 None 박힘 (구식 흐름 호환).
    run_mode: str | None = None

    # 2026-06-08: 텔레그램 매매 알림 — 이 봇 소유자 코드 + 알림 콜백.
    # MultiUserBotManager 가 주입. 콜백 시그니처: async (user_code, TradeEvent).
    # 미주입(None)이거나 user_code 빈 값이면 알림 안 보냄(.exe / 테스트 호환).
    user_code: str = ""
    alert_cb: Any = None
    # 2026-06-13: 일반 안내 콜백 (약관 미동의 등 1회성 사용자 액션 안내).
    notify_cb: Any = None

    state: BotState = field(default=BotState.STOPPED)
    active_position: _ActivePosition | None = field(default=None)
    # #MMBM 2026-07-21: 직전 실행된 셋업의 모델 태그 (_execute_setup 진입 시 setup.source
    # 로 갱신). 매매기록(_record_trade)에서 SB(Origo) 와 MMBM 을 분리하기 위함.
    _active_model: str = field(default=ORIGO_MODEL_NAME)
    # #LIVE-1 fix: marketable limit entry 미체결 대기 상태 (체결되면 active_position 으로 승격).
    _pending_entry: _PendingEntry | None = field(default=None)
    _task: asyncio.Task[None] | None = field(default=None)
    _last_setup_ts_ms: int = field(default=0)  # 동일 setup 중복 진입 방지
    # 2026-06-09: 방향까지 기록 — 같은 봉의 반대 방향(롱→숏 전환)은 차단 안 함.
    _last_setup_direction: Direction | None = field(default=None)
    _last_heartbeat_ms: int = field(default=0)
    # HTF 봉 캐시 — (tf, last_ts_ms) → DataFrame. 같은 봉이면 재fetch 안 함.
    _htf_cache: dict[str, tuple[int, pd.DataFrame]] = field(default_factory=dict)
    # Multi-TF tracker — multi_tf=True 시 lazy init.
    _htf_tracker: HtfSetupTracker | None = field(default=None)
    # 변경 3: HTF FVG map 캐시 — TF 봉 길이 만큼 지나야 재 빌드.
    _htf_fvg_map_cache: list[HtfFvgEntry] = field(default_factory=list)
    _htf_fvg_map_built_at_ms: int = field(default=0)
    # 봉당 1회 flip 검사 — 마지막으로 검사한 5m 봉 ts.
    _last_flip_check_bar_ts: int = field(default=0)
    # 변경 7: 같은 봉 ts 에 WS 가 이미 flip 했으면 step() 보조 검사 skip.
    _flip_done_for_ts: int = field(default=0)
    # 변경 7: 실시간 flip watcher.
    _flip_watcher: object | None = field(default=None)
    # 추세 평가 캐시 — (tf → (last_bar_ts, TrendState)).
    _trend_cache: dict[str, tuple[int, TrendState]] = field(default_factory=dict)
    # 매매 이벤트 영구 저장 (#BUG-2 해소) — lazy init in _record_trade.
    _trades_store: TradesStore | None = field(default=None)
    # 2026-06-17 #SYNC-FIX: record 실패한 이벤트 큐 — 거래소 체결됐는데 기록(JSONL/DB)
    # 이 일시 실패(디스크/락)로 누락되던 문제. 다음 _record_trade 성공 시 재기록.
    _failed_trade_events: list[TradeEvent] = field(default_factory=list)
    # #SAFETY-1 daily loss limit 추적 — NY local 자정 기준 reset.
    _today_realized_pnl_usdt: float = field(default=0.0)
    _today_date_str: str = field(default="")  # "YYYY-MM-DD" NY local
    _today_start_equity: float = field(default=0.0)
    _daily_limit_hit: bool = field(default=False)
    # 2026-06-10 조윤 건의: 일일 수익(TP) 한도 도달 flag — NY 자정 reset.
    _daily_profit_hit: bool = field(default=False)
    # 2026-06-13 약관 미동의(110123) 안내 1회 발송 플래그 (봇 인스턴스 단위).
    _terms_alerted: bool = field(default=False)
    # 2026-06-12 페어별 일일 손실 한도 — 이 심볼의 오늘 실현손익 + sticky flag.
    _today_pair_realized_pnl_usdt: float = field(default=0.0)
    _daily_pair_limit_hit: bool = field(default=False)
    # 키 무효(10003) step 연속 실패 카운터 — 임계치 도달 시 봇 자동 정지.
    _auth_fail_streak: int = field(default=0)

    # 2026-05-28 파트너 요청 — fly.io logs 에서 봇 의사결정 흐름 보이게.
    # 같은 값이 매 step 반복 출력 안 되도록 "직전 값" 캐시 — 변화 시에만 1줄 INFO.
    # 봇 동작 영향 0 (로깅 가시성 전용).
    _last_logged_htf_ema_bias: str = field(default="")  # "bullish"/"bearish"/"neutral"/""
    # align 게이트 skip 로그 변화 감지 — 블랙리스트 제거 후 매 step 재평가되므로
    # 같은 (score, threshold, 방향) skip 은 1회만 INFO (스팸 방지, 2026-06-11).
    _last_align_skip_log: str = field(default="")
    # #SHADOW: 최근 계산된 align 점수 캐시(특징 기록용) + 기록 중복 방지 set.
    _last_align_score: int | None = field(default=None)
    _shadow_seen: dict = field(default_factory=dict)
    _last_logged_htf_fvg_summary: str = field(default="")  # "bull_w=10 bear_w=4 n=8" 형태
    _last_logged_dol_draw: str = field(default="")  # "long"/"short"/"none"/""
    _last_logged_trend_summary: str = field(default="")  # trend_cache 직렬화 핑거프린트

    # 2026-05-27 파트너 요청 — UI 차트 TF 토글 시 로딩 없이 즉시 응답.
    # 봇 start 시 background prefetch 로 모든 TF 채워두고, /ict/ohlcv 가 cache 사용.
    # polling 시 마지막 N봉만 incremental refresh.
    # 2026-07-22: 봇별 캐시 → (symbol,tf) 전역 공유 캐시로 전환(중복제거·메모리절감).
    # 같은 심볼 여러 유저면 심볼당 1벌만 보유. 기본=GLOBAL_OHLCV_CACHE(전역 공유),
    # 테스트는 격리 위해 별도 인스턴스 주입 가능. 저장 자료: (symbol,tf)→ts오름차순 봉.
    _shared_ohlcv: SharedOhlcvCache = field(default_factory=lambda: GLOBAL_OHLCV_CACHE)
    # prefetch background task 핸들 (취소용).
    _prefetch_task: asyncio.Task[None] | None = field(default=None)

    # 2026-05-29 #SILENT-1~5: 조용한 오류 가시화 — 운영 중 어디가 실패하는지
    # judgment 응답 / 로그에서 즉시 확인 가능하게.
    # recovery_failed: 봇 시작 시 거래소 측 fetch_position 실패 (포지션 복원 불가).
    # 다음 step 에서 재시도 가능하도록 flag 유지. True 인 동안은 신규 진입 차단.
    _recovery_failed: bool = field(default=False)
    # 연속 fetch_position 실패 카운트 — 5회 누적 시 ERROR (네트워크/API 장애 의심).
    _sync_failure_streak: int = field(default=0)
    # #MANUAL-POS-RESPECT 2026-07-22: 봇 ENTRY 기록 없어 채택 거절한(유저 수동 추정)
    # 포지션 시그니처. reconcile 이 매 step recovery 를 재호출하는 로그스팸·DB부하
    # 방지 — 같은 시그니처면 조용히 skip. 포지션 소멸/채택 시 None 리셋.
    _declined_manual_sig: tuple[str, float, float] | None = field(default=None)
    # 진입 주문 실패 카운트 (place_order) — 누적 시 운영자 점검 신호.
    _order_failure_count: int = field(default=0)
    # 페어 확장 — 가동 시 fetch_symbol_meta 로 채우는 심볼별 거래소 메타
    # (min_qty / qty_step / max_leverage). 빈 dict 면 BTC 기준 0.001 폴백.
    _symbol_meta: dict[str, float | None] = field(default_factory=dict)
    # SL/TP 박기 실패 카운트 — 무SL 상태 가시화. step 마다 재시도.
    _tpsl_failure_streak: int = field(default=0)
    # 가장 최근 진입 setup direction 들 (recent 10) — judgment 응답에 노출하여
    # "long 비율 우세인데 short 만 진입" 같은 의문을 즉시 해소.
    _recent_setup_directions: list[str] = field(default_factory=list)
    # #REGIME-ROLLING 2026-06-23: 최근 setup |진입추세%| 이력 — 롤링 33분위 floor 계산용.
    _trend_history: deque[float] = field(
        default_factory=lambda: deque(maxlen=REGIME_ROLLING_WINDOW), repr=False,
    )

    async def start(self) -> None:
        """봇 기동 (background task 생성).

        시작 시 거래소 측 활성 포지션 fetch → active_position 복원.
        봇 재시작 시 거래소 측 포지션 인식 못해서 중복 진입 박는 위험 회피.
        """
        if self.state is BotState.RUNNING:
            logger.info("BotIctInstance %s 이미 실행 중", self.symbol)
            return
        # 페어 확장 — 가동 직후 심볼별 거래소 메타(min_qty/lot step/max leverage)
        # 캐시. 미지원 client(구 테스트) 나 조회 실패면 빈 dict → BTC 기준 폴백.
        if hasattr(self.client, "fetch_symbol_meta"):
            try:
                meta = await self.client.fetch_symbol_meta(self.symbol)
                self._symbol_meta = meta if isinstance(meta, dict) else {}
            except Exception as e:  # noqa: BLE001
                logger.warning("symbol meta 로드 실패 (%s): %s", self.symbol, e)
                self._symbol_meta = {}
        await self._recover_position_from_exchange()
        # 2026-06-12 고아 주문 청소 (LINK 사고): 재시작(강제 종료 배포)으로 stop()
        # 의 pending 취소가 못 돈 경우, 전생의 미체결 지정가가 거래소에 남아
        # 봇 모르게 체결된다(무SL 고아 포지션). 시작 시점엔 이 봇의 pending 의도가
        # 없으므로 trading order 전부 취소.
        # 2026-06-13 회귀 수정(#TPSL-STRIP): 활성 포지션이 있으면 skip — Bybit 의
        # 주문 동봉형 SL/TP 가 조건부 주문으로 살아 있어 cancel_all 이 보호장치를
        # 벗겨버렸다 (전 사용자 XRP/LINK 무방비 사고). 포지션 보유 시 고아
        # 지정가는 sync 의 방향 검증·입양이 처리하므로 청소는 무포지션일 때만.
        if self.active_position is None:
            try:
                n_cxl = await self.client.cancel_bot_orders(self.symbol)
                logger.info(
                    "startup 고아 주문 청소 — 봇 주문만 취소 %d건 (유저 수동 주문 보존, %s)",
                    n_cxl or 0, self.symbol,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("startup 고아 주문 청소 실패 (무시): %s", e)
        else:
            logger.info(
                "startup 고아 주문 청소 skip — 활성 포지션 보유 (SL/TP 보존, %s)",
                self.symbol,
            )
        # #RECONCILE 2026-06-06: 재기동(crash) 중 청산돼 trades DB 에 청산 이벤트가
        # 누락된 ENTRY(orphan)를 거래소 closed-pnl 로 대조해 보충. 봇 진행 막지 않게
        # 실패는 무시.
        try:
            await self._reconcile_orphan_entries()
        except Exception as e:  # noqa: BLE001
            logger.warning("reconcile 실패 (무시, 봇 진행): %s", e)
        self.state = BotState.RUNNING
        self._task = asyncio.create_task(self._run_loop())
        # 변경 7: 실시간 flip watcher 가동 (mode C + 활성화 시).
        if self.flip_watch_enabled and self.htf_override_mode == "C":
            from aurora_ict.bot.flip_watcher import FlipWatcher
            self._flip_watcher = FlipWatcher(
                self,
                ws_url=self.flip_watch_ws_url,
                polling_interval_sec=self.flip_watch_polling_interval_sec,
                reconnect_max=self.flip_watch_ws_reconnect_max,
            )
            await self._flip_watcher.start()  # type: ignore[attr-defined]
        # 2026-05-27: 시작 직후 background 로 모든 UI TF 차트 데이터 prefetch.
        # UI 가 TF 토글할 때 cache hit 으로 즉시 응답. await 안 함 (시작 차단 X).
        self._prefetch_task = asyncio.create_task(self._prefetch_all_ohlcv_tfs())
        logger.info("BotIctInstance %s 시작", self.symbol)

    def ensure_prefetch_started(self) -> None:
        """봇 가동 안 해도 OHLCV cache prefetch 시작 (idempotent).

        2026-05-28: SaaS UX — 로그인 후 START 전에도 차트 봉/마커 보이게.
        한 번 시작하면 ``_prefetch_task`` 보관, 이미 시작했거나 진행 중이면 noop.

        ``start()`` 와 별개 — state 는 변경하지 않음 (STOPPED 유지). UI lazy 로딩
        용 진입점이며, 사용자가 명시적으로 START 누르기 전엔 매매 loop 안 돔.
        """
        if self._prefetch_task is not None and not self._prefetch_task.done():
            return  # 이미 진행 중
        # 완료된 task 가 남아 있을 수도 — 새 task 로 교체해 재시도 가능하게.
        self._prefetch_task = asyncio.create_task(self._prefetch_all_ohlcv_tfs())

    @staticmethod
    def _pos_sig(
        direction: Direction, entry_price: float, qty: float,
    ) -> tuple[str, float, float]:
        """포지션 시그니처 (방향+entry+qty 반올림) — 미채택 수동 포지션 재시도 억제용."""
        return (direction.value, round(entry_price, 4), round(qty, 6))

    async def _recover_position_from_exchange(self) -> None:
        """봇 시작 시 거래소 측 활성 포지션 복원.

        fetch_position 호출 → contracts > 0 이면 active_position 채움.
        ts_ms / entry / SL / TP 박은 거 거래소 응답에서 추출 (없으면 추정값).

        2026-05-29 #SILENT-1: 실패 시 ``_recovery_failed=True`` 박음.
        다음 step 의 ``_sync_position_state`` 가 자동 재시도하고, 그동안 신규
        진입은 ``step()`` 에서 차단 (중복 진입 위험 회피).
        """
        try:
            pos = await self.client.fetch_position(self.symbol)
        except Exception as e:  # noqa: BLE001
            # 단순 warning 이었으나 — 복원 실패는 봇 재시작 후 중복 진입 위험.
            logger.error(
                "recover fetch_position 실패 (포지션 복원 불가, 신규 진입 차단): %s",
                e,
            )
            self._recovery_failed = True
            return
        # 정상 호출 됐으면 flag 해제 (정상 복원 또는 contracts=0 모두 OK).
        self._recovery_failed = False
        if pos is None:
            return
        contracts = float(pos.get("contracts", 0) or 0)
        if contracts <= 0:
            return
        # 방향 추정 — Bybit ccxt: side="long"/"short" 또는 size sign.
        side = (pos.get("side") or "").lower()
        if side in ("long", "buy"):
            direction = Direction.LONG
        elif side in ("short", "sell"):
            direction = Direction.SHORT
        else:
            logger.warning("recover: side 인식 실패 (%s) — skip", side)
            return
        entry_price = float(
            pos.get("entryPrice") or pos.get("entry_price") or pos.get("averagePrice") or 0,
        )
        if entry_price <= 0:
            logger.warning("recover: entry_price 인식 실패 — skip")
            return
        # SL/TP — Bybit V5 응답 stopLoss / takeProfit 에서 읽음.
        sl = float(pos.get("stopLossPrice") or pos.get("stop_loss") or 0) or 0.0
        tp = float(pos.get("takeProfitPrice") or pos.get("take_profit") or 0) or 0.0
        # #MANUAL-POS-RESPECT 2026-07-22 (TDAF 유저 컴플레인): 봇은 자기가 연
        # 포지션만 채택·관리한다. 매칭되는 미청산 ENTRY 기록(같은 방향 + entry 가격
        # 1% 이내)이 없으면 유저 수동 포지션으로 간주하고 절대 건드리지 않는다 —
        # 채택·보호SL·TP복원·트레일 전부 skip. (과거엔 무조건 입양해 유저 수동
        # 포지션에 봇이 SL/TP 를 걸어 컴플레인 발생. 봇 지정가 진입은 SL/TP 동봉이라
        # 설령 고아 체결이어도 거래소측 보호는 유지되므로 미채택이 안전.)
        # 단, 소유권 대조는 사용자별 기록(trades_data_dir)이 있는 SaaS 에서만 가능.
        # 단독 .exe(trades_data_dir=None)는 대조 불가라 기존 복구 동작 유지.
        if self.trades_data_dir is not None:
            _own = self._find_unclosed_entry_event(direction)
            _rec_match = (
                _own is not None
                and abs(_own.price - entry_price) <= entry_price * 0.01
            )
            # 봇 ENTRY 기록이 없어도, 거래소 주문 이력에 봇 태그(orderLinkId=AUR*)가
            # 있으면 봇이 연 포지션(고아 체결·DB 유실 대비 — "우리만 아는 표시").
            # 기록 매칭 또는 태그 매칭 중 하나면 봇 포지션으로 채택.
            _tag_match = False
            if not _rec_match:
                _tag_match = await self.client.position_opened_by_bot(
                    self.symbol, direction.value, entry_price, contracts,
                )
            if not _rec_match and not _tag_match:
                logger.warning(
                    "recover: 미추적 포지션(봇 ENTRY 기록·주문태그 모두 없음) — 유저 "
                    "수동 포지션으로 간주해 채택·SL/TP 미설정 (%s %s entry=%.4f qty=%.4f)",
                    self.symbol, direction.value, entry_price, contracts,
                )
                # 재시도 스팸 방지 — 이 포지션은 다시 시도/로그하지 않음.
                self._declined_manual_sig = self._pos_sig(
                    direction, entry_price, contracts,
                )
                return
        # #RESTORE-PARTIAL 2026-06-24: 복원 포지션도 분할익절 대상(파트너 요청).
        # SL 이 본전이면 이미 부분익절된 것으로 보고 재청산 방지, 아니면 tp1 계산.
        _r_tp1, _r_done = self._restore_partial_state(entry_price, sl, direction)
        self.active_position = _ActivePosition(
            direction=direction,
            entry=entry_price,
            stop_loss=sl,
            take_profit=tp,
            qty=contracts,
            setup_ts_ms=0,  # recovery — 원 setup ts_ms 알 수 없어 0
            tp1_price=_r_tp1,
            partial_done=_r_done,
        )
        self._declined_manual_sig = None  # 봇 포지션 채택 — 거절 시그니처 해제
        logger.info(
            "recover: 활성 포지션 복원 — %s %s entry=%.4f qty=%.4f sl=%.4f tp=%.4f",
            self.symbol, direction.value, entry_price, contracts, sl, tp,
        )
        # #BUG-2 해소: trades_store 에 RECOVERED 이벤트 기록.
        self._record_trade(
            TradeEventType.RECOVERED,
            direction=direction,
            price=entry_price,
            qty=contracts,
            reason="bot startup — exchange position fetch",
        )
        # 2026-06-12 파트너: 복구 포지션 TP=0 처리 — 자기 매매 기록(미청산
        # ENTRY 의 context)에서 원 TP 를 찾아 거래소에 재설치. setup_ts 도
        # 함께 복원해 이후 청산 분류(SL/TP)·기록 대조가 정상 동작하게.
        ent = self._find_unclosed_entry_event(direction)
        if ent is not None and abs(ent.price - entry_price) <= entry_price * 0.01:
            self.active_position.setup_ts_ms = ent.setup_ts_ms or 0
            if tp <= 0:
                try:
                    _ctx = json.loads(ent.context_json or "{}")
                except (ValueError, TypeError):
                    _ctx = {}
                rec_tp = float(_ctx.get("tp") or 0.0)
                if rec_tp > 0:
                    ok = False
                    try:
                        ok = await self.client.set_position_tpsl(
                            self.symbol,
                            stop_loss=(sl if sl > 0 else None),
                            take_profit=rec_tp,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning("recover: TP 복원 호출 실패: %s", e)
                    if ok:
                        self.active_position.take_profit = rec_tp
                        tp = rec_tp
                        logger.info(
                            "recover: TP 복원 — %s tp=%.4f (원 ENTRY setup_ts=%s)",
                            self.symbol, rec_tp, ent.setup_ts_ms,
                        )
                    else:
                        # 가격이 이미 TP 를 지나쳤거나 거래소 거부 — 0 유지
                        # (admin 전체 포지션 표에 드러나 수동 판단 가능).
                        logger.warning(
                            "recover: TP 복원 실패 — tp=0 유지 (%s, 원 TP %.4f)",
                            self.symbol, rec_tp,
                        )
                else:
                    logger.info(
                        "recover: 원 ENTRY 에 TP 기록 없음 — tp=0 유지 (%s)",
                        self.symbol,
                    )
        elif tp <= 0:
            logger.info(
                "recover: 매칭되는 미청산 ENTRY 기록 없음 — tp=0 유지 (%s)",
                self.symbol,
            )
        # P1-2: 거래소 측 SL 이 없는(=0) 채 복구되면 무SL 포지션 → 보호 SL 적용 (안 되면 청산).
        if sl <= 0:
            logger.warning("recover: SL 없는 포지션 — 보호 SL 적용 시도 %s", self.symbol)
            await self._ensure_protective_sl(tp if tp > 0 else None, 0.0)
        # #TRAIL-EXCHANGE: 복구 포지션도 트레일 재무장. 거래소에 이미 동일 트레일이
        # 걸려있으면 Bybit 34040(not modified) → alreadySet 처리로 무해. SL 이 본전
        # 이동된 포지션은 risk(|entry-sl|)가 원 R 과 달라질 수 있으나 근사 수용.
        if self.active_position is not None:
            await self._arm_trailing()

    def _find_unclosed_entry_event(self, direction: Direction):
        """복구 보조 — 이 심볼·방향의 청산 기록이 없는 최신 ENTRY 이벤트.

        reconcile(#RECONCILE)과 동일한 대조 규칙: 같은 symbol 의 청산류
        이벤트(setup_ts 기준)에 안 잡힌 ENTRY 중 가장 최근 것. TP/setup_ts
        복원용 — 실패·미발견은 None (복구 자체는 계속).

        Args:
            direction: 복구된 포지션 방향 (ENTRY 방향과 일치해야 매칭).

        Returns:
            TradeEvent 또는 None.
        """
        # 사용자별 기록 디렉토리가 주입된 경우(멀티유저 SaaS)만 동작 — 전역
        # 경로 store 는 단독 .exe/테스트의 무관한 과거 기록이 섞여 있어 엉뚱한
        # TP 를 복원할 수 있다. (store 가 이미 init 됐어도 이 가드가 우선 —
        # _record_trade(RECOVERED) 가 전역 경로로 먼저 init 하는 경우 방어.)
        if self.trades_data_dir is None:
            return None
        if self._trades_store is None:
            try:
                self._trades_store = TradesStore(self.trades_data_dir)
            except Exception as e:  # noqa: BLE001
                logger.warning("recover: TradesStore init 실패 — TP 복원 skip: %s", e)
                return None
        try:
            events = self._trades_store.all_events()
        except Exception as e:  # noqa: BLE001
            logger.warning("recover: all_events 실패 — TP 복원 skip: %s", e)
            return None
        close_types = {
            TradeEventType.SL_HIT, TradeEventType.TP_HIT,
            TradeEventType.SYNC_CLOSE, TradeEventType.MANUAL_CLOSE,
            TradeEventType.FLIP_CLOSE,
        }
        closed_ts = {
            ev.setup_ts_ms for ev in events
            if ev.symbol == self.symbol
            and ev.event_type in close_types and ev.setup_ts_ms
        }
        cands = [
            ev for ev in events
            if ev.symbol == self.symbol
            and ev.event_type is TradeEventType.ENTRY
            and ev.direction == direction.value
            and ev.setup_ts_ms and ev.setup_ts_ms not in closed_ts
        ]
        return max(cands, key=lambda e: e.ts_ms) if cands else None

    async def stop(self) -> None:
        """봇 정지 (background task cancel).

        2026-05-27 파트너 요청 — pending entry (지정가 미체결) 있으면 즉시 취소.
        STOP 후에도 거래소 측 미체결 limit 이 남으면 봇이 다시 켜질 때 의도치
        않게 체결될 수 있음. cancel_all_orders 는 trading order 만 — position
        attached SL/TP conditional 은 영향 X (Bybit V5 category 분리).
        """
        # 1) pending limit entry 취소 (active_position 의 SL/TP 는 무관)
        if self._pending_entry is not None:
            pe = self._pending_entry
            try:
                await self.client.cancel_bot_orders(self.symbol)
                logger.info(
                    "STOP: 지정가 미체결 (entry=%.4f qty=%.6f) 취소",
                    pe.entry, pe.qty,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("STOP pending 취소 실패: %s", e)
            self._pending_entry = None
        self.state = BotState.STOPPED
        # 변경 7: flip watcher 정지.
        if self._flip_watcher is not None:
            try:
                await self._flip_watcher.stop()  # type: ignore[attr-defined]
            except Exception as e:  # noqa: BLE001
                logger.warning("flip watcher 정지 실패: %s", e)
            self._flip_watcher = None
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # 2026-05-27: OHLCV prefetch task 도 같이 취소
        if self._prefetch_task is not None:
            self._prefetch_task.cancel()
            try:
                await self._prefetch_task
            except asyncio.CancelledError:
                pass
            self._prefetch_task = None
        logger.info("BotIctInstance %s 정지", self.symbol)

    async def _run_loop(self) -> None:
        """매 step_interval_sec 마다 step()을 호출하는 메인 루프."""
        import time as _time
        while self.state is BotState.RUNNING:
            try:
                await self.step()
                self._auth_fail_streak = 0  # 정상 step — 인증 실패 카운터 리셋
                # 잔고 조회 등 거래소 호출이 키 무효로 연속 실패하면 step 이 예외를
                # 안 내도(어댑터가 흡수) 자동 정지. TDAF 류(fetch_balance 10003) 차단.
                _bal_streak = getattr(self.client, "auth_fail_streak", 0)
                if _bal_streak >= _AUTH_FAIL_STOP_THRESHOLD:
                    logger.warning(
                        "%s 거래소 키 무효 %d회(잔고 조회) — 봇 자동 정지. 키 재등록 필요.",
                        self.symbol, _bal_streak,
                    )
                    self.state = BotState.STOPPED
                    break
            except AuthenticationError as e:
                # 키 무효(retCode 10003) — 일시 오류가 아니므로 ERROR/traceback 대신
                # WARNING 으로 강등하고 연속 카운트. 임계치 도달 시 봇 자동 정지로
                # 무한 재시도(로그 폭증·502)를 차단한다. 사용자는 키 재등록 후 재가동.
                self._auth_fail_streak += 1
                logger.warning(
                    "%s 거래소 인증 실패(키 무효 추정) %d/%d — %s",
                    self.symbol, self._auth_fail_streak,
                    _AUTH_FAIL_STOP_THRESHOLD, e,
                )
                if self._auth_fail_streak >= _AUTH_FAIL_STOP_THRESHOLD:
                    logger.warning(
                        "%s 키 무효 %d회 연속 — 봇 자동 정지. 거래소 API 키 재등록 필요.",
                        self.symbol, self._auth_fail_streak,
                    )
                    self.state = BotState.STOPPED
                    break
            except Exception as e:  # noqa: BLE001 — step 실패가 loop 전체를 죽이지 않도록
                logger.exception("step 실패: %s", e)
            # Heartbeat — loop 살아있음 주기적 INFO 로그.
            if self.heartbeat_interval_sec > 0:
                now_ms = int(_time.time() * 1000)
                if now_ms - self._last_heartbeat_ms >= self.heartbeat_interval_sec * 1000:
                    has_pos = self.active_position is not None
                    logger.info(
                        "step heartbeat — symbol=%s last_setup_ts_ms=%d active_pos=%s",
                        self.symbol, self._last_setup_ts_ms, has_pos,
                    )
                    self._last_heartbeat_ms = now_ms
            await asyncio.sleep(self.step_interval_sec)

    async def step(self) -> ICTSignal:
        """단일 step — fetch + HTF/Daily bias + signal + execute. 생성된 signal 반환.

        Returns:
            계산된 ``ICTSignal`` (테스트/디버그 용도로 노출).
        """
        if self.multi_tf:
            return await self._step_multi_tf()

        # #LIVE-1: marketable limit 미체결 추적 — 체결되면 active_position 승격,
        # TTL 만료면 취소. 대기 중이면 신규 진입 안 함 (중복 주문 방지).
        if self._pending_entry is not None:
            still_pending = await self._check_pending_entry()
            if still_pending:
                return ICTSignal(
                    action=SignalAction.NO_ACTION,
                    setup=None,
                    symbol=self.symbol,
                    ts_ms=int(time.time() * 1000),
                    reason="marketable limit 체결 대기",
                )

        # #BUG-7 fix: today 실현손익을 거래소 closed-pnl 로 동기화 (NY 자정 reset 포함).
        # daily loss limit / UI today_pnl 이 거래소와 일치하게 (내부 누적 대체).
        # 잔고 조회 실패(None) 시 reset 보류 — 폴백값 baseline 오염 방지.
        equity_now = await self._fetch_equity_or_none()
        self._maybe_reset_daily_pnl(equity_now)
        await self._sync_today_realized_pnl()

        df = await self._fetch_ohlcv()
        htf_bias = await self._compute_htf_bias()
        daily_bias = await self._compute_daily_bias(df)
        bias = self._combine_with_daily(htf_bias, daily_bias)
        # #BIAS-DIRECTION 2026-06-06: HTF EMA 추세로 진입 방향을 결정한다. 양방향
        # setup 중 추세 방향만 진입 → 상승장 숏/하락장 롱 차단(간밤 19연속 숏 방지).
        # None(EMA 비활성/실패/neutral)이면 방향 강제 안 함(양방향 허용 = 기존 동작).
        ema_dir = await self._compute_htf_ema_direction()

        # 2026-05-28: 봇 의사결정 가시성 — 매 step 1줄 INFO (시장 컨디션 스냅샷).
        # fly.io logs 에서 봇이 뭘 보고 있는지 한 눈에 파악 + 추후 사용자 데이터 분석.
        # 5s polling 기준 1줄 / 5s — 비용 적정 범위.
        self._log_step_market_snapshot(df, bias)

        # generate_ict_signal 은 모든 source 의 indicator(swing/fvg/ob/sweep +
        # Phase B turtle/mitigation/implied/rejection) 를 동기 계산하는 CPU 바운드
        # 핵심이다. 이벤트루프에서 직접 돌리면 그 구간 동안 /ict/health(초경량)
        # 응답까지 밀려 Fly health check timeout → 머신 replacing 을 유발했다(#209).
        # 순수 함수(df 읽기만, self/전역 무변경)라 to_thread 로 빼 이벤트루프를
        # 양보 → health 가 항상 즉답하고 거래 step 간격도 안정된다.
        signal = await asyncio.to_thread(
            generate_ict_signal,
            df,
            self.symbol,
            bias=bias,
            min_rr=self.min_rr,
            fvg_min_size_pct=self.fvg_min_size_pct,
            stale_bars=self.setup_stale_bars,
            require_retrace=self.require_retrace,
            expand_to_killzone=self.expand_to_killzone,
            disable_time_filter=self.disable_time_filter,
            min_sl_distance_pct=self.min_sl_distance_pct,
            prefer_direction=ema_dir,
            ote_level=await self._effective_ote(),  # #REGIME-OTE 상승 국면 0.786
        )

        # 추세 평가 캐시 갱신 (현재는 로깅용, 향후 가중치 확장 여지).
        await self._refresh_trend_cache()

        # 진입 중인 position이 있으면 신규 진입은 막고 상태만 동기화 + trail tick + flip.
        if self.active_position is not None:
            # #BE-LOCK (Origo 1.5): 이익 1R 도달 시 SL 본전 — 분할/flip 보다 먼저
            # (가장 싼 보호 동작, 이후 로직과 독립).
            await self._maybe_be_lock(df)
            # #PARTIAL-TP: TP1(1R) 도달 시 50% 부분익절 + 본전SL — trail 보다 먼저
            # (본전 이동이 trail 기준선을 갱신하므로 순서 중요).
            await self._maybe_partial_exit(df)
            if self.enable_trail:
                await self._tick_trail(df)
            await self._maybe_flip(df)
            await self._sync_position_state()
            return signal

        if not signal.is_actionable or signal.setup is None:
            # #MMBM 2026-07-21: SB 셋업 없을 때 2번째 모델(마켓메이커 반전) 시도.
            # 여기 도달 = active_position None(위에서 return) + SB 무셋업. MMBM 은 자체
            # 조건(HTF정합·discount/premium·신선 CHoCH+FVG)으로 검증돼 SB 게이트는
            # 우회하되, _execute_setup 의 리스크레이어(#SAFETY-1 일일한도·서킷브레이커·
            # 사이징·DD스로틀·maker 지정가)는 공유. recovery_failed 게이트가 아래 SB
            # 경로에만 있어 여기서 직접 확인. _pending_entry 있으면 중복진입 방지 skip.
            if (
                self.mmbm_enabled
                and not self._recovery_failed
                and self._pending_entry is None
            ):
                mmbm_setup = detect_mmbm_setup(
                    df,
                    self._mmbm_htf_bias_sign(df),
                    min_rr=self.min_rr,
                    fvg_min_size_pct=self.fvg_min_size_pct,
                )
                dup = mmbm_setup is not None and (
                    mmbm_setup.ts_ms == self._last_setup_ts_ms
                    and mmbm_setup.direction == self._last_setup_direction
                )
                if mmbm_setup is not None and not dup:
                    logger.info(
                        "MMBM setup | dir=%s entry=%.4f sl=%.4f tp=%.4f rr=%.2f",
                        mmbm_setup.direction.value, mmbm_setup.entry,
                        mmbm_setup.stop_loss, mmbm_setup.take_profit,
                        mmbm_setup.risk_reward,
                    )
                    await self._execute_setup(mmbm_setup)
                    self._remember_setup(mmbm_setup)
                    return signal
            # 2026-05-28: setup 미발견 / 조건 불충족 — signal.reason 에 사유 박혀있음.
            # reason 비면 "no setup" 으로 통일. 너무 빈도 높지 않게 — 매 step 1줄 (다른 분기).
            logger.info(
                "setup skip | tf=%s reason=%s",
                self.timeframe, signal.reason or "no setup",
            )
            return signal

        # setup 발견 — 핵심 필드 1줄 INFO (fly logs 에서 후속 게이트 결과와 연결 추적용).
        logger.info(
            "setup found | dir=%s entry=%.4f sl=%.4f tp=%.4f rr=%.2f score=%d source=%s window=%s",
            signal.setup.direction.value, signal.setup.entry,
            signal.setup.stop_loss, signal.setup.take_profit,
            signal.setup.risk_reward, signal.setup.confluence_score,
            signal.setup.source.value if hasattr(signal.setup.source, "value")
            else str(signal.setup.source),
            signal.setup.window,
        )

        # #NYPM-GATE 2026-07-16 (FST#5): NY_PM(NY 13:30-16:00 = 02-05 KST) 진입 차단.
        # 위 killzone 게이트와 달리 disable_time_filter(24h)·구독 두 티어 모두 적용.
        # NY_PM 은 정통 ICT reversal 구간 — 추세추종 Origo 는 여기서 삼중검증 음수
        # (라이브 승률 10%/-29·5년 백테 7/7 페어 음수·6/24 킬존연구 최악). 진입 '시점'
        # 기준(진입이 NY_PM 에 걸릴 때만 차단, 셋업 생성 시점 무관).
        if self.exclude_nypm:
            last_ts_ms = int(df.index[-1].value // 10**6)
            if classify_killzone(last_ts_ms) is KillzoneName.PM:
                logger.info(
                    "진입 skip — NY_PM 차단 (#NYPM-GATE, setup window=%s)",
                    signal.setup.window,
                )
                self._record_shadow(signal.setup, "nypm_skip")
                self._remember_setup(signal.setup)
                return signal

        # #KZ-ENTRY 2026-06-06 (파트너 신고): 진입 '시점' 킬존 게이트 — sub_*
        # (disable_time_filter=False) 한정. silver_bullet 의 시간 필터는 FVG '생성
        # 시점'(fvg.ts_ms)만 검사하므로, 킬존에 형성된 셋업이 retrace 로 한참 뒤
        # '진입 시점'엔 킬존 밖일 수 있다 (13:16 KST=미장 밖 진입 발견). 진입 직전
        # 현재(마지막 닫힌) 봉도 in_trade_window_sub 통과하는지 재확인.
        # referral(24h)·disable_time_filter=True 사용자는 영향 없음.
        if not self.disable_time_filter:
            last_ts_ms = int(df.index[-1].value // 10**6)
            if not in_trade_window_sub(last_ts_ms):
                logger.info(
                    "진입 skip — 현재 봉 킬존 밖 (sub_* 시간 필터, setup window=%s)",
                    signal.setup.window,
                )
                self._record_shadow(signal.setup, "killzone_skip")
                self._remember_setup(signal.setup)
                return signal

        # 2026-05-29 #SILENT-1: 복원 실패 상태에서는 신규 진입 차단.
        # 거래소 측 활성 포지션이 있는데 봇이 인식 못 하면 중복 진입 위험.
        # _sync_position_state 가 성공하면 _recovery_failed 가 False 로 풀린다.
        if self._recovery_failed:
            logger.warning(
                "setup skip — 거래소 측 포지션 복원 실패 상태. sync 성공 전까지 신규 진입 차단.",
            )
            return signal

        # 동일 setup으로 재진입 방지 (중복 주문 X)
        if (
            signal.setup.ts_ms == self._last_setup_ts_ms
            and signal.setup.direction == self._last_setup_direction
        ):
            # 2026-06-09: ts + 방향 동일할 때만 중복 skip. 같은 봉의 반대 방향
            # (롱 청산 직후 숏 진입)은 통과시킨다 — 기존엔 ts 만 봐서 숏 누락.
            logger.info(
                "setup skip | reason=duplicate_ts ts_ms=%d dir=%s",
                signal.setup.ts_ms, signal.setup.direction.value,
            )
            return signal

        if not await self._passes_htf_ema_bias(signal.setup.direction):
            # _passes_htf_ema_bias 내부에서 이미 로그 박힘 — 추가 X.
            # 2026-06-11 리뷰 수정(critical): 여기서 _remember_setup 을 하면
            # "추세 불명확/역방향"이라는 *일시적* 시장 상태 때문에 그 setup 이
            # 영구 블랙리스트(duplicate_ts)에 박혀, 몇 분 뒤 추세가 확정돼도
            # 같은 setup 으론 영영 진입 불가였다. 동적 게이트는 setup 을
            # 기억하지 않고 다음 step 재평가에 맡긴다.
            self._record_shadow(signal.setup, "ema_gate_skip")
            return signal

        # 변경 3: HTF FVG override — 진입 직전 반대 방향 HTF FVG 가중치 평가 (flip target).
        htf_target = await self._evaluate_htf_override(signal.setup, df)
        # 변형 7 B+A 합성 (A): 같은 방향 HTF FVG → confluence_score 보강.
        # qty 산정 (_calc_qty) 에서 confluence_score 가 사용되므로 boost 가 _execute_setup
        # 이전에 적용되어야 효과 발생. _evaluate_htf_override 직후 호출.
        await self._apply_htf_supporting_boost(signal.setup, df)
        # #3 보완: Draw on Liquidity 역방향 진입은 confluence 감점 (게이트 전에 적용).
        self._apply_dol_bias(signal.setup, df)
        # #CISD 2026-06-06: 가격 전달 전환(CISD)이 setup 방향과 일치하면 confluence +1.
        # MSS 1캔들 micro 신호 — 게이트·qty 산정 전에 적용돼야 효과.
        self._apply_cisd_boost(signal.setup, df)
        # #PO3 2026-06-17: AMD Distribution 국면 진입이면 +1 (cisd+po3 5년 robust 흑자).
        self._apply_po3_boost(signal.setup, df)
        # #SMT 2026-06-06: 상관 자산(BTC↔ETH) divergence 가 setup 방향과 일치하면 +1.
        # 게이트·qty 산정 전에 적용. corr 심볼 OHLCV fetch 필요해 async.
        await self._apply_smt_boost(signal.setup, df)
        # #OTE-FIB 2026-06-18: 직전 임펄스 swing leg 의 피보나치 0.618~0.786(ICT OTE)
        # 되돌림 진입이면 confluence +1. 게이트·qty 전에 적용 (net +0.5%p·거래+12).
        self._apply_ote_boost(signal.setup, df)
        # #CT-SL 2026-06-18: 진입 직전 20봉 방향정합 추세 기록 → _execute_setup 이
        # 역추세면 SL 배수를 sl_dist_mult_ct(x4)로 전환 (confluence·게이트엔 영향 0).
        self._set_entry_trend(signal.setup, df)
        self._set_smart_size(signal.setup, df)  # #SMART-SIZE 품질 기반 자금배분
        # #REGIME 2026-06-23: 횡보 국면 회피 게이트 — 진입 직전 추세(|entry_trend_pct|)가
        # 페어별 floor(q33) 미만이면 진입 skip. 안정형 하이브리드 연구: 횡보 국면은 모든
        # TP 가 적자라 "대처 아닌 회피"가 답(회피가 net 흑자의 필수조건). _set_entry_trend
        # 가 기록한 추세를 재사용(추가 계산 0). 미등록 페어는 floor=0 → 게이트 off.
        if self.regime_filter_enabled:
            floor = self._regime_floor()  # 롤링 33분위 or q33 하드코딩 fallback
            cur = abs(signal.setup.entry_trend_pct)
            self._trend_history.append(cur)  # 다음 분위 계산용 누적(현재 판단은 이전 이력 기준)
            if floor > 0 and cur < floor:
                logger.info(
                    "횡보 국면 skip — |trend|=%.3f < floor=%.3f (%s %s)",
                    cur, floor,
                    signal.setup.direction.value, signal.setup.window,
                )
                self._record_shadow(signal.setup, "regime_skip")
                self._remember_setup(signal.setup)
                return signal

        # #COND-ALIGN 2026-07-17 (Origo 2.0, FST#6): 조건부 방향정합 게이트.
        # 약/중추세(|trend| < 강추세 floor q70)에선 진입이 20봉 추세와 정합(같은
        # 방향)일 때만 허용. 강추세(|trend| >= q70)면 반전(터틀수프)도 허용 — 진짜
        # 추세선 역추세 진입이 흑자(+1.5)라 살림. 근거: 극톱질서 역추세 진입만 -1.0,
        # cond_align 이 5년 net +17.7→+21.3(라이브게이트 위 +15.8→+19.0), walk-forward
        # 양반기 robust, 거래 ~22%↓. regime_filter(q33 크기) 와 직교(방향 축).
        if self.cond_align_enabled:
            sign = 1.0 if signal.setup.direction is Direction.LONG else -1.0
            signed_trend = signal.setup.entry_trend_pct * sign
            strong = self._strong_trend_floor()  # 롤링 q70 or 하드코딩 fallback
            mag = abs(signal.setup.entry_trend_pct)
            # signed<0 만 역추세로 차단. signed==0(무추세/데이터부족)은 순추세 취급
            # (기존 #CT-SL 과 일관) → 통과. 연속 백테선 measure-0 이라 결과 불변.
            if strong > 0 and mag < strong and signed_trend < 0:
                logger.info(
                    "역추세 촙 skip — |trend|=%.3f<강추세%.3f & 역추세 (%s %s)",
                    mag, strong,
                    signal.setup.direction.value, signal.setup.window,
                )
                self._record_shadow(signal.setup, "cond_align_skip")
                self._remember_setup(signal.setup)
                return signal
        # B+ 등급 게이트 (#1/#8) — HTF boost 까지 반영된 최종 score 가 기준 미만이면 skip.
        # 빈도↓·품질↑ (하루 ~4~5개 목표). min_confluence=0 이면 비활성(기존 동작).
        if signal.setup.confluence_score < self.min_confluence:
            # 고RR 예외 구멍 — confluence 미달이어도 손익비가 충분히 높고
            # (rr >= high_rr_bypass_min_rr) score>=1 이면 단일신호 셋업도 통과.
            # (파트너 결정 2026-06-04: rr 2.5+ 1점 셋업 진입 허용)
            high_rr_pass = (
                self.high_rr_bypass_min_rr > 0
                and signal.setup.confluence_score >= 1
                and signal.setup.risk_reward >= self.high_rr_bypass_min_rr
            )
            if high_rr_pass:
                logger.info(
                    "고RR 예외 통과 — score=%d rr=%.2f >= %.2f (%s %s)",
                    signal.setup.confluence_score, signal.setup.risk_reward,
                    self.high_rr_bypass_min_rr,
                    signal.setup.direction.value, signal.setup.window,
                )
            else:
                logger.info(
                    "등급 미달 skip — score=%d < min_confluence=%d (%s %s)",
                    signal.setup.confluence_score, self.min_confluence,
                    signal.setup.direction.value, signal.setup.window,
                )
                self._record_shadow(signal.setup, "grade_skip")
                self._remember_setup(signal.setup)
                return signal
        if self.htf_override_mode == "A" and htf_target is not None:
            logger.info(
                "HTF override(A) 진입 차단 — setup=%s 반대 HTF FVG=%s",
                signal.setup.direction.value, htf_target.tf,
            )
            self._record_shadow(signal.setup, "override_block")
            self._remember_setup(signal.setup)
            return signal

        # 2026-05-29 #HTF-LTF-CONFLICT: HTF FVG bull/bear 우세 + LTF 반대 방향 차단.
        # 5-29 #3/#5 (bull weight 우세인데 LTF turtle_soup short 진입 → SL_HIT)
        # 패턴 회고 기반. ratio 임계 이상의 명확 우세에서만 차단 (정상 reversal 신호
        # 까지 막지 않게 보수적). 0 면 비활성.
        if self.htf_ltf_conflict_guard_ratio > 0:
            htf_map = self._htf_fvg_map_cache or []
            bull_w = sum(
                int(e.weight) for e in htf_map
                if getattr(e.type, "value", "") == "bullish"
            )
            bear_w = sum(
                int(e.weight) for e in htf_map
                if getattr(e.type, "value", "") == "bearish"
            )
            if bull_w > 0 and bear_w > 0:
                is_long = signal.setup.direction is Direction.LONG
                ratio_thr = self.htf_ltf_conflict_guard_ratio
                if is_long and (bear_w / bull_w) >= ratio_thr:
                    logger.info(
                        "HTF/LTF 방향 충돌 skip — bear_w=%d > bull_w=%d * %.2f, "
                        "LTF setup=long",
                        bear_w, bull_w, ratio_thr,
                    )
                    self._remember_setup(signal.setup)
                    return signal
                if (not is_long) and (bull_w / bear_w) >= ratio_thr:
                    logger.info(
                        "HTF/LTF 방향 충돌 skip — bull_w=%d > bear_w=%d * %.2f, "
                        "LTF setup=short",
                        bull_w, bear_w, ratio_thr,
                    )
                    self._remember_setup(signal.setup)
                    return signal

        # #SWEEP-GATE (Origo 1.5, 파트너 "3번 판단" 기계화 2026-07-07): 일봉
        # 스윕-반전 후 K일 역방향 차단 — EMA align 지연(7/4 숏 전멸 -165) 방어.
        if await self._sweep_gate_blocked(signal.setup.direction):
            logger.info(
                "스윕-반전 게이트 skip — %s 진입 차단 (최근 %d일 내 반대 스윕-반전)",
                signal.setup.direction.value, self.sweep_gate_days,
            )
            self._record_shadow(signal.setup, "sweep_gate_skip")
            self._remember_setup(signal.setup)
            return signal

        self._record_shadow(signal.setup, "taken")
        await self._execute_setup(signal.setup, htf_flip_target=htf_target)
        self._remember_setup(signal.setup)
        return signal

    def _record_shadow(self, setup: Any, verdict: str) -> None:
        """#SHADOW: 게이트 판정된 setup 을 특징과 함께 JSONL 기록 (행동 영향 0).

        FSD-style 데이터 플라이휠 — 진입한 것뿐 아니라 *거른* 자리(등급 미달/
        align 역방향/킬존 밖/override 차단)도 기록해, 사후 가격과 대조하면
        "거른 게 맞았나"를 라벨링할 수 있는 학습 데이터가 된다.

        같은 (setup ts, 방향, 판정)은 1회만 기록 (step 반복 노이즈 방지).
        실패는 조용히 무시 — 기록이 매매를 막지 않게.

        Args:
            setup: SilverBulletSetup.
            verdict: "taken" / "grade_skip" / "ema_gate_skip" / "killzone_skip"
                / "override_block".
        """
        if not self.shadow_log_enabled:
            return
        try:
            key = (setup.ts_ms, setup.direction.value, verdict)
            if key in self._shadow_seen:
                return
            if len(self._shadow_seen) > 2000:  # 메모리 캡 — 오래된 것 비움
                self._shadow_seen.clear()
            self._shadow_seen[key] = True
            entry = float(setup.entry)
            risk = abs(entry - float(setup.stop_loss))
            rec = {
                "ts_ms": int(time.time() * 1000),
                "symbol": self.symbol,
                "tf": self.timeframe,
                "setup_ts_ms": int(setup.ts_ms),
                "direction": setup.direction.value,
                "verdict": verdict,
                "score": int(setup.confluence_score),
                "rr": round(float(setup.risk_reward), 3),
                "window": setup.window,
                "source": getattr(setup.source, "value", str(setup.source)),
                "entry": entry,
                "stop_loss": float(setup.stop_loss),
                "take_profit": float(setup.take_profit),
                "sl_dist_pct": round(risk / entry * 100, 4) if entry > 0 else None,
                "align_score": self._last_align_score,
                "confluences": list(getattr(setup, "confluences", []) or []),
                # #REGIME-LEARN 2026-06-23: 미진입 setup 까지 진입추세 기록 → 라이브
                # "후보 전체"(진입+횡보skip+등급skip) 분포 학습 모집단 완성. 횡보 임계
                # 정확화·롤링 분위 seed 용(진입거래만 기록하면 분포가 높게 편향됨).
                "entry_trend_pct": round(float(getattr(setup, "entry_trend_pct", 0.0) or 0.0), 4),
            }
            store_dir = Path(self.trades_data_dir or _ict_data_dir())
            store_dir.mkdir(parents=True, exist_ok=True)
            with (store_dir / "shadow_setups.jsonl").open(
                "a", encoding="utf-8",
            ) as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.debug("shadow 기록 실패(무시): %s", e)

    def _remember_setup(self, setup: Any) -> None:
        """동일 setup 재진입 방지 기록 — ts + 방향.

        2026-06-09: 기존엔 ts_ms 만 기록해, 롱 청산 직후 같은 봉에서 나온 숏
        셋업이 duplicate_ts 로 차단돼 진입 누락(라이브 HYPE 버그). 방향까지
        기록해 같은 봉의 반대 방향(롱→숏 전환) 셋업은 차단하지 않는다.
        """
        self._last_setup_ts_ms = setup.ts_ms
        self._last_setup_direction = setup.direction

    async def _step_multi_tf(self) -> ICTSignal:
        """Multi-TF step — HTF tracker + LTF confirmer 결합 (ICT 정통).

        시퀀스:
        1. LTF (self.timeframe) OHLCV fetch
        2. HtfSetupTracker lazy init + 각 HTF fetch → tracker.update_htf
        3. 현재 가격으로 SL/TP hit setup 제거
        4. 가격이 zone 안인 HTF setup 별로 LtfEntryConfirmer 실행
        5. ConfirmedEntry 박힘 + 신규 setup 박힘 + 포지션 없음 → 진입
        """
        if self._htf_tracker is None:
            self._htf_tracker = HtfSetupTracker(
                trade_tf=self.timeframe,
                min_rr=self.min_rr,
                fvg_min_size_pct=self.fvg_min_size_pct,
            )

        ltf_df = await self._fetch_ohlcv()
        last_ts_ms = (
            int(ltf_df.index[-1].value // 10**6) if len(ltf_df) > 0 else 0
        )
        no_action = ICTSignal(
            action=SignalAction.NO_ACTION,
            setup=None,
            symbol=self.symbol,
            ts_ms=last_ts_ms,
            reason="multi_tf",
        )

        if len(ltf_df) < 5:
            return no_action

        current_price = float(ltf_df["close"].iloc[-1])

        # 2026-05-28: multi_tf 경로도 의사결정 스냅샷 1줄 (bias=None — HTF tracker 별도).
        self._log_step_market_snapshot(ltf_df, bias=None)

        # HTF tracker 갱신 — 각 HTF 별 fetch + 최신 setup 감지.
        for htf_tf in self._htf_tracker.htf_list():
            try:
                htf_df = await self._fetch_ohlcv_tf(htf_tf, 200)
            except Exception as e:  # noqa: BLE001
                logger.warning("HTF fetch 실패 (%s): %s", htf_tf, e)
                continue
            self._htf_tracker.update_htf(htf_tf, htf_df)

        # SL/TP hit setup 제거.
        self._htf_tracker.invalidate_if_sl_hit(current_price)
        self._htf_tracker.invalidate_if_tp_hit(current_price)

        # 진입 중이면 sync + trail tick.
        if self.active_position is not None:
            if self.enable_trail:
                await self._tick_trail(ltf_df)
            await self._sync_position_state()
            return no_action

        # 가격이 zone 안인 HTF setup 별로 confirm 시도.
        matching = self._htf_tracker.setups_containing_price(current_price)
        for htf_active in matching:
            # 동일 HTF setup 으로 재진입 방지 (ts + 방향 — 반대 방향은 허용).
            if (
                htf_active.setup.ts_ms == self._last_setup_ts_ms
                and htf_active.setup.direction == self._last_setup_direction
            ):
                continue
            confirmed = confirm_ltf_entry(
                htf_active,
                ltf_df,
                lookback_bars=self.multi_tf_ltf_lookback,
                fvg_min_size_pct=self.fvg_min_size_pct,
            )
            if confirmed is None:
                continue
            setup = self._confirmed_to_setup(confirmed)
            if not await self._passes_htf_ema_bias(setup.direction):
                # 2026-06-11 리뷰 수정: 일시적 게이트 불일치로 수명 긴 HTF setup 을
                # 영구 차단하지 않는다 (다음 step 재평가).
                return no_action
            await self._execute_setup(setup)
            self._remember_setup(htf_active.setup)
            return ICTSignal(
                action=(
                    SignalAction.ENTER_LONG
                    if confirmed.direction is Direction.LONG
                    else SignalAction.ENTER_SHORT
                ),
                setup=setup,
                symbol=self.symbol,
                ts_ms=last_ts_ms,
                reason=f"multi_tf:{htf_active.htf_tf}",
            )

        return no_action

    @staticmethod
    def _confirmed_to_setup(confirmed: ConfirmedEntry) -> SilverBulletSetup:
        """ConfirmedEntry → SilverBulletSetup 변환 (execute_setup 재사용).

        confluence_score=2 박혀있어서 sizing 70% (base 40 + step 15 × 2).
        """
        risk = abs(confirmed.entry - confirmed.stop_loss)
        reward = abs(confirmed.take_profit - confirmed.entry)
        rr = (reward / risk) if risk > 0 else 0.0
        return SilverBulletSetup(
            ts_ms=confirmed.htf_setup_ts_ms,
            direction=confirmed.direction,
            window=f"multi_tf:{confirmed.htf_tf}",
            entry=confirmed.entry,
            stop_loss=confirmed.stop_loss,
            take_profit=confirmed.take_profit,
            risk_reward=rr,
            fvg=confirmed.ltf_fvg,
            confluence_score=2,
            confluences=[f"htf={confirmed.htf_tf}"],
            reasons=[
                f"htf_tf={confirmed.htf_tf}",
                f"htf_setup_ts={confirmed.htf_setup_ts_ms}",
                "multi_tf_confirmed",
            ],
        )

    async def _fetch_ohlcv(self) -> pd.DataFrame:
        """OHLCV fetch + DataFrame 변환 (Trade TF)."""
        return await self._fetch_ohlcv_tf(self.timeframe, self.ohlcv_limit)

    async def _fetch_ohlcv_tf(
        self, tf: str, limit: int, *, symbol: str | None = None,
    ) -> pd.DataFrame:
        """임의 timeframe OHLCV fetch + DataFrame 변환.

        symbol 미지정 시 self.symbol. SMT 등 상관 심볼 fetch 용으로 symbol 주입 가능.
        """
        rows = await self.client.fetch_ohlcv(symbol or self.symbol, tf, limit)
        df = pd.DataFrame(
            rows,
            columns=["ts_ms", "open", "high", "low", "close", "volume"],
        )
        df.index = pd.DatetimeIndex(pd.to_datetime(df["ts_ms"], unit="ms", utc=True))
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    async def _compute_htf_ema_direction(self) -> Direction | None:
        """HTF EMA bias 방향 — bullish→LONG, bearish→SHORT, neutral/비활성/실패→None.

        #BIAS-DIRECTION: 진입 방향 결정용(generate_ict_signal prefer_direction).
        양방향 setup 중 이 방향만 진입시켜 상승장 숏/하락장 롱을 차단한다. None 이면
        방향을 강제하지 않아(양방향 허용) 기존 동작과 같다.
        """
        if not self.htf_ema_bias_enabled:
            return None
        # 2026-06-10 #ALIGN: 다중 EMA 정렬 점수로 방향 결정 (검증됨 — 단일 EMA20
        # 보다 방향 정확도↑). 점수 |s|>=threshold 면 그 방향, 미만이면 None(불명확
        # → 양방향 setup 허용하되 _passes_ema_align_gate 가 진입 자제).
        if self.htf_ema_align_enabled:
            score = await self._compute_ema_align_score()
            if score is None:
                return None
            t = max(1, int(self.htf_ema_align_threshold))
            if score >= t:
                return Direction.LONG
            if score <= -t:
                return Direction.SHORT
            return None
        period = max(2, int(self.htf_ema_bias_period))
        try:
            df = await self._fetch_ohlcv_tf(self.htf_ema_bias_tf, period + 30)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "HTF EMA direction fetch 실패 (tf=%s): %s — 방향 강제 안 함",
                self.htf_ema_bias_tf, e,
            )
            return None
        if len(df) < period + 1:
            return None
        closes = df["close"].astype(float).to_numpy()
        k = 2.0 / (period + 1)
        ema = float(closes[:period].mean())  # SMA 시드
        for px in closes[period:]:
            ema = float(px) * k + ema * (1.0 - k)
        last_close = float(closes[-1])
        if last_close > ema:
            return Direction.LONG
        if last_close < ema:
            return Direction.SHORT
        return None

    async def _passes_htf_ema_bias(self, direction: Direction) -> bool:
        """HTF EMA bias 필터 — setup 방향이 EMA bias 와 일치할 때만 True.

        htf_ema_bias_enabled=False 면 항상 True (필터 비활성).
        OHLCV fetch / EMA 계산 실패 시 안전하게 진입 허용 (True 반환).
        """
        if not self.htf_ema_bias_enabled:
            return True
        # 2026-06-10 #ALIGN: 다중 EMA 정렬 게이트 — override 와 독립 작동(진입 단계
        # 방향 필터). override="C"(진입 후 flip)와 공존. 추세 불명확 시 진입 자제 →
        # prefer_direction=None(양방향 setup)과 짝지어 백테스트 align 동작과 일치.
        if self.htf_ema_align_enabled:
            return await self._passes_ema_align_gate(direction)
        # 변경 4: override 모드 활성이면 단일 EMA bias 는 의미 없음 (override 가 강력).
        if self.htf_override_mode != "off":
            return True
        period = max(2, int(self.htf_ema_bias_period))
        try:
            df = await self._fetch_ohlcv_tf(self.htf_ema_bias_tf, period + 30)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "HTF EMA bias fetch 실패 (tf=%s): %s — bias filter skip",
                self.htf_ema_bias_tf, e,
            )
            return True
        if len(df) < period + 1:
            return True
        closes = df["close"].astype(float).to_numpy()
        k = 2.0 / (period + 1)
        ema = float(closes[:period].mean())  # SMA 시드
        for px in closes[period:]:
            ema = float(px) * k + ema * (1.0 - k)
        last_close = float(closes[-1])
        if last_close > ema:
            bias = "bullish"
        elif last_close < ema:
            bias = "bearish"
        else:
            bias = "neutral"
        # 2026-05-28: HTF EMA bias 전환 시에만 1줄 INFO (예: bullish→bearish 추세 뒤집힘).
        # 매 step 박으면 너무 많음. 변화 감지형.
        if bias != self._last_logged_htf_ema_bias:
            if self._last_logged_htf_ema_bias:  # 빈 초기값은 무시
                logger.info(
                    "HTF EMA bias 변화 | %s → %s (tf=%s ema%d=%.4f close=%.4f)",
                    self._last_logged_htf_ema_bias, bias,
                    self.htf_ema_bias_tf, period, ema, last_close,
                )
            self._last_logged_htf_ema_bias = bias
        want_long = direction is Direction.LONG
        if bias == "bullish" and want_long:
            return True
        if bias == "bearish" and not want_long:
            return True
        if bias == "neutral":
            return True  # 정확히 동일 가격이면 통과
        logger.info(
            "HTF bias 역방향 setup skip — bias=%s setup=%s (tf=%s ema%d=%.4f close=%.4f)",
            bias, "buy" if want_long else "sell",
            self.htf_ema_bias_tf, period, ema, last_close,
        )
        return False

    @staticmethod
    def _ema_last(closes: Any, period: int) -> float:
        """closes 배열의 EMA 마지막값 — SMA 시드 후 재귀 (단일 EMA 게이트와 동일식)."""
        k = 2.0 / (period + 1)
        ema = float(closes[:period].mean())
        for px in closes[period:]:
            ema = float(px) * k + ema * (1.0 - k)
        return ema

    async def _compute_ema_align_score(self) -> int | None:
        """다중 EMA 정렬 점수 (#ALIGN). 방향 결정·게이트가 공유.

        htf_ema_bias_tf(기본 1h) 봉으로 periods 각 EMA 계산 → 인접 쌍(짧은→긴)이
        정배열(짧은>긴)이면 +1, 역배열이면 -1 누적. 범위 -(N-1)~+(N-1).
        강한 상승추세면 +극단, 하락추세면 -극단, 전환/횡보면 0 근처.

        Returns:
            점수 int, 또는 fetch/미성숙 실패 시 None(게이트 skip).
        """
        periods = self.htf_ema_align_periods
        if not periods or len(periods) < 2:
            return None
        pmax = max(periods)
        try:
            df = await self._fetch_ohlcv_tf(self.htf_ema_bias_tf, pmax + 50)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "HTF EMA align fetch 실패 (tf=%s): %s — align gate skip",
                self.htf_ema_bias_tf, e,
            )
            return None
        if len(df) < pmax + 1:
            return None
        closes = df["close"].astype(float).to_numpy()
        emas = [self._ema_last(closes, max(2, int(p))) for p in periods]
        score = 0
        for a, b in zip(emas[:-1], emas[1:], strict=False):
            if a > b:
                score += 1
            elif a < b:
                score -= 1
        self._last_align_score = score  # #SHADOW 특징 기록용 캐시
        # 추세 라벨 변화 시에만 1줄 INFO (단일 EMA bias 로그와 통일).
        bias = "bullish" if score >= 1 else ("bearish" if score <= -1 else "neutral")
        if bias != self._last_logged_htf_ema_bias:
            if self._last_logged_htf_ema_bias:
                logger.info(
                    "HTF EMA align 변화 | %s → %s (score=%d, tf=%s periods=%s)",
                    self._last_logged_htf_ema_bias, bias, score,
                    self.htf_ema_bias_tf, periods,
                )
            self._last_logged_htf_ema_bias = bias
        return score

    async def _passes_ema_align_gate(self, direction: Direction) -> bool:
        """다중 EMA 정렬 방향 게이트 — |점수|>=threshold 면 그 방향만 허용,
        미만이면 추세 불명확으로 진입 자제(False). 계산 실패 시 안전하게 True.
        """
        score = await self._compute_ema_align_score()
        if score is None:
            return True
        t = max(1, int(self.htf_ema_align_threshold))
        is_long = direction is Direction.LONG
        if score >= t:
            ok = is_long       # 상승추세 정렬 → 롱만
        elif score <= -t:
            ok = not is_long   # 하락추세 정렬 → 숏만
        else:
            ok = False         # 추세 불명확 → 진입 자제
        if not ok:
            # 같은 (score, 방향) skip 은 1회만 INFO — 블랙리스트 제거 후 매 step
            # 재평가되므로 변화 감지형으로 스팸 방지. 상태 바뀌면 다시 INFO.
            key = f"{score}|{t}|{is_long}"
            if key != self._last_align_skip_log:
                self._last_align_skip_log = key
                logger.info(
                    "HTF align 게이트 skip — score=%d(T=%d) setup=%s (tf=%s)",
                    score, t, "buy" if is_long else "sell", self.htf_ema_bias_tf,
                )
        else:
            self._last_align_skip_log = ""
        return ok

    async def _compute_htf_bias(self) -> TrendDirection | None:
        """Trade TF 의 상위 HTF1/HTF2 봉을 fetch 해 bias 산출.

        HTF 봉은 LTF 보다 훨씬 느리게 갱신되므로 마지막 봉 ts 기준 캐시 사용.

        Returns:
            - ``TrendDirection`` (UP/DOWN/NONE) — HTF 매핑이 있을 때.
            - ``None`` — Trade TF 가 HTF 매핑이 없을 때 (1m, 3m 등 또는 1w).
              이 경우 silver_bullet 이 같은 df 의 structure 로 자동 추정.
        """
        htf1_tf, htf2_tf = htf_pair(self.timeframe)
        if htf1_tf is None and htf2_tf is None:
            return None  # 매핑 없음 — 자동 추정 위임
        bias1 = await self._htf_bias_for_tf(htf1_tf) if htf1_tf else TrendDirection.NONE
        bias2 = await self._htf_bias_for_tf(htf2_tf) if htf2_tf else TrendDirection.NONE
        return combine_bias(bias1, bias2)

    async def _compute_daily_bias(self, ltf_df: pd.DataFrame) -> TrendDirection:
        """Daily 봉 fetch + 전일 H/L vs 현재가 비교로 daily bias 산출.

        Args:
            ltf_df: Trade TF DataFrame (마지막 close 가 현재가).

        Returns:
            - current > PDH → UP
            - current < PDL → DOWN
            - 범위 안 또는 일봉 부족 → NONE
        """
        if len(ltf_df) == 0:
            return TrendDirection.NONE
        # 일봉 20개로 PDH/PDL + 지난 주 월요일 (최대 13일 전) 까지 안전 커버.
        daily_df = await self._fetch_ohlcv_tf("1d", 20)
        if len(daily_df) < 2:
            return TrendDirection.NONE
        current_close = float(ltf_df["close"].iloc[-1])
        return compute_daily_bias(daily_df, current_close)

    @staticmethod
    def _combine_with_daily(
        htf: TrendDirection | None,
        daily: TrendDirection,
    ) -> TrendDirection | None:
        """HTF bias + Daily bias 결합.

        - HTF None (매핑 없음) → silver_bullet 자동 추정 위임 (None 그대로)
        - 둘 다 같은 방향 → 그 방향
        - 한쪽 NONE → 다른쪽 따름
        - 충돌 → daily 따름 (Daily bias 가 더 무거운 신호라 우선)
        """
        if htf is None:
            return None
        if htf is TrendDirection.NONE:
            return daily
        if daily is TrendDirection.NONE:
            return htf
        if htf == daily:
            return htf
        return daily

    async def _htf_bias_for_tf(self, tf: str) -> TrendDirection:
        """단일 HTF bias — 캐시된 봉 재사용, 새 봉 생겼을 때만 재fetch."""
        # 마지막 봉 ts 만 빠르게 보기 위해 limit=1 prefetch — 캐시 hit 시 skip
        # 단순화: 매 step 마다 limit=200 fetch (HTF 1d 라도 1m 1번이라 부담 작음)
        df = await self._fetch_ohlcv_tf(tf, 200)
        if len(df) == 0:
            return TrendDirection.NONE
        last_ts = int(df.index[-1].value // 10**6)
        cached = self._htf_cache.get(tf)
        if cached is not None and cached[0] == last_ts:
            df = cached[1]
        else:
            self._htf_cache[tf] = (last_ts, df)
        return compute_bias_from_df(df)

    def _mmbm_htf_bias_sign(self, df: pd.DataFrame, lookback: int = 20) -> float:
        """MMBM 용 상위TF(1h) 추세 부호 (+상승/-하락/0중립·데이터부족).

        Trade TF(5m) df 를 1h 로 리샘플해 최근 lookback(20)봉 종가 변화의 부호.
        MMBM 은 HTF 추세 정합이 엣지 필수조건(백테 검증 — 역정합이면 적자)이라
        이 부호로 진입 방향을 게이트한다.

        Args:
            df: Trade TF OHLCV (index=DatetimeIndex).
            lookback: 1h 봉 기준 추세 측정 구간 (기본 20 = 20시간).

        Returns:
            +1.0(상승) / -1.0(하락) / 0.0(중립·데이터부족).
        """
        try:
            h1 = df["close"].resample("1h").last().dropna()
        except (TypeError, ValueError):
            return 0.0
        if len(h1) < lookback + 1:
            return 0.0
        delta = float(h1.iloc[-1] - h1.iloc[-1 - lookback])
        if delta > 0:
            return 1.0
        if delta < 0:
            return -1.0
        return 0.0

    async def _execute_setup(
        self,
        setup: SilverBulletSetup,
        htf_flip_target: HtfFvgEntry | None = None,
        force_qty: float | None = None,
    ) -> None:
        """setup 한 건을 실제 주문으로 실행 (#LIVE-1 fix: marketable limit + SL/TP 동봉).

        - entry = 현재가 바로 앞 marketable limit (슬리피지 0). SL/TP 를 entry 주문에
          동봉 → 체결 시 거래소가 포지션에 conditional 적용 (단일 TP, ICT 정통).
        - 즉시 체결되면 active_position 확정. 미체결이면 ``_pending_entry`` 등록 →
          step 의 ``_check_pending_entry`` 가 체결 승격 / TTL(10분) 만료 취소 추적.
        - use_market_entry=True 면 레거시 즉시 시장가 (slippage 발생, 비권장).
        """
        side = "buy" if setup.direction is Direction.LONG else "sell"

        # #MANUAL-POS-RESPECT 2026-07-22 (리뷰 HIGH): 봇이 flat(active_position None)
        # 인데 거래소에 포지션이 있으면 그건 유저 수동 포지션(또는 미채택 고아)이다.
        # 그 위에 얹거나(one-way 모드 합산) flip 되지 않게 신규 진입을 차단한다 —
        # 과거엔 미채택 포지션을 밟고 진입해 reconcile 이 합산·SL/TP 부착(수동포지션
        # 침해)하거나 비상청산했다. flip 경로(htf_flip_target)는 active_position 보유
        # 상태라 여기 해당 없음. 확인 실패 시에도 안전상 진입 보류.
        if self.active_position is None:
            try:
                _ex = await self.client.fetch_position(self.symbol)
                _exqty = float((_ex or {}).get("contracts", 0) or 0)
            except Exception as e:  # noqa: BLE001
                logger.warning("진입 전 포지션 확인 실패 — 안전상 진입 보류: %s", e)
                return
            if _exqty != 0:
                logger.warning(
                    "진입 차단 — 봇 flat 이나 거래소에 미추적 포지션 존재(유저 수동 "
                    "추정 qty=%.6f) → 신규 진입 보류 (%s)", _exqty, self.symbol,
                )
                return

        # #MMBM 2026-07-21: 이번 실행 셋업의 모델 태그 갱신 (매매기록 분리 실측용).
        # SB(Silver Bullet)=Origo, MMBM=별도 태그. 진입~청산 사이 단일 포지션 유지
        # 규약이라 다음 진입 전까지 이 태그가 유효.
        self._active_model = (
            f"{ORIGO_MODEL_NAME} MMBM"
            if setup.source is SetupSource.MMBM
            else ORIGO_MODEL_NAME
        )

        # #SAFETY-1: 일일 손실 한도 도달 시 새 진입 차단 (active position 은 유지).
        # equity fetch 실패(None) 시 baseline reset 을 보류 — 폴백값으로 하루
        # 기준이 오염되는 것 방지 (2026-06-11 리뷰 수정).
        equity_now = await self._fetch_equity_or_none()
        self._maybe_reset_daily_pnl(equity_now)
        # 2026-06-11 리뷰 수정: 한도는 sticky — 한 번 도달하면 그날 내내 차단.
        # (기존엔 실시간 재계산만 봐서, hit 후 보유 포지션 손절로 한도 아래로
        # 내려가면 진입이 다시 풀렸음. flag 는 NY 자정 reset 또는 사용자가
        # 한도를 올리면 API 가 해제.)
        if self._is_daily_loss_limit_hit():
            self._daily_limit_hit = True
        if self._daily_limit_hit:
            logger.info(
                "setup skip (#SAFETY-1) — daily loss limit %.2f%% hit "
                "(today_pnl=%.2fUSDT / start=%.2f)",
                self.daily_loss_limit_pct,
                self._today_realized_pnl_usdt, self._today_start_equity,
            )
            return
        # 2026-06-12 파트너: 페어별 일일 손실 한도 — 이 페어만 당일 중단 (sticky).
        if self._is_daily_pair_loss_limit_hit():
            self._daily_pair_limit_hit = True
        if self._daily_pair_limit_hit:
            logger.info(
                "setup skip — 페어 일일 손실 한도 %.1fR hit (%s pair_today=%.2fUSDT)",
                self.daily_pair_loss_limit_r, self.symbol,
                self._today_pair_realized_pnl_usdt,
            )
            return
        # 2026-06-10 조윤 건의: 일일 수익(TP) 한도 도달 시 신규 진입 중단.
        if self._is_daily_profit_limit_hit():
            self._daily_profit_hit = True
        if self._daily_profit_hit:
            logger.info(
                "setup skip — daily profit limit %.2f%% hit "
                "(today_pnl=%.2fUSDT / start=%.2f) — 그날 목표 달성, 진입 중단",
                self.daily_profit_limit_pct,
                self._today_realized_pnl_usdt, self._today_start_equity,
            )
            return

        # max_sl_distance_pct skip (비정상 큰 SL 차단). 리스크 기반 sizing 이면
        # SL 거리는 qty 로 관리(손실 고정)되므로 이 상한을 우회한다 → 진입 빈도↑.
        if (
            self.max_sl_distance_pct > 0 and setup.entry > 0
            and not self.risk_based_sizing
        ):
            sl_dist_pct = abs(setup.entry - setup.stop_loss) / setup.entry
            if sl_dist_pct > self.max_sl_distance_pct:
                logger.info(
                    "setup skip — SL 거리 %.4f%% > max %.4f%% (entry=%.4f sl=%.4f)",
                    sl_dist_pct * 100, self.max_sl_distance_pct * 100,
                    setup.entry, setup.stop_loss,
                )
                return

        # 2026-06-11 #EDGE-V2: SL 거리 배수 — 백테스트 10국면 검증(스탑헌트 생존,
        # 배수 키울수록 BTC·ETH·IN·OUT 단조 개선). TP 는 원 RR 유지하게 비례 확장.
        # risk_based_sizing(기본 ON)이면 qty 가 1/배수로 줄어 건당 손실(R) 불변.
        # max_sl 게이트(위)는 원본 거리 기준 통과 후 적용, entry 보정(아래)의
        # 평행이동은 확장된 거리를 그대로 보존한다.
        # #CT-SL 2026-06-18: 역추세(되돌림) 진입이면 SL 배수를 sl_dist_mult_ct(x4)로 전환.
        # signed_trend = 진입 직전 20봉 변화율 × 방향부호. <ct_trend_threshold 면 역추세.
        # 순추세/횡보는 기존 sl_dist_mult(x3) 유지. 7페어 5년 robust(net +4.0%p).
        eff_mult = self.sl_dist_mult
        if self.sl_dist_mult_ct > 0.0:
            _sign = 1.0 if setup.direction is Direction.LONG else -1.0
            if setup.entry_trend_pct * _sign < self.ct_trend_threshold:
                eff_mult = self.sl_dist_mult_ct
        if eff_mult > 0 and eff_mult != 1.0 and setup.entry > 0:
            risk = abs(setup.entry - setup.stop_loss)
            if risk > 0:
                rr0 = abs(setup.take_profit - setup.entry) / risk
                new_risk = risk * eff_mult
                # 2026-06-12 hotfix #LIQ-CAP: 확장 SL 거리가 레버리지 청산 거리
                # (entry/leverage)의 80% 를 넘지 않게 캡. 넘으면 SL 도달 전에
                # 강제청산돼 "건당 1R" 관리가 붕괴 + TP 도 비현실 값이 됨
                # (실거래 ETH 숏 TP 920 / SL>청산가 사고로 발견 — 백테스트는
                # 청산가를 시뮬하지 않아 못 잡았던 갭).
                # 캡이 원본 거리보다 작으면 확장 포기(원본 SL 유지) — 원본을
                # 줄이진 않음 (setup 의미 보존).
                if self.leverage > 0:
                    cap = setup.entry * 0.8 / self.leverage
                    if new_risk > cap:
                        capped = max(risk, cap)
                        logger.info(
                            "SL 확장 청산가 캡 — x%.1f(%.4f) → %.4f (lev=%d)",
                            eff_mult, new_risk, capped, self.leverage,
                        )
                        new_risk = capped
                if setup.direction is Direction.LONG:
                    setup.stop_loss = setup.entry - new_risk
                    setup.take_profit = setup.entry + new_risk * rr0
                else:
                    setup.stop_loss = setup.entry + new_risk
                    setup.take_profit = setup.entry - new_risk * rr0
                logger.info(
                    "SL 거리 확장 적용 — sl=%.4f tp=%.4f (RR %.2f 유지)",
                    setup.stop_loss, setup.take_profit, rr0,
                )

        # 2026-06-03: setup.entry 가 현재가에서 max_entry_distance_pct 초과로
        # 멀리 박혀있으면 entry 가격을 현재가 ±max_entry_distance_pct 안으로
        # 보정해서 진입 진행 (skip X). 너무 멀리 박힌 limit 의 미체결 대기 회피.
        # 2026-06-05 fix(#ENTRY-ADJ-RR): entry 만 당기면 SL/TP 와의 거리가 비대칭
        # 으로 바뀌어 RR 이 붕괴된다(실측 2.65 → 0.55). SL/TP 도 같은 delta 로
        # 평행 이동해 risk/reward 거리를 보존 → RR 유지. setup 자체를 보정하므로
        # 이후 active_position/set_position_tpsl/qty 가 모두 보정값을 따른다.
        if self.max_entry_distance_pct > 0 and setup.entry > 0:
            try:
                current_price = await self.client.fetch_ticker(self.symbol)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "max_entry_distance 체크용 fetch_ticker 실패 (무시, 보정 미적용): %s", e,
                )
                current_price = None
            if current_price is not None and current_price > 0:
                max_dist = current_price * self.max_entry_distance_pct
                if abs(setup.entry - current_price) > max_dist:
                    if setup.entry < current_price:
                        adjusted_entry = current_price - max_dist
                    else:
                        adjusted_entry = current_price + max_dist
                    delta = adjusted_entry - setup.entry
                    setup.entry = adjusted_entry
                    setup.stop_loss += delta       # SL/TP 평행 이동 → RR 보존
                    setup.take_profit += delta
                    logger.info(
                        "entry 가격 보정 — entry/SL/TP 평행 이동 %.4f "
                        "(현재가 %.4f, max %.4f%%, RR 보존 %.2f)",
                        delta, current_price,
                        self.max_entry_distance_pct * 100, setup.risk_reward,
                    )

        equity = await self._fetch_equity()
        # #FORCE-ENTRY 2026-06-25: admin 강제진입은 거래소 최소수량을 직접 지정
        # (force_qty)해 리스크 기반 sizing(_calc_qty)을 우회한다. 일반 진입=None.
        qty = force_qty if force_qty is not None else self._calc_qty(setup, equity)
        if qty <= 0:
            logger.warning("qty 계산 결과 0 이하 → skip: setup=%s", setup.ts_ms)
            return

        logger.info(
            "Execute setup %s %s entry=%.4f sl=%.4f tp=%.4f qty=%.4f rr=%.2f",
            self.symbol, side, setup.entry, setup.stop_loss,
            setup.take_profit, qty, setup.risk_reward,
        )
        # 2026-05-29: judgment 응답 인지 — 최근 setup direction 분포 추적.
        # "long 비율 우세인데 short 만 진입" 의문 해소용 (UI 가 보여줌).
        self._recent_setup_directions.append(setup.direction.value)
        if len(self._recent_setup_directions) > 10:
            self._recent_setup_directions = self._recent_setup_directions[-10:]

        # #LIVE-3 fix: entry = setup.entry (계획가 — FVG mean 등) 에 limit. 가격이 거기
        # retrace 하면 체결. SL/TP 가 setup 기준이라 RR 보존. use_market_entry=True 면
        # 레거시 즉시 시장가 (slippage, 비권장).
        # setup.entry 는 위 max_entry_distance 보정에서 이미 갱신됨(보정 시 SL/TP 도
        # 평행 이동해 RR 보존). 계획가 limit 으로 그대로 사용.
        entry_price = None if self.use_market_entry else setup.entry

        # #LIVE-4 fix: SL/TP 는 entry 주문에 동봉하지 않는다. 계획가(limit) 가 현재가에서
        # 벗어나면 Bybit 가 동봉 SL/TP 의 현재가 기준 방향 검증 (10001 "StopLoss should
        # greater/lower base_price") 으로 주문 자체를 거부 → 진입 0. 대신 체결되어 포지션이
        # 생긴 뒤 set_position_tpsl 로 conditional SL/TP 를 박는다 (체결 시점 가격=계획가라
        # 방향 유효). 즉시 체결은 아래, 미체결(pending) 은 _check_pending_entry 가 처리.
        try:
            order_resp = await self.client.place_order(
                symbol=self.symbol,
                side=side,
                qty=qty,
                price=entry_price,
            )
        except Exception as e:  # noqa: BLE001 — 주문 실패도 봇은 계속 돌아야 함
            # 2026-05-29 #SILENT-3: 진입 주문 실패 가시화.
            # 거래소 reject / API 키 권한 / 잔고 부족 / Bybit 시스템 점검 등 다양한
            # 원인 가능. 봇 자체는 계속 돌아야 하지만 운영자가 즉시 인지해야 한다.
            self._order_failure_count += 1
            logger.error(
                "[%s] place_order 실패 #%d — side=%s qty=%.4f price=%s setup_ts=%d: %s",
                self.user_code or "?", self._order_failure_count, side, qty,
                entry_price,
                setup.ts_ms if hasattr(setup, "ts_ms") else 0, e,
            )
            # 2026-06-13 파트너: 약관 미동의(110123)는 사용자만 풀 수 있음 —
            # 연동 텔레그램으로 1회 안내 (봇 인스턴스당 1번, 스팸 방지).
            err_s = str(e)
            if ("110123" in err_s or "Trading Terms" in err_s) and not self._terms_alerted:
                self._terms_alerted = True
                if self.notify_cb is not None and self.user_code:
                    msg = (
                        f"⚠ <b>{self.symbol}</b> 주문이 거래소에서 거절되고 있어요.\n"
                        "Bybit 정책상 이 컨트랙트는 <b>웹/앱에서 약관 동의 1회</b>가 "
                        "필요합니다.\n"
                        "Bybit 에서 해당 페어 주문 화면을 열어 약관에 동의해 주세요 — "
                        "동의 후 봇이 자동으로 다시 매매합니다."
                    )
                    try:
                        task = asyncio.create_task(self.notify_cb(self.user_code, msg))
                        task.add_done_callback(_log_alert_task_exc)
                    except RuntimeError:
                        pass  # 이벤트 루프 없음(동기 테스트)
            return

        # 체결 여부 — filled_qty / avg_fill_price 로 즉시 체결 판정.
        filled = False
        fill_price = entry_price if entry_price is not None else setup.entry
        if isinstance(order_resp, dict):
            fq = order_resp.get("filled_qty")
            if isinstance(fq, (int, float)) and fq > 0:
                filled = True
            avg = order_resp.get("avg_fill_price")
            if isinstance(avg, (int, float)) and avg > 0:
                fill_price = float(avg)

        if not filled and not self.use_market_entry:
            # 계획가 limit 미체결 → pending 등록. step 의 _check_pending_entry 가
            # 체결 승격 (+ SL/TP 박기) / TTL 취소 추적. active_position 은 아직 X.
            self._pending_entry = _PendingEntry(
                direction=setup.direction,
                entry=fill_price,
                stop_loss=setup.stop_loss,
                take_profit=setup.take_profit,
                qty=qty,
                setup_ts_ms=setup.ts_ms,
                placed_ts_ms=int(time.time() * 1000),
                htf_flip_target=htf_flip_target,
                ltf_weight=TF_WEIGHT.get(self.timeframe, 1),
                # 등록 시점에 컨텍스트 미리 빌드 — 체결되면 ENTRY 기록에 그대로 사용.
                context_json=self._build_entry_context_json(
                    setup, fill_price, qty, htf_flip_target,
                ),
            )
            logger.info(
                "지정가 entry 등록 (계획가 미체결 대기, TTL %ds) — %s %s limit=%.4f "
                "sl=%.4f tp=%.4f qty=%.4f",
                self.entry_limit_ttl_sec, self.symbol, side, fill_price,
                setup.stop_loss, setup.take_profit, qty,
            )
            return

        # 즉시 체결 (시장가 or 계획가==현재가) → active_position 확정 + SL/TP conditional 박기.
        # 2026-05-28: 학습/복기 dataset 위해 진입 context + equity snapshot 같이 박음.
        _market_entry_equity = 0.0
        try:
            _market_entry_equity = float(await self._fetch_equity())
        except Exception:  # noqa: BLE001
            pass
        _market_entry_ctx = self._build_entry_context_json(setup, fill_price, qty, htf_flip_target)
        self.active_position = _ActivePosition(
            direction=setup.direction,
            entry=fill_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            qty=qty,
            setup_ts_ms=setup.ts_ms,
            htf_flip_target=htf_flip_target,
            ltf_weight=TF_WEIGHT.get(self.timeframe, 1),
            entry_ts_ms=int(time.time() * 1000),
            context_json=_market_entry_ctx,
            equity_at_entry=_market_entry_equity,
            tp1_price=self._calc_tp1(fill_price, setup.stop_loss, setup.direction),
        )
        # #BUG-2 해소: ENTRY 이벤트 기록 (체결됨 — 먼저 기록).
        self._record_trade(
            TradeEventType.ENTRY,
            direction=setup.direction,
            price=fill_price,
            qty=qty,
            setup_ts_ms=setup.ts_ms,
            reason=(
                f"confluence={setup.confluence_score} "
                f"window={setup.window} rr={setup.risk_reward:.2f}"
            ),
            context_json=self._build_entry_context_json(
                setup, fill_price, qty, htf_flip_target,
            ),
        )
        # #LIVE-4 + P1(#LIVE-6): 체결 후 유효한 보호 SL 보장. 계획가~체결 괴리로
        # SL 이 현재가 너머면 현재가 기준 재계산, 그래도 실패 시 무SL 방치 금지 위해 청산.
        sl_applied = await self._ensure_protective_sl(
            setup.take_profit, abs(setup.entry - setup.stop_loss),
        )
        # #TRAIL-EXCHANGE: SL 확보 후 트레일 무장 — 성공 시 분할익절 skip.
        _trail_armed = sl_applied and await self._arm_trailing()
        # #PARTIAL-TP-ORDER: SL+swing(Entire) 박힌 후 1.5R 부분 TP(Partial) 추가 등록.
        if sl_applied and self.partial_tp_exchange and not _trail_armed:
            await self._setup_partial_tps()
        if sl_applied and htf_flip_target is not None:
            logger.info(
                "HTF flip target armed — %s zone=[%.4f,%.4f] weight=%d",
                htf_flip_target.tf, htf_flip_target.low, htf_flip_target.high,
                htf_flip_target.weight,
            )

    async def _check_pending_entry(self) -> bool:
        """marketable limit 미체결 추적 (#LIVE-1 fix).

        - 체결됨 (거래소 포지션 contracts > 0) → active_position 승격 + ENTRY 기록.
        - 미체결 + TTL(entry_limit_ttl_sec) 경과 → 주문 취소, pending 해제 (타점 포기).
        - 미체결 + TTL 내 → 계속 대기.

        Returns:
            True = 아직 미체결 대기 중 (step 이 신규 진입 skip).
            False = pending 해소 (체결 승격 or 취소) — step 진행 가능.
        """
        pe = self._pending_entry
        if pe is None:
            return False
        try:
            pos = await self.client.fetch_position(self.symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning("pending 체결 확인 fetch_position 실패: %s — 대기 유지", e)
            return True
        contracts = float(pos.get("contracts", 0) or 0) if pos else 0.0
        if contracts > 0:
            # 체결됨 → active_position 승격. entry 는 거래소 체결가 우선.
            entry_px = pe.entry
            if pos:
                ep = (
                    pos.get("entryPrice")
                    or pos.get("entry_price")
                    or pos.get("averagePrice")
                )
                if isinstance(ep, (int, float)) and float(ep) > 0:
                    entry_px = float(ep)
            # 2026-05-28: 학습/복기 dataset 위해 진입 시점 equity snapshot.
            entry_equity = 0.0
            try:
                entry_equity = float(await self._fetch_equity())
            except Exception:  # noqa: BLE001
                pass
            self.active_position = _ActivePosition(
                direction=pe.direction,
                entry=entry_px,
                stop_loss=pe.stop_loss,
                take_profit=pe.take_profit,
                qty=pe.qty,
                setup_ts_ms=pe.setup_ts_ms,
                htf_flip_target=pe.htf_flip_target,
                ltf_weight=pe.ltf_weight,
                entry_ts_ms=int(time.time() * 1000),
                context_json=pe.context_json,
                equity_at_entry=entry_equity,
                tp1_price=self._calc_tp1(entry_px, pe.stop_loss, pe.direction),
            )
            # #BUG-2: ENTRY 기록 (체결됨 — 먼저 기록).
            self._record_trade(
                TradeEventType.ENTRY,
                direction=pe.direction,
                price=entry_px,
                qty=pe.qty,
                setup_ts_ms=pe.setup_ts_ms,
                reason=f"limit filled entry={entry_px:.4f}",
                context_json=pe.context_json,
            )
            # #LIVE-4 + P1: 체결 후 유효한 보호 SL 보장 (안 되면 청산).
            if await self._ensure_protective_sl(
                pe.take_profit, abs(pe.entry - pe.stop_loss),
            ):
                logger.info(
                    "지정가 체결 — active_position 승격 entry=%.4f sl=%.4f tp=%.4f",
                    entry_px, self.active_position.stop_loss, pe.take_profit,
                )
                # #TRAIL-EXCHANGE: SL 확보 후 트레일 무장 — 성공 시 분할익절 skip.
                _trail_armed = await self._arm_trailing()
                # #PARTIAL-TP-ORDER: SL+swing(Entire) 박힌 후 1.5R 부분 TP(Partial) 추가.
                if self.partial_tp_exchange and not _trail_armed:
                    await self._setup_partial_tps()
            self._pending_entry = None
            return False
        # 미체결 — TTL 만료 체크.
        now_ms = int(time.time() * 1000)
        if now_ms - pe.placed_ts_ms >= self.entry_limit_ttl_sec * 1000:
            try:
                await self.client.cancel_bot_orders(self.symbol)
            except Exception as e:  # noqa: BLE001
                logger.warning("pending limit 취소 실패: %s", e)
            logger.info(
                "marketable limit TTL 만료 (%ds 미체결) — 취소, 타점 포기 (setup_ts=%d)",
                self.entry_limit_ttl_sec, pe.setup_ts_ms,
            )
            self._pending_entry = None
            return False
        return True

    async def cancel_pending_entry(self) -> bool:
        """사용자 명령 — pending limit entry 즉시 취소 (TTL 만료 기다리지 않음).

        2026-05-30 파트너 요청: UI 에 'CANCEL' 버튼 — 사용자가 대기 중인
        지정가 주문을 명시적으로 포기. 거래소 cancel_all_orders + 봇 측
        ``_pending_entry`` 비움. active 포지션엔 영향 X (있어도 따로).

        Returns:
            True: pending 있었고 취소함. False: pending 없었음 (no-op).
        """
        if self._pending_entry is None:
            return False
        try:
            await self.client.cancel_bot_orders(self.symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning("pending entry 수동 취소 — cancel_all_orders 실패: %s", e)
        logger.info(
            "pending entry 사용자 취소 (setup_ts=%d)",
            self._pending_entry.setup_ts_ms,
        )
        self._pending_entry = None
        return True

    async def _tick_trail(self, df: pd.DataFrame) -> None:
        """진입 중 새 swing 형성 시 SL 을 그 자리로 이동 (structure-based trail).

        Args:
            df: 최근 LTF OHLCV DataFrame (swing 검출용).
        """
        pos = self.active_position
        if pos is None:
            return
        update = compute_structure_trail(
            df,
            direction=pos.direction,
            entry=pos.entry,
            current_stop_loss=pos.stop_loss,
            buffer_ratio=self.trail_buffer_ratio,
        )
        if update is None:
            return
        try:
            await self.client.modify_stop_loss(self.symbol, update.new_stop_loss)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "trail SL 수정 실패 (%.4f → %.4f): %s",
                pos.stop_loss, update.new_stop_loss, e,
            )
            return
        logger.info(
            "trail SL 이동 %.4f → %.4f (anchor swing @%.4f)",
            pos.stop_loss, update.new_stop_loss, update.anchor_swing_price,
        )
        pos.stop_loss = update.new_stop_loss

    @staticmethod
    def _is_protective_sl(direction: Direction, sl: float, ref_price: float) -> bool:
        """SL 이 ref_price 의 보호 측인지 — SHORT 은 위, LONG 은 아래.

        bybit 는 SHORT SL ≤ 현재가 / LONG SL ≥ 현재가 면 거부(10001).
        """
        if sl <= 0:
            return False
        if direction is Direction.SHORT:
            return sl > ref_price
        return sl < ref_price

    @staticmethod
    def _protective_sl(direction: Direction, ref_price: float, distance: float) -> float:
        """ref_price 의 보호 측으로 distance 만큼 떨어진 SL (SHORT 위 / LONG 아래)."""
        if direction is Direction.SHORT:
            return ref_price + distance
        return ref_price - distance

    async def _ensure_protective_sl(
        self, take_profit: float | None, sl_distance: float,
    ) -> bool:
        """active_position 에 거래소가 수락하는 유효 SL 보장. 실패 시 포지션 청산.

        P1/#LIVE-6: 계획가~체결 사이 가격 급변으로 계획 SL 이 현재가 너머로 가면
        거래소가 거부(10001) → 무SL 포지션. 현재가 기준 보호 측으로 SL 재계산 후
        set_position_tpsl, 1회 재시도, 그래도 실패하면 무SL 방치 금지를 위해 청산.

        Args:
            take_profit: 함께 걸 TP (None 가능).
            sl_distance: 진입가 대비 SL 거리(절대값). 0 이하면 _FALLBACK_SL_PCT 적용.

        Returns:
            True = 유효 SL 적용 (active_position 유지). False = 적용 실패로 청산함.
        """
        pos = self.active_position
        if pos is None:
            return False
        try:
            ref = await self.client.fetch_ticker(self.symbol)
        except Exception:  # noqa: BLE001
            ref = None
        if not ref or ref <= 0:
            ref = pos.entry
        if sl_distance <= 0:
            sl_distance = ref * _FALLBACK_SL_PCT
        # 계획 SL 이 이미 보호 측이면 존중, 아니면 현재가 기준 재계산.
        desired_sl = (
            pos.stop_loss
            if self._is_protective_sl(pos.direction, pos.stop_loss, ref)
            else self._protective_sl(pos.direction, ref, sl_distance)
        )
        if await self.client.set_position_tpsl(
            self.symbol, stop_loss=desired_sl, take_profit=take_profit,
        ):
            pos.stop_loss = desired_sl
            return True
        # 재시도 — 현재가 재조회 후 보호 측 재계산.
        try:
            ref2 = await self.client.fetch_ticker(self.symbol)
        except Exception:  # noqa: BLE001
            ref2 = None
        ref2 = ref2 if (ref2 and ref2 > 0) else ref
        retry_sl = self._protective_sl(pos.direction, ref2, sl_distance)
        if await self.client.set_position_tpsl(
            self.symbol, stop_loss=retry_sl, take_profit=take_profit,
        ):
            pos.stop_loss = retry_sl
            return True
        logger.error(
            "SL 적용 2회 실패 — 무SL 방치 금지: 포지션 청산 %s %s",
            self.symbol, pos.direction.value,
        )
        await self._emergency_close()
        return False

    async def _arm_trailing(self) -> bool:
        """#TRAIL-EXCHANGE (1.4) + #LIQ-TP (Origo 1.6): 거래소 트레일링 무장.

        보호 SL+setup TP(=다음 미스윕 유동성, ICT 정통)가 박힌 뒤 2차 호출로
        trailingStop+activePrice 만 추가한다. **TP 는 건드리지 않음** — 2026-07-08
        파트너 결정("TP 를 정통으로"): 1.4~1.5 의 5R 원거리 확장이 "닿을 수 없는
        TP" 체감을 만들었고, 정합 백테에서 [유동성TP+trail+BE] 하이브리드가
        +282/DD 218 로 5R 확장(+278/228)과 동률 이상 — 유동성 풀 도달 시 전량
        익절, 못 미치면 트레일이 걷는다. TP 미상(복원 tp=0)일 때만 5R 안전망.

        실패해도 포지션은 SL+setup TP 그대로 → 고정 TP 모드 degrade (무해).
        무장 성공 시 분할익절 skip. activePrice 는 현재가가 활성가 앞일 때만
        (지났으면 생략 = 즉시 활성).

        Returns:
            True = 무장 성공(pos.trail_armed 셋). False = off/실패(고정 TP 유지).
        """
        pos = self.active_position
        if pos is None or self.trail_trigger_r <= 0 or self.trail_dist_r <= 0:
            return False
        risk = abs(pos.entry - pos.stop_loss)
        if risk <= 0:
            return False
        is_long = pos.direction is Direction.LONG
        sign = 1.0 if is_long else -1.0
        # #LIQ-TP: setup TP(유동성 타깃) 유지 — TP 미상일 때만 5R 안전망 등록.
        tp_param: float | None = None
        if pos.take_profit <= 0:
            tp_param = pos.entry + sign * risk * 5.0
        act = pos.entry + sign * risk * self.trail_trigger_r
        try:
            cur = await self.client.fetch_ticker(self.symbol)
        except Exception:  # noqa: BLE001
            cur = None
        # 현재가가 활성가를 이미 지났으면 activePrice 생략 → 즉시 활성.
        act_param: float | None = act
        if cur and cur > 0 and ((is_long and cur >= act) or (not is_long and cur <= act)):
            act_param = None
        resp = await self.client.set_position_tpsl(
            self.symbol,
            take_profit=tp_param,
            trailing_stop=risk * self.trail_dist_r,
            active_price=act_param,
        )
        if resp:
            if tp_param is not None:
                pos.take_profit = tp_param
            pos.trail_armed = True
            logger.info(
                "트레일링 무장 — %s trigger=%.4f(%.1fR) dist=%.4f(%.1fR) tp=%.4f(유동성)%s",
                self.symbol, act, self.trail_trigger_r,
                risk * self.trail_dist_r, self.trail_dist_r, pos.take_profit,
                " (즉시활성)" if act_param is None else "",
            )
            return True
        logger.warning(
            "트레일링 무장 실패 — 고정 TP 모드 유지 (%s, setup TP 그대로)", self.symbol,
        )
        return False

    async def _maybe_be_lock(self, df: pd.DataFrame) -> None:
        """#BE-LOCK (Origo 1.5): 이익 be_trigger_r×R 도달 시 SL 본전 이동 (1회).

        최신 봉 high/low 로 도달 판정(분할익절과 동일 방식). 트레일 활성(2R) 전
        1R~2R 구간에서 이익이 풀 손절로 반전되는 패턴(MFE 실측 23%)을 차단.
        이동 실패는 로깅만 — 기존 SL 유지, 다음 step 재시도.
        """
        pos = self.active_position
        if pos is None or pos.be_moved or self.be_trigger_r <= 0 or len(df) == 0:
            return
        risk = abs(pos.entry - pos.stop_loss)
        if risk <= 0:
            return
        last = df.iloc[-1]
        is_long = pos.direction is Direction.LONG
        prof = (
            float(last["high"]) - pos.entry if is_long
            else pos.entry - float(last["low"])
        )
        if prof < risk * self.be_trigger_r:
            return
        resp = await self.client.set_position_tpsl(self.symbol, stop_loss=pos.entry)
        if resp:
            pos.stop_loss = pos.entry
            pos.be_moved = True
            logger.info(
                "본전 잠금 — %s 이익 %.1fR 도달, SL→entry(%.4f)",
                self.symbol, prof / risk, pos.entry,
            )
        else:
            logger.warning(
                "본전 잠금 실패 — 기존 SL 유지, 다음 step 재시도 (%s)", self.symbol,
            )

    def _peak_equity_path(self):
        """peak equity 영속 파일 — 사용자 데이터 폴더 (심볼 공유, monotonic max)."""
        if self.trades_data_dir is None:
            return None
        return Path(self.trades_data_dir) / "peak_equity.json"

    def _dd_throttle_scale(self, equity: float) -> float:
        """#DD-THROTTLE: 현재 낙폭이 임계 초과면 factor, 아니면 1.0.

        peak 는 파일 영속(전 심볼 공유, 최대값만 갱신) — 재시작 생존. 파일 IO
        실패는 메모리 peak 로 계속(보수). equity<=0 이면 스로틀 없음.
        """
        if self.dd_throttle_pct <= 0 or equity <= 0:
            return 1.0
        path = self._peak_equity_path()
        if self._peak_equity <= 0 and path is not None and path.exists():
            try:
                self._peak_equity = float(
                    json.loads(path.read_text(encoding="utf-8")).get("peak", 0.0))
            except Exception:  # noqa: BLE001
                pass
        if equity > self._peak_equity:
            self._peak_equity = equity
            if path is not None:
                try:
                    path.write_text(json.dumps({"peak": equity}), encoding="utf-8")
                except Exception:  # noqa: BLE001
                    pass
        dd = 1.0 - equity / self._peak_equity if self._peak_equity > 0 else 0.0
        if dd > self.dd_throttle_pct / 100.0:
            logger.info(
                "DD 스로틀 발동 — 낙폭 %.1f%% > %.0f%%, 리스크 x%.1f (%s)",
                dd * 100, self.dd_throttle_pct, self.dd_throttle_factor, self.symbol,
            )
            return self.dd_throttle_factor
        return 1.0

    async def _regime_is_up(self) -> bool:
        """#REGIME-OTE: 상승 국면 여부 — 일봉 20일 수익률 z > 0.75 (마감일 기준).

        z = r20 / (일간수익률 20일 표준편차 × √20). 어제까지의 마감 일봉만 사용
        (연구 분류기와 동일 — 후행 전용, lookahead 없음). 일일 캐시.
        fetch/계산 실패 시 False (기본 OTE 유지, 보수).
        """
        day_key = time.strftime("%Y-%m-%d", time.gmtime())
        st = self._regime_state
        if st is not None and st[0] == day_key:
            return st[1]
        is_up = False
        try:
            d = await self._fetch_ohlcv_tf("1d", 30)
            if d is not None and len(d) >= 23:
                closes = d["close"].astype(float).iloc[:-1]  # 미완성 오늘 제외
                ret = closes.pct_change()
                sig = float(ret.iloc[-20:].std())
                r20 = float(closes.iloc[-1] / closes.iloc[-21] - 1.0)
                if sig > 0:
                    is_up = (r20 / (sig * (20 ** 0.5))) > 0.75
        except Exception as e:  # noqa: BLE001
            logger.warning("국면 판정 1d 실패 — 기본 OTE 유지: %s", e)
        self._regime_state = (day_key, is_up)
        return is_up

    async def _effective_ote(self) -> float:
        """현재 국면에 맞는 OTE 깊이 — 상승 국면 & ote_up_level 설정 시 심화."""
        if self.ote_up_level > 0 and await self._regime_is_up():
            return self.ote_up_level
        return self.ote_level

    async def _sweep_gate_blocked(self, direction: Direction) -> bool:
        """#SWEEP-GATE (Origo 1.5): 일봉 스윕-반전 후 K일 역방향 차단 판정.

        SSL 스윕-반전일 = 저점이 직전 10일 최저 하회 && 종가가 봉 상반부 마감
        → 이후 K일 SHORT 차단. BSL(고점 스윕-반락)은 대칭으로 LONG 차단.
        백테 검증(`sweep_bias_gate.py`)과 동일 규칙. 일봉은 하루 단위로만
        바뀌므로 일일 캐시. 1d fetch 실패 시 차단 없음(기존 동작 보수 유지).

        Args:
            direction: 진입하려는 setup 방향.
        Returns:
            True = 차단 (최근 K일 내 반대 방향 스윕-반전 존재).
        """
        if self.sweep_gate_days <= 0:
            return False
        day_key = time.strftime("%Y-%m-%d", time.gmtime())
        st = self._sweep_gate_state
        if st is None or st[0] != day_key:
            block_s = block_l = False
            d = None
            try:
                d = await self._fetch_ohlcv_tf("1d", 10 + self.sweep_gate_days + 2)
            except Exception as e:  # noqa: BLE001
                logger.warning("스윕 게이트 1d fetch 실패 — 게이트 미적용: %s", e)
            if d is not None and len(d) >= 12:
                dd = d.iloc[:-1]  # 마지막 = 오늘 미완성 봉 제외 (마감일만 판정)
                lows = dd["low"].astype(float)
                highs = dd["high"].astype(float)
                closes = dd["close"].astype(float)
                for j in range(1, self.sweep_gate_days + 1):
                    i = len(dd) - j
                    if i < 10:
                        break
                    mid = (float(lows.iloc[i]) + float(highs.iloc[i])) / 2.0
                    if (float(lows.iloc[i]) < float(lows.iloc[i - 10:i].min())
                            and float(closes.iloc[i]) > mid):
                        block_s = True
                    if (float(highs.iloc[i]) > float(highs.iloc[i - 10:i].max())
                            and float(closes.iloc[i]) < mid):
                        block_l = True
            self._sweep_gate_state = (day_key, block_s, block_l)
            st = self._sweep_gate_state
        return st[1] if direction is Direction.SHORT else st[2]

    @staticmethod
    def _exchange_position_direction(pos: dict[str, Any]) -> Direction | None:
        """거래소 fetch_position 응답에서 포지션 방향 추출 (recover 로직과 동일).

        Bybit ccxt: side="long"/"short" (또는 "buy"/"sell"). 인식 실패 시 None.

        Args:
            pos: fetch_position 응답 dict.

        Returns:
            Direction.LONG / SHORT, 또는 side 인식 실패 시 None.
        """
        side = (pos.get("side") or "").lower()
        if side in ("long", "buy"):
            return Direction.LONG
        if side in ("short", "sell"):
            return Direction.SHORT
        return None

    async def _reconcile_open_position(self, ex_pos: dict[str, Any]) -> None:
        """거래소에 열린 포지션이 봇 인식과 방향·수량 일치하는지 검증 + 보정.

        #POS-SYNC 2026-06-06 (04:14 사건): 기존 sync 는 "거래소에 포지션 있기만
        하면" 통과시켜, 거래소 롱 vs 봇 숏 같은 방향 불일치를 못 잡아 SL/비상청산이
        전부 헛발질했다. 거래소를 진실로 삼아 어긋남을 직접 처리한다.

        - 방향 불일치: 봇이 의도 안 한 포지션 → 즉시 비상청산 (파트너 결정 2026-06-06).
        - 같은 방향 수량 불일치 (사용자 부분 수동 청산 등): 봇 qty 를 거래소 실제로 보정.

        Args:
            ex_pos: fetch_position 응답 (contracts != 0 확인된 상태).
        """
        last_known = self.active_position
        if last_known is None:
            # 2026-06-12 고아 체결 입양 (LINK 사고): 재시작으로 주인 잃은 지정가가
            # 나중에 체결되면 봇 모르는 무SL 포지션이 된다. pending 의도가 없는데
            # 거래소에 포지션이 있으면 복구 절차로 입양 — RECOVERED 기록 +
            # TP 복원 시도 + 무SL 이면 보호 SL (P1-2).
            if self._pending_entry is None:
                # #MANUAL-POS-RESPECT: 이미 유저 수동으로 판정해 거절한 포지션이면
                # 매 step 재시도/재로그 억제(같은 시그니처면 조용히 skip).
                ex_dir0 = self._exchange_position_direction(ex_pos)
                ex_entry0 = float(
                    ex_pos.get("entryPrice") or ex_pos.get("entry_price")
                    or ex_pos.get("averagePrice") or 0,
                )
                ex_qty0 = float(ex_pos.get("contracts", 0) or 0)
                if (
                    ex_dir0 is not None
                    and self._declined_manual_sig is not None
                    and self._pos_sig(ex_dir0, ex_entry0, ex_qty0)
                    == self._declined_manual_sig
                ):
                    return
                logger.warning(
                    "미추적 포지션 발견 — 입양 시도 (%s, 고아 체결/수동 진입)",
                    self.symbol,
                )
                await self._recover_position_from_exchange()
            return
        ex_dir = self._exchange_position_direction(ex_pos)
        if ex_dir is None:
            return  # 방향 인식 실패 — 오판 방지 위해 보정 보류.
        if ex_dir is not last_known.direction:
            logger.error(
                "포지션 방향 불일치 — 봇=%s 거래소=%s. 의도 안 한 포지션 → 즉시 비상청산.",
                last_known.direction.value, ex_dir.value,
            )
            await self._emergency_close()
            return
        # 같은 방향 — 수량 변화 처리.
        contracts = float(ex_pos.get("contracts", 0) or 0)
        # #PARTIAL-TP-FILL 2026-06-25: 거래소 Partial 1.5R TP 체결 감지 — partial_on_exchange
        # 로 등록한 포지션의 qty 가 절반 이하로 줄면 1.5R 부분익절 체결로 간주.
        # partial_done 표시 + (partial_be 면) SL 을 본전(entry)으로 — 아래 #TPSL-VERIFY 가
        # want_sl 변경을 감지해 거래소에 자동 재장착(폴링 _maybe_partial_exit 의 본전SL 과
        # 동일 효과, 거래소 자동체결 경로용). reduce_only 청산을 봇이 안 했어도 동기 일치.
        if (
            self.partial_tp_exchange and last_known.partial_on_exchange
            and not last_known.partial_done and contracts > 0
            and contracts <= last_known.qty * 0.6
        ):
            logger.info(
                "Partial 1.5R TP 체결 감지 — qty %.6f→%.6f (%s), SL 본전 이동",
                last_known.qty, contracts, self.symbol,
            )
            last_known.qty = contracts
            last_known.partial_done = True
            if self.partial_be:
                last_known.stop_loss = last_known.entry
        elif contracts > 0 and abs(contracts - last_known.qty) > last_known.qty * 0.01:
            logger.warning(
                "포지션 수량 불일치 — 봇=%.4f 거래소=%.4f. 거래소 기준 보정 (부분 청산 등).",
                last_known.qty, contracts,
            )
            last_known.qty = contracts
        # 2026-06-13 #TPSL-VERIFY (전 사용자 무방비 사고): 봇은 SL/TP 를 안다고
        # 기억하는데 거래소엔 없는 상태(재시작 청소가 conditional 을 벗김 등)를
        # 매 sync 마다 검증·재장착. 0.1% 허용 오차(틱 라운딩 흡수).
        ex_sl = float(
            ex_pos.get("stopLossPrice") or ex_pos.get("stop_loss") or 0,
        ) or 0.0
        ex_tp = float(
            ex_pos.get("takeProfitPrice") or ex_pos.get("take_profit") or 0,
        ) or 0.0
        want_sl = float(last_known.stop_loss or 0.0)
        want_tp = float(last_known.take_profit or 0.0)
        need = (
            (want_sl > 0 and (ex_sl <= 0 or abs(ex_sl - want_sl) > want_sl * 0.001))
            or (want_tp > 0 and (ex_tp <= 0 or abs(ex_tp - want_tp) > want_tp * 0.001))
        )
        if need:
            # 2026-06-13 추가: SL 이 이미 현재가에 관통된 상태면(거래소에서
            # 벗겨진 사이 가격이 지나감) 재장착은 거래소가 거부한다 — 논리적
            # 으로 손절이 났어야 할 포지션이므로 즉시 비상청산 (무방비 출혈
            # 차단, 오늘 LINK -19% 사례).
            mark_now = 0.0
            try:
                mark_now = float(
                    ex_pos.get("markPrice") or ex_pos.get("mark_price") or 0,
                ) or 0.0
            except (TypeError, ValueError):
                mark_now = 0.0
            if mark_now <= 0:
                try:
                    mark_now = float(await self.client.fetch_ticker(self.symbol) or 0)
                except Exception:  # noqa: BLE001
                    mark_now = 0.0
            sl_breached = (
                want_sl > 0 and ex_sl <= 0 and mark_now > 0
                and (
                    mark_now <= want_sl if last_known.direction is Direction.LONG
                    else mark_now >= want_sl
                )
            )
            if sl_breached:
                logger.error(
                    "#TPSL-VERIFY SL 관통 감지 — %s mark=%.4f sl=%.4f (거래소 무SL). "
                    "손절 자리 지남 → 즉시 비상청산.",
                    self.symbol, mark_now, want_sl,
                )
                await self._emergency_close(
                    reason="TPSL-VERIFY: SL 관통(거래소 무SL) 비상청산",
                )
                return
            logger.warning(
                "#TPSL-VERIFY 재장착 — %s 거래소(sl=%.4f tp=%.4f) vs 봇(sl=%.4f tp=%.4f)",
                self.symbol, ex_sl, ex_tp, want_sl, want_tp,
            )
            try:
                ok = await self.client.set_position_tpsl(
                    self.symbol,
                    stop_loss=(want_sl if want_sl > 0 else None),
                    take_profit=(want_tp if want_tp > 0 else None),
                )
                if not ok:
                    logger.error(
                        "#TPSL-VERIFY 재장착 실패 — 다음 step 재시도 (%s)", self.symbol,
                    )
            except Exception as e:  # noqa: BLE001
                logger.error("#TPSL-VERIFY 재장착 예외 — %s: %s", self.symbol, e)

    async def _emergency_close(
        self, reason: str = "SL 적용 실패 비상청산 (무SL 방지)",
    ) -> None:
        """위험(무SL 등) 상황에서 포지션을 시장가 reduce_only 로 청산.

        2026-06-12: ``reason`` 파라미터화 — admin 강제 청산(/admin/position/
        close)도 이 검증된 경로를 재사용하며 매매 기록 사유만 구분.

        2026-05-29 #SILENT-4: 비상청산 자체가 실패하면 active_position 을 None
        으로 만들면 안 된다 — 거래소에 포지션이 남아있는데 봇이 "닫혔다고 인식"
        하면 더 큰 risk (중복 진입 / SL 없는 포지션 방치). 실패 시 ERROR 로
        알람 + active_position 유지 → 다음 step 의 sync 에서 재확인 + 필요 시
        재시도 가능.

        #POS-SYNC 2026-06-06 (04:14 사건): 봇 인식 방향이 거래소 실제와 어긋났을
        수 있다(봇=숏 vs 거래소=롱 → reduce_only same-side 110017 거부 → 무SL 방치).
        fetch_position 으로 실제 방향·수량을 진실로 삼아 청산한다. 조회 실패 시에만
        봇 인식으로 fallback.
        """
        pos = self.active_position
        if pos is None:
            return
        close_dir = pos.direction
        close_qty = pos.qty
        try:
            ex_pos = await self.client.fetch_position(self.symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning("비상청산 전 fetch_position 실패: %s — 봇 인식 방향으로 진행", e)
            ex_pos = None
        if ex_pos is not None:
            contracts = float(ex_pos.get("contracts", 0) or 0)
            if contracts <= 0:
                # 거래소엔 이미 포지션 없음 — 봇 상태만 정리 (중복 reduce_only 방지).
                logger.info("비상청산 불필요 — 거래소 포지션 없음. active_position 정리.")
                self.active_position = None
                return
            ex_dir = self._exchange_position_direction(ex_pos)
            if ex_dir is not None:
                close_dir = ex_dir
                close_qty = contracts
        # 청산 = 포지션 반대 방향 reduce_only.
        close_side = "buy" if close_dir is Direction.SHORT else "sell"
        try:
            await self.client.place_order(
                self.symbol, side=close_side, qty=close_qty,
                price=None, reduce_only=True,
            )
            self._record_trade(
                TradeEventType.MANUAL_CLOSE,
                direction=close_dir,
                price=pos.entry,
                qty=close_qty,
                # 2026-06-12 리뷰 #3: setup_ts 누락 시 원 ENTRY 가 영구 미청산
                # 으로 남아 reconcile 이중 기록 + TP 복원 오염 — 반드시 전달.
                setup_ts_ms=pos.setup_ts_ms or None,
                reason=reason,
            )
            logger.info(
                "비상청산 완료 — %s %s qty=%.4f",
                self.symbol, close_dir.value, close_qty,
            )
            self.active_position = None
        except Exception as e:  # noqa: BLE001
            # 비상청산 실패 — 거래소에 포지션 남아있을 가능성 매우 큼.
            # active_position 유지로 다음 step 에서 _sync_position_state 가
            # 거래소 상태와 다시 맞춰보도록 한다.
            logger.error(
                "비상청산 실패: %s — active_position 유지, 수동 확인 + 다음 step 재시도",
                e,
            )

    def _log_step_market_snapshot(
        self, df: pd.DataFrame, bias: TrendDirection | None,
    ) -> None:
        """매 step 시작 직후 1줄 INFO — 봇이 보는 시장 컨디션 스냅샷.

        2026-05-28 파트너 요청 — fly.io logs 에서 봇 의사결정 흐름이 보이도록.
        매 step 1줄 (5s polling 기준 1줄/5s = 720줄/h ≈ fly.io 비용 적정).

        Args:
            df: 최근 LTF OHLCV DataFrame (가격 / 봉 ts 추출용).
            bias: 결합된 HTF+Daily bias. multi_tf 경로면 None.

        부수효과:
            HTF FVG map 요약 / trend 캐시 핑거프린트가 직전과 다르면 별도 1줄 INFO.
            (변화 감지형 — 같은 값 반복은 안 박힘.)
        """
        if df is None or len(df) == 0:
            return
        try:
            last_ts_ms = int(df.index[-1].value // 10**6)
            current_price = float(df["close"].iloc[-1])
        except Exception:  # noqa: BLE001 — 로깅이 step 깨면 안 됨
            return
        kz = classify_killzone(last_ts_ms)
        kz_name = kz.value if kz is not None else "none"
        bias_str = bias.value if bias is not None else "n/a"
        has_pos = self.active_position is not None
        logger.info(
            "step | tf=%s price=%.2f kz=%s bias=%s active_pos=%s",
            self.timeframe, current_price, kz_name, bias_str, has_pos,
        )

        # HTF FVG map 요약 — 변화 시에만 1줄 (재 빌드 주기 = 5m 봉 길이).
        # bull/bear weight 합산으로 큰 그림 (개별 FVG 가 아닌 톤).
        # HtfFvgEntry.type 은 FVGType StrEnum ("bullish"/"bearish").
        if self._htf_fvg_map_cache:
            bull_w = sum(
                e.weight for e in self._htf_fvg_map_cache
                if str(e.type) == "bullish"
            )
            bear_w = sum(
                e.weight for e in self._htf_fvg_map_cache
                if str(e.type) == "bearish"
            )
            summary = f"bull_w={bull_w} bear_w={bear_w} n={len(self._htf_fvg_map_cache)}"
            if summary != self._last_logged_htf_fvg_summary:
                logger.info("HTF FVG map | %s", summary)
                self._last_logged_htf_fvg_summary = summary

        # Trend 캐시 핑거프린트 — TF 별 state 문자열 변화 시에만 1줄.
        # _refresh_trend_cache 가 봉 ts 변경 시에만 재평가하므로, 캐시 변화 = 추세 변화 신호.
        # TrendState 는 Literal["up","sideways","down"] 문자열 — 직접 사용.
        if self._trend_cache:
            fingerprint_parts = []
            for tf in ("5m", "15m", "1h", "4h", "1d"):
                tup = self._trend_cache.get(tf)
                if tup is None:
                    continue
                _, state = tup
                fingerprint_parts.append(f"{tf}={state}")
            fingerprint = " ".join(fingerprint_parts)
            if fingerprint and fingerprint != self._last_logged_trend_summary:
                logger.info("trend | %s", fingerprint)
                self._last_logged_trend_summary = fingerprint

    def _apply_dol_bias(self, setup: SilverBulletSetup, df: pd.DataFrame) -> None:
        """Draw on Liquidity 편향 필터 (#3 보완) — 역방향 진입은 confluence 감점.

        정통 ICT: 가격은 가장 가까운 미청산 유동성(DOL)으로 끌린다. 그 draw 와 반대인
        setup 은 추세를 거스를 위험 → confluence_score 를 깎아 B+ 게이트(min_confluence)
        에서 걸러지게 한다 (오르는 장에 계속 숏 치던 문제 완화). 지배적 draw = 위/아래
        DOL 중 현재가에 더 가까운 쪽(magnet). in-place 감점.
        """
        if df is None or len(df) < 5:
            return
        swings = detect_swing_points(df)
        if not swings:
            return
        detect_liquidity_sweeps(df, swings)  # swept 마킹 — 이미 먹힌 유동성 제외
        dols = compute_dol(df, swings)
        if not dols:
            return
        bull = next((d for d in dols if d.type == "bullish"), None)
        bear = next((d for d in dols if d.type == "bearish"), None)
        if bull is not None and bear is not None:
            draw = Direction.LONG if bull.distance < bear.distance else Direction.SHORT
        elif bull is not None:
            draw = Direction.LONG
        elif bear is not None:
            draw = Direction.SHORT
        else:
            return
        # 2026-05-28: 지배적 DOL draw 변화 감지 — 변화 시에만 1줄 INFO.
        # _apply_dol_bias 는 매 setup 발생 시 호출 — 매번 박으면 안 됨, 전환 시점만.
        draw_str = draw.value
        if draw_str != self._last_logged_dol_draw:
            if self._last_logged_dol_draw:
                logger.info(
                    "DOL draw 변화 | %s → %s (bull_dist=%.4f bear_dist=%.4f)",
                    self._last_logged_dol_draw, draw_str,
                    bull.distance if bull is not None else -1.0,
                    bear.distance if bear is not None else -1.0,
                )
            self._last_logged_dol_draw = draw_str
        if setup.direction is not draw:
            setup.confluence_score -= _DOL_COUNTER_PENALTY
            setup.confluences.append(
                f"dol_counter_{draw.value}_-{_DOL_COUNTER_PENALTY}",
            )
            logger.info(
                "DOL 역방향 감점 — setup=%s draw=%s score→%d",
                setup.direction.value, draw.value, setup.confluence_score,
            )

    def _regime_floor(self) -> float:
        """횡보 게이트 floor — 롤링 33분위(표본>=MIN) 또는 페어별 q33 하드코딩 fallback.

        _trend_history(최근 setup |진입추세%|)가 REGIME_ROLLING_MIN 이상 쌓이면
        실시간 33분위를 floor 로(페어 변동성 자동 적응). 표본 부족(초기 배포 직후)이면
        REGIME_TREND_FLOOR 하드코딩값으로 안전 동작. 미등록 페어는 0(게이트 off).
        """
        if self.regime_rolling_enabled and len(self._trend_history) >= REGIME_ROLLING_MIN:
            vals = sorted(self._trend_history)
            return vals[len(vals) // 3]  # 하위 33분위 (백테 q33 동일 방식)
        return self.REGIME_TREND_FLOOR.get(self.symbol, 0.0)

    def _strong_trend_floor(self) -> float:
        """cond_align 강추세 임계 — 롤링 70분위 또는 페어별 q70 하드코딩 fallback.

        _regime_floor 와 같은 _trend_history 재사용(추가 계산 0). 표본 충분하면
        실시간 70분위(페어 변동성 자동 적응), 부족하면 STRONG_TREND_FLOOR 하드코딩.
        미등록 페어는 0 → cond_align 게이트 off(강추세 판정 불가 시 안전).
        """
        if self.regime_rolling_enabled and len(self._trend_history) >= REGIME_ROLLING_MIN:
            vals = sorted(self._trend_history)
            return vals[int(len(vals) * 0.7)]  # 상위30% 경계 (백테 q70 동일)
        return self.STRONG_TREND_FLOOR.get(self.symbol, 0.0)

    def _calc_tp1(self, entry: float, stop_loss: float, direction: Direction) -> float:
        """분할익절 TP1(partial_tp_rr×R) 가격 — 진입 시점 risk 기준(이후 trail 무관).

        Returns:
            TP1 가격. partial_tp_rr<=0 또는 risk<=0 이면 0.0(분할 비대상).
        """
        if self.partial_tp_rr <= 0:
            return 0.0
        risk = abs(entry - stop_loss)
        if risk <= 0:
            return 0.0
        if direction is Direction.LONG:
            return entry + self.partial_tp_rr * risk
        return entry - self.partial_tp_rr * risk

    def _restore_partial_state(
        self, entry: float, stop_loss: float, direction: Direction,
    ) -> tuple[float, bool]:
        """복원 포지션 분할익절 상태 추정 → (tp1_price, partial_done). #RESTORE-PARTIAL.

        SL 이 본전(entry 의 0.1% 이내)이면 이미 부분익절+본전이동된 것으로 보고
        partial_done=True 로 재청산 방지. 무SL(risk<=0)은 분할 비대상(tp1=0). 그 외엔
        거래소 entry/SL 로 tp1 계산(SL 원래값이면 R 정확 — 파트너 6/24 복원도 분할 적용).
        """
        if self.partial_tp_rr <= 0 or entry <= 0 or stop_loss <= 0:
            return 0.0, False  # 무SL(SL<=0)/이상 — 분할 비대상
        risk = abs(entry - stop_loss)
        if risk <= 0:
            return 0.0, False
        if risk / entry < 0.001:  # SL≈entry(본전) → 이미 부분익절된 것으로 추정
            return 0.0, True
        return self._calc_tp1(entry, stop_loss, direction), False

    async def force_entry_long(self, qty: float) -> dict[str, Any]:
        """admin 강제 롱 진입 — partial TP 거래소 등록 실측용 (#FORCE-ENTRY 2026-06-25).

        현재가 기준으로 ``SilverBulletSetup`` 을 합성해 정상 진입 경로(``_execute_setup``)
        를 그대로 태운다 → active_position 구성 + ``_ensure_protective_sl`` +
        (``partial_tp_exchange`` 면) ``_setup_partial_tps`` 까지 실거래 진입과 100% 동일.
        qty 는 거래소 최소수량을 ``force_qty`` 로 직접 지정해 리스크 sizing 을 우회.

        SL=현재가 -0.5%, swing TP=현재가 +1.5%(RR 3). entry=현재가 limit 이라 즉시
        체결(marketable). 이미 active/pending 이면 거부(중복 진입 방지). 소액 실측 전용.

        Args:
            qty: 진입 수량 (거래소 최소수량, 예 BTC 0.001).

        Returns:
            결과 dict — ok / detail / entry / stop_loss / take_profit / qty.
        """
        if self.active_position is not None:
            return {"ok": False, "detail": "이미 active position 있음 — 강제진입 거부"}
        if self._pending_entry is not None:
            return {"ok": False, "detail": "pending entry 대기 중 — 강제진입 거부"}
        try:
            px = await self.client.fetch_ticker(self.symbol)
        except Exception as e:  # noqa: BLE001 — 조회 실패도 봇은 계속 동작
            return {"ok": False, "detail": f"현재가 조회 실패: {e}"}
        if not px or px <= 0:
            return {"ok": False, "detail": "현재가 조회 결과 0 이하"}
        risk = px * 0.005                  # SL 0.5% 아래
        stop_loss = px - risk
        take_profit = px + risk * 3.0      # swing TP — RR 3
        setup = SilverBulletSetup(
            ts_ms=int(time.time() * 1000),
            direction=Direction.LONG,
            window="force_test",
            entry=px,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward=3.0,
            confluence_score=99,
            confluences=["force_entry"],
            reasons=["admin 강제진입 (partial TP 거래소 등록 실측)"],
            # fvg=None 이므로 zone/anchor property 가 fvg 를 안 보게 직접 박는다.
            _zone_high=px,
            _zone_low=stop_loss,
            _anchor_idx=0,
        )
        logger.warning(
            "[FORCE-ENTRY] %s 강제 롱 — px=%.4f sl=%.4f tp=%.4f qty=%.6f",
            self.symbol, px, stop_loss, take_profit, qty,
        )
        await self._execute_setup(setup, force_qty=qty)
        active = self.active_position is not None
        pending = self._pending_entry is not None
        return {
            "ok": active or pending,
            "detail": (
                "진입 체결(active)" if active
                else "지정가 대기(pending)" if pending
                else "진입 실패 — 주문 reject 가능, 로그 확인"
            ),
            "entry": px,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "qty": qty,
        }

    async def _setup_partial_tps(self) -> None:
        """#PARTIAL-TP-ORDER 2026-06-25: 진입 후 활성 포지션에 1.5R 부분 TP(50%)를
        Partial mode 로 거래소 등록 — 폴링(_maybe_partial_exit) 대체. swing TP+SL 은
        _ensure_protective_sl 이 Entire 로 이미 박음(Bybit Entire+Partial 공존, 6/25 실측).

        ⚠️ 진입수량 50%가 거래소 최소주문 미만이면 Partial TP 가 거부된다(6/25 실측:
        BTC 0.001 진입→0.0005 < 최소 0.001 → Entire 만 남음). 그 경우
        partial_on_exchange=False 로 두어 폴링이 1차 익절을 담당(fallback). 등록 성공
        시에만 True → 폴링 skip(이중청산 방지).
        """
        pos = self.active_position
        if pos is None or pos.tp1_price <= 0:
            return
        half = pos.qty * 0.5
        # 사전 최소수량 체크 — partial(50%)이 거래소 최소주문 미만이면 거래소가 거부.
        # 등록 시도조차 안 하고 폴링 fallback(partial_on_exchange=False 유지).
        try:
            meta = await self.client.fetch_symbol_meta(self.symbol)
            min_qty = meta.get("min_qty")
        except Exception:  # noqa: BLE001 — 메타 조회 실패는 retCode 체크로 폴백
            min_qty = None
        if min_qty is not None and half < float(min_qty):
            logger.warning(
                "[%s] partial 수량 %.6f < 최소주문 %.6f — 거래소 Partial TP skip, "
                "폴링(_maybe_partial_exit)이 1차 익절 담당(fallback)",
                self.symbol, half, float(min_qty),
            )
            pos.partial_on_exchange = False
            return
        # 등록 시도 — 성공(retCode 0) 시에만 partial_on_exchange=True → 폴링 skip.
        ok = False
        try:
            res = await self.client.set_position_tpsl(
                self.symbol, take_profit=pos.tp1_price, tp_size=half, tpsl_mode="Partial",
            )
            ok = bool(res) and (
                int(res.get("retCode", -1)) == 0 or bool(res.get("alreadySet"))
            )
        except Exception as e:  # noqa: BLE001 — 등록 실패해도 폴링 분할이 백업
            logger.error("[%s] Partial TP 등록 예외 — %s", self.symbol, e)
        pos.partial_on_exchange = ok
        if ok:
            logger.info(
                "Partial 1.5R TP 등록 — %.4f(50%%) + Entire swing 공존 (%s)",
                pos.tp1_price, self.symbol,
            )
        else:
            logger.warning(
                "[%s] Partial TP 거래소 등록 실패 — 폴링 fallback 유지 (tp1=%.4f size=%.6f)",
                self.symbol, pos.tp1_price, half,
            )

    async def _maybe_partial_exit(self, df: pd.DataFrame) -> None:
        """#PARTIAL-TP 2026-06-23: TP1(1R) 도달 시 50% reduce_only 청산 + 본전SL.

        진입 시 계산한 tp1_price 를 최신 봉 high/low 가 터치하면 1회 부분익절.
        체감승률↑·연속손절↓ 목적 — 나머지 50% runner 가 swing TP 까지 net 보존
        (단타/스윙 트레이드오프 우회). 거래소 복원 포지션(tp1_price=0)·이미
        부분익절(partial_done)은 skip. 부분청산 실패 시 원 포지션 유지하고 계속,
        본전SL 설정 실패도 로깅만(다음 step 재시도 여지).
        """
        # #PARTIAL-TP-ORDER: 거래소 Partial TP 가 실제로 박힌 경우만 폴링 분할 skip
        # (이중청산 방지). 진입수량 50%가 최소주문 미만이라 거래소 등록이 실패한
        # (partial_on_exchange=False) 소액 포지션은 폴링이 1차 익절 담당 — fallback.
        _pos = self.active_position
        if self.partial_tp_exchange and _pos is not None and _pos.partial_on_exchange:
            return
        # #TRAIL-EXCHANGE: 트레일 무장 포지션은 분할익절 없음 — runner 전량 트레일
        # (정합 스윕: 순수 trail +240 > partial+trail +189).
        if _pos is not None and _pos.trail_armed:
            return
        pos = self.active_position
        if pos is None or pos.partial_done or pos.tp1_price <= 0:
            return
        if len(df) == 0:
            return
        last = df.iloc[-1]
        hi = float(last["high"])
        lo = float(last["low"])
        hit = (hi >= pos.tp1_price) if pos.direction is Direction.LONG else (lo <= pos.tp1_price)
        if not hit:
            return
        half = pos.qty * 0.5
        side = "sell" if pos.direction is Direction.LONG else "buy"
        try:
            await self.client.place_order(
                symbol=self.symbol, side=side, qty=half, price=None, reduce_only=True,
            )
        except Exception as e:  # noqa: BLE001 — 부분익절 실패해도 원 포지션 유지하고 계속
            logger.error("[%s] 부분익절 청산 실패 — %s", self.symbol, e)
            return
        pos.qty -= half
        pos.partial_done = True
        # partial_be: 나머지 50% SL 을 본전으로 — TP1 닿은 거래는 최악이 본전(무손실).
        if self.partial_be:
            pos.stop_loss = pos.entry
            try:
                await self.client.set_position_tpsl(
                    self.symbol, stop_loss=pos.entry, take_profit=pos.take_profit,
                )
            except Exception as e:  # noqa: BLE001
                logger.error("[%s] 부분익절 후 본전 SL 설정 실패 — %s", self.symbol, e)
        logger.info(
            "부분익절 — TP1=%.4f 도달, 50%% 청산 + SL 본전 (%s %s, 잔여 qty=%.4f)",
            pos.tp1_price, self.symbol, pos.direction.value, pos.qty,
        )

    def _apply_ote_boost(self, setup: SilverBulletSetup, df: pd.DataFrame) -> None:
        """직전 임펄스 swing leg 의 피보나치 0.618~0.786(ICT OTE) 되돌림 진입 시 +1.

        ZigZag auto-fib 의 핵심(마지막 pivot leg → retracement)을 swing detector 로
        재현. 진입 방향과 임펄스 방향 정합 필요(LONG=상승 leg 되돌림 매수, SHORT=하락
        leg 되돌림 매도). 7페어 5년 검증: net +0.5%p·거래 +12 — ICT 정통 OTE 편입.
        in-place 가산.
        """
        from aurora_ict.indicators.swing_points import SwingType, detect_swing_points

        swings = detect_swing_points(df)
        if len(swings) < 2:
            return
        a, b = swings[-2], swings[-1]  # 직전 leg (a→b)
        is_long = setup.direction is Direction.LONG
        if is_long and b.type is not SwingType.HIGH:
            return
        if (not is_long) and b.type is not SwingType.LOW:
            return
        hi, lo = max(a.price, b.price), min(a.price, b.price)
        if hi <= lo:
            return
        cl = float(df["close"].iloc[-1])
        retr = (hi - cl) / (hi - lo) if is_long else (cl - lo) / (hi - lo)
        if 0.618 <= retr <= 0.786:  # ICT OTE 구간(sweet spot 0.705)
            setup.confluence_score += 1
            setup.confluences.append("ote")
            logger.info(
                "OTE fib 가점 — setup=%s retr=%.3f score→%d",
                setup.direction.value, retr, setup.confluence_score,
            )

    def _set_entry_trend(self, setup: SilverBulletSetup, df: pd.DataFrame) -> None:
        """진입 직전 20봉 변화율(%)을 setup.entry_trend_pct 에 기록 (#CT-SL).

        _execute_setup 이 signed_trend(= 이 값 × 방향부호) < ct_trend_threshold 면
        역추세(되돌림)로 보고 SL 배수를 sl_dist_mult_ct(x4)로 전환. confluence·게이트
        영향 0 (SL 거리만). 데이터 부족(<21봉)이면 0 유지(= 순추세 취급, x3).
        """
        closes = df["close"]
        if len(closes) > 20:
            past = float(closes.iloc[-21])
            if past > 0:
                setup.entry_trend_pct = (float(closes.iloc[-1]) - past) / past * 100.0

    def _set_smart_size(self, setup: SilverBulletSetup, df: pd.DataFrame) -> None:
        """#SMART-SIZE 2026-07-20 (FST#7): 품질 기반 사이즈 배수 계산·기록.

        LuxAlgo 신호계열 대입 결과 유일 walk-forward robust 한 처방. 진입 품질을
        3신호로 점수화(0~3) 후 사이즈 배수로 변환 — _calc_qty 가 risk_amount 에 곱함.
        거래를 거르는 게 아니라(빈도 불변) 좋은 진입에 자금을 더 배분한다.

        품질 신호 (진입 방향과 정합 시 +1):
            - 볼륨: 진입봉 거래량 >= 최근 20봉 평균 (거래 관심 확인)
            - Nadaraya-Watson 중심선: 가우시안 커널 회귀 중심 대비 가격 위치 정합
              (롱=중심 위 / 숏=중심 아래 — 추세 방향 확인)
            - RSI(14): 방향 정합 (롱=RSI>50 / 숏=RSI<50 — 모멘텀 확인)
        배수 = clip(0.7 + q*0.2, 0.4, 1.4). q 평균 1.5 면 배수 ~1.0(중립).

        Args:
            setup: 대상 setup. smart_size_scale 를 in-place 기록.
            df: 5m OHLCV. 데이터 부족(<50봉)이면 1.0 유지(중립).
        """
        if not self.smart_size_enabled or len(df) < 50:
            return
        c = df["close"].to_numpy()
        v = df["volume"].to_numpy()
        is_long = setup.direction is Direction.LONG

        # 1) 볼륨 정합 — 진입봉 vs 최근 20봉 평균.
        vol_ma = float(v[-20:].mean())
        vol_ok = vol_ma > 0 and float(v[-1]) >= vol_ma

        # 2) Nadaraya-Watson 중심선(가우시안 커널, bw=8, 최근 50봉) 대비 위치.
        win = min(50, len(c))
        seg = c[-win:]
        idx = np.arange(win)
        w = np.exp(-((win - 1 - idx) ** 2) / (2 * 8.0 ** 2))  # 최신봉 기준 가중
        nw_center = float(np.sum(seg * w) / np.sum(w))
        nw_ok = (float(c[-1]) > nw_center) == is_long

        # 3) RSI(14) 방향 정합.
        d = np.diff(c[-15:])
        up = float(np.sum(np.where(d > 0, d, 0.0)))
        dn = float(np.sum(np.where(d < 0, -d, 0.0)))
        rsi = 100.0 - 100.0 / (1.0 + up / (dn + 1e-9))
        rsi_ok = (rsi > 50) == is_long

        q = int(vol_ok) + int(nw_ok) + int(rsi_ok)
        setup.smart_size_scale = float(np.clip(0.7 + q * 0.2, 0.4, 1.4))

    def _apply_cisd_boost(self, setup: SilverBulletSetup, df: pd.DataFrame) -> None:
        """CISD(Change in State of Delivery) 순응 시 confluence +1 (#CISD 2026-06-06).

        CISD = 가격 전달 방향 전환의 1캔들 micro 신호 (MSS 의 빠른 버전). 직전 연속
        반대 캔들들의 시초가 라인을 현재 봉이 돌파하면 발생. setup 방향과 같은 방향의
        CISD 가 잡히면 "전환 순응" 가점을 준다 — 정통에서 빠른 진입 confirmation.
        in-place 가산.
        """
        cisd = detect_cisd(df)
        if cisd is None:
            return
        want = CisdType.BULLISH if setup.direction is Direction.LONG else CisdType.BEARISH
        if cisd is want:
            setup.confluence_score += 1
            setup.confluences.append(f"cisd={cisd.value}")
            logger.info(
                "CISD 순응 가점 — setup=%s cisd=%s score→%d",
                setup.direction.value, cisd.value, setup.confluence_score,
            )

    def _apply_po3_boost(self, setup: SilverBulletSetup, df: pd.DataFrame) -> None:
        """Power of 3 (AMD) Distribution 국면 순응 시 confluence +1 (2026-06-17).

        AMD = Accumulation/Manipulation/Distribution. Distribution(NY 진짜 방향성
        단계)에서 진입하면 London 가짜 움직임(Manipulation)을 피하고 추세에 순응한다.
        5년·7페어 백테스트 검증: cisd+po3 조합이 robust 흑자(+3.18%, base −0.76 대비).
        시간 기반(방향 무관)이라 현재 봉(df 마지막) 시각으로 AMD phase 를 판정한다.
        in-place 가산.

        Args:
            setup: confluence_score 를 in-place 가산할 setup.
            df: 현재 OHLCV — 마지막 봉 시각으로 AMD phase 판정 (라인 880 ts_ms 동일 방식).
        """
        ts_ms = int(df.index[-1].value // 10**6)
        if amd_phase(ts_ms) is AmdPhase.DISTRIBUTION:
            setup.confluence_score += 1
            setup.confluences.append("po3_distribution")
            logger.info(
                "PO3 Distribution 가점 — setup=%s score→%d",
                setup.direction.value, setup.confluence_score,
            )

    async def _apply_smt_boost(
        self, setup: SilverBulletSetup, df: pd.DataFrame,
    ) -> None:
        """SMT divergence(상관 자산) 순응 시 confluence +1 (#SMT 2026-06-06).

        정통 ICT: 두 상관 자산(BTC↔ETH)이 같은 시점 swing 에서 한쪽만 새 고/저점을
        박으면 '기관 흐름 누설' — 못 따라온 쪽의 반전 신호. self.symbol swing 과 상관
        심볼 OHLCV 를 비교해 최근 divergence 방향이 setup 과 일치하면 가점.
        짝 없는 알트 심볼·비활성·fetch 실패·divergence 없음 → 무영향(가점 X).
        """
        if not self.smt_enabled:
            return
        corr_symbol = _SMT_CORR_PAIRS.get(self.symbol)
        if corr_symbol is None:
            return  # 상관 짝 없는 심볼 — SMT 적용 불가.
        main_swings = detect_swing_points(df)
        if len(main_swings) < 2:
            return
        try:
            corr_df = await self._fetch_ohlcv_tf(
                self.timeframe, self.ohlcv_limit, symbol=corr_symbol,
            )
        except Exception as e:  # noqa: BLE001 — corr fetch 실패가 진입 막으면 안 됨
            logger.debug("SMT corr OHLCV fetch 실패 (%s): %s — skip", corr_symbol, e)
            return
        if len(corr_df) == 0:
            return
        events = detect_smt_divergence(main_swings, corr_df)
        if not events:
            return
        latest = events[-1]
        want = SmtType.BULLISH if setup.direction is Direction.LONG else SmtType.BEARISH
        if latest.type is want:
            setup.confluence_score += 1
            setup.confluences.append(f"smt={latest.type.value}@{latest.ts_ms}")
            logger.info(
                "SMT 순응 가점 — setup=%s smt=%s (corr=%s) score→%d",
                setup.direction.value, latest.type.value, corr_symbol,
                setup.confluence_score,
            )

    async def _fetch_equity_or_none(self) -> float | None:
        """가용 자산 (USDT equity) 조회 — 실패/형식 불명 시 None.

        2026-06-11 리뷰 수정: 일일 한도 baseline(``_maybe_reset_daily_pnl``)이
        폴백 상수로 오염되지 않게, 실패를 None 으로 구분해 돌려준다.
        """
        try:
            bal = await self.client.fetch_balance()
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_balance 실패: %s", e)
            return None
        if not isinstance(bal, dict):
            return None
        # ccxt 표준 = {"USDT": {"total": float, "free": float, ...}, ...}
        usdt = bal.get("USDT")
        if isinstance(usdt, dict):
            total = usdt.get("total")
            if isinstance(total, (int, float)) and total > 0:
                return float(total)
        # Bybit direct format fallback
        total = bal.get("total")
        if isinstance(total, (int, float)) and total > 0:
            return float(total)
        return None

    async def _fetch_equity(self) -> float:
        """가용 자산 (USDT equity) 조회 — 실패 시 fallback 1000.0 (테스트/오류 대비).

        qty 산정 등 "값이 꼭 필요한" 호출처용. 일일 한도 baseline 은
        ``_fetch_equity_or_none`` 을 써서 실패 시 reset 을 보류한다.
        """
        eq = await self._fetch_equity_or_none()
        return eq if eq is not None else 1000.0

    def _calc_qty_risk_based(
        self, setup: SilverBulletSetup, equity: float,
    ) -> float:
        """리스크 기반 qty — 건당 리스크(equity %)를 고정하고 SL 거리로 역산.

        risk_pct = min(base + step * score, max)
        risk_amount = equity * risk_pct/100
        qty = risk_amount / |entry - stop_loss|   (SL 거리가 멀수록 qty 작아짐)
        → 건당 손실(SL 히트 시) = risk_amount 로 일정. over-leverage 방지를 위해
        필요 notional 이 equity*leverage*position_pct_max% 를 넘지 않게 상한.
        """
        sl_dist = abs(setup.entry - setup.stop_loss)
        if sl_dist <= 0:
            return 0.0
        score = max(0, setup.confluence_score)
        risk_pct = min(
            self.risk_per_trade_base + self.risk_per_trade_step * score,
            self.risk_per_trade_max,
        )
        risk_amount = equity * (risk_pct / 100.0)
        # #DD-THROTTLE: 낙폭 구간 리스크 축소 (변동성 드래그 방어).
        risk_amount *= self._dd_throttle_scale(equity)
        # #SMART-SIZE: 진입 품질(볼륨·NW·RSI) 배수 — 좋은 진입 자금↑ (기본 1.0 중립).
        # 실수일 때만 적용 (테스트 mock setup 방어 — round_amount 와 동일 패턴).
        _ss = getattr(setup, "smart_size_scale", 1.0)
        if isinstance(_ss, (int, float)) and not isinstance(_ss, bool):
            risk_amount *= _ss
        qty = risk_amount / sl_dist
        # over-leverage 상한 — 필요 notional 이 가용 마진×레버리지 한도를 못 넘게.
        max_notional = equity * self.leverage * (self.position_pct_max / 100.0)
        max_qty = max_notional / setup.entry
        return min(qty, max_qty)

    def _calc_qty(self, setup: SilverBulletSetup, equity: float) -> float:
        """진입 qty 계산 — confluence_score 단계별 notional sizing.

        risk_based_sizing=True 면 건당 리스크 고정(_calc_qty_risk_based), 아니면
        기존 고정 % notional sizing:
        pct = min(base + step * score, max)
        margin = equity * pct/100  → leveraged notional = margin * leverage
        qty = leveraged notional / entry_price
        """
        if setup.entry <= 0:
            return 0.0
        if self.risk_based_sizing:
            qty = self._calc_qty_risk_based(setup, equity)
        else:
            score = max(0, setup.confluence_score)
            pct = min(
                self.position_pct_base + self.position_pct_step * score,
                self.position_pct_max,
            )
            margin = equity * (pct / 100.0)
            _ss = getattr(setup, "smart_size_scale", 1.0)  # #SMART-SIZE (mock 방어)
            if isinstance(_ss, (int, float)) and not isinstance(_ss, bool):
                margin *= _ss
            notional = margin * self.leverage
            qty = notional / setup.entry
        # 페어 확장 — 거래소 lot step 에 맞춰 qty 정렬(심볼별 precision). 미지원
        # client 면 원본 유지(안전 폴백). 반환이 실수일 때만 적용(테스트 mock 방어).
        # BTC/ETH 는 precision 관대해 영향 거의 없음.
        if hasattr(self.client, "round_amount"):
            rounded = self.client.round_amount(self.symbol, qty)
            if asyncio.iscoroutine(rounded):
                rounded.close()  # round_amount 는 sync 계약 — mock 의 coroutine 폐기
            elif isinstance(rounded, (int, float)) and not isinstance(rounded, bool):
                qty = float(rounded)
        # 최소 주문수량 미달 시 skip — 심볼 메타가 있으면 그 min_qty, 없으면 Bybit
        # BTC 기준 0.001 폴백. (작은 잔고에서 의도 notional 초과 박는 회귀 회피,
        # 호출처에서 qty 0 이하 skip 분기 활용.)
        meta = self._symbol_meta if isinstance(self._symbol_meta, dict) else {}
        meta_min = meta.get("min_qty")
        min_qty = (
            meta_min
            if isinstance(meta_min, (int, float)) and not isinstance(meta_min, bool)
            and meta_min > 0
            else 0.001
        )
        if qty < min_qty:
            return 0.0
        return qty

    def _maybe_reset_daily_pnl(self, equity_now: float | None) -> None:
        """NY local 자정 기준 일일 누적 손익 reset (#SAFETY-1).

        매일 새 거래일이 시작하면 ``_today_realized_pnl_usdt`` 와 ``_today_start_equity``
        를 갱신하고 ``_daily_limit_hit`` flag 풀어줌. ICT 정통 일일 boundary 정합.

        Args:
            equity_now: 현재 가용 자산 (USDT). 새 날짜 시작 시 baseline 으로 박힘.
                None(잔고 조회 실패)이면 reset 을 **보류** — 다음 성공 fetch 때
                정확한 baseline 으로 reset (2026-06-11 리뷰: 폴백 상수가 하루치
                한도 기준을 오염시키던 문제).
        """
        ny_date = datetime.now(UTC).astimezone(_NY_TZ).strftime("%Y-%m-%d")
        if ny_date != self._today_date_str:
            if equity_now is None:
                logger.warning(
                    "daily PnL reset 보류 — 잔고 조회 실패 (NY %s). 다음 step 재시도.",
                    ny_date,
                )
                return
            self._today_date_str = ny_date
            self._today_realized_pnl_usdt = 0.0
            self._today_start_equity = equity_now if equity_now > 0 else 0.0
            self._daily_limit_hit = False
            # 2026-06-12 페어별 한도도 새 거래일에 함께 reset.
            self._today_pair_realized_pnl_usdt = 0.0
            self._daily_pair_limit_hit = False
            self._daily_profit_hit = False
            logger.info(
                "daily PnL reset (NY %s) — start_equity=%.2f",
                ny_date, self._today_start_equity,
            )

    def _is_daily_pair_loss_limit_hit(self) -> bool:
        """페어별 일일 손실 한도 도달 여부 (R 배수 기준). 2026-06-12 파트너.

        R(1회 리스크 금액) = 시작 equity × risk_per_trade_base%. 이 페어의
        오늘 누적 손실이 limit_r × R 이상이면 True — 이 페어만 당일 중단.
        limit_r=0 또는 시작 equity 미확보면 비활성.
        """
        if self.daily_pair_loss_limit_r <= 0 or self._today_start_equity <= 0:
            return False
        r_usdt = self._today_start_equity * (self.risk_per_trade_base / 100.0)
        if r_usdt <= 0:
            return False
        loss = -self._today_pair_realized_pnl_usdt
        return loss >= self.daily_pair_loss_limit_r * r_usdt

    def _is_daily_loss_limit_hit(self) -> bool:
        """일일 손실 한도 초과 여부 (#SAFETY-1).

        Returns:
            True 면 새 진입 차단. ``daily_loss_limit_pct == 0`` 또는 시작 equity
            미정이면 항상 False.
        """
        if self.daily_loss_limit_pct <= 0:
            return False
        if self._today_start_equity <= 0:
            return False
        loss_pct = -self._today_realized_pnl_usdt / self._today_start_equity * 100.0
        return loss_pct >= self.daily_loss_limit_pct

    def _is_daily_profit_limit_hit(self) -> bool:
        """일일 수익(TP) 한도 초과 여부 (2026-06-10 조윤 건의).

        하루 누적 수익이 ``daily_profit_limit_pct`` 도달하면 그날 신규 진입 중단
        (active position 은 유지). "몇 % 먹으면 그날 종료" 전략.

        Returns:
            True 면 새 진입 차단. ``daily_profit_limit_pct == 0`` 또는 시작 equity
            미정이면 항상 False.
        """
        if self.daily_profit_limit_pct <= 0:
            return False
        if self._today_start_equity <= 0:
            return False
        profit_pct = self._today_realized_pnl_usdt / self._today_start_equity * 100.0
        return profit_pct >= self.daily_profit_limit_pct

    def daily_loss_status(self) -> dict[str, Any]:
        """현재 일일 손익 상태 — API / UI 노출용 (#SAFETY-1).

        Returns:
            dict (limit_pct / today_pnl_usdt / today_pct / start_equity / hit / date).
        """
        loss_pct = (
            -self._today_realized_pnl_usdt / self._today_start_equity * 100.0
            if self._today_start_equity > 0 else 0.0
        )
        return {
            "limit_pct": self.daily_loss_limit_pct,
            "today_pnl_usdt": self._today_realized_pnl_usdt,
            "today_pct": -loss_pct,  # 음수면 손실, 양수면 익절
            "start_equity": self._today_start_equity,
            "hit": self._daily_limit_hit,
            "date_ny": self._today_date_str,
            # 2026-06-10 조윤 건의: 일일 수익(TP) 한도 상태 동봉.
            "profit_limit_pct": self.daily_profit_limit_pct,
            "profit_hit": self._daily_profit_hit,
        }

    async def _sync_today_realized_pnl(self) -> None:
        """거래소 closed-pnl 로 오늘(NY 자정~) 실현손익 동기화 (#BUG-7 해소).

        내부 누적 (SYNC_CLOSE 가 close 가격 모르고 PnL 0 으로 기록하던) 대신 거래소를
        진실로 사용 → today_pnl 이 거래소와 일치 + #SAFETY-1 일일 손실 한도 정확.
        매 step 호출 — NY 자정~now 는 보통 7일 미만 단일 chunk 라 비용 작음.
        """
        ny_midnight = (
            datetime.now(UTC)
            .astimezone(_NY_TZ)
            .replace(hour=0, minute=0, second=0, microsecond=0)
        )
        since_ms = int(ny_midnight.timestamp() * 1000)
        try:
            closed = await self.client.fetch_closed_positions(
                since_ms=since_ms, limit=200,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("today realized pnl 거래소 동기화 실패: %s — 기존값 유지", e)
            return
        total = 0.0
        pair_total = 0.0
        for cp in closed:
            ts = int(getattr(cp, "closed_at_ts", 0) or 0)
            if ts >= since_ms:
                pnl = float(getattr(cp, "pnl_usd", 0.0) or 0.0)
                total += pnl
                # 2026-06-12 페어별 한도 — 이 봇 심볼 분만 따로 누적.
                if str(getattr(cp, "symbol", "") or "") == self.symbol:
                    pair_total += pnl
        self._today_realized_pnl_usdt = total
        self._today_pair_realized_pnl_usdt = pair_total
        if self._is_daily_pair_loss_limit_hit() and not self._daily_pair_limit_hit:
            self._daily_pair_limit_hit = True
            logger.warning(
                "페어 일일 손실 한도 HIT — %s limit=%.1fR pair_today=%.2fUSDT "
                "(이 페어만 당일 진입 중단)",
                self.symbol, self.daily_pair_loss_limit_r,
                self._today_pair_realized_pnl_usdt,
            )
        if self._is_daily_loss_limit_hit() and not self._daily_limit_hit:
            self._daily_limit_hit = True
            logger.warning(
                "daily loss limit HIT (거래소 동기화) — limit=%.2f%% today=%.2fUSDT "
                "(start_equity=%.2f)",
                self.daily_loss_limit_pct,
                self._today_realized_pnl_usdt,
                self._today_start_equity,
            )
        if self._is_daily_profit_limit_hit() and not self._daily_profit_hit:
            self._daily_profit_hit = True
            logger.info(
                "daily profit limit HIT (거래소 동기화) — limit=%.2f%% today=%.2fUSDT "
                "(start_equity=%.2f) — 그날 목표 달성",
                self.daily_profit_limit_pct,
                self._today_realized_pnl_usdt,
                self._today_start_equity,
            )

    def _build_entry_context_json(
        self,
        setup: SilverBulletSetup,
        fill_price: float,
        qty: float,
        htf_flip_target: HtfFvgEntry | None = None,
    ) -> str:
        """ENTRY 이벤트용 풍부한 컨텍스트 JSON — 진입 사유/근거 (#PR-C 거래 저널).

        진입가/SL/TP/qty/레버리지/RR/score/confluences(어떤 지표 작동)/source/window/
        HTF flip 정보를 한 묶음으로 JSON 직렬화. trades.jsonl 의 ``context_json`` 필드
        + trade_journal.log 로 사람이 나중에 분석 가능.
        """
        source = getattr(setup, "source", None)
        if hasattr(source, "value"):
            source_val: str | None = source.value
        elif source is not None:
            source_val = str(source)
        else:
            source_val = None
        ctx: dict[str, Any] = {
            "entry": float(fill_price),
            "sl": float(setup.stop_loss),
            "tp": float(setup.take_profit),
            "qty": float(qty),
            "leverage": int(self.leverage),
            "rr": float(setup.risk_reward),
            "score": int(setup.confluence_score),
            "confluences": list(setup.confluences),
            "window": setup.window,
            "source": source_val,
            "htf_flip_target_armed": htf_flip_target is not None,
        }
        if htf_flip_target is not None:
            ctx["htf_flip_target"] = {
                "tf": htf_flip_target.tf,
                "weight": int(htf_flip_target.weight),
                "low": float(htf_flip_target.low),
                "high": float(htf_flip_target.high),
            }
        return json.dumps(ctx, ensure_ascii=False)

    def _build_close_context_json(
        self,
        entry_pos: _ActivePosition,
        close_price: float,
        pnl_usd: float,
        close_reason: str,
    ) -> str:
        """2026-05-29 청산 시점 봇 판단 스냅샷 — 봇 개선 분석용.

        파트너 요청: "진입 사유 + 당시 봇 판단 — 진짜 구체적으로. 이 데이터들
        가지고 앞으로 봇 개선하게".

        진입 시 ``_build_entry_context_json`` 가 진입 컨텍스트를 박고, 청산 시
        이 함수가 청산 컨텍스트를 박아 사용자가 둘을 비교 분석 가능하게 한다.
        예: "진입할 때 bull_w=250 우세였는데 청산할 때 bear_w=300 으로 flip"
        같은 패턴을 UI 에서 직접 볼 수 있다.

        포함 정보:
            · 청산 가격 / 실현 PnL / 사유 (SL/TP/sync/manual/flip)
            · 진입 vs 청산 가격 변화 (pct)
            · 현재 HTF FVG 가중치 (bull_w / bear_w / n)
            · 마지막 EMA bias / DOL draw / trend / killzone 캐시값
            · 봇 가동 진단 카운터 (recovery_failed / sync_failure_streak 등)
        """
        # 함수 내 import — killzone 모듈 임포트 순환 회피.
        from aurora_ict.timing.killzone import classify_killzone  # noqa: PLC0415
        now_ms = int(time.time() * 1000)
        kz = classify_killzone(now_ms)
        # HTF FVG map cache 그대로 합계.
        htf_map = self._htf_fvg_map_cache or []
        bull_w = sum(
            int(e.weight) for e in htf_map if getattr(e.type, "value", "") == "bullish"
        )
        bear_w = sum(
            int(e.weight) for e in htf_map if getattr(e.type, "value", "") == "bearish"
        )
        # entry 대비 청산 가격 % 변화 (long 은 +면 익, short 은 -면 익).
        if entry_pos.entry > 0:
            move_pct = (close_price - entry_pos.entry) / entry_pos.entry * 100.0
        else:
            move_pct = 0.0
        ctx: dict[str, Any] = {
            "close_price": float(close_price),
            "entry_price": float(entry_pos.entry),
            "move_pct": round(move_pct, 4),
            "pnl_usd": float(pnl_usd),
            "close_reason": close_reason,
            # 청산 시점의 HTF FVG 가중치 스냅샷 (진입과 비교용).
            "htf_fvg_bull_weight": bull_w,
            "htf_fvg_bear_weight": bear_w,
            "htf_fvg_n": len(htf_map),
            # 봇 의사결정 캐시 직전 값들 — log_step_market_snapshot 의 한 줄과 같은 정보.
            "ema_bias_last": self._last_logged_htf_ema_bias or None,
            "dol_draw_last": self._last_logged_dol_draw or None,
            "killzone": kz.value if kz is not None else None,
            # 진단 카운터 — 청산 시점 봇 상태가 정상인지.
            "diagnostics": {
                "recovery_failed": self._recovery_failed,
                "sync_failure_streak": self._sync_failure_streak,
                "order_failure_count": self._order_failure_count,
            },
        }
        return json.dumps(ctx, ensure_ascii=False)

    def _record_trade(
        self,
        event_type: TradeEventType,
        *,
        direction: Direction,
        price: float,
        qty: float,
        entry_for_pnl: float | None = None,
        setup_ts_ms: int | None = None,
        reason: str = "",
        pnl_override: float | None = None,
        context_json: str | None = None,
    ) -> None:
        """매매 이벤트 1건을 trades_store 에 기록 (#BUG-2 해소).

        Args:
            event_type: TradeEventType (ENTRY / SL_HIT / TP_HIT / FLIP_* / RECOVERED /
                SYNC_CLOSE / MANUAL_CLOSE).
            direction: 포지션 방향 (LONG/SHORT).
            price: 체결/발생 가격.
            qty: 수량.
            entry_for_pnl: 청산 이벤트일 때 entry 가격 (PnL 계산용). 진입 이벤트는 None.
            setup_ts_ms: ENTRY/청산 매칭용 setup ts. RECOVERED 등은 None.
            reason: 디버그/분석 사유.
            pnl_override: 거래소 closed-pnl 등 *정확한* 실현 PnL. 제공되면 entry/price
                기반 추정 대신 이 값을 기록(수수료/펀딩 반영된 진실). 청산 이벤트용.

        TradesStore lazy init — 첫 호출 시 ``data_dir()`` 디렉토리 보장.
        실패해도 봇 전체는 멈추지 않게 try/warning 처리.
        """
        if self._trades_store is None:
            # 2026-05-29: 사용자별 격리 — trades_data_dir 가 있으면 그 디렉토리,
            # 없으면 기본 data_dir() (단일 사용자 / .exe 흐름).
            store_dir = self.trades_data_dir or _ict_data_dir()
            try:
                self._trades_store = TradesStore(store_dir)
            except Exception as e:  # noqa: BLE001
                logger.warning("TradesStore 초기화 실패 — 매매 기록 skip: %s", e)
                return
        # PnL 은 거래소 실현치 우선(pnl_override), 없으면 entry/price 기반 추정.
        # #PR-C: closed-pnl 동기화로 fees/funding 반영된 실제 PnL 기록 가능.
        pnl_usdt: float | None = pnl_override
        if pnl_usdt is None and entry_for_pnl is not None and qty > 0:
            sign = 1.0 if direction is Direction.LONG else -1.0
            pnl_usdt = sign * (price - entry_for_pnl) * qty
        event = TradeEvent(
            ts_ms=int(time.time() * 1000),
            event_type=event_type,
            symbol=self.symbol,
            direction="long" if direction is Direction.LONG else "short",
            price=price,
            qty=qty,
            pnl_usdt=pnl_usdt,
            setup_ts_ms=setup_ts_ms,
            reason=reason,
            context_json=context_json,
            # 2026-05-29: DEMO/LIVE 구분 — UI 가 "유형" 옆 컬럼에 표시.
            mode=self.run_mode,
            # 2026-06-17: 봇 모델 태그 — 매매 기록 [모델] 컬럼 (어느 모델로 매매됐는지).
            # #MMBM 2026-07-21: _execute_setup 이 setup.source 로 갱신한 태그 사용
            # (SB=Origo, MMBM=별도) → 실측 시 두 모델 성과 분리.
            model=self._active_model,
        )
        # 2026-06-09: 매매 이벤트를 fly 로그에도 남긴다 — 진입은 "Execute setup"
        # 으로만 찍히고 청산(SL/TP)은 DB 만 기록돼 로그 모니터링에서 누락됐다.
        # event_type(ENTRY/SL_HIT/TP_HIT/FLIP_*/SYNC_CLOSE)이 로그에 찍혀 필터로 잡힘.
        logger.info(
            "매매 %s | %s %s price=%.4f qty=%.6f pnl=%s",
            event.event_type.value.upper(), self.symbol, event.direction,
            price, qty,
            f"{pnl_usdt:+.2f}" if pnl_usdt is not None else "—",
        )
        try:
            self._trades_store.record(event)
            self._flush_failed_trades()  # 이전 실패분이 있으면 재기록 시도
        except Exception as e:  # noqa: BLE001
            # 2026-06-17 #SYNC-FIX: 기록 실패를 ERROR 로 승격 + 큐 보관 → 다음
            # _record_trade 성공 시 _flush_failed_trades 로 재기록. (기존 warning 만 →
            # 거래소 체결됐는데 DB 영구 누락되던 라이브 불일치 해소.) active_position
            # reset 은 그대로 진행하되 기록만 지연 재시도.
            logger.error("trades record 실패 — 큐 보관 후 재시도 예정: %s", e)
            self._failed_trade_events.append(event)
        # 2026-06-08: 매매 알림 — 연동된 사용자에게 텔레그램 발송(fire-and-forget).
        # 미연동·전송 실패는 콜백 내부에서 흡수. 알림이 매매를 막지 않게.
        # 2026-06-12: RECOVERED(재시작 시 기존 포지션 재인식)는 텔레그램 생략 —
        # 배포/재시작마다 같은 포지션 알림이 반복돼 소음(파트너 보고). 기록(DB)은
        # 유지, 포지션 확인은 UI/전체 포지션(admin)으로.
        # 2026-06-12 추가(파트너): SYNC_CLOSE(거래소 측 청산 사후 동기화)도 생략 —
        # SL/TP 미구분이라 정보 가치 낮고 재시작 직후 몰려서 소음. 기록은 유지.
        if event_type in (TradeEventType.RECOVERED, TradeEventType.SYNC_CLOSE):
            return
        if self.alert_cb is not None and self.user_code:
            try:
                task = asyncio.create_task(self.alert_cb(self.user_code, event))
                # 2026-06-10: fire-and-forget task 예외가 silent 로 묻히던 문제 —
                # done callback 으로 알림 task 실패를 로그에 남겨 진단 가능하게.
                task.add_done_callback(_log_alert_task_exc)
            except RuntimeError:
                pass  # 실행 중 이벤트 루프 없음(동기 테스트) — skip

    def _flush_failed_trades(self) -> None:
        """이전에 기록 실패한 이벤트를 재기록 (#SYNC-FIX, 2026-06-17).

        _record_trade 가 record 성공 직후 호출 — 큐에 쌓인 실패분을 순서대로
        재기록 시도. 여전히 실패하면 큐에 남겨 다음 기회에 재시도.
        """
        if not self._failed_trade_events:
            return
        still_failed: list[TradeEvent] = []
        for ev in self._failed_trade_events:
            try:
                self._trades_store.record(ev)
                logger.info(
                    "trades 큐 재기록 성공: %s %s", ev.event_type.value, ev.symbol,
                )
            except Exception as e:  # noqa: BLE001
                still_failed.append(ev)
                logger.warning("trades 큐 재기록 실패 (유지): %s", e)
        self._failed_trade_events = still_failed

    async def _reconcile_orphan_entries(self) -> None:
        """startup — 청산 누락 ENTRY(orphan)를 거래소 closed-pnl 로 보충 (#RECONCILE).

        재기동(crash) 중 청산된 포지션은 active_position 이 없어 _sync_position_state
        가 SYNC_CLOSE 를 기록하지 못한다 → trades DB 에 ENTRY 만 남는다. 거래소
        closed-pnl 과 setup_ts / 진입시각·방향으로 대조해 누락 청산을 SYNC_CLOSE 로
        채운다. 현재 열려있는 포지션의 ENTRY 는 closed-pnl 에 없어 자동 제외된다.
        """
        if self._trades_store is None:
            store_dir = self.trades_data_dir or _ict_data_dir()
            try:
                self._trades_store = TradesStore(store_dir)
            except Exception as e:  # noqa: BLE001
                logger.warning("reconcile: TradesStore init 실패 — skip: %s", e)
                return
        try:
            events = self._trades_store.all_events()
        except Exception as e:  # noqa: BLE001
            logger.warning("reconcile: all_events 실패 — skip: %s", e)
            return
        # 이 심볼의 청산된 setup_ts 집합 + 청산 안 된 ENTRY 수집.
        close_types = {
            TradeEventType.SL_HIT, TradeEventType.TP_HIT,
            TradeEventType.SYNC_CLOSE, TradeEventType.MANUAL_CLOSE,
            TradeEventType.FLIP_CLOSE,
        }
        closed_ts: set[int] = set()
        entries: list[TradeEvent] = []
        for ev in events:
            if ev.symbol != self.symbol:
                continue
            if ev.event_type in close_types and ev.setup_ts_ms:
                closed_ts.add(ev.setup_ts_ms)
            elif ev.event_type is TradeEventType.ENTRY and ev.setup_ts_ms:
                entries.append(ev)
        orphans = [e for e in entries if e.setup_ts_ms not in closed_ts]
        if not orphans:
            return
        logger.info(
            "reconcile: 청산 누락 ENTRY %d건 — 거래소 closed-pnl 대조 시작 (%s)",
            len(orphans), self.symbol,
        )
        oldest = min(e.ts_ms for e in orphans)
        # 2026-06-17 #SYNC-FIX: 거래소 closed-pnl 조회를 3회 backoff 재시도.
        # 1회 실패 후 바로 return 하면 재기동 중 청산된 orphan 이 영구 누락 →
        # API 일시 지연/네트워크 hiccup 흡수. 최종 실패 시 다음 startup 에서 재시도.
        closed = None
        for attempt in range(3):
            try:
                closed = await self.client.fetch_closed_positions(
                    since_ms=oldest - 60_000, limit=200,
                )
                break
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "reconcile: closed-pnl 조회 실패 (시도 %d/3): %s", attempt + 1, e,
                )
                if attempt < 2:
                    await asyncio.sleep(0.5 * (2 ** attempt))
        if closed is None:
            logger.error("reconcile: closed-pnl 조회 최종 실패 — 다음 startup 재시도")
            return
        tol = self._PNL_MATCH_OPENED_TOLERANCE_MS
        used_opened: set[int] = set()  # cp 1건당 orphan 1건 매칭 (중복 방지).
        filled = 0
        for orphan in orphans:
            match = None
            for cp in closed:
                if getattr(cp, "symbol", None) != self.symbol:
                    continue
                if getattr(cp, "direction", None) != orphan.direction:
                    continue
                cp_opened = int(getattr(cp, "opened_at_ts", 0) or 0)
                if cp_opened in used_opened:
                    continue
                # orphan 진입 ts 와 cp opened ±10분 — 같은 거래 인정.
                if cp_opened > 0 and abs(cp_opened - orphan.ts_ms) > tol:
                    continue
                match = cp
                break
            if match is None:
                continue
            used_opened.add(int(getattr(match, "opened_at_ts", 0) or 0))
            direction = (
                Direction.LONG if orphan.direction == "long" else Direction.SHORT
            )
            exit_px = float(getattr(match, "exit_price", 0.0) or 0.0)
            # 2026-06-12: ENTRY context 의 sl/tp 로 SL/TP 분류 — 보충 청산도
            # 매매 기록 유형 필터(TP/SL)에 잡히게.
            try:
                _ctx = json.loads(orphan.context_json or "{}")
            except (ValueError, TypeError):
                _ctx = {}
            evt_type, cls_reason = self._classify_exchange_close(
                direction, float(orphan.price or 0.0),
                float(_ctx.get("sl") or 0.0), float(_ctx.get("tp") or 0.0),
                exit_px,
            )
            self._record_trade(
                evt_type,
                direction=direction,
                price=exit_px,
                qty=orphan.qty,
                entry_for_pnl=orphan.price,
                setup_ts_ms=orphan.setup_ts_ms,
                reason=f"reconcile: 재기동 중 청산 보충 ({cls_reason})",
                pnl_override=float(getattr(match, "pnl_usd", 0.0) or 0.0),
            )
            filled += 1
        logger.info("reconcile: 청산 누락 %d/%d건 보충 완료 (%s)", filled, len(orphans), self.symbol)

    # 2026-05-29 #PNL-MATCH-FIX: cp.opened_at_ts 와 active_position.entry_ts_ms
    # 허용 차이 (Bybit demo 의 createdTime 정밀도 + 봇 fill 인식 지연 흡수).
    # 진입 ts ±10분 안의 cp 만 같은 거래로 인정 — 이전 거래 record 가 잘못 매칭되는
    # propagation 지연 버그 (#5 케이스) 방지.
    _PNL_MATCH_OPENED_TOLERANCE_MS: ClassVar[int] = 10 * 60 * 1000

    async def _fetch_recent_close(self, last_known: _ActivePosition):
        """state-reset 시점 거래소 closed-pnl 에서 매칭되는 close 회수 (#PR-C / #4).

        2026-05-29 #PNL-MATCH-FIX: Bybit demo propagation 지연으로 *이전* 거래의
        cp 가 응답 가장 최근 자리에 와 잘못 매칭되는 버그 해소.
        진입 (active_position.entry_ts_ms) 이후의 cp 만 후보로, 그 중에서도
        cp.opened_at_ts 가 entry_ts_ms 와 ±10분 이내인 것만 같은 거래로 인정.

        Args:
            last_known: state-reset 직전의 active_position.

        Returns:
            매칭 ClosedPosition. 매칭 실패면 None — 다음 step 의 sync 가 재시도
            (지연된 record 가 propagated 된 후 정확히 박힘).
        """
        # since_ms 는 entry_ts_ms 우선 (없으면 setup_ts_ms fallback, 그것도 없으면 1시간 전).
        # entry 이전 거래의 close 가 응답에 섞이지 않게 가장 엄격한 시점부터.
        if last_known.entry_ts_ms > 0:
            since_ms = last_known.entry_ts_ms
        elif last_known.setup_ts_ms > 0:
            since_ms = last_known.setup_ts_ms
        else:
            since_ms = int(time.time() * 1000) - 3_600_000
        try:
            closed = await self.client.fetch_closed_positions(
                since_ms=since_ms, limit=10,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("closed-pnl 조회 실패: %s", e)
            return None
        if not closed:
            # 2026-05-29 #PNL-MATCH: 진단 로그 — 매칭 실패의 가장 흔한 원인 1.
            # since_ms 가 너무 늦거나 거래소 측 응답 propagation 지연.
            logger.info(
                "closed-pnl 빈 응답 — since_ms=%d (entry_ts=%d setup_ts=%d) want=%s/%s",
                since_ms, last_known.entry_ts_ms, last_known.setup_ts_ms,
                self.symbol, last_known.direction.value,
            )
            return None
        want_dir = "long" if last_known.direction is Direction.LONG else "short"
        entry_ts = last_known.entry_ts_ms
        tol = self._PNL_MATCH_OPENED_TOLERANCE_MS
        for cp in closed:
            cp_sym = getattr(cp, "symbol", None)
            cp_dir = getattr(cp, "direction", None)
            if cp_sym != self.symbol or cp_dir != want_dir:
                continue
            # 2026-05-29 #PNL-MATCH-FIX: cp.opened_at_ts (Bybit createdTime) 가
            # entry_ts_ms 와 ±10분 이내인지 검증 — 이전 거래 record 가 우연히
            # 같은 symbol/direction 으로 잡히는 케이스 차단.
            cp_opened = int(getattr(cp, "opened_at_ts", 0) or 0)
            if entry_ts > 0 and cp_opened > 0 and abs(cp_opened - entry_ts) > tol:
                logger.info(
                    "closed-pnl 후보 시간 불일치 skip — cp_opened=%d entry_ts=%d "
                    "diff=%d ms (>%d) — 이전 거래 record 의심",
                    cp_opened, entry_ts, abs(cp_opened - entry_ts), tol,
                )
                continue
            # 매칭 성공 — 정확한 PnL/exit 기록 가능. 진단용 한 줄.
            logger.info(
                "closed-pnl 매칭 — symbol=%s dir=%s exit=%.4f pnl=%.4f USDT "
                "(cp_opened=%d entry_ts=%d)",
                cp_sym, cp_dir,
                float(getattr(cp, "exit_price", 0.0) or 0.0),
                float(getattr(cp, "pnl_usd", 0.0) or 0.0),
                cp_opened, entry_ts,
            )
            return cp
        # 응답은 있는데 want_dir / symbol / 시간 매칭이 안 됨.
        # 응답 첫 cp 의 symbol/direction 까지 보여 운영자가 패턴 파악 가능.
        # None 반환 시 _sync_position_state 가 pnl_override 없이 추정치로 박는데,
        # 다음 step 에서 propagated 된 후 다시 시도 안 됨 (active_position 이미 None).
        # → 차라리 추정치 없이 None 박는 게 통계 신뢰도 보존 (pnl_override=None 으로 위임).
        first = closed[0]
        logger.info(
            "closed-pnl 매칭 실패 — got=%d cps, want=%s/%s, first=%s/%s "
            "(entry_ts=%d). 추정치 fallback 사용.",
            len(closed),
            self.symbol, want_dir,
            getattr(first, "symbol", "?"), getattr(first, "direction", "?"),
            entry_ts,
        )
        return None

    @staticmethod
    def _classify_exchange_close(
        direction: Direction, entry: float, sl: float, tp: float, close_px: float,
    ) -> tuple[TradeEventType, str]:
        """거래소 측 청산 가격으로 SL/TP/미구분 분류 (2026-06-12 파트너 보고).

        기존엔 ``sl > 0 and tp > 0`` 일 때만 분류해서, 복구 포지션처럼 TP=0
        (거래소에서 TP 주문 미발견)인 경우 전부 '미구분'이 됐다 → 매매 기록의
        TP 필터에 익절이 안 잡힘. SL/TP 를 독립 검사하고, 근접 오차 대신
        방향 기준(SL 슬리피지로 더 나쁘게 / TP 가 더 유리하게 체결돼도 인정)
        으로 판정한다.

        Args:
            direction: 포지션 방향.
            entry: 진입가 (오차 기준).
            sl: 손절가 (0 = 미상).
            tp: 목표가 (0 = 미상).
            close_px: 거래소 closed-pnl 의 실제 청산가.

        Returns:
            (TradeEventType, 사유 문자열).
        """
        unknown = (
            TradeEventType.SYNC_CLOSE, "exchange-side close (SL/TP 미구분)",
        )
        if close_px <= 0 or entry <= 0:
            return unknown
        tol = entry * 0.002  # 트리거가-체결가 괴리 + 수수료 반영 오차
        is_long = direction is Direction.LONG
        hit_tp = tp > 0 and (
            close_px >= tp - tol if is_long else close_px <= tp + tol
        )
        hit_sl = sl > 0 and (
            close_px <= sl + tol if is_long else close_px >= sl - tol
        )
        if hit_tp and hit_sl:
            # SL/TP 가 비정상으로 좁아 겹치면 가까운 쪽으로.
            if abs(close_px - tp) <= abs(close_px - sl):
                hit_sl = False
            else:
                hit_tp = False
        if hit_tp:
            return TradeEventType.TP_HIT, "TP_HIT"
        if hit_sl:
            return TradeEventType.SL_HIT, "SL_HIT"
        return unknown

    async def _sync_position_state(self) -> None:
        """거래소 fetch_position 으로 상태 동기화 + 실제 close 정보 회수 (#PR-C/#3+#4).

        활성 포지션이 거래소측에서 닫혔으면 active_position 리셋. closed-pnl 조회로
        실제 exit_price/PnL 회수하고 가능하면 SL_HIT vs TP_HIT 구분해 정확한
        TradeEvent 로 기록.
        """
        try:
            pos = await self.client.fetch_position(self.symbol)
        except Exception as e:  # noqa: BLE001
            # 2026-05-29 #SILENT-2: 연속 실패 누적 가시화.
            # 1~4회: warning, 5회 누적: ERROR (네트워크/API 장애 알람).
            self._sync_failure_streak += 1
            if self._sync_failure_streak >= 5:
                logger.error(
                    "fetch_position 연속 %d회 실패 (sync_position_state): %s — "
                    "거래소 측 상태와 봇 인식 어긋날 위험. 네트워크/API 키 점검 필요.",
                    self._sync_failure_streak, e,
                )
            else:
                logger.warning(
                    "fetch_position 실패 (%d/5): %s",
                    self._sync_failure_streak, e,
                )
            return
        # 성공 — streak reset + recovery_failed 도 해제 (사후 복원 효과).
        self._sync_failure_streak = 0
        if self._recovery_failed:
            logger.info("recover_position 사후 복원 — sync 성공으로 신규 진입 재허용")
            self._recovery_failed = False
        if pos is not None and float(pos.get("contracts", 0) or 0) != 0:
            # 거래소에 포지션 잔존 — '있음'만 보지 말고 방향·수량이 봇 인식과
            # 일치하는지 검증 (#POS-SYNC 2026-06-06, 04:14 방향 불일치 사건).
            await self._reconcile_open_position(pos)
            return
        # 거래소에 포지션 없음 — 이전 거절 시그니처 리셋(다음 포지션은 재평가).
        self._declined_manual_sig = None
        last_known = self.active_position
        if last_known is None:
            return
        # 거래소 closed-pnl 조회 — 실제 exit_price/PnL.
        cp = await self._fetch_recent_close(last_known)
        if cp is not None:
            close_px = float(getattr(cp, "exit_price", 0.0) or 0.0)
            pnl_usd = float(getattr(cp, "pnl_usd", 0.0) or 0.0)
        else:
            close_px = last_known.entry  # placeholder (조회 실패)
            pnl_usd = 0.0
        # SL/TP 구분 — 2026-06-12: 독립 검사 + 방향 기준 (TP=0 복구 포지션도
        # SL 쪽은 분류되게). 분류 실패만 SYNC_CLOSE 로 남는다.
        evt_type = TradeEventType.SYNC_CLOSE
        close_reason = "exchange-side close (SL/TP 미구분)"
        if cp is not None:
            evt_type, close_reason = self._classify_exchange_close(
                last_known.direction, last_known.entry,
                last_known.stop_loss, last_known.take_profit, close_px,
            )
            # #TRAIL-EXCHANGE: 트레일 무장 포지션의 미구분 청산 = 트레일 스탑
            # 체결(SL 원위치도 TP(5R)도 아닌 중간 가격). Cursus 전례처럼 SL_HIT
            # 유형 + trail_stop 사유로 기록 — FST 는 reason 으로 구분.
            if (
                last_known.trail_armed
                and evt_type is TradeEventType.SYNC_CLOSE
                and close_px > 0
            ):
                evt_type = TradeEventType.SL_HIT
                close_reason = "trail_stop (거래소 트레일링 청산)"
        logger.info(
            "position 종료 — entry=%.4f close=%.4f pnl=%.4f USDT (%s)",
            last_known.entry, close_px, pnl_usd, close_reason,
        )
        # 2026-05-29: 청산 시점 봇 판단 스냅샷 — 봇 개선 분석용. UI 가 진입 vs
        # 청산 두 컨텍스트 비교해 패턴 파악 가능.
        close_ctx = self._build_close_context_json(
            last_known, close_px, pnl_usd, close_reason,
        )
        self._record_trade(
            evt_type,
            direction=last_known.direction,
            price=close_px,
            qty=last_known.qty,
            entry_for_pnl=last_known.entry,
            setup_ts_ms=last_known.setup_ts_ms,
            reason=close_reason,
            pnl_override=pnl_usd if cp is not None else None,
            context_json=close_ctx,
        )
        # 2026-05-28: 학습/복기 dataset JSON sidecar 저장 (best-effort, 실패해도 봇 OK)
        try:
            await self._write_trade_dataset(
                last_known, close_px, pnl_usd, close_reason, cp,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("trade dataset 저장 실패: %s", e)
        self.active_position = None

    # ----- 변경 2: 추세 평가 캐시 ---------------------------------------
    async def _refresh_trend_cache(self) -> None:
        """7개 TF (5m/15m/1h/2h/4h/1d/1w) 추세를 봉 ts 변경 시에만 재평가.

        매 step 호출되지만 TF 마지막 봉 ts 가 캐시와 동일하면 skip → fetch 부담 ↓.
        """
        tfs = ("5m", "15m", "1h", "2h", "4h", "1d", "1w")
        for tf in tfs:
            try:
                df = await self._fetch_ohlcv_tf(tf, 60)
            except Exception as e:  # noqa: BLE001
                logger.debug("trend fetch 실패 (%s): %s", tf, e)
                continue
            if len(df) == 0:
                continue
            last_ts = int(df.index[-1].value // 10**6)
            cached = self._trend_cache.get(tf)
            if cached is not None and cached[0] == last_ts:
                continue
            state = evaluate_trend(df)
            self._trend_cache[tf] = (last_ts, state)

    # ----- 변경 3: HTF FVG map 빌드 + override + flip ------------------
    async def _ensure_htf_fvg_map(self, current_ltf_ts_ms: int) -> list[HtfFvgEntry]:
        """HTF FVG map 캐시 — LTF 봉 (5m 가정) 1개 길이 = 5분 = 300_000ms 마다 재 빌드.

        실패 시 기존 캐시 유지 (없으면 빈 리스트).
        """
        rebuild_interval_ms = 300_000  # 5m TF 봉 길이 기준.
        if (
            self._htf_fvg_map_cache
            and (current_ltf_ts_ms - self._htf_fvg_map_built_at_ms) < rebuild_interval_ms
        ):
            return self._htf_fvg_map_cache
        try:
            new_map = await build_htf_fvg_map(
                self._fetch_ohlcv_tf,
                tfs=self.htf_fvg_tfs,
                fvg_min_size_pct=self.fvg_min_size_pct,
                limit=200,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("HTF FVG map 빌드 실패: %s — 기존 캐시 유지", e)
            return self._htf_fvg_map_cache
        # 변경 7: 재빌드 후에도 동일 FVG 의 touch_count 누적 유지.
        prev_touch: dict[tuple, int] = {
            e.key(): e.touch_count for e in self._htf_fvg_map_cache
        }
        for e in new_map:
            tc = prev_touch.get(e.key())
            if tc:
                e.touch_count = tc
        self._htf_fvg_map_cache = new_map
        self._htf_fvg_map_built_at_ms = current_ltf_ts_ms
        return new_map

    async def _evaluate_htf_override(
        self,
        setup: SilverBulletSetup,
        ltf_df: pd.DataFrame,
    ) -> HtfFvgEntry | None:
        """setup 진입 직전 호출 — 반대 방향 HTF FVG 가중치 합이 LTF setup 가중치를
        넘으면 가장 가까운 반대 HTF FVG 반환 (flip target 후보).

        ``htf_override_mode == "off"`` 면 항상 None.
        """
        if self.htf_override_mode == "off":
            return None
        if len(ltf_df) == 0:
            return None
        current_ts = int(ltf_df.index[-1].value // 10**6)
        current_price = float(ltf_df["close"].iloc[-1])
        htf_map = await self._ensure_htf_fvg_map(current_ts)
        if not htf_map:
            return None
        ltf_weight = TF_WEIGHT.get(self.timeframe, 1)
        # 2026-05-29: HTF override threshold 강화 — 새벽 short bias 고착 회고.
        # 기존: threshold = ltf_weight (5m=1) → 거의 모든 반대 FVG 가 override 트리거.
        # ranging 시장에서 현재가 위/아래 FVG 분리 + threshold 낮음 → 진동 / 불안정.
        # 개선: ltf_weight × 3 + max 6 (4h급 가중치 이상만 의미 있는 차단으로 판단).
        # 효과: ranging 잡음 차단 ↓, 정말 큰 HTF 신호일 때만 override 트리거.
        threshold = max(ltf_weight * 3, 6)
        cands = find_opposite_htf_fvg(
            htf_map,
            ltf_direction="buy" if setup.direction is Direction.LONG else "sell",
            current_price=current_price,
            threshold_weight=threshold,
            max_touch_count=self.htf_fvg_max_touch_count,
        )
        if not cands:
            return None
        # #FLIP-REFINE: threshold(합산 가중치) 통과 후 target 은 1h+ 존만 —
        # "가장 가까운 존"이 15m 이면 TP(2.5R) 한참 전(실측 ~0.4R)에서 승자를
        # 자르던 설계 모순 해소. 1h+ 존이 없으면 flip target 미무장(TP/SL 만).
        for c in cands:
            if c.weight >= _FLIP_TARGET_MIN_WEIGHT:
                return c
        return None

    async def _apply_htf_supporting_boost(
        self,
        setup: SilverBulletSetup,
        ltf_df: pd.DataFrame,
    ) -> None:
        """변형 7 B+A 합성의 A — 같은 방향 HTF FVG 가중치로 ``confluence_score`` 보강.

        엄격 안 (사용자 결정 2026-05-20):
        - LTF LONG → 가격 *아래* unswept bullish HTF FVG 만 지지로 인정
        - LTF SHORT → 가격 *위* unswept bearish HTF FVG 만 저항으로 인정

        계단식 점수 매핑 (가중치 합산 기준):
        - 합산 < 4   → +0 (15m 1개 이하 — 보강 X)
        - 합산 4-9   → +1 (1h~4h 1개 정도)
        - 합산 10-19 → +2 (4h 2개 또는 1d)
        - 합산 20+   → +3 (1d 이상)

        ``htf_override_mode == "off"`` 면 동작 안 함 (가중치 시스템 전체 비활성).
        in-place 로 ``setup.confluence_score`` 가산 + ``setup.confluences`` 디버그 기록.
        """
        if self.htf_override_mode == "off":
            return
        if len(ltf_df) == 0:
            return
        current_ts = int(ltf_df.index[-1].value // 10**6)
        current_price = float(ltf_df["close"].iloc[-1])
        htf_map = await self._ensure_htf_fvg_map(current_ts)
        if not htf_map:
            return
        cands = find_supporting_htf_fvg(
            htf_map,
            ltf_direction="buy" if setup.direction is Direction.LONG else "sell",
            current_price=current_price,
            max_touch_count=self.htf_fvg_max_touch_count,
        )
        if not cands:
            return
        total_weight = sum(e.weight for e in cands)
        # 계단식 A — 사용자 결정 2026-05-20.
        # 2026-06-04: 최저 임계 4 → 2 완화 (파트너 결정). 작은 HTF FVG 만
        # 잡히는 시장에서도 boost +1 받게 → score 2 분포 회복 목적.
        # 상위 단계 (20/10) 는 그대로 — 강한 setup 만 큰 boost.
        if total_weight >= 20:
            boost = 3
        elif total_weight >= 10:
            boost = 2
        elif total_weight >= 2:
            boost = 1
        else:
            boost = 0
        if boost <= 0:
            return
        setup.confluence_score += boost
        setup.confluences.append(
            f"htf_support_weight={total_weight}_boost+{boost}",
        )
        logger.info(
            "HTF supporting boost — %s weight_sum=%d boost+%d (cands=%d)",
            setup.direction.value, total_weight, boost, len(cands),
        )

    async def _maybe_flip(self, ltf_df: pd.DataFrame) -> None:
        """봉 close 보조 flip 검사 — WS/polling 이 놓친 거 catch-up.

        - htf_override_mode != "C" 면 동작 안 함.
        - 같은 LTF 봉 ts 에서는 한 번만 검사.
        - WS 가 이미 같은 봉에 flip 발동했으면 (_flip_done_for_ts) skip.
        """
        if self.htf_override_mode != "C":
            return
        pos = self.active_position
        if pos is None or pos.htf_flip_target is None:
            return
        if len(ltf_df) == 0:
            return
        last_ts = int(ltf_df.index[-1].value // 10**6)
        if last_ts == self._last_flip_check_bar_ts:
            return
        self._last_flip_check_bar_ts = last_ts
        # 변경 7: WS 가 이미 같은 봉에 flip 발동했으면 보조 검사 skip.
        if last_ts == self._flip_done_for_ts:
            return

        last_close = float(ltf_df["close"].iloc[-1])
        target = pos.htf_flip_target
        if not target.contains(last_close):
            return
        await self.handle_htf_flip(last_close, last_ts, target)

    async def handle_htf_flip(
        self,
        trigger_price: float,
        ts_ms: int,
        target: HtfFvgEntry,
    ) -> None:
        """HTF FVG flip 실제 실행 — 청산 → 진입 sequential.

        WS flip watcher 와 step() 보조 검사 양쪽에서 호출되는 공통 경로.

        Args:
            trigger_price: flip 발동 시점 가격 (REST 재확인 끝난 값).
            ts_ms: 발동 시점 ts (LTF 봉 ts 또는 tick ts).
            target: 진입 대상 HTF FVG.
        """
        pos = self.active_position
        if pos is None:
            return
        # 같은 봉에 다시 안 박이게 flag.
        self._flip_done_for_ts = ts_ms

        new_direction = (
            Direction.SHORT if pos.direction is Direction.LONG else Direction.LONG
        )
        exit_side = "sell" if pos.direction is Direction.LONG else "buy"
        new_side = "buy" if new_direction is Direction.LONG else "sell"
        logger.info(
            "HTF flip — ltf_%s_%.2f → htf_%s_%.2f (%s FVG, weight %d)",
            "buy" if pos.direction is Direction.LONG else "sell", pos.entry,
            new_side, trigger_price, target.tf, target.weight,
        )

        # 1) 기존 포지션 시장가 청산 — 응답 검증, 실패 시 1회 retry.
        exit_ok = False
        for attempt in (1, 2):
            try:
                resp = await self.client.place_order(
                    symbol=self.symbol, side=exit_side, qty=pos.qty,
                    price=None, reduce_only=True,
                )
                if isinstance(resp, dict) and resp.get("error"):
                    raise RuntimeError(str(resp.get("error")))
                exit_ok = True
                break
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "flip — 기존 포지션 청산 실패 (시도 %d): %s", attempt, e,
                )
                await asyncio.sleep(0.5)
        if not exit_ok:
            # 2026-05-29 #SILENT-5: flip 청산 최종 실패 가시화 강화.
            # 단순 ERROR 로그 → 봇 일관성 위험 (HTF FVG flip 인식했는데 거래소
            # 측 포지션은 유지 → 신호와 실제 상태 어긋남). 운영자가 즉시 인지
            # 필요. 신규 진입 차단 + 사용자별 거래 기록에도 실패 이벤트 박음.
            logger.error(
                "flip 청산 최종 실패 — 신규 진입 중단 + 사용자 알람 필요. "
                "거래소 상태와 봇 인식 어긋남: symbol=%s direction=%s qty=%.4f",
                self.symbol, pos.direction.value, pos.qty,
            )
            # trades 기록에 alarm 이벤트 — UI / 텔레그램에서 즉시 검색 가능.
            self._record_trade(
                TradeEventType.SYNC_CLOSE,
                direction=pos.direction,
                price=pos.entry,  # placeholder
                qty=pos.qty,
                reason="ALARM: flip 청산 실패 — 거래소 측 포지션 유지 가능",
            )
            return

        # #BUG-2: FLIP_CLOSE 기록 — trigger_price 가 실제 청산 가격에 가장 가까운 추정.
        self._record_trade(
            TradeEventType.FLIP_CLOSE,
            direction=pos.direction,
            price=trigger_price,
            qty=pos.qty,
            entry_for_pnl=pos.entry,
            setup_ts_ms=pos.setup_ts_ms,
            reason=f"htf flip target hit @{target.tf} (weight={target.weight})",
        )

        # #FLIP-REFINE (2026-07-02): 역진입 제거 — 청산(방어)까지만.
        # 실측 flip_open 113건 net -301 USDT·승률 19%·전 TF 적자(robust).
        # 반대 FVG 는 "내 포지션의 위험 신호"로는 유효하나 "반대 진입 근거"로는
        # 낙제 → 청산 후 다음 정규 setup 탐색으로 복귀.
        if not _FLIP_REVERSE_ENABLED:
            self.active_position = None
            logger.info(
                "flip 청산 완료 — 역진입 생략(#FLIP-REFINE), 정규 setup 탐색 복귀",
            )
            return

        # 2) 신규 진입 — entry = trigger_price, SL = FVG 반대쪽 (정통 ICT 단순), TP = min_rr R.
        # Why: sl_buffer_ratio 제거(변형 4 정통화). FVG zone 자체 가장자리를 SL 로.
        new_entry = trigger_price
        last_ts = ts_ms
        if new_direction is Direction.LONG:
            new_sl = target.low
        else:
            new_sl = target.high
        sl_dist = abs(new_entry - new_sl)
        if sl_dist <= 0:
            logger.warning("flip — SL 거리 0 이하, skip")
            self.active_position = None
            return
        if new_direction is Direction.LONG:
            new_tp = new_entry + sl_dist * self.min_rr
        else:
            new_tp = new_entry - sl_dist * self.min_rr

        # qty 재산정 — 기존 _calc_qty 재사용 (confluence_score=0 → base pct).
        try:
            equity = await self._fetch_equity()
        except Exception:  # noqa: BLE001
            equity = 1000.0
        fake_setup = SilverBulletSetup(
            ts_ms=last_ts,
            direction=new_direction,
            window="htf_flip",
            entry=new_entry,
            stop_loss=new_sl,
            take_profit=new_tp,
            risk_reward=self.min_rr,
            fvg=None,  # type: ignore[arg-type]
            confluence_score=0,
        )
        new_qty = self._calc_qty(fake_setup, equity)
        try:
            # #FLIP-SL fix: 일반 진입(#LIVE-4)과 동일하게 SL/TP 를 진입 주문에 동봉하지
            # 않는다. flip 은 시장가(price=None)라 대개 통과하지만, 변동성 큰 flip 순간
            # FVG 경계 SL 이 현재가 너머로 가면 Bybit 10001(StopLoss 방향 검증)로 주문
            # 자체가 거부 → 신규 진입 0. SL/TP 는 체결 후 아래 _ensure_protective_sl 가
            # set_position_tpsl 로 박는다 (체결가 기준이라 방향 유효).
            resp = await self.client.place_order(
                symbol=self.symbol, side=new_side, qty=new_qty,
                price=None, stop_loss=None, take_profit=None,
            )
            if isinstance(resp, dict) and resp.get("error"):
                raise RuntimeError(str(resp.get("error")))
        except Exception as e:  # noqa: BLE001
            logger.error(
                "flip — 신규 진입 실패: %s — 포지션 없는 상태 (봇 가동 유지)", e,
            )
            self.active_position = None
            return

        # 2026-05-28: flip 케이스도 진입 컨텍스트 박음 — 학습/복기 dataset 정합.
        _flip_entry_equity = 0.0
        try:
            _flip_entry_equity = float(await self._fetch_equity())
        except Exception:  # noqa: BLE001
            pass
        self.active_position = _ActivePosition(
            direction=new_direction,
            entry=new_entry,
            stop_loss=new_sl,
            take_profit=new_tp,
            qty=new_qty,
            setup_ts_ms=last_ts,
            htf_flip_target=None,  # flip 완료 — 같은 target 재발동 방지.
            ltf_weight=TF_WEIGHT.get(self.timeframe, 1),
            entry_ts_ms=int(time.time() * 1000),
            context_json=f'{{"source":"flip","htf_target_tf":"{target.tf}","htf_target_weight":{target.weight}}}',
            equity_at_entry=_flip_entry_equity,
            tp1_price=self._calc_tp1(new_entry, new_sl, new_direction),
        )
        # #BUG-2: FLIP_OPEN 기록 — 반대 방향 신규 진입.
        self._record_trade(
            TradeEventType.FLIP_OPEN,
            direction=new_direction,
            price=new_entry,
            qty=new_qty,
            setup_ts_ms=last_ts,
            reason=f"htf flip into {target.tf} (weight={target.weight})",
        )
        # #FLIP-TP: flip 진입도 일반 진입과 동일하게 SL+TP 를 거래소에 함께 박는다.
        # 기존엔 modify_stop_loss 로 SL 만 걸어 TP conditional 이 거래소에 없었음 →
        # 봇이 죽으면 TP 미실현 + SYNC_CLOSE 분류 오차. _ensure_protective_sl 가
        # set_position_tpsl 로 SL+TP 동시 적용, 실패 시 무SL 방치 금지로 비상청산.
        await self._ensure_protective_sl(new_tp, sl_dist)

    # ============================================================
    # 2026-05-28: 거래 학습/복기 dataset — per-trade JSON sidecar
    # ============================================================

    async def _write_trade_dataset(
        self,
        pos: _ActivePosition,
        close_px: float,
        pnl_usd: float,
        classification: str,
        closed_position: Any | None,  # ClosedPosition or None
    ) -> None:
        """청산 직후 학습/복기용 dataset JSON 한 건 저장.

        파일: <data_dir>/trades_dataset/<exit_iso>__<direction>__<class>.json
        내용:
          - entry: 진입 ts/시각(UTC/KST/NY) + 가격/SL/TP/qty + context_json + equity
          - exit:  청산 ts/시각 + close_px + pnl + duration + classification
          - ohlcv_snapshot: 1h/5m 봉 window (entry 전후) — cache 활용 (비용 적음)
          - settings_snapshot: 그 시점 봇 설정 (min_rr/min_confluence/leverage 등)
          - bot_version, license_type

        실패해도 봇은 안 죽음 (try/warn 처리 — 호출자가 감쌌음).
        """
        import json as _json
        from datetime import UTC, datetime
        from zoneinfo import ZoneInfo

        from aurora_ict import __version__ as _ict_version

        exit_ts_ms = int(time.time() * 1000)
        entry_ts_ms = pos.entry_ts_ms or 0
        duration_sec = (exit_ts_ms - entry_ts_ms) // 1000 if entry_ts_ms > 0 else None

        def _iso(ts_ms: int, tz: Any) -> str:
            return datetime.fromtimestamp(ts_ms / 1000.0, tz=tz).isoformat()

        utc = UTC
        kst = ZoneInfo("Asia/Seoul")
        ny = ZoneInfo("America/New_York")

        # 진입 context_json (저장된 raw string) → dict 으로 unpack (실패 시 raw 그대로)
        entry_context: Any = None
        if pos.context_json:
            try:
                entry_context = _json.loads(pos.context_json)
            except Exception:  # noqa: BLE001
                entry_context = pos.context_json  # raw fallback

        # 청산 시점 equity (best-effort)
        exit_equity = 0.0
        try:
            exit_equity = float(await self._fetch_equity())
        except Exception:  # noqa: BLE001
            pass

        # OHLCV snapshot — 1h 는 공유 prefetch cache 활용, 5m 은 차트 캐시 대상이
        # 아니라(15m·1h만) close 시점 직접 fetch (저빈도라 비용 미미, 학습 데이터셋의
        # 5m 그래뉼래러티 보존). 실패 시 빈 list.
        ohlcv_1h = self._shared_ohlcv.get(self.symbol, "1h") or []
        try:
            ohlcv_5m = await self.client.fetch_ohlcv(self.symbol, "5m", 300)
        except Exception:  # noqa: BLE001
            ohlcv_5m = []
        # window: entry 50봉 전 ~ exit + 5봉
        def _window(rows: list[list[Any]], entry_ms: int, exit_ms: int) -> list[list[Any]]:
            if not rows or entry_ms <= 0:
                return []
            # ts 오름차순. entry 봉 idx 찾기 (가장 가까운).
            ent_idx = 0
            for i, r in enumerate(rows):
                if r[0] <= entry_ms:
                    ent_idx = i
                else:
                    break
            start = max(0, ent_idx - 50)
            # exit 봉 idx 찾기
            exit_idx = len(rows) - 1
            for i in range(ent_idx, len(rows)):
                if rows[i][0] <= exit_ms:
                    exit_idx = i
                else:
                    break
            end = min(len(rows), exit_idx + 6)
            return [list(r) for r in rows[start:end]]

        snap_1h = _window(ohlcv_1h, entry_ts_ms, exit_ts_ms)
        snap_5m = _window(ohlcv_5m, entry_ts_ms, exit_ts_ms)

        # PnL % 계산 (best-effort)
        pnl_pct_on_margin = None
        if pos.equity_at_entry > 0:
            pnl_pct_on_margin = (pnl_usd / pos.equity_at_entry) * 100.0

        # settings snapshot — 봇 인스턴스 필드 그대로 (학습 시 분류 기준)
        settings_snap = {
            "min_rr": self.min_rr,
            "min_confluence": self.min_confluence,
            "disable_time_filter": self.disable_time_filter,
            "leverage": self.leverage,
            "position_pct_base": self.position_pct_base,
            "position_pct_max": self.position_pct_max,
            "position_pct_step": self.position_pct_step,
            "fvg_min_size_pct": self.fvg_min_size_pct,
            "trail_buffer_ratio": self.trail_buffer_ratio,
            "timeframe": self.timeframe,
        }

        dataset = {
            "version": "v1",
            "bot_version": _ict_version,
            "symbol": self.symbol,
            "direction": "long" if pos.direction is Direction.LONG else "short",
            "classification": classification,
            "entry": {
                "ts_ms": entry_ts_ms,
                "iso_utc": _iso(entry_ts_ms, utc) if entry_ts_ms > 0 else None,
                "iso_kst": _iso(entry_ts_ms, kst) if entry_ts_ms > 0 else None,
                "iso_ny": _iso(entry_ts_ms, ny) if entry_ts_ms > 0 else None,
                "entry_px": pos.entry,
                "sl": pos.stop_loss,
                "tp": pos.take_profit,
                "qty": pos.qty,
                "setup_ts_ms": pos.setup_ts_ms,
                "rr_target": (
                    abs(pos.take_profit - pos.entry) / abs(pos.stop_loss - pos.entry)
                    if pos.stop_loss != pos.entry else None
                ),
                "context": entry_context,
                "equity_at_entry_usdt": pos.equity_at_entry,
            },
            "exit": {
                "ts_ms": exit_ts_ms,
                "iso_utc": _iso(exit_ts_ms, utc),
                "iso_kst": _iso(exit_ts_ms, kst),
                "iso_ny": _iso(exit_ts_ms, ny),
                "close_px": close_px,
                "classification": classification,
                "pnl_usdt": pnl_usd,
                "pnl_pct_on_equity_entry": pnl_pct_on_margin,
                "duration_sec": duration_sec,
                "equity_at_exit_usdt": exit_equity,
                "from_closed_pnl_api": closed_position is not None,
            },
            "ohlcv_snapshot": {
                "1h_entry50_to_exit5": snap_1h,
                "5m_entry50_to_exit5": snap_5m,
            },
            "settings_snapshot": settings_snap,
        }

        out_dir = _ict_data_dir() / "trades_dataset"
        out_dir.mkdir(parents=True, exist_ok=True)
        # 파일명 — exit_iso + dir + class, 정렬 친화적 + 중복 방지
        exit_iso = datetime.fromtimestamp(exit_ts_ms / 1000.0, tz=utc).strftime(
            "%Y%m%dT%H%M%SZ",
        )
        direction_str = "long" if pos.direction is Direction.LONG else "short"
        # classification 에 공백/특수문자 있을 수 있어 sanitize
        class_safe = "".join(c if c.isalnum() else "_" for c in classification)[:40]
        out_path = out_dir / f"{exit_iso}__{direction_str}__{class_safe}.json"
        out_path.write_text(
            _json.dumps(dataset, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.info(
            "trade dataset 저장 — %s (pnl=%.4f duration=%ss)",
            out_path.name, pnl_usd, duration_sec,
        )

    # ============================================================
    # 2026-05-27: UI 차트 OHLCV cache + prefetch (TF 토글 즉시 응답용)
    # ============================================================

    # TF 별 prefetch 한도 — 2026-07-22 파트너 지시로 차트 TF 를 15m·1h 만 유지
    # (나머지 제거). 봇당 캐시가 18.6만봉→2만봉(~9배↓)로 줄어 8GB 에서 동시 봇 상한
    # 을 안전하게 확대(메모리 여유 확보). 트레이드(5m)는 이 캐시와 무관한 별도 fetch.
    # app.js CANDLE_LIMIT 와 정합 유지 필수. 값=합리적 스크롤 여유(기본뷰는 ~300봉).
    _UI_OHLCV_TF_LIMITS: ClassVar[dict[str, int]] = {
        "15m": 10000, "1h": 10000,
    }
    # /ict/ohlcv 가 cache 갱신 트리거할 때 받을 봉 수 (마지막 N봉만 refresh).
    _UI_OHLCV_REFRESH_TAIL: ClassVar[int] = 200

    def _get_ohlcv_lock(self, tf: str) -> asyncio.Lock:
        """(symbol, tf) 별 cache 갱신 lock — 공유 캐시에 위임 (심볼 단위 직렬화)."""
        return self._shared_ohlcv.get_lock(self.symbol, tf)

    async def _prefetch_all_ohlcv_tfs(self) -> None:
        """봇 시작 직후 background — 모든 UI TF prefetch.

        각 TF 별로 _UI_OHLCV_TF_LIMITS 만큼 fetch_ohlcv → 공유 캐시(symbol,tf) 채움.
        같은 심볼을 다른 봇/유저가 이미 채웠으면 skip(심볼당 1회만 fetch). 실패 시
        해당 TF skip (다른 TF prefetch 계속). 작은 봉수 TF 부터 채워 즉시 응답 가능.
        """
        # 작은 봉 수부터 먼저 — 빨리 끝나는 TF 부터 cache 채워 즉시 응답 가능.
        tf_order = sorted(self._UI_OHLCV_TF_LIMITS.items(), key=lambda kv: kv[1])
        for tf, limit in tf_order:
            try:
                async with self._get_ohlcv_lock(tf):
                    if self._shared_ohlcv.has(self.symbol, tf):
                        continue  # 다른 봇/유저가 같은 심볼 이미 채웠으면 skip (공유)
                    rows = await self.client.fetch_ohlcv(self.symbol, tf, limit)
                    if rows:
                        self._shared_ohlcv.set(self.symbol, tf, list(rows))
                        logger.info(
                            "OHLCV prefetch 완료 %s %s — %d봉",
                            self.symbol, tf, len(rows),
                        )
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("OHLCV prefetch %s 실패: %s — skip", tf, e)

    async def _refresh_ohlcv_cache_recent(self, tf: str) -> None:
        """매 UI polling 시 background — 마지막 N봉만 fetch + cache merge.

        cache 가 풀로 채워진 후 호출. 새 봉 추가 + 진행 중 봉 ts update 처리
        (ts 기준 dict merge). limit=200봉 정도면 빠름 (단일 호출).
        """
        try:
            lock = self._get_ohlcv_lock(tf)
            # 이미 누가 refresh 중이면 대기 X 그냥 skip (concurrent polling 중복 방지)
            if lock.locked():
                return
            async with lock:
                new_rows = await self.client.fetch_ohlcv(
                    self.symbol, tf, self._UI_OHLCV_REFRESH_TAIL,
                )
                if not new_rows:
                    return
                cache = self._shared_ohlcv.get(self.symbol, tf) or []
                # ts → row 로 merge (새 봉 add + 진행 중 봉 update)
                by_ts = {r[0]: r for r in cache}
                for r in new_rows:
                    by_ts[r[0]] = r
                merged = sorted(by_ts.values(), key=lambda r: r[0])
                self._shared_ohlcv.set(self.symbol, tf, merged)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.debug("OHLCV refresh %s 실패: %s", tf, e)

    async def get_ohlcv_cached(
        self, tf: str, requested_limit: int,
    ) -> list[list[Any]]:
        """UI /ict/ohlcv 용 — cache hit 우선, miss 면 sync fetch.

        - cache 풀로 채워졌으면 즉시 cache 반환 + background incremental refresh.
        - cache 없으면 작은 limit (max 200봉) 로 sync fetch — UI 빠른 첫 응답
          보장. prefetch task 가 background 로 풀 limit 까지 채움.

        Args:
            tf: timeframe (e.g. "1h").
            requested_limit: UI 가 원하는 봉 수.

        Returns:
            ts 오름차순 rows. requested_limit 못 채워도 있는 만큼 반환.
        """
        cache = self._shared_ohlcv.get(self.symbol, tf)
        if cache and len(cache) > 0:
            # cache hit — background refresh 트리거 (await 안 함)
            asyncio.create_task(self._refresh_ohlcv_cache_recent(tf))
            return cache[-requested_limit:]
        # cache miss — 작은 limit 으로 sync fetch + 풀 prefetch 백그라운드 트리거
        quick_limit = min(requested_limit, 200)
        async with self._get_ohlcv_lock(tf):
            existing = self._shared_ohlcv.get(self.symbol, tf)
            if existing is not None:  # 다른 task/봇 가 채웠을 수도 (심볼 공유)
                return existing[-requested_limit:]
            try:
                rows = await self.client.fetch_ohlcv(self.symbol, tf, quick_limit)
            except Exception as e:  # noqa: BLE001
                logger.warning("get_ohlcv_cached sync fetch %s 실패: %s", tf, e)
                return []
            if rows:
                self._shared_ohlcv.set(self.symbol, tf, list(rows))
            # 풀 prefetch — 이 TF 만 추가로 받음
            target_limit = self._UI_OHLCV_TF_LIMITS.get(tf, requested_limit)
            if target_limit > quick_limit:
                asyncio.create_task(self._prefetch_single_tf(tf, target_limit))
            return list(rows[-requested_limit:]) if rows else []

    async def _prefetch_single_tf(self, tf: str, limit: int) -> None:
        """단일 TF prefetch — get_ohlcv_cached 의 background 트리거용."""
        try:
            async with self._get_ohlcv_lock(tf):
                rows = await self.client.fetch_ohlcv(self.symbol, tf, limit)
                if rows:
                    self._shared_ohlcv.set(self.symbol, tf, list(rows))
                    logger.info(
                        "OHLCV background prefetch %s %s — %d봉",
                        self.symbol, tf, len(rows),
                    )
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.warning("OHLCV background prefetch %s 실패: %s", tf, e)


__all__ = [
    "BotIctInstance",
    "BotState",
    "ExchangeClientProtocol",
]
