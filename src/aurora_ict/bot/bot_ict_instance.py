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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, Protocol
from zoneinfo import ZoneInfo

import pandas as pd

from aurora_ict.bot.structure_trail import compute_structure_trail
from aurora_ict.indicators.daily_bias import compute_daily_bias
from aurora_ict.indicators.dol import compute_dol
from aurora_ict.indicators.liquidity import detect_liquidity_sweeps
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
from aurora_ict.strategy.multi_tf_bias import (
    combine_bias,
    compute_bias_from_df,
    htf_pair,
)
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup
from aurora_ict.strategy.trend_state import TrendState, evaluate_trend
from aurora_ict.timing.killzone import classify_killzone

logger = logging.getLogger(__name__)

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

# DOL 역방향 진입 감점 (#3 보완). 지배적 draw 와 반대인 setup 의 confluence_score 를
# 이만큼 깎아 B+ 게이트(min_confluence)에서 걸러지게 함. 2 = 보통 setup 은 컷, A급만 통과.
_DOL_COUNTER_PENALTY = 2


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
    # FVG 이후 N 봉 안에 retest 없으면 진입 skip. 1h → 10시간.
    setup_stale_bars: int = 10
    # LuxAlgo SB Strict mode — True 면 FVG mean threshold 까지 retrace 된 setup 만 진입.
    require_retrace: bool = False
    # Setup 시간 윈도우 확장 — True 면 Killzone 전체 (Asian/London/NY_AM/Close/PM),
    # False 면 Silver Bullet 1시간 윈도우만 (NY 3-4am/10-11am/2-3pm).
    expand_to_killzone: bool = True
    # 24h 매매 — True 면 SB / Killzone 시간 필터 완전 skip. expand_to_killzone 보다 우선.
    disable_time_filter: bool = True

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
    # max_entry_distance_pct: setup.entry 와 현재가 차이가 이 비율 초과면 setup skip.
    # 0 = 비활성. 너무 멀리 박힌 limit 의 미체결 대기 시간 회피.
    max_entry_distance_pct: float = 0.0
    # heartbeat_interval_sec: step loop 살아있음 INFO 로그 주기. 0 = 비활성.
    heartbeat_interval_sec: int = 0
    # #SAFETY-1: 자본 대비 % 일일 손실 한도. 0 = 비활성. NY local 자정 reset.
    daily_loss_limit_pct: float = 0.0

    # HTF EMA bias 필터 — multi_tf 와 별개. 진입 직전 htf_ema_bias_tf EMA20 vs 가격
    # 비교 → setup.direction 이 bias 와 반대면 진입 skip.
    htf_ema_bias_enabled: bool = False
    htf_ema_bias_tf: str = "1h"
    htf_ema_bias_period: int = 20

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

    state: BotState = field(default=BotState.STOPPED)
    active_position: _ActivePosition | None = field(default=None)
    # #LIVE-1 fix: marketable limit entry 미체결 대기 상태 (체결되면 active_position 으로 승격).
    _pending_entry: _PendingEntry | None = field(default=None)
    _task: asyncio.Task[None] | None = field(default=None)
    _last_setup_ts_ms: int = field(default=0)  # 동일 setup 중복 진입 방지
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
    # #SAFETY-1 daily loss limit 추적 — NY local 자정 기준 reset.
    _today_realized_pnl_usdt: float = field(default=0.0)
    _today_date_str: str = field(default="")  # "YYYY-MM-DD" NY local
    _today_start_equity: float = field(default=0.0)
    _daily_limit_hit: bool = field(default=False)

    # 2026-05-28 파트너 요청 — fly.io logs 에서 봇 의사결정 흐름 보이게.
    # 같은 값이 매 step 반복 출력 안 되도록 "직전 값" 캐시 — 변화 시에만 1줄 INFO.
    # 봇 동작 영향 0 (로깅 가시성 전용).
    _last_logged_htf_ema_bias: str = field(default="")  # "bullish"/"bearish"/"neutral"/""
    _last_logged_htf_fvg_summary: str = field(default="")  # "bull_w=10 bear_w=4 n=8" 형태
    _last_logged_dol_draw: str = field(default="")  # "long"/"short"/"none"/""
    _last_logged_trend_summary: str = field(default="")  # trend_cache 직렬화 핑거프린트

    # 2026-05-27 파트너 요청 — UI 차트 TF 토글 시 로딩 없이 즉시 응답.
    # 봇 start 시 background prefetch 로 모든 TF 채워두고, /ict/ohlcv 가
    # cache 사용. polling 시 마지막 N봉만 incremental refresh.
    # 자료: dict[tf, list[[ts_ms, o, h, l, c, v], ...]] ts 오름차순.
    _ohlcv_cache: dict[str, list[list[Any]]] = field(default_factory=dict)
    # TF 별 동시 갱신 방지 — lazy init (이벤트 루프 필요).
    _ohlcv_cache_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    # prefetch background task 핸들 (취소용).
    _prefetch_task: asyncio.Task[None] | None = field(default=None)

    # 2026-05-29 #SILENT-1~5: 조용한 오류 가시화 — 운영 중 어디가 실패하는지
    # judgment 응답 / 로그에서 즉시 확인 가능하게.
    # recovery_failed: 봇 시작 시 거래소 측 fetch_position 실패 (포지션 복원 불가).
    # 다음 step 에서 재시도 가능하도록 flag 유지. True 인 동안은 신규 진입 차단.
    _recovery_failed: bool = field(default=False)
    # 연속 fetch_position 실패 카운트 — 5회 누적 시 ERROR (네트워크/API 장애 의심).
    _sync_failure_streak: int = field(default=0)
    # 진입 주문 실패 카운트 (place_order) — 누적 시 운영자 점검 신호.
    _order_failure_count: int = field(default=0)
    # SL/TP 박기 실패 카운트 — 무SL 상태 가시화. step 마다 재시도.
    _tpsl_failure_streak: int = field(default=0)
    # 가장 최근 진입 setup direction 들 (recent 10) — judgment 응답에 노출하여
    # "long 비율 우세인데 short 만 진입" 같은 의문을 즉시 해소.
    _recent_setup_directions: list[str] = field(default_factory=list)

    async def start(self) -> None:
        """봇 기동 (background task 생성).

        시작 시 거래소 측 활성 포지션 fetch → active_position 복원.
        봇 재시작 시 거래소 측 포지션 인식 못해서 중복 진입 박는 위험 회피.
        """
        if self.state is BotState.RUNNING:
            logger.info("BotIctInstance %s 이미 실행 중", self.symbol)
            return
        await self._recover_position_from_exchange()
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
        self.active_position = _ActivePosition(
            direction=direction,
            entry=entry_price,
            stop_loss=sl,
            take_profit=tp,
            qty=contracts,
            setup_ts_ms=0,  # recovery — 원 setup ts_ms 알 수 없어 0
        )
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
        # P1-2: 거래소 측 SL 이 없는(=0) 채 복구되면 무SL 포지션 → 보호 SL 적용 (안 되면 청산).
        if sl <= 0:
            logger.warning("recover: SL 없는 포지션 — 보호 SL 적용 시도 %s", self.symbol)
            await self._ensure_protective_sl(tp if tp > 0 else None, 0.0)

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
                await self.client.cancel_all_orders(self.symbol)
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
        equity_now = await self._fetch_equity()
        self._maybe_reset_daily_pnl(equity_now)
        await self._sync_today_realized_pnl()

        df = await self._fetch_ohlcv()
        htf_bias = await self._compute_htf_bias()
        daily_bias = await self._compute_daily_bias(df)
        bias = self._combine_with_daily(htf_bias, daily_bias)

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
        )

        # 추세 평가 캐시 갱신 (현재는 로깅용, 향후 가중치 확장 여지).
        await self._refresh_trend_cache()

        # 진입 중인 position이 있으면 신규 진입은 막고 상태만 동기화 + trail tick + flip.
        if self.active_position is not None:
            if self.enable_trail:
                await self._tick_trail(df)
            await self._maybe_flip(df)
            await self._sync_position_state()
            return signal

        if not signal.is_actionable or signal.setup is None:
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

        # 2026-05-29 #SILENT-1: 복원 실패 상태에서는 신규 진입 차단.
        # 거래소 측 활성 포지션이 있는데 봇이 인식 못 하면 중복 진입 위험.
        # _sync_position_state 가 성공하면 _recovery_failed 가 False 로 풀린다.
        if self._recovery_failed:
            logger.warning(
                "setup skip — 거래소 측 포지션 복원 실패 상태. sync 성공 전까지 신규 진입 차단.",
            )
            return signal

        # 동일 setup으로 재진입 방지 (중복 주문 X)
        if signal.setup.ts_ms == self._last_setup_ts_ms:
            # 2026-05-28: 중복 setup skip — 빈도 매우 낮음 (같은 봉 안에서만).
            logger.info(
                "setup skip | reason=duplicate_ts ts_ms=%d", signal.setup.ts_ms,
            )
            return signal

        if not await self._passes_htf_ema_bias(signal.setup.direction):
            # _passes_htf_ema_bias 내부에서 이미 INFO 로그 박힘 (line 743) — 추가 X.
            self._last_setup_ts_ms = signal.setup.ts_ms
            return signal

        # 변경 3: HTF FVG override — 진입 직전 반대 방향 HTF FVG 가중치 평가 (flip target).
        htf_target = await self._evaluate_htf_override(signal.setup, df)
        # 변형 7 B+A 합성 (A): 같은 방향 HTF FVG → confluence_score 보강.
        # qty 산정 (_calc_qty) 에서 confluence_score 가 사용되므로 boost 가 _execute_setup
        # 이전에 적용되어야 효과 발생. _evaluate_htf_override 직후 호출.
        await self._apply_htf_supporting_boost(signal.setup, df)
        # #3 보완: Draw on Liquidity 역방향 진입은 confluence 감점 (게이트 전에 적용).
        self._apply_dol_bias(signal.setup, df)
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
                self._last_setup_ts_ms = signal.setup.ts_ms
                return signal
        if self.htf_override_mode == "A" and htf_target is not None:
            logger.info(
                "HTF override(A) 진입 차단 — setup=%s 반대 HTF FVG=%s",
                signal.setup.direction.value, htf_target.tf,
            )
            self._last_setup_ts_ms = signal.setup.ts_ms
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
                    self._last_setup_ts_ms = signal.setup.ts_ms
                    return signal
                if (not is_long) and (bull_w / bear_w) >= ratio_thr:
                    logger.info(
                        "HTF/LTF 방향 충돌 skip — bull_w=%d > bear_w=%d * %.2f, "
                        "LTF setup=short",
                        bull_w, bear_w, ratio_thr,
                    )
                    self._last_setup_ts_ms = signal.setup.ts_ms
                    return signal

        await self._execute_setup(signal.setup, htf_flip_target=htf_target)
        self._last_setup_ts_ms = signal.setup.ts_ms
        return signal

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
            # 동일 HTF setup 으로 재진입 방지.
            if htf_active.setup.ts_ms == self._last_setup_ts_ms:
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
                self._last_setup_ts_ms = htf_active.setup.ts_ms
                return no_action
            await self._execute_setup(setup)
            self._last_setup_ts_ms = htf_active.setup.ts_ms
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

    async def _fetch_ohlcv_tf(self, tf: str, limit: int) -> pd.DataFrame:
        """임의 timeframe OHLCV fetch + DataFrame 변환."""
        rows = await self.client.fetch_ohlcv(self.symbol, tf, limit)
        df = pd.DataFrame(
            rows,
            columns=["ts_ms", "open", "high", "low", "close", "volume"],
        )
        df.index = pd.DatetimeIndex(pd.to_datetime(df["ts_ms"], unit="ms", utc=True))
        df = df[["open", "high", "low", "close", "volume"]]
        return df

    async def _passes_htf_ema_bias(self, direction: Direction) -> bool:
        """HTF EMA bias 필터 — setup 방향이 EMA bias 와 일치할 때만 True.

        htf_ema_bias_enabled=False 면 항상 True (필터 비활성).
        OHLCV fetch / EMA 계산 실패 시 안전하게 진입 허용 (True 반환).
        """
        if not self.htf_ema_bias_enabled:
            return True
        # 변경 4: override 모드 활성이면 ema_bias 는 의미 없음 (override 가 강력).
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

    async def _execute_setup(
        self,
        setup: SilverBulletSetup,
        htf_flip_target: HtfFvgEntry | None = None,
    ) -> None:
        """setup 한 건을 실제 주문으로 실행 (#LIVE-1 fix: marketable limit + SL/TP 동봉).

        - entry = 현재가 바로 앞 marketable limit (슬리피지 0). SL/TP 를 entry 주문에
          동봉 → 체결 시 거래소가 포지션에 conditional 적용 (단일 TP, ICT 정통).
        - 즉시 체결되면 active_position 확정. 미체결이면 ``_pending_entry`` 등록 →
          step 의 ``_check_pending_entry`` 가 체결 승격 / TTL(10분) 만료 취소 추적.
        - use_market_entry=True 면 레거시 즉시 시장가 (slippage 발생, 비권장).
        """
        side = "buy" if setup.direction is Direction.LONG else "sell"

        # #SAFETY-1: 일일 손실 한도 도달 시 새 진입 차단 (active position 은 유지).
        # equity fetch 가 _execute_setup 본체에서도 필요해 한 번 미리 가져와 baseline.
        equity_now = await self._fetch_equity()
        self._maybe_reset_daily_pnl(equity_now)
        if self._is_daily_loss_limit_hit():
            logger.info(
                "setup skip (#SAFETY-1) — daily loss limit %.2f%% hit "
                "(today_pnl=%.2fUSDT / start=%.2f)",
                self.daily_loss_limit_pct,
                self._today_realized_pnl_usdt, self._today_start_equity,
            )
            return

        # max_sl_distance_pct skip (비정상 큰 SL 차단).
        if self.max_sl_distance_pct > 0 and setup.entry > 0:
            sl_dist_pct = abs(setup.entry - setup.stop_loss) / setup.entry
            if sl_dist_pct > self.max_sl_distance_pct:
                logger.info(
                    "setup skip — SL 거리 %.4f%% > max %.4f%% (entry=%.4f sl=%.4f)",
                    sl_dist_pct * 100, self.max_sl_distance_pct * 100,
                    setup.entry, setup.stop_loss,
                )
                return

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
        qty = self._calc_qty(setup, equity)
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
                "place_order 실패 #%d — side=%s qty=%.4f price=%s setup_ts=%d: %s",
                self._order_failure_count, side, qty, entry_price,
                setup.ts_ms if hasattr(setup, "ts_ms") else 0, e,
            )
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
            self._pending_entry = None
            return False
        # 미체결 — TTL 만료 체크.
        now_ms = int(time.time() * 1000)
        if now_ms - pe.placed_ts_ms >= self.entry_limit_ttl_sec * 1000:
            try:
                await self.client.cancel_all_orders(self.symbol)
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
            await self.client.cancel_all_orders(self.symbol)
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

    async def _emergency_close(self) -> None:
        """위험(무SL 등) 상황에서 active_position 을 시장가 reduce_only 로 청산.

        2026-05-29 #SILENT-4: 비상청산 자체가 실패하면 active_position 을 None
        으로 만들면 안 된다 — 거래소에 포지션이 남아있는데 봇이 "닫혔다고 인식"
        하면 더 큰 risk (중복 진입 / SL 없는 포지션 방치). 실패 시 ERROR 로
        알람 + active_position 유지 → 다음 step 의 sync 에서 재확인 + 필요 시
        재시도 가능.
        """
        pos = self.active_position
        if pos is None:
            return
        close_side = "buy" if pos.direction is Direction.SHORT else "sell"
        try:
            await self.client.place_order(
                self.symbol, side=close_side, qty=pos.qty,
                price=None, reduce_only=True,
            )
            self._record_trade(
                TradeEventType.MANUAL_CLOSE,
                direction=pos.direction,
                price=pos.entry,
                qty=pos.qty,
                reason="SL 적용 실패 비상청산 (무SL 방지)",
            )
            logger.info(
                "비상청산 완료 — %s %s qty=%.4f",
                self.symbol, pos.direction.value, pos.qty,
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

    async def _fetch_equity(self) -> float:
        """가용 자산 (USDT equity) 조회.

        Bybit V5 ccxt format = ``{"USDT": {"total": ...}, ...}``.
        fallback 1000.0 (테스트/오류 대비).
        """
        try:
            bal = await self.client.fetch_balance()
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_balance 실패: %s — fallback $1000", e)
            return 1000.0
        if not isinstance(bal, dict):
            return 1000.0
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
        return 1000.0

    def _calc_qty(self, setup: SilverBulletSetup, equity: float) -> float:
        """진입 qty 계산 — confluence_score 단계별 notional sizing.

        pct = min(base + step * score, max)
        margin = equity * pct/100  → leveraged notional = margin * leverage
        qty = leveraged notional / entry_price
        """
        score = max(0, setup.confluence_score)
        pct = min(
            self.position_pct_base + self.position_pct_step * score,
            self.position_pct_max,
        )
        margin = equity * (pct / 100.0)
        notional = margin * self.leverage
        if setup.entry <= 0:
            return 0.0
        qty = notional / setup.entry
        # Bybit BTC 최소 주문수량 0.001 — 미달 시 floor 가 아니라 skip (작은 잔고에서
        # 의도 notional 초과 박는 회귀 회피, 호출처에서 qty 0 이하 skip 분기 활용).
        if qty < 0.001:
            return 0.0
        return qty

    def _maybe_reset_daily_pnl(self, equity_now: float) -> None:
        """NY local 자정 기준 일일 누적 손익 reset (#SAFETY-1).

        매일 새 거래일이 시작하면 ``_today_realized_pnl_usdt`` 와 ``_today_start_equity``
        를 갱신하고 ``_daily_limit_hit`` flag 풀어줌. ICT 정통 일일 boundary 정합.

        Args:
            equity_now: 현재 가용 자산 (USDT). 새 날짜 시작 시 baseline 으로 박힘.
        """
        ny_date = datetime.now(UTC).astimezone(_NY_TZ).strftime("%Y-%m-%d")
        if ny_date != self._today_date_str:
            self._today_date_str = ny_date
            self._today_realized_pnl_usdt = 0.0
            self._today_start_equity = equity_now if equity_now > 0 else 0.0
            self._daily_limit_hit = False
            logger.info(
                "daily PnL reset (NY %s) — start_equity=%.2f",
                ny_date, self._today_start_equity,
            )

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
        for cp in closed:
            ts = int(getattr(cp, "closed_at_ts", 0) or 0)
            if ts >= since_ms:
                total += float(getattr(cp, "pnl_usd", 0.0) or 0.0)
        self._today_realized_pnl_usdt = total
        if self._is_daily_loss_limit_hit() and not self._daily_limit_hit:
            self._daily_limit_hit = True
            logger.warning(
                "daily loss limit HIT (거래소 동기화) — limit=%.2f%% today=%.2fUSDT "
                "(start_equity=%.2f)",
                self.daily_loss_limit_pct,
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
        try:
            self._trades_store.record(TradeEvent(
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
            ))
        except Exception as e:  # noqa: BLE001
            logger.warning("trades record 실패: %s", e)

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
            return
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
        # SL/TP 구분 — close 가격이 SL/TP 중 허용 오차 내로 가까우면 해당 이벤트로 분류.
        sl, tp = last_known.stop_loss, last_known.take_profit
        evt_type = TradeEventType.SYNC_CLOSE
        close_reason = "exchange-side close (SL/TP 미구분)"
        if cp is not None and sl > 0 and tp > 0 and close_px > 0:
            # 허용 오차 = max(entry 0.2%, SL 거리 절반) — 슬리피지 + bybit conditional 발동 가격 차이.
            tol = max(last_known.entry * 0.002, abs(sl - last_known.entry) * 0.5)
            d_sl = abs(close_px - sl)
            d_tp = abs(close_px - tp)
            if d_sl <= tol and d_sl <= d_tp:
                evt_type = TradeEventType.SL_HIT
                close_reason = "SL_HIT"
            elif d_tp <= tol and d_tp <= d_sl:
                evt_type = TradeEventType.TP_HIT
                close_reason = "TP_HIT"
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
        return cands[0]

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
            resp = await self.client.place_order(
                symbol=self.symbol, side=new_side, qty=new_qty,
                price=None, stop_loss=new_sl, take_profit=None,
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

        # OHLCV snapshot — prefetch cache 활용 (없으면 빈 list)
        ohlcv_1h = self._ohlcv_cache.get("1h", [])
        ohlcv_5m = self._ohlcv_cache.get("5m", [])
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

    # TF 별 prefetch 한도 — 작은 TF (1m/5m) 는 메모리 부담으로 합리적 max,
    # 큰 TF (1h~1w) 는 Bybit 거래소 history 다 (2020-03~). app.js CANDLE_LIMIT 정합.
    _UI_OHLCV_TF_LIMITS: ClassVar[dict[str, int]] = {
        "1m": 5000, "5m": 20000, "15m": 50000,
        "1h": 60000, "2h": 30000, "4h": 15000,
        "1d": 5000, "1w": 1500,
    }
    # /ict/ohlcv 가 cache 갱신 트리거할 때 받을 봉 수 (마지막 N봉만 refresh).
    _UI_OHLCV_REFRESH_TAIL: ClassVar[int] = 200

    def _get_ohlcv_lock(self, tf: str) -> asyncio.Lock:
        """TF 별 cache 갱신 lock — lazy init (이벤트 루프 안에서)."""
        lock = self._ohlcv_cache_locks.get(tf)
        if lock is None:
            lock = asyncio.Lock()
            self._ohlcv_cache_locks[tf] = lock
        return lock

    async def _prefetch_all_ohlcv_tfs(self) -> None:
        """봇 시작 직후 background — 모든 UI TF prefetch.

        각 TF 별로 _UI_OHLCV_TF_LIMITS 만큼 fetch_ohlcv → _ohlcv_cache 채움.
        실패 시 해당 TF skip (다른 TF prefetch 계속). 사용자가 봇 가동 후
        바로 차트 띄우면 작은 TF (1d/1w) 부터 차례로 채워져 즉시 응답 가능.
        """
        # 작은 봉 수부터 먼저 — 빨리 끝나는 TF 부터 cache 채워 즉시 응답 가능.
        tf_order = sorted(self._UI_OHLCV_TF_LIMITS.items(), key=lambda kv: kv[1])
        for tf, limit in tf_order:
            try:
                async with self._get_ohlcv_lock(tf):
                    if tf in self._ohlcv_cache:
                        continue  # 이미 누가 채웠으면 skip
                    rows = await self.client.fetch_ohlcv(self.symbol, tf, limit)
                    if rows:
                        self._ohlcv_cache[tf] = list(rows)
                        logger.info(
                            "OHLCV prefetch 완료 %s — %d봉", tf, len(rows),
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
                cache = self._ohlcv_cache.get(tf, [])
                # ts → row 로 merge (새 봉 add + 진행 중 봉 update)
                by_ts = {r[0]: r for r in cache}
                for r in new_rows:
                    by_ts[r[0]] = r
                merged = sorted(by_ts.values(), key=lambda r: r[0])
                self._ohlcv_cache[tf] = merged
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
        cache = self._ohlcv_cache.get(tf)
        if cache and len(cache) > 0:
            # cache hit — background refresh 트리거 (await 안 함)
            asyncio.create_task(self._refresh_ohlcv_cache_recent(tf))
            return cache[-requested_limit:]
        # cache miss — 작은 limit 으로 sync fetch + 풀 prefetch 백그라운드 트리거
        quick_limit = min(requested_limit, 200)
        async with self._get_ohlcv_lock(tf):
            if tf in self._ohlcv_cache:  # 다른 task 가 채웠을 수도
                return self._ohlcv_cache[tf][-requested_limit:]
            try:
                rows = await self.client.fetch_ohlcv(self.symbol, tf, quick_limit)
            except Exception as e:  # noqa: BLE001
                logger.warning("get_ohlcv_cached sync fetch %s 실패: %s", tf, e)
                return []
            if rows:
                self._ohlcv_cache[tf] = list(rows)
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
                    self._ohlcv_cache[tf] = list(rows)
                    logger.info(
                        "OHLCV background prefetch %s — %d봉", tf, len(rows),
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
