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

    # 2026-05-27 파트너 요청 — UI 차트 TF 토글 시 로딩 없이 즉시 응답.
    # 봇 start 시 background prefetch 로 모든 TF 채워두고, /ict/ohlcv 가
    # cache 사용. polling 시 마지막 N봉만 incremental refresh.
    # 자료: dict[tf, list[[ts_ms, o, h, l, c, v], ...]] ts 오름차순.
    _ohlcv_cache: dict[str, list[list[Any]]] = field(default_factory=dict)
    # TF 별 동시 갱신 방지 — lazy init (이벤트 루프 필요).
    _ohlcv_cache_locks: dict[str, asyncio.Lock] = field(default_factory=dict)
    # prefetch background task 핸들 (취소용).
    _prefetch_task: asyncio.Task[None] | None = field(default=None)

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

    async def _recover_position_from_exchange(self) -> None:
        """봇 시작 시 거래소 측 활성 포지션 복원.

        fetch_position 호출 → contracts > 0 이면 active_position 채움.
        ts_ms / entry / SL / TP 박은 거 거래소 응답에서 추출 (없으면 추정값).
        """
        try:
            pos = await self.client.fetch_position(self.symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning("recover fetch_position 실패: %s — skip", e)
            return
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
        # SL/TP 박은 거 박은 거 박은 거 — Bybit V5 응답 stopLoss / takeProfit.
        sl = float(pos.get("stopLossPrice") or pos.get("stop_loss") or 0) or 0.0
        tp = float(pos.get("takeProfitPrice") or pos.get("take_profit") or 0) or 0.0
        self.active_position = _ActivePosition(
            direction=direction,
            entry=entry_price,
            stop_loss=sl,
            take_profit=tp,
            qty=contracts,
            setup_ts_ms=0,  # recovery 박은 거 박은 거 박은 거 박은 거 ts_ms 모름
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

        signal = generate_ict_signal(
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
            return signal

        # 동일 setup으로 재진입 방지 (중복 주문 X)
        if signal.setup.ts_ms == self._last_setup_ts_ms:
            return signal

        if not await self._passes_htf_ema_bias(signal.setup.direction):
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

        # #LIVE-3 fix: entry = setup.entry (계획가 — FVG mean 등) 에 limit. 가격이 거기
        # retrace 하면 체결. SL/TP 가 setup 기준이라 RR 보존. use_market_entry=True 면
        # 레거시 즉시 시장가 (slippage, 비권장).
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
            logger.exception("place_order 실패: %s", e)
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
        self.active_position = _ActivePosition(
            direction=setup.direction,
            entry=fill_price,
            stop_loss=setup.stop_loss,
            take_profit=setup.take_profit,
            qty=qty,
            setup_ts_ms=setup.ts_ms,
            htf_flip_target=htf_flip_target,
            ltf_weight=TF_WEIGHT.get(self.timeframe, 1),
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
            self.active_position = _ActivePosition(
                direction=pe.direction,
                entry=entry_px,
                stop_loss=pe.stop_loss,
                take_profit=pe.take_profit,
                qty=pe.qty,
                setup_ts_ms=pe.setup_ts_ms,
                htf_flip_target=pe.htf_flip_target,
                ltf_weight=pe.ltf_weight,
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
        """위험(무SL 등) 상황에서 active_position 을 시장가 reduce_only 로 청산."""
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
        except Exception as e:  # noqa: BLE001
            logger.error("비상청산 실패: %s — 수동 확인 필요", e)
        self.active_position = None

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
            try:
                self._trades_store = TradesStore(_ict_data_dir())
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
            ))
        except Exception as e:  # noqa: BLE001
            logger.warning("trades record 실패: %s", e)

    async def _fetch_recent_close(self, last_known: _ActivePosition):
        """state-reset 시점 거래소 closed-pnl 에서 매칭되는 close 회수 (#PR-C / #4).

        Args:
            last_known: state-reset 직전의 active_position (방향·setup ts 매칭 기준).

        Returns:
            매칭 ClosedPosition (가장 최근, 같은 symbol+direction). 실패/미매칭이면 None.
        """
        since_ms = (
            last_known.setup_ts_ms
            if last_known.setup_ts_ms > 0
            else int(time.time() * 1000) - 3_600_000
        )
        try:
            closed = await self.client.fetch_closed_positions(
                since_ms=since_ms, limit=10,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("closed-pnl 조회 실패: %s", e)
            return None
        if not closed:
            return None
        want_dir = "long" if last_known.direction is Direction.LONG else "short"
        for cp in closed:
            cp_sym = getattr(cp, "symbol", None)
            cp_dir = getattr(cp, "direction", None)
            if cp_sym != self.symbol or cp_dir != want_dir:
                continue
            return cp  # 가장 최근(신→구 정렬) 매칭
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
            logger.warning("fetch_position 실패: %s", e)
            return
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
        self._record_trade(
            evt_type,
            direction=last_known.direction,
            price=close_px,
            qty=last_known.qty,
            entry_for_pnl=last_known.entry,
            setup_ts_ms=last_known.setup_ts_ms,
            reason=close_reason,
            pnl_override=pnl_usd if cp is not None else None,
        )
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
        cands = find_opposite_htf_fvg(
            htf_map,
            ltf_direction="buy" if setup.direction is Direction.LONG else "sell",
            current_price=current_price,
            threshold_weight=ltf_weight,
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
        if total_weight >= 20:
            boost = 3
        elif total_weight >= 10:
            boost = 2
        elif total_weight >= 4:
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
            logger.error("flip — 청산 최종 실패 — 신규 진입 중단 (포지션 유지)")
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
        # SL 별도 박기 (place_order 가 inline 으로 못 받는 경우 보강).
        try:
            await self.client.modify_stop_loss(self.symbol, new_sl)
        except Exception as e:  # noqa: BLE001
            logger.warning("flip — SL 적용 실패 (%.4f): %s — 포지션은 유지", new_sl, e)

        self.active_position = _ActivePosition(
            direction=new_direction,
            entry=new_entry,
            stop_loss=new_sl,
            take_profit=new_tp,
            qty=new_qty,
            setup_ts_ms=last_ts,
            htf_flip_target=None,  # flip 완료 — 같은 target 재발동 방지.
            ltf_weight=TF_WEIGHT.get(self.timeframe, 1),
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
