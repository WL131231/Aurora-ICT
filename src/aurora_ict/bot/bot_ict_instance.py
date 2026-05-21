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
import logging
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

import pandas as pd

from aurora_ict.bot.structure_trail import compute_structure_trail
from aurora_ict.indicators.daily_bias import compute_daily_bias
from aurora_ict.indicators.structure import TrendDirection
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

    async def modify_stop_loss(
        self, symbol: str, new_stop_loss: float,
    ) -> dict[str, Any]: ...

    async def set_leverage(
        self, symbol: str, leverage: int,
    ) -> dict[str, Any]: ...


class BotState(StrEnum):
    """봇 상태."""

    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


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

    # use_market_entry: True 면 setup 검출 시 limit (FVG mean retest) 대신 즉시 시장가 진입.
    # 진입률 100%, ICT 정통 retrace 철학에서 살짝 벗어남.
    use_market_entry: bool = False
    # min_sl_distance_pct: SL 거리 / entry 가 이 비율 미만이면 setup skip (noise-trap 회피).
    min_sl_distance_pct: float = 0.0
    # max_sl_distance_pct: SL 거리 / entry 가 이 비율 초과면 setup skip (비정상 큰 SL 차단).
    # 0 = 비활성.
    max_sl_distance_pct: float = 0.0
    # heartbeat_interval_sec: step loop 살아있음 INFO 로그 주기. 0 = 비활성.
    heartbeat_interval_sec: int = 0

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

    async def stop(self) -> None:
        """봇 정지 (background task cancel)."""
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
        """setup 한 건을 실제 주문으로 실행 (ICT 정통 단일 TP).

        - entry limit order 등록 (SL 동봉, 시장가 진입은 fill 후 SL 별도)
        - take_profit = next BSL/SSL 단일 reduce_only limit 1건
        """
        side = "buy" if setup.direction is Direction.LONG else "sell"
        exit_side = "sell" if setup.direction is Direction.LONG else "buy"

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

        # 실제 fill 가격 / SL / TP — 시장가 진입 시 fill 가격이 setup 과 다를 수 있어
        # 응답에서 추출 후 SL/TP 모두 fill 기준으로 재계산.
        effective_entry = setup.entry
        effective_sl = setup.stop_loss
        effective_tp = setup.take_profit
        try:
            # 시장가 진입 시: SL 은 entry 주문에 동봉하지 않고 fill 확인 후 별도 박음.
            # 그래야 SL 이 실제 fill 가격 기준의 risk distance 로 정확하게 박힘.
            entry_price = None if self.use_market_entry else setup.entry
            inline_sl = None if self.use_market_entry else setup.stop_loss
            order_resp = await self.client.place_order(
                symbol=self.symbol,
                side=side,
                qty=qty,
                price=entry_price,
                stop_loss=inline_sl,
                take_profit=None,
            )
            # Fill 가격 추출 — adapter 가 avg_fill_price / filled_qty 박아 반환.
            avg_fill = None
            if isinstance(order_resp, dict):
                v = order_resp.get("avg_fill_price")
                if isinstance(v, (int, float)) and v > 0:
                    avg_fill = float(v)
            if self.use_market_entry and avg_fill is not None and avg_fill > 0:
                # setup 가격 → fill 가격 shift. SL/TP 동일 distance 유지.
                sl_dist = setup.entry - setup.stop_loss
                tp_dist = setup.take_profit - setup.entry
                effective_entry = avg_fill
                effective_sl = avg_fill - sl_dist
                effective_tp = avg_fill + tp_dist
                logger.info(
                    "fill 가격 반영 — setup=%.4f fill=%.4f sl=%.4f→%.4f tp=%.4f→%.4f",
                    setup.entry, avg_fill, setup.stop_loss, effective_sl,
                    setup.take_profit, effective_tp,
                )
                # SL 을 fill 기준으로 박음 (시장가 진입 시).
                try:
                    await self.client.modify_stop_loss(self.symbol, effective_sl)
                except Exception as e:  # noqa: BLE001
                    logger.warning("fill 기반 SL 적용 실패: %s", e)
            # 정통 ICT: 단일 TP 한 건 (next BSL/SSL).
            await self.client.place_order(
                symbol=self.symbol,
                side=exit_side,
                qty=qty,
                price=effective_tp,
                reduce_only=True,
            )
        except Exception as e:  # noqa: BLE001 — 주문 실패도 봇은 계속 돌아야 함
            logger.exception("place_order 실패: %s", e)
            return

        self.active_position = _ActivePosition(
            direction=setup.direction,
            entry=effective_entry,
            stop_loss=effective_sl,
            take_profit=effective_tp,
            qty=qty,
            setup_ts_ms=setup.ts_ms,
            htf_flip_target=htf_flip_target,
            ltf_weight=TF_WEIGHT.get(self.timeframe, 1),
        )
        if htf_flip_target is not None:
            logger.info(
                "HTF flip target armed — %s zone=[%.4f,%.4f] weight=%d",
                htf_flip_target.tf, htf_flip_target.low, htf_flip_target.high,
                htf_flip_target.weight,
            )

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

    async def _sync_position_state(self) -> None:
        """거래소 fetch_position()으로 현재 position 상태 동기화.

        SL/TP가 트리거되어 거래소 쪽에서 닫혔으면 ``active_position`` 리셋.
        """
        try:
            pos = await self.client.fetch_position(self.symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_position 실패: %s", e)
            return

        if pos is None or float(pos.get("contracts", 0) or 0) == 0:
            # 거래소 측 position 없음 → SL/TP hit으로 청산된 것으로 간주
            logger.info("position 종료 감지 (SL/TP hit 추정) — state reset")
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


__all__ = [
    "BotIctInstance",
    "BotState",
    "ExchangeClientProtocol",
]
