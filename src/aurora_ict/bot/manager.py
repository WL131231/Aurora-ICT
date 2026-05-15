"""BotManager — 단일 BotIctInstance의 lifecycle + 모드 toggle 관리.

담당 책임:
- BotIctInstance 생성/소유 (또는 인스턴스 외부 주입)
- start / stop / 모드 전환 (demo ↔ live)
- 상태 스냅샷 제공 (UI / API 호출용)

모드 전환 흐름:
- 기존 인스턴스 정지 → settings 갱신 → 새 client로 새 BotIctInstance 생성

단일 인스턴스 전제 (멀티 심볼/유저는 상위 계층에서 처리).
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from aurora_ict.bot.bot_ict_instance import (
    BotIctInstance,
    BotState,
    ExchangeClientProtocol,
)
from aurora_ict.config.settings import IctSettings, RunMode, get_settings

logger = logging.getLogger(__name__)


# Client factory — settings를 받아 client를 만들어 반환하는 async callable.
ClientFactory = Callable[[IctSettings], Awaitable[ExchangeClientProtocol]]


@dataclass(slots=True)
class BotStatus:
    """봇 상태 스냅샷 (UI/API 응답용)."""

    state: BotState
    run_mode: RunMode
    enabled: bool
    symbol: str
    has_credentials: bool
    has_active_position: bool
    last_setup_ts_ms: int


@dataclass(slots=True)
class BotManager:
    """단일 BotIctInstance의 lifecycle + 모드 전환을 관리.

    Attributes:
        client_factory: settings로부터 ExchangeClient를 만드는 async factory.
        settings: 사용할 설정. ``None``이면 ``get_settings()`` 사용.
    """

    client_factory: ClientFactory
    settings: IctSettings = field(default_factory=get_settings)
    _bot: BotIctInstance | None = field(default=None)

    async def start(self) -> BotStatus:
        """봇 기동. ``enabled=False``이거나 API 키가 없으면 ValueError."""
        if not self.settings.enabled:
            raise ValueError("봇이 disabled 상태 — settings.enabled=False")
        if not self.settings.has_credentials():
            raise ValueError(
                f"API 키 없음 (run_mode={self.settings.run_mode.value})",
            )
        if self._bot is not None and self._bot.state is BotState.RUNNING:
            logger.info("이미 실행 중 — re-start 무시")
            return self.status()

        client = await self.client_factory(self.settings)
        self._bot = BotIctInstance(
            client=client,
            symbol=self.settings.symbol,
            timeframe=self.settings.timeframe,
            leverage=self.settings.leverage,
            position_pct_base=self.settings.position_pct_base,
            position_pct_max=self.settings.position_pct_max,
            position_pct_step=self.settings.position_pct_step,
            min_rr=self.settings.min_rr,
            fvg_min_size_pct=self.settings.fvg_min_size_pct,
            step_interval_sec=self.settings.step_interval_sec,
            ohlcv_limit=self.settings.ohlcv_limit,
            # v0.4.30 이후 진입 완화 옵션 — settings 에서 명시 주입.
            setup_stale_bars=self.settings.setup_stale_bars,
            disable_time_filter=self.settings.disable_time_filter,
            # multi-TF (ICT 정통 HTF setup + LTF confirm) 옵션.
            multi_tf=self.settings.multi_tf,
            multi_tf_ltf_lookback=self.settings.multi_tf_ltf_lookback,
            enable_trail=self.settings.enable_trail,
            trail_buffer_ratio=self.settings.trail_buffer_ratio,
            use_market_entry=self.settings.use_market_entry,
            enable_partial_tp=self.settings.enable_partial_tp,
        )
        # 거래소 측 leverage 를 settings 에 맞춤 — qty 계산 일치 보장.
        # 실패해도 봇 시작 자체는 진행 (warning 만, 사용자가 수동 박은 거 박혀있을 수 있음).
        try:
            await client.set_leverage(self.settings.symbol, self.settings.leverage)
        except Exception as e:  # noqa: BLE001
            logger.warning("set_leverage 호출 자체 실패: %s", e)
        await self._bot.start()
        logger.info(
            "BotManager started — mode=%s symbol=%s leverage=%dx",
            self.settings.run_mode.value, self.settings.symbol, self.settings.leverage,
        )
        return self.status()

    async def stop(self) -> BotStatus:
        """봇 정지."""
        if self._bot is not None:
            await self._bot.stop()
        return self.status()

    async def set_run_mode(self, mode: RunMode) -> BotStatus:
        """모드 전환 — 실행 중이면 stop → settings 변경 → restart."""
        was_running = self._bot is not None and self._bot.state is BotState.RUNNING
        if was_running:
            await self.stop()
        self.settings.run_mode = mode
        logger.info("run_mode 변경 → %s", mode.value)
        if was_running:
            await self.start()
        return self.status()

    async def set_enabled(self, enabled: bool) -> BotStatus:
        """``enabled`` 토글. False면 stop, True면 start (필요 시)."""
        was_running = self._bot is not None and self._bot.state is BotState.RUNNING
        self.settings.enabled = enabled
        if not enabled and was_running:
            await self.stop()
        elif enabled and not was_running:
            await self.start()
        return self.status()

    def status(self) -> BotStatus:
        """현재 상태 스냅샷."""
        if self._bot is None:
            return BotStatus(
                state=BotState.STOPPED,
                run_mode=self.settings.run_mode,
                enabled=self.settings.enabled,
                symbol=self.settings.symbol,
                has_credentials=self.settings.has_credentials(),
                has_active_position=False,
                last_setup_ts_ms=0,
            )
        return BotStatus(
            state=self._bot.state,
            run_mode=self.settings.run_mode,
            enabled=self.settings.enabled,
            symbol=self._bot.symbol,
            has_credentials=self.settings.has_credentials(),
            has_active_position=self._bot.active_position is not None,
            last_setup_ts_ms=self._bot._last_setup_ts_ms,
        )

    @property
    def bot(self) -> BotIctInstance | None:
        """현재 BotIctInstance 노출 (debug/내부 접근용)."""
        return self._bot


def _make_status_dict(status: BotStatus) -> dict[str, Any]:
    """BotStatus → dict (UI/API 직렬화용)."""
    return {
        "state": status.state.value,
        "run_mode": status.run_mode.value,
        "enabled": status.enabled,
        "symbol": status.symbol,
        "has_credentials": status.has_credentials,
        "has_active_position": status.has_active_position,
        "last_setup_ts_ms": status.last_setup_ts_ms,
    }


__all__ = [
    "BotManager",
    "BotStatus",
    "ClientFactory",
]
