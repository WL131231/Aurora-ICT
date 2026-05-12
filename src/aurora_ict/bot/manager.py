"""BotManager — 단일 BotIctInstance 박힌 거 박힘 lifecycle + 모드 toggle.

박은 거 박은 거 박힘:
- BotIctInstance 박힌 거 박힘 박힘 박힘 박힘 박힘 박힘 (또는 인스턴스 외부 박힘)
- start / stop / 모드 박힘 (demo ↔ live) 박힘 박힘
- 상태 박힌 거 박힘 박힘 박힘 (UI / API 박힌 거 박힙 박힘 박힘)

모드 박힘 박힘 박힘 박힘:
- 박힌 인스턴스 박힘 박힘 박힘 박힘 → settings 박힘 박힘 박힘 → 박은 client 박힘 박힘
  새 BotIctInstance 박힘

박힌 거 박힌 거 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘
박힘 박힘 박힘 박힘 박힘 박힘 박힘.
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


# Client factory 박힌 거 박힙 박힘 — settings 박힙 박힘 client 박힘 박힘 박힘 박힘 callable.
ClientFactory = Callable[[IctSettings], Awaitable[ExchangeClientProtocol]]


@dataclass(slots=True)
class BotStatus:
    """봇 상태 박힘 박힘 (UI/API 박힙 박힘)."""

    state: BotState
    run_mode: RunMode
    enabled: bool
    symbol: str
    has_credentials: bool
    has_active_position: bool
    last_setup_ts_ms: int


@dataclass(slots=True)
class BotManager:
    """단일 BotIctInstance 박힌 거 박힙 lifecycle + 모드 박힘.

    Attributes:
        client_factory: settings 박힙 박힘 ExchangeClient 박힘 박힘 박힘 박힘 (async).
        settings: 박힘 박힘 박힘 박힘 박힘. ``None`` 박힘 박힘 ``get_settings()`` 박힘.
    """

    client_factory: ClientFactory
    settings: IctSettings = field(default_factory=get_settings)
    _bot: BotIctInstance | None = field(default=None)

    async def start(self) -> BotStatus:
        """봇 박힘 박힘. ``enabled=False`` 박힘 박힘 박힘 박힘 ValueError."""
        if not self.settings.enabled:
            raise ValueError("봇 박힘 박은 disabled — settings.enabled=False")
        if not self.settings.has_credentials():
            raise ValueError(
                f"API 키 박힘 X (run_mode={self.settings.run_mode.value})",
            )
        if self._bot is not None and self._bot.state is BotState.RUNNING:
            logger.info("이미 박힘 박힘 — re-start 박힘 X")
            return self.status()

        client = await self.client_factory(self.settings)
        self._bot = BotIctInstance(
            client=client,
            symbol=self.settings.symbol,
            timeframe=self.settings.timeframe,
            risk_per_trade_pct=self.settings.risk_per_trade_pct,
            leverage=self.settings.leverage,
            min_rr=self.settings.min_rr,
            fvg_min_size_pct=self.settings.fvg_min_size_pct,
            step_interval_sec=self.settings.step_interval_sec,
            ohlcv_limit=self.settings.ohlcv_limit,
        )
        await self._bot.start()
        logger.info(
            "BotManager 박힘 박힘 — mode=%s symbol=%s",
            self.settings.run_mode.value, self.settings.symbol,
        )
        return self.status()

    async def stop(self) -> BotStatus:
        """봇 박힘 박힘 박힘."""
        if self._bot is not None:
            await self._bot.stop()
        return self.status()

    async def set_run_mode(self, mode: RunMode) -> BotStatus:
        """모드 박힘 — 박힙 박힘 박힘 박힘 박힘 (running 박힙 박힘 stop 박힘 → 박힘 박힘
        박힘 변경 → restart)."""
        was_running = self._bot is not None and self._bot.state is BotState.RUNNING
        if was_running:
            await self.stop()
        self.settings.run_mode = mode
        logger.info("run_mode 박힘 박힘 → %s", mode.value)
        if was_running:
            await self.start()
        return self.status()

    async def set_enabled(self, enabled: bool) -> BotStatus:
        """``enabled`` 박힘 박힘. False 박힙 박힘 stop, True 박힙 박힘 start (박힌 박힘
        박힙 박힘 박힘)."""
        was_running = self._bot is not None and self._bot.state is BotState.RUNNING
        self.settings.enabled = enabled
        if not enabled and was_running:
            await self.stop()
        elif enabled and not was_running:
            await self.start()
        return self.status()

    def status(self) -> BotStatus:
        """현재 상태 박힘."""
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
        """박은 박힌 BotIctInstance 박힘 박힘 박힘 박힘 (debug 박힘)."""
        return self._bot


def _make_status_dict(status: BotStatus) -> dict[str, Any]:
    """BotStatus → dict (UI/API 박힘 박힘)."""
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
