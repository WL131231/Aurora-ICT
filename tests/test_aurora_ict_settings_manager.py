"""Settings + BotManager — Aurora-ICT v0.2.0."""

from __future__ import annotations

import os
from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.bot_ict_instance import BotState
from aurora_ict.bot.manager import BotManager
from aurora_ict.config.settings import IctSettings, RunMode, reload_settings


@pytest.fixture(autouse=True)
def _clean_env():
    """env 박힘 박힘 박힘 박힘 박힘 박힘 settings 박힘 박힘 박힘 박힙 박힘 박힘 reset 박힘."""
    keep = {k: v for k, v in os.environ.items() if k.startswith("AURORA_ICT_")}
    for k in list(keep):
        del os.environ[k]
    yield
    for k in list(os.environ):
        if k.startswith("AURORA_ICT_"):
            del os.environ[k]
    for k, v in keep.items():
        os.environ[k] = v


# ============================================================
# IctSettings
# ============================================================


def test_settings_defaults() -> None:
    """기본 값 박힙 박힘 demo 박힙 disabled."""
    s = IctSettings()
    assert s.run_mode is RunMode.DEMO
    assert s.enabled is False
    assert s.symbol == "BTC/USDT:USDT"
    assert s.has_credentials() is False


def test_settings_from_env() -> None:
    """env 박힙 박힘 박힙 박힙 박힘 박힙 박힙."""
    os.environ["AURORA_ICT_RUN_MODE"] = "live"
    os.environ["AURORA_ICT_ENABLED"] = "true"
    os.environ["AURORA_ICT_DEMO_API_KEY"] = "demo-key-xxx"
    os.environ["AURORA_ICT_LIVE_API_KEY"] = "live-key-yyy"
    os.environ["AURORA_ICT_LIVE_API_SECRET"] = "live-secret-zzz"
    s = IctSettings()
    assert s.run_mode is RunMode.LIVE
    assert s.enabled is True
    assert s.active_api_key == "live-key-yyy"
    assert s.active_api_secret == "live-secret-zzz"
    assert s.has_credentials() is True


def test_settings_active_keys_switch_with_mode() -> None:
    """run_mode 박힘 박힘 active key 박힘 박힙."""
    s = IctSettings(
        demo_api_key="d-key", demo_api_secret="d-sec",
        live_api_key="l-key", live_api_secret="l-sec",
    )
    s.run_mode = RunMode.DEMO
    assert s.active_api_key == "d-key"
    s.run_mode = RunMode.LIVE
    assert s.active_api_key == "l-key"


def test_settings_has_credentials_partial() -> None:
    """secret 박힙 박힘 박힘 X → False."""
    s = IctSettings(demo_api_key="x")
    assert s.has_credentials() is False


def test_reload_settings() -> None:
    """reload 박힘 박힘 박힙 박힘 박힙 박힙."""
    os.environ["AURORA_ICT_SYMBOL"] = "ETH/USDT:USDT"
    s = reload_settings()
    assert s.symbol == "ETH/USDT:USDT"


# ============================================================
# BotManager
# ============================================================


def _mock_client():
    c = AsyncMock()
    c.fetch_ohlcv = AsyncMock(return_value=[])
    c.place_order = AsyncMock(return_value={})
    c.fetch_position = AsyncMock(return_value=None)
    c.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    return c


async def _factory_with_mock(_settings):
    return _mock_client()


@pytest.mark.asyncio
async def test_manager_start_without_credentials_raises() -> None:
    """API 키 박힘 X → start 박힙 박힘 ValueError."""
    settings = IctSettings(enabled=True)  # no credentials
    mgr = BotManager(client_factory=_factory_with_mock, settings=settings)
    with pytest.raises(ValueError, match="API"):
        await mgr.start()


@pytest.mark.asyncio
async def test_manager_start_disabled_raises() -> None:
    """enabled=False → start 박힙 박힘 ValueError."""
    settings = IctSettings(
        enabled=False,
        demo_api_key="k", demo_api_secret="s",
    )
    mgr = BotManager(client_factory=_factory_with_mock, settings=settings)
    with pytest.raises(ValueError, match="disabled"):
        await mgr.start()


@pytest.mark.asyncio
async def test_manager_start_stop_lifecycle() -> None:
    """start → state RUNNING, stop → STOPPED."""
    settings = IctSettings(
        enabled=True,
        demo_api_key="k", demo_api_secret="s",
        step_interval_sec=3600,
    )
    mgr = BotManager(client_factory=_factory_with_mock, settings=settings)
    status = await mgr.start()
    assert status.state is BotState.RUNNING
    assert status.run_mode is RunMode.DEMO
    status = await mgr.stop()
    assert status.state is BotState.STOPPED


@pytest.mark.asyncio
async def test_manager_set_run_mode_restarts() -> None:
    """running 박힌 거 박힙 박힘 모드 박힌 거 박힘 박힙 박힘 restart 박힘."""
    settings = IctSettings(
        enabled=True,
        demo_api_key="dk", demo_api_secret="ds",
        live_api_key="lk", live_api_secret="ls",
        step_interval_sec=3600,
    )
    mgr = BotManager(client_factory=_factory_with_mock, settings=settings)
    await mgr.start()
    assert mgr.status().run_mode is RunMode.DEMO

    status = await mgr.set_run_mode(RunMode.LIVE)
    assert status.run_mode is RunMode.LIVE
    assert status.state is BotState.RUNNING
    await mgr.stop()


@pytest.mark.asyncio
async def test_manager_set_run_mode_when_stopped() -> None:
    """stopped 박힌 거 박힙 박힘 모드 박힘 박힙 → 박힘 박힘 박힙 박힘."""
    settings = IctSettings(
        demo_api_key="k", demo_api_secret="s",
    )
    mgr = BotManager(client_factory=_factory_with_mock, settings=settings)
    status = await mgr.set_run_mode(RunMode.LIVE)
    assert status.run_mode is RunMode.LIVE
    assert status.state is BotState.STOPPED


@pytest.mark.asyncio
async def test_manager_set_enabled_toggles() -> None:
    """enabled=True 박힘 → start, False 박힘 → stop."""
    settings = IctSettings(
        demo_api_key="k", demo_api_secret="s",
        step_interval_sec=3600,
    )
    mgr = BotManager(client_factory=_factory_with_mock, settings=settings)
    status = await mgr.set_enabled(True)
    assert status.enabled is True
    assert status.state is BotState.RUNNING

    status = await mgr.set_enabled(False)
    assert status.enabled is False
    assert status.state is BotState.STOPPED


@pytest.mark.asyncio
async def test_manager_double_start_is_idempotent() -> None:
    """start 박힌 두 번째 박힘 박힙 박힙 박힘 박힙 박힘 박힙 박힌 거 박힙."""
    settings = IctSettings(
        enabled=True,
        demo_api_key="k", demo_api_secret="s",
        step_interval_sec=3600,
    )
    mgr = BotManager(client_factory=_factory_with_mock, settings=settings)
    await mgr.start()
    first_bot = mgr.bot
    await mgr.start()
    # 박힌 박힌 인스턴스 박힘 박힘 그대로 박힘
    assert mgr.bot is first_bot
    await mgr.stop()
