"""실제 SQLite/봇/API 함수로 사용자 설정과 재시작 경계를 검증한다."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException, Request

from aurora_ict.api.app import (
    DailyLossLimitRequest,
    ModelRequest,
    TimeframeRequest,
    _pending_payload,
    create_app,
)
from aurora_ict.auth import keystore, pin, users_db
from aurora_ict.auth.middleware import SESSION_COOKIE_NAME
from aurora_ict.auth.router import ApiKeysRequest, create_auth_router
from aurora_ict.bot import aurora_client_factory
from aurora_ict.bot.bot_ict_instance import BotIctInstance, BotState, _ActivePosition, _PendingEntry
from aurora_ict.bot.bot_trend_instance import BotTrendInstance, _PendingLimitEntry
from aurora_ict.bot.multi_user_manager import MultiUserBotManager, _UserBotSlot
from aurora_ict.config.settings import CURSUS_MODEL_NAME, ORIGO_MODEL_NAME, IctSettings
from aurora_ict.saas import auto_resume_running_bots
from aurora_ict.strategy.silver_bullet import Direction

USER = "AICT-LIFE-AAAA-AAAA"
OTHER = "AICT-LIFE-BBBB-BBBB"
BTC, ETH, SOL = "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"


@pytest.fixture
def manager(tmp_path: Path) -> MultiUserBotManager:
    """실제 factory를 주입하되 이 모듈에서는 거래소 생성 경로를 호출하지 않는다."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    master = Fernet.generate_key()
    for code in (USER, OTHER):
        users_db.create_user(db, code)
        users_db.set_api_keys(
            db, code, "synthetic-public", keystore.encrypt_secret("synthetic-secret", key=master),
        )
        users_db.set_last_model(db, code, ORIGO_MODEL_NAME)
    return MultiUserBotManager(
        client_factory=aurora_client_factory, db_path=db, master_key=master,
        base_settings=IctSettings(_env_file=None, enabled=True, timeframe="1h"),
    )


def endpoint(manager: MultiUserBotManager, path: str, method: str = "POST"):
    app = create_app(
        multi_user=True, multi_user_manager=manager,
        auth_db_path=manager.db_path, master_key=manager.master_key,
    )
    return next(r.endpoint for r in app.routes if getattr(r, "path", None) == path
                and method in getattr(r, "methods", set()))


def add_origo(manager: MultiUserBotManager, symbol: str, *, code: str = USER):
    settings = manager._build_user_settings(code, symbol)
    bot = BotIctInstance(client=None, symbol=symbol, timeframe=settings.timeframe)
    manager._slots[(code, symbol)] = _UserBotSlot(
        symbol=symbol, settings=settings, bot=bot,
        credential_version=users_db.get_credential_state(manager.db_path, code, "demo")["version"],
    )
    return bot


@pytest.mark.asyncio
async def test_loss_limit_survives_reconstruction_and_is_user_owned(manager):
    for symbol in (ETH, SOL):
        add_origo(manager, symbol)
    result = await endpoint(manager, "/ict/daily_loss_limit")(
        DailyLossLimitRequest(pct=2), user_code=USER,
    )
    assert result["limit_pct"] == 2
    assert manager._build_user_settings(USER, ETH).daily_loss_limit_pct == 2
    for (code, _), slot in manager._slots.items():
        assert code == USER
        assert slot.bot.daily_loss_limit_pct == slot.settings.daily_loss_limit_pct == 2
    assert users_db.get_user_by_code(manager.db_path, OTHER)["daily_loss_limit_pct"] is None
    manager._slots.clear()
    assert (await endpoint(manager, "/ict/daily_loss_limit", "GET")(user_code=USER))["limit_pct"] == 2


@pytest.mark.asyncio
async def test_subscription_loss_limit_revalidates_persisted_value(manager):
    users_db.set_license(manager.db_path, code=USER, license_type="sub_30d", expires_at=None)
    save = endpoint(manager, "/ict/daily_loss_limit")
    assert (await save(DailyLossLimitRequest(pct=2), user_code=USER))["limit_pct"] == 2
    assert manager._build_user_settings(USER, ETH).daily_loss_limit_pct == 2
    assert (await save(DailyLossLimitRequest(pct=50), user_code=USER))["limit_pct"] == 15
    assert manager._build_user_settings(USER, ETH).daily_loss_limit_pct == 15
    with pytest.raises(HTTPException):
        await save(DailyLossLimitRequest(pct=float("nan")), user_code=USER)


@pytest.mark.asyncio
async def test_timeframe_without_btc_updates_all_own_slots_only(manager):
    for symbol in (ETH, SOL):
        add_origo(manager, symbol)
    other_bot = add_origo(manager, ETH, code=OTHER)
    result = await endpoint(manager, "/ict/timeframe")(TimeframeRequest(timeframe="15m"), user_code=USER)
    assert result["timeframe"] == "15m"
    assert manager.base_settings.timeframe == other_bot.timeframe == "1h"
    assert manager._build_user_settings(OTHER, BTC).timeframe == "1h"
    for symbol in (ETH, SOL):
        slot = manager._slots[(USER, symbol)]
        assert slot.settings.timeframe == slot.bot.timeframe == "15m"
    manager._slots.clear()
    assert manager._build_user_settings(USER, ETH).timeframe == "15m"


@pytest.mark.asyncio
async def test_defer_persists_old_model_and_pauses_new_entries(manager):
    bot = add_origo(manager, ETH)
    bot.state = BotState.RUNNING
    bot.active_position = _ActivePosition(Direction.LONG, 100, 98, 104, 1, 1000)
    result = await endpoint(manager, "/ict/model")(
        ModelRequest(model=CURSUS_MODEL_NAME, on_position="defer"), user_code=USER,
    )
    assert result["deferred"] == [ETH]
    assert manager._slots[(USER, ETH)].bot is bot
    assert bot.entry_paused is True
    assert users_db.get_last_model(manager.db_path, USER) == CURSUS_MODEL_NAME
    fresh = MultiUserBotManager(
        client_factory=aurora_client_factory, db_path=manager.db_path,
        base_settings=manager.base_settings, master_key=manager.master_key,
    )
    assert fresh.model_for_slot(USER, ETH) == ORIGO_MODEL_NAME
    assert fresh.model_for_slot(USER, BTC) == CURSUS_MODEL_NAME
    assert fresh.model_for_slot(OTHER, ETH) == ORIGO_MODEL_NAME
    assert (await manager.reconcile_models()) == {"switched": 0, "held": 1}
    assert users_db.get_deferred_models(manager.db_path, USER) == {ETH: ORIGO_MODEL_NAME}


def test_model_selection_atomically_replaces_only_own_overrides(manager):
    users_db.set_model_selection(manager.db_path, OTHER, CURSUS_MODEL_NAME, {SOL: ORIGO_MODEL_NAME})
    users_db.set_model_selection(manager.db_path, USER, CURSUS_MODEL_NAME, {ETH: ORIGO_MODEL_NAME})
    users_db.set_model_selection(manager.db_path, USER, ORIGO_MODEL_NAME, {ETH: ORIGO_MODEL_NAME})
    assert users_db.get_deferred_models(manager.db_path, USER) == {}
    assert users_db.get_deferred_models(manager.db_path, OTHER) == {SOL: ORIGO_MODEL_NAME}


@pytest.mark.asyncio
async def test_defer_before_auto_resume_keeps_running_db_slot_original_model(manager):
    users_db.set_bot_running(manager.db_path, USER, True, symbol=ETH)
    select = endpoint(manager, "/ict/model")
    result = await select(ModelRequest(model=CURSUS_MODEL_NAME), user_code=USER)
    assert result["needs_choice"] is True
    result = await select(ModelRequest(model=CURSUS_MODEL_NAME, on_position="defer"), user_code=USER)
    assert result["deferred"] == [ETH]
    assert manager.model_for_slot(USER, ETH) == ORIGO_MODEL_NAME
    assert manager._slots == {}


@pytest.mark.asyncio
async def test_cursus_pending_display_keeps_existing_behavior(manager):
    bot = BotTrendInstance(client=None, symbol=ETH)
    bot._pending_limit = _PendingLimitEntry(Direction.LONG, 100, 2, 98, "synthetic-order", 1000, 4000)
    manager._slots[(USER, ETH)] = _UserBotSlot(ETH, manager._build_user_settings(USER, ETH), bot)
    payload = _pending_payload(bot)
    assert payload is None
    result = await endpoint(manager, "/ict/position", "GET")(user_code=USER)
    assert result["active"] is False and result["pending"] == payload
    old = os.environ.get("AURORA_ICT_ADMIN_TOKEN")
    os.environ["AURORA_ICT_ADMIN_TOKEN"] = "synthetic-admin-token"
    try:
        result = await endpoint(manager, "/admin/positions", "GET")(
            x_admin_token="synthetic-admin-token", cookie_token=None,
        )
    finally:
        if old is None:
            os.environ.pop("AURORA_ICT_ADMIN_TOKEN", None)
        else:
            os.environ["AURORA_ICT_ADMIN_TOKEN"] = old
    assert result["count"] == 0
    assert bot._pending_limit is not None


@pytest.mark.asyncio
async def test_auth_stop_blocks_manual_and_auto_resume_without_client_creation(manager):
    state = users_db.get_credential_state(manager.db_path, USER, "demo")
    assert users_db.block_credentials(manager.db_path, USER, "demo", state["version"], "auth_expired")
    users_db.set_bot_running(manager.db_path, USER, True, symbol=ETH)
    users_db.set_last_run_mode(manager.db_path, USER, "demo")
    with pytest.raises(ValueError, match="API"):
        await manager.start(USER, ETH)
    assert await auto_resume_running_bots(manager, manager.db_path) == {"attempted": 1, "succeeded": 0, "failed": 1}
    assert manager._slots == {}
    status = await manager.status(USER, ETH)
    assert status["stop_reason"] == "auth_expired"
    assert status["state"] == "stopped"


def test_only_verified_registration_unlocks_and_old_callback_cannot_reblock(manager):
    version = users_db.get_credential_state(manager.db_path, USER, "demo")["version"]
    assert users_db.block_credentials(manager.db_path, USER, "demo", version, "api_permission")
    secret = keystore.encrypt_secret("synthetic-secret", key=manager.master_key)
    users_db.set_api_keys(manager.db_path, USER, "synthetic-public", secret)
    assert users_db.get_credential_state(manager.db_path, USER, "demo")["stop_reason"] == "api_permission"
    users_db.set_api_keys(manager.db_path, USER, "synthetic-public", secret, verified=True)
    assert users_db.get_credential_state(manager.db_path, USER, "demo")["stop_reason"] is None
    assert not users_db.block_credentials(manager.db_path, USER, "demo", version, "auth_expired")
    assert users_db.get_credential_state(manager.db_path, USER, "live")["stop_reason"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["/ict/stop", "/ict/stop-all"])
async def test_unconfirmed_pending_stop_retains_slot_and_reports_conflict(manager, path):
    bot = add_origo(manager, ETH)
    bot.state = BotState.RUNNING
    bot._pending_entry = _PendingEntry(Direction.LONG, 100, 98, 104, 1, 1000, 1000)
    users_db.set_bot_running(manager.db_path, USER, True, symbol=ETH)
    kwargs = {"symbol": ETH} if path == "/ict/stop" else {}
    with pytest.raises(HTTPException) as raised:
        await endpoint(manager, path)(user_code=USER, **kwargs)
    assert raised.value.status_code == 409
    assert manager._slots[(USER, ETH)].bot is bot
    assert bot._pending_entry is not None and bot.entry_paused
    assert users_db.get_bot_running_symbols(manager.db_path, USER) == []
    await manager.reconcile_models()
    assert bot.entry_paused is True


@pytest.mark.asyncio
@pytest.mark.parametrize("license_type,model,expected", [
    ("sub_30d", ORIGO_MODEL_NAME, "5m"),
    ("referral", CURSUS_MODEL_NAME, "15m"),
])
async def test_saved_timeframe_obeys_subscription_and_model_policy(manager, license_type, model, expected):
    users_db.set_license(manager.db_path, code=USER, license_type=license_type, expires_at=None)
    users_db.set_last_model(manager.db_path, USER, model)
    result = await endpoint(manager, "/ict/timeframe")(TimeframeRequest(timeframe="15m"), user_code=USER)
    assert result["timeframe"] == expected
    assert users_db.get_last_timeframe(manager.db_path, USER) == expected
    assert manager._build_user_settings(USER, ETH).timeframe == expected
    assert manager.base_settings.timeframe == ("15m" if model == CURSUS_MODEL_NAME else "1h")


@pytest.mark.asyncio
async def test_blocked_registration_cannot_unlock_without_verifier(manager):
    state = users_db.get_credential_state(manager.db_path, USER, "demo")
    users_db.block_credentials(manager.db_path, USER, "demo", state["version"], "auth_expired")
    router = create_auth_router(manager.db_path, master_key=manager.master_key)
    register = next(r.endpoint for r in router.routes if r.path == "/auth/api-keys" and "POST" in r.methods)
    previous_db = pin.get_session_db_path()
    pin.set_session_db_path(manager.db_path)
    try:
        token = pin.create_session(USER)
        request = Request({"type": "http", "headers": [
            (b"cookie", f"{SESSION_COOKIE_NAME}={token}".encode("ascii")),
        ]})
        with pytest.raises(HTTPException) as raised:
            await register(ApiKeysRequest(api_key="synthetic-public", api_secret="synthetic-secret"), request)
        assert raised.value.status_code == 400
    finally:
        pin.set_session_db_path(previous_db)
    assert users_db.get_credential_state(manager.db_path, USER, "demo") == {
        "version": state["version"], "stop_reason": "auth_expired",
    }


@pytest.mark.asyncio
async def test_cursus_keeps_existing_runtime_and_nonpersistent_loss_behavior(manager):
    users_db.set_last_model(manager.db_path, USER, CURSUS_MODEL_NAME)
    bot = BotTrendInstance(client=None, symbol=ETH)
    settings = manager._build_user_settings(USER, ETH)
    manager._slots[(USER, ETH)] = _UserBotSlot(ETH, settings, bot)
    await endpoint(manager, "/ict/daily_loss_limit")(DailyLossLimitRequest(pct=50), user_code=USER)
    assert bot.daily_loss_limit_pct == settings.daily_loss_limit_pct == 50
    assert users_db.get_user_by_code(manager.db_path, USER)["daily_loss_limit_pct"] is None
    await endpoint(manager, "/ict/timeframe")(TimeframeRequest(timeframe="15m"), user_code=USER)
    assert bot.timeframe == "1h"
    assert bot.state is BotState.STOPPED
    assert manager.base_settings.timeframe == "15m"


@pytest.mark.asyncio
async def test_cursus_defer_does_not_inject_origo_lifecycle_fields(manager):
    users_db.set_last_model(manager.db_path, USER, CURSUS_MODEL_NAME)
    bot = BotTrendInstance(client=None, symbol=ETH)
    pending = _PendingLimitEntry(Direction.LONG, 100, 2, 98, "synthetic-order", 1000, 4000)
    bot._pending_limit = pending
    manager._slots[(USER, ETH)] = _UserBotSlot(ETH, manager._build_user_settings(USER, ETH), bot)
    result = await endpoint(manager, "/ict/model")(
        ModelRequest(model=ORIGO_MODEL_NAME, on_position="defer"), user_code=USER,
    )
    assert result["deferred"] == [ETH]
    assert manager._slots[(USER, ETH)].bot is bot
    assert bot._pending_limit is pending
    assert users_db.get_deferred_models(manager.db_path, USER) == {}
