"""saas.auto_resume_running_bots — Fly machine 재시작 후 자동 재가동 검증.

파트너 결정 2026-05-28 — 봇 인스턴스가 in-memory dict 라 OOM/재배포 시 사라짐.
``users_db.bot_running=1`` 인 사용자만 자동 재가동되는지, 실패 사용자가 다른
사용자에게 영향을 주지 않는지 확인.

mock 0 — 실제 SQLite + Fake ExchangeClient (Protocol 만족). 외부 네트워크 X.

검증 시나리오:
    1. 가동 마킹된 사용자 1명 → 새 manager 가 자동 재가동 + slot 채워짐
    2. 가동 마킹된 사용자 2명 + STOP 마킹 사용자 1명 → 가동 마킹만 재가동
    3. 한 사용자가 ValueError (API 키 미등록) 던져도 다른 사용자는 정상 가동

담당: 지영민 (SaaS 영속화 PR)
"""

from __future__ import annotations

from typing import Any

import pytest
from cryptography.fernet import Fernet

from aurora_ict.auth import keystore, users_db
from aurora_ict.bot.bot_ict_instance import BotState
from aurora_ict.bot.multi_user_manager import MultiUserBotManager
from aurora_ict.config.settings import IctSettings
from aurora_ict.saas import auto_resume_running_bots

# ============================================================
# Fake ExchangeClient — Protocol 만족 최소 구현 (mock 0)
# multi_user_manager 테스트와 같은 패턴.
# ============================================================


class FakeExchangeClient:
    """ExchangeClientProtocol 만족하는 최소 fake — 모든 메서드 no-op / 정적 값."""

    def __init__(self) -> None:
        self.leverage_calls: list[tuple[str, int]] = []

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int,
    ) -> list[list[Any]]:
        return []

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float | None = None,
        reduce_only: bool = False,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        return {"orderId": "FAKE", "filled_qty": qty, "avg_fill_price": price or 0.0}

    async def fetch_position(self, symbol: str) -> dict[str, Any] | None:
        return None

    async def fetch_balance(self) -> dict[str, Any]:
        return {"USDT": {"total": 1000.0}}

    async def fetch_ticker(self, symbol: str) -> float | None:
        return 100.0

    async def cancel_all_orders(self, symbol: str) -> None:
        return None

    async def fetch_closed_positions(
        self, since_ms: int | None = None, limit: int = 200,
    ) -> list[Any]:
        return []

    async def modify_stop_loss(
        self, symbol: str, new_stop_loss: float,
    ) -> dict[str, Any]:
        return {"ok": True}

    async def set_position_tpsl(
        self,
        symbol: str,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        return {"retCode": 0}

    async def set_leverage(
        self, symbol: str, leverage: int,
    ) -> dict[str, Any]:
        self.leverage_calls.append((symbol, leverage))
        return {"ok": True}


# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def master_key() -> bytes:
    return Fernet.generate_key()


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "users.db"
    users_db.init_db(path)
    return path


@pytest.fixture
def base_settings() -> IctSettings:
    """테스트용 — step_interval 길게 둬서 background loop 거의 안 돌게."""
    return IctSettings(
        enabled=True,
        step_interval_sec=3600,
        fvg_min_size_pct=0.001,
        min_rr=1.0,
    )


def _factory_factory(created_clients: list[FakeExchangeClient]):
    """매 client 생성 시 created_clients 에 append."""
    async def factory(_settings: IctSettings) -> FakeExchangeClient:
        c = FakeExchangeClient()
        created_clients.append(c)
        return c
    return factory


def _register_user_with_keys(
    db_path, code: str, master_key: bytes,
) -> None:
    """헬퍼 — 사용자 등록 + API 키 등록 (가동 가능한 상태로 만듦)."""
    users_db.create_user(db_path, code)
    enc = keystore.encrypt_secret(f"sec_{code}", key=master_key)
    users_db.set_api_keys(db_path, code, f"pub_{code}", enc)


# ============================================================
# 시나리오 1 — 가동 마킹 사용자 자동 재가동
# ============================================================


@pytest.mark.asyncio
async def test_auto_resume_starts_marked_users(
    db_path, base_settings, master_key,
) -> None:
    """auto_resume — bot_running=1 사용자가 새 manager 에서 자동 재가동."""
    code = "AICT-RES1-RES1-RES1"
    _register_user_with_keys(db_path, code, master_key)
    # 이전 fly machine 에서 가동 중이었다고 가정 — DB 에 마킹.
    users_db.set_bot_running(db_path, code, True)

    # 새 machine 부팅 — 빈 슬롯 상태의 manager 생성.
    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    # 처음엔 슬롯이 비어있음.
    assert mu.list_users() == []

    stats = await auto_resume_running_bots(mu, db_path)
    assert stats == {"attempted": 1, "succeeded": 1, "failed": 0}

    # 봇이 살아있고 RUNNING 상태.
    assert mu.list_users() == [code]
    bot = await mu.get_or_create_bot(code)
    assert bot.state is BotState.RUNNING

    await mu.stop(code)


# ============================================================
# 시나리오 2 — STOP 마킹 사용자는 재가동 대상 X
# ============================================================


@pytest.mark.asyncio
async def test_auto_resume_skips_stopped_users(
    db_path, base_settings, master_key,
) -> None:
    """auto_resume — bot_running=0 사용자는 스킵, 가동 마킹만 시도."""
    _register_user_with_keys(db_path, "AICT-RUN1-RUN1-RUN1", master_key)
    _register_user_with_keys(db_path, "AICT-RUN2-RUN2-RUN2", master_key)
    _register_user_with_keys(db_path, "AICT-SKIP-SKIP-SKIP", master_key)

    users_db.set_bot_running(db_path, "AICT-RUN1-RUN1-RUN1", True)
    users_db.set_bot_running(db_path, "AICT-RUN2-RUN2-RUN2", True)
    # SKIP 사용자는 그대로 0 (STOP 상태).

    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    stats = await auto_resume_running_bots(mu, db_path)
    assert stats == {"attempted": 2, "succeeded": 2, "failed": 0}

    # 가동 마킹된 2명만 슬롯에 있음.
    assert sorted(mu.list_users()) == ["AICT-RUN1-RUN1-RUN1", "AICT-RUN2-RUN2-RUN2"]

    for c in ["AICT-RUN1-RUN1-RUN1", "AICT-RUN2-RUN2-RUN2"]:
        await mu.stop(c)


@pytest.mark.asyncio
async def test_auto_resume_noop_when_no_running_users(
    db_path, base_settings, master_key,
) -> None:
    """auto_resume — bot_running=1 사용자가 0명이면 attempted=0."""
    # 사용자는 등록되어 있지만 모두 STOP 상태.
    _register_user_with_keys(db_path, "AICT-OFF1-OFF1-OFF1", master_key)
    _register_user_with_keys(db_path, "AICT-OFF2-OFF2-OFF2", master_key)

    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    stats = await auto_resume_running_bots(mu, db_path)
    assert stats == {"attempted": 0, "succeeded": 0, "failed": 0}
    assert mu.list_users() == []


# ============================================================
# 시나리오 3 — 한 사용자 실패가 다른 사용자에게 영향 X
# ============================================================


@pytest.mark.asyncio
async def test_auto_resume_isolates_failures(
    db_path, base_settings, master_key,
) -> None:
    """auto_resume — API 키 없는 사용자 1명 실패해도 정상 사용자 1명은 가동.

    사용자 A: 등록만 하고 API 키 미등록 → mu.start 시 ValueError.
    사용자 B: 키 등록 완료 → 정상 가동.
    둘 다 bot_running=1 마킹. auto_resume 가 A 실패 후에도 B 가동 성공해야.
    """
    # A — 키 없음, 그러나 bot_running=1 (이전에 키 등록 후 가동했다가 키 revoke 된 케이스).
    users_db.create_user(db_path, "AICT-BADD-BADD-BADD")
    users_db.set_bot_running(db_path, "AICT-BADD-BADD-BADD", True)
    # B — 정상.
    _register_user_with_keys(db_path, "AICT-GOOD-GOOD-GOOD", master_key)
    users_db.set_bot_running(db_path, "AICT-GOOD-GOOD-GOOD", True)

    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    stats = await auto_resume_running_bots(mu, db_path)
    # 2명 시도, 1명 성공, 1명 실패.
    assert stats["attempted"] == 2
    assert stats["succeeded"] == 1
    assert stats["failed"] == 1

    # B 만 슬롯에 들어있음, RUNNING 상태.
    assert mu.list_users() == ["AICT-GOOD-GOOD-GOOD"]
    bot = await mu.get_or_create_bot("AICT-GOOD-GOOD-GOOD")
    assert bot.state is BotState.RUNNING

    await mu.stop("AICT-GOOD-GOOD-GOOD")
