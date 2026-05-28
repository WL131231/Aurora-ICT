"""MultiUserBotManager — 사용자별 봇 격리 관리 테스트.

mock 0 정책 — Fake ExchangeClient (Protocol 만족) 직접 구현.
SQLite + Fernet 실제 사용, BotIctInstance 실제 실행 (외부 네트워크 X).

검증 시나리오:
    1. get_or_create_bot — DB 미존재 → ValueError, 정상 등록 후 인스턴스 반환
    2. 같은 user_code 재호출 — 같은 인스턴스 반환 (캐시)
    3. start / stop — bot.state 전이
    4. status — STOPPED → RUNNING → STOPPED
    5. stop_all — 모든 사용자 정지 (멱등)
    6. 사용자 격리 — 2명 → 각자 settings/bot 분리, api_secret 복호화 일치

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

from typing import Any

import pytest
from cryptography.fernet import Fernet

from aurora_ict.auth import keystore, users_db
from aurora_ict.bot.bot_ict_instance import BotState
from aurora_ict.bot.multi_user_manager import MultiUserBotManager
from aurora_ict.config.settings import IctSettings

# ============================================================
# Fake ExchangeClient — Protocol 만족 최소 구현 (mock 0)
# ============================================================


class FakeExchangeClient:
    """ExchangeClientProtocol 만족하는 최소 fake — 모든 메서드 no-op / 정적 값.

    BotIctInstance.start → _recover_position_from_exchange (fetch_position 호출)
    + _run_loop background task 가 실제로 돌지만, step_interval_sec 가 크고
    fetch_ohlcv 가 빈 list 반환이라 NO_ACTION 만 나옴 (sleep 만 돌다 cancel).
    """

    def __init__(self) -> None:
        self.placed_orders: list[dict[str, Any]] = []
        self.leverage_calls: list[tuple[str, int]] = []
        self.closed = False

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
        self.placed_orders.append({
            "symbol": symbol, "side": side, "qty": qty, "price": price,
        })
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
    """테스트용 기본 settings — step_interval 길게 둬서 background loop 거의 안 돌게."""
    return IctSettings(
        enabled=True,
        step_interval_sec=3600,  # 1시간 — 테스트 중 사실상 1회만 시도
        fvg_min_size_pct=0.001,
        min_rr=1.0,
    )


def _factory_factory(created_clients: list[FakeExchangeClient]):
    """factory of factory — 매 client 생성 시 created_clients 에 append."""
    async def factory(_settings: IctSettings) -> FakeExchangeClient:
        c = FakeExchangeClient()
        created_clients.append(c)
        return c
    return factory


# ============================================================
# 1. get_or_create_bot
# ============================================================


@pytest.mark.asyncio
async def test_get_or_create_raises_when_user_missing(
    db_path, base_settings, master_key,
) -> None:
    """get_or_create_bot — 사용자 미존재 시 ValueError."""
    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    with pytest.raises(ValueError, match="DB"):
        await mu.get_or_create_bot("AICT-NONE-NONE-NONE")


@pytest.mark.asyncio
async def test_get_or_create_raises_when_no_api_keys(
    db_path, base_settings, master_key,
) -> None:
    """get_or_create_bot — 사용자는 있지만 api_key 미등록 시 ValueError."""
    users_db.create_user(db_path, "AICT-NOKY-NOKY-NOKY")
    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    with pytest.raises(ValueError, match="API"):
        await mu.get_or_create_bot("AICT-NOKY-NOKY-NOKY")


@pytest.mark.asyncio
async def test_get_or_create_returns_same_instance_on_repeat(
    db_path, base_settings, master_key,
) -> None:
    """같은 user_code 재호출 — 같은 인스턴스 (캐시)."""
    code = "AICT-ONCE-ONCE-ONCE"
    users_db.create_user(db_path, code)
    enc = keystore.encrypt_secret("plain_secret", key=master_key)
    users_db.set_api_keys(db_path, code, "pub", enc)

    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    bot1 = await mu.get_or_create_bot(code)
    bot2 = await mu.get_or_create_bot(code)
    assert bot1 is bot2
    # client 도 1개만 생성 (재호출 캐시).
    assert len(clients) == 1


# ============================================================
# 2. start / stop / status
# ============================================================


@pytest.mark.asyncio
async def test_start_then_stop_transitions(
    db_path, base_settings, master_key,
) -> None:
    """start → state=RUNNING, stop → STOPPED, set_leverage 호출 확인."""
    code = "AICT-STRT-STRT-STRT"
    users_db.create_user(db_path, code)
    enc = keystore.encrypt_secret("plain", key=master_key)
    users_db.set_api_keys(db_path, code, "pub", enc)

    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    await mu.start(code)
    bot = await mu.get_or_create_bot(code)
    assert bot.state is BotState.RUNNING
    # set_leverage 가 사용자 settings 의 symbol/leverage 로 호출됨.
    assert clients[0].leverage_calls
    sym, lev = clients[0].leverage_calls[0]
    assert sym == base_settings.symbol
    assert lev == base_settings.leverage

    await mu.stop(code)
    assert bot.state is BotState.STOPPED


@pytest.mark.asyncio
async def test_status_reports_stopped_when_no_slot(
    db_path, base_settings, master_key,
) -> None:
    """status — 슬롯 없는 사용자도 stopped 응답 (has_credentials 만 DB 조회)."""
    code = "AICT-NOSLOT-XXXX"
    users_db.create_user(db_path, code)
    # api 키 없음.
    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    st = await mu.status(code)
    assert st["state"] == "stopped"
    assert st["has_credentials"] is False
    assert st["has_active_position"] is False


@pytest.mark.asyncio
async def test_status_reports_running_after_start(
    db_path, base_settings, master_key,
) -> None:
    """status — start 직후 RUNNING + has_credentials=True."""
    code = "AICT-RUNS-RUNS-RUNS"
    users_db.create_user(db_path, code)
    enc = keystore.encrypt_secret("plain", key=master_key)
    users_db.set_api_keys(db_path, code, "pub", enc)

    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    await mu.start(code)
    st = await mu.status(code)
    assert st["state"] == "running"
    assert st["has_credentials"] is True

    await mu.stop(code)


# ============================================================
# 3. stop_all + 사용자 격리
# ============================================================


@pytest.mark.asyncio
async def test_two_users_isolated_bots(
    db_path, base_settings, master_key,
) -> None:
    """2명 사용자 — 각자 봇 인스턴스 분리, secret 복호화도 독립."""
    users_db.create_user(db_path, "AICT-IS01-IS01-IS01")
    users_db.create_user(db_path, "AICT-IS02-IS02-IS02")
    enc1 = keystore.encrypt_secret("secret_user1", key=master_key)
    enc2 = keystore.encrypt_secret("secret_user2", key=master_key)
    users_db.set_api_keys(db_path, "AICT-IS01-IS01-IS01", "pub1", enc1)
    users_db.set_api_keys(db_path, "AICT-IS02-IS02-IS02", "pub2", enc2)

    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    bot1 = await mu.get_or_create_bot("AICT-IS01-IS01-IS01")
    bot2 = await mu.get_or_create_bot("AICT-IS02-IS02-IS02")
    assert bot1 is not bot2
    assert len(clients) == 2  # 각자 client 1개씩

    # list_users 양쪽 코드 다 보임.
    listed = sorted(mu.list_users())
    assert listed == ["AICT-IS01-IS01-IS01", "AICT-IS02-IS02-IS02"]


@pytest.mark.asyncio
async def test_stop_all_stops_every_user(
    db_path, base_settings, master_key,
) -> None:
    """stop_all — 모든 사용자 봇 정지 (state=STOPPED)."""
    codes = ["AICT-SA01-SA01-SA01", "AICT-SA02-SA02-SA02"]
    for c in codes:
        users_db.create_user(db_path, c)
        enc = keystore.encrypt_secret("plain", key=master_key)
        users_db.set_api_keys(db_path, c, "pub", enc)

    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    for c in codes:
        await mu.start(c)
    bots = [await mu.get_or_create_bot(c) for c in codes]
    assert all(b.state is BotState.RUNNING for b in bots)

    await mu.stop_all()
    assert all(b.state is BotState.STOPPED for b in bots)


@pytest.mark.asyncio
async def test_stop_unknown_user_is_noop(
    db_path, base_settings, master_key,
) -> None:
    """stop — 슬롯 없는 사용자도 예외 없이 통과 (멱등)."""
    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    # 예외 없이 그냥 통과.
    await mu.stop("AICT-GHST-GHST-GHST")
    await mu.stop_all()
