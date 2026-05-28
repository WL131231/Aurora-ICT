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


# ============================================================
# 4. status timeframe 필드 (2026-05-28 UI Trade TF 토글 버그 fix)
# ============================================================


@pytest.mark.asyncio
async def test_status_includes_timeframe_when_no_slot(
    db_path, base_settings, master_key,
) -> None:
    """status — 슬롯 없을 때도 base_settings.timeframe 응답.

    UI 의 trade-tf 토글 (app.js b.dataset.tradeTf === s.timeframe) 이 작동하려면
    state=stopped 일 때도 timeframe 필드가 필요. base_settings 의 기본 TF 반영.
    """
    code = "AICT-TFST-TFST-TFST"
    users_db.create_user(db_path, code)
    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    st = await mu.status(code)
    assert "timeframe" in st
    assert st["timeframe"] == base_settings.timeframe


@pytest.mark.asyncio
async def test_status_includes_timeframe_when_slot_active(
    db_path, base_settings, master_key,
) -> None:
    """status — slot 활성 상태에서도 settings.timeframe 응답."""
    code = "AICT-TFRN-TFRN-TFRN"
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
    assert "timeframe" in st
    assert st["timeframe"] == base_settings.timeframe
    await mu.stop(code)


@pytest.mark.asyncio
async def test_status_timeframe_default_when_base_settings_none(
    db_path, master_key,
) -> None:
    """status — base_settings=None 이어도 timeframe 필드 누락 X (IctSettings 기본 TF)."""
    code = "AICT-TFDF-TFDF-TFDF"
    users_db.create_user(db_path, code)
    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=None,
        master_key=master_key,
    )
    st = await mu.status(code)
    assert "timeframe" in st
    assert st["timeframe"] == IctSettings().timeframe


# ============================================================
# 5. ensure_bot_ready (2026-05-28 차트 lazy 로딩)
# ============================================================


@pytest.mark.asyncio
async def test_ensure_bot_ready_raises_when_no_api_keys(
    db_path, base_settings, master_key,
) -> None:
    """ensure_bot_ready — API 키 미등록 시 ValueError (호출자가 404 매핑)."""
    code = "AICT-EBNK-EBNK-EBNK"
    users_db.create_user(db_path, code)
    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    with pytest.raises(ValueError, match="API"):
        await mu.ensure_bot_ready(code)


@pytest.mark.asyncio
async def test_ensure_bot_ready_creates_bot_and_starts_prefetch_without_start(
    db_path, base_settings, master_key,
) -> None:
    """ensure_bot_ready — 봇 인스턴스 생성 + prefetch 시작, state 는 STOPPED 유지.

    START 누르기 전에도 차트 봉/마커 데이터 받을 수 있도록 cache prefetch 만 가동.
    state 가 STOPPED 임이 핵심 — 매매 loop 안 돔.
    """
    code = "AICT-EBOK-EBOK-EBOK"
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
    bot = await mu.ensure_bot_ready(code)
    # 봇 인스턴스 존재 + STOPPED 유지 (start() 호출 X).
    assert bot is not None
    assert bot.state is BotState.STOPPED
    # prefetch task 생성됨.
    assert bot._prefetch_task is not None
    # set_leverage 는 호출되지 않음 (start 와 별개).
    assert clients[0].leverage_calls == []


@pytest.mark.asyncio
async def test_ensure_bot_ready_is_idempotent(
    db_path, base_settings, master_key,
) -> None:
    """ensure_bot_ready 재호출 — 같은 봇 인스턴스, prefetch 중복 시작 X."""
    code = "AICT-EBID-EBID-EBID"
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
    bot1 = await mu.ensure_bot_ready(code)
    first_task = bot1._prefetch_task
    bot2 = await mu.ensure_bot_ready(code)
    assert bot1 is bot2
    # 첫 task 가 아직 진행 중이면 같은 task — 재시작 X (idempotent).
    if first_task is not None and not first_task.done():
        assert bot2._prefetch_task is first_task


# ============================================================
# 6. 봇 가동 상태 영속화 (2026-05-28 Fly OOM/재배포 자동 복원)
# ============================================================


@pytest.mark.asyncio
async def test_start_persists_bot_running_flag(
    db_path, base_settings, master_key,
) -> None:
    """start — 성공 시 users.bot_running=1 DB 박힘."""
    code = "AICT-BRST-BRST-BRST"
    users_db.create_user(db_path, code)
    enc = keystore.encrypt_secret("plain", key=master_key)
    users_db.set_api_keys(db_path, code, "pub", enc)

    # 시작 전 — 0.
    user_before = users_db.get_user_by_code(db_path, code)
    assert user_before["bot_running"] == 0

    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    await mu.start(code)

    # 시작 후 — 1.
    user_after = users_db.get_user_by_code(db_path, code)
    assert user_after["bot_running"] == 1
    assert users_db.list_running_codes(db_path) == [code]

    await mu.stop(code)


@pytest.mark.asyncio
async def test_stop_clears_bot_running_flag(
    db_path, base_settings, master_key,
) -> None:
    """stop — DB bot_running=0 + list_running_codes 에서 빠짐."""
    code = "AICT-BRSP-BRSP-BRSP"
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
    assert users_db.get_user_by_code(db_path, code)["bot_running"] == 1

    await mu.stop(code)
    assert users_db.get_user_by_code(db_path, code)["bot_running"] == 0
    assert users_db.list_running_codes(db_path) == []


# ============================================================
# 2026-05-29: Live 모드 지원 — demo/live 키 분리 + run_mode 검증
# ============================================================


@pytest.mark.asyncio
async def test_build_settings_loads_both_demo_and_live_keys(
    db_path, base_settings, master_key,
) -> None:
    """_build_user_settings — demo/live 모두 등록되어 있으면 양쪽 슬롯 채움."""
    code = "AICT-DUAL-DUAL-DUAL"
    users_db.create_user(db_path, code)
    demo_enc = keystore.encrypt_secret("demo_plain", key=master_key)
    live_enc = keystore.encrypt_secret("live_plain", key=master_key)
    users_db.set_api_keys(db_path, code, "demo_pub", demo_enc, mode="demo")
    users_db.set_api_keys(db_path, code, "live_pub", live_enc, mode="live")

    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    settings = mu._build_user_settings(code)
    # demo / live 슬롯 모두 채워져야 — 모드 전환이 사용자별로 자유롭게 가능.
    assert settings.demo_api_key.get_secret_value() == "demo_pub"
    assert settings.demo_api_secret.get_secret_value() == "demo_plain"
    assert settings.live_api_key.get_secret_value() == "live_pub"
    assert settings.live_api_secret.get_secret_value() == "live_plain"


@pytest.mark.asyncio
async def test_build_settings_live_mode_requires_live_key(
    db_path, master_key,
) -> None:
    """run_mode=LIVE 인데 live 키 없으면 ValueError (demo 키만 있어도 차단)."""
    from aurora_ict.config.settings import RunMode
    code = "AICT-LVNO-LVNO-LVNO"
    users_db.create_user(db_path, code)
    # demo 만 등록.
    demo_enc = keystore.encrypt_secret("demo_plain", key=master_key)
    users_db.set_api_keys(db_path, code, "demo_pub", demo_enc, mode="demo")

    base = IctSettings(
        enabled=True,
        step_interval_sec=3600,
        run_mode=RunMode.LIVE,  # LIVE 인데 live 키 없는 상황.
    )
    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base,
        master_key=master_key,
    )
    with pytest.raises(ValueError, match="LIVE"):
        mu._build_user_settings(code)


@pytest.mark.asyncio
async def test_status_reports_both_credentials_flags(
    db_path, base_settings, master_key,
) -> None:
    """status — has_demo_credentials / has_live_credentials 각각 노출."""
    code = "AICT-STAT-STAT-STAT"
    users_db.create_user(db_path, code)
    clients: list[FakeExchangeClient] = []
    mu = MultiUserBotManager(
        client_factory=_factory_factory(clients),
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )
    # 키 미등록 — 둘 다 False.
    st = await mu.status(code)
    assert st["has_demo_credentials"] is False
    assert st["has_live_credentials"] is False
    assert st["has_credentials"] is False  # 현재 모드 (default demo) 기준

    # demo 만 등록.
    demo_enc = keystore.encrypt_secret("demo_plain", key=master_key)
    users_db.set_api_keys(db_path, code, "demo_pub", demo_enc, mode="demo")
    st = await mu.status(code)
    assert st["has_demo_credentials"] is True
    assert st["has_live_credentials"] is False
    assert st["has_credentials"] is True  # default demo 모드라서

    # live 도 등록.
    live_enc = keystore.encrypt_secret("live_plain", key=master_key)
    users_db.set_api_keys(db_path, code, "live_pub", live_enc, mode="live")
    st = await mu.status(code)
    assert st["has_demo_credentials"] is True
    assert st["has_live_credentials"] is True
