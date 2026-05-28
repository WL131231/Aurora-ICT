"""SaaS multi-user 분기 — /ict/* 전체 endpoint 통합 테스트 (mock 0).

검증 시나리오:
    1. 인증 없이 호출 시 401 — 모든 보호 endpoint
    2. /ict/health 는 인증 없이 200
    3. 인증 + 봇 미가동 — config/daily_loss_limit/position/judgment/equity/
       closed_pnl 모두 200 + 그럴듯한 응답
    4. markers/ohlcv — 봇 미가동 시 404
    5. start 후 — markers/ohlcv 200
    6. timeframe 변경 — 허용 목록 검증 + 가동 중 재시작
    7. credentials POST — multi-user 에서는 400 (auth 경로 안내)
    8. run-mode / enabled 토글
    9. 2명 사용자 격리 — 각자 봇 상태 독립

mock 0: Fake ExchangeClient (Protocol 만족) + 실제 SQLite + Fernet.
세션은 모듈 전역 dict → 각 test 시작 시 reset.

담당: 지영민 (SaaS 1차 출시 PR — 누락 endpoint 보완)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from aurora_ict.api.app import create_app
from aurora_ict.auth import pin, users_db
from aurora_ict.bot.multi_user_manager import MultiUserBotManager
from aurora_ict.config.settings import IctSettings

# ============================================================
# Fake ExchangeClient — Protocol 만족 + OHLCV 데이터 제공 (mock 0)
# ============================================================


def _synthetic_bars(n: int = 200) -> list[list[Any]]:
    """결정론적 OHLCV — markers/ohlcv endpoint 가 200 응답 가능하도록."""
    start_ms = 1778598000000  # 2026-05-12 14:00 UTC = NY 10:00 EDT
    rows: list[list[Any]] = []
    price = 100.0
    for i in range(n):
        o = price
        h = price + 5
        lo = price - 5
        cl = price + (1 if i % 2 == 0 else -1)
        rows.append([start_ms + i * 60_000, o, h, lo, cl, 100.0])
        price = cl
    return rows


class FakeExchangeClient:
    """ExchangeClientProtocol 만족 + fetch_ohlcv 에 200봉 반환."""

    def __init__(self) -> None:
        self.placed_orders: list[dict[str, Any]] = []
        self.leverage_calls: list[tuple[str, int]] = []
        self._bars = _synthetic_bars(200)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int,
    ) -> list[list[Any]]:
        return self._bars[-limit:] if limit > 0 else self._bars

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
        return {
            "orderId": "FAKE", "filled_qty": qty, "avg_fill_price": price or 0.0,
        }

    async def fetch_position(self, symbol: str) -> dict[str, Any] | None:
        return None

    async def fetch_balance(self) -> dict[str, Any]:
        return {"USDT": {"total": 1234.56}}

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
    return IctSettings(
        enabled=True,
        step_interval_sec=3600,  # background loop 거의 안 돌게
        fvg_min_size_pct=0.001,
        min_rr=1.0,
    )


@pytest.fixture(autouse=True)
def _reset_sessions() -> Iterator[None]:
    pin.revoke_all_sessions()
    yield
    pin.revoke_all_sessions()


@pytest.fixture
def created_clients() -> list[FakeExchangeClient]:
    return []


@pytest.fixture
def mu(db_path, base_settings, master_key, created_clients) -> MultiUserBotManager:
    async def factory(_settings: IctSettings) -> FakeExchangeClient:
        c = FakeExchangeClient()
        created_clients.append(c)
        return c

    return MultiUserBotManager(
        client_factory=factory,
        db_path=db_path,
        base_settings=base_settings,
        master_key=master_key,
    )


@pytest.fixture
def client(mu, db_path, master_key) -> TestClient:
    app = create_app(
        multi_user=True,
        multi_user_manager=mu,
        auth_db_path=db_path,
        secure_cookie=False,
        master_key=master_key,
    )
    return TestClient(app)


def _register_user(
    client: TestClient, code: str = "AICT-SAAS-SAAS-SAAS",
) -> None:
    """setup-pin + api-keys 세팅 — 이후 cookie 가 client 에 박힘."""
    r = client.post(
        "/auth/setup-pin",
        json={"code": code, "pin": "Aa1!aaaa", "pin_confirm": "Aa1!aaaa"},
    )
    assert r.status_code == 200, r.text
    r = client.post(
        "/auth/api-keys",
        json={"api_key": "pub_xx", "api_secret": "sec_xx"},
    )
    assert r.status_code == 200, r.text


# ============================================================
# 1. 인증 dependency — 모든 보호 endpoint 가 401 반환
# ============================================================


PROTECTED_GET = [
    "/ict/status",
    "/ict/config",
    "/ict/daily_loss_limit",
    "/ict/position",
    "/ict/judgment",
    "/ict/equity",
    "/ict/closed_pnl",
    "/ict/markers",
    "/ict/ohlcv",
]

PROTECTED_POST: list[tuple[str, dict[str, Any]]] = [
    ("/ict/start", {}),
    ("/ict/stop", {}),
    ("/ict/run-mode", {"mode": "demo"}),
    ("/ict/enabled", {"enabled": True}),
    ("/ict/test-connection", {}),
    ("/ict/timeframe", {"timeframe": "1h"}),
    ("/ict/daily_loss_limit", {"pct": 5.0}),
    ("/ict/credentials", {"mode": "demo", "api_key": "x", "api_secret": "y"}),
    ("/ict/position/close", {"fraction": 1.0}),
]


@pytest.mark.parametrize("path", PROTECTED_GET)
def test_protected_get_returns_401_without_auth(
    client: TestClient, path: str,
) -> None:
    resp = client.get(path)
    assert resp.status_code == 401, f"{path} → {resp.status_code}"


@pytest.mark.parametrize(("path", "body"), PROTECTED_POST)
def test_protected_post_returns_401_without_auth(
    client: TestClient, path: str, body: dict[str, Any],
) -> None:
    resp = client.post(path, json=body)
    assert resp.status_code == 401, f"{path} → {resp.status_code}"


# ============================================================
# 2. /ict/health — 인증 없이 200
# ============================================================


def test_health_does_not_require_auth(client: TestClient) -> None:
    resp = client.get("/ict/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# ============================================================
# 3. 인증 + 봇 미가동 — config / daily_loss_limit / position / judgment /
#    equity / closed_pnl 모두 200 + 그럴듯한 응답
# ============================================================


def test_config_returns_safe_dict_after_auth(client: TestClient) -> None:
    _register_user(client)
    resp = client.get("/ict/config")
    assert resp.status_code == 200
    d = resp.json()
    # api key 평문 노출 X — has_*_credentials 만 노출.
    assert "demo_api_key" not in d
    assert "live_api_key" not in d
    assert "has_demo_credentials" in d
    assert "allowed_trade_timeframes" in d


def test_daily_loss_limit_get_when_bot_stopped(client: TestClient) -> None:
    _register_user(client)
    resp = client.get("/ict/daily_loss_limit")
    assert resp.status_code == 200
    d = resp.json()
    assert "limit_pct" in d
    assert d["today_pnl_usdt"] == 0.0
    assert d["hit"] is False


def test_position_when_bot_stopped(client: TestClient) -> None:
    _register_user(client)
    resp = client.get("/ict/position")
    assert resp.status_code == 200
    d = resp.json()
    assert d["active"] is False
    assert d["pending"] is None


def test_judgment_when_bot_stopped(client: TestClient) -> None:
    _register_user(client)
    resp = client.get("/ict/judgment")
    assert resp.status_code == 200
    d = resp.json()
    assert d["direction"]["label"] == "봇 미가동"
    assert d["reasons"] == []


def test_equity_when_bot_stopped(client: TestClient) -> None:
    _register_user(client)
    resp = client.get("/ict/equity")
    assert resp.status_code == 200
    d = resp.json()
    assert d["equity"] == 0.0
    assert d["active"] is False
    assert "session_status" in d


def test_closed_pnl_when_bot_stopped(client: TestClient) -> None:
    _register_user(client)
    resp = client.get("/ict/closed_pnl")
    assert resp.status_code == 200
    assert resp.json() == {"trades": []}


# ============================================================
# 4. markers / ohlcv — 2026-05-28 lazy 로딩: 봇 미가동·키 등록 상태면 200,
#    봇 state 는 STOPPED 유지 (매매 X). 키 미등록만 404.
# ============================================================


def test_markers_lazy_loads_when_bot_stopped_with_keys(
    client: TestClient, mu,
) -> None:
    """봇 미가동 + API 키 등록 — markers 자동 lazy 로딩 (이전 404 → 200)."""
    _register_user(client)
    resp = client.get("/ict/markers?limit=50")
    assert resp.status_code == 200
    body = resp.json()
    assert "markers" in body
    assert "count" in body
    # 봇 state 는 그대로 STOPPED — 매매 가동 X.
    st = client.get("/ict/status").json()
    assert st["state"] == "stopped"


def test_ohlcv_lazy_loads_when_bot_stopped_with_keys(
    client: TestClient, mu,
) -> None:
    """봇 미가동 + API 키 등록 — ohlcv 자동 lazy 로딩 (이전 404 → 200)."""
    _register_user(client)
    resp = client.get("/ict/ohlcv?limit=50")
    assert resp.status_code == 200
    body = resp.json()
    assert body["timeframe"]
    assert len(body["candles"]) > 0
    # 봇 state 는 그대로 STOPPED.
    st = client.get("/ict/status").json()
    assert st["state"] == "stopped"


def test_markers_404_when_no_api_keys(client: TestClient, db_path) -> None:
    """API 키 미등록 사용자 — markers 는 404 (ensure_bot_ready ValueError 전파)."""
    code = "AICT-NOKM-NOKM-NOKM"
    users_db.create_user(db_path, code)
    # PIN 만 세팅, api-keys 는 안 함.
    r = client.post(
        "/auth/setup-pin",
        json={"code": code, "pin": "Aa1!aaaa", "pin_confirm": "Aa1!aaaa"},
    )
    assert r.status_code == 200
    resp = client.get("/ict/markers")
    assert resp.status_code == 404


def test_ohlcv_404_when_no_api_keys(client: TestClient, db_path) -> None:
    """API 키 미등록 사용자 — ohlcv 는 404 (ensure_bot_ready ValueError 전파)."""
    code = "AICT-NOKO-NOKO-NOKO"
    users_db.create_user(db_path, code)
    r = client.post(
        "/auth/setup-pin",
        json={"code": code, "pin": "Aa1!aaaa", "pin_confirm": "Aa1!aaaa"},
    )
    assert r.status_code == 200
    resp = client.get("/ict/ohlcv")
    assert resp.status_code == 404


# ============================================================
# 5. start 후 — markers/ohlcv 200, status running
# ============================================================


def test_start_then_markers_and_ohlcv_ok(client: TestClient) -> None:
    _register_user(client)
    r = client.post("/ict/start")
    assert r.status_code == 200
    assert r.json()["state"] == "running"

    r = client.get("/ict/ohlcv?limit=50")
    assert r.status_code == 200
    body = r.json()
    assert body["timeframe"]
    assert len(body["candles"]) > 0

    r = client.get("/ict/markers?limit=50")
    assert r.status_code == 200
    body = r.json()
    assert "markers" in body
    assert "count" in body

    # cleanup
    client.post("/ict/stop")


def test_start_then_judgment_has_session_reason(client: TestClient) -> None:
    """봇 가동 후 judgment — reasons 에 최소 Session 항목 1개 포함."""
    _register_user(client)
    client.post("/ict/start")
    r = client.get("/ict/judgment")
    assert r.status_code == 200
    body = r.json()
    terms = [x["term"] for x in body["reasons"]]
    assert any(t in ("Killzone", "US Session", "Session") for t in terms)
    client.post("/ict/stop")


# ============================================================
# 6. timeframe 변경
# ============================================================


def test_timeframe_change_valid(client: TestClient) -> None:
    _register_user(client)
    r = client.post("/ict/timeframe", json={"timeframe": "4h"})
    assert r.status_code == 200
    body = r.json()
    assert body["timeframe"] == "4h"
    assert body["restarted"] is False


def test_timeframe_change_invalid_400(client: TestClient) -> None:
    _register_user(client)
    r = client.post("/ict/timeframe", json={"timeframe": "7m"})
    assert r.status_code == 400


def test_timeframe_change_while_running_restarts(client: TestClient) -> None:
    _register_user(client)
    client.post("/ict/start")
    r = client.post("/ict/timeframe", json={"timeframe": "2h"})
    assert r.status_code == 200
    assert r.json()["restarted"] is True
    # 재시작 후 status 다시 running.
    st = client.get("/ict/status").json()
    assert st["state"] == "running"
    client.post("/ict/stop")


# ============================================================
# 7. credentials POST — multi-user 에서는 400 (auth 경로 안내)
# ============================================================


def test_credentials_endpoint_redirects_to_auth(client: TestClient) -> None:
    _register_user(client)
    r = client.post(
        "/ict/credentials",
        json={"mode": "demo", "api_key": "x", "api_secret": "y"},
    )
    assert r.status_code == 400
    assert "/auth/api-keys" in r.json()["detail"]


# ============================================================
# 8. run-mode / enabled 토글
# ============================================================


def test_run_mode_demo_to_live(client: TestClient) -> None:
    _register_user(client)
    r = client.post("/ict/run-mode", json={"mode": "live"})
    assert r.status_code == 200
    cfg = client.get("/ict/config").json()
    assert cfg["run_mode"] == "live"


def test_run_mode_invalid_400(client: TestClient) -> None:
    _register_user(client)
    r = client.post("/ict/run-mode", json={"mode": "fake"})
    assert r.status_code == 400


def test_enabled_false_then_true(client: TestClient) -> None:
    _register_user(client)
    # 가동 안 한 상태에서 enabled=False → state 그대로 stopped, settings.enabled=False.
    r = client.post("/ict/enabled", json={"enabled": False})
    assert r.status_code == 200
    # enabled=True → 봇 가동 (manager.start 동작).
    r = client.post("/ict/enabled", json={"enabled": True})
    assert r.status_code == 200
    assert r.json()["state"] == "running"
    client.post("/ict/stop")


# ============================================================
# 9. test-connection — fake client 가 balance 반환
# ============================================================


def test_test_connection_returns_balance(client: TestClient) -> None:
    _register_user(client)
    r = client.post("/ict/test-connection")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["balance_usdt"] == 1234.56


# ============================================================
# 10. daily_loss_limit POST — 정상/범위 초과 검증
# ============================================================


def test_daily_loss_limit_set_valid(client: TestClient) -> None:
    _register_user(client)
    r = client.post("/ict/daily_loss_limit", json={"pct": 5.0})
    assert r.status_code == 200
    assert r.json()["limit_pct"] == 5.0


def test_daily_loss_limit_set_out_of_range_400(client: TestClient) -> None:
    _register_user(client)
    r = client.post("/ict/daily_loss_limit", json={"pct": 99.0})
    assert r.status_code == 400


# ============================================================
# 11. position/close — 활성 포지션 없으면 404
# ============================================================


def test_position_close_404_when_no_position(client: TestClient) -> None:
    _register_user(client)
    client.post("/ict/start")
    r = client.post("/ict/position/close", json={"fraction": 1.0})
    assert r.status_code == 404
    client.post("/ict/stop")


def test_position_close_fraction_out_of_range_400(client: TestClient) -> None:
    _register_user(client)
    r = client.post("/ict/position/close", json={"fraction": 1.5})
    assert r.status_code == 400


# ============================================================
# 12. 2명 사용자 격리 — start/stop 독립
# ============================================================


def test_two_users_isolated_status(client: TestClient, mu) -> None:
    """두 client 가 같은 app 을 공유해도 봇 상태는 사용자별 격리."""
    app = client.app
    c1 = TestClient(app)
    c2 = TestClient(app)
    _register_user(c1, "AICT-MU01-MU01-MU01")
    _register_user(c2, "AICT-MU02-MU02-MU02")

    # c1 만 start.
    r = c1.post("/ict/start")
    assert r.status_code == 200
    assert r.json()["state"] == "running"

    # c2 는 여전히 stopped.
    s2 = c2.get("/ict/status").json()
    assert s2["state"] == "stopped"

    # c1 stop 정리.
    c1.post("/ict/stop")
