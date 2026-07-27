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
            "reduce_only": reduce_only,
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


@pytest.fixture(autouse=True)
def _prereg_codes(db_path) -> None:
    """#SEC 2026-07-13: setup-pin 이 사전 등록 코드만 허용하므로, 이 모듈이
    쓰는 테스트 코드들을 admin 발급(set_license) 상태로 미리 심는다.
    (라이선스 봇의 /admin/user/license pre-insert 재현.)
    """
    for _code in (
        "AICT-SAAS-SAAS-SAAS", "AICT-POLL-POLL-POLL", "AICT-MULT-MULT-MULT",
        "AICT-ALLS-ALLS-ALLS", "AICT-APOS-APOS-APOS", "AICT-CLOS-CLOS-CLOS",
        "AICT-BKFL-BKFL-BKFL", "AICT-RDCO-RDCO-RDCO", "AICT-FLAT-FLAT-FLAT",
        "AICT-MU01-MU01-MU01", "AICT-MU02-MU02-MU02",
    ):
        users_db.set_license(db_path, code=_code, license_type="referral",
                             expires_at=None)


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
    with_live_key: bool = True,
) -> None:
    """setup-pin + api-keys 세팅 — 이후 cookie 가 client 에 박힘.

    with_live_key=True(기본): demo+live 둘 다 등록 — 첫 가동이 LIVE 기본
    (2026-06-06 DEMO UI 제거)이라 live 키 필요. False 면 demo 만 (live 키 가드 테스트용).
    """
    r = client.post(
        "/auth/setup-pin",
        json={"code": code, "pin": "Aa1!aaaa", "pin_confirm": "Aa1!aaaa"},
    )
    assert r.status_code == 200, r.text
    modes = ("demo", "live") if with_live_key else ("demo",)
    for _mode in modes:
        r = client.post(
            "/auth/api-keys",
            json={"api_key": "pub_xx", "api_secret": "sec_xx", "mode": _mode},
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
    """LIVE 전환은 live 슬롯 키가 따로 등록돼 있어야 성공 (2026-05-29).

    _register_user 는 demo 키만 등록 → 바로 LIVE 전환 시 400 (해당 모드 키 없음).
    Live 키 별도 등록 후엔 200.
    """
    _register_user(client, with_live_key=False)
    # demo 키만 있는 상태에서 LIVE 전환 시도 → 안전 가드로 400.
    r = client.post("/ict/run-mode", json={"mode": "live"})
    assert r.status_code == 400
    assert "LIVE" in r.json()["detail"]

    # live 키 등록 후 재시도 → 200.
    r = client.post(
        "/auth/api-keys",
        json={"api_key": "live_pub", "api_secret": "live_sec", "mode": "live"},
    )
    assert r.status_code == 200, r.text

    r = client.post("/ict/run-mode", json={"mode": "live"})
    assert r.status_code == 200
    cfg = client.get("/ict/config").json()
    assert cfg["run_mode"] == "live"


def test_run_mode_live_blocked_when_no_live_key(client: TestClient) -> None:
    """LIVE 전환 안전 가드 — live 키 등록되지 않은 사용자는 400 (2026-05-29)."""
    _register_user(client, with_live_key=False)  # demo 키만 등록
    r = client.post("/ict/run-mode", json={"mode": "live"})
    assert r.status_code == 400
    assert "LIVE" in r.json()["detail"]


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


def test_daily_profit_limit_set_valid(client: TestClient) -> None:
    """일일 수익(TP) 한도 — 2026-06-10 조윤 건의."""
    _register_user(client)
    r = client.post("/ict/daily_profit_limit", json={"pct": 8.0})
    assert r.status_code == 200
    assert r.json()["profit_limit_pct"] == 8.0


def test_daily_profit_limit_out_of_range_400(client: TestClient) -> None:
    """수익 한도는 0~100 — 초과 시 400."""
    _register_user(client)
    r = client.post("/ict/daily_profit_limit", json={"pct": 150.0})
    assert r.status_code == 400


def test_daily_limit_set_does_not_pollute_base_settings(
    client: TestClient, mu, base_settings,
) -> None:
    """2026-06-11 리뷰 수정: 슬롯 없는 사용자의 한도 설정이 base_settings(전
    사용자 공유)를 오염시키지 않는다 — 크로스 테넌트 누출 방지."""
    _register_user(client, "AICT-POLL-POLL-POLL")
    before_loss = base_settings.daily_loss_limit_pct
    before_profit = base_settings.daily_profit_limit_pct
    assert client.post("/ict/daily_loss_limit", json={"pct": 9.0}).status_code == 200
    assert client.post("/ict/daily_profit_limit", json={"pct": 7.0}).status_code == 200
    assert base_settings.daily_loss_limit_pct == before_loss
    assert base_settings.daily_profit_limit_pct == before_profit


def test_daily_limit_set_applies_to_all_user_slots(
    client: TestClient, mu,
) -> None:
    """한도 설정이 BTC 만이 아니라 사용자의 모든 가동 페어 봇에 반영."""
    code = "AICT-ALLS-ALLS-ALLS"
    _register_user(client, code)
    client.post("/ict/start?symbol=BTC/USDT:USDT")
    client.post("/ict/start?symbol=ETH/USDT:USDT")
    assert client.post("/ict/daily_loss_limit", json={"pct": 6.5}).status_code == 200
    for sym in ("BTC/USDT:USDT", "ETH/USDT:USDT"):
        bot = mu._slots[(code, sym)].bot
        assert bot.daily_loss_limit_pct == 6.5
    client.post("/ict/stop?symbol=BTC/USDT:USDT")
    client.post("/ict/stop?symbol=ETH/USDT:USDT")


def test_daily_loss_limit_get_includes_profit_fields(client: TestClient) -> None:
    """GET 응답에 profit_limit_pct/profit_hit 동봉(UI TP Limit 표시용)."""
    _register_user(client)
    d = client.get("/ict/daily_loss_limit").json()
    assert "profit_limit_pct" in d
    assert "profit_hit" in d


def test_set_preferences_valid(client: TestClient) -> None:
    """언어/시간대 저장 — 텔레그램 알림 출력용 (2026-06-10)."""
    _register_user(client)
    r = client.post(
        "/ict/preferences",
        json={"language": "en", "timezone": "America/New_York"},
    )
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_set_preferences_invalid_language_400(client: TestClient) -> None:
    _register_user(client)
    r = client.post("/ict/preferences", json={"language": "xx"})
    assert r.status_code == 400


def test_set_preferences_invalid_timezone_400(client: TestClient) -> None:
    _register_user(client)
    r = client.post("/ict/preferences", json={"timezone": "Not/AZone"})
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


# ============================================================
# 13. PR C — 멀티 페어 endpoint (?symbol + /ict/running-pairs)
# ============================================================


def test_start_with_symbol_query_starts_specified_pair(
    client: TestClient,
) -> None:
    """/ict/start?symbol=ETH/USDT:USDT — ETH 슬롯 가동, BTC 는 영향 X."""
    _register_user(client)
    # ETH 단독 가동.
    r = client.post("/ict/start?symbol=ETH/USDT:USDT")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["state"] == "running"
    assert body["symbol"] == "ETH/USDT:USDT"

    # running-pairs 에 ETH 만.
    r = client.get("/ict/running-pairs")
    assert r.status_code == 200
    assert r.json()["running_symbols"] == ["ETH/USDT:USDT"]

    # BTC status 조회 — 아직 stopped.
    r = client.get("/ict/status?symbol=BTC/USDT:USDT")
    assert r.status_code == 200
    assert r.json()["state"] == "stopped"

    # cleanup
    client.post("/ict/stop?symbol=ETH/USDT:USDT")


def test_running_pairs_lists_all_active_symbols(
    client: TestClient,
) -> None:
    """/ict/running-pairs — BTC + ETH 동시 가동 시 둘 다 반환."""
    _register_user(client)
    # 둘 다 가동.
    assert client.post("/ict/start?symbol=BTC/USDT:USDT").status_code == 200
    assert client.post("/ict/start?symbol=ETH/USDT:USDT").status_code == 200

    r = client.get("/ict/running-pairs")
    assert r.status_code == 200
    syms = set(r.json()["running_symbols"])
    assert syms == {"BTC/USDT:USDT", "ETH/USDT:USDT"}

    # BTC 만 정지 — ETH 는 그대로.
    assert client.post("/ict/stop?symbol=BTC/USDT:USDT").status_code == 200
    r = client.get("/ict/running-pairs")
    assert r.json()["running_symbols"] == ["ETH/USDT:USDT"]

    # cleanup
    client.post("/ict/stop?symbol=ETH/USDT:USDT")


def test_stop_with_symbol_query_does_not_affect_other_pair(
    client: TestClient,
) -> None:
    """/ict/stop?symbol=BTC 만 정지하고 ETH 는 계속 가동."""
    _register_user(client)
    client.post("/ict/start?symbol=BTC/USDT:USDT")
    client.post("/ict/start?symbol=ETH/USDT:USDT")

    # BTC 만 정지.
    r = client.post("/ict/stop?symbol=BTC/USDT:USDT")
    assert r.status_code == 200

    # BTC status = stopped, ETH status = running 유지.
    btc = client.get("/ict/status?symbol=BTC/USDT:USDT").json()
    eth = client.get("/ict/status?symbol=ETH/USDT:USDT").json()
    assert btc["state"] == "stopped"
    assert eth["state"] == "running"

    # cleanup
    client.post("/ict/stop?symbol=ETH/USDT:USDT")


def test_running_pairs_requires_auth(client: TestClient) -> None:
    """미인증 → 401."""
    r = client.get("/ict/running-pairs")
    assert r.status_code == 401


def _inject_active(bot: Any, direction: Any, entry: float, qty: float) -> None:
    """봇에 활성 포지션 주입 — 멀티 페어 표시 테스트용(실제 진입 로직 우회)."""
    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    bot.active_position = _ActivePosition(
        direction=direction,
        entry=entry,
        stop_loss=entry * 1.01,
        take_profit=entry * 0.95,
        qty=qty,
        setup_ts_ms=1778598000000,
    )


def test_position_lists_all_pairs(client: TestClient, mu) -> None:
    """2026-06-10 버그픽스: BTC+ETH 동시 포지션이 모두 positions 에 보여야.

    기존엔 기본 심볼(BTC) 슬롯만 조회해 ETH 가 UI 에서 사라졌다.
    """
    from aurora_ict.strategy.silver_bullet import Direction
    code = "AICT-MULT-MULT-MULT"
    _register_user(client, code)
    assert client.post("/ict/start?symbol=BTC/USDT:USDT").status_code == 200
    assert client.post("/ict/start?symbol=ETH/USDT:USDT").status_code == 200

    btc = mu._slots[(code, "BTC/USDT:USDT")].bot
    eth = mu._slots[(code, "ETH/USDT:USDT")].bot
    _inject_active(btc, Direction.SHORT, 100.0, 1.0)
    _inject_active(eth, Direction.LONG, 50.0, 2.0)

    r = client.get("/ict/position")
    assert r.status_code == 200
    d = r.json()
    syms = {p["symbol"] for p in d["positions"]}
    assert syms == {"BTC/USDT:USDT", "ETH/USDT:USDT"}
    # 둘 다 active.
    assert all(p["active"] for p in d["positions"])
    # 레거시 단일 필드는 기본 심볼(BTC) 기준.
    assert d["symbol"] == "BTC/USDT:USDT"

    client.post("/ict/stop?symbol=BTC/USDT:USDT")
    client.post("/ict/stop?symbol=ETH/USDT:USDT")


def test_admin_all_positions_lists_and_flags_risky(
    client: TestClient, mu, monkeypatch,
) -> None:
    """2026-06-12 파트너 요청: 전 사용자 포지션 한눈에 + SL>청산가 경고.

    20x 청산 거리 ≈ 4.75% — SL 6% 밖이면 sl_beyond_liq=True (#LIQ-CAP 사고 탐지).
    """
    from aurora_ict.strategy.silver_bullet import Direction
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "admin-tok")
    code = "AICT-APOS-APOS-APOS"
    _register_user(client, code)
    client.post("/ict/start?symbol=BTC/USDT:USDT")
    client.post("/ict/start?symbol=ETH/USDT:USDT")
    btc = mu._slots[(code, "BTC/USDT:USDT")].bot
    eth = mu._slots[(code, "ETH/USDT:USDT")].bot
    # BTC: 안전한 SL (1% — 청산 4.75% 안쪽). ETH: 위험한 SL (숏인데 +6% 위).
    _inject_active(btc, Direction.LONG, 100.0, 1.0)
    btc.active_position.stop_loss = 99.0
    _inject_active(eth, Direction.SHORT, 100.0, 1.0)
    eth.active_position.stop_loss = 106.0  # 청산(≈104.75) 너머 — 위험

    # 미인증 → 401
    assert client.get("/admin/positions").status_code in (401, 503)
    r = client.get("/admin/positions", headers={"X-Admin-Token": "admin-tok"})
    assert r.status_code == 200
    d = r.json()
    assert d["count"] == 2
    assert d["risky"] == 1
    # 위험 포지션이 먼저 정렬
    first = d["positions"][0]
    assert first["symbol"] == "ETH/USDT:USDT"
    assert first["sl_beyond_liq"] is True
    assert d["positions"][1]["sl_beyond_liq"] is False

    client.post("/ict/stop?symbol=BTC/USDT:USDT")
    client.post("/ict/stop?symbol=ETH/USDT:USDT")


def test_position_close_routes_by_symbol(client: TestClient, mu) -> None:
    """청산 ?symbol — ETH 청산이 BTC 포지션을 건드리지 않아야."""
    from aurora_ict.strategy.silver_bullet import Direction
    code = "AICT-CLOS-CLOS-CLOS"
    _register_user(client, code)
    client.post("/ict/start?symbol=BTC/USDT:USDT")
    client.post("/ict/start?symbol=ETH/USDT:USDT")
    btc = mu._slots[(code, "BTC/USDT:USDT")].bot
    eth = mu._slots[(code, "ETH/USDT:USDT")].bot
    _inject_active(btc, Direction.SHORT, 100.0, 1.0)
    _inject_active(eth, Direction.LONG, 50.0, 2.0)

    # ETH 전체 청산.
    r = client.post(
        "/ict/position/close", json={"fraction": 1.0, "symbol": "ETH/USDT:USDT"},
    )
    assert r.status_code == 200
    assert r.json()["active"] is False
    # ETH 는 비고 BTC 는 그대로.
    assert eth.active_position is None
    assert btc.active_position is not None

    client.post("/ict/stop?symbol=BTC/USDT:USDT")
    client.post("/ict/stop?symbol=ETH/USDT:USDT")


def test_status_includes_running_symbols_field(
    client: TestClient,
) -> None:
    """status 응답에 running_symbols 배열 노출 — UI 페어 토글 동기화용."""
    _register_user(client)
    r = client.get("/ict/status")
    assert r.status_code == 200
    body = r.json()
    assert "running_symbols" in body
    assert isinstance(body["running_symbols"], list)


def test_admin_trades_backfill_fills_missing_closes(
    client: TestClient, mu, monkeypatch,
) -> None:
    """2026-06-12 백필: 거래소 closed-pnl 에만 있는 청산을 SYNC_CLOSE 로 보충.

    멱등성: 같은 요청 반복 시 중복 기록 0.
    """
    import time as _t
    from types import SimpleNamespace

    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "admin-tok")
    code = "AICT-BKFL-BKFL-BKFL"
    _register_user(client, code)
    client.post("/ict/start?symbol=BTC/USDT:USDT")
    slot = mu._slots[(code, "BTC/USDT:USDT")]
    now_ms = int(_t.time() * 1000)
    # 주의: mu 가 data_dir 미주입이라 사용자 매매 기록이 실행 간 영속 —
    # 실행마다 고유 심볼로 이전 실행 백필과의 중복 판정을 차단 (결정론).
    uniq_sym = f"BK{now_ms % 100000}/USDT:USDT"
    cp = SimpleNamespace(
        symbol=uniq_sym, direction="short", qty=10170.0,
        entry_price=0.17, exit_price=0.173, pnl_usd=-30.43,
        closed_at_ts=now_ms - 3_600_000, opened_at_ts=now_ms - 7_200_000,
        leverage=10,
    )

    async def _fake_closed(since_ms=None, limit=200):
        return [cp]

    slot.client.fetch_closed_positions = _fake_closed  # self-spy 대체 주입

    r = client.post(
        "/admin/trades/backfill?days=7",
        headers={"X-Admin-Token": "admin-tok"},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["results"].get(code) == "+1건", repr(d)
    # 멱등 — 재실행 시 중복 0.
    r2 = client.post(
        "/admin/trades/backfill?days=7",
        headers={"X-Admin-Token": "admin-tok"},
    )
    assert r2.json()["results"][code] == "+0건"
    # 기록 확인 — 사용자 매매 기록에 SYNC_CLOSE 백필 행.
    from aurora_ict.interfaces.trades_store import TradesStore
    store = TradesStore(mu._user_data_dir(code))
    evs = [e for e in store.all_events() if e.symbol == uniq_sym]
    assert len(evs) == 1
    assert evs[0].pnl_usdt == -30.43
    assert "backfill" in evs[0].reason
    client.post("/ict/stop?symbol=BTC/USDT:USDT")


def test_close_position_uses_reduce_only_and_exchange_qty(
    client: TestClient, mu, created_clients,
) -> None:
    """2026-06-13 보안 H1: 수동 청산이 거래소 실잔량 기준 + reduce_only 로 나간다."""
    from aurora_ict.strategy.silver_bullet import Direction
    code = "AICT-RDCO-RDCO-RDCO"
    _register_user(client, code)
    client.post("/ict/start?symbol=BTC/USDT:USDT")
    bot = mu._slots[(code, "BTC/USDT:USDT")].bot
    _inject_active(bot, Direction.SHORT, 100.0, 2.0)
    fc = bot.client
    # 거래소 실잔량 = 1.5 (봇 인식 2.0과 다름) → 거래소 기준이어야.
    async def _fp(symbol):
        return {"contracts": 1.5, "side": "short", "entryPrice": 100.0}
    fc.fetch_position = _fp  # type: ignore[method-assign]

    r = client.post("/ict/position/close", json={"fraction": 1.0, "symbol": "BTC/USDT:USDT"})
    assert r.status_code == 200
    last = fc.placed_orders[-1]
    assert last["reduce_only"] is True       # 신규 반대 포지션 방지
    assert last["side"] == "buy"             # short 청산 = buy
    assert abs(last["qty"] - 1.5) < 1e-9     # 거래소 실잔량 기준
    client.post("/ict/stop?symbol=BTC/USDT:USDT")


def test_close_position_noop_when_exchange_flat(
    client: TestClient, mu,
) -> None:
    """거래소에 포지션이 이미 없으면(SL/TP 선체결) 주문 없이 상태만 정리."""
    from aurora_ict.strategy.silver_bullet import Direction
    code = "AICT-FLAT-FLAT-FLAT"
    _register_user(client, code)
    client.post("/ict/start?symbol=BTC/USDT:USDT")
    bot = mu._slots[(code, "BTC/USDT:USDT")].bot
    _inject_active(bot, Direction.SHORT, 100.0, 2.0)
    fc = bot.client
    n_before = len(fc.placed_orders)
    async def _fp(symbol):
        return {"contracts": 0, "side": "short"}  # 거래소 잔량 0
    fc.fetch_position = _fp  # type: ignore[method-assign]

    r = client.post("/ict/position/close", json={"fraction": 1.0, "symbol": "BTC/USDT:USDT"})
    assert r.status_code == 200
    assert r.json()["active"] is False
    assert len(fc.placed_orders) == n_before  # 신규 주문 0건
    assert bot.active_position is None
    client.post("/ict/stop?symbol=BTC/USDT:USDT")


def test_model_switch_normalizes_stale_name(client, mu, db_path) -> None:
    """#MODEL-SWITCH-FIX 2026-07-27: UI 하드코딩 옛 버전명("Origo 1.2")도 패밀리
    정규화로 현행 Origo 로 전환돼야 (기존엔 400 조용한 실패 → Cursus→Origo 불가)."""
    from aurora_ict.config.settings import CURSUS_MODEL_NAME, ORIGO_MODEL_NAME

    _register_user(client)
    code = "AICT-SAAS-SAAS-SAAS"
    # Cursus 로 전환 후 → 옛 이름 "Origo 1.2" 로 복귀 시도.
    r = client.post("/ict/model", json={"model": CURSUS_MODEL_NAME})
    assert r.status_code == 200
    r = client.post("/ict/model", json={"model": "Origo 1.2"})
    assert r.status_code == 200
    assert r.json()["model"] == ORIGO_MODEL_NAME       # 정규화된 현행명
    assert users_db.get_last_model(db_path, code) == ORIGO_MODEL_NAME
    # 전략 id 로도 허용 ("origo"/"cursus")
    r = client.post("/ict/model", json={"model": "cursus"})
    assert r.status_code == 200
    assert r.json()["model"] == CURSUS_MODEL_NAME
    # 진짜 모르는 이름은 여전히 400
    r = client.post("/ict/model", json={"model": "Nexus 9"})
    assert r.status_code == 400


def test_judgment_returns_cursus_payload_for_trend_bot(client, mu, db_path) -> None:
    """#2026-07-27 fix: Cursus 사용자 judgment 는 추세형 판단 응답 — 기존 hasattr
    덕타이핑이 추세형의 동명 심 필드에 오판해 ICT 판단(등급/RR/혼조)을 반환했었음."""
    from aurora_ict.config.settings import CURSUS_MODEL_NAME

    _register_user(client)
    r = client.post("/ict/model", json={"model": CURSUS_MODEL_NAME})
    assert r.status_code == 200
    # 슬롯 생성(차트 lazy 로딩 경로) 후 judgment 조회.
    client.get("/ict/ohlcv?timeframe=1h&limit=10")
    r = client.get("/ict/judgment")
    assert r.status_code == 200
    j = r.json()
    blob = str(j.get("direction", {})) + str(j.get("entry_condition", {}))
    assert "Cursus" in blob                    # 추세형 페이로드
    assert "등급" not in blob                  # ICT entry_condition 아님
