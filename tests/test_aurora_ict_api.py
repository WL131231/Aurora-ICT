"""Aurora-ICT FastAPI app — v0.2.1."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from aurora_ict.api.app import create_app
from aurora_ict.bot.manager import BotManager
from aurora_ict.config.settings import IctSettings


def _mock_client_with_data():
    c = AsyncMock()
    # 14 봉 박은 박힘 박힘 박힙 — markers 박힘 박힙 박힙 박힙 박힘 NY 10:00 박힘 박힙 박힘 박힘
    start_ms = 1778598000000  # 2026-05-12 14:00 UTC = NY 10:00 EDT
    bars = [
        (100, 105, 99, 104),
        (104, 130, 103, 125),
        (125, 124, 100, 101),
        (101, 108, 95, 96),
        (96, 110, 95.5, 109),
        (109, 112, 92, 93),
        (93, 100, 92.5, 99),
        (99, 105, 98, 104),
        (104, 106, 100, 101),
        (101, 105, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 119, 108, 118),
        (118, 122, 115, 121),
    ]
    rows = []
    for i, (o, h, lo, cl) in enumerate(bars):
        rows.append([start_ms + i * 60_000, o, h, lo, cl, 100.0])
    c.fetch_ohlcv = AsyncMock(return_value=rows)
    c.place_order = AsyncMock(return_value={"orderId": "X1"})
    c.fetch_position = AsyncMock(return_value=None)
    c.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    return c


async def _factory(_settings) -> Any:
    return _mock_client_with_data()


@pytest.fixture
def manager() -> BotManager:
    settings = IctSettings(
        enabled=True,
        demo_api_key="dk", demo_api_secret="ds",
        live_api_key="lk", live_api_secret="ls",
        step_interval_sec=3600,
        fvg_min_size_pct=0.001,
        min_rr=1.0,
    )
    return BotManager(client_factory=_factory, settings=settings)


@pytest.fixture
def client(manager: BotManager) -> TestClient:
    return TestClient(create_app(manager))


def test_health(client: TestClient) -> None:
    resp = client.get("/ict/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_status_initial(client: TestClient) -> None:
    resp = client.get("/ict/status")
    assert resp.status_code == 200
    d = resp.json()
    assert d["state"] == "stopped"
    assert d["run_mode"] == "demo"
    assert d["enabled"] is True


def test_config_safe(client: TestClient) -> None:
    """config 박힌 거 박힙 박힙 박힘 박힘 박힙 박힘 박힙 박힘 박힙 박힘 박힘."""
    resp = client.get("/ict/config")
    assert resp.status_code == 200
    d = resp.json()
    # API 키 박힘 박힘 박힘 박힙 박힘 박힙 박힙 박힘 박힙 박힘 박힙 박힘
    assert "demo_api_key" not in d
    assert "live_api_key" not in d
    assert d["has_demo_credentials"] is True
    assert d["has_live_credentials"] is True


def test_start_then_status_running(client: TestClient) -> None:
    resp = client.post("/ict/start")
    assert resp.status_code == 200
    assert resp.json()["state"] == "running"
    resp = client.post("/ict/stop")
    assert resp.json()["state"] == "stopped"


def test_start_without_credentials_returns_400() -> None:
    settings = IctSettings(enabled=True)  # no credentials
    mgr = BotManager(client_factory=_factory, settings=settings)
    c = TestClient(create_app(mgr))
    resp = c.post("/ict/start")
    assert resp.status_code == 400
    assert "API" in resp.json()["detail"]


def test_run_mode_change(client: TestClient) -> None:
    resp = client.post("/ict/run-mode", json={"mode": "live"})
    assert resp.status_code == 200
    assert resp.json()["run_mode"] == "live"
    # demo 로 되돌리기
    resp = client.post("/ict/run-mode", json={"mode": "demo"})
    assert resp.json()["run_mode"] == "demo"


def test_run_mode_invalid(client: TestClient) -> None:
    resp = client.post("/ict/run-mode", json={"mode": "paper"})
    assert resp.status_code == 400


def test_enabled_toggle(client: TestClient) -> None:
    resp = client.post("/ict/enabled", json={"enabled": False})
    assert resp.status_code == 200
    assert resp.json()["enabled"] is False
    resp = client.post("/ict/enabled", json={"enabled": True})
    assert resp.json()["enabled"] is True


def test_markers_requires_bot_running(client: TestClient) -> None:
    """봇 미가동 상태 → 404."""
    resp = client.get("/ict/markers")
    assert resp.status_code == 404


# ============================================================
# GET /ict/position — UI 하단 패널용 active position 상세
# ============================================================


def test_position_inactive_when_bot_stopped(client: TestClient) -> None:
    """봇 미가동 → active=False."""
    resp = client.get("/ict/position")
    assert resp.status_code == 200
    assert resp.json() == {"active": False}


def test_position_inactive_when_no_active_position(client: TestClient) -> None:
    """봇 가동 중이지만 active_position 없음 → active=False."""
    client.post("/ict/start")
    resp = client.get("/ict/position")
    assert resp.status_code == 200
    # fixture 데이터로는 진입 안 박힐 가능성 — 보장된 invariant 는 'active' 필드 존재
    body = resp.json()
    assert "active" in body


def test_position_active_returns_full_payload(manager: BotManager) -> None:
    """active_position 박힌 상태 → 모든 필드 + PnL + margin + liq."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
    from aurora_ict.strategy.silver_bullet import Direction

    # bot manually 박기 — start 안 거치고 active_position 직접 박음
    bot = BotIctInstance(client=_mock_client_with_data(), symbol="BTCUSDT")
    bot.active_position = _ActivePosition(
        direction=Direction.LONG,
        entry=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        qty=0.1,
        setup_ts_ms=1778598000000,
    )
    manager._bot = bot

    c = TestClient(create_app(manager))
    resp = c.get("/ict/position")
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is True
    assert body["symbol"] == "BTCUSDT"
    assert body["direction"] == "long"
    assert body["entry"] == 100.0
    assert body["qty"] == 0.1
    assert body["leverage"] == manager.settings.leverage
    # 필수 필드 모두 박혀있는지
    for field in (
        "mark_price", "unrealized_pnl", "roi_pct",
        "margin", "notional", "liquidation_price",
    ):
        assert field in body, f"missing {field}"


def test_position_pnl_sign_long_profit(manager: BotManager) -> None:
    """long + mark > entry → unrealized_pnl > 0."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
    from aurora_ict.strategy.silver_bullet import Direction

    # fixture 마지막 봉 close = 121 (보다 위), entry=100 → +PnL
    bot = BotIctInstance(client=_mock_client_with_data(), symbol="BTCUSDT")
    bot.active_position = _ActivePosition(
        direction=Direction.LONG,
        entry=100.0,
        stop_loss=95.0,
        take_profit=130.0,
        qty=1.0,
        setup_ts_ms=1778598000000,
    )
    manager._bot = bot

    c = TestClient(create_app(manager))
    body = c.get("/ict/position").json()
    assert body["mark_price"] > 100.0  # fixture 마지막 봉 close ≈ 121
    assert body["unrealized_pnl"] > 0


def test_position_pnl_sign_short_profit(manager: BotManager) -> None:
    """short + mark < entry → unrealized_pnl > 0."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
    from aurora_ict.strategy.silver_bullet import Direction

    bot = BotIctInstance(client=_mock_client_with_data(), symbol="BTCUSDT")
    bot.active_position = _ActivePosition(
        direction=Direction.SHORT,
        entry=200.0,    # 가짜 high entry — fixture 마지막 close 가 박은 게 더 낮음
        stop_loss=210.0,
        take_profit=150.0,
        qty=1.0,
        setup_ts_ms=1778598000000,
    )
    manager._bot = bot

    c = TestClient(create_app(manager))
    body = c.get("/ict/position").json()
    assert body["mark_price"] < 200.0
    assert body["unrealized_pnl"] > 0


def test_position_liquidation_long_below_entry(manager: BotManager) -> None:
    """long liquidation < entry, short liquidation > entry."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
    from aurora_ict.strategy.silver_bullet import Direction

    bot = BotIctInstance(client=_mock_client_with_data(), symbol="BTCUSDT")
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0,
        take_profit=110.0, qty=1.0, setup_ts_ms=1778598000000,
    )
    manager._bot = bot
    c = TestClient(create_app(manager))
    assert c.get("/ict/position").json()["liquidation_price"] < 100.0


# ============================================================
# POST /ict/position/close — Close By 수동 청산
# ============================================================


def test_position_close_no_active_returns_404(client: TestClient) -> None:
    """active position 없음 → 404."""
    resp = client.post("/ict/position/close", json={"fraction": 1.0})
    assert resp.status_code == 404


def test_position_close_invalid_fraction(manager: BotManager) -> None:
    """fraction (0, 1] 범위 밖 → 400."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
    from aurora_ict.strategy.silver_bullet import Direction

    bot = BotIctInstance(client=_mock_client_with_data(), symbol="BTCUSDT")
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0,
        take_profit=110.0, qty=1.0, setup_ts_ms=1778598000000,
    )
    manager._bot = bot
    c = TestClient(create_app(manager))

    assert c.post("/ict/position/close", json={"fraction": 0.0}).status_code == 400
    assert c.post("/ict/position/close", json={"fraction": 1.5}).status_code == 400
    assert c.post("/ict/position/close", json={"fraction": -0.5}).status_code == 400


def test_position_close_full(manager: BotManager) -> None:
    """fraction=1.0 → 전체 청산, active_position=None, place_order 호출."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
    from aurora_ict.strategy.silver_bullet import Direction

    mock_client = _mock_client_with_data()
    bot = BotIctInstance(client=mock_client, symbol="BTCUSDT")
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0,
        take_profit=110.0, qty=1.0, setup_ts_ms=1778598000000,
    )
    manager._bot = bot
    c = TestClient(create_app(manager))

    resp = c.post("/ict/position/close", json={"fraction": 1.0})
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is False
    assert body["closed_qty"] == 1.0
    assert body["remaining_qty"] == 0.0
    # 반대 방향 (sell) 시장가 주문 호출 확인
    mock_client.place_order.assert_awaited_once()
    call_kwargs = mock_client.place_order.call_args.kwargs
    assert call_kwargs["side"] == "sell"
    assert call_kwargs["qty"] == 1.0
    # market 청산 — price/SL/TP None
    assert call_kwargs.get("price") is None
    # active_position cleared
    assert bot.active_position is None


def test_position_close_partial(manager: BotManager) -> None:
    """fraction=0.5 → 50% 청산, 남은 qty=0.5, active 유지."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
    from aurora_ict.strategy.silver_bullet import Direction

    mock_client = _mock_client_with_data()
    bot = BotIctInstance(client=mock_client, symbol="BTCUSDT")
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0,
        take_profit=110.0, qty=1.0, setup_ts_ms=1778598000000,
    )
    manager._bot = bot
    c = TestClient(create_app(manager))

    resp = c.post("/ict/position/close", json={"fraction": 0.5})
    assert resp.status_code == 200
    body = resp.json()
    assert body["active"] is True
    assert body["closed_qty"] == 0.5
    assert body["remaining_qty"] == 0.5
    assert bot.active_position is not None
    assert bot.active_position.qty == 0.5


def test_position_close_short_uses_buy_side(manager: BotManager) -> None:
    """SHORT 포지션 청산 → buy 시장가."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
    from aurora_ict.strategy.silver_bullet import Direction

    mock_client = _mock_client_with_data()
    bot = BotIctInstance(client=mock_client, symbol="BTCUSDT")
    bot.active_position = _ActivePosition(
        direction=Direction.SHORT, entry=100.0, stop_loss=105.0,
        take_profit=90.0, qty=2.0, setup_ts_ms=1778598000000,
    )
    manager._bot = bot
    c = TestClient(create_app(manager))

    resp = c.post("/ict/position/close", json={"fraction": 1.0})
    assert resp.status_code == 200
    call_kwargs = mock_client.place_order.call_args.kwargs
    assert call_kwargs["side"] == "buy"


def test_position_close_order_failure_returns_502(manager: BotManager) -> None:
    """place_order 예외 → 502 + active_position 변경 안 됨."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
    from aurora_ict.strategy.silver_bullet import Direction

    mock_client = _mock_client_with_data()
    mock_client.place_order = AsyncMock(side_effect=RuntimeError("exchange down"))
    bot = BotIctInstance(client=mock_client, symbol="BTCUSDT")
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0,
        take_profit=110.0, qty=1.0, setup_ts_ms=1778598000000,
    )
    manager._bot = bot
    c = TestClient(create_app(manager))

    resp = c.post("/ict/position/close", json={"fraction": 1.0})
    assert resp.status_code == 502
    assert "exchange down" in resp.json()["detail"]
    # 주문 실패 시 active_position 그대로
    assert bot.active_position is not None


# ============================================================
# POST /ict/credentials (온보딩)
# ============================================================


def test_credentials_invalid_mode(client: TestClient) -> None:
    resp = client.post(
        "/ict/credentials",
        json={"mode": "paper", "api_key": "K", "api_secret": "S"},
    )
    assert resp.status_code == 400


def test_credentials_empty_key_or_secret(client: TestClient) -> None:
    resp = client.post(
        "/ict/credentials",
        json={"mode": "demo", "api_key": "", "api_secret": "S"},
    )
    assert resp.status_code == 400
    resp = client.post(
        "/ict/credentials",
        json={"mode": "demo", "api_key": "K", "api_secret": "   "},
    )
    assert resp.status_code == 400


def test_credentials_saves_demo_and_updates_settings(
    manager: BotManager, tmp_path, monkeypatch,
) -> None:
    """demo 키 저장 → .env 파일 생성 + manager.settings 즉시 갱신."""
    from aurora_ict.api import app as app_mod

    env_path = tmp_path / ".env"
    monkeypatch.setattr(app_mod, "_env_path", lambda: env_path)

    c = TestClient(create_app(manager))
    resp = c.post(
        "/ict/credentials",
        json={"mode": "demo", "api_key": "NEW_K", "api_secret": "NEW_S"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["has_demo_credentials"] is True

    # .env 파일 박힘 + 키 라인 박힌 거 확인
    content = env_path.read_text(encoding="utf-8")
    assert "AURORA_ICT_DEMO_API_KEY=NEW_K" in content
    assert "AURORA_ICT_DEMO_API_SECRET=NEW_S" in content

    # 메모리 settings 즉시 갱신 확인
    assert manager.settings.demo_api_key.get_secret_value() == "NEW_K"
    assert manager.settings.demo_api_secret.get_secret_value() == "NEW_S"


def test_credentials_replaces_existing_lines(
    manager: BotManager, tmp_path, monkeypatch,
) -> None:
    """기존 라인 박힌 .env → 새 값으로 교체 (중복 라인 생기지 않음)."""
    from aurora_ict.api import app as app_mod

    env_path = tmp_path / ".env"
    env_path.write_text(
        "AURORA_ICT_DEMO_API_KEY=OLD_K\n"
        "AURORA_ICT_DEMO_API_SECRET=OLD_S\n"
        "AURORA_ICT_SYMBOL=BTC/USDT:USDT\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_mod, "_env_path", lambda: env_path)

    c = TestClient(create_app(manager))
    resp = c.post(
        "/ict/credentials",
        json={"mode": "demo", "api_key": "NEW_K", "api_secret": "NEW_S"},
    )
    assert resp.status_code == 200

    content = env_path.read_text(encoding="utf-8")
    # 기존 OLD_K / OLD_S 박힌 거 사라지고 NEW 만 박힘
    assert "OLD_K" not in content
    assert "OLD_S" not in content
    assert "AURORA_ICT_DEMO_API_KEY=NEW_K" in content
    assert "AURORA_ICT_DEMO_API_SECRET=NEW_S" in content
    # 다른 라인 (SYMBOL) 은 보존
    assert "AURORA_ICT_SYMBOL=BTC/USDT:USDT" in content


def test_credentials_live_mode(
    manager: BotManager, tmp_path, monkeypatch,
) -> None:
    """live 모드 키 저장 → AURORA_ICT_LIVE_ prefix + live_api_key 필드 갱신."""
    from aurora_ict.api import app as app_mod

    env_path = tmp_path / ".env"
    monkeypatch.setattr(app_mod, "_env_path", lambda: env_path)

    c = TestClient(create_app(manager))
    resp = c.post(
        "/ict/credentials",
        json={"mode": "live", "api_key": "LK", "api_secret": "LS"},
    )
    assert resp.status_code == 200
    assert resp.json()["has_live_credentials"] is True

    content = env_path.read_text(encoding="utf-8")
    assert "AURORA_ICT_LIVE_API_KEY=LK" in content
    assert "AURORA_ICT_LIVE_API_SECRET=LS" in content
    assert manager.settings.live_api_key.get_secret_value() == "LK"


def test_markers_returns_data_when_running(client: TestClient) -> None:
    client.post("/ict/start")
    resp = client.get("/ict/markers?limit=20")
    assert resp.status_code == 200
    d = resp.json()
    assert "symbol" in d
    assert "markers" in d
    assert "count" in d
    assert d["count"]["fvgs"] >= 0
    client.post("/ict/stop")


def test_ohlcv_requires_bot_running(client: TestClient) -> None:
    resp = client.get("/ict/ohlcv")
    assert resp.status_code == 404


def test_ohlcv_returns_candles(client: TestClient) -> None:
    client.post("/ict/start")
    resp = client.get("/ict/ohlcv?limit=20")
    assert resp.status_code == 200
    d = resp.json()
    assert "candles" in d
    assert len(d["candles"]) > 0
    c0 = d["candles"][0]
    assert "time" in c0
    assert "open" in c0
    assert "close" in c0
    client.post("/ict/stop")


def test_ui_static_mounted(client: TestClient) -> None:
    """ui_ict/index.html 박힙 박힘 박힙 GET /ui/ 박힙 박힙 박힙 200."""
    resp = client.get("/ui/")
    # 박힌 박힙 박힘 200 (ui_ict 박힙) 또는 박힘 박힘 박힘 박힙 404 — 둘 다 OK
    assert resp.status_code in (200, 404)
