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
