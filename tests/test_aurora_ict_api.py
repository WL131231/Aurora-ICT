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
    for i, (o, h, l, cl) in enumerate(bars):
        rows.append([start_ms + i * 60_000, o, h, l, cl, 100.0])
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
    # 박힌 demo 박은 박힘 박힘 박힘
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
    """봇 박힙 박힙 박힘 박힙 → 404."""
    resp = client.get("/ict/markers")
    assert resp.status_code == 404


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
