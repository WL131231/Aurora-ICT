"""2026-07-22: 루트(/) → UI(/ui/) 리다이렉트 테스트.

UI 는 /ui 마운트라 도메인 루트(/)엔 라우트가 없어 {"detail":"Not Found"} 가
떴음(파트너가 모바일서 도메인만 쳐서 빈 화면). 루트 접속 시 대시보드로 보냄.
mock 0 — BotManager 를 결정론적 합성 client 로 구성(기존 test_aurora_ict_api 패턴).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from aurora_ict.api.app import create_app
from aurora_ict.bot.manager import BotManager
from aurora_ict.config.settings import IctSettings


async def _factory(_settings) -> Any:
    c = AsyncMock()
    c.fetch_ohlcv = AsyncMock(return_value=[])
    c.fetch_position = AsyncMock(return_value=None)
    c.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    return c


def _client() -> TestClient:
    settings = IctSettings(
        enabled=True,
        demo_api_key="dk", demo_api_secret="ds",
        live_api_key="lk", live_api_secret="ls",
        step_interval_sec=3600,
    )
    return TestClient(create_app(BotManager(client_factory=_factory, settings=settings)))


def test_root_redirects_to_ui() -> None:
    """GET / → 307 redirect /ui/ (도메인만 쳐도 대시보드로)."""
    r = _client().get("/", follow_redirects=False)
    assert r.status_code in (307, 308)
    assert r.headers["location"] == "/ui/"


def test_root_redirect_lands_on_ui() -> None:
    """리다이렉트 따라가면 최종 UI(/ui/) 200 (index.html 서빙)."""
    r = _client().get("/", follow_redirects=True)
    assert r.status_code == 200
    assert str(r.url).rstrip("/").endswith("/ui")
