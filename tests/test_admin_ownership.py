"""admin 포지션 소유권 진단 — /admin/position/ownership.

2026-07-31 파트너 요청: 미추적 포지션이 봇 것인지 유저 수동인지 **확정 판정**할
수단이 없었다. 레버리지가 봇 값(20x)이라 의심됐으나, 봇이 설정한 레버리지는
계정에 남아 유저 수동 진입에도 그대로 적용되므로 근거가 되지 못한다.

판정 근거 세 갈래를 모두 노출한다:
    tag_match  거래소 주문 이력의 봇 태그(orderLinkId=AUR*) — **유일한 확정 증거**
    entry_rec  봇 매매기록의 미청산 ENTRY
    has_sl     봇은 진입 시 SL 등록에 실패하면 비상청산하므로, SL 부재는 수동 신호

읽기 전용 — 포지션을 건드리지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from aurora_ict.api.app import create_app
from aurora_ict.bot.multi_user_manager import MultiUserBotManager
from aurora_ict.config.settings import IctSettings

ADMIN_TOKEN = "test-admin-token"
CODE, SYM = "AICT-TEST-CODE-XXXX", "BTC/USDT:USDT"


@dataclass
class _Slot:
    symbol: str
    settings: Any
    bot: Any
    client: Any


def _exchange(*, tag: bool, sl: float, contracts: float = 0.003) -> AsyncMock:
    """TDAF 실측 케이스 기본값 — BTC short 0.003 @ 63700.2, SL 없음."""
    ex = AsyncMock()
    ex.fetch_position = AsyncMock(return_value={
        "contracts": contracts, "side": "short", "entryPrice": 63700.2,
        "info": {"stopLoss": str(sl)},
    })
    ex.position_opened_by_bot = AsyncMock(return_value=tag)
    return ex


@pytest.fixture
def make_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", ADMIN_TOKEN)

    def _build(ex: AsyncMock, bot: Any = None) -> TestClient:
        db = tmp_path / "users.db"
        mu = MultiUserBotManager(
            client_factory=lambda *a, **k: ex, db_path=db,
            base_settings=IctSettings(enabled=True, step_interval_sec=3600),
            master_key=Fernet.generate_key(),
        )
        mu._slots[(CODE, SYM)] = _Slot(SYM, None, bot, ex)
        app = create_app(multi_user=True, multi_user_manager=mu,
                         auth_db_path=db, secure_cookie=False,
                         master_key=mu.master_key)
        return TestClient(app)

    return _build


def _get(c: TestClient, token: str | None = ADMIN_TOKEN):
    headers = {"X-Admin-Token": token} if token else {}
    return c.get("/admin/position/ownership",
                 params={"code": CODE, "symbol": SYM}, headers=headers)


def test_manual_position_verdict(make_client: Any) -> None:
    """★ TDAF 실측 재현 — 태그·기록 없고 SL 도 없으면 manual_likely."""
    ex = _exchange(tag=False, sl=0.0)
    r = _get(make_client(ex))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["tag_match"] is False
    assert d["has_sl"] is False
    assert d["entry_record"] is None
    assert d["verdict"] == "manual_likely"


def test_bot_position_by_tag(make_client: Any) -> None:
    """봇 태그가 잡히면 verdict=bot — SL 유무와 무관한 확정 증거."""
    ex = _exchange(tag=True, sl=0.0)
    r = _get(make_client(ex))
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["tag_match"] is True
    assert d["verdict"] == "bot"


def test_sl_present_but_no_evidence_is_unknown(make_client: Any) -> None:
    """SL 은 있는데 태그·기록이 없으면 단정하지 않는다(unknown)."""
    ex = _exchange(tag=False, sl=65000.0)
    r = _get(make_client(ex))
    assert r.status_code == 200, r.text
    assert r.json()["verdict"] == "unknown"


def test_read_only_no_orders(make_client: Any) -> None:
    """진단은 읽기 전용 — 주문이 나가면 안 된다."""
    ex = _exchange(tag=False, sl=0.0)
    _get(make_client(ex))
    assert ex.place_order.await_count == 0


def test_requires_admin_token(make_client: Any) -> None:
    """토큰 없으면 거부 — 읽기 전용이어도 사용자 계정 정보다."""
    ex = _exchange(tag=False, sl=0.0)
    r = _get(make_client(ex), token=None)
    assert r.status_code in (401, 403)


def test_no_active_position_404(make_client: Any) -> None:
    """활성 포지션이 없으면 404."""
    ex = _exchange(tag=False, sl=0.0, contracts=0.0)
    r = _get(make_client(ex))
    assert r.status_code == 404
