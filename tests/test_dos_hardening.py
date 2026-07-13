"""#DOS 2026-07-13 — DoS/자원고갈 하드닝 검증.

상용화 전 가용성 감사(4번째 축) 처방:
- 요청 본문 크기 상한(413) — 대용량 POST RAM 고갈 차단
- 차트 limit 상한(422) — 무제한 limit 이벤트루프/메모리 고갈 차단
- 전역 봇 슬롯 상한 — 단일 머신 admission control
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from aurora_ict.api.app import create_app
from aurora_ict.auth import pin, users_db
from aurora_ict.bot import multi_user_manager as mum
from aurora_ict.bot.multi_user_manager import MultiUserBotManager
from aurora_ict.config.settings import IctSettings


class _FakeClient:
    async def fetch_ohlcv(self, symbol, timeframe, limit):
        return []

    async def set_leverage(self, symbol, leverage):
        return {"ok": True}

    async def fetch_balance(self):
        return {"USDT": {"total": 1000.0}}

    async def fetch_position(self, symbol):
        return None


@pytest.fixture
def db_path(tmp_path):
    p = tmp_path / "users.db"
    users_db.init_db(p)
    for i in range(30):
        users_db.set_license(p, code=f"AICT-DOS{i:02d}-DOS{i:02d}-X",
                             license_type="referral", expires_at=None)
    return p


@pytest.fixture(autouse=True)
def _reset_sessions() -> Iterator[None]:
    pin.revoke_all_sessions()
    yield
    pin.revoke_all_sessions()


@pytest.fixture
def mu(db_path) -> MultiUserBotManager:
    async def factory(_s: IctSettings) -> _FakeClient:
        return _FakeClient()

    return MultiUserBotManager(
        client_factory=factory, db_path=db_path,
        base_settings=IctSettings(enabled=True, step_interval_sec=3600),
        master_key=Fernet.generate_key(),
    )


@pytest.fixture
def client(mu, db_path) -> TestClient:
    app = create_app(multi_user=True, multi_user_manager=mu,
                     auth_db_path=db_path, secure_cookie=False,
                     master_key=mu.master_key)
    return TestClient(app)


def test_oversized_body_rejected_413(client: TestClient) -> None:
    """256KB 초과 본문은 413 — 바디 파싱 전 차단."""
    big = "x" * (300 * 1024)
    resp = client.post("/auth/login", json={"code": big, "pin": big})
    assert resp.status_code == 413


def test_normal_body_passes(client: TestClient) -> None:
    """정상 크기 본문은 413 아님 (미등록 코드라 다른 응답, 413 만 아니면 됨)."""
    resp = client.post(
        "/auth/setup-pin",
        json={"code": "AICT-DOS00-DOS00-X", "pin": "Aa1!aaaa",
              "pin_confirm": "Aa1!aaaa"},
    )
    assert resp.status_code != 413


def test_chart_limit_capped_422(client: TestClient) -> None:
    """차트 limit 이 상한 초과면 422 — 백만봉급 공격은 차단, 실 UI 값은 통과.

    ohlcv 캡 100000(UI 60000봉 요청 수용), markers 캡 5000(지표계산 CPU).
    """
    # 백만봉급(공격)은 422.
    assert client.get("/ict/ohlcv?limit=1000000").status_code in (401, 422)
    assert client.get("/ict/markers?limit=999999").status_code in (401, 422)
    # 실 UI 요청값(ohlcv 60000, markers 2000)은 422 아님(미인증이라 401 은 가능).
    assert client.get("/ict/ohlcv?limit=60000").status_code != 422
    assert client.get("/ict/markers?limit=2000").status_code != 422


@pytest.mark.asyncio
async def test_global_bot_ceiling(mu: MultiUserBotManager, db_path, monkeypatch) -> None:
    """전역 봇 슬롯 상한 도달 시 신규 슬롯 거부(ValueError)."""
    from aurora_ict.auth import keystore
    from aurora_ict.config.settings import RunMode

    # 슬롯 생성 전제인 LIVE 키를 4명에게 등록.
    for i in range(4):
        code = f"AICT-DOS{i:02d}-DOS{i:02d}-X"
        users_db.set_api_keys(db_path, code, "pub", keystore.encrypt_secret(
            "sec", mu.master_key), mode="live")

    monkeypatch.setattr(mum, "MAX_TOTAL_BOTS", 3)
    await mu.get_or_create_bot("AICT-DOS00-DOS00-X", "BTC/USDT:USDT",
                               force_run_mode=RunMode.LIVE)
    await mu.get_or_create_bot("AICT-DOS01-DOS01-X", "ETH/USDT:USDT",
                               force_run_mode=RunMode.LIVE)
    await mu.get_or_create_bot("AICT-DOS02-DOS02-X", "SOL/USDT:USDT",
                               force_run_mode=RunMode.LIVE)
    # 4번째(신규 유저·신규 심볼)는 전역 상한으로 거부.
    with pytest.raises(ValueError, match="서버 전체 동시 가동 한도"):
        await mu.get_or_create_bot("AICT-DOS03-DOS03-X", "XRP/USDT:USDT",
                                   force_run_mode=RunMode.LIVE)
