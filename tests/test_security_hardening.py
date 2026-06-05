"""보안 강화 PR 검증 — path traversal 차단 / docs 비활성 / 보안 헤더 / rate limit.

대상:
    - auth.router._CODE_PATTERN — license code 입력단 형식 검증(path traversal 1차)
    - MultiUserBotManager._user_data_dir — 디렉토리 생성단 경로 봉쇄(2차)
    - create_app(multi_user=True) — /docs 비활성, 보안 헤더, rate limit 미들웨어

mock 0 정책: TestClient + 더미 client_factory(봇 미가동이라 호출 안 됨)만 사용.

담당: 지영민 (보안 강화 PR)
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient
from pydantic import ValidationError

from aurora_ict.api.app import create_app
from aurora_ict.auth import pin, users_db
from aurora_ict.auth.router import LoginRequest, SetupPinRequest
from aurora_ict.bot.multi_user_manager import MultiUserBotManager
from aurora_ict.config.settings import IctSettings

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


@pytest.fixture(autouse=True)
def _reset_sessions() -> Iterator[None]:
    pin.revoke_all_sessions()
    yield
    pin.revoke_all_sessions()


@pytest.fixture
def mu(db_path, master_key) -> MultiUserBotManager:
    async def factory(_settings: IctSettings) -> Any:
        # 봇을 start 하지 않는 테스트라 factory 는 호출되지 않음.
        raise AssertionError("client_factory 가 호출되면 안 됨")

    return MultiUserBotManager(
        client_factory=factory,
        db_path=db_path,
        base_settings=IctSettings(step_interval_sec=3600),
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


# ============================================================
# 1. Path traversal — code 형식 검증 (입력단 1차 방어)
# ============================================================


_BAD_CODES = ["../../etc/passwd", "../evil", "a/b", "a\\b", "..", "x y", "코드한글"]


@pytest.mark.parametrize("bad", _BAD_CODES)
def test_setup_pin_model_rejects_bad_code(bad: str) -> None:
    """SetupPinRequest.code 가 영숫자/_/- 외 문자를 거부(ValidationError)."""
    with pytest.raises(ValidationError):
        SetupPinRequest(code=bad, pin="Aa1!aaaa", pin_confirm="Aa1!aaaa")


@pytest.mark.parametrize("bad", _BAD_CODES)
def test_login_model_rejects_bad_code(bad: str) -> None:
    """LoginRequest.code 도 동일 규칙으로 거부."""
    with pytest.raises(ValidationError):
        LoginRequest(code=bad, pin="Aa1!aaaa")


def test_valid_license_code_accepted() -> None:
    """정상 라이선스 코드(영숫자+하이픈)는 통과."""
    req = SetupPinRequest(
        code="AICT-0Q8B-D1YU-VFRN", pin="Aa1!aaaa", pin_confirm="Aa1!aaaa",
    )
    assert req.code == "AICT-0Q8B-D1YU-VFRN"


def test_setup_pin_endpoint_rejects_traversal(client: TestClient) -> None:
    """API 경유 — traversal code 는 422(검증 실패)로 차단, 계정 생성 안 됨."""
    r = client.post(
        "/auth/setup-pin",
        json={"code": "../../tmp/evil", "pin": "Aa1!aaaa", "pin_confirm": "Aa1!aaaa"},
    )
    assert r.status_code == 422


# ============================================================
# 2. Path traversal — _user_data_dir 디렉토리 봉쇄 (2차 방어)
# ============================================================


@pytest.mark.parametrize("bad", _BAD_CODES + ["../../../../tmp/x"])
def test_user_data_dir_rejects_traversal(mu: MultiUserBotManager, bad: str) -> None:
    """_user_data_dir 가 부적합 user_code 를 ValueError 로 차단(mkdir 전)."""
    with pytest.raises(ValueError):
        mu._user_data_dir(bad)


def test_user_data_dir_accepts_valid(mu: MultiUserBotManager) -> None:
    """정상 코드는 users/ 하위 경로를 반환."""
    path = mu._user_data_dir("AICT-GOOD-GOOD-GOOD")
    assert path.name == "AICT-GOOD-GOOD-GOOD"
    assert path.parent.name == "users"


# ============================================================
# 3. /docs·/openapi.json 비활성 (운영 정보노출 차단)
# ============================================================


def test_docs_disabled_in_saas(client: TestClient) -> None:
    assert client.get("/docs").status_code == 404
    assert client.get("/redoc").status_code == 404
    assert client.get("/openapi.json").status_code == 404


# ============================================================
# 4. 보안 헤더 주입
# ============================================================


def test_security_headers_present(client: TestClient) -> None:
    """모든 응답(인증 불필요한 health 포함)에 보안 헤더가 붙는다."""
    r = client.get("/ict/health")
    assert r.status_code == 200
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["Referrer-Policy"] == "strict-origin-when-cross-origin"
    assert "max-age=" in r.headers["Strict-Transport-Security"]


# ============================================================
# 5. Rate limiting — 인증 엔드포인트 분당 시도 제한
# ============================================================


def test_setup_pin_rate_limited(client: TestClient) -> None:
    """동일 IP 의 /auth/setup-pin 연속 호출이 6번째에 429 로 차단된다.

    1번째: 200(생성), 2~5번째: 400(이미 PIN), 6번째: 429(rate limit).
    sleep 없는 경로라 빠르게 검증 가능.
    """
    payload = {
        "code": "AICT-RATE-RATE-RATE",
        "pin": "Aa1!aaaa",
        "pin_confirm": "Aa1!aaaa",
    }
    statuses = [client.post("/auth/setup-pin", json=payload).status_code
                for _ in range(6)]
    # 처음 5번은 통과(라우트 도달), 6번째는 미들웨어가 429 로 차단.
    assert statuses[0] == 200
    assert all(s != 429 for s in statuses[:5])
    assert statuses[5] == 429


def test_rate_limit_isolated_per_path(client: TestClient) -> None:
    """한 경로의 한도 소진이 다른 경로(health)를 막지 않는다."""
    payload = {
        "code": "AICT-PATH-PATH-PATH",
        "pin": "Aa1!aaaa",
        "pin_confirm": "Aa1!aaaa",
    }
    for _ in range(6):
        client.post("/auth/setup-pin", json=payload)
    # setup-pin 은 한도 소진됐어도 health 는 정상.
    assert client.get("/ict/health").status_code == 200
