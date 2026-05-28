"""auth/router.py + middleware.py — FastAPI TestClient 통합 테스트 (mock 0).

검증 시나리오:
    1. setup-pin → 200 + 세션 cookie
    2. status (cookie 포함) → authenticated=True
    3. api-keys → 200, DB 에 암호화된 secret 저장 확인
    4. login (다른 client) → 200 + 다른 토큰
    5. logout → 200 + status=False
    6. PIN 강도 미달 → 400
    7. 잘못된 코드/PIN → 401
    8. 2명 사용자 독립

mock 0: 진짜 SQLite + Fernet 사용 (tmp_path 격리).
세션은 모듈 전역 dict → 각 test 시작 시 reset 필요.

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aurora_ict.auth import keystore, pin, users_db
from aurora_ict.auth.middleware import SESSION_COOKIE_NAME
from aurora_ict.auth.router import create_auth_router


@pytest.fixture
def master_key() -> bytes:
    """테스트 격리용 Fernet 키 — 운영 keystore 와 무관."""
    return Fernet.generate_key()


@pytest.fixture
def db_path(tmp_path):
    """격리된 users.db — init_db 까지 완료된 상태."""
    path = tmp_path / "users.db"
    users_db.init_db(path)
    return path


@pytest.fixture(autouse=True)
def _reset_sessions() -> Iterator[None]:
    """각 test 전후 세션 dict reset — 격리 보장."""
    pin.revoke_all_sessions()
    yield
    pin.revoke_all_sessions()


@pytest.fixture
def app(db_path, master_key) -> FastAPI:
    """auth router 만 포함한 최소 FastAPI app."""
    app = FastAPI()
    app.include_router(
        create_auth_router(db_path, secure_cookie=False, master_key=master_key),
    )
    return app


@pytest.fixture
def client(app) -> TestClient:
    return TestClient(app)


# ============================================================
# 1. setup-pin
# ============================================================


def test_setup_pin_creates_user_and_session(client: TestClient, db_path) -> None:
    """setup-pin — 새 코드면 자동 생성 + PIN 해시 저장 + 세션 cookie 발급."""
    resp = client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-NEW1-NEW1-NEW1",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["code"] == "AICT-NEW1-NEW1-NEW1"
    assert body["license_type"] == "referral"

    # 세션 cookie 박힘 확인 (TestClient 가 자동 보관).
    assert SESSION_COOKIE_NAME in client.cookies

    # DB 상태 확인 — pin_hash 있고, 평문 X.
    user = users_db.get_user_by_code(db_path, "AICT-NEW1-NEW1-NEW1")
    assert user is not None
    assert user["pin_hash"] is not None
    assert user["pin_hash"].startswith("pbkdf2_sha256$")
    assert "Aa1!aaaa" not in user["pin_hash"]  # 평문 절대 저장 X


def test_setup_pin_mismatch_confirm_400(client: TestClient) -> None:
    """setup-pin — pin ≠ pin_confirm → 400."""
    resp = client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-MISM-MISM-MISM",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!bbbb",
        },
    )
    assert resp.status_code == 400


def test_setup_pin_weak_pin_400(client: TestClient) -> None:
    """setup-pin — PIN 강도 미달 (숫자/특수 부재) → 400."""
    resp = client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-WEAK-WEAK-WEAK",
            "pin": "abcdefgh",  # 영문만, 숫자/특수 X
            "pin_confirm": "abcdefgh",
        },
    )
    assert resp.status_code == 400
    detail = resp.json()["detail"]
    # detail 은 {"code": ..., "message": ...} dict
    assert isinstance(detail, dict)
    assert detail["code"] in ("no_digit", "no_special")


def test_setup_pin_existing_pin_rejects(client: TestClient) -> None:
    """setup-pin — 이미 PIN 있는 코드는 재설정 거부 (별도 flow)."""
    client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-DUPE-DUPE-DUPE",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    # 같은 코드 재설정 시도.
    resp = client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-DUPE-DUPE-DUPE",
            "pin": "Bb2!bbbb",
            "pin_confirm": "Bb2!bbbb",
        },
    )
    assert resp.status_code == 400
    assert "이미 PIN" in resp.json()["detail"]


# ============================================================
# 2. status
# ============================================================


def test_status_unauth_needs_pin_setup_when_empty(client: TestClient) -> None:
    """status — DB 가 비었으면 needs_pin_setup=True."""
    resp = client.get("/auth/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["authenticated"] is False
    assert body["needs_pin_setup"] is True


def test_status_unauth_no_setup_needed_after_someone_registered(
    client: TestClient,
) -> None:
    """status — 누군가 등록되어 있으면 needs_pin_setup=False (cookie 없는 새 client)."""
    client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-OTHR-OTHR-OTHR",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    # cookie 비워서 미인증 상태로 status 확인.
    client.cookies.clear()
    resp = client.get("/auth/status")
    body = resp.json()
    assert body["authenticated"] is False
    assert body["needs_pin_setup"] is False


def test_status_authed_after_setup(client: TestClient) -> None:
    """status — setup-pin 직후 같은 client (cookie 보관) 면 authenticated=True."""
    client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-AUTH-AUTH-AUTH",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    resp = client.get("/auth/status")
    body = resp.json()
    assert body["authenticated"] is True
    assert body["code"] == "AICT-AUTH-AUTH-AUTH"
    assert body["license_type"] == "referral"
    assert body["has_api_keys"] is False  # 아직 api-keys 안 박음


def test_status_includes_license_fields_for_referral(client: TestClient) -> None:
    """status — referral 사용자: expires_at=None (무기한), created_at 채워짐.

    2026-05-28 — 우측 라이선스 카드용 응답 확장 검증.
    """
    client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-REFR-REFR-REFR",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    body = client.get("/auth/status").json()
    assert body["authenticated"] is True
    # 응답 키 존재.
    assert "expires_at" in body
    assert "created_at" in body
    # referral 은 만료 없음 → None.
    assert body["expires_at"] is None
    # 가입 시각은 항상 NOT NULL. ISO 8601 UTC (Z 접미사).
    assert isinstance(body["created_at"], str)
    assert body["created_at"].endswith("Z")


def test_status_includes_license_fields_for_subscription(
    client: TestClient, db_path,
) -> None:
    """status — sub_30d 등 구독 사용자: expires_at 값 그대로 전달.

    DB 에 사전 등록 (라이선스 발급 시 별도 도구가 하듯) → setup-pin 으로 PIN 채움
    → status 응답에서 expires_at 노출 검증.
    """
    # 구독 라이선스 사전 등록 (운영 흐름 모사) — expires_at 명시.
    expiry = "2027-05-28T00:00:00Z"
    users_db.create_user(
        db_path,
        code="AICT-SUB1-SUB1-SUB1",
        license_type="sub_365d",
        expires_at=expiry,
    )
    # PIN 채움 (자동 로그인).
    client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-SUB1-SUB1-SUB1",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    body = client.get("/auth/status").json()
    assert body["authenticated"] is True
    assert body["license_type"] == "sub_365d"
    assert body["expires_at"] == expiry
    assert isinstance(body["created_at"], str)
    assert body["created_at"].endswith("Z")


# ============================================================
# 3. api-keys
# ============================================================


def test_api_keys_requires_auth(client: TestClient) -> None:
    """api-keys — 인증 없이 호출 시 401."""
    resp = client.post(
        "/auth/api-keys",
        json={"api_key": "pub_x", "api_secret": "sec_x"},
    )
    assert resp.status_code == 401


def test_api_keys_stores_encrypted_secret(
    client: TestClient, db_path, master_key,
) -> None:
    """api-keys — 인증된 사용자 secret 암호화 후 DB 저장."""
    client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-KEYS-KEYS-KEYS",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    resp = client.post(
        "/auth/api-keys",
        json={"api_key": "pub_abc123", "api_secret": "very_secret_value"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # DB 상태 — secret 평문 X, 암호문 저장.
    user = users_db.get_user_by_code(db_path, "AICT-KEYS-KEYS-KEYS")
    assert user is not None
    assert user["api_key"] == "pub_abc123"
    assert user["api_secret_enc"] is not None
    assert "very_secret_value" not in user["api_secret_enc"]  # 평문 X
    # 복호화 시 원래 평문 복원.
    plaintext = keystore.decrypt_secret(user["api_secret_enc"], key=master_key)
    assert plaintext == "very_secret_value"

    # status 가 has_api_keys=True 로 바뀜.
    status_resp = client.get("/auth/status")
    assert status_resp.json()["has_api_keys"] is True


# ============================================================
# 4. login
# ============================================================


def test_login_success_with_separate_client(app, db_path) -> None:
    """login — 다른 client (cookie 격리) 로 호출 시 새 토큰 발급."""
    # 사전 등록 (별도 client).
    c1 = TestClient(app)
    c1.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-LOGN-LOGN-LOGN",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    token_a = c1.cookies.get(SESSION_COOKIE_NAME)
    assert token_a is not None

    # 새 client — cookie 없는 상태에서 login.
    c2 = TestClient(app)
    resp = c2.post(
        "/auth/login",
        json={"code": "AICT-LOGN-LOGN-LOGN", "pin": "Aa1!aaaa"},
    )
    assert resp.status_code == 200
    token_b = c2.cookies.get(SESSION_COOKIE_NAME)
    assert token_b is not None
    # 같은 사용자라도 세션 토큰은 다름.
    assert token_b != token_a


def test_login_wrong_pin_401(client: TestClient) -> None:
    """login — PIN 틀리면 401."""
    client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-WRNG-WRNG-WRNG",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    client.cookies.clear()
    resp = client.post(
        "/auth/login",
        json={"code": "AICT-WRNG-WRNG-WRNG", "pin": "Wrong!9X"},
    )
    assert resp.status_code == 401


def test_login_unknown_code_401(client: TestClient) -> None:
    """login — 미등록 코드도 401 (코드 enum 정보 노출 안 함)."""
    resp = client.post(
        "/auth/login",
        json={"code": "AICT-NONE-NONE-NONE", "pin": "Aa1!aaaa"},
    )
    assert resp.status_code == 401


# ============================================================
# 5. logout
# ============================================================


def test_logout_revokes_session(client: TestClient) -> None:
    """logout — 세션 무효화 후 status=False."""
    client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-LOUT-LOUT-LOUT",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    assert client.get("/auth/status").json()["authenticated"] is True

    resp = client.post("/auth/logout")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # logout 이후 status — TestClient 의 cookie 가 비워졌어야 함.
    # delete_cookie 가 max-age=0 헤더로 박혀서 client.cookies 가 비워짐.
    status_body = client.get("/auth/status").json()
    assert status_body["authenticated"] is False


def test_logout_idempotent_without_session(client: TestClient) -> None:
    """logout — 세션 없어도 200 (멱등)."""
    resp = client.post("/auth/logout")
    assert resp.status_code == 200


# ============================================================
# 6. 2명 사용자 독립
# ============================================================


def test_two_users_independent_sessions(app, db_path) -> None:
    """다중 사용자 — 각 client 가 자기 코드만 보고, 서로 격리."""
    c1 = TestClient(app)
    c2 = TestClient(app)

    c1.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-USR1-USR1-USR1",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    c2.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-USR2-USR2-USR2",
            "pin": "Bb2!bbbb",
            "pin_confirm": "Bb2!bbbb",
        },
    )

    s1 = c1.get("/auth/status").json()
    s2 = c2.get("/auth/status").json()
    assert s1["code"] == "AICT-USR1-USR1-USR1"
    assert s2["code"] == "AICT-USR2-USR2-USR2"
    assert s1["code"] != s2["code"]

    # 각자 api-keys 독립.
    c1.post(
        "/auth/api-keys",
        json={"api_key": "pub_user1", "api_secret": "secret_user1"},
    )
    c2.post(
        "/auth/api-keys",
        json={"api_key": "pub_user2", "api_secret": "secret_user2"},
    )

    u1 = users_db.get_user_by_code(db_path, "AICT-USR1-USR1-USR1")
    u2 = users_db.get_user_by_code(db_path, "AICT-USR2-USR2-USR2")
    assert u1 is not None and u2 is not None
    assert u1["api_key"] == "pub_user1"
    assert u2["api_key"] == "pub_user2"
    assert u1["api_secret_enc"] != u2["api_secret_enc"]


def test_session_cookie_is_httponly(client: TestClient) -> None:
    """세션 cookie — HttpOnly 속성 확인 (JS 접근 차단)."""
    resp = client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-HTTP-ONLY-XXXX",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    set_cookie = resp.headers.get("set-cookie", "")
    assert SESSION_COOKIE_NAME in set_cookie
    assert "HttpOnly" in set_cookie
    # SameSite=Lax
    assert "SameSite=Lax".lower() in set_cookie.lower()


# ============================================================
# 7. /auth/status app_version 필드 (2026-05-28 UI 자동 갱신)
# ============================================================


def test_status_unauth_includes_app_version(client: TestClient) -> None:
    """미인증 응답에도 app_version 포함 — 로그인 화면에서도 drift 감지 필요."""
    from aurora_ict import __version__

    resp = client.get("/auth/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["authenticated"] is False
    assert "app_version" in body
    assert body["app_version"] == __version__


def test_status_authed_includes_app_version(client: TestClient) -> None:
    """인증 응답에도 app_version 포함 — polling 으로 재배포 후 자동 reload 트리거."""
    from aurora_ict import __version__

    client.post(
        "/auth/setup-pin",
        json={
            "code": "AICT-VRSN-VRSN-VRSN",
            "pin": "Aa1!aaaa",
            "pin_confirm": "Aa1!aaaa",
        },
    )
    resp = client.get("/auth/status")
    body = resp.json()
    assert body["authenticated"] is True
    assert "app_version" in body
    assert body["app_version"] == __version__
    # 빈 문자열 / None 이면 UI drift 비교 불가 — 비어있지 않은 string 임을 확인.
    assert isinstance(body["app_version"], str)
    assert body["app_version"]
