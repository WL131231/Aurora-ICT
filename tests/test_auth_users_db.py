"""users_db 모듈 — 실제 SQLite (in tmp_path) 사용, mock 0.

검증 대상 함수:
    - init_db (idempotent)
    - create_user (정상 + 중복 코드 IntegrityError)
    - get_user_by_code (존재 / 미존재)
    - set_pin (정상 + 평문 거부 + 미존재 code)
    - set_api_keys (정상 + 빈 값 거부 + 미존재 code)
    - update_last_login (정상 + 미존재 code)

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

import sqlite3

import pytest

from aurora_ict.auth import users_db
from aurora_ict.auth.pin import hash_pin


@pytest.fixture
def db_path(tmp_path):
    """격리된 .db 경로 — init_db 까지 완료된 상태로 제공."""
    path = tmp_path / "users.db"
    users_db.init_db(path)
    return path


def test_init_db_creates_users_table(tmp_path):
    """init_db — 신규 경로에서 users 테이블 + 인덱스 생성."""
    path = tmp_path / "fresh.db"
    users_db.init_db(path)

    assert path.exists()
    with sqlite3.connect(str(path)) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users'",
        ).fetchall()
        assert len(rows) == 1


def test_init_db_idempotent(db_path):
    """init_db — 같은 경로 재실행해도 예외 없음."""
    # db_path fixture 는 이미 init 됐고, 한 번 더 호출.
    users_db.init_db(db_path)
    users_db.init_db(db_path)  # 3회째도 OK
    # 데이터 손실 없는지 확인 — 사용자 한 명 만들고 재 init 후에도 살아있어야 함.
    users_db.create_user(db_path, "AICT-AAAA-AAAA-AAAA")
    users_db.init_db(db_path)
    assert users_db.get_user_by_code(db_path, "AICT-AAAA-AAAA-AAAA") is not None


def test_create_user_returns_row_id_and_persists(db_path):
    """create_user — id 반환 + 조회 시 모든 필드 일치."""
    row_id = users_db.create_user(
        db_path,
        code="AICT-1111-2222-3333",
        license_type="sub_30d",
        expires_at="2026-06-30T23:59:59Z",
    )
    assert isinstance(row_id, int)
    assert row_id >= 1

    user = users_db.get_user_by_code(db_path, "AICT-1111-2222-3333")
    assert user is not None
    assert user["code"] == "AICT-1111-2222-3333"
    assert user["license_type"] == "sub_30d"
    assert user["expires_at"] == "2026-06-30T23:59:59Z"
    assert user["pin_hash"] is None
    assert user["api_key"] is None
    assert user["api_secret_enc"] is None
    assert user["last_login_at"] is None
    assert user["created_at"] is not None
    assert user["updated_at"] is not None


def test_create_user_defaults_to_referral(db_path):
    """create_user — license_type 미지정 시 referral, expires_at None 가능."""
    users_db.create_user(db_path, "AICT-REFR-0001-0001")
    user = users_db.get_user_by_code(db_path, "AICT-REFR-0001-0001")
    assert user is not None
    assert user["license_type"] == "referral"
    assert user["expires_at"] is None


def test_create_user_duplicate_code_raises(db_path):
    """create_user — UNIQUE 제약 위반 시 IntegrityError."""
    users_db.create_user(db_path, "AICT-DUPE-DUPE-DUPE")
    with pytest.raises(sqlite3.IntegrityError):
        users_db.create_user(db_path, "AICT-DUPE-DUPE-DUPE")


def test_get_user_by_code_missing(db_path):
    """get_user_by_code — 미존재 코드면 None."""
    assert users_db.get_user_by_code(db_path, "AICT-NOPE-NOPE-NOPE") is None


def test_set_pin_success(db_path):
    """set_pin — 정상 해시 저장 + updated_at 갱신."""
    users_db.create_user(db_path, "AICT-PIN1-PIN1-PIN1")
    before = users_db.get_user_by_code(db_path, "AICT-PIN1-PIN1-PIN1")
    assert before is not None
    pin_h = hash_pin("Aa1!aaaa")
    ok = users_db.set_pin(db_path, "AICT-PIN1-PIN1-PIN1", pin_h)
    assert ok is True

    after = users_db.get_user_by_code(db_path, "AICT-PIN1-PIN1-PIN1")
    assert after is not None
    assert after["pin_hash"] == pin_h
    # updated_at 은 이전과 같거나 더 큰 ISO 문자열 (초 단위 정밀도라 같을 수 있음).
    assert after["updated_at"] >= before["updated_at"]


def test_set_pin_rejects_plaintext(db_path):
    """set_pin — algo prefix 없는 (평문 의심) 값은 ValueError."""
    users_db.create_user(db_path, "AICT-PIN2-PIN2-PIN2")
    with pytest.raises(ValueError, match="해시 형식"):
        users_db.set_pin(db_path, "AICT-PIN2-PIN2-PIN2", "myPin12!")


def test_set_pin_missing_code_returns_false(db_path):
    """set_pin — 미존재 code 면 False 반환 (예외 X)."""
    ok = users_db.set_pin(
        db_path, "AICT-NONE-NONE-NONE", hash_pin("Aa1!aaaa"),
    )
    assert ok is False


def test_set_api_keys_success(db_path):
    """set_api_keys — 키/암호문 저장 성공."""
    users_db.create_user(db_path, "AICT-KEY1-KEY1-KEY1")
    ok = users_db.set_api_keys(
        db_path,
        code="AICT-KEY1-KEY1-KEY1",
        api_key="pub_abc123",
        api_secret_enc="gAAAA_pretend_fernet_token",
    )
    assert ok is True

    user = users_db.get_user_by_code(db_path, "AICT-KEY1-KEY1-KEY1")
    assert user is not None
    assert user["api_key"] == "pub_abc123"
    assert user["api_secret_enc"] == "gAAAA_pretend_fernet_token"


def test_set_api_keys_rejects_empty(db_path):
    """set_api_keys — 빈 문자열은 ValueError."""
    users_db.create_user(db_path, "AICT-KEY2-KEY2-KEY2")
    with pytest.raises(ValueError):
        users_db.set_api_keys(db_path, "AICT-KEY2-KEY2-KEY2", "", "x")
    with pytest.raises(ValueError):
        users_db.set_api_keys(db_path, "AICT-KEY2-KEY2-KEY2", "x", "")


def test_set_api_keys_missing_code_returns_false(db_path):
    """set_api_keys — 미존재 code 면 False."""
    ok = users_db.set_api_keys(
        db_path, "AICT-NONE-NONE-NONE", "pub", "ciphertext",
    )
    assert ok is False


def test_update_last_login_success(db_path):
    """update_last_login — last_login_at 채워짐."""
    users_db.create_user(db_path, "AICT-LOGN-LOGN-LOGN")
    before = users_db.get_user_by_code(db_path, "AICT-LOGN-LOGN-LOGN")
    assert before is not None
    assert before["last_login_at"] is None

    ok = users_db.update_last_login(db_path, "AICT-LOGN-LOGN-LOGN")
    assert ok is True

    after = users_db.get_user_by_code(db_path, "AICT-LOGN-LOGN-LOGN")
    assert after is not None
    assert after["last_login_at"] is not None
    # ISO 8601 UTC Z 접미사 검증
    assert after["last_login_at"].endswith("Z")


def test_update_last_login_missing_code_returns_false(db_path):
    """update_last_login — 미존재 code 면 False."""
    ok = users_db.update_last_login(db_path, "AICT-NONE-NONE-NONE")
    assert ok is False


def test_multiple_users_coexist(db_path):
    """다중 사용자 동시 저장 — 페러럴 모델 시뮬레이션."""
    users_db.create_user(db_path, "AICT-USR1-USR1-USR1", "sub_30d", "2026-06-30T00:00:00Z")
    users_db.create_user(db_path, "AICT-USR2-USR2-USR2", "sub_90d", "2026-08-31T00:00:00Z")

    u1 = users_db.get_user_by_code(db_path, "AICT-USR1-USR1-USR1")
    u2 = users_db.get_user_by_code(db_path, "AICT-USR2-USR2-USR2")
    assert u1 is not None and u2 is not None
    assert u1["id"] != u2["id"]
    assert u1["license_type"] == "sub_30d"
    assert u2["license_type"] == "sub_90d"
