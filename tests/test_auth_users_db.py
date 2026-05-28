"""users_db 모듈 — 실제 SQLite (in tmp_path) 사용, mock 0.

검증 대상 함수:
    - init_db (idempotent, sessions 테이블 포함)
    - create_user (정상 + 중복 코드 IntegrityError)
    - get_user_by_code (존재 / 미존재)
    - set_pin (정상 + 평문 거부 + 미존재 code)
    - set_api_keys (정상 + 빈 값 거부 + 미존재 code)
    - update_last_login (정상 + 미존재 code)
    - create_session_row / get_session / delete_session
    - delete_sessions_for_user / delete_all_sessions
    - cleanup_expired_sessions (만료 row 일괄 정리)

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

import sqlite3
import time

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


# ============================================================
# 세션 영속화 (Fly.io 재배포 대비, 2026-05-28)
# ============================================================


def test_init_db_creates_sessions_table(tmp_path):
    """init_db — sessions 테이블 + idx_sessions_expires 인덱스 생성."""
    path = tmp_path / "sess.db"
    users_db.init_db(path)

    with sqlite3.connect(str(path)) as conn:
        tables = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            ).fetchall()
        }
        assert "sessions" in tables
        indices = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'",
            ).fetchall()
        }
        assert "idx_sessions_expires" in indices


def test_create_session_row_and_get_session_round_trip(db_path):
    """create_session_row → get_session 으로 같은 user_code 회수."""
    users_db.create_user(db_path, "AICT-SES1-SES1-SES1")
    ok = users_db.create_session_row(
        db_path,
        token="tok_round_trip_1",
        user_code="AICT-SES1-SES1-SES1",
        ttl_sec=3600,
    )
    assert ok is True

    row = users_db.get_session(db_path, "tok_round_trip_1")
    assert row is not None
    assert row["token"] == "tok_round_trip_1"
    assert row["user_code"] == "AICT-SES1-SES1-SES1"
    # ISO 8601 UTC Z 접미사
    assert row["issued_at"].endswith("Z")
    assert row["expires_at"].endswith("Z")
    # expires > issued
    assert row["expires_at"] > row["issued_at"]


def test_get_session_returns_none_for_missing_token(db_path):
    """get_session — 미존재 토큰이면 None."""
    assert users_db.get_session(db_path, "no_such_token") is None
    assert users_db.get_session(db_path, "") is None


def test_get_session_excludes_expired_rows(db_path):
    """get_session — 만료된 세션은 row 가 있어도 None 반환."""
    users_db.create_user(db_path, "AICT-EXPI-EXPI-EXPI")
    # 1초 TTL — 즉시 만료시키기 위해 sleep.
    users_db.create_session_row(
        db_path, token="tok_expires_soon",
        user_code="AICT-EXPI-EXPI-EXPI", ttl_sec=1,
    )
    # 만료 직전엔 valid.
    assert users_db.get_session(db_path, "tok_expires_soon") is not None
    # 만료 통과 대기 — ISO 초 정밀도 + buffer.
    time.sleep(1.5)
    assert users_db.get_session(db_path, "tok_expires_soon") is None


def test_create_session_row_rejects_empty_token(db_path):
    """create_session_row — 빈 토큰은 ValueError."""
    with pytest.raises(ValueError, match="token"):
        users_db.create_session_row(
            db_path, token="", user_code="AICT-X-X-X", ttl_sec=60,
        )


def test_create_session_row_rejects_empty_user_code(db_path):
    """create_session_row — 빈 user_code 는 ValueError (DB 백엔드 정합)."""
    with pytest.raises(ValueError, match="user_code"):
        users_db.create_session_row(
            db_path, token="tok_x", user_code="", ttl_sec=60,
        )


def test_create_session_row_rejects_non_positive_ttl(db_path):
    """create_session_row — ttl_sec 0 또는 음수는 ValueError."""
    users_db.create_user(db_path, "AICT-TTLZ-TTLZ-TTLZ")
    with pytest.raises(ValueError, match="ttl_sec"):
        users_db.create_session_row(
            db_path, token="tok_ttl0",
            user_code="AICT-TTLZ-TTLZ-TTLZ", ttl_sec=0,
        )
    with pytest.raises(ValueError, match="ttl_sec"):
        users_db.create_session_row(
            db_path, token="tok_ttlneg",
            user_code="AICT-TTLZ-TTLZ-TTLZ", ttl_sec=-5,
        )


def test_delete_session_removes_row(db_path):
    """delete_session — 1건 삭제 + 멱등."""
    users_db.create_user(db_path, "AICT-DEL1-DEL1-DEL1")
    users_db.create_session_row(
        db_path, token="tok_del", user_code="AICT-DEL1-DEL1-DEL1", ttl_sec=60,
    )
    assert users_db.delete_session(db_path, "tok_del") is True
    # 두 번째 호출 — 이미 없음, False (멱등).
    assert users_db.delete_session(db_path, "tok_del") is False
    assert users_db.get_session(db_path, "tok_del") is None


def test_delete_session_empty_token_returns_false(db_path):
    """delete_session — 빈/None 토큰은 False (예외 X)."""
    assert users_db.delete_session(db_path, "") is False


def test_delete_sessions_for_user_removes_all_for_code(db_path):
    """delete_sessions_for_user — 특정 사용자 세션만 일괄 제거."""
    users_db.create_user(db_path, "AICT-MUL1-MUL1-MUL1")
    users_db.create_user(db_path, "AICT-MUL2-MUL2-MUL2")
    for i in range(3):
        users_db.create_session_row(
            db_path, token=f"u1_tok_{i}",
            user_code="AICT-MUL1-MUL1-MUL1", ttl_sec=600,
        )
    users_db.create_session_row(
        db_path, token="u2_tok_only",
        user_code="AICT-MUL2-MUL2-MUL2", ttl_sec=600,
    )

    removed = users_db.delete_sessions_for_user(db_path, "AICT-MUL1-MUL1-MUL1")
    assert removed == 3
    # 다른 사용자 세션은 살아있음.
    assert users_db.get_session(db_path, "u2_tok_only") is not None
    # 자기 세션은 모두 사라짐.
    for i in range(3):
        assert users_db.get_session(db_path, f"u1_tok_{i}") is None


def test_delete_sessions_for_user_unknown_returns_zero(db_path):
    """delete_sessions_for_user — 미존재 user_code 면 0."""
    assert users_db.delete_sessions_for_user(db_path, "AICT-NONE-NONE-NONE") == 0
    # 빈 user_code 도 0 (안전).
    assert users_db.delete_sessions_for_user(db_path, "") == 0


def test_delete_all_sessions_clears_table(db_path):
    """delete_all_sessions — sessions 테이블 전체 비움 + count 반환."""
    users_db.create_user(db_path, "AICT-ALL1-ALL1-ALL1")
    users_db.create_user(db_path, "AICT-ALL2-ALL2-ALL2")
    users_db.create_session_row(
        db_path, token="all_t1", user_code="AICT-ALL1-ALL1-ALL1", ttl_sec=60,
    )
    users_db.create_session_row(
        db_path, token="all_t2", user_code="AICT-ALL2-ALL2-ALL2", ttl_sec=60,
    )
    removed = users_db.delete_all_sessions(db_path)
    assert removed == 2
    assert users_db.get_session(db_path, "all_t1") is None
    assert users_db.get_session(db_path, "all_t2") is None


def test_cleanup_expired_sessions_removes_only_expired(db_path):
    """cleanup_expired_sessions — 만료 row 만 제거, 유효 row 는 보존."""
    users_db.create_user(db_path, "AICT-CLN1-CLN1-CLN1")
    # 1초 TTL — 만료 예정.
    users_db.create_session_row(
        db_path, token="expired_tok",
        user_code="AICT-CLN1-CLN1-CLN1", ttl_sec=1,
    )
    # 1시간 TTL — 유효 유지.
    users_db.create_session_row(
        db_path, token="alive_tok",
        user_code="AICT-CLN1-CLN1-CLN1", ttl_sec=3600,
    )
    time.sleep(1.5)
    removed = users_db.cleanup_expired_sessions(db_path)
    assert removed == 1

    # 유효 세션은 살아있고, 만료 세션은 row 자체가 사라짐.
    assert users_db.get_session(db_path, "alive_tok") is not None
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE token = ?",
            ("expired_tok",),
        ).fetchone()
        assert row[0] == 0


def test_cleanup_expired_sessions_empty_table_returns_zero(db_path):
    """cleanup_expired_sessions — 빈 테이블에선 0."""
    assert users_db.cleanup_expired_sessions(db_path) == 0


# ============================================================
# 봇 가동 상태 영속화 (Fly machine OOM/재배포 자동 복원, 2026-05-28)
# ============================================================


def test_init_db_creates_bot_running_column_with_default_zero(tmp_path):
    """init_db — 신규 DB 에 bot_running 컬럼이 NOT NULL DEFAULT 0 로 생김."""
    path = tmp_path / "br_fresh.db"
    users_db.init_db(path)

    with sqlite3.connect(str(path)) as conn:
        cols = {row[1]: row for row in conn.execute(
            "PRAGMA table_info(users)",
        ).fetchall()}
        assert "bot_running" in cols
        # tuple: (cid, name, type, notnull, dflt_value, pk)
        col = cols["bot_running"]
        assert col[2].upper() == "INTEGER"
        assert col[3] == 1  # NOT NULL
        assert str(col[4]) == "0"  # DEFAULT 0


def test_init_db_migrates_old_db_adds_bot_running(tmp_path):
    """init_db — bot_running 컬럼 없는 구버전 DB 도 ALTER TABLE 마이그레이션.

    구버전 스키마를 직접 만든 뒤 init_db 재호출 → 컬럼 추가 + 기존 row 보존 + 기본값 0.
    """
    path = tmp_path / "br_legacy.db"
    # 구버전 스키마 (bot_running 컬럼 없음) — 운영 DB 시뮬레이션.
    with sqlite3.connect(str(path)) as conn:
        conn.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE NOT NULL,
                pin_hash TEXT,
                api_key TEXT,
                api_secret_enc TEXT,
                license_type TEXT NOT NULL DEFAULT 'referral',
                expires_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                last_login_at TEXT
            )
        """)
        conn.execute(
            "INSERT INTO users (code, created_at, updated_at) VALUES (?, ?, ?)",
            ("AICT-LEGCY-LEG-LEG", "2026-01-01T00:00:00Z", "2026-01-01T00:00:00Z"),
        )
        conn.commit()

    # 신버전 init_db 호출 — ALTER TABLE 가 컬럼 추가해야 함.
    users_db.init_db(path)

    user = users_db.get_user_by_code(path, "AICT-LEGCY-LEG-LEG")
    assert user is not None
    # 기존 row 보존.
    assert user["code"] == "AICT-LEGCY-LEG-LEG"
    # 신규 컬럼 — 기본값 0.
    assert user["bot_running"] == 0

    # 재실행 idempotent — 두 번째 호출도 예외 없음.
    users_db.init_db(path)
    user2 = users_db.get_user_by_code(path, "AICT-LEGCY-LEG-LEG")
    assert user2["bot_running"] == 0


def test_set_bot_running_toggles_flag(db_path):
    """set_bot_running — True/False 박은 뒤 get_user_by_code 로 확인."""
    users_db.create_user(db_path, "AICT-BR01-BR01-BR01")
    user = users_db.get_user_by_code(db_path, "AICT-BR01-BR01-BR01")
    assert user is not None
    assert user["bot_running"] == 0  # 신규 사용자 기본값

    ok = users_db.set_bot_running(db_path, "AICT-BR01-BR01-BR01", True)
    assert ok is True
    user = users_db.get_user_by_code(db_path, "AICT-BR01-BR01-BR01")
    assert user["bot_running"] == 1

    ok = users_db.set_bot_running(db_path, "AICT-BR01-BR01-BR01", False)
    assert ok is True
    user = users_db.get_user_by_code(db_path, "AICT-BR01-BR01-BR01")
    assert user["bot_running"] == 0


def test_set_bot_running_missing_code_returns_false(db_path):
    """set_bot_running — 미존재 code 면 False (예외 X)."""
    assert users_db.set_bot_running(db_path, "AICT-NONE-NONE-NONE", True) is False


def test_list_running_codes_filters_by_flag(db_path):
    """list_running_codes — bot_running=1 사용자만 반환."""
    # 3명 등록, 그 중 2명만 가동 마킹.
    users_db.create_user(db_path, "AICT-RUN1-RUN1-RUN1")
    users_db.create_user(db_path, "AICT-RUN2-RUN2-RUN2")
    users_db.create_user(db_path, "AICT-STOP-STOP-STOP")
    users_db.set_bot_running(db_path, "AICT-RUN1-RUN1-RUN1", True)
    users_db.set_bot_running(db_path, "AICT-RUN2-RUN2-RUN2", True)
    # STOP 사용자는 그대로 0.

    codes = users_db.list_running_codes(db_path)
    assert sorted(codes) == ["AICT-RUN1-RUN1-RUN1", "AICT-RUN2-RUN2-RUN2"]


def test_list_running_codes_empty_when_none_running(db_path):
    """list_running_codes — 가동 중 사용자 없으면 빈 list."""
    users_db.create_user(db_path, "AICT-ZRO1-ZRO1-ZRO1")
    users_db.create_user(db_path, "AICT-ZRO2-ZRO2-ZRO2")
    assert users_db.list_running_codes(db_path) == []
