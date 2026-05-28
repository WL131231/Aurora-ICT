"""세션 영속화 — pin 모듈 DB 백엔드 동작 검증 (mock 0, 실 SQLite).

파트너 결정 2026-05-28 — Fly.io 가 PR 머지마다 재배포 → 봇 프로세스 죽으면 메모리
dict 가 비워져 사용자가 매번 다시 로그인. 이걸 막기 위해 pin 모듈이 ``users.db`` 의
``sessions`` 테이블로 토큰 영속화하는지 검증.

핵심 시나리오:
    1. set_session_db_path 후 create_session → DB row 생성 + 토큰 유효
    2. 모듈 메모리 (``_active_sessions``) 비워도 validate_session 여전히 True
       — 봇 재시작 시뮬레이션 (메모리는 휘발, DB 는 영속)
    3. revoke_session 이 DB row 도 삭제
    4. revoke_sessions_for_user / revoke_all_sessions DB 반영
    5. 메모리 모드 (set_session_db_path 미호출 / None) 는 기존 동작 유지 — .exe 호환

담당: 지영민 (SaaS 세션 영속화 PR)
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator

import pytest

from aurora_ict.auth import pin, users_db


@pytest.fixture
def db_path(tmp_path):
    """격리된 users.db — sessions 테이블까지 init 된 상태."""
    path = tmp_path / "users.db"
    users_db.init_db(path)
    # FK 검증을 위해 사용자 row 미리 생성 — sessions.user_code 가 users.code 참조.
    users_db.create_user(path, "AICT-PERS-PERS-PERS")
    users_db.create_user(path, "AICT-PER2-PER2-PER2")
    return path


@pytest.fixture(autouse=True)
def _reset_pin_state(db_path) -> Iterator[None]:
    """각 테스트 격리 — DB path 와 메모리 dict 모두 초기화."""
    pin.set_session_db_path(None)
    pin._active_sessions.clear()  # 메모리 fallback 잔재 제거.
    yield
    pin.set_session_db_path(None)
    pin._active_sessions.clear()


# ============================================================
# 1. DB 백엔드 활성화 — create / validate 흐름
# ============================================================


def test_set_session_db_path_activates_db_backend(db_path) -> None:
    """set_session_db_path 호출 후 get_session_db_path 동일 경로 반환."""
    assert pin.get_session_db_path() is None
    pin.set_session_db_path(db_path)
    assert pin.get_session_db_path() == db_path


def test_create_session_persists_to_db(db_path) -> None:
    """create_session — DB 모드에서 sessions 테이블에 row 1건 생성."""
    pin.set_session_db_path(db_path)
    token = pin.create_session(user_code="AICT-PERS-PERS-PERS", ttl_sec=3600)

    assert isinstance(token, str)
    assert len(token) > 0

    # 메모리 dict 에는 들어가지 X — DB 백엔드는 메모리 우회.
    assert token not in pin._active_sessions

    # DB row 직접 확인.
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT user_code FROM sessions WHERE token = ?",
            (token,),
        ).fetchone()
        assert row is not None
        assert row[0] == "AICT-PERS-PERS-PERS"


def test_create_session_db_backend_requires_user_code(db_path) -> None:
    """DB 백엔드는 빈 user_code 거부 — sessions.user_code NOT NULL."""
    pin.set_session_db_path(db_path)
    with pytest.raises(ValueError, match="user_code"):
        pin.create_session(user_code="", ttl_sec=3600)


def test_validate_session_survives_memory_wipe(db_path) -> None:
    """봇 재시작 시뮬레이션 — 메모리 비워도 DB 백엔드라 토큰 유효.

    이게 이 PR 의 핵심: Fly.io 재배포 시 ``_active_sessions`` 가 날아가도
    사용자는 다시 로그인할 필요 없음.
    """
    pin.set_session_db_path(db_path)
    token = pin.create_session(user_code="AICT-PERS-PERS-PERS", ttl_sec=3600)

    # 봇 재시작 시뮬레이션 — 메모리 dict 강제 초기화 (실제 프로세스 재시작과 동등).
    pin._active_sessions.clear()

    # DB 백엔드라 여전히 유효.
    assert pin.validate_session(token) is True
    assert pin.get_user_from_session(token) == "AICT-PERS-PERS-PERS"


def test_get_user_from_session_returns_code_from_db(db_path) -> None:
    """get_user_from_session — DB 백엔드에서 user_code 정상 회수."""
    pin.set_session_db_path(db_path)
    token = pin.create_session(user_code="AICT-PER2-PER2-PER2", ttl_sec=3600)

    pin._active_sessions.clear()  # 메모리 fallback 의존 X 확인.
    assert pin.get_user_from_session(token) == "AICT-PER2-PER2-PER2"


def test_get_user_from_session_returns_none_for_unknown_token(db_path) -> None:
    """get_user_from_session — DB 백엔드에서 미존재 토큰이면 None."""
    pin.set_session_db_path(db_path)
    assert pin.get_user_from_session("no_such_token_xyz") is None
    assert pin.get_user_from_session("") is None
    assert pin.get_user_from_session(None) is None


# ============================================================
# 2. revoke 동작 — DB 반영
# ============================================================


def test_revoke_session_deletes_db_row(db_path) -> None:
    """revoke_session — DB row 즉시 삭제, validate_session=False."""
    pin.set_session_db_path(db_path)
    token = pin.create_session(user_code="AICT-PERS-PERS-PERS", ttl_sec=3600)
    assert pin.validate_session(token) is True

    pin.revoke_session(token)
    assert pin.validate_session(token) is False

    # DB 직접 확인 — row 자체가 없어야 함.
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE token = ?",
            (token,),
        ).fetchone()
        assert row[0] == 0


def test_revoke_session_idempotent(db_path) -> None:
    """revoke_session — 미존재 토큰 / None / 빈 문자열 모두 예외 X."""
    pin.set_session_db_path(db_path)
    pin.revoke_session(None)
    pin.revoke_session("")
    pin.revoke_session("nonexistent_token")
    # 정상 토큰 revoke 후 재호출도 OK.
    token = pin.create_session(user_code="AICT-PERS-PERS-PERS", ttl_sec=3600)
    pin.revoke_session(token)
    pin.revoke_session(token)  # 두 번째도 예외 없음.


def test_revoke_sessions_for_user_only_affects_owner(db_path) -> None:
    """revoke_sessions_for_user — 다른 사용자 세션은 살아있어야."""
    pin.set_session_db_path(db_path)
    t1a = pin.create_session(user_code="AICT-PERS-PERS-PERS", ttl_sec=3600)
    t1b = pin.create_session(user_code="AICT-PERS-PERS-PERS", ttl_sec=3600)
    t2 = pin.create_session(user_code="AICT-PER2-PER2-PER2", ttl_sec=3600)

    count = pin.revoke_sessions_for_user("AICT-PERS-PERS-PERS")
    assert count == 2
    assert pin.validate_session(t1a) is False
    assert pin.validate_session(t1b) is False
    # 다른 사용자 토큰은 살아있음.
    assert pin.validate_session(t2) is True


def test_revoke_all_sessions_clears_db_table(db_path) -> None:
    """revoke_all_sessions — DB 의 모든 세션 row 제거, 개수 반환."""
    pin.set_session_db_path(db_path)
    pin.create_session(user_code="AICT-PERS-PERS-PERS", ttl_sec=3600)
    pin.create_session(user_code="AICT-PER2-PER2-PER2", ttl_sec=3600)

    count = pin.revoke_all_sessions()
    assert count == 2
    with sqlite3.connect(str(db_path)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()
        assert row[0] == 0


# ============================================================
# 3. 메모리 백엔드 (.exe 호환) — set_session_db_path 미호출
# ============================================================


def test_memory_backend_when_db_path_not_set() -> None:
    """set_session_db_path 미호출 — 기존 메모리 dict 동작 그대로."""
    # autouse fixture 가 None 으로 reset 한 상태.
    assert pin.get_session_db_path() is None

    token = pin.create_session(user_code="legacy_user", ttl_sec=3600)
    # 메모리 dict 에 박힘.
    assert token in pin._active_sessions
    assert pin.validate_session(token) is True
    assert pin.get_user_from_session(token) == "legacy_user"


def test_memory_backend_allows_empty_user_code() -> None:
    """메모리 백엔드는 빈 user_code 허용 — 단일 사용자 .exe 레거시 경로."""
    assert pin.get_session_db_path() is None
    token = pin.create_session(user_code="", ttl_sec=3600)
    # validate 는 True (토큰 자체는 유효).
    assert pin.validate_session(token) is True
    # 하지만 user_code 가 비어있어 SaaS 인증 컨텍스트에서는 None.
    assert pin.get_user_from_session(token) is None


def test_memory_backend_revoke_clears_dict() -> None:
    """메모리 백엔드 revoke 흐름 — DB 호출 X."""
    token = pin.create_session(user_code="mem_user", ttl_sec=3600)
    pin.revoke_session(token)
    assert pin.validate_session(token) is False
    assert token not in pin._active_sessions


def test_switching_backend_does_not_leak_memory_sessions(db_path) -> None:
    """메모리 모드에서 만든 토큰은 DB 백엔드 전환 후 무효 — 백엔드 격리.

    Why: 두 백엔드를 동시에 살려두면 일관성 깨짐 (DB 토큰을 메모리에서 못 찾고 그 반대도).
    set_session_db_path 가 백엔드 스위치 역할을 한다는 계약 명시.
    """
    # 메모리 모드 세션 생성.
    mem_token = pin.create_session(user_code="legacy", ttl_sec=3600)
    assert pin.validate_session(mem_token) is True

    # DB 백엔드 전환.
    pin.set_session_db_path(db_path)
    # DB 에 없는 토큰이라 무효.
    assert pin.validate_session(mem_token) is False
