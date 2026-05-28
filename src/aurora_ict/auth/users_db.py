"""사용자 DB — 라이선스 코드 / PIN 해시 / 암호화된 API 키 / 만료 영속화.

파트너 결정 2026-05-28 — Aurora-ICT 를 진짜 SaaS 로 전환하기 위한 다중 사용자 저장소.

설계:
    - SQLite (Python 내장 ``sqlite3``) — 별도 의존성 X, 단일 파일 (.db)
    - 저장 위치: ``<data_dir>/users.db`` (``aurora_ict.paths.data_dir`` 활용)
    - 한 봇 인스턴스가 여러 사용자를 서빙할 수 있음 (페러럴 모델 대응)
    - 평문 PIN 절대 저장 X — ``aurora_ict.auth.pin.hash_pin`` 결과만 보관
    - 평문 API secret 절대 저장 X — ``aurora_ict.auth.keystore.encrypt_secret`` 결과만 보관

테이블 스키마:
    ``users(id, code, pin_hash, api_key, api_secret_enc,
            license_type, expires_at, created_at, updated_at, last_login_at)``

    - ``code`` UNIQUE — 라이선스 발급 코드 (예: ``AICT-XXXX-XXXX-XXXX``)
    - ``license_type`` — ``referral`` / ``sub_30d`` / ``sub_90d`` / ``sub_365d``
    - ``expires_at`` — ISO 8601 UTC (예: ``2026-12-31T23:59:59Z``)
    - 모든 시각 필드는 UTC ISO 8601 — 봇이 KST 변환은 표시 직전에만

보안:
    - 모든 쿼리 parameterized (``?`` placeholder) — SQL injection 방지
    - 컨텍스트 매니저 (``with sqlite3.connect``) — 트랜잭션 안전
    - 외부 호출자에게 평문 비밀 노출 X (조회 결과에 ``api_secret_enc`` 그대로 반환,
      복호화는 호출자가 ``keystore.decrypt_secret`` 명시적으로 호출)

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# DDL — idempotent 하게 init_db 에서 사용. CREATE TABLE IF NOT EXISTS 라 재실행 안전.
_DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
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
"""

# code 로 lookup 빈도 높음 — UNIQUE 인덱스가 이미 SQLite 가 자동 생성하지만 명시.
_DDL_INDEX_CODE = "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_code ON users(code)"


def _utcnow_iso() -> str:
    """현재 UTC 시각을 ISO 8601 (초 단위, Z 접미사) 으로 반환.

    Returns:
        예: ``"2026-05-28T12:34:56Z"``
    """
    # microsecond 제거 — DB 가독성 + 충분한 정밀도
    return datetime.now(tz=UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _connect(db_path: Path | str) -> sqlite3.Connection:
    """SQLite 연결 — 외래키 켜고 Row factory 적용.

    Args:
        db_path: .db 파일 경로 (상위 디렉토리는 호출자 책임).

    Returns:
        ``sqlite3.Connection``. 사용 후 ``with`` 컨텍스트로 자동 close 권장.
    """
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    # 외래키 — 현 스키마엔 FK 없지만 향후 license_events 등 확장 대비.
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: Path | str) -> None:
    """users.db 테이블 생성 — idempotent (이미 있으면 no-op).

    Args:
        db_path: 생성/오픈할 SQLite 파일 경로. 부모 디렉토리는 미리 존재해야 함.

    Raises:
        sqlite3.OperationalError: 디스크 권한 없음 / 경로 문제.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(path) as conn:
        conn.execute(_DDL_USERS)
        conn.execute(_DDL_INDEX_CODE)
        conn.commit()


def create_user(
    db_path: Path | str,
    code: str,
    license_type: str = "referral",
    expires_at: str | None = None,
) -> int:
    """라이선스 코드 발급 직후 신규 사용자 row 생성.

    PIN / API 키는 사용자가 첫 로그인 / 키 등록 단계에서 채움 (NULL 로 시작).

    Args:
        db_path: users.db 경로.
        code: 발급된 라이선스 코드 (``AICT-XXXX-XXXX-XXXX``). UNIQUE.
        license_type: ``referral`` / ``sub_30d`` / ``sub_90d`` / ``sub_365d``.
        expires_at: ISO 8601 UTC 만료. ``referral`` 은 None 가능.

    Returns:
        새로 생성된 row 의 id (INTEGER PRIMARY KEY).

    Raises:
        sqlite3.IntegrityError: ``code`` 가 이미 존재.
    """
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO users (code, license_type, expires_at, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (code, license_type, expires_at, now, now),
        )
        conn.commit()
        return int(cur.lastrowid)


def get_user_by_code(db_path: Path | str, code: str) -> dict[str, Any] | None:
    """라이선스 코드로 사용자 조회.

    Args:
        db_path: users.db 경로.
        code: 조회할 라이선스 코드.

    Returns:
        dict (컬럼명 → 값) 또는 미존재 시 None. ``api_secret_enc`` 는 암호문 상태 그대로.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE code = ?",
            (code,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)


def set_pin(db_path: Path | str, code: str, pin_hash: str) -> bool:
    """첫 PIN 설정 또는 변경 — ``pin_hash`` 는 ``aurora_ict.auth.pin.hash_pin`` 결과.

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        pin_hash: ``pbkdf2_sha256$...$...$...`` 형식 해시 (평문 PIN 금지).

    Returns:
        True 면 업데이트 성공, False 면 해당 code 존재 X.

    Raises:
        ValueError: ``pin_hash`` 가 명백히 평문 (algo prefix 부재) 인 경우.
    """
    # 가드 — 실수로 평문 PIN 이 넘어오는 사고 방지. hash_pin 결과는 무조건 algo$ 접두.
    if "$" not in pin_hash:
        raise ValueError(
            "pin_hash 가 해시 형식이 아닙니다 (평문 PIN 저장 시도 가능성).",
        )
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE users
            SET pin_hash = ?, updated_at = ?
            WHERE code = ?
            """,
            (pin_hash, now, code),
        )
        conn.commit()
        return cur.rowcount > 0


def set_api_keys(
    db_path: Path | str,
    code: str,
    api_key: str,
    api_secret_enc: str,
) -> bool:
    """거래소 API 키 등록 — ``api_secret_enc`` 는 keystore.encrypt_secret 결과여야 함.

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        api_key: 거래소 발급 public key (평문 OK).
        api_secret_enc: Fernet 으로 암호화된 secret (base64 문자열).

    Returns:
        True 면 업데이트 성공, False 면 해당 code 존재 X.

    Raises:
        ValueError: ``api_key`` 또는 ``api_secret_enc`` 가 빈 문자열.
    """
    if not api_key or not api_secret_enc:
        raise ValueError("api_key 와 api_secret_enc 는 비어있을 수 없습니다.")
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE users
            SET api_key = ?, api_secret_enc = ?, updated_at = ?
            WHERE code = ?
            """,
            (api_key, api_secret_enc, now, code),
        )
        conn.commit()
        return cur.rowcount > 0


def update_last_login(db_path: Path | str, code: str) -> bool:
    """로그인 성공 시 ``last_login_at`` 갱신.

    Args:
        db_path: users.db 경로.
        code: 로그인 성공한 사용자 코드.

    Returns:
        True 면 갱신 성공, False 면 code 미존재.
    """
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE users
            SET last_login_at = ?, updated_at = ?
            WHERE code = ?
            """,
            (now, now, code),
        )
        conn.commit()
        return cur.rowcount > 0


__all__ = [
    "create_user",
    "get_user_by_code",
    "init_db",
    "set_api_keys",
    "set_pin",
    "update_last_login",
]
