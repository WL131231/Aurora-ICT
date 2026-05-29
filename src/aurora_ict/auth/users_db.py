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

    ``sessions(token, user_code, issued_at, expires_at)``

    - ``token`` PRIMARY KEY — secrets.token_urlsafe(32) 결과
    - ``user_code`` — users.code FK (소프트 참조 — 사용자 row 삭제 시 cascade X,
      revoke_sessions_for_user 가 명시 정리)
    - 만료 시각도 ISO 8601 UTC — 인덱스 ``idx_sessions_expires`` 로 cleanup 가속

    파트너 결정 2026-05-28 — Fly.io 재배포 시 봇 프로세스가 죽어도 세션 유지
    하기 위해 메모리 dict 를 DB 로 영속화. 재배포 후에도 사용자가 다시 로그인할
    필요 없음.

보안:
    - 모든 쿼리 parameterized (``?`` placeholder) — SQL injection 방지
    - 컨텍스트 매니저 (``with sqlite3.connect``) — 트랜잭션 안전
    - 외부 호출자에게 평문 비밀 노출 X (조회 결과에 ``api_secret_enc`` 그대로 반환,
      복호화는 호출자가 ``keystore.decrypt_secret`` 명시적으로 호출)

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

import json
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# DDL — idempotent 하게 init_db 에서 사용. CREATE TABLE IF NOT EXISTS 라 재실행 안전.
#
# 2026-05-28: ``bot_running`` 컬럼 추가 (봇 가동 상태 영속화) — Fly.io machine
# OOM / 재배포로 프로세스가 죽어도 다시 살아날 때 list_running_codes 로
# 자동 재가동. 기존 DB 는 ``_ensure_bot_running_column`` 가 ALTER TABLE
# 마이그레이션 (idempotent, PRAGMA table_info 확인 후 없으면 추가).
#
# 2026-05-29 (Live 모드 지원): 거래소 API 키를 demo/live 슬롯으로 분리 저장.
# 기존 ``api_key`` / ``api_secret_enc`` 는 deprecated 이지만 backward compat
# 위해 컬럼 유지 — ``_ensure_dual_key_columns`` 가 ALTER TABLE 로 demo/live
# 컬럼 추가하고 기존 row 의 api_key 를 demo 슬롯으로 복사 (가장 보수적 가정).
# 이유: Bybit 은 Demo Trading 과 Live 가 별도 시스템 → API 키도 분리 발급.
# 같은 키로 양쪽 호출 시 401 / sign error 발생 → 모드별로 다른 키 보관 필수.
_DDL_USERS = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT UNIQUE NOT NULL,
    pin_hash TEXT,
    api_key TEXT,
    api_secret_enc TEXT,
    demo_api_key TEXT,
    demo_api_secret_enc TEXT,
    live_api_key TEXT,
    live_api_secret_enc TEXT,
    license_type TEXT NOT NULL DEFAULT 'referral',
    expires_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_login_at TEXT,
    bot_running INTEGER NOT NULL DEFAULT 0
)
"""

# code 로 lookup 빈도 높음 — UNIQUE 인덱스가 이미 SQLite 가 자동 생성하지만 명시.
_DDL_INDEX_CODE = "CREATE UNIQUE INDEX IF NOT EXISTS idx_users_code ON users(code)"

# 세션 영속화 — Fly.io 재배포 시 사용자 강제 로그아웃 방지 (2026-05-28).
# FK 는 선언만 — PRAGMA foreign_keys 가 켜져 있어도 users 삭제 흐름이 현재 없으므로
# 실 cascade 동작은 발생 X. revoke_sessions_for_user 가 명시적으로 정리.
_DDL_SESSIONS = """
CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_code TEXT NOT NULL,
    issued_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    FOREIGN KEY (user_code) REFERENCES users(code)
)
"""

# 만료 cleanup 가속용 — startup 시 _cleanup_expired_sessions 가 full scan 대신 인덱스 활용.
_DDL_INDEX_SESSIONS_EXPIRES = (
    "CREATE INDEX IF NOT EXISTS idx_sessions_expires ON sessions(expires_at)"
)


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


def _ensure_last_run_mode_column(conn: sqlite3.Connection) -> None:
    """2026-05-29 hot-fix: 사용자별 마지막 run_mode (demo/live) 영속화.

    배경: fly machine 재시작 시 ``auto_resume_running_bots`` 가
    ``base_settings.run_mode = DEMO`` 기준으로 봇 재가동 시도. LIVE 모드로
    가동 중이던 사용자도 DEMO 로 가동 시도 → DEMO 키 미등록 → 봇 죽음.
    이 컬럼이 있으면 사용자가 마지막에 가동한 run_mode 로 정확하게 재가동.

    마이그레이션: ``last_run_mode TEXT`` 컬럼 idempotent 추가. NULL 허용
    (구식 사용자는 base_settings.run_mode fallback).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "last_run_mode" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN last_run_mode TEXT")


def _ensure_bot_running_symbols_column(conn: sqlite3.Connection) -> None:
    """2026-05-29 PR B: 멀티 페어 (BTC + ETH 동시 진입) 영속화 — bot_running
    boolean → bot_running_symbols JSON 배열 컬럼.

    마이그레이션:
      1) bot_running_symbols TEXT 컬럼 idempotent 추가.
      2) 기존 ``bot_running = 1`` row 의 bot_running_symbols 가 비어있으면
         ``'["BTC/USDT:USDT"]'`` 로 자동 설정 (기존 단일 페어 가동 상태 보존).

    bot_running 컬럼은 backward compat 위해 그대로 남겨둠 (사용 안 함).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "bot_running_symbols" not in cols:
        conn.execute(
            "ALTER TABLE users ADD COLUMN bot_running_symbols TEXT",
        )
    # 기존 bot_running=1 인데 bot_running_symbols 비어있는 row 만 마이그레이션.
    conn.execute(
        """
        UPDATE users
        SET bot_running_symbols = '["BTC/USDT:USDT"]'
        WHERE bot_running = 1
          AND (bot_running_symbols IS NULL OR bot_running_symbols = '')
        """,
    )


def _ensure_bot_running_column(conn: sqlite3.Connection) -> None:
    """기존 users 테이블에 ``bot_running`` 컬럼 없으면 ALTER TABLE 로 추가.

    2026-05-28 — 봇 가동 상태 영속화 마이그레이션. 이미 컬럼이 있으면 no-op.
    신규 DB 는 ``_DDL_USERS`` 가 처음부터 컬럼을 포함하므로 PRAGMA 결과에
    이미 ``bot_running`` 이 들어 있어 ALTER 가 호출되지 않는다.

    Args:
        conn: 열려있는 SQLite 연결 (호출자가 commit 책임).
    """
    # PRAGMA table_info 는 (cid, name, type, notnull, dflt_value, pk) 튜플 반환.
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "bot_running" not in cols:
        # NOT NULL DEFAULT 0 — 기존 row 도 자동으로 0 (정지) 으로 채워짐.
        conn.execute(
            "ALTER TABLE users ADD COLUMN bot_running INTEGER NOT NULL DEFAULT 0",
        )


def _ensure_dual_key_columns(conn: sqlite3.Connection) -> None:
    """기존 users 테이블에 demo/live 키 컬럼 없으면 ALTER TABLE 로 추가.

    2026-05-29 (Live 모드 지원) — 단일 ``api_key`` / ``api_secret_enc`` 슬롯을
    demo/live 로 분리. 마이그레이션 정책:

      1) 4 컬럼 (``demo_api_key``, ``demo_api_secret_enc``, ``live_api_key``,
         ``live_api_secret_enc``) 이 없으면 추가 (각각 nullable TEXT).
      2) 기존 ``api_key`` / ``api_secret_enc`` 가 NOT NULL 인데 새 demo 컬럼은
         비어있는 row 들은 demo 슬롯으로 1회 복사 — 사용자가 Live 키 신규 등록
         전까지 데모 모드는 끊김 없이 동작.

    Why 가장 보수적 가정 = demo: 기본 ``run_mode`` 가 demo 이고, 지금까지 사용자가
    등록한 키는 Bybit Demo Trading 키였음. Live 슬롯으로 복사하면 잘못된 키로
    실거래 가능성 있어 위험 — 무조건 demo 로 박는다.

    Args:
        conn: 열려있는 SQLite 연결 (호출자가 commit 책임).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    # 누락 컬럼만 골라 ALTER (idempotent — 이미 있으면 skip).
    for col in (
        "demo_api_key",
        "demo_api_secret_enc",
        "live_api_key",
        "live_api_secret_enc",
    ):
        if col not in cols:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT")

    # 기존 key 를 demo 슬롯으로 1회 복사 — demo 슬롯 비어있는 row 만 대상.
    # 이미 demo 슬롯에 값이 있으면 (신규 등록 완료 후) 덮어쓰지 않음 (멱등).
    conn.execute(
        """
        UPDATE users
        SET demo_api_key = api_key,
            demo_api_secret_enc = api_secret_enc
        WHERE api_key IS NOT NULL
          AND api_secret_enc IS NOT NULL
          AND (demo_api_key IS NULL OR demo_api_key = '')
        """,
    )


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
        conn.execute(_DDL_SESSIONS)
        conn.execute(_DDL_INDEX_SESSIONS_EXPIRES)
        # 기존 DB 마이그레이션 — 누락 컬럼 추가 + 기존 데이터 보존.
        _ensure_bot_running_column(conn)
        _ensure_dual_key_columns(conn)
        # 2026-05-29 PR B: 멀티 페어 영속화 컬럼 + 기존 데이터 마이그레이션.
        _ensure_bot_running_symbols_column(conn)
        # 2026-05-29 hot-fix: LIVE 사용자 자동 재가동 버그 해소용 last_run_mode.
        _ensure_last_run_mode_column(conn)
        conn.commit()
    # 2026-05-28: 공지사항 테이블도 같은 파일에 idempotent 생성.
    from aurora_ict.auth.notices_db import init_notices_table
    init_notices_table(path)


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
    mode: str = "demo",
) -> bool:
    """거래소 API 키 등록 — ``mode`` 슬롯 (demo/live) 에 저장.

    2026-05-29: Live 모드 지원 — 기존 단일 슬롯 (``api_key`` / ``api_secret_enc``)
    이 demo/live 로 분리됨. ``mode`` 파라미터 추가, 기본 ``"demo"`` 로 backward compat.

    동기화 정책: ``mode="demo"`` 일 때는 legacy ``api_key`` / ``api_secret_enc``
    컬럼도 함께 갱신 (기존 코드 경로가 이 컬럼을 읽고 있을 수 있어 안전망).
    ``mode="live"`` 는 live 슬롯만 갱신하고 legacy 컬럼은 건드리지 않음.

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        api_key: 거래소 발급 public key (평문 OK).
        api_secret_enc: Fernet 으로 암호화된 secret (base64 문자열).
        mode: ``"demo"`` 또는 ``"live"``. 그 외 값은 ValueError.

    Returns:
        True 면 업데이트 성공, False 면 해당 code 존재 X.

    Raises:
        ValueError: ``api_key`` 또는 ``api_secret_enc`` 가 빈 문자열,
            또는 ``mode`` 가 허용 외 값.
    """
    if not api_key or not api_secret_enc:
        raise ValueError("api_key 와 api_secret_enc 는 비어있을 수 없습니다.")
    if mode not in ("demo", "live"):
        raise ValueError(f"mode 는 'demo' 또는 'live' 여야 합니다: {mode!r}")
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        if mode == "demo":
            # demo 슬롯 + legacy 컬럼 동시 갱신 (backward compat).
            cur = conn.execute(
                """
                UPDATE users
                SET demo_api_key = ?,
                    demo_api_secret_enc = ?,
                    api_key = ?,
                    api_secret_enc = ?,
                    updated_at = ?
                WHERE code = ?
                """,
                (api_key, api_secret_enc, api_key, api_secret_enc, now, code),
            )
        else:  # live
            cur = conn.execute(
                """
                UPDATE users
                SET live_api_key = ?,
                    live_api_secret_enc = ?,
                    updated_at = ?
                WHERE code = ?
                """,
                (api_key, api_secret_enc, now, code),
            )
        conn.commit()
        return cur.rowcount > 0


def get_api_keys(
    db_path: Path | str,
    code: str,
    mode: str,
) -> tuple[str, str] | None:
    """모드별 거래소 API 키 조회 — ``(api_key, api_secret_enc)`` 반환.

    2026-05-29: Live 모드 지원. demo 슬롯이 비어 있고 legacy ``api_key`` 가 있는
    구식 row 대비 — demo 조회 시 legacy 컬럼도 fallback 으로 본다.

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        mode: ``"demo"`` 또는 ``"live"``.

    Returns:
        ``(api_key, api_secret_enc)`` 튜플, 키 미등록이면 None.
        ``api_secret_enc`` 는 암호문 그대로 (호출자가 keystore.decrypt_secret).

    Raises:
        ValueError: ``mode`` 가 허용 외 값.
    """
    if mode not in ("demo", "live"):
        raise ValueError(f"mode 는 'demo' 또는 'live' 여야 합니다: {mode!r}")
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT api_key, api_secret_enc,
                   demo_api_key, demo_api_secret_enc,
                   live_api_key, live_api_secret_enc
            FROM users WHERE code = ?
            """,
            (code,),
        ).fetchone()
        if row is None:
            return None
    if mode == "live":
        ak = row["live_api_key"]
        sk = row["live_api_secret_enc"]
    else:
        # demo — 신규 슬롯 우선, 없으면 legacy fallback (마이그레이션 전 row 대응).
        ak = row["demo_api_key"] or row["api_key"]
        sk = row["demo_api_secret_enc"] or row["api_secret_enc"]
    if not ak or not sk:
        return None
    return (str(ak), str(sk))


def has_api_keys(db_path: Path | str, code: str, mode: str) -> bool:
    """모드별 API 키 등록 여부 — boolean.

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        mode: ``"demo"`` 또는 ``"live"``.

    Returns:
        해당 모드 슬롯에 키가 모두 들어있으면 True.

    Raises:
        ValueError: ``mode`` 가 허용 외 값.
    """
    return get_api_keys(db_path, code, mode) is not None


def update_license(
    db_path: Path | str,
    code: str,
    license_type: str,
    expires_at: str | None,
) -> bool:
    """라이선스 type / 만료일 갱신 — 라이선스 서버 동기화 후 호출 (2026-05-29).

    배경: ``setup_pin`` 시 라이선스 서버 검증 없이 기본 ``'referral'`` 로 박혀,
    실제 sub_30d / sub_90d / sub_365d 사용자가 UI 에서 "레퍼럴 / 무기한" 으로
    잘못 표시되는 버그. 관리자 (또는 추후 자동 sync) 가 정정 가능하도록.

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        license_type: ``referral`` / ``sub_30d`` / ``sub_90d`` / ``sub_365d``.
            그 외 값은 ValueError.
        expires_at: ISO 8601 UTC (예: ``"2027-05-28T23:59:59Z"``) 또는 None
            (referral 무기한). 형식 검증은 호출자 책임.

    Returns:
        True 면 UPDATE 1건 성공, False 면 code 미존재.

    Raises:
        ValueError: ``license_type`` 이 허용 외 값.
    """
    allowed = {"referral", "sub_30d", "sub_90d", "sub_365d"}
    if license_type not in allowed:
        raise ValueError(
            f"license_type 허용 외 값: {license_type!r} (허용: {sorted(allowed)})",
        )
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE users
            SET license_type = ?, expires_at = ?, updated_at = ?
            WHERE code = ?
            """,
            (license_type, expires_at, now, code),
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


# ============================================================
# 봇 가동 상태 영속화 — Fly.io machine OOM/재배포 후 자동 복원 (2026-05-28)
# ============================================================


def _parse_running_symbols(raw: str | None) -> list[str]:
    """bot_running_symbols 컬럼 (JSON 배열 문자열) → list[str]. 손상/빈값은 [].
    """
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [str(s) for s in data if isinstance(s, str) and s]


def get_bot_running_symbols(db_path: Path | str, code: str) -> list[str]:
    """사용자 코드의 가동 중 symbol list 조회 — 2026-05-29 PR B.

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.

    Returns:
        가동 중인 symbol list (예: ``["BTC/USDT:USDT", "ETH/USDT:USDT"]``).
        사용자 미존재 / 가동 없음이면 빈 list.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT bot_running_symbols FROM users WHERE code = ?",
            (code,),
        ).fetchone()
    if row is None:
        return []
    return _parse_running_symbols(row["bot_running_symbols"])


def set_bot_running(
    db_path: Path | str,
    code: str,
    running: bool,
    symbol: str = "BTC/USDT:USDT",
) -> bool:
    """봇 가동 상태 플래그 박기 — symbol 단위 (멀티 페어, 2026-05-29 PR B).

    Why: ``MultiUserBotManager._slots`` 는 in-memory dict 라 Fly machine
    OOM / 재배포 시 사라짐. 이 컬럼 (JSON 배열) 을 영속화해 둬야 startup hook
    (``saas.py``) 이 ``list_running_bots`` 로 페어별 자동 재가동 가능.

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        running: True 면 symbol 을 list 에 추가, False 면 제거.
        symbol: ccxt 형식 symbol (예: "BTC/USDT:USDT"). 기본값 backward compat.

    Returns:
        True 면 UPDATE 1건 성공, False 면 해당 code 가 DB 에 없음 (no-op).
    """
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT bot_running_symbols FROM users WHERE code = ?",
            (code,),
        ).fetchone()
        if row is None:
            return False
        cur_list = _parse_running_symbols(row["bot_running_symbols"])
        cur_set = set(cur_list)
        if running:
            cur_set.add(symbol)
        else:
            cur_set.discard(symbol)
        # 안정 정렬 — 순서 보장으로 startup 재가동 흐름 결정적.
        new_arr = sorted(cur_set)
        new_json = json.dumps(new_arr, ensure_ascii=False)
        # 기존 컬럼 (bot_running) 도 호환 위해 동기화 — list 비어있으면 0,
        # 1개 이상이면 1. 옛 코드 경로 (단일 사용자 흐름 등) 가 그 값 읽어도 OK.
        legacy_flag = 1 if new_arr else 0
        conn.execute(
            """
            UPDATE users
            SET bot_running_symbols = ?, bot_running = ?, updated_at = ?
            WHERE code = ?
            """,
            (new_json, legacy_flag, now, code),
        )
        conn.commit()
        return True


def list_running_bots(db_path: Path | str) -> list[tuple[str, str]]:
    """가동 중 (user_code, symbol) 쌍 목록 — startup 자동 재가동용.

    bot_running_symbols JSON 배열 파싱 후 expand.

    Args:
        db_path: users.db 경로.

    Returns:
        list[(user_code, symbol)] — code 오름차순 + symbol 오름차순.
    """
    out: list[tuple[str, str]] = []
    with _connect(db_path) as conn:
        rows = conn.execute(
            "SELECT code, bot_running_symbols FROM users "
            "WHERE bot_running_symbols IS NOT NULL AND bot_running_symbols != ''",
        ).fetchall()
    for row in rows:
        code = str(row["code"])
        for sym in _parse_running_symbols(row["bot_running_symbols"]):
            out.append((code, sym))
    out.sort()
    return out


def set_last_run_mode(db_path: Path | str, code: str, mode: str) -> bool:
    """사용자 마지막 run_mode 영속화 — 2026-05-29 hot-fix.

    Why: fly machine 재시작 시 auto_resume 가 base_settings 의 default
    run_mode (DEMO) 로 재가동 시도. LIVE 사용자도 DEMO 로 가동 시도해서
    DEMO 키 미등록 시 봇 죽음. 이 컬럼에 마지막 가동 모드 박아 정확히 복원.

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        mode: ``"demo"`` 또는 ``"live"``. 그 외 값은 ValueError.

    Returns:
        True 면 UPDATE 1건 성공, False 면 code 미존재.

    Raises:
        ValueError: ``mode`` 가 허용 외 값.
    """
    if mode not in ("demo", "live"):
        raise ValueError(f"mode 는 'demo' 또는 'live' 여야 합니다: {mode!r}")
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE users
            SET last_run_mode = ?, updated_at = ?
            WHERE code = ?
            """,
            (mode, now, code),
        )
        conn.commit()
        return cur.rowcount > 0


def get_last_run_mode(db_path: Path | str, code: str) -> str | None:
    """사용자 마지막 run_mode 조회 — 2026-05-29 hot-fix.

    Returns:
        ``"demo"`` / ``"live"`` / None (미설정 또는 사용자 미존재).
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_run_mode FROM users WHERE code = ?",
            (code,),
        ).fetchone()
    if row is None:
        return None
    val = row["last_run_mode"]
    if val in ("demo", "live"):
        return str(val)
    return None


def list_running_codes(db_path: Path | str) -> list[str]:
    """[Deprecated 2026-05-29 PR B] 기존 boolean 흐름 호환용.

    내부적으로 ``list_running_bots`` 를 호출해 unique user_code 만 반환.
    신규 코드는 ``list_running_bots`` 직접 사용 권장.
    """
    bots = list_running_bots(db_path)
    seen: set[str] = set()
    out = []
    for code, _ in bots:
        if code not in seen:
            seen.add(code)
            out.append(code)
    return out


# ============================================================
# 세션 영속화 — Fly.io 재배포 대비 (2026-05-28)
# ============================================================


# 기본 세션 TTL — pin 모듈 / router 와 동일 (30일). 호출자가 명시 override 가능.
_DEFAULT_SESSION_TTL_SEC = 30 * 24 * 3600


def _iso_from_epoch(epoch_sec: float) -> str:
    """epoch 초 → ISO 8601 UTC (Z 접미사).

    Args:
        epoch_sec: ``time.time()`` 결과 형식의 초 (float 허용).

    Returns:
        예: ``"2026-05-28T12:34:56Z"`` — microsecond 절사.
    """
    return (
        datetime.fromtimestamp(epoch_sec, tz=UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def create_session_row(
    db_path: Path | str,
    token: str,
    user_code: str,
    ttl_sec: int = _DEFAULT_SESSION_TTL_SEC,
) -> bool:
    """세션 1건 영속화 — Fly.io 재배포 후에도 토큰 유효성 유지.

    Args:
        db_path: users.db 경로 (sessions 테이블 포함).
        token: ``secrets.token_urlsafe(32)`` 결과 (UNIQUE 가정).
        user_code: 토큰이 귀속될 사용자 라이선스 코드. 비어있으면 ValueError.
        ttl_sec: 만료까지 초 (기본 30일). 음수면 ValueError.

    Returns:
        True 면 INSERT 성공, False 면 PK 충돌 (실제로는 거의 발생 X — 32바이트 무작위).

    Raises:
        ValueError: ``token`` / ``user_code`` 가 빈 문자열 또는 ``ttl_sec`` 가 양수가 아님.
    """
    # 방어적 가드 — 빈 토큰 / 빈 user_code 는 호출 측 실수 가능성 높음.
    if not token or not isinstance(token, str):
        raise ValueError("token 은 비어있을 수 없습니다.")
    if not user_code or not isinstance(user_code, str):
        raise ValueError("user_code 는 비어있을 수 없습니다.")
    if not isinstance(ttl_sec, int) or ttl_sec <= 0:
        raise ValueError("ttl_sec 은 양의 정수여야 합니다.")

    now_sec = time.time()
    issued = _iso_from_epoch(now_sec)
    expires = _iso_from_epoch(now_sec + ttl_sec)
    try:
        with _connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (token, user_code, issued_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (token, user_code, issued, expires),
            )
            conn.commit()
            return True
    except sqlite3.IntegrityError:
        # token PK 충돌 — token_urlsafe(32) 가 우연히 겹치는 천문학적 확률.
        return False


def get_session(db_path: Path | str, token: str) -> dict[str, Any] | None:
    """토큰으로 세션 조회 — 만료된 row 는 자동 제외 (반환 None).

    만료된 row 를 호출 시점에 DELETE 하지는 않음 — cleanup_expired_sessions 가 일괄
    정리. 조회 핫패스에서 쓰기 트랜잭션 일으키지 않기 위함.

    Args:
        db_path: users.db 경로.
        token: 조회할 세션 토큰.

    Returns:
        ``{"token", "user_code", "issued_at", "expires_at"}`` dict, 또는 미존재/만료 시 None.
    """
    if not token or not isinstance(token, str):
        return None
    now_iso = _utcnow_iso()
    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT token, user_code, issued_at, expires_at
            FROM sessions
            WHERE token = ? AND expires_at > ?
            """,
            (token, now_iso),
        ).fetchone()
        if row is None:
            return None
        return dict(row)


def delete_session(db_path: Path | str, token: str) -> bool:
    """단일 세션 삭제 — 로그아웃 시 호출. 미존재 토큰도 False 반환 (예외 X).

    Args:
        db_path: users.db 경로.
        token: 삭제할 세션 토큰.

    Returns:
        True 면 1건 삭제, False 면 미존재 (멱등 보장).
    """
    if not token or not isinstance(token, str):
        return False
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM sessions WHERE token = ?", (token,))
        conn.commit()
        return cur.rowcount > 0


def delete_sessions_for_user(db_path: Path | str, user_code: str) -> int:
    """특정 사용자의 모든 세션 삭제 — PIN 변경 / 계정 잠금 시 호출.

    Args:
        db_path: users.db 경로.
        user_code: 대상 사용자 라이선스 코드. 빈 문자열이면 0 반환 (안전).

    Returns:
        삭제된 세션 개수 (0 이상).
    """
    if not user_code or not isinstance(user_code, str):
        return 0
    with _connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM sessions WHERE user_code = ?",
            (user_code,),
        )
        conn.commit()
        return int(cur.rowcount)


def delete_all_sessions(db_path: Path | str) -> int:
    """sessions 테이블 전체 비우기 — 보안 사고 / 마스터 키 회전 시.

    Args:
        db_path: users.db 경로.

    Returns:
        삭제된 세션 개수.
    """
    with _connect(db_path) as conn:
        cur = conn.execute("DELETE FROM sessions")
        conn.commit()
        return int(cur.rowcount)


def cleanup_expired_sessions(db_path: Path | str) -> int:
    """만료된 세션 row 일괄 제거 — 서버 startup 시 1회 호출 권장.

    idx_sessions_expires 인덱스가 있어 row 수가 많아도 빠름.

    Args:
        db_path: users.db 경로.

    Returns:
        제거된 세션 개수.
    """
    now_iso = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            "DELETE FROM sessions WHERE expires_at <= ?",
            (now_iso,),
        )
        conn.commit()
        return int(cur.rowcount)


__all__ = [
    "cleanup_expired_sessions",
    "create_session_row",
    "create_user",
    "delete_all_sessions",
    "delete_session",
    "delete_sessions_for_user",
    "get_api_keys",
    "get_session",
    "get_user_by_code",
    "has_api_keys",
    "init_db",
    "list_running_codes",
    "set_api_keys",
    "set_bot_running",
    "get_bot_running_symbols",
    "get_last_run_mode",
    "list_running_bots",
    "set_last_run_mode",
    "set_pin",
    "update_last_login",
    "update_license",
]
