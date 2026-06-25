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
import logging
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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


def _ensure_last_model_column(conn: sqlite3.Connection) -> None:
    """2026-06-25 #CURSUS: 사용자별 선택 봇 모델(Origo/Cursus) 영속화.

    모델 선택값을 저장해 봇 가동·재가동(auto_resume) 시 multi_user 가 어느 봇
    클래스(Origo=BotIctInstance / Cursus=BotTrendInstance)를 생성할지 분기한다.
    NULL 허용 — 구식 사용자/미선택은 DEFAULT_MODEL_NAME(Origo) fallback.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "last_model" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN last_model TEXT")


def _ensure_last_timeframe_column(conn: sqlite3.Connection) -> None:
    """2026-06-08: 사용자별 마지막 매매 timeframe 영속화.

    배경: 가동 중 매매 TF 변경 시 봇이 재시작되는데, 사용자별 timeframe 저장이
    없어 재시작이 전역 base_settings.timeframe(기본값)으로 회귀(1h 눌러도 15m
    로 돌아감). 이 컬럼으로 사용자가 마지막 설정한 TF 를 정확히 복원.

    마이그레이션: ``last_timeframe TEXT`` 컬럼 idempotent 추가. NULL 허용
    (구식 사용자는 base_settings.timeframe fallback).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "last_timeframe" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN last_timeframe TEXT")


def _ensure_pref_columns(conn: sqlite3.Connection) -> None:
    """2026-06-10: 사용자별 언어/시간대 영속화 — 텔레그램 알림 출력용.

    UI 언어/시간대는 브라우저 localStorage 에만 있어 서버(텔레그램 발송)가
    몰랐다. 이 컬럼으로 서버가 사용자 언어/시간대를 알아 알림을 해당 언어·
    시간대로 보낸다. NULL 허용(미설정 → ko / Asia/Seoul fallback).

    마이그레이션: ``pref_language`` / ``pref_timezone`` TEXT idempotent 추가.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "pref_language" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN pref_language TEXT")
    if "pref_timezone" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN pref_timezone TEXT")


def _ensure_telegram_chat_id_column(conn: sqlite3.Connection) -> None:
    """2026-06-08: 사용자별 텔레그램 chat_id 영속화 — 매매 알림 발송용.

    사용자가 텔레그램 봇에 자기 라이선스 코드를 입력하면 그 chat_id 를 코드에
    연결 저장. 매매 이벤트 시 이 chat_id 로 알림 발송. NULL 허용(미연동).

    마이그레이션: ``telegram_chat_id TEXT`` 컬럼 idempotent 추가.
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "telegram_chat_id" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN telegram_chat_id TEXT")


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


def _ensure_last_active_pairs_column(conn: sqlite3.Connection) -> None:
    """2026-06-06: 봇 STOP→START 시 페어 선택 기억 — last_active_pairs JSON 배열.

    bot_running_symbols(현재 가동 중)와 별개로, 사용자가 마지막으로 선택한 페어
    집합. 전체 STOP 시에도 유지돼 START 시 복원된다(페어 칩으로 끄면 제거).

    마이그레이션: 컬럼 idempotent 추가 + 기존 bot_running_symbols 를 초기값으로
    복사(이미 가동 중인 사용자의 선호 보존).
    """
    cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
    if "last_active_pairs" not in cols:
        conn.execute("ALTER TABLE users ADD COLUMN last_active_pairs TEXT")
    conn.execute(
        """
        UPDATE users
        SET last_active_pairs = bot_running_symbols
        WHERE (last_active_pairs IS NULL OR last_active_pairs = '')
          AND bot_running_symbols IS NOT NULL AND bot_running_symbols != ''
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

    2026-06-12 #DB-RESILIENCE: 파일이 깨져 있으면(file is not a database —
    강제종료로 SQLite 헤더 손상) 부팅 크래시 무한루프 대신, 깨진 파일을
    ``users.db.corrupt-<ts>`` 로 보존해 두고 새 DB 로 부팅을 계속한다.
    (2026-06-11 장애: 머신 강제종료 → users.db 손상 → init_db 크래시 →
    fly 재시작 루프로 2시간+ 전면 다운. 데이터는 볼륨 스냅샷으로 복구.)

    Args:
        db_path: 생성/오픈할 SQLite 파일 경로. 부모 디렉토리는 미리 존재해야 함.

    Raises:
        sqlite3.OperationalError: 디스크 권한 없음 / 경로 문제.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        _init_db_inner(path)
    except sqlite3.DatabaseError as e:
        if "not a database" not in str(e).lower():
            raise
        ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        quarantine = path.with_name(f"{path.name}.corrupt-{ts}")
        path.rename(quarantine)
        logger.critical(
            "users.db 손상 감지 — %s 로 격리 후 새 DB 생성. "
            "볼륨 스냅샷에서 복구 검토 필요! (%s)", quarantine.name, e,
        )
        _init_db_inner(path)


def backup_db(db_path: Path | str, keep: int = 3) -> Path | None:
    """users.db 일일 백업 — SQLite backup API 로 일관된 사본 생성.

    2026-06-12 #DB-RESILIENCE: 손상 사고 대비 애플리케이션 레벨 백업
    (fly 볼륨 스냅샷의 보조). 같은 날짜 백업이 이미 있으면 skip,
    오래된 백업은 ``keep`` 개만 남기고 정리.

    Args:
        db_path: 원본 users.db 경로.
        keep: 보관할 백업 개수 (기본 3 = 최근 3일).

    Returns:
        생성된(또는 기존) 백업 경로. 원본 없음/실패 시 None.
    """
    path = Path(db_path)
    if not path.exists():
        return None
    day = datetime.now(UTC).strftime("%Y%m%d")
    bak = path.with_name(f"{path.name}.bak-{day}")
    try:
        if not bak.exists():
            with sqlite3.connect(str(path)) as src, sqlite3.connect(str(bak)) as dst:
                src.backup(dst)
            logger.info("users.db 백업 생성 — %s", bak.name)
        # 오래된 백업 정리 — 이름 정렬 = 날짜 정렬.
        baks = sorted(path.parent.glob(f"{path.name}.bak-*"))
        for old in baks[:-keep]:
            old.unlink(missing_ok=True)
        return bak
    except Exception as e:  # noqa: BLE001
        logger.warning("users.db 백업 실패(무시): %s", e)
        return None


def _init_db_inner(path: Path) -> None:
    """init_db 본체 — 테이블 생성 + 마이그레이션 (손상 격리 래퍼와 분리).

    실패 시에도 연결을 확실히 닫는다 — Windows 에선 열린 핸들이 있으면
    손상 파일 rename(격리)이 PermissionError 로 막히기 때문.
    """
    conn = _connect(path)
    try:
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
        # 2026-06-25 #CURSUS: 사용자별 선택 봇 모델(Origo/Cursus).
        _ensure_last_model_column(conn)
        # 2026-06-06: 봇 STOP→START 시 페어 선택 기억용 last_active_pairs.
        _ensure_last_active_pairs_column(conn)
        # 2026-06-08: 가동 중 매매 TF 변경이 재시작 시 회귀하는 버그 — last_timeframe.
        _ensure_last_timeframe_column(conn)
        # 2026-06-08: 매매 알림 텔레그램 연동 — 사용자별 chat_id.
        _ensure_telegram_chat_id_column(conn)
        _ensure_pref_columns(conn)
        conn.commit()
    finally:
        # Windows: 핸들 열려 있으면 손상 파일 rename(격리)이 막힘 — 확실히 닫기.
        conn.close()
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


def delete_api_keys(
    db_path: Path | str,
    code: str,
    mode: str,
) -> bool:
    """모드별 거래소 API 키 삭제 — 슬롯을 NULL 로 비움.

    2026-05-30 파트너 요청: cred-slot UI 에 '삭제' 버튼 — 사용자가 등록한
    DEMO/LIVE 키를 명시 제거. mode 별로 별도 처리, 다른 모드 키는 그대로.

    ``mode='demo'`` 는 legacy ``api_key`` / ``api_secret_enc`` 컬럼도 함께
    비움 (set_api_keys 와 같은 동기화 정책).

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        mode: ``"demo"`` 또는 ``"live"``. 그 외 값은 ValueError.

    Returns:
        True 면 업데이트 성공 (row 존재), False 면 code 미존재.

    Raises:
        ValueError: ``mode`` 가 허용 외 값.
    """
    if mode not in ("demo", "live"):
        raise ValueError(f"mode 는 'demo' 또는 'live' 여야 합니다: {mode!r}")
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        if mode == "demo":
            cur = conn.execute(
                """
                UPDATE users
                SET demo_api_key = NULL,
                    demo_api_secret_enc = NULL,
                    api_key = NULL,
                    api_secret_enc = NULL,
                    updated_at = ?
                WHERE code = ?
                """,
                (now, code),
            )
        else:
            cur = conn.execute(
                """
                UPDATE users
                SET live_api_key = NULL,
                    live_api_secret_enc = NULL,
                    updated_at = ?
                WHERE code = ?
                """,
                (now, code),
            )
        conn.commit()
        return cur.rowcount > 0


def set_license(
    db_path: Path | str,
    code: str,
    license_type: str,
    expires_at: str | None,
) -> dict[str, Any]:
    """라이선스 코드 발급/정정 — idempotent (create-or-update).

    2026-05-29 (라이선스 근본 fix): 라이선스 봇 (aurora-ict-license/admin_bot.py)
    이 ``/issue_sub 365`` 발급 시 SaaS 의 ``/admin/license/issue`` 호출 →
    이 함수가 users.db 에 pre-insert. 그러면 사용자가 SaaS UI 에 코드 입력 +
    PIN 설정해도 ``setup_pin`` 이 default ``referral`` 로 덮어쓰지 않고
    pre-insert 된 license_type 그대로 유지.

    Args:
        db_path: users.db 경로.
        code: 라이선스 코드 (``AICT-XXXX-XXXX-XXXX``).
        license_type: ``referral`` / ``sub_30d`` / ``sub_90d`` / ``sub_365d``.
        expires_at: ISO 8601 UTC 만료 (``referral`` 은 None 허용).

    Returns:
        ``{"created": bool, "code": str, "license_type": str, "expires_at": str|None}``.
        - created=True: 신규 row INSERT (사용자가 아직 setup_pin 안 함).
        - created=False: 기존 row UPDATE (정정 흐름).

    Raises:
        ValueError: ``license_type`` 이 허용 외 값.
    """
    allowed = {"referral", "sub_30d", "sub_90d", "sub_365d"}
    if license_type not in allowed:
        raise ValueError(
            f"license_type 허용 외 값: {license_type!r} (허용: {sorted(allowed)})",
        )
    existing = get_user_by_code(db_path, code)
    if existing is None:
        # 신규 — create_user 호출 (license_type/expires_at 박힘).
        create_user(db_path, code, license_type, expires_at)
        return {
            "created": True,
            "code": code,
            "license_type": license_type,
            "expires_at": expires_at,
        }
    # 기존 — update_license 호출 (정정).
    update_license(db_path, code, license_type, expires_at)
    return {
        "created": False,
        "code": code,
        "license_type": license_type,
        "expires_at": expires_at,
    }


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


def get_last_active_pairs(db_path: Path | str, code: str) -> list[str]:
    """사용자가 마지막으로 선택한 페어 집합 — 봇 STOP→START 복원용.

    bot_running_symbols(현재 가동 중)와 달리 전체 STOP 시에도 유지된다.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_active_pairs FROM users WHERE code = ?", (code,),
        ).fetchone()
    if row is None:
        return []
    return _parse_running_symbols(row["last_active_pairs"])


def set_last_active_pair(
    db_path: Path | str,
    code: str,
    symbol: str,
    active: bool,
) -> bool:
    """선호 페어 집합에 symbol 추가(active=True)/제거(False).

    페어 가동 시 add, 페어 칩으로 끌 때 remove. 전체 STOP 은 이 값을 안 건드려
    START 시 복원된다.

    Returns:
        True 면 UPDATE 성공, False 면 code 미존재(no-op).
    """
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_active_pairs FROM users WHERE code = ?", (code,),
        ).fetchone()
        if row is None:
            return False
        cur = set(_parse_running_symbols(row["last_active_pairs"]))
        if active:
            cur.add(symbol)
        else:
            cur.discard(symbol)
        conn.execute(
            "UPDATE users SET last_active_pairs = ?, updated_at = ? WHERE code = ?",
            (json.dumps(sorted(cur), ensure_ascii=False), now, code),
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


def set_last_model(db_path: Path | str, code: str, model: str) -> bool:
    """사용자 선택 봇 모델 영속화 (#CURSUS 2026-06-25).

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        model: 모델 표시명 (settings.AVAILABLE_MODELS 의 key, 예 "Origo 1.2"/"Cursus 1.0").

    Returns:
        True 면 UPDATE 1건 성공, False 면 code 미존재.
    """
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE users
            SET last_model = ?, updated_at = ?
            WHERE code = ?
            """,
            (model, now, code),
        )
        conn.commit()
        return cur.rowcount > 0


def get_last_model(db_path: Path | str, code: str) -> str | None:
    """사용자 선택 봇 모델 조회 — 미설정/미존재면 None (호출처 DEFAULT_MODEL_NAME fallback)."""
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_model FROM users WHERE code = ?",
            (code,),
        ).fetchone()
    return row[0] if row and row[0] else None


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


def set_last_timeframe(db_path: Path | str, code: str, timeframe: str) -> bool:
    """사용자 마지막 매매 timeframe 영속화 — 2026-06-08.

    Why: 가동 중 매매 TF 변경 시 봇 재시작이 전역 base_settings(기본값)로
    회귀하는 버그(1h 눌러도 15m). 이 컬럼에 마지막 TF 박아 재시작·재가동 시
    정확히 복원. TRADE_TIMEFRAMES 검증은 호출처(라우터)에서.

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        timeframe: 매매 timeframe 문자열.

    Returns:
        True 면 UPDATE 1건 성공, False 면 code 미존재.
    """
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE users
            SET last_timeframe = ?, updated_at = ?
            WHERE code = ?
            """,
            (timeframe, now, code),
        )
        conn.commit()
        return cur.rowcount > 0


def get_last_timeframe(db_path: Path | str, code: str) -> str | None:
    """사용자 마지막 매매 timeframe 조회 — 2026-06-08.

    Returns:
        timeframe 문자열 또는 None (미설정 또는 사용자 미존재).
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT last_timeframe FROM users WHERE code = ?",
            (code,),
        ).fetchone()
    if row is None:
        return None
    val = row["last_timeframe"]
    return str(val) if val else None


def set_telegram_chat_id(db_path: Path | str, code: str, chat_id: str) -> bool:
    """사용자 텔레그램 chat_id 연결 — 매매 알림 발송 대상. 2026-06-08.

    Args:
        db_path: users.db 경로.
        code: 대상 사용자 라이선스 코드.
        chat_id: 텔레그램 chat_id (숫자지만 문자열로 저장).

    Returns:
        True 면 UPDATE 1건 성공, False 면 code 미존재.
    """
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE users
            SET telegram_chat_id = ?, updated_at = ?
            WHERE code = ?
            """,
            (str(chat_id), now, code),
        )
        conn.commit()
        return cur.rowcount > 0


def clear_telegram_chat_id_by_chat(db_path: Path | str, chat_id: str) -> int:
    """이 chat_id 에 연동된 모든 코드의 연동 해제 — '코드 재등록' 용. 2026-06-12.

    재등록 버튼을 누른 채팅이 새 코드를 보내면, 기존에 이 채팅에 묶여 있던
    코드들을 먼저 풀어 알림이 두 코드로 중복 발송되는 것을 막는다.

    Args:
        db_path: users.db 경로.
        chat_id: 텔레그램 chat_id (문자열).

    Returns:
        해제된 행 수.
    """
    now = _utcnow_iso()
    with _connect(db_path) as conn:
        cur = conn.execute(
            """
            UPDATE users
            SET telegram_chat_id = NULL, updated_at = ?
            WHERE telegram_chat_id = ?
            """,
            (now, str(chat_id)),
        )
        conn.commit()
        return cur.rowcount


def get_telegram_chat_id(db_path: Path | str, code: str) -> str | None:
    """사용자 텔레그램 chat_id 조회 — 매매 알림 발송용. 2026-06-08.

    Returns:
        chat_id 문자열 또는 None (미연동 또는 사용자 미존재).
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT telegram_chat_id FROM users WHERE code = ?",
            (code,),
        ).fetchone()
    if row is None:
        return None
    val = row["telegram_chat_id"]
    return str(val) if val else None


def set_user_prefs(
    db_path: Path | str,
    code: str,
    language: str | None = None,
    timezone: str | None = None,
) -> bool:
    """사용자 언어/시간대 저장 — 텔레그램 알림 출력용. 2026-06-10.

    None 인 인자는 갱신하지 않음(부분 업데이트). 둘 다 None 이면 no-op True.

    Returns:
        True 면 UPDATE 성공(또는 갱신할 값 없음), False 면 code 미존재.
    """
    sets: list[str] = []
    params: list[Any] = []
    if language is not None:
        sets.append("pref_language = ?")
        params.append(language)
    if timezone is not None:
        sets.append("pref_timezone = ?")
        params.append(timezone)
    if not sets:
        return True
    sets.append("updated_at = ?")
    params.append(_utcnow_iso())
    params.append(code)
    with _connect(db_path) as conn:
        cur = conn.execute(
            f"UPDATE users SET {', '.join(sets)} WHERE code = ?",  # noqa: S608
            params,
        )
        conn.commit()
        return cur.rowcount > 0


def get_user_prefs(db_path: Path | str, code: str) -> dict[str, str | None]:
    """사용자 언어/시간대 조회 — 미설정/미존재면 값 None. 2026-06-10.

    Returns:
        ``{"language": str|None, "timezone": str|None}``.
    """
    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT pref_language, pref_timezone FROM users WHERE code = ?",
            (code,),
        ).fetchone()
    if row is None:
        return {"language": None, "timezone": None}
    return {
        "language": row["pref_language"] or None,
        "timezone": row["pref_timezone"] or None,
    }


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
    "set_last_timeframe",
    "get_last_timeframe",
    "set_telegram_chat_id",
    "get_telegram_chat_id",
    "set_user_prefs",
    "get_user_prefs",
    "delete_api_keys",
    "set_license",
    "set_pin",
    "update_last_login",
    "update_license",
]
