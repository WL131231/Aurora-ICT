"""공지사항 SQLite 저장소 — 모든 사용자에게 표시되는 운영 공지.

users.db 와 같은 파일을 공유하지만 별도 테이블 (notices) 로 격리.
관리자 (Telegram admin bot 또는 직접 API) 가 공지 등록 → SaaS UI 우상단 banner.

스키마:
    CREATE TABLE notices (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message TEXT NOT NULL,
        severity TEXT NOT NULL DEFAULT 'info',  -- info / warn / critical
        expires_at TEXT,                          -- ISO 8601 UTC, NULL = 영구
        created_at TEXT NOT NULL                  -- ISO 8601 UTC + Z
    )
    CREATE INDEX idx_notices_expires ON notices(expires_at)

활성 공지 결정 — `expires_at IS NULL OR expires_at > now()` 중 가장 최근 (id DESC) 1개.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _iso_now() -> str:
    """현재 시각 ISO 8601 UTC + Z 접미사 — DB 비교 일관성."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


_DDL_NOTICES = """
CREATE TABLE IF NOT EXISTS notices (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message TEXT NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info',
    expires_at TEXT,
    created_at TEXT NOT NULL
)
"""

_DDL_INDEX_EXPIRES = (
    "CREATE INDEX IF NOT EXISTS idx_notices_expires ON notices(expires_at)"
)


def init_notices_table(db_path: Path) -> None:
    """notices 테이블 + 인덱스 idempotent 생성. users_db.init_db 가 함께 호출."""
    with sqlite3.connect(db_path) as conn:
        conn.execute(_DDL_NOTICES)
        conn.execute(_DDL_INDEX_EXPIRES)
        conn.commit()


def create_notice(
    db_path: Path,
    message: str,
    severity: str = "info",
    expires_at: str | None = None,
) -> int:
    """공지 1건 등록 → 새로 생성된 id 반환.

    Args:
        db_path: SQLite 파일 경로.
        message: 공지 본문 (필수, 빈 문자열 금지).
        severity: info / warn / critical 중 하나. 그 외 값은 ValueError.
        expires_at: ISO 8601 UTC 만료 시각. None 이면 영구.

    Returns:
        새 공지의 id (INTEGER PK).

    Raises:
        ValueError: message 가 빈 문자열 또는 severity 가 허용 외 값.
    """
    if not message or not message.strip():
        raise ValueError("message 가 비어있습니다.")
    if severity not in ("info", "warn", "critical"):
        raise ValueError(f"severity 허용 값 아님: {severity!r}")
    now = _iso_now()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO notices(message, severity, expires_at, created_at) "
            "VALUES (?, ?, ?, ?)",
            (message.strip(), severity, expires_at, now),
        )
        conn.commit()
        return int(cur.lastrowid or 0)


def get_active_notice(db_path: Path) -> dict[str, Any] | None:
    """현재 활성 공지 1건 — 만료 안 된 것 중 가장 최근 (id DESC).

    Returns:
        dict (id, message, severity, expires_at, created_at) 또는 None.
    """
    now = _iso_now()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        # expires_at IS NULL (영구) OR expires_at > now (미만료)
        cur = conn.execute(
            "SELECT id, message, severity, expires_at, created_at "
            "FROM notices "
            "WHERE expires_at IS NULL OR expires_at > ? "
            "ORDER BY id DESC LIMIT 1",
            (now,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)


def delete_notice(db_path: Path, notice_id: int) -> bool:
    """공지 삭제. 멱등 — 없는 id 면 False.

    Returns:
        삭제 성공 여부.
    """
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute("DELETE FROM notices WHERE id = ?", (notice_id,))
        conn.commit()
        return cur.rowcount > 0


def list_notices(
    db_path: Path,
    include_expired: bool = False,
) -> list[dict[str, Any]]:
    """공지 목록 — admin 조회용 (만료 포함 여부 선택).

    Args:
        include_expired: True 면 만료된 공지도 포함.

    Returns:
        list of dict, id DESC 순.
    """
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if include_expired:
            cur = conn.execute(
                "SELECT id, message, severity, expires_at, created_at "
                "FROM notices ORDER BY id DESC",
            )
        else:
            now = _iso_now()
            cur = conn.execute(
                "SELECT id, message, severity, expires_at, created_at "
                "FROM notices "
                "WHERE expires_at IS NULL OR expires_at > ? "
                "ORDER BY id DESC",
                (now,),
            )
        return [dict(r) for r in cur.fetchall()]


__all__ = [
    "create_notice",
    "delete_notice",
    "get_active_notice",
    "init_notices_table",
    "list_notices",
]
