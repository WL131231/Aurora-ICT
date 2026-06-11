"""#DB-RESILIENCE — users.db 손상 격리 + 일일 백업 (2026-06-11 장애 재발 방지).

장애: 머신 강제종료 → users.db SQLite 헤더 손상 → init_db 크래시 →
fly 재시작 무한루프로 전면 다운. 이제 손상 파일은 격리하고 부팅은 계속.
"""
from __future__ import annotations

import sqlite3

from aurora_ict.auth import users_db


def test_init_db_quarantines_corrupt_file(tmp_path) -> None:
    """깨진 파일 → .corrupt-<ts> 로 격리 + 새 DB 생성 (크래시 X)."""
    db = tmp_path / "users.db"
    db.write_bytes(b"this is not a sqlite database at all" * 10)
    users_db.init_db(db)  # 크래시 없이 통과해야 함
    # 격리 파일 존재 + 원본 내용 보존
    quarantined = list(tmp_path.glob("users.db.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes().startswith(b"this is not")
    # 새 DB 는 정상 동작
    users_db.create_user(db, "AICT-RESI-RESI-RESI")
    assert users_db.get_user_by_code(db, "AICT-RESI-RESI-RESI") is not None


def test_init_db_normal_path_unaffected(tmp_path) -> None:
    """정상 DB 는 기존과 동일 — 격리 파일 안 생김, idempotent."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    users_db.create_user(db, "AICT-NORM-NORM-NORM")
    users_db.init_db(db)  # 재호출 no-op
    assert users_db.get_user_by_code(db, "AICT-NORM-NORM-NORM") is not None
    assert list(tmp_path.glob("*.corrupt-*")) == []


def test_backup_db_creates_and_prunes(tmp_path) -> None:
    """백업 생성(같은 날 1회) + keep 개수 초과분 정리 + 복원 가능."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    users_db.create_user(db, "AICT-BKUP-BKUP-BKUP")
    bak = users_db.backup_db(db)
    assert bak is not None and bak.exists()
    assert users_db.backup_db(db) == bak  # 같은 날 재호출 — 동일 파일(skip)
    # 백업으로 실제 조회 가능 (일관 사본 검증)
    with sqlite3.connect(str(bak)) as conn:
        row = conn.execute(
            "SELECT code FROM users WHERE code='AICT-BKUP-BKUP-BKUP'",
        ).fetchone()
    assert row is not None
    # keep 정리 — 가짜 옛 백업 4개 만들고 keep=3 적용 시 오래된 것 삭제
    for d in ("20260101", "20260102", "20260103", "20260104"):
        (tmp_path / f"users.db.bak-{d}").write_bytes(b"old")
    users_db.backup_db(db, keep=3)
    remaining = sorted(p.name for p in tmp_path.glob("users.db.bak-*"))
    assert len(remaining) == 3
    assert remaining[-1].endswith(bak.name.split("-")[-1])  # 오늘자 보존


def test_backup_db_missing_source_returns_none(tmp_path) -> None:
    assert users_db.backup_db(tmp_path / "nope.db") is None
