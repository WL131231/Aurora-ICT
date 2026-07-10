"""admin PIN 초기화 검증 — clear_pin + setup-pin 재설정 flow (2026-07-10).

PIN 은 pbkdf2 해시라 원문 복구 불가 — clear 후 setup-pin 재통과가 복구 경로.
"""

from __future__ import annotations

from pathlib import Path

from aurora_ict.auth import users_db
from aurora_ict.auth.pin import hash_pin


def _db(tmp_path: Path) -> Path:
    p = tmp_path / "users.db"
    users_db.init_db(p)
    return p


def test_clear_pin_resets_hash_and_allows_resetup(tmp_path: Path) -> None:
    db = _db(tmp_path)
    users_db.create_user(db, "AICT-TEST-0001-PINX", license_type="sub_30d")
    users_db.set_pin(db, "AICT-TEST-0001-PINX", hash_pin("Secret12!"))
    assert users_db.get_user_by_code(db, "AICT-TEST-0001-PINX")["pin_hash"]

    assert users_db.clear_pin(db, "AICT-TEST-0001-PINX") is True
    assert users_db.get_user_by_code(db, "AICT-TEST-0001-PINX")["pin_hash"] is None

    # 초기화 후 새 PIN 설정 가능 (setup-pin 경로의 전제)
    users_db.set_pin(db, "AICT-TEST-0001-PINX", hash_pin("NewPin34!"))
    assert users_db.get_user_by_code(db, "AICT-TEST-0001-PINX")["pin_hash"].startswith(
        "pbkdf2_sha256$",
    )


def test_clear_pin_unknown_code_returns_false(tmp_path: Path) -> None:
    db = _db(tmp_path)
    assert users_db.clear_pin(db, "AICT-NOPE-0000-0000") is False
