"""#SEC 2026-08-03 — PIN 해시 강도 + 하위 호환.

상용화 전 보안 점검에서 PBKDF2 반복이 100k(OWASP 2023 권장 600k)였다. 올리되
**기존 사용자가 로그인하지 못하면 안 되므로**, 검증이 저장된 해시의 iter 값을
쓰는지(형식에 포함) 확인한다.
"""

from __future__ import annotations

import base64
import hashlib

from aurora_ict.auth import pin


def test_new_hash_uses_600k() -> None:
    """새로 만드는 해시는 OWASP 권장 반복을 쓴다."""
    h = pin.hash_pin("TestPin!23")
    assert h.split("$")[1] == "600000"


def test_legacy_100k_hash_still_verifies() -> None:
    """★ 기존 사용자(100k 해시)가 계속 로그인된다 — 형식에 iter 가 들어 있다."""
    raw = "TestPin!23"
    salt = b"0123456789abcdef"
    derived = hashlib.pbkdf2_hmac("sha256", raw.encode(), salt, 100_000)
    legacy = "pbkdf2_sha256$100000$" + base64.b64encode(salt).decode() + "$" \
             + base64.b64encode(derived).decode()

    assert pin.verify_pin(raw, legacy) is True
    assert pin.verify_pin("WrongPin!23", legacy) is False


def test_roundtrip() -> None:
    """새 해시도 정상 검증된다."""
    h = pin.hash_pin("TestPin!23")
    assert pin.verify_pin("TestPin!23", h) is True
    assert pin.verify_pin("TestPin!24", h) is False


def test_salt_is_random() -> None:
    """같은 PIN 이라도 해시가 달라야 한다(레인보우 테이블 방어)."""
    assert pin.hash_pin("TestPin!23") != pin.hash_pin("TestPin!23")
