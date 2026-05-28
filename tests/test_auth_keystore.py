"""keystore 모듈 — 실제 Fernet 사용, mock 0.

검증:
    - encrypt/decrypt round-trip (한글/긴 문자열/특수문자 포함)
    - 다른 키로 복호화 시 InvalidToken
    - env 우선순위 (env 설정 시 파일 무시)
    - 파일 fallback (env 없을 때 자동 생성)
    - env 형식 검증 (잘못된 길이/base64 → ValueError)
    - 평문 손상 시 InvalidToken
    - DB 라운드트립 (encrypt → DB 저장 → 조회 → decrypt) 시나리오

격리:
    - ``AURORA_ICT_DATA_DIR`` env 를 tmp_path 로 설정 → 사용자 PC 영향 X
    - ``AURORA_ICT_MASTER_KEY`` env 는 테스트별로 monkeypatch 로 set/unset

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

import base64

import pytest
from cryptography.fernet import Fernet, InvalidToken

from aurora_ict.auth import keystore, users_db


@pytest.fixture(autouse=True)
def _isolate_data_dir(monkeypatch, tmp_path):
    """모든 keystore 테스트 — data_dir 을 tmp_path 로 redirect.

    Why: 사용자 PC LOCALAPPDATA/Aurora/master.key 절대 안 건드림.
    """
    monkeypatch.setenv("AURORA_ICT_DATA_DIR", str(tmp_path))
    # 마스터 키 env 도 기본 unset — 각 테스트가 명시적으로 setenv 함.
    monkeypatch.delenv("AURORA_ICT_MASTER_KEY", raising=False)


def test_round_trip_default_key(tmp_path):
    """기본 마스터 키 (파일 자동 생성) 로 암호화/복호화 라운드트립."""
    plaintext = "binance_secret_abc_XYZ_123!@#"
    ct = keystore.encrypt_secret(plaintext)

    assert isinstance(ct, str)
    assert ct != plaintext
    # Fernet 토큰은 'gAAAA' 로 시작
    assert ct.startswith("gAAAA")

    # 같은 프로세스 — 파일에 저장된 키 재사용해 복호화
    pt = keystore.decrypt_secret(ct)
    assert pt == plaintext

    # 마스터 키 파일 생성됐는지
    assert (tmp_path / "master.key").exists()


def test_round_trip_unicode_and_long():
    """한글/긴 문자열도 무결성 보장."""
    plaintext = "비밀키_" + "X" * 500 + "_끝!"
    ct = keystore.encrypt_secret(plaintext)
    pt = keystore.decrypt_secret(ct)
    assert pt == plaintext


def test_explicit_key_round_trip():
    """key 인자 명시 시 그 키로 암복호화 — 마스터 키 무관."""
    explicit = Fernet.generate_key()
    ct = keystore.encrypt_secret("hello", key=explicit)
    pt = keystore.decrypt_secret(ct, key=explicit)
    assert pt == "hello"


def test_decrypt_with_wrong_key_raises():
    """다른 키로 복호화 시도 시 InvalidToken."""
    key_a = Fernet.generate_key()
    key_b = Fernet.generate_key()
    ct = keystore.encrypt_secret("topsecret", key=key_a)

    with pytest.raises(InvalidToken):
        keystore.decrypt_secret(ct, key=key_b)


def test_env_key_takes_priority_over_file(tmp_path, monkeypatch):
    """env 가 설정되면 파일 master.key 는 무시되어야 함."""
    # 파일 키 (A) 와 env 키 (B) 가 다르면 — env 가 우선이라 파일 키로 복호화 시 InvalidToken.
    file_key = Fernet.generate_key()
    (tmp_path / "master.key").write_bytes(file_key)

    env_key = Fernet.generate_key()
    monkeypatch.setenv("AURORA_ICT_MASTER_KEY", env_key.decode("ascii"))

    # encrypt 는 env 키 사용
    ct = keystore.encrypt_secret("payload")
    # 같은 함수 호출 — 여전히 env 사용해 성공
    assert keystore.decrypt_secret(ct) == "payload"

    # 파일 키 (A) 로는 복호화 실패해야 (env 가 다른 키였음을 증명)
    with pytest.raises(InvalidToken):
        keystore.decrypt_secret(ct, key=file_key)


def test_file_fallback_when_no_env(tmp_path):
    """env 미설정 — 파일이 없으면 자동 생성, 있으면 재사용."""
    assert not (tmp_path / "master.key").exists()

    # 첫 호출 — 파일 자동 생성
    ct1 = keystore.encrypt_secret("v1")
    key_file = tmp_path / "master.key"
    assert key_file.exists()
    saved_key = key_file.read_bytes()

    # 두 번째 호출 — 같은 키로 복호화 가능
    pt1 = keystore.decrypt_secret(ct1)
    assert pt1 == "v1"

    # 파일 내용이 안 바뀌었는지 (재생성 X)
    assert key_file.read_bytes() == saved_key


def test_invalid_env_base64_raises(monkeypatch):
    """env 가 base64 디코드 불가 → ValueError."""
    monkeypatch.setenv("AURORA_ICT_MASTER_KEY", "!!!not-valid-base64!!!")
    with pytest.raises(ValueError):
        keystore.encrypt_secret("x")


def test_invalid_env_wrong_length_raises(monkeypatch):
    """env 가 base64 디코드 되지만 32바이트 아닐 때 → ValueError."""
    # 16바이트만 인코딩 — Fernet 요구 (32바이트) 불일치
    short = base64.urlsafe_b64encode(b"\x00" * 16).decode("ascii")
    monkeypatch.setenv("AURORA_ICT_MASTER_KEY", short)
    with pytest.raises(ValueError, match="32바이트"):
        keystore.encrypt_secret("x")


def test_get_master_key_env_path(monkeypatch):
    """get_master_key — env 설정 시 그 값을 그대로 반환."""
    env_key = Fernet.generate_key()
    monkeypatch.setenv("AURORA_ICT_MASTER_KEY", env_key.decode("ascii"))
    assert keystore.get_master_key() == env_key


def test_get_master_key_file_path(tmp_path):
    """get_master_key — env 없을 때 파일 생성/재사용."""
    k1 = keystore.get_master_key()
    k2 = keystore.get_master_key()
    assert k1 == k2  # 같은 키 재사용
    assert (tmp_path / "master.key").exists()


def test_corrupted_ciphertext_raises():
    """평문/잘못된 형식 입력 시 InvalidToken."""
    with pytest.raises(InvalidToken):
        keystore.decrypt_secret("not-a-real-token")


def test_encrypt_rejects_non_string():
    """plaintext 가 str 이 아니면 TypeError."""
    with pytest.raises(TypeError):
        keystore.encrypt_secret(b"bytes-not-str")  # type: ignore[arg-type]


def test_decrypt_rejects_non_string():
    """ciphertext 가 str 이 아니면 TypeError."""
    with pytest.raises(TypeError):
        keystore.decrypt_secret(b"bytes-not-str")  # type: ignore[arg-type]


def test_integration_with_users_db(tmp_path):
    """end-to-end — encrypt → users_db 저장 → 조회 → decrypt 라운드트립."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    users_db.create_user(db, "AICT-INTG-INTG-INTG")

    plaintext = "exchange_api_secret_super_sensitive"
    ct = keystore.encrypt_secret(plaintext)
    ok = users_db.set_api_keys(db, "AICT-INTG-INTG-INTG", "pub_key_xyz", ct)
    assert ok is True

    row = users_db.get_user_by_code(db, "AICT-INTG-INTG-INTG")
    assert row is not None
    # DB 엔 암호문 그대로 — 평문 노출 X
    assert row["api_secret_enc"] == ct
    assert plaintext not in row["api_secret_enc"]

    # 복호화 시 원본 일치
    recovered = keystore.decrypt_secret(row["api_secret_enc"])
    assert recovered == plaintext
