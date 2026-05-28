"""API secret 암호화 키스토어 — Fernet (AES-128-CBC + HMAC-SHA256) 기반.

파트너 결정 2026-05-28 — 다중 사용자 SaaS 전환을 위한 거래소 secret 보호 모듈.

설계:
    - ``cryptography.fernet.Fernet`` — 검증된 대칭 암호화 (AEAD 유사 보장)
    - 마스터 키 32 바이트 (base64 url-safe 인코딩 시 44 글자)
    - 한 봇 인스턴스 = 한 마스터 키 → 그 봇이 저장한 모든 사용자 secret 복호화

마스터 키 소스 우선순위:
    1. 환경변수 ``AURORA_ICT_MASTER_KEY`` — 운영/CI 권장 (secrets manager 연동 가능)
    2. ``<data_dir>/master.key`` 파일 — 첫 실행 시 자동 생성 (개발/단독 호스팅)
    3. (없으면 2번 경로에 신규 발급해서 저장)

중요 — 데이터 디렉토리 이전:
    ``users.db`` 만 옮기고 ``master.key`` 를 안 옮기면 모든 secret 복호화 실패.
    반드시 두 파일 함께 이전 (또는 환경변수로 같은 키 주입). 운영 가이드에 명시 필요.

파일 권한:
    POSIX 에서 ``master.key`` 는 0o600 (소유자만 read/write).
    Windows 는 NTFS ACL 이 별도라 chmod 무시되지만, 사용자 디렉토리 (LOCALAPPDATA)
    자체가 이미 다른 사용자 접근 차단되어 기본 충분.

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

import base64
import os
import stat
from pathlib import Path

from cryptography.fernet import Fernet, InvalidToken

from aurora_ict.paths import data_dir

_ENV_VAR = "AURORA_ICT_MASTER_KEY"
_KEY_FILENAME = "master.key"

# Fernet 키 길이 — 32 바이트 (raw) / base64 url-safe 인코딩 시 44 글자.
_FERNET_KEY_BYTES = 32


def _key_file_path() -> Path:
    """마스터 키 파일 절대 경로 — ``<data_dir>/master.key``.

    Returns:
        Path. 존재 여부는 보장 X.
    """
    return data_dir() / _KEY_FILENAME


def _generate_and_save_key(path: Path) -> bytes:
    """새 Fernet 키 생성 → 파일에 저장 → bytes 반환.

    Args:
        path: 저장 경로 (부모 디렉토리 자동 생성).

    Returns:
        Fernet 키 (base64 url-safe 인코딩된 bytes, 44 글자).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    key = Fernet.generate_key()
    path.write_bytes(key)
    # POSIX 0o600 — 소유자만 read/write. Windows 에선 NTFS ACL 우선이라 no-op 에 가까움.
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    except OSError:
        # Windows 일부 환경에서 chmod 가 무시/실패해도 동작에 영향 X.
        pass
    return key


def get_master_key() -> bytes:
    """마스터 키 획득 — env 우선, 없으면 파일 (없으면 생성).

    Returns:
        Fernet 호환 키 bytes (base64 url-safe 44 글자).

    Raises:
        ValueError: 환경변수가 설정되었으나 형식 (길이/base64) 이 잘못된 경우.
    """
    env_val = os.environ.get(_ENV_VAR)
    if env_val:
        # 환경변수 — 형식 검증 (잘못된 키로 운영하면 모든 secret 영구 복호화 불가).
        env_bytes = env_val.encode("ascii")
        try:
            raw = base64.urlsafe_b64decode(env_bytes)
        except (ValueError, base64.binascii.Error) as e:
            raise ValueError(
                f"{_ENV_VAR} 가 유효한 base64 url-safe 가 아닙니다.",
            ) from e
        if len(raw) != _FERNET_KEY_BYTES:
            raise ValueError(
                f"{_ENV_VAR} 디코드 길이가 {_FERNET_KEY_BYTES}바이트가 아닙니다 "
                f"(현재 {len(raw)}바이트).",
            )
        return env_bytes

    path = _key_file_path()
    if path.exists():
        return path.read_bytes().strip()
    return _generate_and_save_key(path)


def _fernet(key: bytes | None = None) -> Fernet:
    """Fernet 인스턴스 생성 — 키 미지정 시 ``get_master_key()`` 사용.

    Args:
        key: 명시적 키 (테스트용). None 이면 마스터 키 사용.

    Returns:
        Fernet 인스턴스.
    """
    return Fernet(key if key is not None else get_master_key())


def encrypt_secret(plaintext: str, key: bytes | None = None) -> str:
    """문자열 평문 → Fernet 암호문 (base64 url-safe 문자열).

    Args:
        plaintext: 암호화할 평문 (거래소 API secret 등).
        key: 명시적 키 (테스트/회전용). None 이면 마스터 키.

    Returns:
        ASCII 문자열 (DB 저장 가능). Fernet 토큰은 ``gAAAA...`` 로 시작.

    Raises:
        TypeError: ``plaintext`` 가 str 이 아닌 경우.
    """
    if not isinstance(plaintext, str):
        raise TypeError("plaintext 는 str 이어야 합니다.")
    token = _fernet(key).encrypt(plaintext.encode("utf-8"))
    return token.decode("ascii")


def decrypt_secret(ciphertext: str, key: bytes | None = None) -> str:
    """Fernet 암호문 → 평문 문자열.

    Args:
        ciphertext: ``encrypt_secret`` 결과 또는 동일 형식의 Fernet 토큰.
        key: 명시적 키 (테스트/회전용). None 이면 마스터 키.

    Returns:
        평문 문자열 (UTF-8 디코드).

    Raises:
        InvalidToken: 키 불일치 / 토큰 손상 / 형식 오류.
        TypeError: ``ciphertext`` 가 str 이 아닌 경우.
    """
    if not isinstance(ciphertext, str):
        raise TypeError("ciphertext 는 str 이어야 합니다.")
    raw = _fernet(key).decrypt(ciphertext.encode("ascii"))
    return raw.decode("utf-8")


__all__ = [
    "InvalidToken",
    "decrypt_secret",
    "encrypt_secret",
    "get_master_key",
]
