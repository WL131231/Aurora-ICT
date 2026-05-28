"""PIN 인증 — 라이선스 코드와 함께 사용하는 2-factor 단순 인증.

파트너 결정 2026-05-28 — 아이패드/모바일에서 봇 접속 위해 SaaS 인증 추가.
완전 SaaS 클라우드 호스팅 전 단계로, 로컬 봇 (LAN 노출) + PIN 보호.

PIN 정책:
    - 최소 8자리
    - 영문 (대/소문자) 포함
    - 숫자 포함
    - 특수문자 포함 (!@#$%^&* 등)

저장:
    - PIN 평문 X — PBKDF2-HMAC-SHA256 (100,000 iter) + 16바이트 salt
    - license.json 의 ``pin_hash`` 필드 (base64 인코딩 통합 문자열)
    - 형식: ``pbkdf2_sha256$<iter>$<base64_salt>$<base64_hash>``

세션 토큰:
    - secrets.token_urlsafe(32)
    - 봇 메모리 dict (재시작 시 사라짐 — 매번 다시 로그인)
    - 만료 30일 (default), 갱신 시 새 토큰 발급
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import string
import time
from dataclasses import dataclass

_ALGO = "pbkdf2_sha256"
_ITERATIONS = 100_000
_SALT_BYTES = 16

# PIN 정책 — 8자리+ 영/숫/특수 혼합
_PIN_MIN_LEN = 8
_SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;':\",./<>?`~"


@dataclass(slots=True)
class PinStrengthError:
    """PIN 강도 검증 실패 — UI 가 사용자에게 표시할 사유."""

    code: str       # "too_short" / "no_letter" / "no_digit" / "no_special"
    message: str    # 한국어 안내문


def validate_pin_strength(pin: str) -> PinStrengthError | None:
    """PIN 정책 검증 — 통과 시 None, 실패 시 PinStrengthError.

    Args:
        pin: 검증할 PIN 문자열.

    Returns:
        None 이면 통과. 실패 시 첫 위반 사항 1건 반환.
    """
    if not isinstance(pin, str):
        return PinStrengthError("invalid_type", "PIN은 문자열이어야 합니다.")
    if len(pin) < _PIN_MIN_LEN:
        return PinStrengthError(
            "too_short", f"PIN은 최소 {_PIN_MIN_LEN}자 이상이어야 합니다.",
        )
    has_letter = any(c in string.ascii_letters for c in pin)
    if not has_letter:
        return PinStrengthError(
            "no_letter", "영문 (대/소문자) 을 1자 이상 포함해야 합니다.",
        )
    has_digit = any(c in string.digits for c in pin)
    if not has_digit:
        return PinStrengthError(
            "no_digit", "숫자를 1자 이상 포함해야 합니다.",
        )
    has_special = any(c in _SPECIAL_CHARS for c in pin)
    if not has_special:
        return PinStrengthError(
            "no_special",
            f"특수문자 ({_SPECIAL_CHARS[:10]}…) 를 1자 이상 포함해야 합니다.",
        )
    return None


def hash_pin(pin: str) -> str:
    """PIN → 저장 가능한 해시 문자열.

    Returns:
        ``pbkdf2_sha256$<iter>$<base64_salt>$<base64_hash>`` 형식.
    """
    salt = secrets.token_bytes(_SALT_BYTES)
    derived = hashlib.pbkdf2_hmac(
        "sha256", pin.encode("utf-8"), salt, _ITERATIONS,
    )
    b64_salt = base64.b64encode(salt).decode("ascii")
    b64_hash = base64.b64encode(derived).decode("ascii")
    return f"{_ALGO}${_ITERATIONS}${b64_salt}${b64_hash}"


def verify_pin(pin: str, stored_hash: str) -> bool:
    """저장된 해시와 입력 PIN 비교 — timing-safe (hmac.compare_digest).

    Returns:
        True 면 일치, False 면 불일치 또는 형식 오류.
    """
    if not isinstance(pin, str) or not isinstance(stored_hash, str):
        return False
    parts = stored_hash.split("$")
    if len(parts) != 4 or parts[0] != _ALGO:
        return False
    try:
        iterations = int(parts[1])
        salt = base64.b64decode(parts[2])
        expected = base64.b64decode(parts[3])
    except (ValueError, base64.binascii.Error):
        return False
    derived = hashlib.pbkdf2_hmac(
        "sha256", pin.encode("utf-8"), salt, iterations,
    )
    return hmac.compare_digest(derived, expected)


# ============================================================
# 세션 토큰 — 봇 메모리 dict (단일 사용자 봇 가정)
# ============================================================


@dataclass(slots=True)
class _Session:
    """발급된 세션 1건.

    user_code:
        세션이 귀속된 사용자 라이선스 코드. SaaS 다중 사용자 전환 (2026-05-28)
        이후 토큰 → user_code 매핑 필수. 단일 사용자 호환을 위해 ""(빈 문자열)
        도 허용 (레거시 호출 경로 — middleware 가 user_code 미존재로 보고 거절).
    """

    token: str
    issued_at_ms: int
    expires_at_ms: int
    user_code: str = ""


# 봇 프로세스 메모리에만 존재 — 재시작 시 모든 세션 무효 (다시 로그인 필요).
_active_sessions: dict[str, _Session] = {}

_DEFAULT_TTL_SEC = 30 * 24 * 3600  # 30일 — cookie Max-Age 와 일치


def create_session(
    user_code: str = "",
    ttl_sec: int = _DEFAULT_TTL_SEC,
) -> str:
    """새 세션 토큰 발급 → 메모리 dict 에 저장 → 토큰 반환.

    Args:
        user_code: 세션이 귀속될 사용자 라이선스 코드. SaaS 흐름에서는 필수.
            레거시 (단일 사용자) 호출자는 빈 문자열로 호출 가능.
        ttl_sec: 만료까지 초 (기본 30일).

    Returns:
        토큰 문자열 (URL-safe base64, 32바이트).
    """
    token = secrets.token_urlsafe(32)
    now_ms = int(time.time() * 1000)
    _active_sessions[token] = _Session(
        token=token,
        issued_at_ms=now_ms,
        expires_at_ms=now_ms + ttl_sec * 1000,
        user_code=user_code,
    )
    return token


def validate_session(token: str | None) -> bool:
    """토큰 유효성 — 존재 + 만료 X.

    Args:
        token: Authorization 헤더 또는 cookie 의 토큰.

    Returns:
        True 면 유효 세션, False 면 무효/만료/미발급.
    """
    if not token or not isinstance(token, str):
        return False
    sess = _active_sessions.get(token)
    if sess is None:
        return False
    now_ms = int(time.time() * 1000)
    if now_ms >= sess.expires_at_ms:
        # 만료 — 메모리에서 제거
        _active_sessions.pop(token, None)
        return False
    return True


def get_user_from_session(token: str | None) -> str | None:
    """토큰 → 사용자 라이선스 코드 — 만료/미존재/user_code 부재 시 None.

    Args:
        token: 세션 토큰 (cookie 또는 Authorization 헤더).

    Returns:
        user_code 문자열 (비어있지 않을 때만), 그 외 None.
    """
    if not token or not isinstance(token, str):
        return None
    sess = _active_sessions.get(token)
    if sess is None:
        return None
    now_ms = int(time.time() * 1000)
    if now_ms >= sess.expires_at_ms:
        _active_sessions.pop(token, None)
        return None
    # user_code 가 빈 문자열인 레거시 세션은 SaaS 인증 컨텍스트에서 거절.
    return sess.user_code or None


def revoke_session(token: str | None) -> None:
    """로그아웃 — 토큰 즉시 무효화."""
    if token:
        _active_sessions.pop(token, None)


def revoke_all_sessions() -> int:
    """전체 세션 종료 — 보안 사고 시 또는 PIN 변경 시.

    Returns:
        제거된 세션 개수.
    """
    count = len(_active_sessions)
    _active_sessions.clear()
    return count


def revoke_sessions_for_user(user_code: str) -> int:
    """특정 사용자 세션 전부 무효화 — PIN 변경/계정 잠금 등.

    Args:
        user_code: 대상 사용자 라이선스 코드.

    Returns:
        무효화된 세션 개수.
    """
    to_remove = [
        tok for tok, sess in _active_sessions.items()
        if sess.user_code == user_code
    ]
    for tok in to_remove:
        _active_sessions.pop(tok, None)
    return len(to_remove)


__all__ = [
    "PinStrengthError",
    "create_session",
    "get_user_from_session",
    "hash_pin",
    "revoke_all_sessions",
    "revoke_session",
    "revoke_sessions_for_user",
    "validate_pin_strength",
    "validate_session",
    "verify_pin",
]
