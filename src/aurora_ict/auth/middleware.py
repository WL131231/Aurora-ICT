"""FastAPI 인증 미들웨어 — cookie 세션 토큰 → user_code 변환 dependency.

파트너 결정 2026-05-28 — SaaS 전환 시 인증 보호 endpoint 의 공통 진입점.

설계:
    - FastAPI ``Depends`` 호환 async callable.
    - cookie ``aurora_ict_session`` 에서 토큰 추출 → ``pin.get_user_from_session``
      으로 user_code 조회 → 미인증/만료/user_code 부재 시 401.
    - 평문 비밀 노출 X — 검증 실패 사유는 단순 메시지만 (사용자 enum 노출 방지).

세션 cookie 사양 (router.py 와 일치):
    - 이름: ``aurora_ict_session``
    - HttpOnly=True, SameSite=Lax, Max-Age=30일
    - Secure 는 HTTPS 환경에서만 True (개발 환경에선 False)

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from fastapi import HTTPException, Request, status

from aurora_ict.auth import users_db
from aurora_ict.auth.pin import get_session_db_path, get_user_from_session

logger = logging.getLogger(__name__)

# cookie 이름 — router.py 와 단일 source of truth (서로 import 해 일관 유지).
SESSION_COOKIE_NAME = "aurora_ict_session"


def is_license_expired(expires_at: str | None) -> bool:
    """라이선스 만료 여부 판정 (ISO 8601 UTC 문자열 기준).

    Args:
        expires_at: ``create_user`` 가 박은 만료 시각 (예: ``2026-12-31T23:59:59Z``).
            ``None``/빈 문자열은 referral(무기한) → 만료 아님.

    Returns:
        ``True`` = 만료됨. ``False`` = 유효 / 무기한.

    Why:
        파싱 불가한 값이면 ``False`` (차단 안 함) — 손상된 데이터로 정상 사용자를
        잠그는 것보다, 만료 미적용이 안전한 기본값. 잘못된 포맷은 로그로 남긴다.
    """
    if not expires_at:
        return False
    raw = expires_at.strip()
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        exp = datetime.fromisoformat(raw)
    except ValueError:
        logger.warning("expires_at 파싱 실패 — 만료 미적용: %r", expires_at)
        return False
    if exp.tzinfo is None:
        exp = exp.replace(tzinfo=UTC)
    return datetime.now(UTC) >= exp


def extract_session_token(request: Request) -> str | None:
    """Request cookie 에서 세션 토큰 추출 — 없으면 None.

    Args:
        request: FastAPI Request 객체.

    Returns:
        토큰 문자열 또는 None.
    """
    token = request.cookies.get(SESSION_COOKIE_NAME)
    if not token:
        return None
    return token


async def require_auth(request: Request) -> str:
    """인증된 사용자 라이선스 코드 반환 — 미인증 시 401.

    Args:
        request: FastAPI Request (cookie 헤더 포함).

    Returns:
        세션에 귀속된 user_code (라이선스 코드).

    Raises:
        HTTPException: 토큰 부재/만료/user_code 부재 → 401 Unauthorized.
    """
    token = extract_session_token(request)
    if token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="세션 토큰이 없습니다 — 로그인이 필요합니다.",
        )
    user_code = get_user_from_session(token)
    if user_code is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="세션이 만료되었거나 유효하지 않습니다 — 다시 로그인해 주세요.",
        )
    # 라이선스 만료 게이트 — DB 백엔드(SaaS) 에서만 적용. 메모리 모드(데스크탑
    # 단일 사용자)는 _db_path 가 None 이라 skip. referral(expires_at None)은 무기한.
    # 만료 시 403 — 세션/토큰은 유효하나 구독이 끝난 상태(401 재로그인과 구분).
    # /auth/status 는 require_auth 의존성이 없어 만료돼도 조회 가능(UI 가 만료 안내).
    db = get_session_db_path()
    if db is not None:
        user = users_db.get_user_by_code(db, user_code)
        if user is not None and is_license_expired(user.get("expires_at")):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="구독이 만료되었습니다 — 갱신 후 이용 가능합니다.",
            )
    return user_code


async def optional_auth(request: Request) -> str | None:
    """인증된 user_code 또는 None — endpoint 가 비인증도 허용할 때 사용.

    Args:
        request: FastAPI Request.

    Returns:
        user_code 또는 None (비인증).
    """
    token = extract_session_token(request)
    if token is None:
        return None
    return get_user_from_session(token)


__all__ = [
    "SESSION_COOKIE_NAME",
    "extract_session_token",
    "optional_auth",
    "require_auth",
]
