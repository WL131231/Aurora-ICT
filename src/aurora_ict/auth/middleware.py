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

from fastapi import HTTPException, Request, status

from aurora_ict.auth.pin import get_user_from_session

# cookie 이름 — router.py 와 단일 source of truth (서로 import 해 일관 유지).
SESSION_COOKIE_NAME = "aurora_ict_session"


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
