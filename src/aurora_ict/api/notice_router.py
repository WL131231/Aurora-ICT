"""공지사항 router — 사용자 GET + admin POST/DELETE.

설계:
    - GET /ict/notice — 인증된 사용자에게 활성 공지 1건 반환 (없으면 active=False).
    - POST /admin/notice — admin token 으로 보호. 공지 등록.
    - DELETE /admin/notice/{id} — admin token. 공지 삭제.
    - GET /admin/notice — admin token. 만료 포함 목록.

Admin token 검증:
    - env ``AURORA_ICT_ADMIN_TOKEN`` 와 헤더 ``X-Admin-Token`` 일치 (hmac.compare_digest).
    - env 미설정 시 admin endpoint 모두 503 (의도된 비활성).
"""

from __future__ import annotations

import hmac
import logging
import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from aurora_ict.auth import notices_db
from aurora_ict.auth.middleware import require_auth

logger = logging.getLogger(__name__)

_ADMIN_TOKEN_ENV = "AURORA_ICT_ADMIN_TOKEN"


def _admin_token() -> str | None:
    """env 에서 admin token 읽기 — 미설정 시 None."""
    tok = os.environ.get(_ADMIN_TOKEN_ENV, "").strip()
    return tok or None


def _verify_admin_token(x_admin_token: str) -> bool:
    """헤더 토큰 검증 — timing-safe compare. env 미설정 시 항상 False (비활성)."""
    expected = _admin_token()
    if not expected:
        return False
    return hmac.compare_digest(x_admin_token or "", expected)


class NoticeCreate(BaseModel):
    """공지 등록 요청 본문."""

    message: str = Field(..., min_length=1, max_length=500)
    severity: Literal["info", "warn", "critical"] = "info"
    expires_at: str | None = None  # ISO 8601 UTC


def create_notice_router(db_path: Path) -> APIRouter:
    """공지 router 생성 — db_path 클로저로 박힘."""
    router = APIRouter()

    @router.get("/ict/notice")
    async def get_notice(_user_code: str = Depends(require_auth)) -> dict:
        """인증 사용자에게 활성 공지 1건 반환."""
        notice = notices_db.get_active_notice(db_path)
        if notice is None:
            return {"active": False}
        return {
            "active": True,
            "id": notice["id"],
            "message": notice["message"],
            "severity": notice["severity"],
            "expires_at": notice.get("expires_at"),
            "created_at": notice.get("created_at"),
        }

    @router.post("/admin/notice")
    async def create_notice_admin(
        payload: NoticeCreate,
        x_admin_token: str = Header(..., alias="X-Admin-Token"),
    ) -> dict:
        """공지 등록 — admin token 필수.

        env AURORA_ICT_ADMIN_TOKEN 미설정 시 503 (의도된 비활성).
        토큰 불일치 시 401.
        """
        if _admin_token() is None:
            raise HTTPException(503, "admin token 미설정")
        if not _verify_admin_token(x_admin_token):
            raise HTTPException(401, "invalid admin token")
        try:
            nid = notices_db.create_notice(
                db_path,
                payload.message,
                payload.severity,
                payload.expires_at,
            )
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        logger.info(
            "공지 등록 — id=%d severity=%s expires=%s",
            nid, payload.severity, payload.expires_at,
        )
        return {"id": nid}

    @router.delete("/admin/notice/{notice_id}")
    async def delete_notice_admin(
        notice_id: int,
        x_admin_token: str = Header(..., alias="X-Admin-Token"),
    ) -> dict:
        """공지 삭제."""
        if _admin_token() is None:
            raise HTTPException(503, "admin token 미설정")
        if not _verify_admin_token(x_admin_token):
            raise HTTPException(401, "invalid admin token")
        ok = notices_db.delete_notice(db_path, notice_id)
        logger.info("공지 삭제 시도 — id=%d removed=%s", notice_id, ok)
        return {"deleted": ok}

    @router.get("/admin/notice")
    async def list_notices_admin(
        x_admin_token: str = Header(..., alias="X-Admin-Token"),
        include_expired: bool = False,
    ) -> dict:
        """공지 목록 (admin 조회용)."""
        if _admin_token() is None:
            raise HTTPException(503, "admin token 미설정")
        if not _verify_admin_token(x_admin_token):
            raise HTTPException(401, "invalid admin token")
        items = notices_db.list_notices(db_path, include_expired=include_expired)
        return {"items": items}

    return router


__all__ = ["create_notice_router"]
