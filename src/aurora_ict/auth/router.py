"""FastAPI 인증 router — 코드 + PIN 기반 SaaS 로그인 흐름.

파트너 결정 2026-05-28 — Aurora-ICT 다중 사용자 SaaS 전환.

제공 endpoint:
    - ``POST /auth/setup-pin``   — 신규/기존 사용자 첫 PIN 설정 + 자동 로그인
    - ``POST /auth/login``       — 코드 + PIN 검증 → 세션 cookie 발급
    - ``POST /auth/logout``      — 세션 무효화 (멱등)
    - ``GET  /auth/status``      — 현재 세션 상태 + 사용자 정보
    - ``POST /auth/api-keys``    — 인증된 사용자 거래소 API 키 등록/갱신

설계 메모:
    - 세션 cookie 이름은 ``middleware.SESSION_COOKIE_NAME`` 와 일관.
    - cookie 속성: HttpOnly=True / SameSite=Lax / Max-Age=30일.
      Secure 는 router 생성 시 ``secure_cookie`` flag 로 토글 (HTTPS 환경 True).
    - PIN 강도 검증은 ``pin.validate_pin_strength`` 위임 (단일 진실).
    - 로그인 실패 시 sleep 0.5s — 타이밍 공격 완화 (timing attack mitigation).
    - DB 경로/keystore master key 는 router factory 인자로 주입 — 테스트 격리 용이.

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import BaseModel, Field

from aurora_ict import __version__
from aurora_ict.auth import keystore, pin, users_db
from aurora_ict.auth.middleware import SESSION_COOKIE_NAME, extract_session_token

# 세션 cookie TTL — pin._DEFAULT_TTL_SEC 와 일치 (30일).
_SESSION_TTL_SEC = 30 * 24 * 3600

# 로그인 실패 시 타이밍 공격 완화 sleep — 0.5초 (사용자 체감 가능하지만 brute-force 차단).
_LOGIN_FAIL_SLEEP_SEC = 0.5


# ============================================================
# Request 모델 — pydantic v2
# ============================================================


class SetupPinRequest(BaseModel):
    """PIN 최초 설정 요청."""

    # 라이선스 코드 형식 — ``AICT-XXXX-XXXX-XXXX`` 가정, 최소 길이만 검증.
    code: str = Field(min_length=4, max_length=64)
    pin: str = Field(min_length=1, max_length=128)
    pin_confirm: str = Field(min_length=1, max_length=128)


class LoginRequest(BaseModel):
    """로그인 요청."""

    code: str = Field(min_length=4, max_length=64)
    pin: str = Field(min_length=1, max_length=128)


class ApiKeysRequest(BaseModel):
    """거래소 API 키 등록 요청.

    2026-05-29 (Live 모드 지원): ``mode`` 추가 — ``"demo"`` / ``"live"`` 슬롯
    선택. 미지정 시 ``"demo"`` (backward compat: 구 UI 호출도 그대로 동작).
    """

    api_key: str = Field(min_length=1, max_length=256)
    api_secret: str = Field(min_length=1, max_length=512)
    mode: str = Field(default="demo", pattern="^(demo|live)$")


# ============================================================
# Router factory
# ============================================================


def create_auth_router(
    db_path: Path | str,
    *,
    secure_cookie: bool = False,
    master_key: bytes | None = None,
) -> APIRouter:
    """``/auth/*`` router 생성 — DB 경로/master key 를 주입.

    Args:
        db_path: ``users.db`` 절대 경로. 부모 디렉토리는 호출자가 보장.
        secure_cookie: True 면 cookie ``Secure`` 속성 활성 (HTTPS 전용).
            로컬/테스트 (HTTP) 에서는 False.
        master_key: keystore Fernet 키. None 이면 ``keystore.get_master_key()``
            기본 동작 (환경변수 / 파일) 사용. 테스트에서는 명시 주입.

    Returns:
        prefix ``/auth`` 의 APIRouter.
    """
    # DB 초기화 — idempotent. 첫 호출이면 테이블 생성.
    users_db.init_db(db_path)

    router = APIRouter(prefix="/auth", tags=["auth"])

    def _set_session_cookie(response: Response, token: str) -> None:
        """세션 cookie 박기 — SaaS 일관 속성.

        HttpOnly=True (JS 접근 차단), SameSite=Lax (CSRF 완화), Max-Age=30d.
        """
        response.set_cookie(
            key=SESSION_COOKIE_NAME,
            value=token,
            max_age=_SESSION_TTL_SEC,
            httponly=True,
            secure=secure_cookie,
            samesite="lax",
        )

    def _clear_session_cookie(response: Response) -> None:
        """세션 cookie 제거 — 로그아웃 시."""
        response.delete_cookie(
            key=SESSION_COOKIE_NAME,
            httponly=True,
            secure=secure_cookie,
            samesite="lax",
        )

    @router.post("/setup-pin")
    async def setup_pin(req: SetupPinRequest, response: Response) -> dict[str, object]:
        """첫 PIN 설정 — code 없으면 referral 로 자동 생성, 있으면 PIN 채움.

        Why 자동 생성:
            referral 모델은 "코드 받자마자 PIN 등록" 흐름 — 사전 DB 등록 없음.
            sub_* 라이선스는 별도 발급 시 사전 row 생성됨 (이 path 로 PIN 만 채움).

        실패 조건:
            - PIN ≠ pin_confirm → 400
            - PIN 강도 미달 → 400
            - 이미 pin_hash 있는 사용자 → 400 (재설정 별도 flow 필요)
        """
        if req.pin != req.pin_confirm:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="PIN 과 PIN 확인이 일치하지 않습니다.",
            )
        err = pin.validate_pin_strength(req.pin)
        if err is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"code": err.code, "message": err.message},
            )

        user = users_db.get_user_by_code(db_path, req.code)
        if user is None:
            # 신규 사용자 — referral 기본 발급. sub_* 는 별도 관리 도구로 사전 등록.
            try:
                users_db.create_user(db_path, req.code, license_type="referral")
            except sqlite3.IntegrityError as e:
                # 거의 발생 X (위에서 None 확인) — race condition 방어.
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="이미 등록된 코드입니다.",
                ) from e
            user = users_db.get_user_by_code(db_path, req.code)
            assert user is not None  # 방금 생성 → None 불가
        elif user.get("pin_hash"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="이미 PIN 이 설정된 코드입니다 — 재설정은 별도 절차가 필요합니다.",
            )

        pin_hash = pin.hash_pin(req.pin)
        ok = users_db.set_pin(db_path, req.code, pin_hash)
        if not ok:
            # 위에서 row 확정 후라 거의 도달 X — 방어적.
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PIN 저장 실패.",
            )
        users_db.update_last_login(db_path, req.code)

        token = pin.create_session(user_code=req.code, ttl_sec=_SESSION_TTL_SEC)
        _set_session_cookie(response, token)
        return {
            "ok": True,
            "code": req.code,
            "license_type": user.get("license_type", "referral"),
        }

    @router.post("/login")
    async def login(req: LoginRequest, response: Response) -> dict[str, object]:
        """코드 + PIN 검증 → 세션 cookie 발급.

        실패 시:
            - 무조건 401 (코드 미존재 / PIN 불일치 구분 X — 사용자 enum 방지)
            - sleep 0.5s 박아 timing attack 완화 (코드 미존재가 더 빨라지지 X)
        """
        user = users_db.get_user_by_code(db_path, req.code)
        # PIN 검증 — user 미존재여도 dummy 해시로 동일 시간 소비.
        stored = user.get("pin_hash") if user else None
        if stored is None:
            # dummy 해시 — verify_pin 가 항상 같은 비용 소비하도록.
            await asyncio.sleep(_LOGIN_FAIL_SLEEP_SEC)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="코드 또는 PIN 이 올바르지 않습니다.",
            )
        if not pin.verify_pin(req.pin, stored):
            await asyncio.sleep(_LOGIN_FAIL_SLEEP_SEC)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="코드 또는 PIN 이 올바르지 않습니다.",
            )

        users_db.update_last_login(db_path, req.code)
        token = pin.create_session(user_code=req.code, ttl_sec=_SESSION_TTL_SEC)
        _set_session_cookie(response, token)
        return {
            "ok": True,
            "code": req.code,
            "license_type": user.get("license_type", "referral"),
        }

    @router.post("/logout")
    async def logout(request: Request, response: Response) -> dict[str, bool]:
        """현재 세션 무효화 — 토큰 없어도 200 (멱등)."""
        token = extract_session_token(request)
        if token is not None:
            pin.revoke_session(token)
        _clear_session_cookie(response)
        return {"ok": True}

    @router.get("/status")
    async def status_endpoint(request: Request) -> dict[str, object]:
        """현재 세션 상태 — UI 가 초기 로딩 시 호출.

        Returns:
            미인증: ``{"authenticated": False, "needs_pin_setup": bool}``
            인증:   ``{"authenticated": True, "code": ..., "license_type": ...,
                       "has_api_keys": bool, "expires_at": str|None,
                       "created_at": str|None}``

        2026-05-28 — 파트너 요청. UI 우측 라이선스 카드용으로 만료일/가입일 노출.
        ``expires_at`` 은 referral 사용자는 None (무기한), sub_* 는 ISO 8601 UTC.
        ``created_at`` 은 가입 시각 (UI 가 잔여일 계산에 활용).
        """
        token = extract_session_token(request)
        user_code = pin.get_user_from_session(token) if token else None
        if user_code is None:
            # 미인증 — 가입자가 한 명도 없으면 "초기 셋업 필요" 표시.
            with sqlite3.connect(str(db_path)) as conn:
                row = conn.execute(
                    "SELECT COUNT(*) FROM users WHERE pin_hash IS NOT NULL",
                ).fetchone()
            has_any_pin = bool(row and row[0] > 0)
            return {
                "authenticated": False,
                "needs_pin_setup": not has_any_pin,
                # 2026-05-28: UI 자동 버전 갱신 — 미인증 응답에도 포함해야 로그인
                # 화면에서도 버전 drift 감지 가능.
                "app_version": __version__,
            }
        user = users_db.get_user_by_code(db_path, user_code)
        if user is None:
            # 세션은 있는데 DB row 사라짐 (드물지만 운영 사고 대비) — 세션 무효화.
            pin.revoke_session(token)
            return {
                "authenticated": False,
                "needs_pin_setup": False,
                "app_version": __version__,
            }
        # 2026-05-29 (Live 모드 지원): demo/live 키 등록 여부를 별도로 노출.
        # 기존 ``has_api_keys`` 는 둘 중 하나라도 있으면 True (backward compat) —
        # 구식 UI 도 "키 등록됨" 으로 인식하도록.
        has_demo = users_db.has_api_keys(db_path, user["code"], "demo")
        has_live = users_db.has_api_keys(db_path, user["code"], "live")
        return {
            "authenticated": True,
            "code": user["code"],
            "license_type": user.get("license_type", "referral"),
            "has_api_keys": bool(has_demo or has_live),
            "has_demo_keys": has_demo,
            "has_live_keys": has_live,
            # referral 은 expires_at 이 NULL — UI 가 "무기한" 으로 표시.
            "expires_at": user.get("expires_at"),
            # 가입 시각 — UI 가 잔여일 / 기간 표시에 사용. NOT NULL 컬럼.
            "created_at": user.get("created_at"),
            # 2026-05-28: UI 자동 버전 갱신 — 부팅 시점 캡쳐된 버전과 polling 시점
            # 버전을 비교해 다르면 location.reload() (app.js _checkVersionDrift).
            "app_version": __version__,
        }

    @router.post("/api-keys")
    async def set_api_keys_endpoint(
        req: ApiKeysRequest, request: Request,
    ) -> dict[str, object]:
        """거래소 API 키 등록/갱신 — secret 은 Fernet 으로 암호화 후 저장.

        인증 필요 — cookie 세션 토큰 검증.
        """
        token = extract_session_token(request)
        user_code = pin.get_user_from_session(token) if token else None
        if user_code is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="로그인이 필요합니다.",
            )
        api_key = req.api_key.strip()
        api_secret = req.api_secret.strip()
        if not api_key or not api_secret:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="api_key 와 api_secret 은 비어있을 수 없습니다.",
            )
        # secret 만 암호화 — public api_key 는 평문 OK (조회용).
        secret_enc = keystore.encrypt_secret(api_secret, key=master_key)
        ok = users_db.set_api_keys(
            db_path, user_code, api_key, secret_enc, mode=req.mode,
        )
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="해당 사용자가 존재하지 않습니다.",
            )
        return {"ok": True, "mode": req.mode}

    @router.delete("/api-keys")
    async def delete_api_keys_endpoint(
        request: Request,
        mode: str,
    ) -> dict[str, object]:
        """거래소 API 키 삭제 — 슬롯 NULL 로 비움 (2026-05-30 파트너 요청).

        UI 의 cred-slot '삭제' 버튼이 호출. 현재 모드와 무관하게 지정 모드의
        키만 삭제. 다른 모드 키는 그대로.

        Args:
            mode: ``demo`` 또는 ``live``. 쿼리 파라미터.

        Returns:
            ``{"ok": True, "mode": ...}``.
        """
        if mode not in ("demo", "live"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"mode 는 'demo' 또는 'live' 여야 합니다: {mode!r}",
            )
        token = extract_session_token(request)
        user_code = pin.get_user_from_session(token) if token else None
        if user_code is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="로그인이 필요합니다.",
            )
        ok = users_db.delete_api_keys(db_path, user_code, mode)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="해당 사용자가 존재하지 않습니다.",
            )
        return {"ok": True, "mode": mode}

    return router


__all__ = [
    "ApiKeysRequest",
    "LoginRequest",
    "SetupPinRequest",
    "create_auth_router",
]
