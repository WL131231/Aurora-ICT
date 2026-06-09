"""매매 로그 read API — 본인 + admin (2026-05-29).

엔드포인트:
    * ``GET  /ict/trades`` — 본인 거래 (require_auth). limit/since_ms/event_type 필터.
    * ``GET  /ict/trades/export`` — 본인 거래 CSV 다운로드 (require_auth).
    * ``GET  /admin/trades`` — 특정 사용자 거래 조회 (X-Admin-Token).
    * ``GET  /admin/trades/export`` — 특정 사용자 거래 CSV (X-Admin-Token).
    * ``GET  /admin/trades/all_users`` — 전체 사용자 24h 통계 (X-Admin-Token).
    * ``GET  /admin/trades/backup`` — JSONL raw 다운로드 (X-Admin-Token).

Why 별도 router:
    notice_router 와 패턴 일관 — auth 와 admin token 양쪽을 모두 다루는
    router 는 별도 파일로 격리해 saas/app.py 비대화 회피.

저장소 형태:
    ``<data_dir>/users/<code>/trades.jsonl`` (source of truth)
    ``<data_dir>/users/<code>/trades.db`` (분석 쿼리 — 빠른 SELECT)
    ``<data_dir>/users/<code>/trade_journal.log`` (사람용 텍스트)

조회는 SQLite 기준. JSONL 은 raw 백업.

담당: 지영민 (SaaS 매매 로그 격리 PR)
"""

from __future__ import annotations

import csv
import hmac
import io
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    Cookie,
    Depends,
    Header,
    HTTPException,
    Query,
    Response,
)
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


_ADMIN_TOKEN_ENV = "AURORA_ICT_ADMIN_TOKEN"

# 2026-05-29 (v2): UI 전용 별도 admin 패스워드 — 텔레그램 봇과 같은 토큰을 UI
# 로그인 비번으로 쓰는 게 불편하다는 파트너 피드백. 텔레그램 봇 호환은
# AURORA_ICT_ADMIN_TOKEN 유지, UI 사용자는 본인이 설정한 비번으로 로그인.
# 둘 중 하나만 일치해도 통과 (헤더/쿠키 모두).
_ADMIN_UI_PASSWORD_ENV = "AURORA_ICT_ADMIN_UI_PASSWORD"

# 2026-05-29: admin token 인증 쿠키 — UI 에서 X-Admin-Token 한 번 입력 후
# HttpOnly Secure SameSite cookie 로 보관해 이후 admin endpoints 자동 호출.
_ADMIN_COOKIE_NAME = "aurora_admin_token"
_ADMIN_COOKIE_TTL_SEC = 30 * 24 * 3600  # 30일

# CSV 컬럼 순서 — UI / Excel 호환.
# 2026-05-29: mode 컬럼 추가 (DEMO/LIVE 구분, 파트너 요청).
_CSV_HEADERS = (
    "ts_ms", "event_type", "mode", "symbol", "direction", "price", "qty",
    "pnl_usdt", "setup_ts_ms", "reason",
)


# user_code 는 파일 경로(<data_dir>/users/<code>/...) 에 직접 들어간다. admin
# endpoint 의 user_code 는 신뢰 경계 밖 입력이라, path traversal(../, 절대경로,
# 경로 구분자)을 막기 위해 영숫자/하이픈/언더스코어만 허용한다.
_SAFE_USER_CODE = re.compile(r"^[A-Za-z0-9_-]+$")


def _safe_user_code(user_code: str) -> str:
    """user_code 안전성 검증 — 경로 탈출 문자 차단. 위반 시 400.

    모든 경로 조립의 중앙 게이트 (admin/본인용 공통). Query pattern 과 함께
    defense-in-depth.
    """
    if not _SAFE_USER_CODE.match(user_code):
        raise HTTPException(status_code=400, detail="잘못된 user_code 형식입니다.")
    return user_code


def _trades_db_path(data_dir: Path, user_code: str) -> Path:
    """사용자별 trades.db 경로 — `<data_dir>/users/<code>/trades.db`."""
    return data_dir / "users" / _safe_user_code(user_code) / "trades.db"


def _trades_jsonl_path(data_dir: Path, user_code: str) -> Path:
    """사용자별 trades.jsonl 경로."""
    return data_dir / "users" / _safe_user_code(user_code) / "trades.jsonl"


def _query_trades(
    db_path: Path,
    limit: int,
    since_ms: int | None,
    event_type: str | None,
) -> list[dict[str, Any]]:
    """SQLite SELECT — 시간 내림차순 (최근 우선).

    Args:
        db_path: 사용자별 trades.db.
        limit: 반환 row 수 상한 (1~5000).
        since_ms: 이 ts_ms 이상만 (None = 전체).
        event_type: 이벤트 유형 필터 (None = 전체).

    Returns:
        dict list. db 부재 시 빈 list.
    """
    if not db_path.exists():
        return []
    where = []
    params: list[Any] = []
    if since_ms is not None and since_ms > 0:
        where.append("ts_ms >= ?")
        params.append(since_ms)
    if event_type:
        where.append("event_type = ?")
        params.append(event_type)
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT ts_ms, event_type, symbol, direction, price, qty, "
        "pnl_usdt, setup_ts_ms, reason, context_json, mode "
        f"FROM trades{where_sql} "
        "ORDER BY ts_ms DESC LIMIT ?"
    )
    params.append(limit)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        # 2026-05-29 hot-fix: 옛 trades.db (PR #166 이전 사용자) 에 mode
        # 컬럼 없으면 SELECT 가 OperationalError → HTTP 500. TradesStore
        # __init__ 의 ALTER 가 봇 가동 시점에만 실행되므로, admin 이 봇 비가동
        # 사용자 조회 시 마이그레이션 안 됨. 여기서 idempotent ALTER 한 번 더.
        try:
            conn.execute("ALTER TABLE trades ADD COLUMN mode TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # 이미 컬럼 있음 또는 trades 테이블 자체 없음 (빈 DB)
        try:
            rows = conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError as e:
            # trades 테이블 자체 없음 — 사용자 봇이 한 번도 가동된 적 없는 경우.
            # 빈 list 반환해 UI 가 "데이터 없음" 표시 (HTTP 500 회피).
            logger.warning(
                "trades 테이블 조회 실패 (빈 DB 추정): %s", e,
            )
            return []
    return [dict(r) for r in rows]


def _trades_to_csv(
    rows: list[dict[str, Any]], *, with_user_code: bool = False,
) -> str:
    """CSV 직렬화 — Excel 한국어 호환을 위해 BOM 없이 UTF-8.

    with_user_code=True 면 맨 앞에 ``user_code`` 컬럼 추가 — 전체 사용자 일괄
    export 시 어느 사용자 거래인지 구분용.
    """
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    headers = ["user_code", *_CSV_HEADERS] if with_user_code else list(_CSV_HEADERS)
    writer.writerow(headers)
    for r in rows:
        cols = [
            r.get("ts_ms", ""),
            r.get("event_type", ""),
            r.get("mode") or "",  # DEMO/LIVE — 구식 row 는 빈값.
            r.get("symbol", ""),
            r.get("direction", ""),
            r.get("price", ""),
            r.get("qty", ""),
            r.get("pnl_usdt") if r.get("pnl_usdt") is not None else "",
            r.get("setup_ts_ms") if r.get("setup_ts_ms") is not None else "",
            r.get("reason", ""),
        ]
        if with_user_code:
            cols = [r.get("user_code", ""), *cols]
        writer.writerow(cols)
    return buf.getvalue()


def _expected_admin_secrets() -> list[str]:
    """허용된 admin 비밀값 목록 — ADMIN_TOKEN (텔레그램 봇) + ADMIN_UI_PASSWORD (UI).

    2026-05-29 (v2): UI 사용자가 별도 비번을 설정할 수 있게 분리. 둘 다
    옵션 — 환경변수 미설정 시 해당 값은 후보에서 제외. 둘 다 미설정이면
    503 (admin 기능 비활성).
    """
    out = []
    tok = os.environ.get(_ADMIN_TOKEN_ENV)
    if tok:
        out.append(tok)
    pwd = os.environ.get(_ADMIN_UI_PASSWORD_ENV)
    if pwd:
        out.append(pwd)
    return out


def _check_admin_token(x_admin_token: str | None) -> None:
    """Admin token 검증 — timing-safe (hmac.compare_digest).

    텔레그램 봇 등 server-to-server 호출이 헤더로 ADMIN_TOKEN 보내는 케이스.
    ADMIN_UI_PASSWORD 도 같이 후보로 두면 UI 헤더 직접 호출도 통과.

    Raises:
        HTTPException 401: 환경변수 미설정 또는 토큰 불일치.
    """
    expected_list = _expected_admin_secrets()
    if not expected_list:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{_ADMIN_TOKEN_ENV} 또는 {_ADMIN_UI_PASSWORD_ENV} 환경변수가 "
                "설정되지 않았습니다."
            ),
        )
    if not x_admin_token:
        raise HTTPException(status_code=401, detail="invalid admin token")
    for expected in expected_list:
        if hmac.compare_digest(x_admin_token, expected):
            return
    raise HTTPException(status_code=401, detail="invalid admin token")


def _check_admin_cookie_or_header(
    cookie_token: str | None, header_token: str | None,
) -> None:
    """헤더 또는 쿠키 둘 중 하나가 admin token / UI password 와 일치하면 통과.

    2026-05-29 (v2): 환경변수 2개 (텔레그램 봇용 ADMIN_TOKEN + UI 전용
    ADMIN_UI_PASSWORD) 중 어느 값과 일치해도 통과. 후보 비교는 timing-safe.

    UI 사용자는 1회 본인의 password 헤더 입력 → 쿠키 발급 → 이후 쿠키만으로
    admin endpoint 접근. 텔레그램 봇은 ADMIN_TOKEN 헤더로 호출.

    Raises:
        HTTPException 401: 둘 다 없거나 모든 후보와 불일치.
        HTTPException 503: 두 환경변수 모두 미설정.
    """
    expected_list = _expected_admin_secrets()
    if not expected_list:
        raise HTTPException(
            status_code=503,
            detail=(
                f"{_ADMIN_TOKEN_ENV} 또는 {_ADMIN_UI_PASSWORD_ENV} 환경변수가 "
                "설정되지 않았습니다."
            ),
        )
    candidates = [header_token, cookie_token]
    for tok in candidates:
        if not tok:
            continue
        for expected in expected_list:
            if hmac.compare_digest(tok, expected):
                return
    raise HTTPException(status_code=401, detail="invalid admin token")


def _list_user_codes(data_dir: Path) -> list[str]:
    """`<data_dir>/users/` 하위 모든 사용자 코드 목록 — admin 통계용."""
    root = data_dir / "users"
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def _aggregate_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """거래 행에서 카운트·총 PnL·승률 산출 — 청산 이벤트만 PnL 집계."""
    closed_events = {"sl_hit", "tp_hit", "flip_close", "sync_close", "manual_close"}
    closed = [r for r in rows if r.get("event_type") in closed_events]
    pnls = [r.get("pnl_usdt") for r in closed if r.get("pnl_usdt") is not None]
    wins = [p for p in pnls if p and p > 0]
    losses = [p for p in pnls if p and p <= 0]
    return {
        "total_events": len(rows),
        "closed_events": len(closed),
        "pnl_sum_usdt": round(sum(pnls), 2) if pnls else 0.0,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate_pct": (
            round(len(wins) / len(pnls) * 100, 2) if pnls else 0.0
        ),
    }


async def _resolve_seed(seed_provider: Any, user_code: str) -> float | None:
    """seed_provider(user_code) 안전 호출 — 미주입/실패 시 None.

    매매기록 화면의 '현재 시드' 카드용. 거래소 조회 실패가 매매기록 표시
    자체를 막지 않도록 예외를 삼키고 None 반환.
    """
    if seed_provider is None:
        return None
    try:
        return await seed_provider(user_code)
    except Exception:  # noqa: BLE001
        return None


class _LicenseUpdateRequest(BaseModel):
    """라이선스 정보 정정 요청 — admin 만 호출 가능 (2026-05-29)."""

    code: str = Field(min_length=4, max_length=64)
    license_type: str = Field(pattern="^(referral|sub_30d|sub_90d|sub_365d)$")
    # ISO 8601 UTC (예: "2027-05-28T23:59:59Z") 또는 None (referral 무기한).
    expires_at: str | None = None


def create_trades_router(
    data_dir: Path,
    require_auth_dep: Any,
    *,
    secure_cookie: bool = True,
    auth_db_path: Path | None = None,
    seed_provider: Any = None,
) -> APIRouter:
    """매매 로그 router — auth dep 를 주입 (SaaS / .exe 양쪽 호환).

    Args:
        data_dir: SaaS 데이터 루트 (Fly: ``/data``). 사용자별 trades.* 가
            `<data_dir>/users/<code>/` 에 저장됨.
        require_auth_dep: 본인 인증 dependency — ``user_code`` 반환.
        secure_cookie: True (기본) 면 admin 쿠키에 Secure 플래그. 운영(HTTPS)
            에서 항상 True. TestClient (HTTP) 에서는 False 로 주입해야
            쿠키가 후속 요청에 자동 포함된다.
        auth_db_path: users.db 경로. 라이선스 admin endpoint
            (``/admin/user/license``) 가 사용. None 이면 license endpoint 등록
            안 함 (단일 사용자 / .exe 흐름 호환).
        seed_provider: async (user_code) -> float | None. 사용자의 현재 USDT
            지갑 잔고(시드) 조회 콜백. SaaS 는 mu_manager.get_seed_usdt 주입,
            단일/.exe 는 None(시드 표시 생략).

    Returns:
        FastAPI APIRouter — prefix 없음 (`/ict/trades`, `/admin/trades` 등).
    """
    router = APIRouter(tags=["trades"])

    # ==========================================================
    # 본인 거래 — /ict/trades
    # ==========================================================

    @router.get("/ict/trades")
    async def list_my_trades(
        limit: int = Query(default=200, ge=1, le=5000),
        since_ms: int | None = Query(default=None, ge=0),
        event_type: str | None = Query(default=None),
        user_code: str = Depends(require_auth_dep),
    ) -> dict[str, Any]:
        """본인 거래 list — 시간 내림차순. SQLite 부재 시 빈 list."""
        db = _trades_db_path(data_dir, user_code)
        rows = _query_trades(db, limit, since_ms, event_type)
        return {
            "trades": rows,
            "count": len(rows),
            "stats": _aggregate_stats(rows),
            "current_seed_usdt": await _resolve_seed(seed_provider, user_code),
        }

    @router.get("/ict/trades/export")
    async def export_my_trades_csv(
        since_ms: int | None = Query(default=None, ge=0),
        event_type: str | None = Query(default=None),
        user_code: str = Depends(require_auth_dep),
    ) -> StreamingResponse:
        """본인 거래 CSV 다운로드 — 최대 50000행."""
        db = _trades_db_path(data_dir, user_code)
        rows = _query_trades(db, limit=50000, since_ms=since_ms, event_type=event_type)
        csv_text = _trades_to_csv(rows)
        filename = f"trades_{user_code}.csv"
        return StreamingResponse(
            iter([csv_text]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    # ==========================================================
    # Admin — /admin/trades (X-Admin-Token)
    # ==========================================================

    @router.get("/admin/trades")
    async def admin_list_trades(
        user_code: str = Query(..., min_length=4, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
        limit: int = Query(default=500, ge=1, le=10000),
        since_ms: int | None = Query(default=None, ge=0),
        event_type: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        cookie_token: str | None = Cookie(default=None, alias=_ADMIN_COOKIE_NAME),
    ) -> dict[str, Any]:
        """지정한 사용자 거래 — 운영 진단용."""
        _check_admin_cookie_or_header(cookie_token, x_admin_token)
        db = _trades_db_path(data_dir, user_code)
        rows = _query_trades(db, limit, since_ms, event_type)
        return {
            "user_code": user_code,
            "trades": rows,
            "count": len(rows),
            "stats": _aggregate_stats(rows),
            "current_seed_usdt": await _resolve_seed(seed_provider, user_code),
        }

    @router.get("/admin/trades/export")
    async def admin_export_trades_csv(
        user_code: str = Query(..., min_length=4, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
        since_ms: int | None = Query(default=None, ge=0),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        cookie_token: str | None = Cookie(default=None, alias=_ADMIN_COOKIE_NAME),
    ) -> StreamingResponse:
        """지정 사용자 CSV — 운영 백업/오프라인 분석."""
        _check_admin_cookie_or_header(cookie_token, x_admin_token)
        db = _trades_db_path(data_dir, user_code)
        rows = _query_trades(db, limit=200000, since_ms=since_ms, event_type=None)
        csv_text = _trades_to_csv(rows)
        filename = f"trades_{user_code}.csv"
        return StreamingResponse(
            iter([csv_text]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.get("/admin/trades/export-all")
    async def admin_export_all_trades_csv(
        since_ms: int | None = Query(default=None, ge=0),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        cookie_token: str | None = Cookie(default=None, alias=_ADMIN_COOKIE_NAME),
    ) -> StreamingResponse:
        """전체 사용자 거래를 한 CSV 로 — user_code 컬럼 포함 (운영 일괄 분석용).

        사용자별로 일일이 받지 않고 모든 사용자 매매를 한 파일로 합쳐 내려준다.
        """
        _check_admin_cookie_or_header(cookie_token, x_admin_token)
        all_rows: list[dict[str, Any]] = []
        for code in _list_user_codes(data_dir):
            db = _trades_db_path(data_dir, code)
            rows = _query_trades(
                db, limit=200000, since_ms=since_ms, event_type=None,
            )
            for r in rows:
                r["user_code"] = code
            all_rows.extend(rows)
        csv_text = _trades_to_csv(all_rows, with_user_code=True)
        return StreamingResponse(
            iter([csv_text]),
            media_type="text/csv; charset=utf-8",
            headers={
                "Content-Disposition": 'attachment; filename="trades_all_users.csv"',
            },
        )

    @router.get("/admin/trades/all_users")
    async def admin_all_users_summary(
        since_ms: int | None = Query(default=None, ge=0),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        cookie_token: str | None = Cookie(default=None, alias=_ADMIN_COOKIE_NAME),
    ) -> dict[str, Any]:
        """전체 사용자 통계 — 최근 N (since_ms) 거래 + 사용자별 집계.

        Returns:
            ``{users: [{code, count, pnl_sum, win_rate, ...}], total_pnl, ...}``.
        """
        _check_admin_cookie_or_header(cookie_token, x_admin_token)
        codes = _list_user_codes(data_dir)
        per_user: list[dict[str, Any]] = []
        total_pnl = 0.0
        for code in codes:
            db = _trades_db_path(data_dir, code)
            rows = _query_trades(db, limit=5000, since_ms=since_ms, event_type=None)
            stats = _aggregate_stats(rows)
            stats["code"] = code
            per_user.append(stats)
            total_pnl += stats["pnl_sum_usdt"]
        return {
            "user_count": len(codes),
            "total_pnl_usdt": round(total_pnl, 2),
            "users": per_user,
        }

    # ==========================================================
    # 2026-05-29: Admin 인증 (UI 전용) + 보조 endpoints
    # ==========================================================

    @router.post("/admin/login")
    async def admin_login(
        response: Response,
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        """Admin token 검증 후 HttpOnly Secure 쿠키 발급.

        UI 흐름:
            1. 사용자가 모달에 token 입력 → 이 endpoint 로 X-Admin-Token 헤더 POST
            2. 통과 시 ``aurora_admin_token`` 쿠키 (HttpOnly+Secure+SameSite=Lax) set
            3. 이후 admin endpoints 가 쿠키 자동 읽음 → UI 추가 입력 X.

        쿠키 보안:
            - HttpOnly: JS 접근 차단 (XSS 시 token 유출 방지)
            - Secure: HTTPS 만 (Fly 는 force_https=true 라 항상 만족)
            - SameSite=Lax: CSRF 완화
            - Max-Age 30일
        """
        _check_admin_token(x_admin_token)
        # 쿠키에는 실제 token 그대로 박는다 — 서버가 받을 때 다시 hmac 비교.
        # 별도 세션 store 없이 stateless 검증 (token 자체가 비밀이므로 OK).
        response.set_cookie(
            key=_ADMIN_COOKIE_NAME,
            value=x_admin_token or "",
            max_age=_ADMIN_COOKIE_TTL_SEC,
            httponly=True,
            secure=secure_cookie,
            samesite="lax",
        )
        return {"ok": True}

    @router.post("/admin/logout")
    async def admin_logout(response: Response) -> dict[str, Any]:
        """Admin 쿠키 제거 — 멱등 (없는 쿠키 delete 도 200)."""
        response.delete_cookie(
            key=_ADMIN_COOKIE_NAME,
            httponly=True,
            secure=secure_cookie,
            samesite="lax",
        )
        return {"ok": True}

    @router.get("/admin/session")
    async def admin_session(
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        cookie_token: str | None = Cookie(default=None, alias=_ADMIN_COOKIE_NAME),
    ) -> dict[str, Any]:
        """현재 admin 인증 여부 — UI 가 부팅 시 호출해 admin UI 분기."""
        try:
            _check_admin_cookie_or_header(cookie_token, x_admin_token)
            return {"authenticated": True}
        except HTTPException:
            return {"authenticated": False}

    @router.get("/admin/users")
    async def admin_list_users(
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        cookie_token: str | None = Cookie(default=None, alias=_ADMIN_COOKIE_NAME),
    ) -> dict[str, Any]:
        """매매 기록 디렉토리에 존재하는 사용자 코드 목록 — UI dropdown 용."""
        _check_admin_cookie_or_header(cookie_token, x_admin_token)
        return {"users": _list_user_codes(data_dir)}

    @router.post("/admin/trades/rebuild")
    async def admin_rebuild_sqlite(
        user_code: str = Query(..., min_length=4, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        cookie_token: str | None = Cookie(default=None, alias=_ADMIN_COOKIE_NAME),
    ) -> dict[str, Any]:
        """지정 사용자의 trades.db 를 trades.jsonl 에서 재생성.

        용도:
            마이그레이션 직후 또는 SQLite 일부 row 누락/손상 의심될 때. JSONL 은
            source of truth 라 항상 보존. 재빌드는 기존 ``trades`` 테이블을 비운 뒤
            JSONL 전체를 다시 insert (TradesStore.rebuild_sqlite 활용).

        Returns:
            ``{ok: True, user_code, inserted_count}`` 또는 jsonl 부재 시
            ``{ok: False, reason: "no_jsonl"}``.
        """
        _check_admin_cookie_or_header(cookie_token, x_admin_token)
        from aurora_ict.interfaces.trades_store import TradesStore
        user_dir = data_dir / "users" / user_code
        jsonl = user_dir / "trades.jsonl"
        if not jsonl.exists():
            return {"ok": False, "reason": "no_jsonl", "user_code": user_code}
        # TradesStore __init__ 가 directory 만 받고 자동으로 db / jsonl 열어준다.
        store = TradesStore(user_dir)
        try:
            inserted = store.rebuild_sqlite()
        finally:
            store.close()
        return {
            "ok": True,
            "user_code": user_code,
            "inserted_count": inserted,
        }

    @router.get("/admin/trades/backup", response_class=PlainTextResponse)
    async def admin_backup_jsonl(
        user_code: str = Query(..., min_length=4, max_length=64, pattern=r"^[A-Za-z0-9_-]+$"),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        cookie_token: str | None = Cookie(default=None, alias=_ADMIN_COOKIE_NAME),
    ) -> PlainTextResponse:
        """JSONL raw 다운로드 — source of truth 백업용. 큰 파일은 그대로 stream."""
        _check_admin_cookie_or_header(cookie_token, x_admin_token)
        path = _trades_jsonl_path(data_dir, user_code)
        if not path.exists():
            return PlainTextResponse("", status_code=200)
        return PlainTextResponse(
            path.read_text(encoding="utf-8"),
            media_type="application/x-ndjson",
            headers={
                "Content-Disposition": (
                    f'attachment; filename="trades_{user_code}.jsonl"'
                ),
            },
        )

    # ==========================================================
    # 2026-05-29: Admin 라이선스 정정 — UI 가 "레퍼럴 / 무기한" 으로 잘못 표시되는
    # 버그 (setup_pin 시 라이선스 서버 검증 없이 기본 referral 박힘) 처방.
    #
    # 2026-05-29 근본 fix: set_license (create-or-update idempotent) 사용.
    # - 사용자가 아직 setup_pin 안 했어도 (DB row 없어도) 라이선스 봇이 미리
    #   호출해서 pre-insert 가능.
    # - admin_bot.py 의 /issue_sub 시 자동 호출 → setup_pin 시 default
    #   referral 로 덮어쓰지 않음.
    # ==========================================================

    if auth_db_path is not None:
        @router.post("/admin/user/license")
        async def admin_update_license(
            req: _LicenseUpdateRequest,
            x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
            cookie_token: str | None = Cookie(default=None, alias=_ADMIN_COOKIE_NAME),
        ) -> dict[str, Any]:
            """사용자 라이선스 type / 만료일 박기 (admin 만, idempotent).

            기존 사용자 → UPDATE. 신규 코드 (아직 setup_pin 안 함) → INSERT.
            라이선스 봇이 코드 발급 직후 호출하면 pre-insert 되어 setup_pin
            때 default referral 로 덮어쓰지 않음.

            예 (sub_365d, 2027-05-28 만료):
                POST /admin/user/license
                {
                  "code": "AICT-XXXX-XXXX-XXXX",
                  "license_type": "sub_365d",
                  "expires_at": "2027-05-28T23:59:59Z"
                }

            응답 — 신규 INSERT 면 created=True, 정정 UPDATE 면 created=False.
            """
            _check_admin_cookie_or_header(cookie_token, x_admin_token)
            # users_db 는 외부 import (순환 회피).
            from aurora_ict.auth import users_db
            try:
                result = users_db.set_license(
                    auth_db_path,
                    code=req.code,
                    license_type=req.license_type,
                    expires_at=req.expires_at,
                )
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
            return {"ok": True, **result}

    return router


__all__ = ["create_trades_router"]
