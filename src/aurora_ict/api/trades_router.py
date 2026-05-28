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
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from fastapi.responses import PlainTextResponse, StreamingResponse

logger = logging.getLogger(__name__)


_ADMIN_TOKEN_ENV = "AURORA_ICT_ADMIN_TOKEN"

# CSV 컬럼 순서 — UI / Excel 호환.
_CSV_HEADERS = (
    "ts_ms", "event_type", "symbol", "direction", "price", "qty",
    "pnl_usdt", "setup_ts_ms", "reason",
)


def _trades_db_path(data_dir: Path, user_code: str) -> Path:
    """사용자별 trades.db 경로 — `<data_dir>/users/<code>/trades.db`."""
    return data_dir / "users" / user_code / "trades.db"


def _trades_jsonl_path(data_dir: Path, user_code: str) -> Path:
    """사용자별 trades.jsonl 경로."""
    return data_dir / "users" / user_code / "trades.jsonl"


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
        "pnl_usdt, setup_ts_ms, reason, context_json "
        f"FROM trades{where_sql} "
        "ORDER BY ts_ms DESC LIMIT ?"
    )
    params.append(limit)
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _trades_to_csv(rows: list[dict[str, Any]]) -> str:
    """CSV 직렬화 — Excel 한국어 호환을 위해 BOM 없이 UTF-8."""
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_CSV_HEADERS)
    for r in rows:
        writer.writerow([
            r.get("ts_ms", ""),
            r.get("event_type", ""),
            r.get("symbol", ""),
            r.get("direction", ""),
            r.get("price", ""),
            r.get("qty", ""),
            r.get("pnl_usdt") if r.get("pnl_usdt") is not None else "",
            r.get("setup_ts_ms") if r.get("setup_ts_ms") is not None else "",
            r.get("reason", ""),
        ])
    return buf.getvalue()


def _check_admin_token(x_admin_token: str | None) -> None:
    """Admin token 검증 — timing-safe (hmac.compare_digest).

    Raises:
        HTTPException 401: 환경변수 미설정 또는 토큰 불일치.
    """
    expected = os.environ.get(_ADMIN_TOKEN_ENV)
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=f"{_ADMIN_TOKEN_ENV} 환경변수가 설정되지 않았습니다.",
        )
    if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
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


def create_trades_router(
    data_dir: Path,
    require_auth_dep: Any,
) -> APIRouter:
    """매매 로그 router — auth dep 를 주입 (SaaS / .exe 양쪽 호환).

    Args:
        data_dir: SaaS 데이터 루트 (Fly: ``/data``). 사용자별 trades.* 가
            `<data_dir>/users/<code>/` 에 저장됨.
        require_auth_dep: 본인 인증 dependency — ``user_code`` 반환.

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
        user_code: str = Query(..., min_length=4, max_length=64),
        limit: int = Query(default=500, ge=1, le=10000),
        since_ms: int | None = Query(default=None, ge=0),
        event_type: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        """지정한 사용자 거래 — 운영 진단용."""
        _check_admin_token(x_admin_token)
        db = _trades_db_path(data_dir, user_code)
        rows = _query_trades(db, limit, since_ms, event_type)
        return {
            "user_code": user_code,
            "trades": rows,
            "count": len(rows),
            "stats": _aggregate_stats(rows),
        }

    @router.get("/admin/trades/export")
    async def admin_export_trades_csv(
        user_code: str = Query(..., min_length=4, max_length=64),
        since_ms: int | None = Query(default=None, ge=0),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> StreamingResponse:
        """지정 사용자 CSV — 운영 백업/오프라인 분석."""
        _check_admin_token(x_admin_token)
        db = _trades_db_path(data_dir, user_code)
        rows = _query_trades(db, limit=200000, since_ms=since_ms, event_type=None)
        csv_text = _trades_to_csv(rows)
        filename = f"trades_{user_code}.csv"
        return StreamingResponse(
            iter([csv_text]),
            media_type="text/csv; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )

    @router.get("/admin/trades/all_users")
    async def admin_all_users_summary(
        since_ms: int | None = Query(default=None, ge=0),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> dict[str, Any]:
        """전체 사용자 통계 — 최근 N (since_ms) 거래 + 사용자별 집계.

        Returns:
            ``{users: [{code, count, pnl_sum, win_rate, ...}], total_pnl, ...}``.
        """
        _check_admin_token(x_admin_token)
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

    @router.get("/admin/trades/backup", response_class=PlainTextResponse)
    async def admin_backup_jsonl(
        user_code: str = Query(..., min_length=4, max_length=64),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    ) -> PlainTextResponse:
        """JSONL raw 다운로드 — source of truth 백업용. 큰 파일은 그대로 stream."""
        _check_admin_token(x_admin_token)
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

    return router


__all__ = ["create_trades_router"]
