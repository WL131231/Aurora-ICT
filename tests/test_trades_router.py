"""trades_router 테스트 — 본인 / admin 분기 + CSV / 통계.

mock 0 — 실제 sqlite + TestClient 사용. require_auth 는 monkey-patched dep
대신 직접 호출한 적 없는 헬퍼로 우회.

검증:
    1. /ict/trades — 본인 거래만 반환 (사용자 격리)
    2. /ict/trades/export — CSV header + row 일치
    3. /admin/trades — admin token 없으면 401, 있으면 200
    4. /admin/trades/all_users — 사용자별 통계 집계
    5. _query_trades — limit / since_ms / event_type 필터 동작

담당: 지영민 (매매 로그 격리 PR)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from aurora_ict.api.trades_router import (
    _aggregate_stats,
    _query_trades,
    _trades_db_path,
    create_trades_router,
)

# ============================================================
# Fixtures
# ============================================================


@pytest.fixture
def data_dir(tmp_path) -> Path:
    """격리된 데이터 루트."""
    return tmp_path


def _create_user_trades(
    data_dir: Path, code: str, events: list[dict],
) -> Path:
    """사용자 디렉토리에 trades.db 생성 + 이벤트 insert."""
    user_dir = data_dir / "users" / code
    user_dir.mkdir(parents=True, exist_ok=True)
    db_path = user_dir / "trades.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS trades (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            symbol TEXT NOT NULL,
            direction TEXT NOT NULL,
            price REAL NOT NULL,
            qty REAL NOT NULL,
            pnl_usdt REAL,
            setup_ts_ms INTEGER,
            reason TEXT NOT NULL DEFAULT '',
            context_json TEXT
        )
        """,
    )
    for e in events:
        conn.execute(
            "INSERT INTO trades(ts_ms, event_type, symbol, direction, price, "
            "qty, pnl_usdt, setup_ts_ms, reason) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                e["ts_ms"],
                e["event_type"],
                e.get("symbol", "BTC/USDT:USDT"),
                e.get("direction", "short"),
                e.get("price", 73000.0),
                e.get("qty", 1.0),
                e.get("pnl_usdt"),
                e.get("setup_ts_ms"),
                e.get("reason", ""),
            ),
        )
    conn.commit()
    conn.close()
    # JSONL 도 같이 박아둠 (admin backup endpoint 용).
    jsonl = user_dir / "trades.jsonl"
    with jsonl.open("w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return db_path


def _make_app(data_dir: Path, current_user: str = "AICT-TEST-USER-0001") -> FastAPI:
    """trades_router 만 등록한 가벼운 FastAPI app — require_auth 는 고정 코드 반환.

    secure_cookie=False — TestClient 는 HTTP 라 Secure 쿠키가 후속 요청에
    안 실린다. 운영 SaaS 는 saas.py 에서 secure_cookie=True 로 주입.
    """
    app = FastAPI()

    async def _fake_require_auth() -> str:
        return current_user

    app.include_router(
        create_trades_router(data_dir, _fake_require_auth, secure_cookie=False),
    )
    return app


# ============================================================
# 1. _query_trades — 필터 동작
# ============================================================


def test_query_trades_returns_empty_when_db_missing(tmp_path):
    """db 파일 부재 시 빈 list."""
    assert _query_trades(tmp_path / "missing.db", limit=10, since_ms=None, event_type=None) == []


def test_query_trades_orders_by_ts_desc(data_dir):
    """ts_ms 내림차순 — 최근 우선."""
    code = "AICT-ORDR-ORDR-ORDR"
    _create_user_trades(data_dir, code, [
        {"ts_ms": 1000, "event_type": "entry"},
        {"ts_ms": 3000, "event_type": "sl_hit", "pnl_usdt": -10.0},
        {"ts_ms": 2000, "event_type": "tp_hit", "pnl_usdt": 5.0},
    ])
    rows = _query_trades(_trades_db_path(data_dir, code), limit=10, since_ms=None, event_type=None)
    assert [r["ts_ms"] for r in rows] == [3000, 2000, 1000]


def test_query_trades_since_filter(data_dir):
    """since_ms 보다 작은 ts 는 제외."""
    code = "AICT-SINC-SINC-SINC"
    _create_user_trades(data_dir, code, [
        {"ts_ms": 1000, "event_type": "entry"},
        {"ts_ms": 5000, "event_type": "sl_hit", "pnl_usdt": -1.0},
    ])
    rows = _query_trades(_trades_db_path(data_dir, code), limit=10, since_ms=2000, event_type=None)
    assert [r["ts_ms"] for r in rows] == [5000]


def test_query_trades_event_type_filter(data_dir):
    """event_type 필터 — 정확히 일치만."""
    code = "AICT-EVTF-EVTF-EVTF"
    _create_user_trades(data_dir, code, [
        {"ts_ms": 1000, "event_type": "entry"},
        {"ts_ms": 2000, "event_type": "sl_hit", "pnl_usdt": -1.0},
        {"ts_ms": 3000, "event_type": "tp_hit", "pnl_usdt": 2.0},
    ])
    rows = _query_trades(_trades_db_path(data_dir, code), limit=10, since_ms=None, event_type="sl_hit")
    assert len(rows) == 1
    assert rows[0]["event_type"] == "sl_hit"


# ============================================================
# 2. _aggregate_stats — 청산만 PnL 집계
# ============================================================


def test_aggregate_stats_separates_wins_losses():
    rows = [
        {"event_type": "entry"},  # 진입은 PnL 집계 X
        {"event_type": "sl_hit", "pnl_usdt": -10.0},
        {"event_type": "tp_hit", "pnl_usdt": 30.0},
        {"event_type": "tp_hit", "pnl_usdt": 5.0},
    ]
    stats = _aggregate_stats(rows)
    assert stats["total_events"] == 4
    assert stats["closed_events"] == 3
    assert stats["wins"] == 2
    assert stats["losses"] == 1
    assert stats["pnl_sum_usdt"] == 25.0  # -10 + 30 + 5
    assert stats["win_rate_pct"] == round(2 / 3 * 100, 2)


def test_aggregate_stats_empty():
    stats = _aggregate_stats([])
    assert stats["total_events"] == 0
    assert stats["pnl_sum_usdt"] == 0.0
    assert stats["win_rate_pct"] == 0.0


# ============================================================
# 3. /ict/trades — 본인 거래만
# ============================================================


def test_get_my_trades_isolated_per_user(data_dir):
    """본인 user_code 의 거래만 보임 — 다른 사용자 데이터 노출 X."""
    _create_user_trades(data_dir, "AICT-MINE-MINE-MINE", [
        {"ts_ms": 1000, "event_type": "entry"},
    ])
    _create_user_trades(data_dir, "AICT-OTHE-OTHE-OTHE", [
        {"ts_ms": 2000, "event_type": "sl_hit", "pnl_usdt": -50.0},
    ])
    app = _make_app(data_dir, current_user="AICT-MINE-MINE-MINE")
    client = TestClient(app)
    r = client.get("/ict/trades")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 1
    assert data["trades"][0]["ts_ms"] == 1000


def test_get_my_trades_returns_empty_when_no_db(data_dir):
    """본인 거래 데이터 없는 신규 사용자도 200 + 빈 list."""
    app = _make_app(data_dir, current_user="AICT-NEWB-NEWB-NEWB")
    client = TestClient(app)
    r = client.get("/ict/trades")
    assert r.status_code == 200
    assert r.json() == {"trades": [], "count": 0, "stats": _aggregate_stats([])}


# ============================================================
# 4. /ict/trades/export — CSV
# ============================================================


def test_export_my_trades_csv(data_dir):
    """CSV header + row, content-type, attachment 헤더."""
    code = "AICT-CSV1-CSV1-CSV1"
    _create_user_trades(data_dir, code, [
        {"ts_ms": 1000, "event_type": "entry", "direction": "short"},
        {"ts_ms": 2000, "event_type": "sl_hit", "pnl_usdt": -10.5, "direction": "short"},
    ])
    app = _make_app(data_dir, current_user=code)
    client = TestClient(app)
    r = client.get("/ict/trades/export")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    text = r.text
    # header
    assert text.startswith("ts_ms,event_type,symbol,direction,price,qty,pnl_usdt")
    # 두 행 모두 포함
    assert "1000" in text
    assert "2000" in text
    assert "-10.5" in text


# ============================================================
# 5. /admin/trades — token 검증
# ============================================================


def test_admin_trades_without_token_503(data_dir, monkeypatch):
    """ADMIN_TOKEN env 미설정 → 503."""
    monkeypatch.delenv("AURORA_ICT_ADMIN_TOKEN", raising=False)
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.get("/admin/trades?user_code=AICT-X-X-X")
    assert r.status_code == 503


def test_admin_trades_wrong_token_401(data_dir, monkeypatch):
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.get(
        "/admin/trades?user_code=AICT-X-X-X",
        headers={"X-Admin-Token": "wrong"},
    )
    assert r.status_code == 401


def test_admin_trades_valid_token_returns_user_data(data_dir, monkeypatch):
    """admin 이 다른 사용자 데이터 조회 — 정상 200 + 격리된 user 의 row 반환."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    code = "AICT-ADMR-ADMR-ADMR"
    _create_user_trades(data_dir, code, [
        {"ts_ms": 9999, "event_type": "tp_hit", "pnl_usdt": 100.0},
    ])
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.get(
        f"/admin/trades?user_code={code}",
        headers={"X-Admin-Token": "secret-token-xyz"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["user_code"] == code
    assert data["count"] == 1
    assert data["stats"]["pnl_sum_usdt"] == 100.0


# ============================================================
# 6. /admin/trades/all_users — 통계 집계
# ============================================================


def test_admin_all_users_aggregates(data_dir, monkeypatch):
    """모든 사용자별 PnL 합산."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    _create_user_trades(data_dir, "AICT-USR1-USR1-USR1", [
        {"ts_ms": 1000, "event_type": "tp_hit", "pnl_usdt": 50.0},
    ])
    _create_user_trades(data_dir, "AICT-USR2-USR2-USR2", [
        {"ts_ms": 2000, "event_type": "sl_hit", "pnl_usdt": -30.0},
    ])
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.get(
        "/admin/trades/all_users",
        headers={"X-Admin-Token": "secret-token-xyz"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["user_count"] == 2
    assert data["total_pnl_usdt"] == 20.0  # 50 - 30
    codes = {u["code"] for u in data["users"]}
    assert codes == {"AICT-USR1-USR1-USR1", "AICT-USR2-USR2-USR2"}


# ============================================================
# 7. /admin/trades/backup — raw JSONL
# ============================================================


def test_admin_backup_jsonl(data_dir, monkeypatch):
    """JSONL raw 파일 반환 + Content-Disposition 헤더."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    code = "AICT-BKUP-BKUP-BKUP"
    _create_user_trades(data_dir, code, [
        {"ts_ms": 1000, "event_type": "entry"},
    ])
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.get(
        f"/admin/trades/backup?user_code={code}",
        headers={"X-Admin-Token": "secret-token-xyz"},
    )
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]
    assert '"event_type": "entry"' in r.text


# ============================================================
# 2026-05-29: Admin 인증 cookie + rebuild + users 목록
# ============================================================


def test_admin_login_sets_cookie(data_dir, monkeypatch):
    """/admin/login 통과 시 aurora_admin_token 쿠키 발급."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.post(
        "/admin/login",
        headers={"X-Admin-Token": "secret-token-xyz"},
    )
    assert r.status_code == 200
    assert r.json() == {"ok": True}
    # 쿠키가 세팅됨.
    assert "aurora_admin_token" in client.cookies
    assert client.cookies["aurora_admin_token"] == "secret-token-xyz"


def test_admin_login_invalid_token_401(data_dir, monkeypatch):
    """잘못된 토큰은 쿠키 발급 X."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.post(
        "/admin/login",
        headers={"X-Admin-Token": "wrong"},
    )
    assert r.status_code == 401
    assert "aurora_admin_token" not in client.cookies


def test_admin_endpoints_accept_cookie_after_login(data_dir, monkeypatch):
    """로그인 후 쿠키만으로 admin endpoint 호출 가능."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    code = "AICT-CKAY-CKAY-CKAY"
    _create_user_trades(data_dir, code, [
        {"ts_ms": 1000, "event_type": "entry"},
    ])
    app = _make_app(data_dir)
    client = TestClient(app)
    # 로그인 → 쿠키
    r = client.post("/admin/login", headers={"X-Admin-Token": "secret-token-xyz"})
    assert r.status_code == 200
    # 헤더 없이 쿠키만으로 admin endpoint 호출.
    r = client.get(f"/admin/trades?user_code={code}")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] == 1


def test_admin_session_endpoint(data_dir, monkeypatch):
    """/admin/session — 로그인 전 false, 후 true."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    app = _make_app(data_dir)
    client = TestClient(app)
    assert client.get("/admin/session").json() == {"authenticated": False}
    client.post("/admin/login", headers={"X-Admin-Token": "secret-token-xyz"})
    assert client.get("/admin/session").json() == {"authenticated": True}


def test_admin_logout_clears_cookie(data_dir, monkeypatch):
    """로그아웃 후 쿠키 제거 → admin endpoint 401."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    app = _make_app(data_dir)
    client = TestClient(app)
    client.post("/admin/login", headers={"X-Admin-Token": "secret-token-xyz"})
    r = client.post("/admin/logout")
    assert r.status_code == 200
    # 쿠키 만료 (max_age=0). httpx 는 expired 쿠키를 제거하므로 jar 에 없거나 빈 값.
    assert (
        "aurora_admin_token" not in client.cookies
        or client.cookies.get("aurora_admin_token") in (None, "", '""')
    )
    # 헤더도 없으면 401.
    r = client.get("/admin/trades?user_code=AICT-X-X-X")
    assert r.status_code == 401


def test_admin_users_list(data_dir, monkeypatch):
    """/admin/users — `<data_dir>/users/` 하위 디렉토리 목록."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    _create_user_trades(data_dir, "AICT-USR1-USR1-USR1", [
        {"ts_ms": 1000, "event_type": "entry"},
    ])
    _create_user_trades(data_dir, "AICT-USR2-USR2-USR2", [
        {"ts_ms": 2000, "event_type": "tp_hit", "pnl_usdt": 5.0},
    ])
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.get(
        "/admin/users",
        headers={"X-Admin-Token": "secret-token-xyz"},
    )
    assert r.status_code == 200
    users = r.json()["users"]
    assert "AICT-USR1-USR1-USR1" in users
    assert "AICT-USR2-USR2-USR2" in users


def test_admin_rebuild_sqlite(data_dir, monkeypatch):
    """JSONL 기준으로 SQLite 재생성 — 기존 db 비우고 jsonl 전체 insert."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    code = "AICT-RBLD-RBLD-RBLD"
    # 사용자 디렉토리 + 최소 trades.jsonl (event_type=entry 3건).
    user_dir = data_dir / "users" / code
    user_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "ts_ms": 1000 + i,
            "event_type": "entry",
            "symbol": "BTC/USDT:USDT",
            "direction": "short",
            "price": 73000.0,
            "qty": 1.0,
            "pnl_usdt": None,
            "setup_ts_ms": None,
            "reason": "",
            "context_json": None,
        }
        for i in range(3)
    ]
    with (user_dir / "trades.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    # rebuild
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.post(
        f"/admin/trades/rebuild?user_code={code}",
        headers={"X-Admin-Token": "secret-token-xyz"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["inserted_count"] == 3
    # 이후 admin/trades 조회로 확인.
    r = client.get(
        f"/admin/trades?user_code={code}",
        headers={"X-Admin-Token": "secret-token-xyz"},
    )
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_admin_rebuild_no_jsonl(data_dir, monkeypatch):
    """JSONL 부재 시 ok=False, reason=no_jsonl."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.post(
        "/admin/trades/rebuild?user_code=AICT-MISS-MISS-MISS",
        headers={"X-Admin-Token": "secret-token-xyz"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is False
    assert data["reason"] == "no_jsonl"
