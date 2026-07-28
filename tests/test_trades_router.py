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
from typing import Any

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
            context_json TEXT,
            mode TEXT
        )
        """,
    )
    for e in events:
        conn.execute(
            "INSERT INTO trades(ts_ms, event_type, symbol, direction, price, "
            "qty, pnl_usdt, setup_ts_ms, reason, mode) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                e.get("mode"),
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


def _make_app(
    data_dir: Path,
    current_user: str = "AICT-TEST-USER-0001",
    *,
    seed_provider: Any = None,
) -> FastAPI:
    """trades_router 만 등록한 가벼운 FastAPI app — require_auth 는 고정 코드 반환.

    secure_cookie=False — TestClient 는 HTTP 라 Secure 쿠키가 후속 요청에
    안 실린다. 운영 SaaS 는 saas.py 에서 secure_cookie=True 로 주입.
    """
    app = FastAPI()

    async def _fake_require_auth() -> str:
        return current_user

    app.include_router(
        create_trades_router(
            data_dir, _fake_require_auth, secure_cookie=False,
            seed_provider=seed_provider,
        ),
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
    assert r.json() == {
        "trades": [],
        "count": 0,
        "stats": _aggregate_stats([]),
        "current_seed_usdt": None,
    }


def test_my_trades_includes_seed_when_provider_set(data_dir):
    """seed_provider 주입 시 current_seed_usdt 에 잔고 반영."""
    async def _seed(user_code: str) -> float:
        return 1234.56

    app = _make_app(
        data_dir, current_user="AICT-SEED-SEED-SEED", seed_provider=_seed,
    )
    client = TestClient(app)
    r = client.get("/ict/trades")
    assert r.status_code == 200
    assert r.json()["current_seed_usdt"] == 1234.56


def test_my_trades_seed_none_when_provider_raises(data_dir):
    """seed_provider 가 예외를 던져도 매매기록은 정상 + current_seed_usdt=None."""
    async def _seed_boom(user_code: str) -> float:
        raise RuntimeError("거래소 조회 실패")

    app = _make_app(
        data_dir, current_user="AICT-SEED-BOOM-0001", seed_provider=_seed_boom,
    )
    client = TestClient(app)
    r = client.get("/ict/trades")
    assert r.status_code == 200
    assert r.json()["current_seed_usdt"] is None


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
    # header — 2026-05-29: mode 컬럼 추가 (DEMO/LIVE 구분).
    assert text.startswith(
        "ts_ms,event_type,mode,model,symbol,direction,price,qty,pnl_usdt",
    )
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
# 7. /admin/trades/export-all — 전체 사용자 단일 CSV
# ============================================================


def test_admin_export_all_requires_token(data_dir, monkeypatch):
    """토큰 없으면 503/401 — 운영 데이터라 admin 게이팅 필수."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.get("/admin/trades/export-all", headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 401


def test_admin_export_all_combines_users_with_code_column(data_dir, monkeypatch):
    """모든 사용자 거래가 한 CSV — user_code 컬럼 + 각 사용자 row 포함."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    _create_user_trades(data_dir, "AICT-AAAA-AAAA-AAAA", [
        {"ts_ms": 1000, "event_type": "entry", "direction": "long"},
    ])
    _create_user_trades(data_dir, "AICT-BBBB-BBBB-BBBB", [
        {"ts_ms": 2000, "event_type": "sl_hit", "pnl_usdt": -7.5, "direction": "short"},
    ])
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.get(
        "/admin/trades/export-all",
        headers={"X-Admin-Token": "secret-token-xyz"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/csv")
    assert "trades_all_users.csv" in r.headers["content-disposition"]
    text = r.text
    # user_code 가 맨 앞 컬럼.
    assert text.startswith("user_code,ts_ms,event_type,mode,model,symbol,direction")
    assert "AICT-AAAA-AAAA-AAAA" in text
    assert "AICT-BBBB-BBBB-BBBB" in text
    assert "1000" in text
    assert "-7.5" in text


def test_admin_export_all_since_filter(data_dir, monkeypatch):
    """since_ms 보다 오래된 row 는 제외 — 기간 필터 동작."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    _create_user_trades(data_dir, "AICT-OLDN-OLDN-OLDN", [
        {"ts_ms": 1000, "event_type": "entry"},
        {"ts_ms": 9000, "event_type": "sl_hit", "pnl_usdt": -1.0},
    ])
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.get(
        "/admin/trades/export-all?since_ms=5000",
        headers={"X-Admin-Token": "secret-token-xyz"},
    )
    assert r.status_code == 200
    text = r.text
    assert "9000" in text
    # ts=1000 은 since 필터로 제외 — data row 에는 없어야.
    assert ",1000," not in text


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


# ============================================================
# 2026-05-29: Admin 라이선스 정정 endpoint
# ============================================================


def _make_app_with_auth_db(
    data_dir: Path, auth_db_path: Path,
    current_user: str = "AICT-TEST-USER-0001",
):
    """auth_db_path 주입 — admin license endpoint 활성."""
    from fastapi import FastAPI
    app = FastAPI()

    async def _fake_require_auth() -> str:
        return current_user

    app.include_router(
        create_trades_router(
            data_dir, _fake_require_auth,
            secure_cookie=False,
            auth_db_path=auth_db_path,
        ),
    )
    return app


def test_admin_update_license_success(data_dir, tmp_path, monkeypatch):
    """admin 이 사용자 라이선스 type / expires_at 갱신."""
    from aurora_ict.auth import users_db
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    auth_db = tmp_path / "users.db"
    users_db.init_db(auth_db)
    code = "AICT-LIC1-LIC1-LIC1"
    users_db.create_user(auth_db, code)  # default referral / NULL
    app = _make_app_with_auth_db(data_dir, auth_db)
    client = TestClient(app)
    r = client.post(
        "/admin/user/license",
        headers={"X-Admin-Token": "secret-token-xyz"},
        json={
            "code": code,
            "license_type": "sub_365d",
            "expires_at": "2027-05-28T23:59:59Z",
        },
    )
    assert r.status_code == 200, r.text
    # 2026-05-29: set_license idempotent — created 필드 추가 (False = UPDATE).
    body = r.json()
    assert body["ok"] is True
    assert body["created"] is False
    assert body["code"] == code
    assert body["license_type"] == "sub_365d"
    assert body["expires_at"] == "2027-05-28T23:59:59Z"
    # DB 실측.
    user = users_db.get_user_by_code(auth_db, code)
    assert user["license_type"] == "sub_365d"
    assert user["expires_at"] == "2027-05-28T23:59:59Z"


def test_admin_update_license_invalid_type_400(data_dir, tmp_path, monkeypatch):
    """허용 외 license_type → 400."""
    from aurora_ict.auth import users_db
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    auth_db = tmp_path / "users.db"
    users_db.init_db(auth_db)
    users_db.create_user(auth_db, "AICT-X-X-X")
    app = _make_app_with_auth_db(data_dir, auth_db)
    client = TestClient(app)
    r = client.post(
        "/admin/user/license",
        headers={"X-Admin-Token": "secret-token-xyz"},
        json={
            "code": "AICT-X-X-X",
            "license_type": "lifetime",  # 허용 외
            "expires_at": None,
        },
    )
    # Pydantic Field pattern 검증으로 422.
    assert r.status_code == 422


def test_admin_update_license_creates_when_user_missing(
    data_dir, tmp_path, monkeypatch,
):
    """2026-05-29 근본 fix: 사용자 row 없으면 INSERT (라이선스 봇 pre-issue 흐름).
    이전엔 404 였음 — 이제 created:True 로 신규 생성.
    """
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    auth_db = tmp_path / "users.db"
    users_db_init = __import__("aurora_ict.auth.users_db", fromlist=["init_db"])
    users_db_init.init_db(auth_db)
    app = _make_app_with_auth_db(data_dir, auth_db)
    client = TestClient(app)
    r = client.post(
        "/admin/user/license",
        headers={"X-Admin-Token": "secret-token-xyz"},
        json={
            "code": "AICT-MISS-MISS-MISS",
            "license_type": "sub_30d",
            "expires_at": "2026-06-30T23:59:59Z",
        },
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] is True  # 신규 INSERT
    assert body["license_type"] == "sub_30d"


def test_admin_update_license_no_auth_db_skipped(data_dir, tmp_path, monkeypatch):
    """auth_db_path=None 이면 endpoint 자체 미등록."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    app = _make_app(data_dir)  # auth_db_path 없이 등록
    client = TestClient(app)
    r = client.post(
        "/admin/user/license",
        headers={"X-Admin-Token": "secret-token-xyz"},
        json={
            "code": "AICT-X-X-X",
            "license_type": "sub_30d",
            "expires_at": None,
        },
    )
    assert r.status_code == 404


def test_admin_update_license_requires_admin_token(data_dir, tmp_path, monkeypatch):
    """admin 토큰 없으면 401."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-token-xyz")
    auth_db = tmp_path / "users.db"
    users_db_init = __import__("aurora_ict.auth.users_db", fromlist=["init_db"])
    users_db_init.init_db(auth_db)
    app = _make_app_with_auth_db(data_dir, auth_db)
    client = TestClient(app)
    r = client.post(
        "/admin/user/license",
        json={
            "code": "AICT-X-X-X",
            "license_type": "sub_30d",
            "expires_at": None,
        },
    )
    assert r.status_code == 401


# ============================================================
# 2026-05-29 (v2): UI 전용 별도 admin password (AURORA_ICT_ADMIN_UI_PASSWORD)
# 텔레그램 봇 호환은 ADMIN_TOKEN 유지, UI 사용자는 본인 비번으로 로그인
# ============================================================


def test_admin_login_accepts_ui_password(data_dir, monkeypatch):
    """ADMIN_UI_PASSWORD 만으로도 로그인 통과."""
    monkeypatch.delenv("AURORA_ICT_ADMIN_TOKEN", raising=False)
    monkeypatch.setenv("AURORA_ICT_ADMIN_UI_PASSWORD", "my-easy-pw-1234")
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.post(
        "/admin/login", headers={"X-Admin-Token": "my-easy-pw-1234"},
    )
    assert r.status_code == 200
    assert client.cookies["aurora_admin_token"] == "my-easy-pw-1234"


def test_admin_login_accepts_token_when_both_set(data_dir, monkeypatch):
    """ADMIN_TOKEN + ADMIN_UI_PASSWORD 둘 다 박혀있고, token 으로 로그인 통과."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "telegram-bot-token-xyz")
    monkeypatch.setenv("AURORA_ICT_ADMIN_UI_PASSWORD", "my-ui-pw")
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.post(
        "/admin/login", headers={"X-Admin-Token": "telegram-bot-token-xyz"},
    )
    assert r.status_code == 200


def test_admin_login_accepts_ui_password_when_both_set(data_dir, monkeypatch):
    """ADMIN_TOKEN + ADMIN_UI_PASSWORD 둘 다 박혀있고, ui pw 로 로그인 통과."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "telegram-bot-token-xyz")
    monkeypatch.setenv("AURORA_ICT_ADMIN_UI_PASSWORD", "my-ui-pw")
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.post(
        "/admin/login", headers={"X-Admin-Token": "my-ui-pw"},
    )
    assert r.status_code == 200


def test_admin_login_503_when_both_env_unset(data_dir, monkeypatch):
    """ADMIN_TOKEN + ADMIN_UI_PASSWORD 둘 다 미설정 → 503."""
    monkeypatch.delenv("AURORA_ICT_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("AURORA_ICT_ADMIN_UI_PASSWORD", raising=False)
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.post(
        "/admin/login", headers={"X-Admin-Token": "anything"},
    )
    assert r.status_code == 503


def test_admin_endpoints_accept_ui_password_cookie(data_dir, monkeypatch):
    """UI pw 로 로그인 후 쿠키만으로 admin endpoint 호출 가능."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_UI_PASSWORD", "my-pw-99")
    code = "AICT-UIPW-UIPW-UIPW"
    _create_user_trades(data_dir, code, [
        {"ts_ms": 1000, "event_type": "entry"},
    ])
    app = _make_app(data_dir)
    client = TestClient(app)
    client.post("/admin/login", headers={"X-Admin-Token": "my-pw-99"})
    r = client.get(f"/admin/trades?user_code={code}")
    assert r.status_code == 200
    assert r.json()["count"] == 1


def test_admin_login_rejects_wrong_value(data_dir, monkeypatch):
    """ADMIN_TOKEN 과 ADMIN_UI_PASSWORD 둘 다 박혀있어도 어느 쪽과도 다르면 401."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "secret-tok")
    monkeypatch.setenv("AURORA_ICT_ADMIN_UI_PASSWORD", "secret-pw")
    app = _make_app(data_dir)
    client = TestClient(app)
    r = client.post(
        "/admin/login", headers={"X-Admin-Token": "wrong-value"},
    )
    assert r.status_code == 401


def test_query_trades_legacy_db_without_mode_column_auto_migrates(
    data_dir,
):
    """옛 trades.db (mode 컬럼 없는 PR #166 이전 schema) 도 SELECT 성공.

    파트너 보고 2026-05-29 — admin 페이지에서 다른 사용자 조회 시 HTTP 500.
    원인: TradesStore.__init__ 의 ALTER 가 봇 가동 시점에만 실행. admin 이
    봇 비가동 사용자 조회 시 mode 컬럼 없어 OperationalError.

    Fix: _query_trades 가 SELECT 직전 idempotent ALTER 시도.
    """
    code = "AICT-LEGACY-XX-XX"
    user_dir = data_dir / "users" / code
    user_dir.mkdir(parents=True, exist_ok=True)
    db_path = user_dir / "trades.db"
    # 옛 schema — mode 컬럼 없음 (PR #166 이전).
    conn = sqlite3.connect(str(db_path))
    conn.executescript(
        """
        CREATE TABLE trades (
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
    conn.execute(
        "INSERT INTO trades(ts_ms, event_type, symbol, direction, price, qty) "
        "VALUES (1000, 'entry', 'BTC/USDT:USDT', 'short', 73000.0, 1.0)",
    )
    conn.commit()
    conn.close()

    # _query_trades 직접 호출 — mode 컬럼 없는 옛 DB 라도 빈 list 아님.
    from aurora_ict.api.trades_router import _query_trades
    rows = _query_trades(db_path, limit=100, since_ms=None, event_type=None)
    assert len(rows) == 1
    assert rows[0]["event_type"] == "entry"
    # 마이그레이션 완료된 컬럼은 NULL.
    assert rows[0]["mode"] is None


def test_query_trades_empty_db_returns_empty_list(data_dir):
    """trades 테이블 자체 없는 빈 DB 도 HTTP 500 안 내고 빈 list 반환."""
    code = "AICT-EMPTY-XX-XX"
    user_dir = data_dir / "users" / code
    user_dir.mkdir(parents=True, exist_ok=True)
    db_path = user_dir / "trades.db"
    # 빈 DB 파일만 생성 (테이블 없음).
    conn = sqlite3.connect(str(db_path))
    conn.close()

    from aurora_ict.api.trades_router import _query_trades
    rows = _query_trades(db_path, limit=100, since_ms=None, event_type=None)
    assert rows == []


# ============================================================
# 14. /admin/user/license — set_license idempotent (2026-05-29 근본 fix)
# ============================================================


def test_admin_license_creates_user_when_not_exists(
    data_dir, tmp_path, monkeypatch,
):
    """라이선스 봇이 코드 발급 시 호출 — 사용자 row 없으면 INSERT.

    파트너 보고: setup_pin 이 default referral 로 박는 버그. 근본 fix —
    라이선스 봇이 미리 /admin/user/license 호출하면 pre-insert 됨.
    """
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "tok-1")
    auth_db = tmp_path / "users.db"
    from aurora_ict.auth import users_db
    users_db.init_db(auth_db)
    app = _make_app_with_auth_db(data_dir, auth_db)
    client = TestClient(app)
    # 신규 코드 (DB 에 없음).
    r = client.post(
        "/admin/user/license",
        json={
            "code": "AICT-NEW-NEW-NEW",
            "license_type": "sub_365d",
            "expires_at": "2027-05-28T23:59:59Z",
        },
        headers={"X-Admin-Token": "tok-1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] is True
    assert body["code"] == "AICT-NEW-NEW-NEW"
    assert body["license_type"] == "sub_365d"


def test_admin_license_updates_existing_user(
    data_dir, tmp_path, monkeypatch,
):
    """기존 사용자 정정 — UPDATE 흐름."""
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "tok-2")
    # 기존 사용자 (referral 로 박혀있음).
    from aurora_ict.auth import users_db
    auth_db = tmp_path / "users.db"
    users_db.init_db(auth_db)
    users_db.create_user(auth_db, "AICT-OLD-OLD-OLD", license_type="referral")

    app = _make_app_with_auth_db(data_dir, auth_db)
    client = TestClient(app)
    r = client.post(
        "/admin/user/license",
        json={
            "code": "AICT-OLD-OLD-OLD",
            "license_type": "sub_90d",
            "expires_at": "2026-08-28T23:59:59Z",
        },
        headers={"X-Admin-Token": "tok-2"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["created"] is False  # UPDATE
    # DB 에 실제 박혀있는지 확인.
    user = users_db.get_user_by_code(auth_db, "AICT-OLD-OLD-OLD")
    assert user["license_type"] == "sub_90d"
    assert user["expires_at"] == "2026-08-28T23:59:59Z"


def test_admin_license_rejects_invalid_type(data_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("AURORA_ICT_ADMIN_TOKEN", "tok-3")
    from aurora_ict.auth import users_db
    auth_db = tmp_path / "users.db"
    users_db.init_db(auth_db)
    app = _make_app_with_auth_db(data_dir, auth_db)
    client = TestClient(app)
    r = client.post(
        "/admin/user/license",
        json={
            "code": "AICT-INV-INV-INV",
            "license_type": "premium",  # 미지원 — Pydantic Field pattern 차단.
            "expires_at": None,
        },
        headers={"X-Admin-Token": "tok-3"},
    )
    assert r.status_code == 422  # Pydantic validation


def test_admin_license_requires_admin_auth(data_dir, tmp_path):
    from aurora_ict.auth import users_db
    auth_db = tmp_path / "users.db"
    users_db.init_db(auth_db)
    app = _make_app_with_auth_db(data_dir, auth_db)
    client = TestClient(app)
    r = client.post(
        "/admin/user/license",
        json={
            "code": "AICT-XX-XX-XX",
            "license_type": "sub_365d",
            "expires_at": "2027-05-28T23:59:59Z",
        },
    )
    assert r.status_code in (401, 503)  # 인증 실패 또는 admin 비활성


# ============================================================
# path traversal 방어 (#ADMIN-PATH-TRAVERSAL)
# ============================================================


def test_safe_user_code_blocks_traversal() -> None:
    """user_code 경로 탈출 문자 차단 — path 함수 중앙 게이트.

    admin endpoint 의 user_code 는 신뢰 경계 밖이라, ../ · 구분자 · 절대경로가
    파일 경로 조립에 들어가면 디렉토리 밖 접근 위험. _safe_user_code 가 400.
    """
    from fastapi import HTTPException

    from aurora_ict.api.trades_router import _safe_user_code, _trades_db_path

    # 정상 코드(하이픈/언더스코어 포함)는 통과.
    assert _safe_user_code("AICT-PERS-PERS-PERS") == "AICT-PERS-PERS-PERS"
    assert _safe_user_code("user_01") == "user_01"

    # 경로 탈출 시도는 전부 400 (forward slash / 상위참조 / 절대경로).
    bad_codes = ["../etc", "a/b", "a/../b", "code/../../master", "/abs", "a.b/c"]
    for bad in bad_codes:
        with pytest.raises(HTTPException) as exc:
            _safe_user_code(bad)
        assert exc.value.status_code == 400

    # 윈도우 구분자(역슬래시)도 차단.
    with pytest.raises(HTTPException):
        _safe_user_code("a" + chr(92) + "b")

    # path 함수도 동일 방어 (중앙 게이트 경유).
    with pytest.raises(HTTPException):
        _trades_db_path(Path("/data"), "../../etc")


def test_attach_roi_computes_margin_pct() -> None:
    """#ROI 2026-07-28: 청산 행에 증거금 대비 % 소급 부착 — lev 해석 3단계."""
    from aurora_ict.api.trades_router import _attach_roi

    rows = [
        # ENTRY (ctx 에 leverage 15) — 매칭 소스
        dict(event_type="entry", symbol="ETH/USDT:USDT", setup_ts_ms=111,
             price=2000.0, qty=1.0, pnl_usdt=None, model="Origo 2.2",
             context_json='{"leverage": 15}'),
        # 청산 — ENTRY 매칭으로 lev 15: margin=2000*1/15=133.33, pnl 10 → +7.5%
        dict(event_type="tp_hit", symbol="ETH/USDT:USDT", setup_ts_ms=111,
             price=2000.0, qty=1.0, pnl_usdt=10.0, model="Origo 2.2",
             context_json=None),
        # 청산 — 매칭 없음 + Cursus 알트 → 기본 7x: margin=100*7/7=100, pnl -5 → -5%
        dict(event_type="sl_hit", symbol="DOGE/USDT:USDT", setup_ts_ms=None,
             price=100.0, qty=7.0, pnl_usdt=-5.0, model="Cursus 1.0",
             context_json=None),
        # 비청산 행 — None 유지
        dict(event_type="recovered", symbol="BTC/USDT:USDT", setup_ts_ms=None,
             price=60000.0, qty=1.0, pnl_usdt=None, model="Origo 2.2",
             context_json=None),
    ]
    out = _attach_roi(rows)
    assert out[1]["roi_pct"] == 7.5
    assert out[2]["roi_pct"] == -5.0
    assert out[0]["roi_pct"] is None
    assert out[3]["roi_pct"] is None
