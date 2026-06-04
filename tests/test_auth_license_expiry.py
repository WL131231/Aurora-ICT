"""라이선스 만료 서버측 강제 (#LICENSE-EXPIRY) — require_auth 게이트 검증.

기존엔 require_auth / login 어디서도 ``expires_at`` 을 검증하지 않아, 구독이
만료(sub_30d/90d/365d)돼도 로그인·봇 가동이 그대로 가능했다. require_auth 에
만료 게이트를 추가해 만료 시 403 을 던지도록 한 것을 검증한다.

- referral(expires_at None) = 무기한 → 통과
- 메모리 모드(데스크탑 .exe) = 게이트 skip
- 만료 = 403, 유효 구독 = 통과
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi import HTTPException

from aurora_ict.auth import pin, users_db
from aurora_ict.auth.middleware import (
    SESSION_COOKIE_NAME,
    is_license_expired,
    require_auth,
)


def _iso(dt: datetime) -> str:
    """datetime → ISO 8601 UTC (Z 접미)."""
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class _Req:
    """require_auth 가 쓰는 request.cookies 만 흉내내는 최소 stub."""

    def __init__(self, token: str | None = None) -> None:
        self.cookies = {SESSION_COOKIE_NAME: token} if token else {}


@pytest.fixture
def db_path(tmp_path):
    """격리된 users.db — referral 사용자 1명 미리 생성."""
    path = tmp_path / "users.db"
    users_db.init_db(path)
    users_db.create_user(path, "AICT-PERS-PERS-PERS")  # referral (expires_at None)
    return path


@pytest.fixture(autouse=True)
def _reset_pin_state(db_path) -> Iterator[None]:
    pin.set_session_db_path(None)
    pin._active_sessions.clear()
    yield
    pin.set_session_db_path(None)
    pin._active_sessions.clear()


# ============================================================
# is_license_expired 순수 함수
# ============================================================


def test_none_and_empty_are_unlimited() -> None:
    assert is_license_expired(None) is False
    assert is_license_expired("") is False


def test_past_is_expired() -> None:
    assert is_license_expired(_iso(datetime.now(UTC) - timedelta(days=1))) is True


def test_future_is_valid() -> None:
    assert is_license_expired(_iso(datetime.now(UTC) + timedelta(days=1))) is False


def test_bad_format_not_blocked() -> None:
    """손상된 값으로 정상 사용자 잠그지 않음 — 만료 미적용."""
    assert is_license_expired("garbage") is False
    assert is_license_expired("2026-99-99") is False


def test_naive_datetime_treated_as_utc() -> None:
    """tz 없는 ISO 도 UTC 로 간주 (과거면 만료)."""
    assert is_license_expired("2000-01-01T00:00:00") is True


# ============================================================
# require_auth 만료 게이트
# ============================================================


@pytest.mark.asyncio
async def test_require_auth_blocks_expired_license(db_path) -> None:
    pin.set_session_db_path(db_path)
    past = _iso(datetime.now(UTC) - timedelta(days=1))
    users_db.create_user(
        db_path, "AICT-EXPI-EXPI-EXPI", license_type="sub_30d", expires_at=past,
    )
    token = pin.create_session(user_code="AICT-EXPI-EXPI-EXPI", ttl_sec=3600)
    with pytest.raises(HTTPException) as exc:
        await require_auth(_Req(token))
    assert exc.value.status_code == 403


@pytest.mark.asyncio
async def test_require_auth_allows_referral(db_path) -> None:
    pin.set_session_db_path(db_path)
    token = pin.create_session(user_code="AICT-PERS-PERS-PERS", ttl_sec=3600)
    assert await require_auth(_Req(token)) == "AICT-PERS-PERS-PERS"


@pytest.mark.asyncio
async def test_require_auth_allows_valid_subscription(db_path) -> None:
    pin.set_session_db_path(db_path)
    fut = _iso(datetime.now(UTC) + timedelta(days=10))
    users_db.create_user(
        db_path, "AICT-VALI-VALI-VALI", license_type="sub_90d", expires_at=fut,
    )
    token = pin.create_session(user_code="AICT-VALI-VALI-VALI", ttl_sec=3600)
    assert await require_auth(_Req(token)) == "AICT-VALI-VALI-VALI"


@pytest.mark.asyncio
async def test_require_auth_memory_mode_skips_gate() -> None:
    """메모리 모드(데스크탑)는 만료 게이트 자체를 적용하지 않는다."""
    pin.set_session_db_path(None)
    pin._active_sessions.clear()
    token = pin.create_session(user_code="legacy", ttl_sec=3600)
    assert await require_auth(_Req(token)) == "legacy"
