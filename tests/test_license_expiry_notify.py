"""G-3a 라이선스 만료 알림 + 백그라운드 verify 단위 테스트.

검증 범위:
    - ``days_until_expiry()`` — 구독제만 계산, 레퍼럴/expires 없음 → None
    - ``_compute_expiry_warning_level()`` — days → 등급 매핑 경계
    - ``_attach_expiry_fields()`` — get_license_status result 에 필드 통합
    - ``LauncherApi.start_background_verify`` / ``stop_background_verify`` 라이프사이클

CLAUDE.md mock 0 정책 — 실 네트워크 / 실 OS 호출 X. monkeypatch + threading.Event.
"""
from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from aurora_ict_launcher import launcher, license_client
from aurora_ict_launcher.launcher import (
    LauncherApi,
    _attach_expiry_fields,
    _compute_expiry_warning_level,
)

NOW = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)


# ============================================================
# days_until_expiry
# ============================================================


def test_days_until_expiry_subscription_future():
    """구독제 + 5일 후 만료 → 5 (또는 정확히 5일 미만이면 4)."""
    payload = {
        "type": "sub_30d",
        "expires_at": (NOW + timedelta(days=5)).isoformat(),
    }
    result = license_client.days_until_expiry(payload, now=NOW)
    # timedelta.days 는 truncate — 정확히 5일이면 5, 그 미만이면 4
    assert result == 5


def test_days_until_expiry_subscription_past():
    """구독제 + 이미 만료 → 음수."""
    payload = {
        "type": "sub_30d",
        "expires_at": (NOW - timedelta(days=2)).isoformat(),
    }
    result = license_client.days_until_expiry(payload, now=NOW)
    assert result is not None
    assert result < 0


def test_days_until_expiry_referral_returns_none():
    """레퍼럴은 expires_at 없음 → 만료 개념 X → None."""
    payload = {"type": "referral", "expires_at": None}
    assert license_client.days_until_expiry(payload, now=NOW) is None


def test_days_until_expiry_unknown_type_returns_none():
    """type 누락/알 수 없는 → None."""
    assert license_client.days_until_expiry({}, now=NOW) is None
    assert license_client.days_until_expiry({"type": "unknown"}, now=NOW) is None


def test_days_until_expiry_missing_expires_at_returns_none():
    """구독제인데 expires_at 누락 → None."""
    payload = {"type": "sub_30d", "expires_at": None}
    assert license_client.days_until_expiry(payload, now=NOW) is None


# ============================================================
# _compute_expiry_warning_level
# ============================================================


def test_warning_level_none_for_none_days():
    assert _compute_expiry_warning_level(None) == "none"


def test_warning_level_none_for_far_future():
    assert _compute_expiry_warning_level(30) == "none"
    assert _compute_expiry_warning_level(4) == "none"


def test_warning_level_d3_boundary():
    """3일 / 2일 → d3."""
    assert _compute_expiry_warning_level(3) == "d3"
    assert _compute_expiry_warning_level(2) == "d3"


def test_warning_level_d1_boundary():
    """1일 → d1."""
    assert _compute_expiry_warning_level(1) == "d1"


def test_warning_level_today():
    """0일 → today (만료 당일)."""
    assert _compute_expiry_warning_level(0) == "today"


def test_warning_level_expired_negative():
    """음수 → expired."""
    assert _compute_expiry_warning_level(-1) == "expired"
    assert _compute_expiry_warning_level(-100) == "expired"


# ============================================================
# _attach_expiry_fields
# ============================================================


def test_attach_expiry_fields_with_subscription_payload():
    """구독제 payload → days/level 필드 채워짐.

    timedelta(days=2, hours=1) — datetime.now(UTC) 와 함수 내 호출 사이 마이크로
    초 흐름으로 days=1 되는 케이스 방지 (정확히 2일 이상 보장).
    """
    payload = {
        "type": "sub_30d",
        "expires_at": (datetime.now(UTC) + timedelta(days=2, hours=1)).isoformat(),
    }
    result = _attach_expiry_fields({"has_license": True}, payload)
    assert result["days_until_expiry"] == 2
    assert result["expiry_warning_level"] == "d3"


def test_attach_expiry_fields_with_none_payload():
    """payload=None (license.json 없음) → days=None, level='none'."""
    result = _attach_expiry_fields({"has_license": False}, None)
    assert result["days_until_expiry"] is None
    assert result["expiry_warning_level"] == "none"


def test_attach_expiry_fields_with_referral_payload():
    """레퍼럴 payload → days=None, level='none'."""
    payload = {"type": "referral", "expires_at": None}
    result = _attach_expiry_fields({"has_license": True}, payload)
    assert result["days_until_expiry"] is None
    assert result["expiry_warning_level"] == "none"


# ============================================================
# LauncherApi.get_license_status — expiry 필드 통합
# ============================================================


def _stub_data_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_aurora_ict_data_dir", lambda: tmp_path)


def test_get_license_status_includes_expiry_fields_for_subscription(monkeypatch, tmp_path: Path):
    """구독제 라이선스 + verify 성공 → days_until_expiry + warning_level 박힘."""
    _stub_data_dir(monkeypatch, tmp_path)

    # hours=1 여유 — datetime.now 호출 간 마이크로초 흐름 보정 (정확히 2일 보장).
    expires = datetime.now(UTC) + timedelta(days=2, hours=1)
    license_client.save_license(tmp_path, {
        "code": "AICT-AAAA-BBBB-CCCC",
        "type": "sub_30d",
        "license_token": "tok",
        "expires_at": expires.isoformat(),
        "last_verified_at": datetime.now(UTC).isoformat(),
    })
    monkeypatch.setattr(license_client, "verify_license",
                        lambda c, m, t, ctx: (200, {
                            "ok": True, "type": "sub_30d",
                            "expires_at": expires.isoformat(),
                        }))
    monkeypatch.setattr(license_client, "get_machine_id", lambda: "x" * 64)

    api = LauncherApi()
    result = api.get_license_status()

    assert "days_until_expiry" in result
    assert "expiry_warning_level" in result
    assert result["days_until_expiry"] == 2
    assert result["expiry_warning_level"] == "d3"


def test_get_license_status_no_license_includes_expiry_fields(monkeypatch, tmp_path: Path):
    """license.json 없음 → 두 필드 모두 None/'none'."""
    _stub_data_dir(monkeypatch, tmp_path)

    api = LauncherApi()
    result = api.get_license_status()
    assert result["days_until_expiry"] is None
    assert result["expiry_warning_level"] == "none"


def test_get_license_status_referral_has_none_expiry_fields(monkeypatch, tmp_path: Path):
    """레퍼럴 + verify 200 → expiry 필드 None / 'none'."""
    _stub_data_dir(monkeypatch, tmp_path)

    license_client.save_license(tmp_path, {
        "code": "AICT-X-Y-Z",
        "type": "referral",
        "license_token": "tok",
        "expires_at": None,
        "last_verified_at": datetime.now(UTC).isoformat(),
    })
    monkeypatch.setattr(license_client, "verify_license",
                        lambda c, m, t, ctx: (200, {"ok": True, "type": "referral"}))
    monkeypatch.setattr(license_client, "get_machine_id", lambda: "x" * 64)

    api = LauncherApi()
    result = api.get_license_status()
    assert result["days_until_expiry"] is None
    assert result["expiry_warning_level"] == "none"


# ============================================================
# start_background_verify / stop_background_verify
# ============================================================


def test_start_background_verify_starts_thread():
    """start 호출 후 thread 가 활성 + alive."""
    api = LauncherApi()
    assert api._license_verify_thread is None

    api.start_background_verify()
    try:
        assert api._license_verify_thread is not None
        assert api._license_verify_thread.is_alive()
    finally:
        api.stop_background_verify()
        api._license_verify_thread.join(timeout=2.0)


def test_start_background_verify_idempotent():
    """두 번 호출해도 thread 한 개만 — race 회피."""
    api = LauncherApi()
    api.start_background_verify()
    try:
        first_thread = api._license_verify_thread
        api.start_background_verify()  # 두 번째 호출
        assert api._license_verify_thread is first_thread  # 같은 thread
    finally:
        api.stop_background_verify()
        api._license_verify_thread.join(timeout=2.0)


def test_stop_background_verify_signals_thread_to_exit():
    """stop_background_verify → Event.set → thread 가 wait 깨고 즉시 종료."""
    api = LauncherApi()
    # VERIFY_INTERVAL_SEC 를 잠시 짧게 monkey patch 안 함 — Event.wait 가 즉시
    # set 되면 깨야 하므로 sleep 으로 검증 가능.
    api.start_background_verify()
    time.sleep(0.05)  # thread 가 wait 진입할 시간
    api.stop_background_verify()
    api._license_verify_thread.join(timeout=2.0)
    assert not api._license_verify_thread.is_alive()
