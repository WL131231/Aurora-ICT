"""LauncherApi 의 라이선스 게이트 메서드 단위 테스트 (G-2b).

검증 범위:
    - ``get_license_status()`` — license.json 없음 / verify 성공 / 4xx / 네트워크 fail + grace
    - ``redeem_code()`` — 입력 정리 (strip + upper) / 성공 시 저장 / 4xx 매핑 / 네트워크 fail
    - ``_mask_code()`` — 코드 마스킹 형식

CLAUDE.md 정책 준수: mock 0 — license_client 의 HTTP/저장 함수를 monkeypatch 로
결정론화 (외부 네트워크 / 실제 OS 부품 식별 호출 X).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from aurora_ict_launcher import launcher
from aurora_ict_launcher.launcher import LauncherApi, _mask_code

# ============================================================
# _mask_code 헬퍼
# ============================================================


def test_mask_code_typical_format():
    """AICT-XXXX-XXXX-XXXX → AICT-****-****-XXXX."""
    assert _mask_code("AICT-J5MD-X59Y-57O7") == "AICT-****-****-57O7"


def test_mask_code_empty():
    """빈 문자열 → 전체 마스킹."""
    assert _mask_code("") == "AICT-****-****-****"


def test_mask_code_no_dashes():
    """대시 없는 입력 → 마지막 그룹만 노출."""
    assert _mask_code("AICTJ5MDX59Y57O7") == "AICT-****-****-AICTJ5MDX59Y57O7"


# ============================================================
# get_license_status
# ============================================================


def _stub_data_dir(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(launcher, "_aurora_ict_data_dir", lambda: tmp_path)


def test_get_license_status_no_license_file(monkeypatch, tmp_path: Path):
    """license.json 없으면 has_license=False, verify 호출 X."""
    _stub_data_dir(monkeypatch, tmp_path)

    api = LauncherApi()
    result = api.get_license_status()

    assert result["has_license"] is False
    assert result["verify_ok"] is None
    assert result["verify_error"] is None
    assert result["grace_ok"] is False


def test_get_license_status_verify_success(monkeypatch, tmp_path: Path):
    """저장된 라이선스 있고 verify 200 OK → verify_ok=True + last_verified_at 갱신."""
    _stub_data_dir(monkeypatch, tmp_path)

    # 1) 기존 license.json 박기
    from aurora_ict_launcher import license_client
    saved = {
        "code": "AICT-J5MD-X59Y-57O7",
        "type": "sub_30d",
        "license_token": "fake-token",
        "expires_at": "2026-06-20T00:00:00+00:00",
        "last_verified_at": "2026-05-20T00:00:00+00:00",
    }
    license_client.save_license(tmp_path, saved)

    # 2) verify 200 응답 stub
    def fake_verify(code, mid, tok, ctx):
        return 200, {"ok": True, "type": "sub_30d", "expires_at": "2026-06-25T00:00:00+00:00"}

    monkeypatch.setattr(license_client, "verify_license", fake_verify)
    monkeypatch.setattr(license_client, "get_machine_id", lambda: "x" * 64)

    api = LauncherApi()
    result = api.get_license_status()

    assert result["has_license"] is True
    assert result["verify_ok"] is True
    assert result["type"] == "sub_30d"
    assert result["expires_at"] == "2026-06-25T00:00:00+00:00"  # 서버 응답으로 갱신

    # last_verified_at 이 갱신됐는지 확인 (저장 파일)
    reloaded = license_client.load_license(tmp_path)
    assert reloaded["last_verified_at"] != "2026-05-20T00:00:00+00:00"
    assert reloaded["expires_at"] == "2026-06-25T00:00:00+00:00"


def test_get_license_status_verify_4xx_expired(monkeypatch, tmp_path: Path):
    """verify 403 expired → verify_ok=False + verify_error='expired'."""
    _stub_data_dir(monkeypatch, tmp_path)

    from aurora_ict_launcher import license_client
    license_client.save_license(tmp_path, {
        "code": "AICT-AAAA-BBBB-CCCC",
        "type": "sub_30d",
        "license_token": "tok",
        "expires_at": "2026-05-01T00:00:00+00:00",
        "last_verified_at": "2026-05-19T00:00:00+00:00",
    })

    monkeypatch.setattr(license_client, "verify_license",
                        lambda c, m, t, ctx: (403, {"ok": False, "error": "expired"}))
    monkeypatch.setattr(license_client, "get_machine_id", lambda: "x" * 64)

    api = LauncherApi()
    result = api.get_license_status()

    assert result["has_license"] is True
    assert result["verify_ok"] is False
    assert result["verify_error"] == "expired"
    assert result["grace_ok"] is False


def test_get_license_status_network_fail_grace_within(monkeypatch, tmp_path: Path):
    """verify 네트워크 fail + 레퍼럴 grace 안 → verify_ok=False, grace_ok=True."""
    _stub_data_dir(monkeypatch, tmp_path)

    from aurora_ict_launcher import license_client
    now = datetime.now(UTC)
    license_client.save_license(tmp_path, {
        "code": "AICT-X-Y-Z",
        "type": "referral",
        "license_token": "tok",
        "expires_at": None,
        "last_verified_at": (now - timedelta(days=3)).isoformat(),  # 3일 전 → 7일 grace 안
    })

    monkeypatch.setattr(license_client, "verify_license",
                        lambda c, m, t, ctx: (0, {"error": "network"}))
    monkeypatch.setattr(license_client, "get_machine_id", lambda: "x" * 64)

    api = LauncherApi()
    result = api.get_license_status()

    assert result["verify_ok"] is False
    assert result["verify_error"] == "network"
    assert result["grace_ok"] is True


def test_get_license_status_network_fail_grace_expired(monkeypatch, tmp_path: Path):
    """verify 네트워크 fail + 레퍼럴 grace 만료 → grace_ok=False."""
    _stub_data_dir(monkeypatch, tmp_path)

    from aurora_ict_launcher import license_client
    now = datetime.now(UTC)
    license_client.save_license(tmp_path, {
        "code": "AICT-X-Y-Z",
        "type": "referral",
        "license_token": "tok",
        "expires_at": None,
        "last_verified_at": (now - timedelta(days=10)).isoformat(),  # 10일 전 → grace 만료
    })

    monkeypatch.setattr(license_client, "verify_license",
                        lambda c, m, t, ctx: (0, {"error": "network"}))
    monkeypatch.setattr(license_client, "get_machine_id", lambda: "x" * 64)

    api = LauncherApi()
    result = api.get_license_status()

    assert result["verify_ok"] is False
    assert result["grace_ok"] is False


def test_get_license_status_masks_code(monkeypatch, tmp_path: Path):
    """code_masked 가 마지막 4자리만 노출."""
    _stub_data_dir(monkeypatch, tmp_path)

    from aurora_ict_launcher import license_client
    license_client.save_license(tmp_path, {
        "code": "AICT-J5MD-X59Y-57O7",
        "type": "sub_30d",
        "license_token": "t",
        "expires_at": "2027-01-01T00:00:00+00:00",
        "last_verified_at": "2026-05-21T00:00:00+00:00",
    })
    monkeypatch.setattr(license_client, "verify_license",
                        lambda c, m, t, ctx: (200, {"ok": True, "type": "sub_30d",
                                                     "expires_at": "2027-01-01T00:00:00+00:00"}))
    monkeypatch.setattr(license_client, "get_machine_id", lambda: "x" * 64)

    api = LauncherApi()
    result = api.get_license_status()
    assert result["code_masked"] == "AICT-****-****-57O7"


# ============================================================
# redeem_code
# ============================================================


def test_redeem_code_empty_input_returns_error(monkeypatch, tmp_path: Path):
    """빈 문자열/공백만 입력 → ok=False + 안내 메시지."""
    _stub_data_dir(monkeypatch, tmp_path)

    api = LauncherApi()

    r1 = api.redeem_code("")
    assert r1["ok"] is False
    assert "코드를 입력" in r1["message"]

    r2 = api.redeem_code("   ")
    assert r2["ok"] is False


def test_redeem_code_strips_and_uppercases(monkeypatch, tmp_path: Path):
    """입력의 앞뒤 공백 + 소문자 → strip + upper 후 전송."""
    _stub_data_dir(monkeypatch, tmp_path)

    from aurora_ict_launcher import license_client
    captured = {}

    def fake_redeem(code, mid, ctx):
        captured["code"] = code
        return 200, {"ok": True, "type": "referral", "expires_at": None,
                     "license_token": "tok"}

    monkeypatch.setattr(license_client, "redeem_code", fake_redeem)
    monkeypatch.setattr(license_client, "get_machine_id", lambda: "m" * 64)

    api = LauncherApi()
    api.redeem_code("  aict-abcd-efgh-ijkl  ")

    assert captured["code"] == "AICT-ABCD-EFGH-IJKL"


def test_redeem_code_success_saves_license(monkeypatch, tmp_path: Path):
    """redeem 200 OK → license.json 저장 + last_verified_at 기록."""
    _stub_data_dir(monkeypatch, tmp_path)

    from aurora_ict_launcher import license_client

    monkeypatch.setattr(license_client, "redeem_code",
                        lambda c, m, ctx: (200, {
                            "ok": True, "type": "sub_30d",
                            "expires_at": "2026-06-20T00:00:00+00:00",
                            "license_token": "abc123",
                        }))
    monkeypatch.setattr(license_client, "get_machine_id", lambda: "m" * 64)

    api = LauncherApi()
    result = api.redeem_code("AICT-J5MD-X59Y-57O7")

    assert result["ok"] is True
    assert result["type"] == "sub_30d"

    saved = license_client.load_license(tmp_path)
    assert saved["code"] == "AICT-J5MD-X59Y-57O7"
    assert saved["license_token"] == "abc123"
    assert saved["last_verified_at"]  # 비어있지 않아야


def test_redeem_code_error_maps_to_message(monkeypatch, tmp_path: Path):
    """4xx error → 한국어 메시지 매핑."""
    _stub_data_dir(monkeypatch, tmp_path)

    from aurora_ict_launcher import license_client
    monkeypatch.setattr(license_client, "get_machine_id", lambda: "m" * 64)

    cases = [
        ("invalid", "존재하지 않는"),
        ("voided", "무효화된"),
        ("already_used", "다른 PC"),
    ]
    api = LauncherApi()
    for err_code, expected_msg_part in cases:
        monkeypatch.setattr(license_client, "redeem_code",
                            lambda c, m, ctx, err=err_code: (404, {"ok": False, "error": err}))
        result = api.redeem_code("AICT-X-Y-Z")
        assert result["ok"] is False
        assert expected_msg_part in result["message"], f"err_code={err_code}"


def test_redeem_code_network_fail(monkeypatch, tmp_path: Path):
    """네트워크 실패 → ok=False + 인터넷 연결 안내."""
    _stub_data_dir(monkeypatch, tmp_path)

    from aurora_ict_launcher import license_client
    monkeypatch.setattr(license_client, "redeem_code",
                        lambda c, m, ctx: (0, {"error": "network"}))
    monkeypatch.setattr(license_client, "get_machine_id", lambda: "m" * 64)

    api = LauncherApi()
    result = api.redeem_code("AICT-X-Y-Z")
    assert result["ok"] is False
    assert "네트워크" in result["message"]


def test_verify_license_now_delegates_to_get_license_status(monkeypatch, tmp_path: Path):
    """verify_license_now 가 get_license_status 와 동일 결과."""
    _stub_data_dir(monkeypatch, tmp_path)

    api = LauncherApi()
    # license.json 없는 상태에서 둘이 동일 응답
    r1 = api.get_license_status()
    r2 = api.verify_license_now()
    assert r1 == r2
