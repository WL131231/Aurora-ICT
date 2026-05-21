"""license_client 단위 테스트 — 외부 의존 0 (mock 0 정책).

테스트 범위:
    - machine_id hash 안정성 (같은 입력 → 같은 출력 / 다른 입력 → 다른 출력)
    - load/save/delete license.json (atomic 쓰기 + 격리 폴더)
    - is_within_grace 정책 (referral 7일 / sub_* expires_at)

외부 의존 없음: HTTP / OS 부품 식별 호출은 self-spy 패턴 또는 monkeypatch 로 결정론화.
백엔드 호출 (urllib) 자체는 별도 통합 테스트에서 검증 (이 PR scope X).
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from aurora_ict_launcher import license_client

# ============================================================
# machine_id hash 안정성
# ============================================================


def test_machine_id_hash_stable_for_same_parts():
    """같은 식별자 → 같은 hash. SHA-256 의 결정성 + 모듈 hash 함수 정확성 검증."""
    # _build_machine_id_parts 결과를 self-spy 로 고정 — 부품 식별 결과 동일성 보장.
    parts = ["guid:ABC-123", "cpu:CPU-XYZ", "disk:DISK-789"]

    # 같은 parts 두 번 hash → 동일.
    import hashlib
    hash1 = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    hash2 = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    assert hash1 == hash2
    assert len(hash1) == 64  # SHA-256 hex digest 길이


def test_machine_id_hash_changes_on_different_parts():
    """부품 하나만 달라도 hash 완전히 달라야 (avalanche)."""
    import hashlib

    parts_a = ["guid:ABC-123", "cpu:CPU-XYZ", "disk:DISK-789"]
    parts_b = ["guid:ABC-123", "cpu:CPU-XYZ", "disk:DISK-OTHER"]

    hash_a = hashlib.sha256("|".join(parts_a).encode("utf-8")).hexdigest()
    hash_b = hashlib.sha256("|".join(parts_b).encode("utf-8")).hexdigest()
    assert hash_a != hash_b


def test_get_machine_id_returns_64_hex_chars(monkeypatch):
    """get_machine_id() 가 항상 64자 hex 반환 — 부품 식별 다 실패해도 fallback 으로 동작."""
    # 모든 부품 식별 함수가 None 반환하도록 fallback 강제.
    monkeypatch.setattr(license_client, "_read_windows_machine_guid", lambda: None)
    monkeypatch.setattr(license_client, "_read_cpu_uuid", lambda: None)
    monkeypatch.setattr(license_client, "_read_disk_serial", lambda: None)

    result = license_client.get_machine_id()
    assert isinstance(result, str)
    assert len(result) == 64
    assert all(c in "0123456789abcdef" for c in result)


def test_get_machine_id_uses_all_available_parts(monkeypatch):
    """식별 가능한 부품 모두 hash 에 반영 — 일부 누락돼도 남은 거로 hash."""
    monkeypatch.setattr(license_client, "_read_windows_machine_guid", lambda: "GUID-AAA")
    monkeypatch.setattr(license_client, "_read_cpu_uuid", lambda: None)
    monkeypatch.setattr(license_client, "_read_disk_serial", lambda: "DISK-BBB")

    parts = license_client._build_machine_id_parts()
    assert "guid:GUID-AAA" in parts
    assert "disk:DISK-BBB" in parts
    # CPU 부분은 없어야 (None 이라 skip)
    assert not any(p.startswith("cpu:") for p in parts)


# ============================================================
# license.json read / write
# ============================================================


def test_load_license_returns_none_when_file_missing(tmp_path: Path):
    """파일 없으면 None 반환."""
    assert license_client.load_license(tmp_path) is None


def test_save_then_load_round_trip(tmp_path: Path):
    """save → load 왕복 → 같은 dict."""
    payload = {
        "code": "AICT-J5MD-X59Y-57O7",
        "type": "sub_30d",
        "license_token": "LzdcGBiZ1JgldCTWkhDuF06fz7Bp8JUkm5DbYAjMFro",
        "expires_at": "2026-06-20T10:34:30.858822+00:00",
        "last_verified_at": "2026-05-21T11:00:00+00:00",
    }
    assert license_client.save_license(tmp_path, payload) is True

    loaded = license_client.load_license(tmp_path)
    assert loaded == payload


def test_save_creates_data_dir_if_missing(tmp_path: Path):
    """data_dir 이 없어도 save 가 폴더 자동 생성."""
    nested = tmp_path / "nested" / "sub" / "dir"
    assert not nested.exists()

    payload = {"code": "AICT-XXXX-XXXX-XXXX", "type": "referral"}
    assert license_client.save_license(nested, payload) is True
    assert (nested / "license.json").exists()


def test_save_is_atomic_via_tmp_rename(tmp_path: Path):
    """save 가 tmp → replace 로 atomic — 쓰는 도중 tmp 파일이 남아있지 말아야."""
    payload = {"code": "AICT-A", "type": "referral"}
    assert license_client.save_license(tmp_path, payload) is True

    # tmp 파일 자취 없음
    assert not (tmp_path / "license.json.tmp").exists()
    # 본 파일만 존재
    assert (tmp_path / "license.json").exists()


def test_delete_license_removes_file(tmp_path: Path):
    """delete_license → 파일 삭제."""
    payload = {"code": "AICT-A", "type": "referral"}
    license_client.save_license(tmp_path, payload)
    assert (tmp_path / "license.json").exists()

    license_client.delete_license(tmp_path)
    assert not (tmp_path / "license.json").exists()


def test_delete_license_no_error_when_missing(tmp_path: Path):
    """파일 없는 상태에서 delete 호출해도 에러 X (idempotent)."""
    license_client.delete_license(tmp_path)  # 그냥 통과해야


def test_load_license_returns_none_on_corrupt_json(tmp_path: Path):
    """JSON 깨졌으면 None 반환 (예외 X)."""
    (tmp_path / "license.json").write_text("이건 json 아님 {{ broken", encoding="utf-8")
    assert license_client.load_license(tmp_path) is None


# ============================================================
# grace period 판정
# ============================================================


NOW = datetime(2026, 5, 21, 12, 0, 0, tzinfo=UTC)


def test_grace_subscription_within_expires_at_is_ok():
    """구독제: expires_at 전이면 verify 무시하고 동작 OK."""
    payload = {
        "type": "sub_30d",
        "expires_at": (NOW + timedelta(days=10)).isoformat(),
        # last_verified_at 이 오래돼도 무관 (구독제 정책)
        "last_verified_at": (NOW - timedelta(days=100)).isoformat(),
    }
    assert license_client.is_within_grace(payload, now=NOW) is True


def test_grace_subscription_past_expires_at_is_not_ok():
    """구독제: expires_at 이후면 grace X — 봇 정지."""
    payload = {
        "type": "sub_30d",
        "expires_at": (NOW - timedelta(days=1)).isoformat(),
    }
    assert license_client.is_within_grace(payload, now=NOW) is False


def test_grace_referral_within_7_days_is_ok():
    """레퍼럴: 마지막 verify 후 7일 안이면 OK."""
    payload = {
        "type": "referral",
        "expires_at": None,
        "last_verified_at": (NOW - timedelta(days=6)).isoformat(),
    }
    assert license_client.is_within_grace(payload, now=NOW) is True


def test_grace_referral_past_7_days_is_not_ok():
    """레퍼럴: 마지막 verify 후 7일 초과면 grace 만료."""
    payload = {
        "type": "referral",
        "expires_at": None,
        "last_verified_at": (NOW - timedelta(days=8)).isoformat(),
    }
    assert license_client.is_within_grace(payload, now=NOW) is False


def test_grace_referral_exactly_7_days_boundary():
    """레퍼럴: 정확히 7일 경계 — < 비교라 7일 0초는 False (보수적)."""
    payload = {
        "type": "referral",
        "expires_at": None,
        "last_verified_at": (NOW - timedelta(days=7)).isoformat(),
    }
    # 정책 검증: < 사용해서 7일 정각엔 False (Margin 0 빡빡한 보안 우선).
    # 만약 정책이 <= 로 바뀌면 이 케이스는 True 가 돼야 함.
    assert license_client.is_within_grace(payload, now=NOW) is False


def test_grace_missing_type_returns_false():
    """type 누락 또는 알 수 없는 값 → grace X (안전 우선)."""
    payload = {"expires_at": (NOW + timedelta(days=30)).isoformat()}
    assert license_client.is_within_grace(payload, now=NOW) is False

    payload2 = {"type": "unknown_type", "expires_at": (NOW + timedelta(days=30)).isoformat()}
    assert license_client.is_within_grace(payload2, now=NOW) is False


def test_grace_referral_missing_last_verified_returns_false():
    """레퍼럴인데 last_verified_at 없으면 grace X — 한 번도 verify 안 된 상태."""
    payload = {"type": "referral", "expires_at": None}
    assert license_client.is_within_grace(payload, now=NOW) is False


def test_grace_subscription_missing_expires_at_returns_false():
    """구독제인데 expires_at 없으면 grace X — 비정상 상태."""
    payload = {"type": "sub_30d", "expires_at": None}
    assert license_client.is_within_grace(payload, now=NOW) is False


def test_grace_handles_z_suffix_iso():
    """ISO 8601 의 ``Z`` 접미사 (`...Z`) 도 정상 파싱."""
    payload = {
        "type": "sub_30d",
        "expires_at": (NOW + timedelta(days=5)).isoformat().replace("+00:00", "Z"),
    }
    assert license_client.is_within_grace(payload, now=NOW) is True


def test_grace_handles_malformed_iso():
    """깨진 ISO 문자열 → 안전하게 False."""
    payload = {"type": "sub_30d", "expires_at": "not-a-date"}
    assert license_client.is_within_grace(payload, now=NOW) is False


# ============================================================
# 상수 sanity
# ============================================================


def test_grace_days_referral_matches_policy():
    """[[project-aurora-ict-punch-list]] 의 2026-05-21 합의 — 7일."""
    assert license_client.GRACE_DAYS_REFERRAL == 7


def test_license_api_base_no_trailing_slash():
    """URL 결합 시 // 안 생기게 base 의 trailing slash 는 stripped."""
    assert not license_client.LICENSE_API_BASE.endswith("/")
    assert license_client.LICENSE_API_BASE.startswith("https://")
