"""G-3b 라이선스 티어별 settings 분기 테스트.

검증 범위:
    - IctSettings.license_type validator — 잘못된 값 → referral fallback
    - model_validator 의 정책 강제:
        * referral → disable_time_filter 사용자 값 유지 (기본 True)
        * sub_* → disable_time_filter=False 강제 (사용자 True override 무시)
    - launcher._inject_license_type_env — license.json 존재/누락별 env 박힘

CLAUDE.md mock 0 정책 — env 조작 / license.json 박기 외 외부 호출 0.
"""
from __future__ import annotations

import os
from pathlib import Path

from aurora_ict.config.settings import IctSettings
from aurora_ict_launcher import launcher, license_client

# ============================================================
# IctSettings.license_type validator + model_validator
# ============================================================


def _clean_env(monkeypatch):
    """AURORA_ICT_ prefix env 전체 제거 — 외부 .env 영향 차단."""
    for key in list(os.environ):
        if key.startswith("AURORA_ICT_"):
            monkeypatch.delenv(key, raising=False)


def test_license_type_default_is_referral(monkeypatch):
    """env 미설정 시 license_type='referral'. 2026-05-27 referral 24h(True) 복원."""
    _clean_env(monkeypatch)
    s = IctSettings(_env_file=None)
    assert s.license_type == "referral"
    # referral 은 default True (24h 매매). sub_* 만 model_validator 가 False 강제.
    assert s.disable_time_filter is True


def test_license_type_sub_30d_forces_time_filter(monkeypatch):
    """sub_30d → disable_time_filter=False 강제 (Killzone 시간만)."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_30d")
    s = IctSettings(_env_file=None)
    assert s.license_type == "sub_30d"
    assert s.disable_time_filter is False


def test_license_type_sub_90d_forces_time_filter(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_90d")
    s = IctSettings(_env_file=None)
    assert s.disable_time_filter is False


def test_license_type_sub_365d_forces_time_filter(monkeypatch):
    _clean_env(monkeypatch)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_365d")
    s = IctSettings(_env_file=None)
    assert s.disable_time_filter is False


def test_subscription_overrides_user_attempt_to_enable_24h(monkeypatch):
    """구독제 사용자가 .env 로 24h 매매 강제 시도해도 무시 — 라이선스 정책 우선."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_30d")
    monkeypatch.setenv("AURORA_ICT_DISABLE_TIME_FILTER", "true")
    s = IctSettings(_env_file=None)
    # 사용자가 True 박았지만 model_validator 가 False 로 강제
    assert s.disable_time_filter is False


def test_referral_respects_user_disable_time_filter_setting(monkeypatch):
    """레퍼럴은 정책 강제 X — 사용자가 .env 로 박은 값 그대로."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "referral")
    monkeypatch.setenv("AURORA_ICT_DISABLE_TIME_FILTER", "false")
    s = IctSettings(_env_file=None)
    # 사용자가 False 박았으면 그대로 (정책 강제 X)
    assert s.disable_time_filter is False


def test_subscription_enforces_edge_v2(monkeypatch):
    """2026-06-11 #EDGE-V2: 구독제 = 등급4 + RR2.5 + SLx3 + 대기120분 강제."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_90d")
    monkeypatch.setenv("AURORA_ICT_MIN_CONFLUENCE", "2")
    s = IctSettings(_env_file=None)
    assert s.min_confluence == 4
    assert s.min_rr == 2.5
    assert s.sl_dist_mult == 3.0
    assert s.entry_limit_ttl_sec == 7200
    assert s.disable_time_filter is False  # 킬존 유지


def test_subscription_respects_more_conservative_values(monkeypatch):
    """구독제 강제는 '최소' 방향 — 사용자가 더 보수적이면 유지."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_90d")
    monkeypatch.setenv("AURORA_ICT_MIN_CONFLUENCE", "5")
    monkeypatch.setenv("AURORA_ICT_MIN_RR", "3.0")
    monkeypatch.setenv("AURORA_ICT_SL_DIST_MULT", "4.0")
    monkeypatch.setenv("AURORA_ICT_ENTRY_LIMIT_TTL_SEC", "10800")
    s = IctSettings(_env_file=None)
    assert s.min_confluence == 5
    assert s.min_rr == 3.0
    assert s.sl_dist_mult == 4.0
    assert s.entry_limit_ttl_sec == 10800


def test_referral_keeps_user_values(monkeypatch):
    """레퍼럴은 #EDGE-V2 강제 X — 사용자/기본값 그대로."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "referral")
    monkeypatch.setenv("AURORA_ICT_MIN_CONFLUENCE", "2")
    s = IctSettings(_env_file=None)
    assert s.min_confluence == 2
    assert s.min_rr == 2.0
    assert s.sl_dist_mult == 1.0
    assert s.entry_limit_ttl_sec == 300


def test_invalid_license_type_falls_back_to_referral(monkeypatch):
    """잘못된 type 값 → referral fallback. referral default True (24h)."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_99d")
    s = IctSettings(_env_file=None)
    assert s.license_type == "referral"
    assert s.disable_time_filter is True


def test_empty_license_type_falls_back_to_referral(monkeypatch):
    """빈 문자열 → referral."""
    _clean_env(monkeypatch)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "")
    s = IctSettings(_env_file=None)
    assert s.license_type == "referral"


# ============================================================
# launcher._inject_license_type_env
# ============================================================


def test_inject_license_type_env_with_subscription(monkeypatch, tmp_path: Path):
    """license.json 에 type=sub_30d → env 에 AURORA_ICT_LICENSE_TYPE=sub_30d 박힘."""
    monkeypatch.setattr(launcher, "_aurora_ict_data_dir", lambda: tmp_path)
    license_client.save_license(tmp_path, {
        "code": "AICT-X-Y-Z",
        "type": "sub_30d",
        "license_token": "tok",
        "expires_at": "2026-06-20T00:00:00+00:00",
    })

    env = {}
    launcher._inject_license_type_env(env)
    assert env.get("AURORA_ICT_LICENSE_TYPE") == "sub_30d"


def test_inject_license_type_env_with_referral(monkeypatch, tmp_path: Path):
    """레퍼럴도 동일 박힘 — 본체 settings 가 referral 일 때 정책 강제 X 라 무해."""
    monkeypatch.setattr(launcher, "_aurora_ict_data_dir", lambda: tmp_path)
    license_client.save_license(tmp_path, {
        "code": "AICT-X-Y-Z",
        "type": "referral",
        "license_token": "tok",
        "expires_at": None,
    })

    env = {}
    launcher._inject_license_type_env(env)
    assert env.get("AURORA_ICT_LICENSE_TYPE") == "referral"


def test_inject_license_type_env_no_license_file(monkeypatch, tmp_path: Path):
    """license.json 없으면 env 안 박음 — 본체가 default (referral) 사용."""
    monkeypatch.setattr(launcher, "_aurora_ict_data_dir", lambda: tmp_path)
    env = {}
    launcher._inject_license_type_env(env)
    assert "AURORA_ICT_LICENSE_TYPE" not in env


def test_inject_license_type_env_missing_type_field(monkeypatch, tmp_path: Path):
    """license.json 있는데 type 키 없으면 env 안 박음."""
    monkeypatch.setattr(launcher, "_aurora_ict_data_dir", lambda: tmp_path)
    license_client.save_license(tmp_path, {"code": "AICT-X-Y-Z"})  # type 누락
    env = {}
    launcher._inject_license_type_env(env)
    assert "AURORA_ICT_LICENSE_TYPE" not in env


def test_inject_license_type_env_does_not_overwrite_other_env(monkeypatch, tmp_path: Path):
    """기존 env 의 다른 키들은 그대로 — license_type 만 박음 (mutation in place)."""
    monkeypatch.setattr(launcher, "_aurora_ict_data_dir", lambda: tmp_path)
    license_client.save_license(tmp_path, {
        "code": "AICT-X-Y-Z",
        "type": "sub_30d",
        "license_token": "tok",
    })

    env = {"PATH": "/usr/bin", "OTHER_VAR": "value"}
    launcher._inject_license_type_env(env)
    assert env["PATH"] == "/usr/bin"
    assert env["OTHER_VAR"] == "value"
    assert env["AURORA_ICT_LICENSE_TYPE"] == "sub_30d"
