"""Aurora-ICT updater — 자동 업데이트 로직 unit test."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import urllib.error

from aurora_ict import updater


# ============================================================
# _parse_version
# ============================================================


def test_parse_version_basic() -> None:
    assert updater._parse_version("v0.3.8") == (0, 3, 8)
    assert updater._parse_version("0.3.8") == (0, 3, 8)


def test_parse_version_strips_pre_release() -> None:
    """rc/beta 박힌 부분은 무시."""
    assert updater._parse_version("v0.3.8-rc1") == (0, 3, 8)


def test_parse_version_partial() -> None:
    """비표준 (1.0a) — 가능한 부분까지만 매핑."""
    assert updater._parse_version("1.0a") == (1, 0)


def test_parse_version_empty() -> None:
    assert updater._parse_version("") == ()
    assert updater._parse_version("v") == ()


# ============================================================
# is_newer
# ============================================================


def test_is_newer_higher_remote() -> None:
    assert updater.is_newer("v0.3.9", "0.3.8") is True
    assert updater.is_newer("v0.4.0", "0.3.99") is True
    assert updater.is_newer("v1.0.0", "0.99.99") is True


def test_is_newer_same_or_lower() -> None:
    assert updater.is_newer("v0.3.8", "0.3.8") is False
    assert updater.is_newer("v0.3.7", "0.3.8") is False


def test_is_newer_invalid_tag() -> None:
    """파싱 실패 — False 반환 (skip)."""
    assert updater.is_newer("garbage", "0.3.8") is False


# ============================================================
# fetch_latest_release
# ============================================================


def test_fetch_latest_release_success() -> None:
    """정상 응답 → dict 반환."""
    fake_payload = {"tag_name": "v0.3.9", "assets": []}
    mock_resp = MagicMock()
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = lambda s, *a: None
    mock_resp.read = lambda: json.dumps(fake_payload).encode()

    with patch("aurora_ict.updater.urllib.request.urlopen") as mock_open:
        mock_open.return_value.__enter__ = lambda s: mock_resp
        mock_open.return_value.__exit__ = lambda s, *a: None
        with patch("aurora_ict.updater.json.load", return_value=fake_payload):
            result = updater.fetch_latest_release()
    assert result == fake_payload


def test_fetch_latest_release_network_error() -> None:
    """URLError → None (조용히 skip)."""
    with patch(
        "aurora_ict.updater.urllib.request.urlopen",
        side_effect=urllib.error.URLError("no internet"),
    ):
        assert updater.fetch_latest_release() is None


def test_fetch_latest_release_timeout() -> None:
    with patch(
        "aurora_ict.updater.urllib.request.urlopen",
        side_effect=TimeoutError("slow"),
    ):
        assert updater.fetch_latest_release() is None


# ============================================================
# download_update
# ============================================================


def test_download_update_success(tmp_path) -> None:
    target = tmp_path / "Aurora-ICT.exe.new"

    def _fake_retrieve(url: str, dst: str) -> None:
        from pathlib import Path
        Path(dst).write_bytes(b"fake exe content")

    with patch(
        "aurora_ict.updater.urllib.request.urlretrieve",
        side_effect=_fake_retrieve,
    ):
        assert updater.download_update("http://example.com/x.exe", target) is True
    assert target.exists()


def test_download_update_cleans_partial_on_failure(tmp_path) -> None:
    """실패 시 부분 파일 정리."""
    target = tmp_path / "Aurora-ICT.exe.new"

    def _fail_after_write(url: str, dst: str) -> None:
        from pathlib import Path
        Path(dst).write_bytes(b"partial")
        raise OSError("disk full")

    with patch(
        "aurora_ict.updater.urllib.request.urlretrieve",
        side_effect=_fail_after_write,
    ):
        assert updater.download_update("http://x.com/x.exe", target) is False
    assert not target.exists()


# ============================================================
# apply_pending_update — dev 환경 (frozen=False)
# ============================================================


def test_apply_pending_update_noop_in_dev() -> None:
    """frozen=False → 즉시 False (no-op)."""
    assert updater.apply_pending_update() is False


# ============================================================
# start_background_check — dev 환경 no-op
# ============================================================


def test_start_background_check_noop_in_dev() -> None:
    """frozen=False → Thread 생성 X."""
    with patch("aurora_ict.updater.threading.Thread") as mock_t:
        updater.start_background_check()
        mock_t.assert_not_called()


# ============================================================
# ASSET_NAME 매핑
# ============================================================


def test_asset_name_mapping() -> None:
    """release.yml 의 매트릭스 산출물 이름과 일치."""
    assert updater.ASSET_NAME["Windows"] == "Aurora-ICT-windows.exe"
    assert updater.ASSET_NAME["Darwin"] == "Aurora-ICT-macOS.zip"


def test_github_api_url_points_to_releases_repo() -> None:
    """코드 repo(Aurora-ICT)가 아니라 public release repo(Aurora-ICT-releases) 가리킴."""
    assert "Aurora-ICT-releases" in updater.GITHUB_API_LATEST
    assert "releases/latest" in updater.GITHUB_API_LATEST


# ============================================================
# 통합 시나리오 — _check_and_download_sync
# ============================================================


def test_check_and_download_sync_noop_in_dev() -> None:
    """dev → 즉시 return, fetch 호출 X."""
    with patch("aurora_ict.updater.fetch_latest_release") as mock_fetch:
        updater._check_and_download_sync()
        mock_fetch.assert_not_called()


def test_check_and_download_sync_skip_when_current_is_latest() -> None:
    """frozen 가정 — 현재 버전 ≥ remote → 다운로드 안 함."""
    fake_release = {"tag_name": "v0.0.1", "assets": []}
    with patch("aurora_ict.updater._is_frozen", return_value=True):
        with patch("aurora_ict.updater.platform.system", return_value="Windows"):
            with patch(
                "aurora_ict.updater.fetch_latest_release",
                return_value=fake_release,
            ):
                with patch("aurora_ict.updater.download_update") as mock_dl:
                    updater._check_and_download_sync()
                    mock_dl.assert_not_called()


def test_check_and_download_sync_skip_when_asset_missing() -> None:
    """새 버전이지만 Windows asset 박지 않음 → 다운로드 skip."""
    fake_release = {
        "tag_name": "v999.0.0",
        "assets": [{"name": "Aurora-ICT-macOS.zip", "browser_download_url": "x"}],
    }
    with patch("aurora_ict.updater._is_frozen", return_value=True):
        with patch("aurora_ict.updater.platform.system", return_value="Windows"):
            with patch(
                "aurora_ict.updater.fetch_latest_release",
                return_value=fake_release,
            ):
                with patch("aurora_ict.updater.download_update") as mock_dl:
                    updater._check_and_download_sync()
                    mock_dl.assert_not_called()


@pytest.mark.skip(reason="frozen 상태에서 .exe.new 박힌 통합 — manual test only")
def test_check_and_download_sync_full_flow() -> None:
    """전체 flow integration — frozen 환경 박은 거 박은 거 박은 거 박은 실제 빌드에서만 검증 가능."""
