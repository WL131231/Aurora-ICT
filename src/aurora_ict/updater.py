"""GitHub Releases 기반 자동 업데이터 — Aurora-ICT .exe 자기 갱신.

흐름 (사용자 마찰 0):
    1. **시작 시** ``apply_pending_update()`` — 직전 실행에서 다운로드된
       ``Aurora-ICT.exe.new`` 있으면 현재 exe 와 swap → 새 버전 재시작.
       (사용자는 한 번 끄면 새 버전.)
    2. **백그라운드** ``start_background_check()`` — GitHub Releases API 호출 →
       최신 tag 가 ``__version__`` 보다 높으면 ``Aurora-ICT.exe.new`` 로 다운로드
       (사용자 GUI 사용 안 막음).
    3. **다음 시작** 1번이 swap → 사용자는 GUI 다시 켰을 때 자동으로 새 버전.

플랫폼:
    - **Windows**: 같은 디렉토리 내 rename 으로 lock 우회 (PyInstaller --onefile).
    - **macOS**: ``.app`` 번들 swap 미지원 — 사용자가 release 페이지 수동 다운로드
      (Phase 후속).

환경:
    - dev/pytest (``sys.frozen`` False) → 모든 함수 no-op.
    - 네트워크 실패·API rate limit → 조용히 skip (다음 시작 때 재시도).

의존성: 표준 라이브러리만 (urllib + threading) — PyInstaller bundle 부담 0.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from aurora_ict import __version__

logger = logging.getLogger(__name__)


# ============================================================
# 설정 상수
# ============================================================

# 코드 repo (Aurora-ICT) 는 private. release artifact 는 별도 public repo
# (Aurora-ICT-releases) 로 호스팅. self-update 는 본 URL 에서 fetch.
GITHUB_API_LATEST = (
    "https://api.github.com/repos/WL131231/Aurora-ICT-releases/releases/latest"
)
HTTP_TIMEOUT_SEC = 5
DOWNLOAD_TIMEOUT_SEC = 300

# 플랫폼별 release asset 이름 (.github/workflows/release.yml 의 산출물 이름과 일치)
ASSET_NAME = {
    "Windows": "Aurora-ICT-windows.exe",
    "Darwin":  "Aurora-ICT-macOS.zip",  # 자동 swap 미지원 (수동 다운로드)
}


# ============================================================
# 내부 헬퍼
# ============================================================


def _is_frozen() -> bool:
    """PyInstaller bundle 환경 여부 — dev/pytest 에서는 항상 False."""
    return bool(getattr(sys, "frozen", False))


def _parse_version(raw: str) -> tuple[int, ...]:
    """``"v0.3.8"`` → ``(0, 3, 8)`` — semantic 비교용."""
    s = raw.lstrip("v").split("-", 1)[0]
    parts: list[int] = []
    for chunk in s.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            break
    return tuple(parts)


def _exe_path() -> Path:
    """현재 실행 중 PyInstaller .exe 의 절대 경로."""
    return Path(sys.executable).resolve()


# Windows subprocess.Popen creation flags — 부모 process 와 완전 분리.
_DETACHED_PROCESS = 0x00000008
_CREATE_NEW_PROCESS_GROUP = 0x00000200
_CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def _spawn_clean_env(exe: Path) -> None:
    """새 .exe 를 부모 PyInstaller 환경과 완전 분리해 spawn.

    Why: ``apply_pending_update`` 가 단순 spawn 하면 새 process 가 부모의
    ``_MEIPASS`` / ``_PYI_*`` env 상속 → 부모 atexit hook 의 _MEI 디렉토리 정리와
    새 process 의 numpy 등 import race 발생. 3중 격리로 방지.

    Args:
        exe: 새로 시작할 .exe 경로 (swap 직후의 정상 .exe).
    """
    clean_env = {
        k: v for k, v in os.environ.items()
        if not (k.startswith("_MEI") or k.startswith("_PYI"))
    }
    flags = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP | _CREATE_BREAKAWAY_FROM_JOB
    subprocess.Popen(  # noqa: S603 — 자기 자신 재시작, 신뢰 가능
        [str(exe)],
        env=clean_env,
        creationflags=flags,
        close_fds=True,
        cwd=str(exe.parent),
    )


# ============================================================
# Public API
# ============================================================


def apply_pending_update() -> bool:
    """직전 다운로드된 업데이트가 있으면 swap → 새 버전 재시작.

    launcher main() 가장 처음 호출. swap 성공하면 ``sys.exit(0)`` 으로 종료 후 새
    exe 실행 → 본 함수는 반환되지 않음. swap 실패·해당 없음 시 ``False`` 반환.
    """
    if not _is_frozen():
        return False
    if platform.system() != "Windows":
        return False  # macOS .app swap 미지원

    exe = _exe_path()
    new_path = exe.with_suffix(exe.suffix + ".new")
    old_path = exe.with_suffix(exe.suffix + ".old")

    if not new_path.exists():
        return False

    try:
        if old_path.exists():
            old_path.unlink()
        exe.rename(old_path)
        new_path.rename(exe)
        logger.info("auto-update applied: %s → %s (재시작)", new_path.name, exe.name)
        _spawn_clean_env(exe)
        time.sleep(0.5)
        sys.exit(0)
    except OSError as e:
        logger.warning("auto-update apply 실패 (사용자 직접 다운 권장): %s", e)
        return False


def fetch_latest_release() -> dict | None:
    """GitHub API ``/releases/latest`` 호출 — 네트워크 실패 시 ``None``."""
    try:
        req = urllib.request.Request(
            GITHUB_API_LATEST,
            headers={"Accept": "application/vnd.github+json"},
        )
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:  # noqa: S310 — https 고정
            return json.load(resp)
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as e:
        logger.debug("update check 실패 (조용히 skip): %s", e)
        return None


def is_newer(remote_tag: str, local_version: str = __version__) -> bool:
    """``remote_tag`` 가 ``local_version`` 보다 높은가."""
    try:
        return _parse_version(remote_tag) > _parse_version(local_version)
    except (ValueError, TypeError):
        return False


def download_update(asset_url: str, target: Path) -> bool:
    """asset URL → ``target`` 경로 다운로드. 실패 시 부분 파일 정리."""
    try:
        urllib.request.urlretrieve(asset_url, str(target))  # noqa: S310 — https 고정
        return True
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        logger.warning("update download 실패: %s", e)
        if target.exists():
            try:
                target.unlink()
            except OSError:
                pass
        return False


def _check_and_download_sync() -> None:
    """백그라운드 thread 의 실제 작업 — check + download 직렬 실행."""
    if not _is_frozen() or platform.system() != "Windows":
        return

    release = fetch_latest_release()
    if release is None:
        return

    tag = release.get("tag_name", "")
    if not is_newer(tag):
        logger.debug("update check: 현재 최신 (%s)", __version__)
        return

    asset_name = ASSET_NAME.get(platform.system())
    if asset_name is None:
        return

    asset_url: str | None = None
    for asset in release.get("assets", []):
        if asset.get("name") == asset_name:
            asset_url = asset.get("browser_download_url")
            break

    if asset_url is None:
        logger.debug(
            "update check: %s 새 버전 있으나 asset 미발견 (%s)",
            tag, asset_name,
        )
        return

    target = _exe_path().with_suffix(_exe_path().suffix + ".new")
    if target.exists():
        logger.info("update %s 이미 다운로드 완료 — 다음 실행 시 적용", tag)
        return

    logger.info("update %s 발견 → 백그라운드 다운로드 시작 (사용자 GUI 사용 가능)", tag)
    if download_update(asset_url, target):
        logger.info("update %s 다운로드 완료 → 다음 실행 시 자동 적용", tag)


def start_background_check() -> None:
    """백그라운드 thread 에서 update check + 다운로드 실행 — launcher main 이 호출."""
    if not _is_frozen():
        return
    t = threading.Thread(
        target=_check_and_download_sync, daemon=True, name="aurora-ict-updater",
    )
    t.start()


__all__ = [
    "ASSET_NAME",
    "GITHUB_API_LATEST",
    "apply_pending_update",
    "download_update",
    "fetch_latest_release",
    "is_newer",
    "start_background_check",
]
