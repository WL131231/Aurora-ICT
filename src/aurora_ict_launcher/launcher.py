"""Aurora Launcher — Pywebview 미니 GUI + 본체 .exe 자동 swap + 실행.

흐름 (사용자 마찰 0):
    1. launcher 더블클릭 → GUI 시작 + 백그라운드에서 GitHub Releases /latest 체크
    2. 새 버전 있으면 다이얼로그 → 사용자 "다운로드" 클릭 → ``Aurora-ICT.exe.new``
       다운 + 본체 .exe 와 swap (본체 실행 X 상태라 race condition 없음)
    3. 사용자 "Aurora 시작" 클릭 → ``subprocess.Popen(Aurora-ICT.exe, env=...)``
       으로 본체 실행 + launcher 종료
    4. 본체 측 자기-swap 은 ``AURORA_ICT_FROM_LAUNCHER`` env 받으면 skip — 중복 방지

본체 호환:
    launcher 안 쓰는 사용자 (직접 본체 .exe 실행) 도 그대로 작동 — 본체의
    ``apply_pending_update`` / ``start_background_check`` 가 fallback.

플랫폼:
    Windows 본 구현. macOS .app 번들 swap 은 본 모듈 미지원 (Phase 3 자체).

의존성:
    표준 라이브러리 + pywebview (GUI). PyInstaller 빌드 시 numpy/pandas 등 본체
    의존성 미포함 → 미니 .exe (~10MB).
"""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import ssl
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from aurora_ict_launcher import __version__

logger = logging.getLogger(__name__)

# ============================================================
# v0.1.66: SSL context — frozen --onefile 환경에서 certifi cacert.pem 명시
# ============================================================
# Why: ChoYoon Claude #133 환기 — 사용자 (huihu) launcher.log 본문
# `[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate`.
# PyInstaller frozen 환경에서 ssl 모듈이 Windows system CA store path 깨짐 →
# urllib HTTPS 핸드셰이크 fail.
# v0.1.63 fix 2 (`--collect-data certifi`) 는 `cacert.pem` 을 bundle 에 박지만
# 코드 측 `certifi.where()` 명시 사용 X → ssl 모듈이 인지 X. 본 PR fix 본질.
try:
    import certifi
    _SSL_CONTEXT: ssl.SSLContext = ssl.create_default_context(cafile=certifi.where())
    _SSL_CTX_SOURCE: str = f"certifi {certifi.where()}"
    _SSL_IMPORT_ERROR: str | None = None
except ImportError as _ssl_err:
    # certifi 미설치 또는 PyInstaller 측 collect-data 측 X — system default fallback.
    # v0.2.18 (ChoYoon #133 P0 ⑦): Windows 사용자 측 launcher.log 측 "system default"
    # 박힘 = certifi import 실제 fail. fail 사유 측 별도 변수 박아 _setup_file_logging
    # 직후 logger 박음 (진단 자료). build_launcher.py 측 --hidden-import certifi 박음.
    _SSL_CONTEXT = ssl.create_default_context()
    _SSL_CTX_SOURCE = "system default"
    _SSL_IMPORT_ERROR = repr(_ssl_err)
except Exception as _ssl_err:  # noqa: BLE001 — certifi 측 다른 fail 측 fallback
    _SSL_CONTEXT = ssl.create_default_context()
    _SSL_CTX_SOURCE = "system default"
    _SSL_IMPORT_ERROR = repr(_ssl_err)

# ============================================================
# 설정 상수
# ============================================================

# v0.2.20: 코드 repo (Aurora) 측 private 박힘 + release artifact 측 별도 public
# repo (Aurora-ICT-releases) 측 호스팅 박음. launcher 측 본 URL 측 fetch — 사용자 측
# 코드 측 access 측 X 라도 release self-update 측 그대로 박힘.
GITHUB_API_LATEST = "https://api.github.com/repos/WL131231/Aurora-ICT-releases/releases/latest"
# v0.1.59: 5 → 15초 보강 (방화벽 / 외부 네트워크 환경에서 GitHub API 응답 5초 넘음 보고).
HTTP_TIMEOUT_SEC = 15

# v0.1.59: GitHub API 공식 정책 — 모든 요청 User-Agent 필수, 미설정 시 403 거부 가능.
# 이전엔 urllib 기본 ("Python-urllib/X.Y") 박혀 일부 환경에서 거부 → "GitHub Releases
# 조회 실패" 에러. ChoYoon Claude #133 코드 점검 verify.
LAUNCHER_USER_AGENT = f"Aurora-Launcher/{__version__}"

# v0.2.18 (ChoYoon #133 P0 ⑧, 사용자 huihu 환기): 본체 .exe 이름 측 release.yml asset
# 정합 박음. v0.1.116 측 `Aurora-ICT.exe` 측 박혀 사용자 측 manual rename workaround
# (= release asset `Aurora-ICT-windows.exe` 측 download 박은 후 미존재 fail). v0.2.18 측
# AURORA_EXE_NAME 측 platform 분기 박아 release asset 측 그대로 spawn.
# macOS: Aurora-ICT.app (별도 .zip 흐름, _body_local_target 측 처리, 본 상수 X)
# Linux: Aurora (dev fallback)
if platform.system() == "Windows":
    AURORA_EXE_NAME = "Aurora-ICT-windows.exe"
else:
    AURORA_EXE_NAME = "Aurora-ICT.exe"

# 본체 데이터 격리 폴더 이름 — v0.1.17~v0.1.21 launcher 옆 ``_aurora/``.
# v0.1.22 부터 ``%LOCALAPPDATA%\Aurora-ICT\`` (Windows 표준 hidden 위치) 로 이전.
# 본 상수는 (a) legacy migration 시 launcher 옆 ``_aurora/`` 탐색,
# (b) Windows 가 아닌 dev 환경 fallback — 두 용도로만 사용.
AURORA_DATA_DIR = "_aurora_ict"
AURORA_LOCALAPPDATA_NAME = "Aurora-ICT"

# 본체에 전달할 env 마커 — 본체 자기-swap 중복 방지
LAUNCHER_ENV_MARKER = "AURORA_ICT_FROM_LAUNCHER"
LAUNCHER_PATH_ENV = "AURORA_ICT_LAUNCHER_PATH"
"""launcher .exe 절대 경로 (frozen 모드만). 본체가 재시작 요청 시 launcher 다시
spawn 하기 위함. v0.1.43 신규 — UI 업데이트 팝업의 '재시작하기' 버튼 흐름."""

LAUNCHER_AUTO_START_ENV = "AURORA_ICT_LAUNCHER_AUTO_START"
"""launcher 가 시작 시 자동으로 START 클릭 (auto-launch 본체). 본체 /relaunch
엔드포인트가 launcher spawn 시 박음. v0.1.43 신규."""

LAUNCHER_KILL_PARENT_PID_ENV = "AURORA_ICT_KILL_PARENT_PID"
"""v0.1.61 신규 — launcher 가 시작 즉시 taskkill /F 할 본체 PID.
본체 /relaunch 엔드포인트가 새 launcher Popen 시 자기 PID 박음. launcher 는
별개 process group 이라 부모-자식 묶임 X → 본체 안 죽는 케이스 무조건 해결.
v0.1.42~v0.1.58 (8회) 본체 자기 죽이기 (os._exit / ExitProcess / cmd taskkill)
모두 일부 환경에서 실패 → 외부 launcher 가 죽이는 게 가장 robust."""

# v0.1.63: GitHub API fetch 마지막 에러 — UI / log 진단용. ChoYoon Claude #133 fix 3.
# fetch_latest_release 가 실패 시 본 변수에 박음. LauncherApi.check_update 가 UI 에 노출.
_last_fetch_error: str | None = None


# ============================================================
# v0.1.93: Launcher single-instance mutex — 중복 실행 차단
# ============================================================
# Why: 사용자 보고 (2026-05-08) — Launcher 가 두 개 실행되는 케이스. v0.1.92 측
# body 만 mutex 박힘 (body 측 중복 차단). launcher 자체 측 mutex 자체 X 라
# self-update 측 spawn-then-exit overlap / 사용자 더블클릭 / 어디 stale process
# 등 시 두 launcher window 동시 표시 가능. body mutex 와 별개 name 박음
# (launcher + body 동시 실행은 정상 흐름 — 같은 mutex 박으면 launcher 살아있는
# 동안 body spawn 차단되는 본질).

_LAUNCHER_MUTEX_HANDLE = None  # noqa: N816 — 모듈 lifetime handle 보유 (GC 방지)
_LAUNCHER_MUTEX_NAME = "Aurora-Launcher-SingleInstance-v0.1.93"
_ERROR_ALREADY_EXISTS = 183


def _acquire_launcher_single_instance_mutex() -> bool:
    """Windows named mutex — Launcher 중복 실행 차단.

    apply_pending_launcher_update() 다음에 박힘 — self-update spawn-then-exit
    측 mutex race 회피 (옛 launcher 가 mutex 잡은 채로 새 launcher spawn 하면
    새 launcher 측 ALREADY_EXISTS 즉시 exit → no launcher 사고).

    Returns:
        ``True`` — primary launcher (mutex 획득). ``False`` — duplicate (exit 권장).
    """
    global _LAUNCHER_MUTEX_HANDLE  # noqa: PLW0603
    if sys.platform != "win32":
        return True
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
    except (AttributeError, OSError, ImportError):
        return True  # ctypes 미지원 — fallback OK
    _LAUNCHER_MUTEX_HANDLE = kernel32.CreateMutexW(None, True, _LAUNCHER_MUTEX_NAME)
    last_error = kernel32.GetLastError()
    return last_error != _ERROR_ALREADY_EXISTS


# ============================================================
# 헬퍼
# ============================================================


def _is_frozen() -> bool:
    """PyInstaller bundle 환경 여부."""
    return bool(getattr(sys, "frozen", False))


def _launcher_dir() -> Path:
    """launcher .exe 가 있는 디렉토리 (본체 .exe 도 같은 폴더 가정)."""
    if _is_frozen():
        return Path(sys.executable).resolve().parent
    # dev 환경 — repo root
    return Path(__file__).resolve().parents[2]


def _aurora_ict_data_dir() -> Path:
    """본체 데이터 격리 폴더 — OS 표준 hidden 위치.

    플랫폼:
        - Windows: ``%LOCALAPPDATA%\\Aurora``
        - macOS:   ``~/Library/Application Support/Aurora`` (v0.1.67)
        - Linux 또는 LOCALAPPDATA 미설정: launcher 옆 ``_aurora/`` fallback (dev)

    Why: launcher 옆 폴더가 보이지 않도록. v0.1.67 — macOS 표준 위치 분기 박음
    (이전엔 .app 번들 안 옆 fallback → AppTranslocation 임시 위치 등 위험).
    ChoYoon Claude #133 7번째 cycle fix A.
    """
    sys_name = platform.system()
    if sys_name == "Windows":
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            return Path(local_app) / AURORA_LOCALAPPDATA_NAME
    elif sys_name == "Darwin":
        return Path.home() / "Library" / "Application Support" / AURORA_LOCALAPPDATA_NAME
    return _launcher_dir() / AURORA_DATA_DIR


def _body_artifact_name() -> str:
    """v0.1.67: 플랫폼별 GitHub release asset 이름 — release.yml 정합.

    ChoYoon Claude #133 8번째 cycle fix D — macOS launcher 가 Windows .exe 다운로드
    하던 결함 (find_aurora_exe_url Windows hardcoded → Exec format error).

    Linux = release.yml 빌드 X (사용자 영향 0) — CI/dev fallback 으로 Windows asset.
    """
    if platform.system() == "Darwin":
        return "Aurora-ICT-macOS.zip"  # release.yml L51 정합
    return "Aurora-ICT-windows.exe"  # Windows + Linux fallback


def _body_local_target() -> Path:
    """v0.1.67 + v0.2.18: 플랫폼별 로컬 본체 path. release asset 이름 정합.

    macOS:   Aurora-ICT.app (zip 풀어서 박힘)
    Windows: Aurora-ICT-windows.exe (= release asset 그대로, AURORA_EXE_NAME 측 동일)
    Linux:   Aurora-ICT.exe (dev fallback)
    """
    data_dir = _aurora_ict_data_dir()
    if platform.system() == "Darwin":
        return data_dir / "Aurora-ICT.app"
    return data_dir / AURORA_EXE_NAME


def _aurora_exe_path() -> Path:
    """본체 절대 경로 — 격리 폴더 안. 플랫폼별 (.exe / .app).

    v0.1.67: 단순 wrapper — 호출자 호환성 위해 유지. 신규 코드는 _body_local_target() 직접.
    """
    return _body_local_target()


def _migrate_legacy_layout() -> None:
    """legacy layout → 현재 격리 폴더로 자동 이전 (best-effort).

    두 단계 마이그레이션:
        1. v0.1.16 이전: ``<launcher>/Aurora-ICT.exe`` → 격리 폴더
        2. v0.1.21 이전: ``<launcher>/_aurora/`` 폴더 통째 → ``%LOCALAPPDATA%\\Aurora-ICT\\``

    실패해도 launcher 시작 차단 X (재다운으로 복구 가능).
    """
    new_data_dir = _aurora_ict_data_dir()
    legacy_data_dir = _launcher_dir() / AURORA_DATA_DIR  # <launcher>/_aurora/

    # ── 1단계: legacy ``<launcher>/_aurora/`` 폴더 → 새 위치 (v0.1.22 이전).
    # 새 위치와 legacy 가 같으면 (Windows 아닌 fallback) skip.
    if (
        legacy_data_dir.exists()
        and legacy_data_dir.resolve() != new_data_dir.resolve()
        and not new_data_dir.exists()
    ):
        try:
            new_data_dir.parent.mkdir(parents=True, exist_ok=True)
            new_data_dir.mkdir(parents=True, exist_ok=True)
            for item in legacy_data_dir.iterdir():
                target = new_data_dir / item.name
                try:
                    item.rename(target)
                except OSError as e:
                    logger.warning("legacy 파일 이전 실패 (%s): %s", item.name, e)
            try:
                legacy_data_dir.rmdir()  # 빈 폴더면 정리
            except OSError:
                pass  # 일부 파일 남아 있으면 그냥 둠
            logger.info("legacy %s → %s 이전 완료", legacy_data_dir, new_data_dir)
        except OSError as e:
            logger.warning("legacy data_dir 이전 실패: %s", e)

    # ── 2단계: launcher 옆 잔재 (v0.1.16 이전) → 새 위치.
    legacy_exe = _launcher_dir() / AURORA_EXE_NAME
    new_exe = _aurora_exe_path()
    if legacy_exe.exists() and not new_exe.exists():
        try:
            new_exe.parent.mkdir(parents=True, exist_ok=True)
            legacy_exe.rename(new_exe)
            logger.info("legacy %s → %s 이전 완료", legacy_exe, new_exe)
        except OSError as e:
            logger.warning("legacy exe 이전 실패 (재다운 필요): %s", e)
    # ── 3단계 (v0.2.18, ChoYoon #133 P0 ⑧): Windows 측 v0.1.116 까지 박혔던
    # `Aurora-ICT.exe` 측 v0.2.18 측 `Aurora-ICT-windows.exe` 측 rename 박음.
    # release asset 이름 정합 박은 본질 = 사용자 huihu 측 manual rename workaround
    # 박은 거 측 자동 처리. 측 새 본체 측 release 측 download 박힐 때 .exe.new
    # 측 정합 — 본 step 측 사용자 측 v0.1.116 까지 박힌 본체 측 보존.
    if platform.system() == "Windows":
        legacy_v0116_exe = new_data_dir / "Aurora-ICT.exe"
        new_v0218_exe = new_data_dir / "Aurora-ICT-windows.exe"
        if legacy_v0116_exe.exists() and not new_v0218_exe.exists():
            try:
                legacy_v0116_exe.rename(new_v0218_exe)
                logger.info(
                    "v0.2.18 본체 이름 자동 migration: %s → %s",
                    legacy_v0116_exe, new_v0218_exe,
                )
            except OSError as e:
                logger.warning("v0.2.18 본체 rename 실패 (재다운 필요): %s", e)
    # launcher 옆 잔재 (.new / .old / .aurora_ict_version) 정리.
    # v0.1.24: launcher.exe.old 는 swap 직후 unlink 되지만 release 잔재 가능 → 정리.
    # legacy launcher.exe.new 는 apply_pending_launcher_update 가 호환 처리하므로
    # 여기선 unlink X (swap 흐름 방해 방지).
    for name in (
        "Aurora-ICT.exe.new", "Aurora-ICT.exe.old", ".aurora_ict_version",
        "Aurora-ICT-windows.exe.new", "Aurora-ICT-windows.exe.old",
        "Aurora-ICT-launcher.exe.old",
    ):
        legacy = _launcher_dir() / name
        if legacy.exists():
            try:
                legacy.unlink()
            except OSError:
                pass


def _parse_version(raw: str) -> tuple[int, ...]:
    """``"v0.1.10"`` → ``(0, 1, 10)``."""
    s = raw.lstrip("v").split("-", 1)[0]
    parts: list[int] = []
    for chunk in s.split("."):
        try:
            parts.append(int(chunk))
        except ValueError:
            break
    return tuple(parts)


# ============================================================
# GitHub API + 다운로드
# ============================================================


def fetch_latest_release() -> dict | None:
    """GitHub Releases /latest 호출 — 네트워크 실패 시 None.

    v0.1.59: User-Agent 헤더 필수 (GitHub API 정책) + HTTPError 명시 catch +
    INFO/WARNING 로그 (이전엔 debug silently skip → 사용자 진단 불가).
    v0.1.63: 실패 시 ``_last_fetch_error`` 전역 변수에 박음 — UI 노출 + 진단.
    """
    global _last_fetch_error  # noqa: PLW0603 — 진단 채널
    try:
        req = urllib.request.Request(
            GITHUB_API_LATEST,
            headers={
                "Accept": "application/vnd.github+json",
                "User-Agent": LAUNCHER_USER_AGENT,
            },
        )
        with urllib.request.urlopen(  # noqa: S310 — User-Agent + SSL context 박힘
            req, timeout=HTTP_TIMEOUT_SEC, context=_SSL_CONTEXT,
        ) as resp:
            data = json.load(resp)
        _last_fetch_error = None  # 성공 시 reset
        return data
    except urllib.error.HTTPError as e:
        # 403 (rate limit / User-Agent reject) / 404 등 — 명시 로그
        msg = f"HTTP {e.code} {e.reason}"
        logger.warning(
            "update check %s — GitHub API 거부 (User-Agent / rate limit / proxy)", msg,
        )
        _last_fetch_error = msg
        return None
    except urllib.error.URLError as e:
        # DNS / 방화벽 / SSL / 연결 차단 — 명시 로그
        msg = f"URLError: {e.reason}"
        logger.warning("update check 네트워크 실패: %s", e.reason)
        _last_fetch_error = msg
        return None
    except (json.JSONDecodeError, TimeoutError) as e:
        msg = f"{type(e).__name__}: {e}"
        logger.warning("update check 응답 파싱/타임아웃 실패: %s", msg)
        _last_fetch_error = msg
        return None


def find_aurora_exe_url(release: dict) -> str | None:
    """release assets 에서 본체 URL 반환 — 플랫폼별 asset 이름 (v0.1.67).

    Windows = Aurora-ICT-windows.exe / macOS = Aurora-ICT-macOS.zip.
    ChoYoon Claude #133 fix D — 이전엔 Windows hardcoded 라 macOS launcher 가
    Windows binary 다운로드 → Exec format error → "본체 .exe 미존재" misleading.
    """
    target_name = _body_artifact_name()
    for asset in release.get("assets", []):
        if asset.get("name") == target_name:
            url = asset.get("browser_download_url")
            return str(url) if url else None
    return None


def find_launcher_url(release: dict) -> str | None:
    """release assets 에서 launcher URL 반환 — 플랫폼별 asset name (v0.1.109).

    Windows: ``Aurora-ICT-launcher.exe``
    macOS:   ``Aurora-ICT-launcher-macOS.zip`` (release.yml 측 ditto -ck 박은 .app 번들)
    """
    sys_name = platform.system()
    if sys_name == "Darwin":
        target_name = "Aurora-ICT-launcher-macOS.zip"
    else:
        target_name = "Aurora-ICT-launcher.exe"
    for asset in release.get("assets", []):
        if asset.get("name") == target_name:
            url = asset.get("browser_download_url")
            return str(url) if url else None
    return None


def download_to(url: str, target: Path) -> bool:
    """url → target 경로 다운로드. 실패 시 부분 파일 정리.

    v0.1.59: urlretrieve 대신 명시 Request + urlopen — User-Agent 헤더 박음.
    v0.1.66: SSL context (certifi cacert) + timeout 명시 — frozen 환경 SSL 핸드셰이크
    + 무한 hang 방지. ChoYoon Claude #133 fix B.
    """
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": LAUNCHER_USER_AGENT},
        )
        with (
            urllib.request.urlopen(  # noqa: S310 — User-Agent + SSL + timeout 박힘
                req, timeout=HTTP_TIMEOUT_SEC, context=_SSL_CONTEXT,
            ) as resp,
            target.open("wb") as f,
        ):
            shutil.copyfileobj(resp, f)
        return True
    except urllib.error.HTTPError as e:
        logger.warning("download HTTP %d: %s (%s)", e.code, e.reason, url)
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        logger.warning("download 실패: %s", e)
    if target.exists():
        try:
            target.unlink()
        except OSError:
            pass
    return False


def get_local_aurora_ict_version() -> str | None:
    """본체 .exe 의 버전 추정 — Aurora-ICT.exe 옆 .version 파일 또는 모름.

    PyInstaller 빌드된 .exe 는 외부에서 버전 추출 어려움. 본체가 시작 시 .version
    파일 작성하는 패턴 (별도 PR) 또는 처음에는 None 반환.
    None 일 때는 launcher 가 무조건 최신 버전 다운 권유.
    """
    version_file = _aurora_ict_data_dir() / ".aurora_ict_version"
    if version_file.exists():
        try:
            return version_file.read_text(encoding="utf-8").strip()
        except OSError:
            return None
    return None


# ============================================================
# Swap (본체 .exe 갱신)
# ============================================================


# ============================================================
# Launcher self-update (v0.1.19) — PR #71 본체 swap race fix 패턴 차용
# ============================================================


def _launcher_exe_path() -> Path:
    """현재 실행 중 Aurora-ICT-launcher.exe 경로 (frozen 환경)."""
    return Path(sys.executable).resolve()


def _launcher_new_path() -> Path:
    """다운된 launcher.new 임시 위치 — 격리 폴더 안 (v0.1.24).

    v0.1.23 이전: launcher 옆 ``Aurora-ICT-launcher.exe.new``  ← 사용자 눈에 보임
    v0.1.24~:    ``%LOCALAPPDATA%\\Aurora-ICT\\Aurora-ICT-launcher.exe.new``  ← 숨김
    v0.1.109:    macOS 측 ``Aurora-ICT-launcher-macOS.zip.new`` 박음 (ChoYoon 권장 c).
    """
    if platform.system() == "Darwin":
        return _aurora_ict_data_dir() / "Aurora-ICT-launcher-macOS.zip.new"
    return _aurora_ict_data_dir() / "Aurora-ICT-launcher.exe.new"


def apply_pending_launcher_update() -> bool:
    """직전 다운된 launcher.new 가 있으면 swap → 새 launcher 재시작 (race fix).

    main() 가장 처음 호출. swap 시 race condition 회피를 위해 PR #71 의
    _spawn_clean_env 패턴 차용 (env 정리 + DETACHED + CREATE_BREAKAWAY_FROM_JOB).

    v0.1.24: ``.new`` 가 LocalAppData 격리 폴더에 있을 수도, legacy (launcher 옆) 일 수도.
    호환성 위해 두 위치 모두 체크 — 어느쪽이든 발견 시 swap.

    v0.1.109: macOS 분기 박음 (ChoYoon 권장 c) — .zip 측 ditto unzip + .app
    swap + ``open -n`` spawn. 이전 측 Windows only 박혀있어 macOS 측 self-update
    자체 X.

    Returns:
        True (실제로는 도달 X — sys.exit). False — 해당 없음 / 실패.
    """
    if not _is_frozen():
        return False
    sys_name = platform.system()
    if sys_name == "Darwin":
        return _apply_pending_launcher_update_macos()
    if sys_name != "Windows":
        return False

    exe = _launcher_exe_path()
    new_path = _launcher_new_path()                          # LocalAppData (v0.1.24~)
    legacy_new = exe.with_suffix(exe.suffix + ".new")        # launcher 옆 (v0.1.23 이전 호환)
    old_path = exe.with_suffix(exe.suffix + ".old")

    # v0.1.24 위치 우선, 없으면 legacy 위치 (마이그레이션 호환).
    src: Path | None = None
    if new_path.exists():
        src = new_path
    elif legacy_new.exists():
        src = legacy_new
    if src is None:
        return False

    try:
        if old_path.exists():
            old_path.unlink()
        exe.rename(old_path)
        # ``.new`` 가 다른 볼륨 (LocalAppData = C:, launcher.exe = D: 가능) 일 수 있음 →
        # ``rename()`` 대신 ``shutil.move()`` 사용. 같은 볼륨이면 rename 으로 fast path.
        shutil.move(str(src), str(exe))
        logger.info("launcher self-update applied: %s 재시작", exe.name)

        # 새 launcher 분리 spawn (race fix — PR #71 패턴)
        _DETACHED_PROCESS = 0x00000008  # noqa: N806
        _CREATE_NEW_PROCESS_GROUP = 0x00000200  # noqa: N806
        _CREATE_BREAKAWAY_FROM_JOB = 0x01000000  # noqa: N806
        clean_env = {
            k: v for k, v in os.environ.items()
            if not (k.startswith("_MEI") or k.startswith("_PYI"))
        }
        flags = _DETACHED_PROCESS | _CREATE_NEW_PROCESS_GROUP | _CREATE_BREAKAWAY_FROM_JOB
        subprocess.Popen(  # noqa: S603 — 자기 재시작
            [str(exe)],
            env=clean_env,
            creationflags=flags,
            close_fds=True,
            cwd=str(exe.parent),
        )
        time.sleep(0.5)  # 새 process _MEI 풀 시간 확보
        sys.exit(0)
    except OSError as e:
        logger.warning("launcher self-update apply 실패: %s", e)
        return False


def _apply_pending_launcher_update_macos() -> bool:
    """v0.1.109: macOS 측 launcher self-update — .zip 측 .app 번들 swap.

    흐름:
        1. _launcher_new_path() = ``%LAD%/Aurora-ICT-launcher-macOS.zip.new``
        2. (v0.2.22) metadata file 측 version 비교 — 자기 측 ≥ pending 측 단순 삭제
        3. ditto -x -k 측 zip 풀어 임시 디렉토리 박음
        4. 옛 .app 측 .app.old 박음 + 새 .app 측 위치 박음
        5. ``open -n`` 측 새 .app 분리 spawn + sys.exit(0)

    v0.2.22 (ChoYoon #133 18 cycle P1 ②): 사용자 측 manual 측 새 version 다운 박은
    후 첫 실행 시점 측 PyObjC race 회피. self_version >= pending_version 측 단순
    .zip.new + metadata 측 삭제 박음 (swap 의무 X).

    sys.executable 측 frozen .app 측 ``.../Aurora-ICT-launcher.app/Contents/MacOS/Aurora-ICT-launcher``.
    parents[2] 측 .app 번들.
    """
    src_zip = _launcher_new_path()
    if not src_zip.exists():
        return False

    # v0.2.22: metadata file 측 version 비교
    metadata = src_zip.with_suffix(".new.version")  # Aurora-ICT-launcher-macOS.zip.new.version
    if metadata.exists():
        try:
            pending_v = metadata.read_text(encoding="utf-8").strip().lstrip("v")
            if _parse_version(pending_v) <= _parse_version(__version__):
                logger.info(
                    "pending launcher v%s 측 자기 측 v%s 정합 ↑ → 단순 삭제 (PyObjC race 회피)",
                    pending_v, __version__,
                )
                try:
                    src_zip.unlink(missing_ok=True)
                    metadata.unlink(missing_ok=True)
                except OSError as e:
                    logger.warning("pending launcher cleanup 실패: %s", e)
                return False
        except (OSError, ValueError, TypeError) as e:
            logger.debug("pending version 비교 실패 (계속 swap): %s", e)
    try:
        app_bundle = Path(sys.executable).resolve().parents[2]
    except IndexError:
        logger.warning("macOS launcher .app 번들 위치 측 X — sys.executable=%s", sys.executable)
        return False
    if app_bundle.suffix != ".app":
        logger.warning("macOS launcher 측 .app 번들 X (frozen 환경 변형): %s", app_bundle)
        return False

    tmp_dir = src_zip.parent / "_launcher_swap_tmp"
    try:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        # ditto -x -k = zip extract (macOS 표준, .app metadata 보존)
        r = subprocess.run(  # noqa: S603, S607
            ["ditto", "-x", "-k", str(src_zip), str(tmp_dir)],
            capture_output=True, timeout=60, check=False,
        )
        if r.returncode != 0:
            err = r.stderr.decode("utf-8", errors="replace").strip()
            logger.warning("ditto unzip 실패 (rc=%d): %s", r.returncode, err)
            return False
        # 새 .app 찾기
        new_app = next(tmp_dir.glob("*.app"), None)
        if new_app is None:
            logger.warning("zip 안 측 .app 번들 X")
            return False
        # 옛 .app → .old
        old_app = app_bundle.with_suffix(".app.old")
        if old_app.exists():
            shutil.rmtree(old_app)
        shutil.move(str(app_bundle), str(old_app))
        # 새 .app 측 옛 위치 박음
        shutil.move(str(new_app), str(app_bundle))
        # 정리
        try:
            src_zip.unlink()
            shutil.rmtree(tmp_dir)
        except OSError:
            pass
        logger.info("macOS launcher self-update applied: %s 재시작", app_bundle.name)
        # 새 .app 측 ``open -n`` 분리 spawn (Finder 표준 lifecycle)
        subprocess.Popen(  # noqa: S603, S607 — 자기 재시작
            ["open", "-n", str(app_bundle)],
            close_fds=True,
        )
        time.sleep(0.5)
        sys.exit(0)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("macOS launcher self-update 실패: %s", e)
        # tmp 정리 (실패 시)
        if tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir)
            except OSError:
                pass
        return False


def _check_and_download_launcher_update() -> None:
    """백그라운드 thread — launcher 자기 update check + 다운로드 (다음 시작 시 apply).

    v0.1.109: macOS 분기 박음 — 이전 측 Windows only 박혀있어 macOS 측 launcher
    self-update 자체 X. ChoYoon 권장 (c).
    """
    if not _is_frozen():
        return
    if platform.system() not in ("Windows", "Darwin"):
        return
    release = fetch_latest_release()
    if release is None:
        return
    tag = release.get("tag_name", "")
    try:
        if _parse_version(tag) <= _parse_version(__version__):
            return  # 현재 launcher 가 최신 또는 더 높음
    except (ValueError, TypeError):
        return
    url = find_launcher_url(release)
    if url is None:
        return
    # v0.1.24: launcher 옆 X → 격리 폴더 (LocalAppData) 에 다운로드 (사용자 눈에 안 보임)
    target = _launcher_new_path()
    if target.exists():
        logger.info("launcher %s 이미 다운 완료 — 다음 시작 시 적용", tag)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    logger.info("launcher %s 발견 → 백그라운드 다운로드 (%s)", tag, target)
    if download_to(url, target):
        logger.info("launcher %s 다운 완료 → 다음 시작 시 자동 swap", tag)
        # v0.2.22 (ChoYoon #133 18 cycle P1 ②): metadata file 측 version 박음.
        # 다음 launcher 시작 시점 측 _apply_pending_launcher_update_macos 측 measured
        # version 비교 박아 PyObjC race 회피 박음 (자기 ≥ pending 측 단순 삭제).
        try:
            metadata = target.with_suffix(".new.version")
            metadata.write_text(tag, encoding="utf-8")
            logger.debug("pending launcher metadata 박음: %s = %s", metadata, tag)
        except OSError as e:
            logger.warning("pending launcher metadata 박음 실패 (계속 진행): %s", e)


def start_background_launcher_check() -> None:
    """launcher 시작 시 백그라운드 자기 update check thread 띄우기."""
    if not _is_frozen():
        return
    t = threading.Thread(
        target=_check_and_download_launcher_update,
        daemon=True,
        name="launcher-self-updater",
    )
    t.start()


def apply_swap(downloaded_new: Path) -> bool:
    """다운로드된 .new → 본체와 swap. 플랫폼별 흐름 (v0.1.67).

    Windows: ``Aurora-ICT.exe.new`` → ``Aurora-ICT.exe`` 단순 rename.
    macOS:   ``Aurora-ICT-macOS.zip`` 풀어서 ``Aurora-ICT.app`` 박음 (zip 본질).

    본체 실행 X 상태라 lock 없음 → 안전 swap.

    ChoYoon Claude #133 fix E — macOS asset 은 zip 형식이라 rename 만으론 부족.
    """
    target = _body_local_target()
    sys_name = platform.system()

    # macOS — zip 풀음 흐름. v0.1.73: suffix 검사 → is_zipfile 박음
    # (download target = ``.zip.new`` 본질 fix H 정합 — suffix=".new" 라 옛 분기
    # SKIP 됐던 결함 회피). zipfile.extractall → ditto -x -k 로 변경 (.app 번들의
    # symlink / xattr / executable bit metadata 보존). ChoYoon #133 fix I.
    if sys_name == "Darwin":
        import zipfile
        if not zipfile.is_zipfile(downloaded_new):
            logger.warning("macOS swap: zip 형식 X — %s", downloaded_new)
            return False
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # 기존 .app 제거 (깔끔 swap)
            if target.exists():
                shutil.rmtree(target)
            # ditto -x -k = macOS 표준 zip extract. .app 번들 metadata 보존
            # (symlinks / xattr / resource forks / executable bit). 별도 의존성 X.
            result = subprocess.run(  # noqa: S603, S607
                ["ditto", "-x", "-k", str(downloaded_new), str(target.parent)],
                capture_output=True, timeout=60, check=False,
            )
            if result.returncode != 0:
                err = result.stderr.decode("utf-8", errors="replace")
                logger.error(
                    "ditto -xk 실패 (rc=%d): %s", result.returncode, err.strip(),
                )
                return False
            downloaded_new.unlink()
            # quarantine xattr 정리 — urllib 다운은 quarantine X 박지만 안전망
            subprocess.run(  # noqa: S603, S607
                ["xattr", "-cr", str(target)],
                capture_output=True, timeout=10, check=False,
            )
            # .app 디렉토리 +x — Finder 진입 가능
            try:
                target.chmod(0o755)
            except OSError:
                pass
            logger.info("macOS .app swap 완료 (ditto): %s", target)
            return True
        except (OSError, subprocess.TimeoutExpired) as e:
            logger.warning("macOS swap 실패: %s", e)
            return False

    # Windows / 기타 — rename 흐름 (기존)
    old_path = target.with_suffix(target.suffix + ".old")
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if old_path.exists():
            old_path.unlink()
        if target.exists():
            target.rename(old_path)
        downloaded_new.rename(target)
        if old_path.exists():
            try:
                old_path.unlink()
            except OSError:
                pass
        return True
    except OSError as e:
        logger.warning("swap 실패: %s", e)
        return False


# ============================================================
# 본체 실행
# ============================================================


def _kill_other_aurora_processes() -> None:
    """v0.1.100: launcher 시작 시 다른 Aurora 측 process 강제 정리.

    사용자 보고 (2026-05-08): launcher 2개 + body 3개 동시 박힘 → 모두 죽고
    재시작 cascade. mutex 측 timing race 측 100% 못 잡힘 + v0.1.99 auto-respawn
    측 cascade 키움. 시작 시 강제 정리 박아 단일 launcher + 단일 body 보장.

    자기 PID + 부모 PID 제외 — v0.1.102 fix:
    - self-update spawn 시 옛 launcher = 부모, 새 launcher = 자식
    - 새 launcher 측 옛 launcher 측 ``taskkill /F /T`` 박으면 tree-kill 측
      자식 (= 자기 자신) 도 같이 박힘 → 사용자 보고 \"런처 누르면 안 나옴\"
    - 부모 PID 측 skip 박아 self-tree-kill 회피. 옛 launcher 측 sys.exit(0)
      자체로 곧 죽음 (kill 안 해도 OK).
    """
    if platform.system() != "Windows":
        return
    my_pid = os.getpid()
    # v0.1.102: 부모 PID 측 self-update chain 측 \"새 launcher 측 옛 launcher
    # 측 tree-kill 측 자기도 죽이는\" 본질 차단. os.getppid 측 PyInstaller
    # frozen 환경 측 정상 박힘.
    try:
        my_parent_pid = os.getppid()
    except OSError:
        my_parent_pid = -1
    try:
        for image_name in ("Aurora-ICT.exe", "Aurora-ICT-launcher.exe"):
            # tasklist 측 PID 받아 자기 PID 제외 + taskkill
            r = subprocess.run(  # noqa: S603, S607
                ["tasklist", "/FI", f"IMAGENAME eq {image_name}", "/FO", "CSV", "/NH"],
                capture_output=True, timeout=5, check=False,
                creationflags=0x08000000,  # CREATE_NO_WINDOW (cmd flash 차단)
            )
            if r.returncode != 0:
                continue
            text = r.stdout.decode("cp949", errors="replace")
            for line in text.splitlines():
                if not line.strip():
                    continue
                # CSV: "Aurora-ICT.exe","12345","Console","1","100,000 K"
                parts = [p.strip().strip('"') for p in line.split(",")]
                if len(parts) < 2:
                    continue
                try:
                    pid = int(parts[1])
                except ValueError:
                    continue
                if pid == my_pid:
                    continue
                if pid == my_parent_pid:
                    logger.info(
                        "v0.1.102 startup nuke: %s PID=%d 측 부모 PID — skip "
                        "(tree-kill 측 self 차단)", image_name, pid,
                    )
                    continue
                logger.info(
                    "v0.1.100 startup nuke: %s PID=%d kill",
                    image_name, pid,
                )
                try:
                    subprocess.run(  # noqa: S603, S607
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True, timeout=5, check=False,
                        creationflags=0x08000000,  # CREATE_NO_WINDOW
                    )
                except (OSError, subprocess.TimeoutExpired) as e:
                    logger.warning("kill 실패 PID=%d: %s", pid, e)
        # OS 측 process slot release 시간 (mutex / port release)
        time.sleep(0.3)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("startup nuke 실패 (계속 진행): %s", e)


def _kill_existing_aurora_on_port(port: int = 8765) -> None:
    """v0.1.64: 옛 본체가 API 포트 점유 중이면 taskkill.

    Why: 사용자 제안 — 재시작 = launcher GUI 로 돌아가는 흐름. 본체 자기 죽이기
    의무 X (v0.1.42~v0.1.61 모두 일부 환경 fail). launcher 가 새 본체 spawn 직전
    port 점유 검사 → 점유 PID kill → 새 본체 spawn. 무조건 동작.

    Windows netstat -ano 로 PID 찾기. listening state 만 본체로 간주.
    """
    if platform.system() != "Windows":
        return
    try:
        result = subprocess.run(  # noqa: S603, S607
            ["netstat", "-ano"],
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=0x08000000,  # CREATE_NO_WINDOW (v0.1.102)
        )
        if result.returncode != 0:
            return
        port_marker = f":{port} "
        # cp949 / utf-8 fallback (Windows netstat 환경 의존)
        text = result.stdout.decode("cp949", errors="replace")
        my_pid = os.getpid()
        for line in text.splitlines():
            if port_marker not in line or "LISTENING" not in line:
                continue
            parts = line.split()
            if not parts:
                continue
            try:
                pid = int(parts[-1])
            except ValueError:
                continue
            if pid == my_pid:
                continue
            logger.info("옛 본체 발견 (port %d 점유): PID=%d → taskkill /F", port, pid)
            try:
                kill_result = subprocess.run(  # noqa: S603, S607
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=5,
                    check=False,
                    creationflags=0x08000000,  # CREATE_NO_WINDOW (v0.1.102)
                )
                if kill_result.returncode == 0:
                    logger.info("옛 본체 kill OK: PID=%d", pid)
                    # listening 점유 해제 시간 잠깐 대기 (TIME_WAIT 등)
                    time.sleep(0.5)
                else:
                    logger.warning(
                        "옛 본체 kill returncode=%d", kill_result.returncode,
                    )
            except (OSError, subprocess.TimeoutExpired) as e:
                logger.warning("옛 본체 kill 실패: %s", e)
            return  # 1 PID 만 처리 (보통 1 listener)
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("port %d 점유 검사 실패: %s", port, e)


def launch_aurora() -> subprocess.Popen | None:
    """본체 실행 — env 마커 전달로 본체 자기-swap 중복 방지. 플랫폼별 흐름 (v0.1.67).

    Windows: ``Aurora-ICT.exe`` 직접 Popen (DETACHED_PROCESS).
    macOS:   ``open Aurora-ICT.app`` 명령 (Finder 등록된 앱처럼 실행).

    v0.1.80: Popen 객체 반환 (이전 bool) — LauncherApi 측 본체 process 보관 →
    polling 으로 본체 종료 감지 → launcher webview show 본질. 사용자 제안 패러다임:
    "launcher 가 항상 살아있음 + 본체 spawn 시 hide + 본체 종료 시 show" 본질.

    macOS = `open` 명령 자체가 detach 라 Popen 객체 반환해도 실제 본체 PID X.
    Windows = DETACHED_PROCESS Popen 의 PID 보관 가능 — polling 정합.

    Returns:
        Popen 객체 (Windows 정상 / macOS = open wrapper Popen) /
        None (본체 미존재 또는 실패).
    """
    target = _body_local_target()
    if not target.exists():
        logger.error("본체 미존재: %s", target)
        return None

    # v0.1.64: 옛 본체 (port 점유) 자동 정리 — 사용자 시각 "재시작 = launcher GUI"
    # 흐름. 본체 자기 죽이기 의무 X. 본체 /relaunch 가 launcher 새로 spawn 시
    # KILL_PARENT_PID 도 박지만, 그게 fail 해도 본 단계가 안전망.
    _kill_existing_aurora_on_port()

    env = os.environ.copy()
    env[LAUNCHER_ENV_MARKER] = "1"
    # v0.1.43: launcher 경로 박음 — 본체 /relaunch 가 launcher 다시 spawn 가능.
    # frozen 환경 (sys.executable = launcher.exe) 만 의미. dev 환경은 skip.
    if _is_frozen():
        env[LAUNCHER_PATH_ENV] = sys.executable
    # auto-start env 는 launcher 가 본체 spawn 시 절대 박지 않음 — 새 본체가 자기
    # 다시 재시작 명령 무한 루프 위험 차단.
    env.pop(LAUNCHER_AUTO_START_ENV, None)

    sys_name = platform.system()
    try:
        if sys_name == "Darwin":
            # macOS: `open` 명령 — .app 번들 표준 실행 흐름. Info.plist 처리 + Finder
            # 표준 lifecycle 정합. cwd 박음 X (open 이 .app/Contents/MacOS/ 자동).
            # v0.1.73: stderr 캡처 + 짧은 timeout. ChoYoon #133 fix J — 본체 시작
            # 흐름 진단 자료 박음. open 자체는 비동기 spawn → 정상 = 즉시 종료.
            # fail 시 open 이 stderr 메시지 박음 (e.g. "이 응용 프로그램은 이 Mac
            # 에서 지원되지 않습니다") → launcher.log 측 가시화.
            logger.info("본체 실행 시도 (open .app): %s", target)
            proc = subprocess.Popen(  # noqa: S603, S607 — open 명령 + .app path
                ["open", str(target)],
                env=env,
                close_fds=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                _, stderr = proc.communicate(timeout=2.0)
                if proc.returncode != 0:
                    err = stderr.decode("utf-8", errors="replace").strip()
                    logger.error(
                        "open 명령 fail (rc=%d): %s", proc.returncode, err,
                    )
                    return None
                logger.info("open 명령 성공 (rc=0)")
            except subprocess.TimeoutExpired:
                # 정상 — open 이 .app 비동기 spawn 후 자체 종료 (보통 빠름)
                logger.info("본체 시작 명령 박힘 (비동기 detach)")
            return proc

        # Windows / 기타 — 직접 Popen (DETACHED_PROCESS)
        DETACHED_PROCESS = 0x00000008  # noqa: N806
        CREATE_NEW_PROCESS_GROUP = 0x00000200  # noqa: N806
        flags = (
            DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
            if sys_name == "Windows"
            else 0
        )
        proc = subprocess.Popen(  # noqa: S603 — 본체 실행, 신뢰 가능
            [str(target)],
            env=env,
            creationflags=flags,
            close_fds=True,
            # v0.1.17: cwd = launcher 옆. 본체가 .env / config_store 등을 launcher
            # 옆에서 찾게 → 사용자가 .env 를 launcher 옆에 두면 인식 OK.
            cwd=str(_launcher_dir()),
        )
        # v0.1.101: body 측 SetForegroundWindow 권한 박음 — Windows Vista+ 측
        # focus stealing prevention 정책 측 다른 process 측 focus 뺏을 때 OS
        # 측 거부. launcher (현재 focus 보유 process) 측 AllowSetForegroundWindow
        # 박아 body PID 측 권한 양도 → body 측 SetForegroundWindow 측 성공.
        if sys_name == "Windows":
            try:
                import ctypes
                if not ctypes.windll.user32.AllowSetForegroundWindow(proc.pid):
                    logger.debug(
                        "AllowSetForegroundWindow PID=%d 측 fail (계속 진행)",
                        proc.pid,
                    )
            except Exception as e:  # noqa: BLE001
                logger.debug("AllowSetForegroundWindow 측 예외 (무시): %s", e)
        return proc
    except OSError as e:
        logger.error("본체 실행 실패: %s", e)
        return None


# ============================================================
# Pywebview API — JS bridge
# ============================================================


class LauncherApi:
    """Pywebview JS bridge — UI 가 호출하는 백엔드 메서드."""

    def __init__(self, *, auto_start: bool = False) -> None:
        # v0.1.43: 본체 /relaunch 가 자식 launcher spawn 시 env 박음.
        # UI 가 ``is_auto_start()`` 로 확인 후 START 자동 클릭.
        self._auto_start = auto_start
        # v0.1.80: 본체 process 보관 + polling thread 측 본체 종료 감지.
        # 사용자 제안 패러다임 — launcher 항상 살아있음 + 본체 spawn 시 hide
        # + 본체 종료 시 show. 본체 자기 죽이기 의무 자체 X.
        self._aurora_proc: subprocess.Popen | None = None
        # v0.1.99: WebView2 crash 감지용 — 본체 짧게 살다가 정상 종료 시 likely
        # crash → 자동 respawn (최대 3회). spawn 시각 + 카운트 박음.
        self._aurora_spawn_at: float = 0.0  # time.time() 박음
        self._aurora_crash_respawn_count: int = 0

    def is_auto_start(self) -> bool:
        """v0.1.43: auto-start 모드 여부 — 본체 재시작 흐름에서 launcher 가
        spawn 됐을 때 True. UI 가 START 자동 클릭 결정에 사용."""
        return self._auto_start

    def get_local_version(self) -> str:
        """현재 설치된 본체 버전. 모르면 'unknown'."""
        v = get_local_aurora_ict_version()
        return v if v else "unknown"

    def get_launcher_version(self) -> str:
        """launcher 자체 버전."""
        return __version__

    def check_update(self) -> dict:
        """업데이트 체크 — 결과를 UI 에 dict 로 반환.

        Returns:
            ``{"latest": str | None, "has_update": bool, "url": str | None,
              "error": str | None}``
        """
        # v0.1.116: ChoYoon #133 진단 강화 — 진입/결과 logger.info 박음.
        # 다음 swap fail 시점 측 정확 진단 자료 박힘.
        logger.info("check_update 진입")
        release = fetch_latest_release()
        if release is None:
            detail = f" [{_last_fetch_error}]" if _last_fetch_error else ""
            logger.warning("check_update: release=None%s", detail)
            return {
                "latest": None, "has_update": False, "url": None,
                "error": f"GitHub Releases 조회 실패{detail}",
            }
        latest_tag = release.get("tag_name", "")
        url = find_aurora_exe_url(release)
        local_v = get_local_aurora_ict_version()
        has_update = False
        if local_v is not None:
            try:
                has_update = _parse_version(latest_tag) > _parse_version(local_v)
            except (ValueError, TypeError):
                has_update = False
        else:
            # 로컬 버전 미상 — .aurora_ict_version 파일 없음. 안전하게 has_update=True
            # 로 가정해 사용자 다운 권유. 첫 다운 후 .aurora_ict_version 작성됨 → 다음
            # 부터는 정상 비교. (v0.1.14 fix — 이전엔 exe 존재 시 has_update=False
            # 라 사용자가 launcher 통해 swap 못 받음.)
            has_update = True
        logger.info(
            "check_update 결과: latest=%s local=%s has_update=%s url=%s",
            latest_tag, local_v, has_update, url,
        )
        return {"latest": latest_tag, "has_update": has_update, "url": url,
                "error": None}

    def download_and_swap(self, url: str) -> dict:
        """본체 다운로드 + swap. 플랫폼별 download target (v0.1.73).

        Windows: ``Aurora-ICT.exe.new`` (rename swap)
        macOS:   ``Aurora-ICT-macOS.zip.new`` (zip 형식 보존 — apply_swap 가 ditto 풀음)

        Why: ChoYoon Claude #133 10th cycle fix H — 이전 흐름은 macOS 측에서
        ``Aurora-ICT.app.new`` (단일 파일 path) 박음 → apply_swap 의 .zip 분기 SKIP
        → Windows rename 흐름 fall-through → ``Aurora-ICT.app`` 가 zip 바이트 단일
        파일로 박힘 (디렉토리 X, 번들 X) → macOS "지원되지 않음" 에러.

        Returns:
            ``{"success": bool, "message": str}``
        """
        # v0.1.116: ChoYoon #133 진단 강화 — 진입/결과 logger.info 박음.
        logger.info("download_and_swap 진입: url=%s", url)
        if platform.system() == "Darwin":
            # macOS: download target = .zip.new — apply_swap 가 zip 풀어서 .app 박음.
            target = _aurora_ict_data_dir() / "Aurora-ICT-macOS.zip.new"
        else:
            target = _aurora_exe_path().with_suffix(_aurora_exe_path().suffix + ".new")
        # 폴더 자동 생성 (첫 다운 시)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not download_to(url, target):
            logger.warning("download_and_swap: download_to 실패 url=%s", url)
            return {"success": False, "message": "다운로드 실패 (네트워크 확인)"}
        if not apply_swap(target):
            logger.warning("download_and_swap: apply_swap 실패 target=%s", target)
            return {"success": False, "message": "swap 실패 (본체 권한 확인)"}
        # 새 버전 기록
        try:
            release = fetch_latest_release()
            if release is not None:
                version_file = _aurora_ict_data_dir() / ".aurora_ict_version"
                version_file.parent.mkdir(parents=True, exist_ok=True)
                version_file.write_text(
                    release.get("tag_name", "").lstrip("v"), encoding="utf-8",
                )
        except OSError:
            pass
        logger.info("download_and_swap 완료: success=True")
        return {"success": True, "message": "업데이트 적용 완료"}

    def launch(self) -> dict:
        """본체 실행 — v0.1.80 사용자 제안 패러다임 + v0.1.116 readiness polling:
        launcher 항상 살아있음 + 본체 spawn 시 readiness 측 wait → ready 박힌 후
        webview hide → 본체 GUI 차지. 본체 종료 시 show.

        v0.1.116 (ChoYoon #133): 이전 흐름 측 spawn 즉시 hide → 사용자 측 까만
        화면 51초+37초 본질 (launcher swap + 본체 startup). 신규 흐름 측 spawn
        후 launcher webview 측 그대로 + status "본체 시작 중..." 박음 → /health
        200 OK 박힐 때까지 polling → ready 시점 hide. 사용자 측 마찰 ↓↓↓.
        """
        # v0.1.116: spawn 직후 launcher status 측 "본체 시작 중..." 박음
        self._update_status_js("본체 시작 중...", "var(--text-2)")
        proc = launch_aurora()
        if proc is None:
            self._update_status_js("✗ 본체 미존재", "#fb7185")
            return {"success": False, "message": "본체 .exe 미존재 — 먼저 업데이트"}
        # 본체 process 보관 + polling thread 시작
        self._aurora_proc = proc
        self._aurora_spawn_at = time.time()  # v0.1.99: crash 감지용
        self._aurora_crash_respawn_count = 0  # v0.1.99: 자동 respawn 카운터
        self._start_aurora_polling()
        # v0.1.116: readiness polling 시작 — ready 박힐 때 hide
        self._start_readiness_polling()
        return {"success": True, "message": "Aurora 시작됨"}

    def _start_readiness_polling(
        self, ready_timeout: float = 60.0, ready_interval: float = 0.5,
    ) -> None:
        """v0.1.116 (ChoYoon #133 ⭐⭐⭐⭐⭐): 본체 startup 측 ``/health`` 200 OK
        박힐 때까지 polling → ready 시점 launcher hide.

        매 0.5초 체크, 60초 timeout (default). 사용자 측 시간 분해 — 본체 startup
        ~37초 + buffer 23초. timeout 초과 측 launcher hide 강행 (본체 측 startup
        실패 가능, 사용자 측 status 측 표시).

        Args:
            ready_timeout: polling deadline (테스트 측 짧게 박음).
            ready_interval: poll attempt 간격.

        Status 갱신:
            - 매 5초 측 elapsed 박음 ("본체 시작 중... (12초)")
            - ready 박힐 때 ✓ 표기 + 0.3초 후 hide
            - timeout 측 ⚠ 표기 + 1초 후 hide 강행
        """
        ready_url = "http://127.0.0.1:8765/health"

        def _poll() -> None:
            poll_start = time.time()
            deadline = poll_start + ready_timeout
            attempt = 0
            while time.time() < deadline:
                attempt += 1
                try:
                    req = urllib.request.Request(ready_url)
                    with urllib.request.urlopen(req, timeout=2.0) as resp:  # noqa: S310
                        if resp.status == 200:
                            elapsed = time.time() - self._aurora_spawn_at
                            logger.info(
                                "readiness OK (시작 후 %.1f초, %d 시도) → launcher hide",
                                elapsed, attempt,
                            )
                            self._update_status_js(
                                f"✓ Aurora 시작됨 ({elapsed:.0f}초)",
                                "#34d399",
                            )
                            # v0.2.24: progress bar 100% + 짧은 빈 보여주기
                            self._update_progress_js(100.0, "✓ 시작됨")
                            time.sleep(0.3)
                            self._hide_launcher_window()
                            return
                except Exception:  # noqa: BLE001 — startup 중 connection refused 정상
                    pass
                # v0.2.24 (ChoYoon #133 P1 ③): progress bar 측 매 5 시도 (2.5초) 갱신
                # — 사용자 측 멈춤 의심 회피 + 시각 자료. percent 측 95% 측 cap 박음
                # (마지막 5% 측 ready 박힘 시점 측 100%).
                if attempt % 5 == 0:
                    elapsed = time.time() - self._aurora_spawn_at
                    percent = min(95.0, (elapsed / ready_timeout) * 100)
                    self._update_progress_js(
                        percent, f"본체 준비 중... ({elapsed:.0f}초)",
                    )
                # 매 10번 (5초) 측 status 갱신
                if attempt % 10 == 0:
                    elapsed = time.time() - self._aurora_spawn_at
                    self._update_status_js(
                        f"본체 시작 중... ({elapsed:.0f}초)",
                        "var(--text-2)",
                    )
                time.sleep(ready_interval)
            # timeout — hide 강행
            elapsed = time.time() - self._aurora_spawn_at
            logger.warning(
                "readiness timeout (%.1f초, %d 시도) → launcher hide 강행",
                elapsed, attempt,
            )
            self._update_status_js(
                f"⚠ 본체 응답 X ({elapsed:.0f}초) — hide 강행",
                "#fbbf24",
            )
            # v0.2.24: timeout 측 progress 100% 박지 X — 사용자 측 fail 정합 인지
            time.sleep(1.0)
            self._hide_launcher_window()

        threading.Thread(target=_poll, daemon=True, name="readiness-poll").start()

    def _update_status_js(self, text: str, color: str) -> None:
        """v0.1.116: launcher webview status_line 측 외부 thread 측 갱신.

        readiness polling thread 측 status 측 매 5초 갱신 본질. JS 측 setStatus
        측 window 박혀있어 evaluate_js 측 호출 가능.
        """
        try:
            import webview  # type: ignore[import-not-found]
            if not webview.windows:
                return
            text_esc = text.replace("\\", "\\\\").replace("'", "\\'")
            color_esc = color.replace("\\", "\\\\").replace("'", "\\'")
            webview.windows[0].evaluate_js(
                f"if (window.setStatus) setStatus('{text_esc}', '{color_esc}');",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("status JS update fail: %s", e)

    def _update_progress_js(self, percent: float, text: str) -> None:
        """v0.2.24 (ChoYoon #133 P1 ③): launcher webview progress bar 측 갱신.

        readiness polling thread 측 매 2.5초 측 percent 갱신 박음. 사용자 측
        멈춤 의심 회피 + 시각 자료. JS 측 setProgress 측 window 박혀있어
        evaluate_js 측 호출 가능. ready 박힐 때 100% 박음 + 짧은 fade.

        Args:
            percent: 0.0 ~ 100.0
            text: progress 측 아래 표기 텍스트
        """
        try:
            import webview  # type: ignore[import-not-found]
            if not webview.windows:
                return
            text_esc = text.replace("\\", "\\\\").replace("'", "\\'")
            webview.windows[0].evaluate_js(
                f"if (window.setProgress) setProgress({percent:.1f}, '{text_esc}');",
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("progress JS update fail: %s", e)

    def _hide_launcher_window(self) -> None:
        """v0.1.80: launcher webview hide — 백그라운드 살아있음.

        v0.1.116 (ChoYoon #133): macOS 측 ``webview.windows[0].hide()`` 만으론
        dock icon 측 그대로 박힘 → process 측 visible=false 박아 완전 hide.
        Windows 측 영향 X.
        """
        try:
            import webview  # type: ignore[import-not-found]
            if webview.windows:
                webview.windows[0].hide()
                logger.info("launcher webview hide — 본체 GUI 차지")
                # v0.1.116: macOS 측 osascript 측 process visible=false 박음
                if platform.system() == "Darwin":
                    try:
                        subprocess.run(  # noqa: S603, S607
                            ["osascript", "-e",
                             'tell application "System Events" to set visible'
                             ' of process "Aurora-ICT-launcher" to false'],
                            capture_output=True, timeout=3, check=False,
                        )
                        logger.debug("macOS osascript hide 박힘")
                    except (OSError, subprocess.TimeoutExpired) as e:
                        logger.debug("osascript hide 실패 (계속 진행): %s", e)
        except Exception as e:  # noqa: BLE001
            logger.warning("launcher hide 실패: %s", e)

    def _show_launcher_window(self) -> None:
        """v0.1.80: launcher webview show — 본체 종료 후 사용자 화면 등장.

        v0.1.109 (ChoYoon 권장 b): macOS 측 hide 측 dock icon 측 그대로 박혀 있어
        focus 측 본체 측 잡혀있을 가능성. show 시 NSApp 측 forefront 박음 — pywebview
        측 노출 X 라 osascript 박아 \"frontmost = true\" 박음.

        v0.2.18 (ChoYoon #133 P0 ①): v0.1.116 측 hide 측 ``set visible=false`` 박은
        symmetric 정합 박힘 본질 — show 측 ``set visible=true`` 측 추가 의무 박음.
        Windows 측 정상 동작 verify 박힘 (사용자 huihu) → macOS-specific 회귀 박힘.
        """
        try:
            import webview  # type: ignore[import-not-found]
            if webview.windows:
                webview.windows[0].show()
                logger.info("launcher webview show — 본체 종료 감지 + 등장")
                # v0.1.109 + v0.2.18: macOS 측 forefront + visible 박음.
                # show 만으론 dock 박힌 채 사용자 화면 측 안 떠올라옴.
                if platform.system() == "Darwin":
                    # v0.2.18: hide 측 set visible=false 박았으므로 show 측
                    # set visible=true 박아야 dock + 사용자 화면 측 표시 회복.
                    try:
                        subprocess.run(  # noqa: S603, S607
                            ["osascript", "-e",
                             'tell application "System Events" to set visible'
                             ' of process "Aurora-ICT-launcher" to true'],
                            capture_output=True, timeout=3, check=False,
                        )
                        logger.debug("macOS osascript visible=true 박힘")
                    except (OSError, subprocess.TimeoutExpired) as e:
                        logger.debug("osascript visible=true 실패 (계속 진행): %s", e)
                    try:
                        subprocess.run(  # noqa: S603, S607
                            ["osascript", "-e",
                             'tell application "Aurora-ICT-launcher" to activate'],
                            capture_output=True, timeout=3, check=False,
                        )
                    except (OSError, subprocess.TimeoutExpired) as e:
                        logger.debug("osascript activate 실패 (계속 진행): %s", e)
        except Exception as e:  # noqa: BLE001
            logger.warning("launcher show 실패: %s", e)

    def _start_aurora_polling(self) -> None:
        """v0.1.80: 본체 process 종료 감지 polling thread.
        v0.1.83: marker file 검사 추가 — 본체 측 /relaunch 시 marker 박음
        → launcher 가 본체 process.terminate() 호출 (본체 자체 종료 의무 X).
        v0.1.92: marker 발동 종료면 새 본체 자동 spawn — 사용자 시각 \"재시작\"
        한 번 클릭 흐름 (이전 v0.1.83 측 사용자가 START 다시 클릭 본질 X).

        흐름:
            - 본체 정상 종료 (X 클릭 / crash) → launcher show + START 활성
            - 본체 marker 종료 (재시작 요청) → 새 본체 자동 spawn (launcher 그대로 hide)

        v0.2.22 (ChoYoon #133 18 cycle P1 ④): macOS 측 ``open .app`` wrapper Popen 측
        즉시 종료 박음 → ``proc.poll()`` 측 의미 X 박음. 즉 spawn 직후 \"본체 종료
        감지\" log 측 false positive 박힘 (legacy 오발화). macOS 측 polling skip.
        macOS 측 본체 종료 감지 → launcher show 흐름 측 v0.2.18 launcher show-back
        측 측 dock icon 측 사용자 측 직접 클릭 박는 흐름 박음.
        """
        # v0.2.22: macOS 측 polling skip — open wrapper Popen 측 PID 측 추적 의미 X
        if platform.system() == "Darwin":
            logger.info(
                "_start_aurora_polling: macOS skip — open wrapper Popen 측 즉시 종료, "
                "본체 종료 감지 측 사용자 측 dock icon 클릭 흐름 박음",
            )
            return
        marker_path = _aurora_ict_data_dir() / ".relaunch_request"

        def _poll() -> None:
            while True:
                proc = self._aurora_proc
                if proc is None:
                    break
                # v0.1.83: marker 검사 — 본체 측 /relaunch 호출 시 박음
                triggered_by_marker = False
                if marker_path.exists():
                    logger.info(
                        "marker 발견 (%s) → 본체 process.terminate() 호출",
                        marker_path,
                    )
                    try:
                        proc.terminate()
                        # 짧은 대기 후 안 죽으면 kill
                        try:
                            proc.wait(timeout=3.0)
                        except subprocess.TimeoutExpired:
                            logger.warning("terminate 후도 살아있음 → kill")
                            proc.kill()
                    except OSError as e:
                        logger.warning("본체 종료 실패: %s", e)
                    # marker 정리
                    try:
                        marker_path.unlink()
                    except OSError:
                        pass
                    triggered_by_marker = True
                rc = proc.poll()
                if rc is not None:
                    # v0.1.92: marker 발동 종료 = 재시작 요청 → 새 본체 자동 spawn
                    if triggered_by_marker:
                        logger.info(
                            "재시작 요청 (marker) — 새 본체 자동 spawn (rc=%s)", rc,
                        )
                        new_proc = launch_aurora()
                        if new_proc is not None:
                            self._aurora_proc = new_proc
                            self._aurora_spawn_at = time.time()
                            logger.info("새 본체 spawn OK — polling 계속")
                            continue  # 새 process 측 polling 계속
                        logger.warning(
                            "새 본체 spawn 실패 — launcher show fallback",
                        )
                    # v0.1.100: v0.1.99 측 auto-respawn 제거 — 사용자 보고
                    # "본체 ↔ launcher 무한 재시작 cascade" 본질. 본체 짧게 죽는
                    # 진짜 원인 (uvicorn port bind fail / 다른 본체 process 충돌
                    # 등) 을 가리고 사용자 환경 더 망가뜨림. 그냥 launcher show 박음.
                    elapsed = time.time() - self._aurora_spawn_at
                    logger.info(
                        "본체 종료 감지 (rc=%s, %.1f초 살아있음) → launcher show + "
                        "START 활성", rc, elapsed,
                    )
                    self._aurora_proc = None
                    self._show_launcher_window()
                    # JS 측 START 버튼 활성 — webview eval
                    try:
                        import webview  # type: ignore[import-not-found]
                        if webview.windows:
                            webview.windows[0].evaluate_js(
                                "document.getElementById('btn-start').disabled = false;"
                                "if (window.setStatus) setStatus('본체 종료됨 — 다시 시작 가능', 'var(--text-2)');",
                            )
                    except Exception as e:  # noqa: BLE001
                        logger.debug("UI START 활성 fail: %s", e)
                    break
                time.sleep(2.0)
        threading.Thread(target=_poll, daemon=True).start()

    def quit(self) -> None:
        """launcher 종료 — pywebview 윈도우 destroy + os._exit (v0.1.18 fix).

        Why: 이전 sys.exit(0) 은 js_api thread 에서 호출되어 pywebview main thread
        가 catch 안 함 → launcher 종료 X (사용자 보고). webview.windows[0].destroy()
        + os._exit(0) 으로 강제 종료.
        """
        try:
            import webview  # type: ignore[import-not-found]
            if webview.windows:
                webview.windows[0].destroy()
        except Exception:  # noqa: BLE001 — 종료 흐름이라 예외 무시
            pass
        os._exit(0)  # noqa: S603 — 의도적 강제 종료


# ============================================================
# GUI 진입점
# ============================================================


def _ui_index_path() -> Path:
    """index.html 경로 — 빌드 / dev 모두 대응."""
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS) / "ui" / "index.html"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent / "ui" / "index.html"


def _setup_file_logging() -> Path | None:
    """v0.1.63: launcher 진단용 file logging — `%LOCALAPPDATA%\\Aurora-ICT\\launcher.log`.

    ChoYoon Claude #133 환기 — frozen `--windowed` 빌드는 콘솔 X → stderr 사라짐
    → ``logger.warning`` 출력 위치 X → 사용자 측 root cause 진단 불가.
    file handler 박아두면 manual 다운로드 후 사용자 측 로그 파일에 자동 기록 →
    다음 cycle 정확한 진단 가능.

    Returns:
        log 파일 절대 경로 (성공 시) / None (디렉토리 권한 등 실패 시).
    """
    import logging.handlers

    try:
        log_dir = _aurora_ict_data_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "launcher.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=1_000_000,  # 1 MB
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        ))
        # root logger 에 박음 — launcher 내부 logger.* 모두 포함.
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        # 이미 file handler 박혔으면 중복 X (자가-update swap 시 main 재진입)
        for existing in list(root_logger.handlers):
            if isinstance(existing, logging.handlers.RotatingFileHandler):
                root_logger.removeHandler(existing)
        root_logger.addHandler(file_handler)
        return log_file
    except OSError:
        return None


def _log_environment_info() -> None:
    """v0.1.63: 시스템 정보 로깅 — proxy / Python / platform / frozen 등.

    사용자 측 root cause (proxy / SSL / certifi) 진단 단서. ChoYoon Claude
    #133 fix 4.
    """
    import platform as _plat

    logger.info("=" * 60)
    logger.info("Aurora-Launcher v%s 시작", __version__)
    logger.info("Python %s", sys.version.replace("\n", " "))
    logger.info("Platform: %s", _plat.platform())
    logger.info("sys.frozen=%s _MEIPASS=%s",
                getattr(sys, "frozen", False),
                getattr(sys, "_MEIPASS", None))
    logger.info("LOCALAPPDATA=%s", os.environ.get("LOCALAPPDATA"))
    logger.info(
        "HTTPS_PROXY=%s HTTP_PROXY=%s",
        os.environ.get("HTTPS_PROXY") or "(unset)",
        os.environ.get("HTTP_PROXY") or "(unset)",
    )
    # v0.1.66: SSL context 출처 — ChoYoon Claude #133 fix C.
    # 다음 cycle launcher.log 에서 "certifi C:\...\_MEI...\certifi\cacert.pem" 박힘
    # → frozen 환경 SSL 핸드셰이크 본질 정합 verify 가능.
    logger.info("SSL context: %s", _SSL_CTX_SOURCE)
    # v0.2.18 (ChoYoon #133 P0 ⑦): Windows 측 "system default" 박힘 = certifi
    # import 측 fail. 사유 박음 → root cause 진단 (ImportError module name 등).
    if _SSL_IMPORT_ERROR is not None:
        logger.warning("certifi import 실패 사유: %s", _SSL_IMPORT_ERROR)
    logger.info("=" * 60)


def _kill_parent_if_requested() -> None:
    """v0.1.61: 본체 /relaunch → launcher Popen 시 박은 부모 PID 강제 종료.

    Why: v0.1.42~v0.1.58 본체 자기 죽이기 (os._exit / ExitProcess / cmd
    watchdog) 모두 일부 사용자 환경에서 실패 (PyInstaller frozen + uvicorn +
    threading + webview 복합 hold). launcher 는 별개 process group 이라 부모
    묶임 X → taskkill /F /T 무조건 동작.

    환경변수 ``AURORA_ICT_KILL_PARENT_PID`` 있으면 그 PID 강제 종료. 없으면 noop
    (일반 launcher 시작 흐름).
    """
    kill_pid_str = os.environ.get(LAUNCHER_KILL_PARENT_PID_ENV)
    if not kill_pid_str:
        return
    try:
        kill_pid = int(kill_pid_str)
    except ValueError:
        logger.warning("KILL_PARENT_PID env 잘못됨: %s", kill_pid_str)
        return
    logger.info("부모 본체 강제 종료 시도: PID=%d", kill_pid)
    try:
        # /F = 강제, /T = 자식 트리도. Windows 만 (launcher = Windows only).
        result = subprocess.run(  # noqa: S603, S607
            ["taskkill", "/F", "/T", "/PID", str(kill_pid)],
            capture_output=True,
            timeout=5,
            check=False,
            creationflags=0x08000000,  # CREATE_NO_WINDOW (v0.1.102)
        )
        if result.returncode == 0:
            logger.info("부모 본체 종료 OK: PID=%d", kill_pid)
        else:
            logger.warning(
                "부모 본체 종료 returncode=%d stderr=%s",
                result.returncode, result.stderr.decode("utf-8", errors="replace"),
            )
    except (OSError, subprocess.TimeoutExpired) as e:
        logger.warning("부모 본체 종료 실패: %s", e)
    # env 정리 — 다음 launcher self-update swap 시 본 env 잔재 X
    os.environ.pop(LAUNCHER_KILL_PARENT_PID_ENV, None)


def main() -> None:
    """launcher 진입점 — pywebview 윈도우 시작."""
    import webview  # type: ignore[import-not-found]

    # v0.1.63: file logging 박기 가장 우선 — 모든 후속 단계 로그 보존.
    # ChoYoon Claude #133 fix 1 (가장 시급). frozen --windowed 환경 stderr 사라짐
    # 진단 차단 메타 issue 해소 — 사용자 측 manual 다운로드 후 launcher.log 박힘.
    log_file = _setup_file_logging()
    _log_environment_info()
    if log_file is not None:
        logger.info("file logging 활성: %s", log_file)
    else:
        logger.warning("file logging 활성 실패 (디렉토리 권한 등)")

    # v0.1.61: 본체 /relaunch 흐름 — 부모 본체 PID 받아 강제 종료 (자기-launcher 호출).
    # 다른 단계 (apply_pending_launcher_update / migrate / update check) 보다 우선 —
    # 부모 살아있으면 새 본체 spawn 시 포트 충돌 등 위험.
    _kill_parent_if_requested()

    # v0.1.100: 다른 Aurora process 강제 정리 — 사용자 보고 launcher / body 다중
    # 실행 cascade fix. mutex 측 timing race 못 잡는 케이스 보강.
    _kill_other_aurora_processes()

    # v0.1.19: 직전 다운된 launcher.new 가 있으면 swap → 새 launcher 재시작.
    # 본 함수는 swap 성공 시 sys.exit(0) — 도달 X.
    apply_pending_launcher_update()

    # v0.1.93: launcher 측 single-instance mutex — 사용자 보고 (2026-05-08)
    # "런처가 2개 실행" 본질. apply_pending_launcher_update() 다음에 박힘 —
    # self-update 측 spawn-then-exit 흐름 측 mutex race 회피.
    if not _acquire_launcher_single_instance_mutex():
        logger.warning(
            "Launcher 이미 실행 중 (Windows named mutex %s) — 중복 실행 차단",
            _LAUNCHER_MUTEX_NAME,
        )
        sys.exit(0)

    # v0.1.17: 이전 layout (launcher 옆 Aurora-ICT.exe) → _aurora/ 자동 이전.
    _migrate_legacy_layout()

    # v0.1.19: 백그라운드 launcher 자기 update check + 다운 (다음 시작 시 apply).
    start_background_launcher_check()

    # v0.1.43: 본체 /relaunch 흐름 — env 또는 sys.argv 로 auto-start 모드 결정.
    auto_start = (
        os.environ.get(LAUNCHER_AUTO_START_ENV) == "1"
        or "--auto-start" in sys.argv
    )
    api = LauncherApi(auto_start=auto_start)
    ui_path = _ui_index_path()

    webview.create_window(
        "Aurora Launcher",
        str(ui_path),
        js_api=api,
        width=1280,                  # 본체 .exe 와 동일 크기 (v0.1.16 redesign)
        height=800,
        min_size=(960, 600),
        resizable=True,
        background_color="#1e202c",  # v0.1.15 brand bg
    )

    # GUI 떠 있는 동안 백그라운드로 자동 update check (1회)
    def _bg_check() -> None:
        api.check_update()  # 캐시 효과 — UI 가 다시 호출하면 빠른 응답

    threading.Thread(target=_bg_check, daemon=True).start()

    webview.start()


if __name__ == "__main__":
    main()
