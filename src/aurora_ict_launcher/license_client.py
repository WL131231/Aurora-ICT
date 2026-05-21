"""Aurora-ICT 라이선스 클라이언트 — 백엔드 통신 + 로컬 토큰 저장 + grace 판정.

기능:
    - ``get_machine_id()`` — PC 식별 hash (Windows MachineGuid + CPU UUID + 디스크 시리얼 → SHA-256).
    - ``redeem_code()`` — POST /api/redeem (첫 코드 사용, PC 묶기).
    - ``verify_license()`` — POST /api/verify (정기 검증).
    - ``load_license()`` / ``save_license()`` / ``delete_license()`` — license.json 관리.
    - ``is_within_grace()`` — 오프라인 grace period 판정.

정책 ([[project-aurora-ict-punch-list]] 와 2026-05-21 합의):
    - 레퍼럴 (referral, 평생): 마지막 verify 성공 후 7일 grace.
    - 구독제 (sub_30d/90d/365d): expires_at 까지 verify 결과 무시 (사놓은 기간 보장).

PC 묶기 강도: Windows MachineGuid + CPU UUID + 디스크 시리얼 (강).
    - SSD/HDD 교체 시 라이선스 무효화 → 관리자 재발급 요청 필요.
    - 부품 안 바꾸면 영구 일정.

의존성: 표준 라이브러리만 (urllib + ssl + winreg). 외부 패키지 X.
백엔드: 환경변수 ``AURORA_LICENSE_API_BASE`` 로 override 가능 (테스트/staging).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import UTC, datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================
# 설정 상수
# ============================================================

LICENSE_API_BASE = os.environ.get(
    "AURORA_LICENSE_API_BASE",
    "https://aurora-ict-license-r8dl.vercel.app",
).rstrip("/")
"""라이선스 백엔드 URL — Vercel prod (aurora-ict-license-r8dl)."""

LICENSE_HTTP_TIMEOUT_SEC = 15
"""HTTP 타임아웃 — launcher 의 GitHub fetch 와 동일 (방화벽/외부망 환경 보수적)."""

GRACE_DAYS_REFERRAL = 7
"""레퍼럴 (평생) 라이선스의 오프라인 허용 기간 (일). 2026-05-21 합의."""

LICENSE_FILE_NAME = "license.json"

USER_AGENT = "Aurora-ICT-Launcher"


# ============================================================
# machine_id 생성
# ============================================================


def _read_windows_machine_guid() -> str | None:
    """Windows: ``HKLM\\SOFTWARE\\Microsoft\\Cryptography\\MachineGuid``.

    OS 재설치하지 않으면 영구 일정. Windows 7+ 모두 보유.
    """
    if sys.platform != "win32":
        return None
    try:
        import winreg
    except ImportError:
        return None
    try:
        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE,
            r"SOFTWARE\Microsoft\Cryptography",
            0,
            winreg.KEY_READ | winreg.KEY_WOW64_64KEY,
        ) as key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
            return str(value).strip()
    except OSError as e:
        logger.debug("MachineGuid 읽기 실패: %s", e)
        return None


def _read_cpu_uuid() -> str | None:
    """CPU/시스템 UUID — Windows wmic / macOS ioreg / Linux /etc/machine-id.

    Windows: ``wmic csproduct get UUID`` — 메인보드 BIOS UUID. 메인보드 바꾸면 변경.
    macOS: ``ioreg`` 의 IOPlatformUUID.
    Linux: ``/etc/machine-id`` (systemd 표준).
    """
    sysname = platform.system()
    try:
        if sysname == "Windows":
            out = subprocess.run(
                ["wmic", "csproduct", "get", "UUID"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in out.stdout.splitlines():
                stripped = line.strip()
                if stripped and stripped.lower() != "uuid":
                    return stripped
        elif sysname == "Darwin":
            out = subprocess.run(
                ["ioreg", "-d2", "-c", "IOPlatformExpertDevice"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in out.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    parts = line.split('"')
                    if len(parts) >= 4:
                        return parts[-2].strip()
        else:
            mid_path = Path("/etc/machine-id")
            if mid_path.exists():
                return mid_path.read_text(encoding="utf-8").strip()
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("CPU UUID 읽기 실패 (%s): %s", sysname, e)
    return None


def _read_disk_serial() -> str | None:
    """첫 번째 디스크 시리얼 — Windows wmic / macOS diskutil.

    Windows: ``wmic diskdrive where Index=0 get SerialNumber`` — 부팅 디스크 (Index=0).
    macOS: ``diskutil info /`` 의 Volume UUID.
    Linux: 별도 표준 X — fallback None (machine-id + cpu uuid 로 충분).
    """
    sysname = platform.system()
    try:
        if sysname == "Windows":
            out = subprocess.run(
                ["wmic", "diskdrive", "where", "Index=0", "get", "SerialNumber"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in out.stdout.splitlines():
                stripped = line.strip()
                if stripped and stripped.lower() != "serialnumber":
                    return stripped
        elif sysname == "Darwin":
            out = subprocess.run(
                ["diskutil", "info", "/"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            for line in out.stdout.splitlines():
                if "Volume UUID" in line:
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        return parts[1].strip()
    except (subprocess.SubprocessError, OSError) as e:
        logger.debug("디스크 시리얼 읽기 실패 (%s): %s", sysname, e)
    return None


def _build_machine_id_parts() -> list[str]:
    """machine_id 의 raw 식별자 리스트 — 테스트 용으로 분리."""
    parts: list[str] = []

    guid = _read_windows_machine_guid()
    if guid:
        parts.append(f"guid:{guid}")

    cpu_uuid = _read_cpu_uuid()
    if cpu_uuid:
        parts.append(f"cpu:{cpu_uuid}")

    disk_serial = _read_disk_serial()
    if disk_serial:
        parts.append(f"disk:{disk_serial}")

    if not parts:
        # 모든 부품 식별 실패 — 호스트네임 + user 로 fallback (덜 안정적이지만 동작).
        # Why: wmic deprecated 환경 (Win 11 신형) 또는 ioreg 권한 없는 케이스 대비.
        parts.append(f"host:{platform.node()}")
        parts.append(f"user:{os.environ.get('USERNAME') or os.environ.get('USER') or ''}")
    return parts


def get_machine_id() -> str:
    """PC 식별 hash — 부품 식별자 조합 SHA-256.

    Returns:
        64자 hex string. 같은 PC 에서는 부품 안 바꾸면 영구 일정.

    Raises:
        없음 — 모든 부품 식별 실패 시에도 fallback 으로 동작.
    """
    parts = _build_machine_id_parts()
    combined = "|".join(parts)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()


# ============================================================
# license.json read / write
# ============================================================


def _license_file_path(data_dir: Path) -> Path:
    return data_dir / LICENSE_FILE_NAME


def load_license(data_dir: Path) -> dict | None:
    """``license.json`` 읽기.

    Args:
        data_dir: 격리 폴더 (Windows: ``%LOCALAPPDATA%\\Aurora-ICT``).

    Returns:
        파싱된 dict, 또는 파일 없음/깨졌으면 ``None``.
    """
    path = _license_file_path(data_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("license.json 읽기 실패: %s", e)
        return None


def save_license(data_dir: Path, payload: dict) -> bool:
    """``license.json`` atomic 쓰기 (tmp → replace).

    Args:
        data_dir: 격리 폴더.
        payload: ``{"code", "type", "license_token", "expires_at", "last_verified_at"}``.

    Returns:
        성공 여부. 실패 시 logger.warning 후 False.
    """
    path = _license_file_path(data_dir)
    tmp_path = path.with_suffix(".json.tmp")
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp_path.replace(path)
        return True
    except OSError as e:
        logger.warning("license.json 쓰기 실패: %s", e)
        return False


def delete_license(data_dir: Path) -> None:
    """``license.json`` 삭제 (라이선스 무효화 / 코드 재입력 흐름)."""
    path = _license_file_path(data_dir)
    if path.exists():
        try:
            path.unlink()
        except OSError as e:
            logger.warning("license.json 삭제 실패: %s", e)


# ============================================================
# HTTP 호출 (urllib + 공유 SSL context)
# ============================================================


def _post_json(
    api_path: str,
    payload: dict,
    ssl_context: ssl.SSLContext,
) -> tuple[int, dict]:
    """POST + JSON 응답.

    Args:
        api_path: ``/api/redeem`` 같은 path.
        payload: 요청 body (JSON 직렬화).
        ssl_context: launcher.py 의 ``_SSL_CONTEXT`` 재사용 (certifi cacert).

    Returns:
        ``(status_code, response_dict)``. 네트워크 실패 시 ``(0, {"error": "network", ...})``.
    """
    url = LICENSE_API_BASE + api_path
    body = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json; charset=utf-8",
        "User-Agent": USER_AGENT,
    }
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(
            req, timeout=LICENSE_HTTP_TIMEOUT_SEC, context=ssl_context,
        ) as resp:
            raw = resp.read().decode("utf-8")
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                data = {"error": "non_json", "raw": raw[:200]}
            return resp.status, data
    except urllib.error.HTTPError as e:
        # 4xx/5xx — body 도 JSON 파싱 시도
        try:
            err_raw = e.read().decode("utf-8")
            err_data = json.loads(err_raw)
        except (json.JSONDecodeError, OSError):
            err_data = {"error": f"http_{e.code}"}
        return e.code, err_data
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        logger.warning("license %s 네트워크 실패: %s", api_path, e)
        return 0, {"error": "network", "detail": str(e)}


def redeem_code(
    code: str,
    machine_id: str,
    ssl_context: ssl.SSLContext,
) -> tuple[int, dict]:
    """POST /api/redeem — 첫 코드 사용 (PC 묶기).

    Args:
        code: ``AICT-XXXX-XXXX-XXXX``.
        machine_id: ``get_machine_id()`` 결과.
        ssl_context: ``launcher._SSL_CONTEXT``.

    Returns:
        백엔드 응답. 200 성공 시 ``license_token`` / ``type`` / ``expires_at`` 포함.
    """
    return _post_json(
        "/api/redeem",
        {"code": code, "machine_id": machine_id},
        ssl_context,
    )


def verify_license(
    code: str,
    machine_id: str,
    license_token: str,
    ssl_context: ssl.SSLContext,
) -> tuple[int, dict]:
    """POST /api/verify — 정기 검증.

    Args:
        code: ``AICT-XXXX-XXXX-XXXX``.
        machine_id: ``get_machine_id()`` 결과.
        license_token: redeem 시 받은 HMAC 토큰.
        ssl_context: ``launcher._SSL_CONTEXT``.

    Returns:
        백엔드 응답. 200 성공 시 ``type`` / ``expires_at`` 포함.
        403 일 때 error 는 ``not_active`` / ``pc_mismatch`` / ``bad_token`` / ``expired`` 중 하나.
    """
    return _post_json(
        "/api/verify",
        {"code": code, "machine_id": machine_id, "license_token": license_token},
        ssl_context,
    )


# ============================================================
# grace period 판정
# ============================================================


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        # 백엔드는 보통 ``2026-06-20T10:34:30.858822+00:00`` 형식.
        # Python 3.11+ 의 ``fromisoformat`` 가 Z 도 처리하지만, 보수적으로 변환.
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def is_within_grace(
    license_payload: dict,
    *,
    now: datetime | None = None,
) -> bool:
    """오프라인 grace 안인지 — verify 네트워크 실패 시 봇 계속 동작 허용 여부.

    정책 (2026-05-21 합의):
        - ``type == "referral"`` (평생): 마지막 verify 성공 후 ``GRACE_DAYS_REFERRAL`` 안.
        - ``type`` 이 ``sub_*``: ``expires_at`` 까지 verify 결과 무시 — 사놓은 기간 보장.

    Args:
        license_payload: ``load_license()`` 결과.
        now: 현재 시각 (UTC). 기본 ``datetime.now(timezone.utc)``. 테스트 주입용.

    Returns:
        grace 안이면 True (봇 동작 OK). False 면 봇 중단 + 코드 입력 화면.
    """
    now = now or datetime.now(UTC)
    license_type = license_payload.get("type", "")

    if license_type.startswith("sub_"):
        expires_at = _parse_iso(license_payload.get("expires_at"))
        if expires_at is None:
            return False
        return now < expires_at

    if license_type == "referral":
        last_verified = _parse_iso(license_payload.get("last_verified_at"))
        if last_verified is None:
            return False
        return now < (last_verified + timedelta(days=GRACE_DAYS_REFERRAL))

    return False
