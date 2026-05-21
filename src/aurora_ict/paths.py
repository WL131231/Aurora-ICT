"""Aurora-ICT 데이터 디렉토리 결정 — launcher 와 일관 위치.

OS 표준 hidden 위치:
    - Windows: ``%LOCALAPPDATA%\\Aurora``
    - macOS:   ``~/Library/Application Support/Aurora``
    - Linux / LOCALAPPDATA 미설정: ``~/.aurora-ict`` fallback (dev)

환경변수 ``AURORA_ICT_DATA_DIR`` 가 설정되어 있으면 그 값을 우선 사용 (테스트 / CI 격리).

launcher (`aurora_ict_launcher.launcher._aurora_ict_data_dir`) 와 동일 위치를 산출하지만
import 순환 회피를 위해 별도 모듈로 둠.
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

_LOCALAPPDATA_NAME = "Aurora"


def data_dir() -> Path:
    """Aurora-ICT 봇 데이터 디렉토리 (trades.jsonl / trades.db / 기타).

    Returns:
        Path. 존재 보장 안 함 — 호출자가 ``mkdir(parents=True, exist_ok=True)``.
    """
    override = os.environ.get("AURORA_ICT_DATA_DIR")
    if override:
        return Path(override)
    sys_name = platform.system()
    if sys_name == "Windows":
        local_app = os.environ.get("LOCALAPPDATA")
        if local_app:
            return Path(local_app) / _LOCALAPPDATA_NAME
    elif sys_name == "Darwin":
        return Path.home() / "Library" / "Application Support" / _LOCALAPPDATA_NAME
    # Linux / LOCALAPPDATA 미설정 fallback
    return Path.home() / ".aurora-ict"


__all__ = ["data_dir"]
