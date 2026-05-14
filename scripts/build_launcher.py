"""PyInstaller 빌드 스크립트 — Aurora-ICT Launcher 미니 .exe.

본체 (Aurora-ICT.exe) 와 별개 작은 wrapper:
    - pywebview + 표준 라이브러리만 (~10MB)
    - 본체 의존성 (numpy / pandas / ccxt / fastapi 등) 미포함

산출물:
    Windows: dist/Aurora-ICT-launcher.exe
    macOS:   dist/Aurora-ICT-launcher.app  (zip 으로 release 첨부)

사용법:
    python scripts/build_launcher.py

진입점: src/aurora_ict_launcher/launcher.py
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_SEP = os.pathsep  # Windows ';' / Unix ':'

PLATFORM_HIDDEN_IMPORTS = {
    "Windows": ["webview.platforms.winforms"],
    "Darwin": ["webview.platforms.cocoa"],
    "Linux": ["webview.platforms.gtk", "webview.platforms.qt"],
}


PLATFORM_ICON = {
    "Windows": "aurora.ico",  # v0.4.44 — 본체 .exe 와 동일 아이콘 (사용자 요청)
    "Darwin": "aurora.icns",
    "Linux": "aurora.png",
}
PLATFORM_ICON_FALLBACK = {
    "Windows": "aurora-launcher.ico",  # 본체 아이콘 미존재 시 기존 launcher 아이콘
    "Darwin": "aurora-launcher.icns",
    "Linux": "aurora-launcher.png",
}


def main() -> int:
    plat = platform.system()
    entry = PROJECT_ROOT / "src" / "aurora_ict_launcher" / "launcher.py"
    ui_dir = PROJECT_ROOT / "src" / "aurora_ict_launcher" / "ui"
    assets_dir = PROJECT_ROOT / "assets"

    cmd: list[str] = [
        sys.executable, "-m", "PyInstaller",
        "--name", "Aurora-ICT-launcher",
        "--windowed",
        "--paths", str(PROJECT_ROOT / "src"),
        # ui/ 번들 (HTML/CSS/JS) — _MEIPASS/ui 아래에 풀림
        "--add-data", f"{ui_dir}{DATA_SEP}ui",
        # v0.1.63: certifi CA bundle 명시 collect — frozen 환경 SSL 핸드셰이크 보장.
        # Why: ChoYoon Claude #133 환기 — frozen --windowed --onefile 일부 환경에서
        # ssl 모듈 CA path 깨짐 → urllib HTTPS 핸드셰이크 fail → "조회 실패" 일반화.
        "--collect-data", "certifi",
        # v0.2.18 (ChoYoon #133 P0 ⑦): Windows launcher 측 "SSL context: system default"
        # 박힘 = certifi import 측 fail (--collect-data 측 data 만 박고 module 측 X).
        # --hidden-import 측 PyInstaller import tracker 측 명시 박음 → frozen .exe
        # 측 import 정합.
        "--hidden-import", "certifi",
        "--clean",
        "--noconfirm",
        "--onefile",
    ]

    # 런처 아이콘 — 검정 다이아몬드 (assets/aurora-ict-launcher.ico, v0.1.14).
    # 미존재 시 본체 그라디언트 아이콘 fallback.
    icon_name = PLATFORM_ICON.get(plat)
    fallback_name = PLATFORM_ICON_FALLBACK.get(plat)
    chosen_icon = None
    if icon_name and (assets_dir / icon_name).exists():
        chosen_icon = assets_dir / icon_name
    elif fallback_name and (assets_dir / fallback_name).exists():
        chosen_icon = assets_dir / fallback_name
        print(f"[warn] launcher icon {icon_name} 미존재 → fallback {fallback_name}")
    if chosen_icon:
        cmd.extend(["--icon", str(chosen_icon)])

    for module in PLATFORM_HIDDEN_IMPORTS.get(plat, []):
        cmd.extend(["--hidden-import", module])

    # 본체 의존성 모두 exclude — launcher 는 표준 + pywebview 만
    for module in [
        "numpy", "pandas", "ccxt", "fastapi", "uvicorn", "pydantic",
        "matplotlib", "scipy", "pyarrow", "aurora_ict.backtest",
    ]:
        cmd.extend(["--exclude-module", module])

    cmd.append(str(entry))

    print(f"[platform] {plat} ({sys.version})")
    print("[exec]", " ".join(cmd))
    rc = subprocess.run(cmd, check=False).returncode

    # v0.1.74 (ChoYoon Claude #133 fix O): macOS launcher.app Info.plist 후처리.
    if rc == 0 and plat == "Darwin":
        _patch_launcher_info_plist()

    return rc


def _patch_launcher_info_plist() -> None:
    """v0.1.74: macOS launcher.app Info.plist 후처리 — LSMinimumSystemVersion."""
    import plistlib

    app_path = PROJECT_ROOT / "dist" / "Aurora-ICT-launcher.app"
    plist_path = app_path / "Contents" / "Info.plist"
    if not plist_path.exists():
        print(f"[warn] launcher Info.plist not found: {plist_path} (skip patch)")
        return
    try:
        with plist_path.open("rb") as f:
            plist = plistlib.load(f)
        plist["LSMinimumSystemVersion"] = "13.0"
        plist["NSHighResolutionCapable"] = True
        try:
            from aurora_ict_launcher import __version__ as _v
            plist["CFBundleVersion"] = _v
            plist["CFBundleShortVersionString"] = _v
        except ImportError:
            pass
        with plist_path.open("wb") as f:
            plistlib.dump(plist, f)
        print("[ok] launcher Info.plist patched: LSMinimumSystemVersion=13.0")
    except (OSError, plistlib.InvalidFileException) as e:
        print(f"[warn] launcher Info.plist patch failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
