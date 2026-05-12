# PyInstaller spec — Aurora-ICT 단일 실행 파일.
#
# 빌드:
#   pyinstaller Aurora-ICT.spec --clean --noconfirm
#
# Windows 결과: dist/Aurora-ICT.exe
# macOS  결과: dist/Aurora-ICT.app

# ruff: noqa: F821 — PyInstaller 가 주입하는 Analysis/PYZ/EXE 변수
from pathlib import Path

BLOCK_CIPHER = None
PROJECT_ROOT = Path(SPECPATH).resolve()
SRC = PROJECT_ROOT / "src"
UI_DIR = PROJECT_ROOT / "ui_ict"
ASSETS_DIR = PROJECT_ROOT / "assets"
ICON_WIN = ASSETS_DIR / "aurora.ico"
ICON_MAC = ASSETS_DIR / "aurora.icns"

a = Analysis(
    [str(SRC / "aurora_ict_launcher" / "__main__.py")],
    pathex=[str(SRC)],
    binaries=[],
    datas=[
        # ui_ict 폴더 통째로 — launcher 가 /ui static mount 로 서빙
        (str(UI_DIR), "ui_ict"),
    ],
    hiddenimports=[
        "aurora_ict",
        "aurora_ict.api.app",
        "aurora_ict.bot.manager",
        "aurora_ict.bot.aurora_adapter",
        "aurora_ict.config.settings",
        "aurora_ict.updater",
        # ccxt 비동기 거래소 (Bybit) — PyInstaller 가 명시 import 만 잡음
        "ccxt.async_support.bybit",
        "ccxt.pro.bybit",
        # uvicorn 의존 (Windows asyncio loop)
        "uvicorn.loops.asyncio",
        "uvicorn.protocols.http.httptools_impl",
        "uvicorn.protocols.websockets.websockets_impl",
        "uvicorn.lifespan.on",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "tkinter",
        "matplotlib",
        "scipy",
        "PIL",
        "IPython",
        "notebook",
        "pytest",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=BLOCK_CIPHER,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=BLOCK_CIPHER)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="Aurora-ICT",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,           # Windows: 콘솔 창 X
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON_WIN) if ICON_WIN.exists() else None,
)

# macOS .app bundle (sys.platform == "darwin" 일 때만 PyInstaller 가 자동 사용)
import sys
if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name="Aurora-ICT.app",
        icon=str(ICON_MAC) if ICON_MAC.exists() else None,
        bundle_identifier="com.aurora.ict",
        info_plist={
            "CFBundleName": "Aurora-ICT",
            "CFBundleDisplayName": "Aurora · ICT",
            "CFBundleShortVersionString": "0.2.3",
            "CFBundleVersion": "0.2.3",
            "NSHighResolutionCapable": "True",
            "LSMinimumSystemVersion": "11.0",
        },
    )
