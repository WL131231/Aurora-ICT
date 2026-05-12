"""Aurora-ICT launcher — pywebview UI + uvicorn 백그라운드 서버.

실행 흐름:
1. 로그 설정
2. settings 로드 (.env)
3. BotManager 생성 (aurora_client_factory 주입)
4. FastAPI app 생성
5. uvicorn을 daemon thread로 띄움 (127.0.0.1:8765)
6. /ict/health 가 응답할 때까지 대기 (최대 10초)
7. pywebview 창 띄움 (http://127.0.0.1:8765/ui/)
8. 창 닫히면 cleanup

frozen (PyInstaller) 환경에서도 동일하게 동작 — sys.frozen 분기 X (uvicorn은 thread 안에서
직접 import).
"""

from __future__ import annotations

import logging
import os
import socket
import sys
import threading
import time
import urllib.request
from contextlib import closing
from pathlib import Path

import uvicorn
import webview

from aurora_ict.api.app import create_app
from aurora_ict.bot import aurora_client_factory
from aurora_ict.bot.manager import BotManager
from aurora_ict.config.settings import get_settings, reload_settings

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
READINESS_TIMEOUT_SEC = 10.0
WINDOW_TITLE = "Aurora · ICT"
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 800


def _setup_logging() -> None:
    level = os.environ.get("AURORA_ICT_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _is_port_free(host: str, port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        try:
            s.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, preferred: int) -> int:
    if _is_port_free(host, preferred):
        return preferred
    # 기본 포트 사용 중이면 OS가 빈 포트 할당
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def _wait_ready(url: str, timeout_sec: float) -> bool:
    """/ict/health 200 응답까지 폴링."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:  # noqa: S310 — local 127.0.0.1
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001
            pass
        time.sleep(0.2)
    return False


def _run_uvicorn(app, host: str, port: int) -> uvicorn.Server:
    """uvicorn을 background thread로 띄움. Server 객체 반환 (shutdown용)."""
    config = uvicorn.Config(app, host=host, port=port, log_level="info", access_log=False)
    server = uvicorn.Server(config)

    def _serve() -> None:
        try:
            server.run()
        except Exception as e:  # noqa: BLE001
            logger.exception("uvicorn 종료: %s", e)

    thread = threading.Thread(target=_serve, name="aurora-ict-uvicorn", daemon=True)
    thread.start()
    return server


def _exe_dir() -> Path:
    """frozen 환경에서는 executable 가 있는 폴더, dev 에서는 cwd 반환."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path.cwd()


def _find_env_file() -> Path | None:
    """exe 옆 → cwd → 부모 폴더 순으로 .env 탐색."""
    candidates = [
        _exe_dir() / ".env",
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def main() -> int:
    _setup_logging()
    logger.info("Aurora-ICT launcher 시작")

    env_file = _find_env_file()
    if env_file is not None:
        logger.info(".env 로드: %s", env_file)
        settings = reload_settings(env_file=env_file)
    else:
        logger.info(".env 없음 — 환경변수 / 기본값 사용")
        settings = get_settings()

    logger.info(
        "settings: mode=%s symbol=%s enabled=%s",
        settings.run_mode.value, settings.symbol, settings.enabled,
    )

    host = os.environ.get("AURORA_ICT_HOST", DEFAULT_HOST)
    preferred_port = int(os.environ.get("AURORA_ICT_PORT", str(DEFAULT_PORT)))
    port = _pick_port(host, preferred_port)
    if port != preferred_port:
        logger.warning("포트 %d 사용 중 — %d 로 fallback", preferred_port, port)

    manager = BotManager(client_factory=aurora_client_factory, settings=settings)
    app = create_app(manager)

    server = _run_uvicorn(app, host, port)

    health_url = f"http://{host}:{port}/ict/health"
    ui_url = f"http://{host}:{port}/ui/"

    if not _wait_ready(health_url, READINESS_TIMEOUT_SEC):
        logger.error("백엔드 준비 실패 (%ds) — UI 열지 않고 종료", READINESS_TIMEOUT_SEC)
        server.should_exit = True
        return 1

    logger.info("UI 열기: %s", ui_url)
    webview.create_window(
        title=WINDOW_TITLE,
        url=ui_url,
        width=WINDOW_WIDTH,
        height=WINDOW_HEIGHT,
        resizable=True,
    )
    try:
        webview.start()
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt")
    finally:
        logger.info("종료 처리 — uvicorn shutdown")
        server.should_exit = True
        # 짧게 대기 (graceful)
        for _ in range(20):
            if not server.started:
                break
            time.sleep(0.1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
