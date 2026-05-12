"""Aurora-ICT 진입점 — ``python -m aurora_ict``.

수행 순서:
1. settings 로드
2. BotManager 생성 (aurora_client_factory 주입)
3. FastAPI app 생성
4. uvicorn 기동 (host 기본 ``127.0.0.1``, port 기본 ``8765``)

UI 접근 → ``http://127.0.0.1:8765/docs``.

환경변수:
- ``AURORA_ICT_HOST`` (default ``127.0.0.1``)
- ``AURORA_ICT_PORT`` (default ``8765``)
- 그 외 ``AURORA_ICT_*`` settings 전반
"""

from __future__ import annotations

import logging
import os

import uvicorn

from aurora_ict.api.app import create_app
from aurora_ict.bot import aurora_client_factory
from aurora_ict.bot.manager import BotManager
from aurora_ict.config.settings import get_settings


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("AURORA_ICT_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger("aurora_ict")

    settings = get_settings()
    logger.info(
        "Aurora-ICT 시작 — mode=%s symbol=%s enabled=%s",
        settings.run_mode.value, settings.symbol, settings.enabled,
    )

    manager = BotManager(
        client_factory=aurora_client_factory,
        settings=settings,
    )
    app = create_app(manager)

    host = os.environ.get("AURORA_ICT_HOST", "127.0.0.1")
    port = int(os.environ.get("AURORA_ICT_PORT", "8765"))

    logger.info("REST API 기동 — http://%s:%d/docs", host, port)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
