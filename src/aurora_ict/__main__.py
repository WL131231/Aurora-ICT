"""Aurora-ICT 진입점 — ``python -m aurora_ict`` 박힘 박힘 박힘.

박힌 거 박힘:
1. settings 박힘 박힘
2. BotManager 박힘 박힘 (aurora_client_factory 박힘)
3. FastAPI app 박힘 박힘
4. uvicorn 박힘 박힘 박힘 (host 박힘 ``127.0.0.1``, port 박힘 ``8765``)

UI 박힘 박힘 박힘 → ``http://127.0.0.1:8765/docs`` 박힘 박힘 박힘 박힘 박힘 박힘.

환경변수:
- ``AURORA_ICT_HOST`` (default ``127.0.0.1``)
- ``AURORA_ICT_PORT`` (default ``8765``)
- 박힘 박힘 ``AURORA_ICT_*`` settings 박힘 박힘 박힘 박힘 박힘
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
        "Aurora-ICT 박힘 박힘 — mode=%s symbol=%s enabled=%s",
        settings.run_mode.value, settings.symbol, settings.enabled,
    )

    manager = BotManager(
        client_factory=aurora_client_factory,
        settings=settings,
    )
    app = create_app(manager)

    host = os.environ.get("AURORA_ICT_HOST", "127.0.0.1")
    port = int(os.environ.get("AURORA_ICT_PORT", "8765"))

    logger.info("REST API 박힘 박힘 박힘 http://%s:%d/docs", host, port)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
