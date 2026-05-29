"""Aurora-ICT SaaS 진입점 — 다중 사용자 FastAPI 서버.

main.py 는 단일 사용자 (.exe + pywebview) 전용이고, 이 모듈은 Docker / Fly.io /
Railway / 자체 VPS 같은 클라우드 호스팅 전용. 차이점:

    - **pywebview 창 없음** — uvicorn 만 foreground. 호스트가 컨테이너 lifecycle 관리.
    - **MultiUserBotManager** — 사용자별 BotIctInstance 격리.
    - **multi_user=True** 로 ``create_app`` 호출 — ``/auth/*`` router + 보호 endpoint.
    - **환경변수 기반** — host/port/data dir/cookie secure 모두 ENV 로.
    - **self-update / launcher 분기 없음** — 컨테이너 이미지 교체로 업데이트.

환경변수 명세:
    - ``AURORA_ICT_HOST``            서버 바인드 호스트 (기본 ``0.0.0.0``)
    - ``AURORA_ICT_PORT``            바인드 포트 (기본 ``8765``)
    - ``AURORA_ICT_DATA_DIR``        ``users.db`` / ``master.key`` 보관 위치
                                     (Docker: ``/data`` volume 권장)
    - ``AURORA_ICT_MASTER_KEY``      Fernet 마스터 키 (44 글자 base64).
                                     없으면 ``<data_dir>/master.key`` 자동 생성.
    - ``AURORA_ICT_SECURE_COOKIE``   ``1`` (기본, HTTPS 환경) / ``0`` (로컬 HTTP)
    - ``AURORA_ICT_SAAS``            컨테이너 헬스체크용 노출 flag (런타임 영향 X)
    - ``AURORA_ICT_LOG_LEVEL``       기본 INFO

shutdown 흐름:
    SIGTERM (k8s / Docker stop) → uvicorn graceful shutdown → ``mu.stop_all()`` 로
    모든 사용자 봇 정지 + ccxt aiohttp 세션 close.

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import uvicorn

from aurora_ict.api.app import create_app
from aurora_ict.auth import pin, users_db
from aurora_ict.bot import aurora_client_factory
from aurora_ict.bot.multi_user_manager import MultiUserBotManager
from aurora_ict.config.settings import IctSettings
from aurora_ict.paths import data_dir

logger = logging.getLogger(__name__)


async def auto_resume_running_bots(
    mu: MultiUserBotManager,
    db_path: Path | str,
) -> dict[str, int]:
    """Fly machine 재시작 / 재배포 후 ``bot_running=1`` 사용자 자동 재가동.

    파트너 결정 2026-05-28 — 봇 인스턴스가 in-memory 라 OOM/재배포 시 사라짐.
    이 함수가 DB 의 영속 플래그를 읽고 best-effort 로 ``mu.start`` 호출.
    한 사용자 실패 (API 키 만료, 거래소 응답 거부 등) 가 다음 사용자 가동에
    영향 X — ExceptionGroup 사용 X (Python 3.11 호환).

    Args:
        mu: MultiUserBotManager 인스턴스 (start 메서드 호출용).
        db_path: users.db 경로 (list_running_codes 조회).

    Returns:
        ``{"attempted": int, "succeeded": int, "failed": int}`` 통계.
        startup 로그 + 테스트 검증용. 한 명도 가동 안 됐으면 attempted=0.
    """
    stats = {"attempted": 0, "succeeded": 0, "failed": 0}
    try:
        codes = users_db.list_running_codes(db_path)
    except Exception as e:  # noqa: BLE001
        logger.warning("bot 자동 재가동 — list_running_codes 실패: %s", e)
        return stats
    if not codes:
        logger.info("bot 자동 재가동 대상 없음 (bot_running=1 사용자 0명)")
        return stats
    stats["attempted"] = len(codes)
    logger.info("bot 자동 재가동 시도 — %d명", len(codes))
    # 2026-05-29 hot-fix: 사용자별 last_run_mode 로 강제 가동. LIVE 사용자가
    # base_settings.run_mode (DEMO 기본) 때문에 DEMO 키 미등록 ValueError 로
    # 봇 죽던 버그 해소. last_run_mode 미설정 (구식 사용자) 은 base 그대로.
    from aurora_ict.config.settings import RunMode
    for code in codes:
        try:
            last_mode_str = users_db.get_last_run_mode(db_path, code)
        except Exception as e:  # noqa: BLE001
            logger.warning("last_run_mode 조회 실패 — %s: %s", code, e)
            last_mode_str = None
        force_mode: RunMode | None = None
        if last_mode_str == "live":
            force_mode = RunMode.LIVE
        elif last_mode_str == "demo":
            force_mode = RunMode.DEMO
        try:
            await mu.start(code, force_run_mode=force_mode)
            stats["succeeded"] += 1
            logger.info(
                "bot 자동 재가동 — %s ✓ (run_mode=%s)",
                code, force_mode.value if force_mode else "base_default",
            )
        except Exception as e:  # noqa: BLE001
            stats["failed"] += 1
            # API 키 미등록 / 만료 / 거래소 응답 실패 등 — 다음 사용자에 영향 X.
            logger.warning("bot 자동 재가동 실패 — %s: %s", code, e)
    return stats


def _setup_logging() -> None:
    """SaaS 진입점 전용 로깅 — stdout 핸들러만.

    main.py 의 ``_setup_logging`` 은 LOCALAPPDATA 파일 핸들러를 추가하지만,
    컨테이너 환경에서는 stdout 으로 모으는 게 표준 (Docker / k8s log driver).
    """
    level = os.environ.get("AURORA_ICT_LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        force=True,
    )


def main() -> int:
    """SaaS 서버 부트스트랩 — DB init → manager 생성 → uvicorn 가동.

    Returns:
        exit code (uvicorn 가 graceful 종료되면 0).
    """
    _setup_logging()
    logger.info("Aurora-ICT SaaS 진입점 시작")

    dd = data_dir()
    dd.mkdir(parents=True, exist_ok=True)
    db_path = dd / "users.db"
    users_db.init_db(db_path)
    logger.info("users.db 경로: %s", db_path)

    # 세션 토큰 영속화 — Fly.io 재배포 후에도 사용자 로그인 상태 유지 (2026-05-28).
    # 이 호출 이전: pin 모듈은 메모리 dict 사용 (.exe 모드).
    # 이 호출 이후: pin 모듈의 create_session/validate_session/revoke_session 등 모두
    # users.db 의 sessions 테이블로 동작. 봇 프로세스가 죽었다 살아도 토큰 유효.
    pin.set_session_db_path(db_path)
    purged = users_db.cleanup_expired_sessions(db_path)
    logger.info(
        "세션 백엔드 = SQLite (sessions 테이블), startup cleanup 으로 %d 건 제거",
        purged,
    )

    # base_settings — 사용자별 IctSettings 가 이 값을 model_copy 후 api_key 만 덮어씀.
    # ENV 로 demo/live mode / leverage / symbol 등 운영 정책 통일 가능 (사용자별 override 추후).
    base_settings = IctSettings()
    logger.info(
        "base_settings: mode=%s symbol=%s timeframe=%s leverage=%s",
        base_settings.run_mode.value,
        base_settings.symbol,
        base_settings.timeframe,
        base_settings.leverage,
    )

    # client_factory 는 main.py 와 정확히 동일한 ``aurora_client_factory`` 재사용.
    # multi_user_manager 가 사용자별 settings 로 호출해 ccxt CcxtClient → AuroraClientAdapter
    # 생성. 시그니처 ``Callable[[IctSettings], Awaitable[ExchangeClientProtocol]]`` 일치.
    mu = MultiUserBotManager(
        client_factory=aurora_client_factory,
        db_path=db_path,
        base_settings=base_settings,
    )

    secure_cookie_env = os.environ.get("AURORA_ICT_SECURE_COOKIE", "1").strip()
    secure_cookie = secure_cookie_env not in ("0", "false", "False", "")
    logger.info("secure_cookie=%s (HTTPS 운영 시 1 권장)", secure_cookie)

    app = create_app(
        multi_user=True,
        multi_user_manager=mu,
        auth_db_path=db_path,
        secure_cookie=secure_cookie,
    )

    # FastAPI startup 이벤트 — Fly machine 재시작 / 재배포 후 자동 복원 (2026-05-28).
    # 실로직은 모듈 함수 ``auto_resume_running_bots`` 가 담당 (테스트 분리 용이).
    @app.on_event("startup")
    async def _auto_resume_hook() -> None:
        # 2026-05-29: 매매 로그 사용자별 격리 마이그레이션 — 1회 실행 (idempotent).
        # AURORA_ICT_LEGACY_TRADES_OWNER 환경변수가 가리키는 사용자 디렉토리로
        # 기존 /data/trades.* 를 이관. 환경변수 미설정이면 skip (안전).
        try:
            from aurora_ict.interfaces.trades_migration import (
                migrate_legacy_trades_to_user_dir,
            )
            from aurora_ict.paths import data_dir as _d
            mig = migrate_legacy_trades_to_user_dir(_d())
            logger.info("legacy trades 마이그레이션 결과 — %s", mig)
        except Exception as e:  # noqa: BLE001
            logger.warning("legacy trades 마이그레이션 실패 (무시, 봇 진행): %s", e)
        await auto_resume_running_bots(mu, db_path)

    # FastAPI shutdown 이벤트 — uvicorn graceful 종료 시 모든 사용자 봇 정지.
    #
    # Why stop_all 이 bot_running=0 을 덮지 않는가:
    #   shutdown 은 보통 "재배포로 새 machine 띄움" 이라 다음 부팅 때 자동 재가동
    #   되어야 함. stop() 은 DB 의 bot_running 을 0 으로 바꾸므로 그대로 호출하면
    #   재배포 후 자동 재가동이 끊김. 따라서 in-memory 봇만 정지하는 별도 경로 필요.
    #   ``stop_all`` 자체는 그대로 두고, 호출 전에 in-memory 봇만 정지하는 우회
    #   루프 사용.
    @app.on_event("shutdown")
    async def _on_shutdown() -> None:
        logger.info("SaaS shutdown 신호 — 모든 사용자 봇 정지 시작 (DB 가동 플래그 보존)")
        # DB 의 bot_running 을 덮지 않도록 stop_all 대신 in-memory 정지만.
        # 사용자가 명시 STOP 을 누른 게 아니라 단지 프로세스 graceful 종료이므로
        # 다음 부팅에서 다시 살아나야 정상.
        for code in list(mu.list_users()):
            slot = mu._slots.get(code)
            if slot is None:
                continue
            try:
                if slot.bot is not None:
                    await slot.bot.stop()
                await mu._close_client(slot)
            except Exception as e:  # noqa: BLE001
                logger.warning("shutdown: 사용자 %s 정지 실패 — %s", code, e)
        logger.info("SaaS shutdown 완료")

    host = os.environ.get("AURORA_ICT_HOST", "0.0.0.0")  # noqa: S104 — 컨테이너 바인드 의도
    port = int(os.environ.get("AURORA_ICT_PORT", "8765"))
    logger.info("uvicorn 시작 — http://%s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
