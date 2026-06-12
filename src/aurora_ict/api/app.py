"""FastAPI app — Aurora-ICT REST 엔드포인트.

제공 엔드포인트:
- ``GET  /ict/status`` — 봇 상태 조회
- ``POST /ict/start`` — 봇 기동
- ``POST /ict/stop`` — 봇 정지
- ``POST /ict/run-mode`` — demo/live 전환 (body: ``{"mode": "demo"|"live"}``)
- ``POST /ict/enabled`` — enabled toggle (body: ``{"enabled": true|false}``)
- ``GET  /ict/config`` — 현재 settings 노출 (api key는 마스킹)
- ``GET  /ict/markers`` — 최근 OHLCV 기반 chart markers (FVG/Sweep/MSS/Setup)
- ``GET  /ict/health`` — health probe

BotManager는 모듈 전역 싱글톤으로 주입한다 (production에선 ``lifespan`` 훅 사용 권장).
"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from ccxt.base.errors import AuthenticationError
from fastapi import Cookie, FastAPI, Header, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, SecretStr

from aurora_ict.api.markers import to_chart_markers
from aurora_ict.bot.manager import BotManager
from aurora_ict.bot.multi_user_manager import (
    _DEFAULT_SYMBOL,
    MAX_CHOICE_PAIRS,
    MAX_PAIRS_PER_USER,
    MultiUserBotManager,
)
from aurora_ict.bot.pair_registry import EXCLUDED_PAIRS, FIXED_PAIRS
from aurora_ict.config.settings import TRADE_TIMEFRAMES, IctSettings, RunMode
from aurora_ict.strategy.silver_bullet import Direction


def _rows_to_candles(rows: list[list[Any]]) -> list[dict[str, float]]:
    """OHLCV rows → 차트 candle dict 리스트.

    수만 봉(파트너 요청: 거래소 시작부터 현재까지) 동기 변환이 이벤트 루프를 막아
    health check(8765) timeout → Fly 재기동을 유발하므로, asyncio.to_thread 로
    호출해 이벤트 루프 밖에서 변환한다 (#OHLCV-NONBLOCK 2026-06-06).
    """
    return [
        {
            "time": int(r[0] // 1000),
            "open": float(r[1]),
            "high": float(r[2]),
            "low": float(r[3]),
            "close": float(r[4]),
        }
        for r in rows
    ]

logger = logging.getLogger(__name__)


class RunModeRequest(BaseModel):
    mode: str  # "demo" | "live"


class EnabledRequest(BaseModel):
    enabled: bool


class CredentialsRequest(BaseModel):
    mode: str  # "demo" | "live"
    api_key: str
    api_secret: str


class PositionCloseRequest(BaseModel):
    """Close By 수동 청산 — fraction 1.0 = 전체, 0.5 = 50%."""

    fraction: float = 1.0  # (0, 1]
    # 2026-06-10: 멀티 페어 — 어느 심볼 포지션을 청산할지. 미지정 시 기본 심볼.
    symbol: str | None = None


class TimeframeRequest(BaseModel):
    """매매 timeframe 변경 — 1h / 2h / 4h / 1d / 1w 중 하나."""

    timeframe: str


class DailyLossLimitRequest(BaseModel):
    """#SAFETY-1 일일 손실 한도 변경 — 자본 대비 % (0 = 비활성, 0~50)."""

    pct: float


class PreferencesRequest(BaseModel):
    """2026-06-10 사용자 언어/시간대 — 텔레그램 알림 출력용. 부분 업데이트."""

    language: str | None = None
    timezone: str | None = None


def _env_path() -> Path:
    """`.env` 위치 — frozen(.exe 옆) / dev(cwd) 분기."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / ".env"
    return Path.cwd() / ".env"


def _write_env_credentials(mode: str, api_key: str, api_secret: str) -> Path:
    """`.env` 파일에 키 갱신. 기존 라인 있으면 교체, 없으면 추가.

    Args:
        mode: "demo" 또는 "live".
        api_key / api_secret: 평문 키.

    Returns:
        실제 갱신된 .env 경로.
    """
    env_path = _env_path()
    lines: list[str] = []
    if env_path.is_file():
        lines = env_path.read_text(encoding="utf-8").splitlines()
    prefix = "AURORA_ICT_LIVE_" if mode == "live" else "AURORA_ICT_DEMO_"
    key_var = f"{prefix}API_KEY"
    secret_var = f"{prefix}API_SECRET"
    filtered = [
        ln for ln in lines
        if not ln.startswith(f"{key_var}=")
        and not ln.startswith(f"{secret_var}=")
    ]
    filtered.append(f"{key_var}={api_key}")
    filtered.append(f"{secret_var}={api_secret}")
    env_path.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    return env_path


def _settings_safe_dict(settings: IctSettings) -> dict[str, Any]:
    """settings → dict (api key는 직접 노출하지 않고 보유 여부만 표시)."""
    from aurora_ict import __version__
    return {
        "version": __version__,
        "run_mode": settings.run_mode.value,
        "enabled": settings.enabled,
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
        "leverage": settings.leverage,
        "position_pct_base": settings.position_pct_base,
        "position_pct_max": settings.position_pct_max,
        "position_pct_step": settings.position_pct_step,
        "multi_tf": settings.multi_tf,
        "multi_tf_ltf_lookback": settings.multi_tf_ltf_lookback,
        "enable_trail": settings.enable_trail,
        "trail_buffer_ratio": settings.trail_buffer_ratio,
        "use_market_entry": settings.use_market_entry,
        "min_sl_distance_pct": settings.min_sl_distance_pct,
        "min_rr": settings.min_rr,
        "fvg_min_size_pct": settings.fvg_min_size_pct,
        "step_interval_sec": settings.step_interval_sec,
        "ohlcv_limit": settings.ohlcv_limit,
        "daily_loss_limit_pct": settings.daily_loss_limit_pct,
        "daily_profit_limit_pct": settings.daily_profit_limit_pct,
        "has_demo_credentials": bool(
            settings.demo_api_key.get_secret_value()
            and settings.demo_api_secret.get_secret_value(),
        ),
        "has_live_credentials": bool(
            settings.live_api_key.get_secret_value()
            and settings.live_api_secret.get_secret_value(),
        ),
        "allowed_trade_timeframes": list(TRADE_TIMEFRAMES),
    }


_KILLZONE_LABEL = {
    "asian": "Asian", "london": "London", "ny_am": "NY AM",
    "london_close": "London Close", "pm": "NY PM",
}


def _compute_session_status() -> dict[str, str]:
    """현재 세션 상태 — 킬존/미장/None (좌상단 표시용, 2026-05-27 추가).

    우선순위: 킬존 안 → 'Kill zone : <name>' / NYSE 시간(09:30-16:00 ET, 평일)
    → 'U.S. stock market Open' / 둘 다 아님 → 'None'.
    """
    import time as _time
    from datetime import datetime, time
    from zoneinfo import ZoneInfo

    from aurora_ict.timing.killzone import classify_killzone

    now_ms = int(_time.time() * 1000)
    kz = classify_killzone(now_ms)
    if kz is not None:
        label = _KILLZONE_LABEL.get(kz.value, kz.value)
        return {"kind": "killzone", "label": f"Kill zone : {label}"}
    ny = datetime.fromtimestamp(now_ms / 1000.0, tz=ZoneInfo("America/New_York"))
    if ny.weekday() < 5 and time(9, 30) <= ny.time() <= time(16, 0):
        return {"kind": "us_open", "label": "U.S. stock market Open"}
    return {"kind": "none", "label": "None"}


def _status_dict(manager: BotManager) -> dict[str, Any]:
    """BotManager status → dict."""
    st = manager.status()
    return {
        "state": st.state.value,
        "run_mode": st.run_mode.value,
        "enabled": st.enabled,
        "symbol": st.symbol,
        "timeframe": manager.settings.timeframe,
        "has_credentials": st.has_credentials,
        "has_active_position": st.has_active_position,
        "last_setup_ts_ms": st.last_setup_ts_ms,
    }


# #SEC rate limiting — 인증 관련 민감 POST 엔드포인트에 IP 당 분당 시도 제한.
# brute-force / 무한 계정생성 / app-layer DoS 완화. 정상 사용자는 분당 5회 안에
# 로그인하므로 영향 없음. (Fly 단일 머신 가정 — 다중 확장 시 공유 저장소 필요.)
_RATE_LIMITED_PATHS = frozenset({"/auth/login", "/auth/setup-pin", "/admin/login"})
_AUTH_RATE_MAX = 5
_AUTH_RATE_WINDOW = 60.0


def _client_ip(request: Request) -> str:
    """프록시(Fly) 뒤 실제 클라이언트 IP 추출 — rate limit 키 용도.

    Fly.io 는 ``Fly-Client-IP`` 헤더로 원 IP 를 전달한다. 없으면
    ``X-Forwarded-For`` 첫 항목, 그것도 없으면 소켓 peer 로 폴백한다.
    """
    fly_ip = request.headers.get("fly-client-ip")
    if fly_ip:
        return fly_ip.strip()
    xff = request.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _register_multi_user_routes(
    app: FastAPI,
    *,
    mu_manager: MultiUserBotManager,
    auth_db_path: str | Path | None,
    secure_cookie: bool,
    master_key: bytes | None,
) -> None:
    """multi-user 모드 라우트 등록 — /auth/* + 사용자별 /ict/* 보호.

    /ict/health 는 인증 없이 응답 (health probe — 로드밸런서 용).
    그 외 /ict/* 는 ``require_auth`` dependency 로 보호.

    Args:
        app: FastAPI 인스턴스.
        mu_manager: 사용자별 봇 격리 관리자.
        auth_db_path: users.db 경로. None 이면 기본 위치.
        secure_cookie: cookie Secure 속성 (HTTPS).
        master_key: keystore Fernet 키 (테스트용 주입).
    """
    from fastapi import Depends

    from aurora_ict.auth.middleware import require_auth
    from aurora_ict.auth.router import create_auth_router
    from aurora_ict.paths import data_dir

    db_path = Path(auth_db_path) if auth_db_path else data_dir() / "users.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # /auth/* router — setup-pin / login / logout / status / api-keys.
    app.include_router(
        create_auth_router(
            db_path,
            secure_cookie=secure_cookie,
            master_key=master_key,
            on_keys_changed=mu_manager.stop_user,
        ),
    )

    # 2026-05-28: 공지 router — /ict/notice (인증) + /admin/notice (admin token).
    from aurora_ict.api.notice_router import create_notice_router
    app.include_router(create_notice_router(db_path))

    # 2026-05-29: 매매 로그 router — /ict/trades (본인) + /admin/trades (admin)
    #             + /admin/user/license (admin, 라이선스 정정용).
    # secure_cookie 는 main router 와 동일 정책 — Fly 는 HTTPS 강제라 True.
    from aurora_ict.api.trades_router import create_trades_router
    from aurora_ict.paths import data_dir as _ict_data_dir_fn
    app.include_router(
        create_trades_router(
            _ict_data_dir_fn(),
            require_auth,
            secure_cookie=secure_cookie,
            auth_db_path=db_path,
            seed_provider=mu_manager.get_seed_usdt,
        ),
    )

    # === #SEC 보안 미들웨어 (SaaS/multi-user 전용) ===
    from starlette.responses import JSONResponse

    from aurora_ict.auth.ratelimit import RateLimiter

    _auth_limiter = RateLimiter(max_hits=_AUTH_RATE_MAX, window_sec=_AUTH_RATE_WINDOW)

    @app.middleware("http")
    async def _rate_limit_mw(request: Request, call_next):  # type: ignore[no-untyped-def]
        """인증 민감 엔드포인트에 IP 단위 rate limit 적용 — 초과 시 429."""
        if request.method == "POST" and request.url.path in _RATE_LIMITED_PATHS:
            client_ip = _client_ip(request)
            if not _auth_limiter.allow(f"{request.url.path}:{client_ip}"):
                logger.warning(
                    "rate limit 초과 — path=%s ip=%s",
                    request.url.path, client_ip,
                )
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "요청이 너무 많습니다. 잠시 후 다시 시도해 주세요.",
                    },
                )
        return await call_next(request)

    @app.middleware("http")
    async def _security_headers_mw(request: Request, call_next):  # type: ignore[no-untyped-def]
        """응답에 기본 보안 헤더 주입 — 클릭재킹/MIME 스니핑/HTTPS 강제."""
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        # HSTS — Fly 가 force_https 하지만 헤더는 앱이 보내야 브라우저가 기억한다.
        resp.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
        return resp

    @app.get("/ict/health")
    async def health_mu() -> dict[str, str]:
        # 인증 없이 응답 — 외부 모니터링/로드밸런서 probe.
        return {"status": "ok"}

    @app.get("/ict/status")
    async def status_mu(
        symbol: str = Query(default=_DEFAULT_SYMBOL),
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        # 2026-05-29 PR C: ?symbol= 쿼리 — 멀티 페어 UI 가 페어별 상태 조회.
        # 미지정 시 BTC default (기존 UI 흐름 그대로 작동).
        return await mu_manager.status(user_code, symbol)

    @app.post("/ict/start")
    async def start_mu(
        symbol: str | None = Query(default=None),
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        # 2026-06-06: symbol 미지정(전체 START 버튼)이면 선호 페어(last_active_pairs)
        # 를 모두 복원 가동 — STOP→START 시 BTC 만 켜지던 문제 해소. symbol 명시
        # (페어 추가/모달)면 그 페어만 개별 가동(기존 동작).
        if not symbol:
            logger.info("/ict/start 전체 — code=%s 선호 페어 복원", user_code)
            try:
                restored = await mu_manager.start_preferred(user_code)
            except Exception as e:  # noqa: BLE001
                logger.exception("/ict/start 복원 예외 — code=%s", user_code)
                raise HTTPException(
                    status_code=500, detail=f"가동 중 내부 오류: {e}",
                ) from e
            logger.info("/ict/start 복원 — code=%s pairs=%s", user_code, restored)
            return await mu_manager.status(user_code)
        logger.info("/ict/start 호출 — code=%s symbol=%s", user_code, symbol)
        try:
            await mu_manager.start(user_code, symbol)
        except ValueError as e:
            logger.warning(
                "/ict/start ValueError — code=%s symbol=%s detail=%s",
                user_code, symbol, e,
            )
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:  # noqa: BLE001 — 모든 예외 로깅 후 500 변환.
            logger.exception(
                "/ict/start 예기치 못한 예외 — code=%s symbol=%s",
                user_code, symbol,
            )
            raise HTTPException(
                status_code=500,
                detail=f"{symbol} 가동 중 내부 오류: {e}",
            ) from e
        logger.info(
            "/ict/start 성공 — code=%s symbol=%s", user_code, symbol,
        )
        return await mu_manager.status(user_code, symbol)

    @app.post("/ict/stop")
    async def stop_mu(
        symbol: str = Query(default=_DEFAULT_SYMBOL),
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        # 2026-05-29 PR C: ?symbol= 쿼리 — 사용자가 BTC 만 정지하고 ETH 는
        # 계속 가동하는 부분 정지 가능. 미지정 시 BTC default. 페어 칩으로 끄는
        # 경우라 forget_preference=True(기본) — 선호에서도 제거.
        await mu_manager.stop(user_code, symbol)
        return await mu_manager.status(user_code, symbol)

    @app.post("/ict/stop-all")
    async def stop_all_mu(
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        # 2026-06-06: 전체 STOP 버튼 — 모든 가동 페어 정지하되 선호(last_active)는
        # 유지해서 START 시 복원되게 한다(개별 칩 stop 과 구분).
        stopped = await mu_manager.stop_user(user_code)
        logger.info("/ict/stop-all — code=%s stopped=%s", user_code, stopped)
        return await mu_manager.status(user_code)

    @app.get("/ict/running-pairs")
    async def running_pairs_mu(
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """사용자의 가동 중인 모든 페어 list — UI 페어 토글 동기화용.

        2026-05-29 PR C: UI 가 페어 토글 active 상태를 백엔드 진실값과 맞추기 위해
        호출. fly 재시작/다른 탭에서 가동 변경된 경우에도 정확히 반영.

        Returns:
            ``{"running_symbols": ["BTC/USDT:USDT", "ETH/USDT:USDT"]}``.
        """
        from aurora_ict.auth import users_db as _users_db_local
        try:
            syms = _users_db_local.get_bot_running_symbols(
                mu_manager.db_path, user_code,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "running-pairs 조회 실패 — %s: %s", user_code, e,
            )
            syms = []
        return {"running_symbols": syms}

    @app.get("/ict/markets")
    async def list_markets_mu(
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """거래 가능 페어 목록 — UI 페어 선택 드롭다운용 (페어 확장).

        거래대금 상위 N + 고정 7. 슬롯 client 미확보(첫 가동 전) 시엔 고정 7만
        반환되고, 봇 가동 후부터 전체 목록을 받는다. 검증 탈락 페어(BNB)는
        시세·목록 모두에서 제외.

        Returns:
            ``{"pairs": [...], "tickers": [...], "max_pairs": 10,
            "fixed_pairs": [고정7], "max_choice_pairs": 3}``.
            tickers 는 페어 선택기 시세 표시용(거래대금 정렬). 시세 조회
            전/실패 시 tickers 는 빈 배열, pairs 는 화이트리스트 폴백.
        """
        try:
            tickers = await mu_manager.list_market_tickers(user_code)
        except Exception as e:  # noqa: BLE001
            logger.warning("markets tickers 조회 실패 — %s: %s", user_code, e)
            tickers = []
        tickers = [t for t in tickers if t.get("symbol") not in EXCLUDED_PAIRS]
        if tickers:
            pairs = [t["symbol"] for t in tickers]
            for fixed in FIXED_PAIRS:
                if fixed not in pairs:
                    pairs.append(fixed)
        else:
            try:
                pairs = await mu_manager.list_tradable_pairs(user_code)
            except Exception as e:  # noqa: BLE001
                logger.warning("markets pairs 폴백 실패 — %s: %s", user_code, e)
                pairs = list(FIXED_PAIRS)
        pairs = [p for p in pairs if p not in EXCLUDED_PAIRS]
        return {
            "pairs": pairs,
            "tickers": tickers,
            "max_pairs": MAX_PAIRS_PER_USER,
            "fixed_pairs": list(FIXED_PAIRS),
            "max_choice_pairs": MAX_CHOICE_PAIRS,
        }

    # ------------------------------------------------------------------
    # 2026-05-28: SaaS 1차 출시 — UI 가 호출하는 /ict/* 전체를 multi-user 분기에도
    # 등록 (이전엔 health/status/start/stop 4개만 → 그 외 endpoint 가 SaaS 에서 404).
    #
    # 핵심 단순화 결정:
    #   - 사용자별 settings 분리는 추후. 현재는 mu_manager.base_settings 공유.
    #   - settings 갱신 (timeframe / daily_loss_limit / run_mode 등) 은 슬롯이 있으면
    #     슬롯 settings 갱신, 슬롯 없으면 base_settings 갱신 (다음 start 시 반영).
    #   - api_key/secret 은 사용자별 (/auth/api-keys 가 DB 에 별도 저장 — 이쪽은 손 안 댐).
    # ------------------------------------------------------------------

    def _user_settings(user_code: str) -> IctSettings:
        """사용자 슬롯의 settings — 없으면 base copy + last_run_mode 반영.

        get_or_create_bot 이 안 호출된 시점(예: 봇 미가동 시 /ict/config 조회) 에서도
        뭔가 반환해야 함. 슬롯이 없으면 base_settings 를 deep copy 한 뒤
        사용자별 last_run_mode (DB) 로 run_mode 정확히 박는다.

        2026-05-29 hot-fix: 이전엔 base_settings 직접 반환했는데, base_settings
        는 프로세스 전역 공유 객체라 사용자별 LIVE/DEMO 가 혼동되고 한 사용자가
        모드 전환 시 다른 사용자에게도 leak. 이제 사용자별 사본을 만들고
        last_run_mode 우선으로 run_mode 결정.
        """
        slot = mu_manager._slots.get((user_code, _DEFAULT_SYMBOL))
        if slot is not None:
            return slot.settings
        base = mu_manager.base_settings or IctSettings()
        s = base.model_copy(deep=True)
        # last_run_mode 로 정확히 박기 — 사용자가 LIVE 로 가동했었으면 LIVE.
        from aurora_ict.auth import users_db as _users_db_local
        try:
            last_mode = _users_db_local.get_last_run_mode(
                mu_manager.db_path, user_code,
            )
        except Exception:  # noqa: BLE001
            last_mode = None
        if last_mode == "live":
            s.run_mode = RunMode.LIVE
        elif last_mode == "demo":
            s.run_mode = RunMode.DEMO
        return s

    @app.get("/ict/config")
    async def get_config_mu(
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        return _settings_safe_dict(_user_settings(user_code))

    @app.post("/ict/run-mode")
    async def set_run_mode_mu(
        req: RunModeRequest,
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """demo / live 전환 — 해당 모드 키 없으면 400 으로 차단 (2026-05-29).

        Live 전환은 실거래 진입 위험 — UI 가 확인 모달을 띄우더라도 서버 단에서
        키 등록 여부를 다시 검증해 잘못된 키로 라이브 도달하는 사고 방지.
        """
        from aurora_ict.auth import users_db as _users_db
        try:
            mode = RunMode(req.mode)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"invalid mode: {req.mode}",
            ) from e
        # 안전 가드 — 전환 대상 모드의 API 키가 없으면 차단.
        target_key = "live" if mode is RunMode.LIVE else "demo"
        if not _users_db.has_api_keys(mu_manager.db_path, user_code, target_key):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"{target_key.upper()} 거래소 API 키가 등록되지 않았습니다. "
                    "키를 먼저 등록한 뒤 모드를 전환해 주세요."
                ),
            )
        # 2026-05-29 hot-fix: base_settings.run_mode 를 건드리지 않는다.
        # base_settings 는 프로세스 전역 공유 객체라 한 사용자가 LIVE 로 바꾸면
        # 다른 사용자의 모드 표시까지 LIVE 로 leak 되고, fly 재배포 후 ENV 기본값
        # (DEMO) 로 리셋되면 status() 가 다시 DEMO 로 떨어져 LIVE 사용자가
        # "또 데모로 넘어가고" 보이는 버그 발생.
        #
        # 대신 last_run_mode 를 즉시 DB 에 박고, 슬롯 폐기 후 재가동.
        # status() 가 slot 없을 때 last_run_mode 를 우선 본다 (모드 표시 정확).
        # auto_resume / 다음 start 흐름이 force_run_mode 로 정확하게 복원.
        slot = mu_manager._slots.get((user_code, _DEFAULT_SYMBOL))
        was_running = (
            slot is not None
            and slot.bot is not None
            and slot.bot.state.value == "running"
        )
        if was_running:
            await mu_manager.stop(user_code)
        # 슬롯 자체를 비워서 다음 start 가 깨끗한 settings 로 다시 빌드하게 함.
        # (slot.settings.run_mode 만 갈아끼면 client 객체는 이전 모드 키로 만들어진
        # 것이라 sign 실패 가능 — 슬롯 폐기가 안전.)
        if slot is not None:
            mu_manager._slots.pop((user_code, _DEFAULT_SYMBOL), None)
        # 사용자 마지막 run_mode 즉시 영속화 — status() 가 이 값을 우선 본다.
        try:
            _users_db.set_last_run_mode(mu_manager.db_path, user_code, mode.value)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "사용자 %s last_run_mode 박기 실패 (무시): %s", user_code, e,
            )
        if was_running:
            try:
                await mu_manager.start(user_code, force_run_mode=mode)
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e)) from e
        return await mu_manager.status(user_code)

    @app.post("/ict/enabled")
    async def set_enabled_mu(
        req: EnabledRequest,
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        # multi-user 에서는 enabled 토글 = base_settings.enabled 갱신 + start/stop 분기.
        # (BotManager.set_enabled 와 의미 동일하게 — False 면 stop, True 면 start.)
        slot = mu_manager._slots.get((user_code, _DEFAULT_SYMBOL))
        was_running = (
            slot is not None
            and slot.bot is not None
            and slot.bot.state.value == "running"
        )
        if slot is not None:
            slot.settings.enabled = req.enabled
        elif mu_manager.base_settings is not None:
            mu_manager.base_settings.enabled = req.enabled
        try:
            if not req.enabled and was_running:
                await mu_manager.stop(user_code)
            elif req.enabled and not was_running:
                await mu_manager.start(user_code)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return await mu_manager.status(user_code)

    @app.post("/ict/test-connection")
    async def test_connection_mu(
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """현재 mode 의 API 키로 거래소 연결 테스트 — multi-user 사용자별 settings 기준.

        슬롯이 없으면 DB 에서 사용자 키를 가져와 settings 구성 (봇은 만들지 않음).
        """
        slot = mu_manager._slots.get((user_code, _DEFAULT_SYMBOL))
        if slot is not None:
            settings = slot.settings
        else:
            try:
                settings = mu_manager._build_user_settings(user_code)
            except ValueError as e:
                # API 키 미등록 → 사용자에게 안내.
                return {
                    "ok": False,
                    "mode": (mu_manager.base_settings or IctSettings()).run_mode.value,
                    "error": str(e),
                }
        mode = settings.run_mode.value
        if not settings.has_credentials():
            return {
                "ok": False,
                "mode": mode,
                "error": f"{mode.upper()} API 키가 등록되어 있지 않음",
            }
        try:
            client = await mu_manager.client_factory(settings)
        except Exception as e:  # noqa: BLE001
            logger.warning("test-connection client 생성 실패: %s", e)
            return {"ok": False, "mode": mode, "error": f"client 생성 실패: {e}"}
        try:
            bal = await client.fetch_balance()
        except Exception as e:  # noqa: BLE001
            logger.warning("test-connection fetch_balance 실패: %s", e)
            return {"ok": False, "mode": mode, "error": f"fetch_balance 실패: {e}"}
        usdt_total: float | None = None
        if isinstance(bal, dict):
            usdt = bal.get("USDT")
            if isinstance(usdt, dict):
                v = usdt.get("total")
                if isinstance(v, (int, float)):
                    usdt_total = float(v)
            if usdt_total is None:
                v = bal.get("total")
                if isinstance(v, (int, float)):
                    usdt_total = float(v)
        return {"ok": True, "mode": mode, "balance_usdt": usdt_total}

    @app.post("/ict/timeframe")
    async def set_trade_timeframe_mu(
        req: TimeframeRequest,
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """매매 timeframe 변경 — multi-user 사용자별 settings + 가동 중이면 재시작."""
        if req.timeframe not in TRADE_TIMEFRAMES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"timeframe '{req.timeframe}' 미지원 — "
                    f"허용 목록: {list(TRADE_TIMEFRAMES)}"
                ),
            )
        slot = mu_manager._slots.get((user_code, _DEFAULT_SYMBOL))
        was_running = (
            slot is not None
            and slot.bot is not None
            and slot.bot.state.value == "running"
        )
        if was_running:
            await mu_manager.stop(user_code)
        if slot is not None:
            slot.settings.timeframe = req.timeframe
        elif mu_manager.base_settings is not None:
            mu_manager.base_settings.timeframe = req.timeframe
        # 2026-06-08: 사용자별 영속화 — 재시작/재가동이 전역 base(기본값)로
        # 회귀하지 않게. _build_user_settings 가 이 값을 복원한다.
        from aurora_ict.auth import users_db as _users_db_tf
        _users_db_tf.set_last_timeframe(db_path, user_code, req.timeframe)
        logger.info(
            "[multi-user] %s trade timeframe 변경 → %s", user_code, req.timeframe,
        )
        if was_running:
            await mu_manager.start(user_code)
        settings = _user_settings(user_code)
        return {
            "timeframe": settings.timeframe,
            "allowed": list(TRADE_TIMEFRAMES),
            "restarted": was_running,
        }

    @app.post("/ict/preferences")
    async def set_preferences_mu(
        req: PreferencesRequest,
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """사용자 언어/시간대 저장 — 텔레그램 알림이 이 값으로 출력 (2026-06-10).

        UI 언어/시간대는 그동안 브라우저에만 있어 서버 발송(텔레그램)이 몰랐다.
        """
        from zoneinfo import ZoneInfo  # noqa: PLC0415

        from aurora_ict.auth import users_db as _users_db_pref  # noqa: PLC0415
        if req.language is not None and req.language not in ("ko", "en", "zh", "ja"):
            raise HTTPException(status_code=400, detail="지원 언어: ko/en/zh/ja")
        if req.timezone is not None:
            try:
                ZoneInfo(req.timezone)
            except Exception as e:  # noqa: BLE001
                raise HTTPException(
                    status_code=400, detail=f"invalid timezone: {req.timezone}",
                ) from e
        _users_db_pref.set_user_prefs(
            db_path, user_code, req.language, req.timezone,
        )
        return {"ok": True}

    @app.get("/ict/daily_loss_limit")
    async def get_daily_loss_limit_mu(
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """#SAFETY-1 한도 + 오늘 누적 손익 상태 — 사용자 봇이 있으면 그 값, 없으면 settings.

        2026-06-11 리뷰 수정: 기본 심볼(BTC) 슬롯 고정 조회 → ETH 만 가동 중인
        사용자도 오늘 손익이 보이게 사용자 슬롯 아무거나(가동 중 우선) 사용.
        """
        user_slots = [
            s for (u, _sym), s in mu_manager._slots.items()
            if u == user_code and s.bot is not None
        ]
        settings = _user_settings(user_code)
        if user_slots:
            running = [s for s in user_slots if s.bot.state.value == "running"]
            return (running[0] if running else user_slots[0]).bot.daily_loss_status()
        return {
            "limit_pct": settings.daily_loss_limit_pct,
            "today_pnl_usdt": 0.0,
            "today_pct": 0.0,
            "start_equity": 0.0,
            "hit": False,
            "date_ny": "",
            "profit_limit_pct": settings.daily_profit_limit_pct,
            "profit_hit": False,
        }

    @app.post("/ict/daily_profit_limit")
    async def set_daily_profit_limit_mu(
        req: DailyLossLimitRequest,
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """일일 수익(TP) 한도 설정 — 사용자 slot.settings + 가동 중 bot 갱신.

        2026-06-10 조윤 건의: "하루 몇 % 먹으면 그날 종료". 0 = 비활성, 0~100.
        """
        if req.pct < 0 or req.pct > 100:
            raise HTTPException(
                status_code=400,
                detail="daily_profit_limit_pct 는 0~100 범위 (0 = 비활성).",
            )
        # 2026-06-11 리뷰 수정: ① BTC 슬롯 고정 → 사용자 *전 슬롯* 순회 (ETH/HYPE
        # 봇에도 반영). ② 슬롯 없을 때 base_settings(전 사용자 공유 객체) 기록
        # 제거 — 다른 사용자 봇 기본값을 오염시키던 크로스 테넌트 버그.
        for (u, _sym), slot in list(mu_manager._slots.items()):
            if u != user_code:
                continue
            slot.settings.daily_profit_limit_pct = req.pct
            if slot.bot is not None:
                slot.bot.daily_profit_limit_pct = req.pct
                if (
                    slot.bot._daily_profit_hit
                    and not slot.bot._is_daily_profit_limit_hit()
                ):
                    slot.bot._daily_profit_hit = False
                    logger.info(
                        "[multi-user] %s/%s daily profit limit 한도 ↑ — hit flag 해제",
                        user_code, _sym,
                    )
        logger.info(
            "[multi-user] %s daily profit limit 설정 → %.2f%%", user_code, req.pct,
        )
        return {"profit_limit_pct": req.pct}

    @app.post("/ict/daily_loss_limit")
    async def set_daily_loss_limit_mu(
        req: DailyLossLimitRequest,
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """#SAFETY-1 한도 설정 — 사용자 slot.settings + 가동 중 bot in-memory 갱신."""
        if req.pct < 0 or req.pct > 50:
            raise HTTPException(
                status_code=400,
                detail="daily_loss_limit_pct 는 0~50 범위 (0 = 비활성).",
            )
        # 2026-06-11 리뷰 수정: 사용자 전 슬롯 순회 + base_settings 오염 제거
        # (위 daily_profit_limit 와 동일 — 크로스 테넌트 누출 방지).
        for (u, _sym), slot in list(mu_manager._slots.items()):
            if u != user_code:
                continue
            slot.settings.daily_loss_limit_pct = req.pct
            if slot.bot is not None:
                slot.bot.daily_loss_limit_pct = req.pct
                if (
                    slot.bot._daily_limit_hit
                    and not slot.bot._is_daily_loss_limit_hit()
                ):
                    slot.bot._daily_limit_hit = False
                    logger.info(
                        "[multi-user] %s/%s daily loss limit 한도 ↑ — hit flag 해제",
                        user_code, _sym,
                    )
        logger.info(
            "[multi-user] %s daily loss limit 설정 → %.2f%%", user_code, req.pct,
        )
        return {"limit_pct": req.pct}

    @app.post("/ict/credentials")
    async def set_credentials_mu(
        req: CredentialsRequest,
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """API 키 등록 — multi-user 에서는 /auth/api-keys 가 정식 경로.

        호환을 위해 이 endpoint 도 노출하지만, /auth/api-keys 로 안내.
        """
        raise HTTPException(
            status_code=400,
            detail=(
                "multi-user 모드에서는 /auth/api-keys 를 사용해 주세요 "
                "(.env 직접 기록 X — DB 암호화 저장)."
            ),
        )

    @app.post("/ict/position/close")
    async def close_position_mu(
        req: PositionCloseRequest,
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """Close By 수동 청산 — multi-user 사용자별 봇 기준."""
        if not (0.0 < req.fraction <= 1.0):
            raise HTTPException(
                status_code=400,
                detail=f"fraction은 (0, 1] 범위 — 받은 값: {req.fraction}",
            )
        slot = mu_manager._slots.get(
            (user_code, req.symbol or _DEFAULT_SYMBOL),
        )
        bot = slot.bot if slot is not None else None
        if bot is None or bot.active_position is None:
            raise HTTPException(status_code=404, detail="active position 없음")
        ap = bot.active_position
        close_qty = ap.qty * req.fraction
        close_side = "sell" if ap.direction is Direction.LONG else "buy"
        try:
            await bot.client.place_order(
                symbol=bot.symbol,
                side=close_side,
                qty=close_qty,
                price=None,
                stop_loss=None,
                take_profit=None,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "[multi-user] %s close_position place_order 실패: %s",
                user_code, e,
            )
            raise HTTPException(
                status_code=502, detail=f"청산 주문 실패: {e}",
            ) from e
        remaining = ap.qty - close_qty
        if req.fraction >= 1.0 or remaining <= 1e-9:
            bot.active_position = None
            logger.info(
                "[multi-user] %s 전체 청산 완료 — qty=%.6f", user_code, close_qty,
            )
            return {
                "closed_qty": close_qty, "remaining_qty": 0.0, "active": False,
            }
        ap.qty = remaining
        logger.info(
            "[multi-user] %s 부분 청산 — closed=%.6f remaining=%.6f",
            user_code, close_qty, remaining,
        )
        return {
            "closed_qty": close_qty, "remaining_qty": remaining, "active": True,
        }

    async def _build_position_payload(
        user_code: str, bot: Any,
    ) -> dict[str, Any]:
        """단일 봇(페어)의 포지션 상세 — active/pending 포함, symbol 항상 명시.

        멀티 페어(BTC/ETH/HYPE) 동시 포지션을 슬롯별로 만들 때 재사용.
        """
        pending: dict[str, Any] | None = None
        if bot._pending_entry is not None:
            pe = bot._pending_entry
            pending = {
                "direction": pe.direction.value,
                "entry": pe.entry,
                "stop_loss": pe.stop_loss,
                "take_profit": pe.take_profit,
                "qty": pe.qty,
                "placed_ts_ms": pe.placed_ts_ms,
            }
        if bot.active_position is None:
            return {"active": False, "symbol": bot.symbol, "pending": pending}

        ap = bot.active_position
        ex_unrealized: float | None = None
        ex_entry = ap.entry
        try:
            ex_pos = await bot.client.fetch_position(bot.symbol)
            if ex_pos:
                up = ex_pos.get("unrealized_pnl")
                if up is None:
                    up = ex_pos.get("unrealizedPnl")
                if isinstance(up, (int, float)):
                    ex_unrealized = float(up)
                ep = (
                    ex_pos.get("entry_price")
                    or ex_pos.get("entryPrice")
                    or ex_pos.get("avgPrice")
                )
                if isinstance(ep, (int, float)) and float(ep) > 0:
                    ex_entry = float(ep)
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "[multi-user] %s %s 거래소 포지션 PnL fetch 실패: %s",
                user_code, bot.symbol, e,
            )

        mark_price = ex_entry
        try:
            rows = await bot.client.fetch_ohlcv(bot.symbol, bot.timeframe, 2)
            if rows:
                mark_price = float(rows[-1][4])
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "[multi-user] %s %s mark_price fetch 실패: %s",
                user_code, bot.symbol, e,
            )

        if ex_unrealized is not None:
            unrealized = ex_unrealized
        elif ap.direction is Direction.LONG:
            unrealized = (mark_price - ex_entry) * ap.qty
        else:
            unrealized = (ex_entry - mark_price) * ap.qty

        # 2026-06-11 리뷰 수정: 기본 심볼 settings 가 아니라 *이 봇*의 leverage.
        # 알트는 슬롯 생성 시 15배 강제라 BTC 설정(예 50배) 기준 청산가/마진이
        # 완전히 틀리게 표시되던 문제.
        lev = int(getattr(bot, "leverage", 0)) or _user_settings(user_code).leverage
        notional = ex_entry * ap.qty
        margin = notional / lev
        if ap.direction is Direction.LONG:
            liq_price = ex_entry * (1.0 - (1.0 / lev) * 0.95)
        else:
            liq_price = ex_entry * (1.0 + (1.0 / lev) * 0.95)
        roi_pct = (unrealized / margin * 100.0) if margin > 0 else 0.0
        return {
            "active": True,
            "symbol": bot.symbol,
            "direction": ap.direction.value,
            "entry": ex_entry,
            "stop_loss": ap.stop_loss,
            "take_profit": ap.take_profit,
            "qty": ap.qty,
            "setup_ts_ms": ap.setup_ts_ms,
            "mark_price": mark_price,
            "unrealized_pnl": unrealized,
            "roi_pct": roi_pct,
            "margin": margin,
            "notional": notional,
            "liquidation_price": liq_price,
            "leverage": lev,
            "pending": pending,
        }

    @app.get("/admin/positions")
    async def admin_all_positions(
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        cookie_token: str | None = Cookie(default=None, alias="aurora_admin_token"),
    ) -> dict[str, Any]:
        """전 사용자 열린 포지션/대기주문 한눈에 — 운영 사고 대응용 (2026-06-12).

        파트너 요청: 6/12 #LIQ-CAP 사고(SL 이 청산가 밖) 같은 위험 포지션을
        사용자별로 일일이 안 보고 한 번에 점검. SL 이 청산가 밖이면
        ``sl_beyond_liq=True`` 경고 플래그.
        """
        from aurora_ict.api.trades_router import _check_admin_cookie_or_header
        _check_admin_cookie_or_header(cookie_token, x_admin_token)
        rows: list[dict[str, Any]] = []
        # 2026-06-12 파트너 요청 v2: 실시간 PnL — 심볼당 현재가 1회 조회(캐시)로
        # 슬롯 수와 무관하게 거래소 호출을 심볼 수로 제한.
        mark_cache: dict[str, float | None] = {}

        async def _mark(bot: Any, sym: str) -> float | None:
            if sym in mark_cache:
                return mark_cache[sym]
            price: float | None = None
            try:
                ohlcv = await bot.client.fetch_ohlcv(sym, "1m", 2)
                if ohlcv:
                    price = float(ohlcv[-1][4])
            except Exception as e:  # noqa: BLE001
                logger.debug("[admin/positions] %s 현재가 조회 실패: %s", sym, e)
            mark_cache[sym] = price
            return price

        for (u, sym), slot in list(mu_manager._slots.items()):
            bot = slot.bot
            if bot is None:
                continue
            lev = int(getattr(bot, "leverage", 0)) or 1
            for kind, pos in (
                ("active", bot.active_position),
                ("pending", bot._pending_entry),
            ):
                if pos is None:
                    continue
                entry = float(pos.entry)
                qty = float(pos.qty)
                is_long = pos.direction is Direction.LONG
                liq = entry * (1.0 - 0.95 / lev) if is_long else entry * (1.0 + 0.95 / lev)
                sl = float(pos.stop_loss)
                # 2026-06-12 리뷰 #2: SL=0(미상)도 위험 — 숏은 0>=liq 가 False 라
                # 빠지던 비대칭 수정 (미추적 행과 동일 기준).
                beyond = sl <= 0 or ((sl <= liq) if is_long else (sl >= liq))
                # 실시간 PnL — active 만 (pending 은 미체결이라 손익 없음).
                mark = unreal = roi = None
                if kind == "active":
                    mark = await _mark(bot, sym)
                    if mark is not None:
                        unreal = (mark - entry) * qty if is_long else (entry - mark) * qty
                        margin = entry * qty / lev
                        roi = (unreal / margin * 100.0) if margin > 0 else 0.0
                rows.append({
                    "user_code": u,
                    "symbol": sym,
                    "kind": kind,
                    "direction": "long" if is_long else "short",
                    "entry": entry,
                    "qty": qty,
                    "stop_loss": sl,
                    "take_profit": float(pos.take_profit),
                    "leverage": lev,
                    "liq_price": round(liq, 6),
                    "sl_beyond_liq": beyond,
                    "bot_state": bot.state.value,
                    "mark_price": mark,
                    "unrealized_pnl": unreal,
                    "roi_pct": roi,
                })
        # 2026-06-12 파트너 보고: 봇 미추적 포지션(수동 진입·재기동 고아)이 안
        # 보임 — 사용자별 거래소 계정 전체를 1회 스캔해 미추적분을 보충한다.
        tracked = {
            (r["user_code"], r["symbol"]) for r in rows if r["kind"] == "active"
        }
        seen_users: set[str] = set()
        for (u, _sym), slot in list(mu_manager._slots.items()):
            if u in seen_users or slot.bot is None:
                continue
            seen_users.add(u)
            fetch_all = getattr(slot.client, "fetch_all_positions", None)
            if not callable(fetch_all):
                continue
            try:
                ex_positions = await fetch_all()
            except Exception as e:  # noqa: BLE001
                logger.debug("[admin/positions] %s 계정 스캔 실패: %s", u, e)
                continue
            for p in ex_positions:
                psym = str(p.get("symbol") or "")
                if not psym or (u, psym) in tracked:
                    continue
                entry = float(p.get("entryPrice") or 0.0)
                qty = float(p.get("qty") or p.get("contracts") or 0.0)
                side = str(p.get("side") or "").lower()
                if entry <= 0 or qty <= 0 or side not in ("long", "short"):
                    continue
                lev = int(float(p.get("leverage") or 0) or 0) or 1
                is_long = side == "long"
                liq = float(p.get("liquidationPrice") or 0.0) or (
                    entry * (1.0 - 0.95 / lev) if is_long
                    else entry * (1.0 + 0.95 / lev)
                )
                info = p.get("info") or {}
                sl = float(info.get("stopLoss") or 0.0)
                tp = float(info.get("takeProfit") or 0.0)
                # SL 없거나 청산가 밖이면 위험 — 미추적은 봇 보호 SL 도 없다.
                beyond = sl <= 0 or ((sl <= liq) if is_long else (sl >= liq))
                mark = float(p.get("markPrice") or 0.0) or None
                unreal = p.get("unrealizedPnl")
                if not isinstance(unreal, (int, float)) and mark is not None:
                    unreal = (
                        (mark - entry) * qty if is_long else (entry - mark) * qty
                    )
                margin = entry * qty / lev
                roi = (
                    float(unreal) / margin * 100.0
                    if isinstance(unreal, (int, float)) and margin > 0 else None
                )
                rows.append({
                    "user_code": u,
                    "symbol": psym,
                    "kind": "untracked",
                    "direction": side,
                    "entry": entry,
                    "qty": qty,
                    "stop_loss": sl,
                    "take_profit": tp,
                    "leverage": lev,
                    "liq_price": round(liq, 6),
                    "sl_beyond_liq": beyond,
                    "bot_state": "untracked",
                    "mark_price": mark,
                    "unrealized_pnl": (
                        float(unreal) if isinstance(unreal, (int, float)) else None
                    ),
                    "roi_pct": roi,
                })
        # 위험(청산가 밖/무 SL) → 미추적 → 코드 순 정렬.
        rows.sort(key=lambda r: (
            not r["sl_beyond_liq"], r["kind"] != "untracked", r["user_code"],
        ))
        total_pnl = sum(
            r["unrealized_pnl"] for r in rows
            if isinstance(r.get("unrealized_pnl"), (int, float))
        )
        return {"positions": rows, "count": len(rows),
                "risky": sum(1 for r in rows if r["sl_beyond_liq"]),
                "untracked": sum(1 for r in rows if r["kind"] == "untracked"),
                "total_unrealized_pnl": round(total_pnl, 4)}

    @app.post("/admin/trades/backfill")
    async def admin_trades_backfill(
        days: int = Query(default=7, ge=1, le=30),
        code: str | None = Query(default=None),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        cookie_token: str | None = Cookie(default=None, alias="aurora_admin_token"),
    ) -> dict[str, Any]:
        """유실 청산 기록 백필 — 거래소 closed-pnl 로 보충 (2026-06-12 파트너).

        6/12 새벽 스냅샷 복원으로 유실 창(~11h)의 매매 기록이 사라져 통계가
        거래소와 어긋남. 사용자별(미지정 시 슬롯 보유 전체) 최근 ``days``일
        closed-pnl 을 읽어, 기록에 없는 청산을 SYNC_CLOSE 로 채운다.

        중복 방지: 같은 symbol·방향·청산시각 ±10분·수량 ±2% 의 기존 청산
        이벤트가 있으면 skip — 정상 기록 구간을 다시 돌려도 안전(멱등).
        """
        import time as _time

        from aurora_ict.api.trades_router import _check_admin_cookie_or_header
        from aurora_ict.interfaces.trades_store import (
            TradeEvent,
            TradeEventType,
            TradesStore,
        )
        _check_admin_cookie_or_header(cookie_token, x_admin_token)
        since_ms = int(_time.time() * 1000) - days * 86_400_000
        close_types = {
            TradeEventType.SL_HIT, TradeEventType.TP_HIT,
            TradeEventType.SYNC_CLOSE, TradeEventType.MANUAL_CLOSE,
            TradeEventType.FLIP_CLOSE,
        }
        # 사용자당 첫 슬롯 client 로 계정 단위 closed-pnl 조회.
        targets: dict[str, Any] = {}
        for (u, _s), slot in list(mu_manager._slots.items()):
            if slot.bot is None or (code and u != code):
                continue
            targets.setdefault(u, slot)
        results: dict[str, str] = {}
        for u, slot in targets.items():
            try:
                cps = await slot.client.fetch_closed_positions(
                    since_ms=since_ms, limit=200,
                )
            except Exception as e:  # noqa: BLE001
                results[u] = f"closed-pnl 조회 실패: {e}"
                continue
            try:
                store = TradesStore(mu_manager._user_data_dir(u))
                events = store.all_events()
            except Exception as e:  # noqa: BLE001
                results[u] = f"기록 접근 실패: {e}"
                continue
            existing = [
                (ev.symbol, ev.direction, ev.ts_ms, ev.qty)
                for ev in events if ev.event_type in close_types
            ]
            added = 0
            failed = False
            for cp in cps or []:
                c_ts = int(getattr(cp, "closed_at_ts", 0) or 0)
                sym = str(getattr(cp, "symbol", "") or "")
                dirn = str(getattr(cp, "direction", "") or "")
                qty = float(getattr(cp, "qty", 0) or 0)
                if (c_ts < since_ms or not sym
                        or dirn not in ("long", "short") or qty <= 0):
                    continue
                dup = any(
                    s == sym and d == dirn
                    and abs(t - c_ts) <= 600_000
                    and abs(q - qty) <= max(qty * 0.02, 1e-9)
                    for (s, d, t, q) in existing
                )
                if dup:
                    continue
                ev = TradeEvent(
                    ts_ms=c_ts,
                    event_type=TradeEventType.SYNC_CLOSE,
                    symbol=sym,
                    direction=dirn,
                    price=float(getattr(cp, "exit_price", 0) or 0),
                    qty=qty,
                    pnl_usdt=float(getattr(cp, "pnl_usd", 0) or 0),
                    setup_ts_ms=None,
                    reason=(
                        "backfill: 거래소 closed-pnl 보충 "
                        f"(entry={float(getattr(cp, 'entry_price', 0) or 0)})"
                    ),
                    mode="live",
                )
                try:
                    store.record(ev)
                except Exception as e:  # noqa: BLE001
                    results[u] = f"기록 쓰기 실패: {e}"
                    failed = True
                    break
                existing.append((sym, dirn, c_ts, qty))
                added += 1
            if not failed:
                results[u] = f"+{added}건"
            logger.info("[admin] 백필 — %s: %s", u, results.get(u))
        return {"ok": True, "days": days, "results": results}

    @app.post("/admin/position/close")
    async def admin_force_close(
        code: str = Query(...),
        symbol: str = Query(...),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        cookie_token: str | None = Cookie(default=None, alias="aurora_admin_token"),
    ) -> dict[str, Any]:
        """admin 강제 청산 — 위험 포지션(TP=0 복구분 등) 즉시 시장가 정리.

        2026-06-12 파트너 요청. 검증된 비상청산 경로(_emergency_close:
        거래소 실방향·수량 fetch 후 reduce_only, 실패 시 포지션 인식 유지)를
        재사용하고 매매 기록 사유만 'admin 강제 청산'으로 남긴다.
        """
        from aurora_ict.api.trades_router import _check_admin_cookie_or_header
        _check_admin_cookie_or_header(cookie_token, x_admin_token)
        slot = mu_manager._slots.get((code, symbol))
        bot = slot.bot if slot is not None else None
        if bot is None or bot.active_position is None:
            raise HTTPException(
                status_code=404, detail="해당 사용자×페어의 활성 포지션 없음",
            )
        logger.warning(
            "[admin] 강제 청산 요청 — %s %s (%s)",
            code, symbol, bot.active_position.direction.value,
        )
        await bot._emergency_close(reason="admin 강제 청산")
        closed = bot.active_position is None
        return {"ok": closed, "code": code, "symbol": symbol,
                "detail": "청산 완료" if closed
                else "청산 주문 실패 — 봇이 다음 step 에서 재동기화"}

    @app.get("/ict/position")
    async def get_position_mu(
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """현재 active/pending 포지션 — 사용자의 모든 페어 슬롯 순회(멀티 페어).

        2026-06-10 버그픽스: 기존엔 기본 심볼(BTC) 슬롯 하나만 조회해 ETH/HYPE
        동시 포지션이 UI 에 안 보였다. 이제 사용자의 전 슬롯을 모아 ``positions``
        list 로 반환. 레거시 단일 필드는 호환 위해 기본 심볼(없으면 첫 항목)로 유지.
        """
        user_slots = sorted(
            (
                (sym, slot)
                for (u, sym), slot in mu_manager._slots.items()
                if u == user_code
            ),
            key=lambda x: x[0],
        )
        positions: list[dict[str, Any]] = []
        for _sym, slot in user_slots:
            bot = slot.bot if slot is not None else None
            if bot is None:
                continue
            payload = await _build_position_payload(user_code, bot)
            # active 도 pending 도 없는 페어는 표에서 제외(빈 행 방지).
            if payload.get("active") or payload.get("pending"):
                positions.append(payload)

        legacy = next(
            (p for p in positions if p.get("symbol") == _DEFAULT_SYMBOL),
            positions[0] if positions else {"active": False, "pending": None},
        )
        return {**legacy, "positions": positions}

    @app.post("/ict/pending/cancel")
    async def cancel_pending_mu(
        symbol: str = Query(default=_DEFAULT_SYMBOL),
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """대기 중인 지정가 entry 주문 즉시 취소 (사용자 명령, 2026-05-30).

        UI 의 positions 표 'CANCEL' 버튼이 호출. 봇이 거래소 cancel_all_orders
        + 내부 ``_pending_entry`` 비움. active 포지션엔 영향 X.

        2026-06-10: 멀티 페어 — ``symbol`` 로 어느 페어 대기주문인지 지정.
        """
        slot = mu_manager._slots.get((user_code, symbol))
        bot = slot.bot if slot is not None else None
        if bot is None:
            return {"ok": False, "reason": "bot_not_running"}
        ok = await bot.cancel_pending_entry()
        return {"ok": ok}

    @app.get("/ict/judgment")
    async def get_judgment_mu(
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """봇 현재 판단 — multi-user 사용자별 (FVG/구조/Sweep + 세션 + entry condition)."""
        slot = mu_manager._slots.get((user_code, _DEFAULT_SYMBOL))
        bot = slot.bot if slot is not None else None
        if bot is None:
            return {
                "direction": {
                    "long_pct": 50, "short_pct": 50, "label": "봇 미가동",
                },
                "reasons": [],
                "entry_condition": {
                    "title": "봇 미가동", "detail": "START 눌러 가동",
                },
            }
        settings = _user_settings(user_code)
        htf_map = bot._htf_fvg_map_cache or []
        bull_w = sum(int(e.weight) for e in htf_map if e.type.value == "bullish")
        bear_w = sum(int(e.weight) for e in htf_map if e.type.value == "bearish")
        tot = bull_w + bear_w
        if tot > 0:
            long_pct = round(bull_w / tot * 100, 1)
            short_pct = round(100 - long_pct, 1)
        else:
            long_pct = short_pct = 50.0
        if long_pct >= 60:
            dir_label = "롱(상승) 우세"
        elif short_pct >= 60:
            dir_label = "숏(하락) 우세"
        else:
            dir_label = "방향 불확실 (혼조)"
        cur_price = 0.0
        try:
            rows = await bot.client.fetch_ohlcv(bot.symbol, bot.timeframe, 2)
            if rows:
                cur_price = float(rows[-1][4])
        except Exception:  # noqa: BLE001
            pass
        reasons: list[dict[str, Any]] = []
        if cur_price > 0 and htf_map:
            bulls = [
                e for e in htf_map
                if e.type.value == "bullish" and e.high < cur_price
            ]
            bears = [
                e for e in htf_map
                if e.type.value == "bearish" and e.low > cur_price
            ]
            bulls.sort(key=lambda e: cur_price - e.high)
            bears.sort(key=lambda e: e.low - cur_price)
            for e in bulls[:2]:
                reasons.append({
                    "color": "green",
                    "term": f"FVG {e.tf}",
                    "range": f"{e.high:,.0f} ~ {e.low:,.0f}",
                    "interpretation": "롱 지지 가능성 (가격이 끌릴 빈 공간)",
                })
            for e in bears[:2]:
                reasons.append({
                    "color": "red",
                    "term": f"FVG {e.tf}",
                    "range": f"{e.high:,.0f} ~ {e.low:,.0f}",
                    "interpretation": "위쪽 저항 (가격 도달 시 막힐 위험)",
                })
        try:
            use_tf = bot.timeframe
            ohlcv_rows = await bot.client.fetch_ohlcv(bot.symbol, use_tf, 200)
            if ohlcv_rows:
                df = pd.DataFrame(
                    ohlcv_rows,
                    columns=["ts_ms", "open", "high", "low", "close", "volume"],
                )
                df.index = pd.DatetimeIndex(
                    pd.to_datetime(df["ts_ms"], unit="ms", utc=True),
                )
                df = df[["open", "high", "low", "close", "volume"]]
                _mk = to_chart_markers(
                    df,
                    min_rr=settings.min_rr,
                    fvg_min_size_pct=settings.fvg_min_size_pct,
                )
                struct_src = _mk.large_structure or _mk.structure
                if struct_src:
                    s = struct_src[-1]
                    struct_meta = {
                        "bos_bullish":   ("green",  "BOS↑",    "상승 추세 지속 (이전 고점 돌파)"),
                        "bos_bearish":   ("red",    "BOS↓",    "하락 추세 지속 (이전 저점 돌파)"),
                        "choch_bullish": ("green",  "CHoCH↑",  "추세 전환 — 상승 우세 (전환 신호)"),
                        "choch_bearish": ("red",    "CHoCH↓",  "추세 전환 — 하락 우세 (전환 신호)"),
                    }
                    color, term, interp = struct_meta.get(
                        s.type, ("yellow", s.type, "구조 변화"),
                    )
                    reasons.append({
                        "color": color, "term": term,
                        "range": f"{s.broken_level:,.0f}",
                        "interpretation": interp,
                    })
                ob_src = _mk.large_order_blocks or _mk.order_blocks
                obs = [o for o in ob_src if not o.mitigated]
                if cur_price > 0 and obs:
                    bull_obs = [
                        o for o in obs
                        if o.type == "bullish" and o.high < cur_price
                    ]
                    bear_obs = [
                        o for o in obs
                        if o.type == "bearish" and o.low > cur_price
                    ]
                    bull_obs.sort(key=lambda o: cur_price - o.high)
                    bear_obs.sort(key=lambda o: o.low - cur_price)
                    for o in bull_obs[:1]:
                        reasons.append({
                            "color": "green", "term": "OB",
                            "range": f"{o.high:,.0f} ~ {o.low:,.0f}",
                            "interpretation": "롱 지지 가능성 (기관 매수 흔적)",
                        })
                    for o in bear_obs[:1]:
                        reasons.append({
                            "color": "red", "term": "OB",
                            "range": f"{o.high:,.0f} ~ {o.low:,.0f}",
                            "interpretation": "위쪽 저항 (기관 매도 흔적)",
                        })
                sweep_src = _mk.large_sweeps or _mk.sweeps
                if sweep_src:
                    sw = sweep_src[-1]
                    if sw.type == "bullish":
                        reasons.append({
                            "color": "yellow", "term": "SSL Sweep",
                            "range": f"{sw.wick_price:,.0f}",
                            "interpretation": "하단 유동성 흡수 — 반등 가능성 (롱 손절 청산)",
                        })
                    elif sw.type == "bearish":
                        reasons.append({
                            "color": "yellow", "term": "BSL Sweep",
                            "range": f"{sw.wick_price:,.0f}",
                            "interpretation": "상단 유동성 흡수 — 하락 가능성 (숏 손절 청산)",
                        })
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "[multi-user] %s judgment markers 추출 실패: %s", user_code, e,
            )
        session = _compute_session_status()
        if session["kind"] == "killzone":
            reasons.append({
                "color": "green", "term": "Killzone",
                "range": session["label"].replace("Kill zone : ", ""),
                "interpretation": "활발한 매매 시간대 (진입 윈도우)",
            })
        elif session["kind"] == "us_open":
            reasons.append({
                "color": "green", "term": "US Session",
                "range": "09:30-16:00 ET",
                "interpretation": "미장 개장 — 변동성 ↑",
            })
        else:
            reasons.append({
                "color": "yellow", "term": "Session", "range": "Off-hours",
                "interpretation": "킬존 밖 — 진입 자제 권장 (referral 은 24h 허용)",
            })
        min_cf = settings.min_confluence
        min_rr = settings.min_rr
        if bot.active_position is not None:
            ec_title = "진입 중 — 신규 진입 차단 (페어당 1포지션)"
            ec_detail = f"방향 {bot.active_position.direction.value}, SL/TP 진행"
        elif bot._pending_entry is not None:
            pe = bot._pending_entry
            ec_title = f"지정가 미체결 대기 ({pe.entry:,.2f})"
            ec_detail = "10분 안 체결 안 되면 자동 취소"
        else:
            ec_title = f"등급 {min_cf} 이상 & RR {min_rr} 이상 setup 대기"
            ec_detail = "현재 활성 setup 없음 (혹은 게이트 미달로 skip)"
        # 2026-05-29: 진단 인지 보강. 파트너 의문 "long 비율 우세인데 short 만 진입"
        # 직접 해소 — recent setup direction 분포 + 봇 내부 진단 카운터 노출.
        recent_dirs = list(getattr(bot, "_recent_setup_directions", []) or [])
        recent_long = sum(1 for d in recent_dirs if d == "long")
        recent_short = sum(1 for d in recent_dirs if d == "short")
        return {
            "direction": {
                "long_pct": long_pct, "short_pct": short_pct,
                "label": dir_label, "bull_weight": bull_w, "bear_weight": bear_w,
                # 사용자 직관 보정 — 위 비율은 HTF FVG 가중치 기준.
                # 실제 진입은 LTF + HTF override + EMA + DOL + 등급 조합 결과로
                # 위 비율과 다를 수 있다.
                "hint": (
                    "위 비율은 HTF FVG 가중치 기준 (장기 구조). "
                    "실제 진입 방향은 LTF + HTF override + EMA + DOL + 등급 필터 "
                    "조합 결과로 위 비율과 다를 수 있음."
                ),
                "recent_setups": {
                    "long": recent_long, "short": recent_short,
                    "last": recent_dirs[-1] if recent_dirs else None,
                },
            },
            "reasons": reasons,
            "entry_condition": {"title": ec_title, "detail": ec_detail},
            # 2026-05-29 #SILENT-*: 조용한 오류 가시화 — 운영 중 실패 위치 즉시 확인.
            "diagnostics": {
                "recovery_failed": bool(getattr(bot, "_recovery_failed", False)),
                "sync_failure_streak": int(
                    getattr(bot, "_sync_failure_streak", 0),
                ),
                "order_failure_count": int(
                    getattr(bot, "_order_failure_count", 0),
                ),
                "tpsl_failure_streak": int(
                    getattr(bot, "_tpsl_failure_streak", 0),
                ),
            },
        }

    @app.get("/ict/equity")
    async def get_equity_mu(
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """현재 잔고(USDT) + 세션 상태 — multi-user 사용자별."""
        slot = mu_manager._slots.get((user_code, _DEFAULT_SYMBOL))
        bot = slot.bot if slot is not None else None
        session = _compute_session_status()
        if bot is None:
            return {"equity": 0.0, "active": False, "session_status": session}
        try:
            eq = await bot._fetch_equity()
        except Exception as e:  # noqa: BLE001
            logger.debug(
                "[multi-user] %s equity fetch 실패: %s", user_code, e,
            )
            return {
                "equity": 0.0, "active": True, "error": "fetch_failed",
                "session_status": session,
            }
        return {"equity": float(eq), "active": True, "session_status": session}

    @app.get("/ict/closed_pnl")
    async def get_closed_pnl_mu(
        limit: int = 50,
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """최근 청산 거래 내역 — multi-user 사용자별 (30일 윈도우)."""
        slot = mu_manager._slots.get((user_code, _DEFAULT_SYMBOL))
        bot = slot.bot if slot is not None else None
        if bot is None:
            return {"trades": []}
        import time as _time
        since_ms = int(_time.time() * 1000) - 30 * 24 * 60 * 60 * 1000
        try:
            closed = await bot.client.fetch_closed_positions(
                since_ms=since_ms, limit=max(1, min(int(limit), 200)),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[multi-user] %s closed_pnl fetch 실패: %s", user_code, e,
            )
            return {"trades": []}
        trades: list[dict[str, Any]] = []
        for cp in closed[: max(1, min(int(limit), 200))]:
            trades.append({
                "symbol": getattr(cp, "symbol", None),
                "direction": getattr(cp, "direction", None),
                "entry_price": getattr(cp, "entry_price", None),
                "exit_price": getattr(cp, "exit_price", None),
                "qty": getattr(cp, "qty", None),
                "pnl_usd": getattr(cp, "pnl_usd", None),
                "roi_pct": getattr(cp, "roi_pct", None),
                "leverage": getattr(cp, "leverage", None),
                "closed_at_ts": getattr(cp, "closed_at_ts", None),
                "opened_at_ts": getattr(cp, "opened_at_ts", None),
            })
        return {"trades": trades}

    @app.get("/ict/markers")
    async def get_markers_mu(
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int = 1500,
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """OHLCV 기반 chart markers — multi-user 사용자별.

        2026-05-28 lazy 로딩 — START 전에도 차트 보이게. 봇 인스턴스 없으면
        ``ensure_bot_ready`` 로 자동 생성 + prefetch 시작 (가동 X).
        API 키 미등록 시 404 응답.
        """
        try:
            bot = await mu_manager.ensure_bot_ready(user_code)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        use_symbol = symbol or bot.symbol
        use_tf = timeframe or bot.timeframe
        try:
            rows = await bot.client.fetch_ohlcv(use_symbol, use_tf, limit)
        except AuthenticationError as e:
            # 키 무효(10003) — 502 대신 빈 마커 반환. 차트는 마커 없이 표시되고
            # 사용자는 키 재등록 유도. (502 는 콘솔 에러 + ASGI exception 유발)
            logger.warning(
                "[multi-user] %s markers 인증 실패(키 무효): %s", user_code, e,
            )
            rows = []
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "[multi-user] %s fetch_ohlcv 실패: %s", user_code, e,
            )
            raise HTTPException(status_code=502, detail=f"fetch_ohlcv: {e}") from e
        df = pd.DataFrame(
            rows,
            columns=["ts_ms", "open", "high", "low", "close", "volume"],
        )
        df.index = pd.DatetimeIndex(pd.to_datetime(df["ts_ms"], unit="ms", utc=True))
        df = df[["open", "high", "low", "close", "volume"]]
        settings = _user_settings(user_code)
        markers = to_chart_markers(
            df,
            min_rr=settings.min_rr,
            fvg_min_size_pct=settings.fvg_min_size_pct,
        )
        return {
            "symbol": use_symbol,
            "timeframe": use_tf,
            "count": {
                "fvgs": len(markers.fvgs),
                "sweeps": len(markers.sweeps),
                "structure": len(markers.structure),
                "swings": len(markers.swings),
                "killzones": len(markers.killzones),
                "setups": len(markers.setups),
                "order_blocks": len(markers.order_blocks),
                "macros": len(markers.macros),
                "trailing": 1 if markers.trailing else 0,
                "internal_swings": len(markers.internal_swings),
                "internal_structure": len(markers.internal_structure),
                "large_swings": len(markers.large_swings),
                "large_structure": len(markers.large_structure),
                "equal_levels": len(markers.equal_levels),
            },
            "markers": markers.to_dict(),
        }

    @app.get("/ict/ohlcv")
    async def get_ohlcv_mu(
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int = 1500,
        user_code: str = Depends(require_auth),
    ) -> dict[str, Any]:
        """OHLCV 봉 — multi-user 사용자별 (bot prefetch cache 활용).

        2026-05-28 lazy 로딩 — START 전에도 차트 보이게. 봇 인스턴스 없으면
        ``ensure_bot_ready`` 로 자동 생성 + prefetch 시작 (가동 X).
        API 키 미등록 시 404 응답.
        """
        try:
            bot = await mu_manager.ensure_bot_ready(user_code)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        use_symbol = symbol or bot.symbol
        use_tf = timeframe or bot.timeframe
        try:
            if use_symbol == bot.symbol:
                rows = await bot.get_ohlcv_cached(use_tf, limit)
            else:
                rows = await bot.client.fetch_ohlcv(use_symbol, use_tf, limit)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"fetch_ohlcv: {e}") from e
        # 수만 봉 변환은 이벤트 루프 밖에서 (#OHLCV-NONBLOCK) — health timeout 방지.
        candles = await asyncio.to_thread(_rows_to_candles, rows)
        return {"symbol": use_symbol, "timeframe": use_tf, "candles": candles}

    # Static UI mount — SaaS 도 ui_ict 그대로 서빙 (브라우저 → cookie 세션 흐름).
    # single-user 분기와 동일 패턴 (frozen 케이스는 SaaS 에서 없음 — dev / Docker layout 만).
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "ui_ict")
    candidates.append(Path(__file__).resolve().parents[3] / "ui_ict")
    for ui_dir in candidates:
        if ui_dir.is_dir():
            app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")
            logger.info("[multi-user] UI mounted from %s", ui_dir)
            break


def create_app(
    manager: BotManager | None = None,
    *,
    multi_user: bool = False,
    multi_user_manager: MultiUserBotManager | None = None,
    auth_db_path: str | Path | None = None,
    secure_cookie: bool = False,
    master_key: bytes | None = None,
) -> FastAPI:
    """FastAPI app 인스턴스 생성.

    Args:
        manager: single-user BotManager. ``multi_user=False`` 면 필수.
        multi_user: True 면 SaaS 모드 — 인증 router 등록 + 사용자별 bot 라우팅.
        multi_user_manager: ``multi_user=True`` 일 때 필수. 사용자별 봇 격리 관리.
        auth_db_path: ``multi_user=True`` 시 users.db 경로. None 이면
            ``aurora_ict.paths.data_dir() / "users.db"`` 자동 사용.
        secure_cookie: 세션 cookie ``Secure`` 속성 (HTTPS 운영 시 True).
        master_key: keystore Fernet 키 (테스트용). None 이면 기본 동작.

    Returns:
        FastAPI app — multi_user 분기 따라 router/dependency 다름.
    """
    if multi_user:
        if multi_user_manager is None:
            raise ValueError(
                "multi_user=True 면 multi_user_manager 인자가 필수입니다.",
            )
    else:
        if manager is None:
            raise ValueError(
                "multi_user=False 면 manager (BotManager) 인자가 필수입니다.",
            )

    # #SEC 운영(SaaS, multi_user)에서는 /docs·/redoc·/openapi.json 을 비활성화한다.
    # 전체 API 스키마(admin 엔드포인트·헤더 이름 포함)가 인증 없이 노출되면 공격
    # 표면 정찰에 직접 쓰인다. 데스크탑(.exe single)은 로컬 전용이라 그대로 유지.
    app = FastAPI(
        title="Aurora-ICT API",
        version="0.2.1",
        description="ICT (Inner Circle Trader) 매매 봇 REST API",
        docs_url=None if multi_user else "/docs",
        redoc_url=None if multi_user else "/redoc",
        openapi_url=None if multi_user else "/openapi.json",
    )

    # CORS — SaaS(multi_user)는 운영 도메인만 허용(쿠키 세션 + credentials 사용이라
    # wildcard 와 함께 두면 위험). 데스크탑(.exe single)은 localhost/file 접속이라
    # 편의상 wildcard 유지.
    _cors_origins = (
        ["https://aurora-ict-one.fly.dev"] if multi_user else ["*"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # multi-user 모드 — /auth/* router + 사용자별 보호 endpoint 등록 (single 와 분리).
    if multi_user:
        _register_multi_user_routes(
            app,
            mu_manager=multi_user_manager,  # type: ignore[arg-type]
            auth_db_path=auth_db_path,
            secure_cookie=secure_cookie,
            master_key=master_key,
        )
        return app

    # 이하 single-user 흐름 — 기존 동작 그대로 (CLI/.exe 호환).
    assert manager is not None  # 위에서 검증됨 (multi_user=False → manager 필수)

    @app.get("/ict/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ict/status")
    async def get_status() -> dict[str, Any]:
        return _status_dict(manager)

    @app.get("/ict/config")
    async def get_config() -> dict[str, Any]:
        return _settings_safe_dict(manager.settings)

    @app.post("/ict/start")
    async def start_bot() -> dict[str, Any]:
        # 2026-05-27: ENABLE 버튼 제거 후 START 가 enable + start 둘 다 담당.
        try:
            await manager.set_enabled(True)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        try:
            await manager.start()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _status_dict(manager)

    @app.post("/ict/stop")
    async def stop_bot() -> dict[str, Any]:
        # 2026-05-27: STOP 은 stop + disable (다음 START 까지 진입 차단).
        await manager.stop()
        try:
            await manager.set_enabled(False)
        except ValueError:
            pass
        return _status_dict(manager)

    @app.post("/ict/run-mode")
    async def set_run_mode(req: RunModeRequest) -> dict[str, Any]:
        try:
            mode = RunMode(req.mode)
        except ValueError as e:
            raise HTTPException(
                status_code=400, detail=f"invalid mode: {req.mode}",
            ) from e
        await manager.set_run_mode(mode)
        return _status_dict(manager)

    @app.post("/ict/enabled")
    async def set_enabled(req: EnabledRequest) -> dict[str, Any]:
        try:
            await manager.set_enabled(req.enabled)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _status_dict(manager)

    @app.post("/ict/test-connection")
    async def test_connection() -> dict[str, Any]:
        """현재 mode의 API 키로 거래소 연결 테스트.

        client_factory 로 client 생성 → fetch_balance() 호출.
        성공: {ok: true, balance_usdt, mode}
        실패: {ok: false, error, mode}
        """
        mode = manager.settings.run_mode.value
        if not manager.settings.has_credentials():
            return {
                "ok": False,
                "mode": mode,
                "error": f"{mode.upper()} API 키가 등록되어 있지 않음",
            }
        try:
            client = await manager.client_factory(manager.settings)
        except Exception as e:  # noqa: BLE001
            logger.warning("test-connection client 생성 실패: %s", e)
            return {"ok": False, "mode": mode, "error": f"client 생성 실패: {e}"}
        try:
            bal = await client.fetch_balance()
        except Exception as e:  # noqa: BLE001
            logger.warning("test-connection fetch_balance 실패: %s", e)
            return {"ok": False, "mode": mode, "error": f"fetch_balance 실패: {e}"}

        # USDT 잔고 추출 (ccxt 형식)
        usdt_total: float | None = None
        if isinstance(bal, dict):
            usdt = bal.get("USDT")
            if isinstance(usdt, dict):
                v = usdt.get("total")
                if isinstance(v, (int, float)):
                    usdt_total = float(v)
            if usdt_total is None:
                v = bal.get("total")
                if isinstance(v, (int, float)):
                    usdt_total = float(v)
        return {
            "ok": True,
            "mode": mode,
            "balance_usdt": usdt_total,
        }

    @app.post("/ict/timeframe")
    async def set_trade_timeframe(req: TimeframeRequest) -> dict[str, Any]:
        """매매 timeframe 변경 — 1h/2h/4h/1d/1w 만 허용.

        가동 중인 봇이 있으면 stop → settings 갱신 → start 로 매끄럽게 재시작.
        """
        if req.timeframe not in TRADE_TIMEFRAMES:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"timeframe '{req.timeframe}' 미지원 — "
                    f"허용 목록: {list(TRADE_TIMEFRAMES)}"
                ),
            )
        was_running = manager.bot is not None and manager.bot.state.value == "running"
        if was_running:
            await manager.stop()
        manager.settings.timeframe = req.timeframe
        logger.info("trade timeframe 변경 → %s", req.timeframe)
        if was_running:
            await manager.start()
        return {
            "timeframe": manager.settings.timeframe,
            "allowed": list(TRADE_TIMEFRAMES),
            "restarted": was_running,
        }

    @app.get("/ict/daily_loss_limit")
    async def get_daily_loss_limit() -> dict[str, Any]:
        """#SAFETY-1 한도 + 오늘 누적 손익 상태 (UI 좌측 박스).

        Returns:
            limit_pct / today_pnl_usdt / today_pct / start_equity / hit / date_ny.
        """
        bot = manager.bot
        if bot is not None:
            status = bot.daily_loss_status()
        else:
            status = {
                "limit_pct": manager.settings.daily_loss_limit_pct,
                "today_pnl_usdt": 0.0,
                "today_pct": 0.0,
                "start_equity": 0.0,
                "hit": False,
                "date_ny": "",
                "profit_limit_pct": manager.settings.daily_profit_limit_pct,
                "profit_hit": False,
            }
        return status

    @app.post("/ict/daily_profit_limit")
    async def set_daily_profit_limit(req: DailyLossLimitRequest) -> dict[str, Any]:
        """일일 수익(TP) 한도 설정 (단일 사용자) — 0 = 비활성, 0~100 자본 %."""
        if req.pct < 0 or req.pct > 100:
            raise HTTPException(
                status_code=400,
                detail="daily_profit_limit_pct 는 0~100 범위 (0 = 비활성).",
            )
        manager.settings.daily_profit_limit_pct = req.pct
        if manager.bot is not None:
            manager.bot.daily_profit_limit_pct = req.pct
            if (
                manager.bot._daily_profit_hit
                and not manager.bot._is_daily_profit_limit_hit()
            ):
                manager.bot._daily_profit_hit = False
                logger.info("daily profit limit 한도 ↑ — hit flag 해제")
        logger.info("daily profit limit 설정 → %.2f%%", req.pct)
        return {"profit_limit_pct": req.pct}

    @app.post("/ict/daily_loss_limit")
    async def set_daily_loss_limit(req: DailyLossLimitRequest) -> dict[str, Any]:
        """#SAFETY-1 한도 설정 — 0 = 비활성, 0~50 자본 % .

        settings + 가동 중 bot 양쪽 in-memory 갱신. .env 파일은 손 안 댐
        (다음 봇 재시작 시 .env 값이 다시 박힘 — 영구 적용은 .env 직접 수정).
        """
        if req.pct < 0 or req.pct > 50:
            raise HTTPException(
                status_code=400,
                detail="daily_loss_limit_pct 는 0~50 범위 (0 = 비활성).",
            )
        manager.settings.daily_loss_limit_pct = req.pct
        if manager.bot is not None:
            manager.bot.daily_loss_limit_pct = req.pct
            # 한도 ↑ 변경 시 이미 hit 였던 flag 해제 (사용자 의도).
            if manager.bot._daily_limit_hit and not manager.bot._is_daily_loss_limit_hit():
                manager.bot._daily_limit_hit = False
                logger.info("daily loss limit 한도 ↑ — hit flag 해제")
        logger.info("daily loss limit 설정 → %.2f%%", req.pct)
        return {"limit_pct": req.pct}

    @app.post("/ict/credentials")
    async def set_credentials(req: CredentialsRequest) -> dict[str, Any]:
        """API 키 등록 — .env 영구 저장 + 메모리 settings 즉시 갱신."""
        if req.mode not in ("demo", "live"):
            raise HTTPException(status_code=400, detail=f"invalid mode: {req.mode}")
        if not req.api_key.strip() or not req.api_secret.strip():
            raise HTTPException(status_code=400, detail="api_key / api_secret 필수")
        try:
            env_path = _write_env_credentials(
                req.mode, req.api_key.strip(), req.api_secret.strip(),
            )
        except OSError as e:
            raise HTTPException(
                status_code=500, detail=f".env 쓰기 실패: {e}",
            ) from e
        # 메모리 settings 즉시 갱신 (다음 start 시 바로 반영)
        if req.mode == "demo":
            manager.settings.demo_api_key = SecretStr(req.api_key.strip())
            manager.settings.demo_api_secret = SecretStr(req.api_secret.strip())
        else:
            manager.settings.live_api_key = SecretStr(req.api_key.strip())
            manager.settings.live_api_secret = SecretStr(req.api_secret.strip())
        logger.info("credentials 갱신 — mode=%s, env=%s", req.mode, env_path)
        return _settings_safe_dict(manager.settings)

    @app.post("/ict/position/close")
    async def close_position(req: PositionCloseRequest) -> dict[str, Any]:
        """Close By 수동 청산 — 반대 방향 시장가 주문 + active_position 갱신.

        Args:
            req.fraction: 0 < fraction ≤ 1. 1.0 = 전체 청산, 0.5 = 50% 부분 청산.

        Returns:
            ``{"closed_qty": ..., "remaining_qty": ..., "active": bool}``
        """
        if not (0.0 < req.fraction <= 1.0):
            raise HTTPException(
                status_code=400,
                detail=f"fraction은 (0, 1] 범위 — 받은 값: {req.fraction}",
            )
        bot = manager.bot
        if bot is None or bot.active_position is None:
            raise HTTPException(status_code=404, detail="active position 없음")

        ap = bot.active_position
        close_qty = ap.qty * req.fraction
        # 반대 방향 시장가 청산 (long → sell, short → buy)
        close_side = "sell" if ap.direction is Direction.LONG else "buy"
        try:
            await bot.client.place_order(
                symbol=bot.symbol,
                side=close_side,
                qty=close_qty,
                # market 청산 — price/SL/TP 없음
                price=None,
                stop_loss=None,
                take_profit=None,
            )
        except Exception as e:  # noqa: BLE001 — 사용자에게 그대로 전달
            logger.exception("close_position place_order 실패: %s", e)
            raise HTTPException(status_code=502, detail=f"청산 주문 실패: {e}") from e

        remaining = ap.qty - close_qty
        if req.fraction >= 1.0 or remaining <= 1e-9:
            bot.active_position = None
            logger.info("전체 청산 완료 — qty=%.6f", close_qty)
            return {"closed_qty": close_qty, "remaining_qty": 0.0, "active": False}

        ap.qty = remaining
        logger.info(
            "부분 청산 — closed=%.6f remaining=%.6f", close_qty, remaining,
        )
        return {"closed_qty": close_qty, "remaining_qty": remaining, "active": True}

    @app.post("/ict/pending/cancel")
    async def cancel_pending() -> dict[str, Any]:
        """대기 중인 지정가 entry 주문 즉시 취소 (단일 사용자 분기).

        2026-06-11 리뷰 수정: mu 분기에만 있어 .exe(legacy) 모드에서 UI CANCEL
        버튼이 404 나던 것 보강.
        """
        bot = manager.bot
        if bot is None:
            return {"ok": False, "reason": "bot_not_running"}
        ok = await bot.cancel_pending_entry()
        return {"ok": ok}

    @app.get("/ict/position")
    async def get_position() -> dict[str, Any]:
        """현재 active position 상세 — UI 하단 패널용 + 지정가 대기 시 pending 필드.

        Returns:
            - ``{"active": False, "pending": None}`` — 봇 미가동/플랫 + 대기 주문 없음
            - ``{"active": False, "pending": {...}}`` — 지정가 미체결 대기 중 (UI 차트 라인용)
            - ``{"active": True, ..., "pending": None}`` — 활성 포지션 (지정가 대기는 없음)
        """
        bot = manager.bot
        # pending entry (지정가 미체결 대기) — active 든 아니든 같이 표시.
        pending: dict[str, Any] | None = None
        if bot is not None and bot._pending_entry is not None:
            pe = bot._pending_entry
            pending = {
                "direction": pe.direction.value,
                "entry": pe.entry,
                "stop_loss": pe.stop_loss,
                "take_profit": pe.take_profit,
                "qty": pe.qty,
                "placed_ts_ms": pe.placed_ts_ms,
            }
        if bot is None or bot.active_position is None:
            return {"active": False, "pending": pending}

        ap = bot.active_position
        # #BUG-7 / 사용자 요청: 거래소 포지션 값 그대로 우선 (unrealized_pnl / entry).
        # 거래소 fetch 실패 시 봇 메모리 (ap) + 봇 계산 fallback.
        ex_unrealized: float | None = None
        ex_entry = ap.entry
        try:
            ex_pos = await bot.client.fetch_position(bot.symbol)
            if ex_pos:
                up = ex_pos.get("unrealized_pnl")
                if up is None:
                    up = ex_pos.get("unrealizedPnl")  # ccxt 표준 키
                if isinstance(up, (int, float)):
                    ex_unrealized = float(up)
                ep = (
                    ex_pos.get("entry_price")
                    or ex_pos.get("entryPrice")
                    or ex_pos.get("avgPrice")
                )
                if isinstance(ep, (int, float)) and float(ep) > 0:
                    ex_entry = float(ep)
        except Exception as e:  # noqa: BLE001
            logger.debug("거래소 포지션 PnL fetch 실패 — 봇 계산 fallback: %s", e)

        # mark price = 마지막 봉 close (fetch 실패 시 entry 로 fallback)
        mark_price = ex_entry
        try:
            rows = await bot.client.fetch_ohlcv(bot.symbol, bot.timeframe, 2)
            if rows:
                mark_price = float(rows[-1][4])
        except Exception as e:  # noqa: BLE001
            logger.debug("mark_price fetch 실패 — entry 사용: %s", e)

        # Unrealized PnL — 거래소 값 우선, 없으면 (mark-entry)*qty 봇 계산.
        if ex_unrealized is not None:
            unrealized = ex_unrealized
        elif ap.direction is Direction.LONG:
            unrealized = (mark_price - ex_entry) * ap.qty
        else:
            unrealized = (ex_entry - mark_price) * ap.qty

        # Notional / Margin / Liquidation (거래소 entry 기준 대략)
        lev = manager.settings.leverage
        notional = ex_entry * ap.qty
        margin = notional / lev
        # liquidation = entry × (1 ∓ 1/lev × 0.95) — 0.95 = maintenance margin buffer
        if ap.direction is Direction.LONG:
            liq_price = ex_entry * (1.0 - (1.0 / lev) * 0.95)
        else:
            liq_price = ex_entry * (1.0 + (1.0 / lev) * 0.95)

        # Unrealized PnL % (ROI on margin)
        roi_pct = (unrealized / margin * 100.0) if margin > 0 else 0.0

        return {
            "active": True,
            "symbol": bot.symbol,
            "direction": ap.direction.value,
            "entry": ex_entry,
            "stop_loss": ap.stop_loss,
            "take_profit": ap.take_profit,
            "qty": ap.qty,
            "setup_ts_ms": ap.setup_ts_ms,
            "mark_price": mark_price,
            "unrealized_pnl": unrealized,
            "roi_pct": roi_pct,
            "margin": margin,
            "notional": notional,
            "liquidation_price": liq_price,
            "leverage": lev,
            "pending": pending,
        }

    @app.get("/ict/judgment")
    async def get_judgment() -> dict[str, Any]:
        """봇 현재 판단 — 좌하단 패널 (2026-05-27 추가).

        Returns:
            - ``direction``: 롱/숏 확률 (HTF FVG bull/bear 가중치 비율)
            - ``reasons``: [{color, term, range, interpretation}, ...] 활성 지표 목록
            - ``entry_condition``: 진입 조건 + 현재 막힌 사유
        """
        bot = manager.bot
        if bot is None:
            return {
                "direction": {"long_pct": 50, "short_pct": 50, "label": "봇 미가동"},
                "reasons": [],
                "entry_condition": {"title": "봇 미가동", "detail": "START 눌러 가동"},
            }
        # HTF FVG 가중치 합산 → 방향 확률
        htf_map = bot._htf_fvg_map_cache or []
        bull_w = sum(int(e.weight) for e in htf_map if e.type.value == "bullish")
        bear_w = sum(int(e.weight) for e in htf_map if e.type.value == "bearish")
        tot = bull_w + bear_w
        if tot > 0:
            long_pct = round(bull_w / tot * 100, 1)
            short_pct = round(100 - long_pct, 1)
        else:
            long_pct = short_pct = 50.0
        if long_pct >= 60:
            dir_label = "롱(상승) 우세"
        elif short_pct >= 60:
            dir_label = "숏(하락) 우세"
        else:
            dir_label = "방향 불확실 (혼조)"
        # 현재가
        cur_price = 0.0
        try:
            rows = await bot.client.fetch_ohlcv(bot.symbol, bot.timeframe, 2)
            if rows:
                cur_price = float(rows[-1][4])
        except Exception:  # noqa: BLE001
            pass
        # Reasons — 가까운 unswept HTF FVG 위/아래 2개씩
        reasons: list[dict[str, Any]] = []
        if cur_price > 0 and htf_map:
            bulls = [e for e in htf_map if e.type.value == "bullish" and e.high < cur_price]
            bears = [e for e in htf_map if e.type.value == "bearish" and e.low > cur_price]
            bulls.sort(key=lambda e: cur_price - e.high)  # 가까운 순
            bears.sort(key=lambda e: e.low - cur_price)
            for e in bulls[:2]:
                reasons.append({
                    "color": "green",
                    "term": f"FVG {e.tf}",
                    "range": f"{e.high:,.0f} ~ {e.low:,.0f}",
                    "interpretation": "롱 지지 가능성 (가격이 끌릴 빈 공간)",
                })
            for e in bears[:2]:
                reasons.append({
                    "color": "red",
                    "term": f"FVG {e.tf}",
                    "range": f"{e.high:,.0f} ~ {e.low:,.0f}",
                    "interpretation": "위쪽 저항 (가격 도달 시 막힐 위험)",
                })
        # 2026-05-27: 파트너 요청 — "FVG 만 나옴" → BOS/CHoCH + OB + Sweep 도 표시.
        # 봇 본체는 다 보고 있음 (차트 마커에 표시). 판단 패널만 FVG 위주였던 거 보완.
        try:
            use_tf = bot.timeframe
            ohlcv_rows = await bot.client.fetch_ohlcv(bot.symbol, use_tf, 200)
            if ohlcv_rows:
                df = pd.DataFrame(
                    ohlcv_rows,
                    columns=["ts_ms", "open", "high", "low", "close", "volume"],
                )
                df.index = pd.DatetimeIndex(
                    pd.to_datetime(df["ts_ms"], unit="ms", utc=True),
                )
                df = df[["open", "high", "low", "close", "volume"]]
                _mk = to_chart_markers(
                    df,
                    min_rr=manager.settings.min_rr,
                    fvg_min_size_pct=manager.settings.fvg_min_size_pct,
                )
                # 1) 최근 구조 변화 (BOS/CHoCH) — large 우선, 1개
                struct_src = _mk.large_structure or _mk.structure
                if struct_src:
                    s = struct_src[-1]
                    struct_meta = {
                        "bos_bullish":   ("green",  "BOS↑",    "상승 추세 지속 (이전 고점 돌파)"),
                        "bos_bearish":   ("red",    "BOS↓",    "하락 추세 지속 (이전 저점 돌파)"),
                        "choch_bullish": ("green",  "CHoCH↑",  "추세 전환 — 상승 우세 (전환 신호)"),
                        "choch_bearish": ("red",    "CHoCH↓",  "추세 전환 — 하락 우세 (전환 신호)"),
                    }
                    color, term, interp = struct_meta.get(
                        s.type, ("yellow", s.type, "구조 변화"),
                    )
                    reasons.append({
                        "color": color, "term": term,
                        "range": f"{s.broken_level:,.0f}",
                        "interpretation": interp,
                    })
                # 2) 가까운 unmitigated OB — 위/아래 1개씩 (large 우선)
                ob_src = _mk.large_order_blocks or _mk.order_blocks
                obs = [o for o in ob_src if not o.mitigated]
                if cur_price > 0 and obs:
                    bull_obs = [o for o in obs if o.type == "bullish" and o.high < cur_price]
                    bear_obs = [o for o in obs if o.type == "bearish" and o.low > cur_price]
                    bull_obs.sort(key=lambda o: cur_price - o.high)
                    bear_obs.sort(key=lambda o: o.low - cur_price)
                    for o in bull_obs[:1]:
                        reasons.append({
                            "color": "green", "term": "OB",
                            "range": f"{o.high:,.0f} ~ {o.low:,.0f}",
                            "interpretation": "롱 지지 가능성 (기관 매수 흔적)",
                        })
                    for o in bear_obs[:1]:
                        reasons.append({
                            "color": "red", "term": "OB",
                            "range": f"{o.high:,.0f} ~ {o.low:,.0f}",
                            "interpretation": "위쪽 저항 (기관 매도 흔적)",
                        })
                # 3) 최근 liquidity sweep — 1개 (large 우선)
                sweep_src = _mk.large_sweeps or _mk.sweeps
                if sweep_src:
                    sw = sweep_src[-1]
                    if sw.type == "bullish":
                        reasons.append({
                            "color": "yellow", "term": "SSL Sweep",
                            "range": f"{sw.wick_price:,.0f}",
                            "interpretation": "하단 유동성 흡수 — 반등 가능성 (롱 손절 청산)",
                        })
                    elif sw.type == "bearish":
                        reasons.append({
                            "color": "yellow", "term": "BSL Sweep",
                            "range": f"{sw.wick_price:,.0f}",
                            "interpretation": "상단 유동성 흡수 — 하락 가능성 (숏 손절 청산)",
                        })
        except Exception as e:  # noqa: BLE001
            logger.debug("judgment markers 추출 실패: %s", e)
        # 세션 상태 한 줄
        session = _compute_session_status()
        if session["kind"] == "killzone":
            reasons.append({
                "color": "green", "term": "Killzone", "range": session["label"].replace("Kill zone : ", ""),
                "interpretation": "활발한 매매 시간대 (진입 윈도우)",
            })
        elif session["kind"] == "us_open":
            reasons.append({
                "color": "green", "term": "US Session", "range": "09:30-16:00 ET",
                "interpretation": "미장 개장 — 변동성 ↑",
            })
        else:
            reasons.append({
                "color": "yellow", "term": "Session", "range": "Off-hours",
                "interpretation": "킬존 밖 — 진입 자제 권장 (referral 은 24h 허용)",
            })
        # Entry condition
        min_cf = manager.settings.min_confluence
        min_rr = manager.settings.min_rr
        if bot.active_position is not None:
            ec_title = "진입 중 — 신규 진입 차단 (페어당 1포지션)"
            ec_detail = f"방향 {bot.active_position.direction.value}, SL/TP 진행"
        elif bot._pending_entry is not None:
            pe = bot._pending_entry
            ec_title = f"지정가 미체결 대기 ({pe.entry:,.2f})"
            ec_detail = "10분 안 체결 안 되면 자동 취소"
        else:
            ec_title = f"등급 {min_cf} 이상 & RR {min_rr} 이상 setup 대기"
            ec_detail = "현재 활성 setup 없음 (혹은 게이트 미달로 skip)"
        # 2026-05-29: 진단 인지 보강. 파트너 의문 "long 비율 우세인데 short 만 진입"
        # 직접 해소 — recent setup direction 분포 + 봇 내부 진단 카운터 노출.
        recent_dirs = list(getattr(bot, "_recent_setup_directions", []) or [])
        recent_long = sum(1 for d in recent_dirs if d == "long")
        recent_short = sum(1 for d in recent_dirs if d == "short")
        return {
            "direction": {
                "long_pct": long_pct, "short_pct": short_pct,
                "label": dir_label, "bull_weight": bull_w, "bear_weight": bear_w,
                # 사용자 직관 보정 — 위 비율은 HTF FVG 가중치 기준.
                # 실제 진입은 LTF + HTF override + EMA + DOL + 등급 조합 결과로
                # 위 비율과 다를 수 있다.
                "hint": (
                    "위 비율은 HTF FVG 가중치 기준 (장기 구조). "
                    "실제 진입 방향은 LTF + HTF override + EMA + DOL + 등급 필터 "
                    "조합 결과로 위 비율과 다를 수 있음."
                ),
                "recent_setups": {
                    "long": recent_long, "short": recent_short,
                    "last": recent_dirs[-1] if recent_dirs else None,
                },
            },
            "reasons": reasons,
            "entry_condition": {"title": ec_title, "detail": ec_detail},
            # 2026-05-29 #SILENT-*: 조용한 오류 가시화 — 운영 중 실패 위치 즉시 확인.
            "diagnostics": {
                "recovery_failed": bool(getattr(bot, "_recovery_failed", False)),
                "sync_failure_streak": int(
                    getattr(bot, "_sync_failure_streak", 0),
                ),
                "order_failure_count": int(
                    getattr(bot, "_order_failure_count", 0),
                ),
                "tpsl_failure_streak": int(
                    getattr(bot, "_tpsl_failure_streak", 0),
                ),
            },
        }

    @app.get("/ict/equity")
    async def get_equity() -> dict[str, Any]:
        """현재 잔고(USDT) + 현재 세션 상태(킬존/미장/None) — 좌측·좌상단 표시용.

        2026-05-27 추가. session_status:
          - {"kind":"killzone","label":"Kill zone : Asian"} — 킬존 안
          - {"kind":"us_open","label":"U.S. stock market Open"} — NYSE 시간(킬존 밖)
          - {"kind":"none","label":"None"} — 둘 다 아님
        """
        bot = manager.bot
        session = _compute_session_status()
        if bot is None:
            return {"equity": 0.0, "active": False, "session_status": session}
        try:
            eq = await bot._fetch_equity()
        except Exception as e:  # noqa: BLE001
            logger.debug("equity fetch 실패: %s", e)
            return {
                "equity": 0.0, "active": True, "error": "fetch_failed",
                "session_status": session,
            }
        return {"equity": float(eq), "active": True, "session_status": session}

    @app.get("/ict/closed_pnl")
    async def get_closed_pnl(limit: int = 50) -> dict[str, Any]:
        """최근 청산 거래 내역 — UI 우측 P&L 패널용 (Bybit P&L 화면 모사).

        거래소 closed-pnl history 를 직접 조회 (수수료/펀딩 반영된 실현치).

        Args:
            limit: 반환할 최대 거래 수 (기본 50, 30일 윈도우 정합).

        Returns:
            ``{"trades": [{symbol, direction, entry_price, exit_price, qty,
              pnl_usd, roi_pct, leverage, closed_at_ts, opened_at_ts}, ...]}``
            신→구 정렬. 봇 미가동 또는 fetch 실패 시 ``{"trades": []}``.
        """
        bot = manager.bot
        if bot is None:
            return {"trades": []}
        # 2026-05-27: 파트너 요청 — 7일 → 30일 윈도우.
        # bybit V5 closed-pnl 은 chunk 당 7일 한도지만 fetch_closed_positions 가
        # 7일 chunk 페이지네이션 (max_chunks=30) 으로 자동 분할 처리.
        import time as _time
        since_ms = int(_time.time() * 1000) - 30 * 24 * 60 * 60 * 1000
        try:
            closed = await bot.client.fetch_closed_positions(
                since_ms=since_ms, limit=max(1, min(int(limit), 200)),
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("closed_pnl fetch 실패: %s", e)
            return {"trades": []}
        trades: list[dict[str, Any]] = []
        for cp in closed[: max(1, min(int(limit), 200))]:
            trades.append({
                "symbol": getattr(cp, "symbol", None),
                "direction": getattr(cp, "direction", None),
                "entry_price": getattr(cp, "entry_price", None),
                "exit_price": getattr(cp, "exit_price", None),
                "qty": getattr(cp, "qty", None),
                "pnl_usd": getattr(cp, "pnl_usd", None),
                "roi_pct": getattr(cp, "roi_pct", None),
                "leverage": getattr(cp, "leverage", None),
                "closed_at_ts": getattr(cp, "closed_at_ts", None),
                "opened_at_ts": getattr(cp, "opened_at_ts", None),
            })
        return {"trades": trades}

    @app.get("/ict/markers")
    async def get_markers(
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int = 1500,
    ) -> dict[str, Any]:
        """OHLCV fetch 후 chart marker 일체를 계산해 반환.

        2026-05-27 파트너 요청 — 봉 수 더 많이 보이게. default 200 → 1500
        (Bybit V5 한도 1000 초과는 ccxt_client.fetch_ohlcv 가 자동 페이지네이션).
        """
        bot = manager.bot
        if bot is None:
            raise HTTPException(
                status_code=404,
                detail="봇이 실행 중이 아닙니다 — /ict/start 먼저 호출하세요",
            )
        use_symbol = symbol or bot.symbol
        use_tf = timeframe or bot.timeframe
        try:
            rows = await bot.client.fetch_ohlcv(use_symbol, use_tf, limit)
        except AuthenticationError as e:
            # 키 무효(10003) — 502 대신 빈 마커. 차트는 마커 없이 표시.
            logger.warning("markers 인증 실패(키 무효): %s", e)
            rows = []
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_ohlcv 실패: %s", e)
            raise HTTPException(status_code=502, detail=f"fetch_ohlcv: {e}") from e

        df = pd.DataFrame(
            rows,
            columns=["ts_ms", "open", "high", "low", "close", "volume"],
        )
        df.index = pd.DatetimeIndex(pd.to_datetime(df["ts_ms"], unit="ms", utc=True))
        df = df[["open", "high", "low", "close", "volume"]]

        markers = to_chart_markers(
            df,
            min_rr=manager.settings.min_rr,
            fvg_min_size_pct=manager.settings.fvg_min_size_pct,
        )
        return {
            "symbol": use_symbol,
            "timeframe": use_tf,
            "count": {
                "fvgs": len(markers.fvgs),
                "sweeps": len(markers.sweeps),
                "structure": len(markers.structure),
                "swings": len(markers.swings),
                "killzones": len(markers.killzones),
                "setups": len(markers.setups),
                "order_blocks": len(markers.order_blocks),
                "macros": len(markers.macros),
                "trailing": 1 if markers.trailing else 0,
                "internal_swings": len(markers.internal_swings),
                "internal_structure": len(markers.internal_structure),
                "large_swings": len(markers.large_swings),
                "large_structure": len(markers.large_structure),
                "equal_levels": len(markers.equal_levels),
            },
            "markers": markers.to_dict(),
        }

    @app.get("/ict/ohlcv")
    async def get_ohlcv(
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int = 1500,
    ) -> dict[str, Any]:
        """OHLCV 봉 그대로 반환 (UI lightweight-charts candle 입력용).

        2026-05-27: 봇 prefetch cache 사용 — TF 토글 즉시 응답.
        cache hit 시 즉시 반환 + background incremental refresh.
        symbol override 시는 cache 우회 (멀티 심볼 미지원 — 기본 봇 심볼만).
        """
        bot = manager.bot
        if bot is None:
            raise HTTPException(
                status_code=404, detail="봇이 실행 중이 아닙니다",
            )
        use_symbol = symbol or bot.symbol
        use_tf = timeframe or bot.timeframe
        try:
            if use_symbol == bot.symbol:
                rows = await bot.get_ohlcv_cached(use_tf, limit)
            else:
                rows = await bot.client.fetch_ohlcv(use_symbol, use_tf, limit)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"fetch_ohlcv: {e}") from e
        # 수만 봉 변환은 이벤트 루프 밖에서 (#OHLCV-NONBLOCK) — health timeout 방지.
        candles = await asyncio.to_thread(_rows_to_candles, rows)
        return {"symbol": use_symbol, "timeframe": use_tf, "candles": candles}

    # Static UI mount — frozen (PyInstaller) 환경 대응:
    # - dev: <repo_root>/ui_ict/
    # - frozen: sys._MEIPASS/ui_ict/ (PyInstaller spec datas 로 묶임)
    import sys
    candidates = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        candidates.append(Path(meipass) / "ui_ict")
    candidates.append(Path(__file__).resolve().parents[3] / "ui_ict")
    for ui_dir in candidates:
        if ui_dir.is_dir():
            app.mount("/ui", StaticFiles(directory=str(ui_dir), html=True), name="ui")
            logger.info("UI mounted from %s", ui_dir)
            break

    return app


__all__ = ["create_app"]
