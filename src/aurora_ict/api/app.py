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

import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, SecretStr

from aurora_ict.api.markers import to_chart_markers
from aurora_ict.bot.manager import BotManager
from aurora_ict.config.settings import TRADE_TIMEFRAMES, IctSettings, RunMode
from aurora_ict.strategy.silver_bullet import Direction

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


class TimeframeRequest(BaseModel):
    """매매 timeframe 변경 — 1h / 2h / 4h / 1d / 1w 중 하나."""

    timeframe: str


class DailyLossLimitRequest(BaseModel):
    """#SAFETY-1 일일 손실 한도 변경 — 자본 대비 % (0 = 비활성, 0~50)."""

    pct: float


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


def create_app(manager: BotManager) -> FastAPI:
    """FastAPI app 인스턴스 생성.

    Args:
        manager: 라우터에서 사용할 BotManager 싱글톤.
    """
    app = FastAPI(
        title="Aurora-ICT API",
        version="0.2.1",
        description="ICT (Inner Circle Trader) 매매 봇 REST API",
    )

    # CORS — dev 편의를 위해 wildcard (production에서는 도메인 제한 필요)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
        try:
            await manager.start()
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        return _status_dict(manager)

    @app.post("/ict/stop")
    async def stop_bot() -> dict[str, Any]:
        await manager.stop()
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
            }
        return status

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
    async def get_closed_pnl(limit: int = 20) -> dict[str, Any]:
        """최근 청산 거래 내역 — UI 우측 P&L 패널용 (Bybit P&L 화면 모사).

        거래소 closed-pnl history 를 직접 조회 (수수료/펀딩 반영된 실현치).

        Args:
            limit: 반환할 최대 거래 수 (기본 20).

        Returns:
            ``{"trades": [{symbol, direction, entry_price, exit_price, qty,
              pnl_usd, roi_pct, leverage, closed_at_ts, opened_at_ts}, ...]}``
            신→구 정렬. 봇 미가동 또는 fetch 실패 시 ``{"trades": []}``.
        """
        bot = manager.bot
        if bot is None:
            return {"trades": []}
        # 최근 7일 윈도우 (bybit 단일 chunk 범위 한도)
        import time as _time
        since_ms = int(_time.time() * 1000) - 7 * 24 * 60 * 60 * 1000
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
        limit: int = 200,
    ) -> dict[str, Any]:
        """OHLCV fetch 후 chart marker 일체를 계산해 반환."""
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
        limit: int = 200,
    ) -> dict[str, Any]:
        """OHLCV 봉 그대로 반환 (UI lightweight-charts candle 입력용)."""
        bot = manager.bot
        if bot is None:
            raise HTTPException(
                status_code=404, detail="봇이 실행 중이 아닙니다",
            )
        use_symbol = symbol or bot.symbol
        use_tf = timeframe or bot.timeframe
        try:
            rows = await bot.client.fetch_ohlcv(use_symbol, use_tf, limit)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=502, detail=f"fetch_ohlcv: {e}") from e
        candles = [
            {
                "time": int(r[0] // 1000),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
            }
            for r in rows
        ]
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
