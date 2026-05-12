"""FastAPI app — Aurora-ICT REST 박힘.

박힌 endpoint:
- ``GET  /ict/status`` — 봇 상태 박힘
- ``POST /ict/start`` — 봇 박힘 박힘
- ``POST /ict/stop`` — 봇 박힘 박힘
- ``POST /ict/run-mode`` — demo/live 박힘 박힘 (body: ``{"mode": "demo"|"live"}``)
- ``POST /ict/enabled`` — enabled toggle (body: ``{"enabled": true|false}``)
- ``GET  /ict/config`` — 박힌 settings 박힙 박힘 (api key 박힘 박힘 박힘 박힘)
- ``GET  /ict/markers`` — 박힌 OHLCV 박힘 박힘 박힘 chart markers (FVG/Sweep/MSS/Setup)
- ``GET  /ict/health`` — health probe

박힌 거 박힙 박힘 BotManager 박힌 거 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘
박힙 박힘 박힘 박힘 박힘 박힘 박힘 (production 박힙 박힘 박힘 ``lifespan`` 박힘 박힘 박힘).
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
from aurora_ict.config.settings import IctSettings, RunMode
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
    """settings → dict (api key 박힘 박힘 박힘 박힘 박힘)."""
    return {
        "run_mode": settings.run_mode.value,
        "enabled": settings.enabled,
        "symbol": settings.symbol,
        "timeframe": settings.timeframe,
        "risk_per_trade_pct": settings.risk_per_trade_pct,
        "leverage": settings.leverage,
        "min_rr": settings.min_rr,
        "fvg_min_size_pct": settings.fvg_min_size_pct,
        "step_interval_sec": settings.step_interval_sec,
        "ohlcv_limit": settings.ohlcv_limit,
        "has_demo_credentials": bool(
            settings.demo_api_key.get_secret_value()
            and settings.demo_api_secret.get_secret_value(),
        ),
        "has_live_credentials": bool(
            settings.live_api_key.get_secret_value()
            and settings.live_api_secret.get_secret_value(),
        ),
    }


def _status_dict(manager: BotManager) -> dict[str, Any]:
    """BotManager status → dict."""
    st = manager.status()
    return {
        "state": st.state.value,
        "run_mode": st.run_mode.value,
        "enabled": st.enabled,
        "symbol": st.symbol,
        "has_credentials": st.has_credentials,
        "has_active_position": st.has_active_position,
        "last_setup_ts_ms": st.last_setup_ts_ms,
    }


def create_app(manager: BotManager) -> FastAPI:
    """FastAPI app 박힘 박힘 박힘.

    Args:
        manager: 박힌 BotManager 박힌 거 박힘 박힘 박힘 박힘.
    """
    app = FastAPI(
        title="Aurora-ICT API",
        version="0.2.1",
        description="ICT (Inner Circle Trader) 매매 박힌 봇 REST API",
    )

    # CORS — dev 박힘 박힘 박힘 박힘 박힘 (production 박힘 박힘 박힘 박힘 박힘 박힘 박힘)
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

    @app.get("/ict/position")
    async def get_position() -> dict[str, Any]:
        """현재 active position 상세 — UI 하단 패널용.

        Returns:
            - ``{"active": False}`` 봇 미가동 또는 포지션 없음
            - ``{"active": True, ...}`` 박힌 active position + mark price + PnL + margin
        """
        bot = manager.bot
        if bot is None or bot.active_position is None:
            return {"active": False}

        ap = bot.active_position
        # mark price = 마지막 봉 close (fetch 실패 시 entry 로 fallback)
        mark_price = ap.entry
        try:
            rows = await bot.client.fetch_ohlcv(bot.symbol, bot.timeframe, 2)
            if rows:
                mark_price = float(rows[-1][4])
        except Exception as e:  # noqa: BLE001
            logger.debug("mark_price fetch 실패 — entry 사용: %s", e)

        # Unrealized PnL = (mark - entry) * qty (long) / (entry - mark) * qty (short)
        if ap.direction is Direction.LONG:
            unrealized = (mark_price - ap.entry) * ap.qty
        else:
            unrealized = (ap.entry - mark_price) * ap.qty

        # Notional / Margin / Liquidation (대략)
        lev = manager.settings.leverage
        notional = ap.entry * ap.qty
        margin = notional / lev
        # liquidation = entry × (1 ∓ 1/lev × 0.95) — 0.95 = maintenance margin buffer
        if ap.direction is Direction.LONG:
            liq_price = ap.entry * (1.0 - (1.0 / lev) * 0.95)
        else:
            liq_price = ap.entry * (1.0 + (1.0 / lev) * 0.95)

        # Unrealized PnL % (ROI on margin)
        roi_pct = (unrealized / margin * 100.0) if margin > 0 else 0.0

        return {
            "active": True,
            "symbol": bot.symbol,
            "direction": ap.direction.value,
            "entry": ap.entry,
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
        }

    @app.get("/ict/markers")
    async def get_markers(
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """OHLCV fetch 박힌 거 박힘 박힘 markers 박힘 박힘."""
        bot = manager.bot
        if bot is None:
            raise HTTPException(
                status_code=404,
                detail="봇 박힙 박힘 박힘 박힘 — /ict/start 박힘 박힙 박힘 박힙 박힘",
            )
        use_symbol = symbol or bot.symbol
        use_tf = timeframe or bot.timeframe
        try:
            rows = await bot.client.fetch_ohlcv(use_symbol, use_tf, limit)
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_ohlcv 박힘: %s", e)
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
            },
            "markers": markers.to_dict(),
        }

    @app.get("/ict/ohlcv")
    async def get_ohlcv(
        symbol: str | None = None,
        timeframe: str | None = None,
        limit: int = 200,
    ) -> dict[str, Any]:
        """OHLCV 봉 박힘 박힘 박힘 (UI 박힘 박힘 lightweight-charts candles)."""
        bot = manager.bot
        if bot is None:
            raise HTTPException(
                status_code=404, detail="봇 박힙 박힘 박힘 박힘",
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
