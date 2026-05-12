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
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from aurora_ict.api.markers import to_chart_markers
from aurora_ict.bot.manager import BotManager
from aurora_ict.config.settings import IctSettings, RunMode

logger = logging.getLogger(__name__)


class RunModeRequest(BaseModel):
    mode: str  # "demo" | "live"


class EnabledRequest(BaseModel):
    enabled: bool


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

    # ============================================================
    # Static UI mount — ui_ict/ 박힌 거 박힙 박힘 박힙 박힙 박힘
    # ============================================================
    # 박힌 거 박힙 박힘 박힙 박힘 박힙 박힘 박힙 박힘 박힙 박힘 박힙 박힙 박힙 박힘 박힙 박힘.
    # Aurora-ICT 박힌 거 박힘 박힘 박힘 ui_ict/ 박힙 박힘 박힘 박힙 박힘 박힙 박힘 박힙 박힙.
    _ui_dir = Path(__file__).resolve().parents[3] / "ui_ict"
    if _ui_dir.is_dir():
        app.mount("/ui", StaticFiles(directory=str(_ui_dir), html=True), name="ui")

    return app


__all__ = ["create_app"]
