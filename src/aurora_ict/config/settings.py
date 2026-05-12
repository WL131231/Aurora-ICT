"""Aurora-ICT settings — pydantic v2 Settings + 환경변수 박힘.

박은 거 박힘:
- ``run_mode`` = ``demo`` (기본) / ``live`` — 박힌 모드 박힘 박힘 API 키 박힘 박힘 박힘
- ``enabled`` = bot 박힘 ON/OFF (start 박은 박힘 박힘 박힘 박힘 박힘)
- Bybit demo / live API 키 박힘 박힘 박힘 박힘
- 매매 박힌 파라미터 — risk_per_trade_pct / leverage / symbol / min_rr 등

env 박힌 거 박힘 박힘 prefix = ``AURORA_ICT_``. 예:
- ``AURORA_ICT_RUN_MODE=demo``
- ``AURORA_ICT_DEMO_API_KEY=...``
- ``AURORA_ICT_LIVE_API_KEY=...``

.env 박힌 거 박힘 박힘 박힘 박힘 박힘 박힙 박힘 박힘 박힘 박힘 박힘.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class RunMode(str, Enum):
    """봇 박힘 박힌 모드."""

    DEMO = "demo"
    LIVE = "live"


class IctSettings(BaseSettings):
    """Aurora-ICT 박힌 설정 — 환경변수 박힘 자동 박힘.

    Attributes:
        run_mode: 박은 모드 (demo / live). 기본 demo.
        enabled: bot 박힘 박은 start 박힘 박은 박힘 (False 박힙 박힘 start 박힘 X).
        symbol: 박힌 symbol (e.g. "BTC/USDT:USDT" ccxt 박힘 박힘 박힘 박힘 박힘).
        timeframe: OHLCV timeframe.
        risk_per_trade_pct: 박힌 trade 박힌 risk %.
        leverage: 박힌 leverage.
        min_rr: 최소 RR.
        fvg_min_size_pct: FVG 박힌 최소 % size.
        step_interval_sec: bot step 박힘 interval.
        ohlcv_limit: fetch 봉 수.

        demo_api_key / demo_api_secret: Bybit Demo Trading 박힌 키.
        live_api_key / live_api_secret: Bybit 실매매 박힌 키 (박힙 박힘 박힘 박힘 박힘 빈).
    """

    model_config = SettingsConfigDict(
        env_prefix="AURORA_ICT_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    run_mode: RunMode = Field(default=RunMode.DEMO)
    enabled: bool = Field(default=False)

    symbol: str = Field(default="BTC/USDT:USDT")
    timeframe: str = Field(default="1m")
    risk_per_trade_pct: float = Field(default=0.5, ge=0.01, le=5.0)
    leverage: int = Field(default=5, ge=1, le=20)
    min_rr: float = Field(default=2.0, ge=1.0)
    fvg_min_size_pct: float = Field(default=0.0005, ge=0)
    step_interval_sec: int = Field(default=60, ge=10)
    ohlcv_limit: int = Field(default=200, ge=50, le=1000)

    demo_api_key: SecretStr = Field(default=SecretStr(""))
    demo_api_secret: SecretStr = Field(default=SecretStr(""))
    live_api_key: SecretStr = Field(default=SecretStr(""))
    live_api_secret: SecretStr = Field(default=SecretStr(""))

    @property
    def active_api_key(self) -> str:
        """``run_mode`` 박힌 박힌 API key 박힙 박힘 박힘."""
        if self.run_mode is RunMode.LIVE:
            return self.live_api_key.get_secret_value()
        return self.demo_api_key.get_secret_value()

    @property
    def active_api_secret(self) -> str:
        """``run_mode`` 박힌 박힌 API secret."""
        if self.run_mode is RunMode.LIVE:
            return self.live_api_secret.get_secret_value()
        return self.demo_api_secret.get_secret_value()

    @property
    def is_live(self) -> bool:
        return self.run_mode is RunMode.LIVE

    @property
    def is_demo(self) -> bool:
        return self.run_mode is RunMode.DEMO

    def has_credentials(self) -> bool:
        """박힌 모드 박힌 박힌 API 키 박힘 박힘 박힘 박힘."""
        return bool(self.active_api_key and self.active_api_secret)


_singleton: IctSettings | None = None


def get_settings() -> IctSettings:
    """싱글톤 settings 박힘.

    첫 호출 박힘 .env 박힘 박힘 박힘 박힘 박힘 — 박힙 박힘 박힘 cache 박힘.
    """
    global _singleton
    if _singleton is None:
        _singleton = IctSettings()
    return _singleton


def reload_settings(env_file: str | Path | None = None) -> IctSettings:
    """settings 박힘 박힘 박힘 박힘 박힙 박힘 박힘 — 테스트 박힘 / 박힌 박힘 박힘 박힘 박힘 박힘.

    Args:
        env_file: 명시적 박힌 .env 박힘 박힘 박힙 박힘 박힘. ``None`` 박힘 박힘 박힘 박힘.
    """
    global _singleton
    if env_file is not None:
        _singleton = IctSettings(_env_file=str(env_file))  # type: ignore[call-arg]
    else:
        _singleton = IctSettings()
    return _singleton


__all__ = [
    "IctSettings",
    "RunMode",
    "get_settings",
    "reload_settings",
]
