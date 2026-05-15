"""Aurora-ICT settings — pydantic v2 Settings + 환경변수 기반.

담는 항목:
- ``run_mode`` = ``demo`` (기본) / ``live`` — 모드에 따라 사용하는 API 키 분기
- ``enabled`` = bot ON/OFF (start 시 이 값이 False면 가동 불가)
- Bybit demo / live API 키 (각 모드별로 별도 보관)
- 매매 파라미터 — risk_per_trade_pct / leverage / symbol / min_rr 등

환경변수 prefix = ``AURORA_ICT_``. 예:
- ``AURORA_ICT_RUN_MODE=demo``
- ``AURORA_ICT_DEMO_API_KEY=...``
- ``AURORA_ICT_LIVE_API_KEY=...``

.env 파일도 같은 prefix로 읽어 들인다.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 매매 timeframe 허용 목록 — 5m 이상.
# LTF (5m/15m/30m) 는 ICT 정통 entry TF (HTF bias + LTF refined entry).
# 1m 은 노이즈 과대로 비허용.
TRADE_TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w")


class RunMode(StrEnum):
    """봇 실행 모드."""

    DEMO = "demo"
    LIVE = "live"


class IctSettings(BaseSettings):
    """Aurora-ICT 설정 — 환경변수에서 자동 로드.

    Attributes:
        run_mode: 운용 모드 (demo / live). 기본 demo.
        enabled: bot 가동 허용 플래그 (False면 start 불가).
        symbol: 거래 symbol (e.g. "BTC/USDT:USDT", ccxt unified symbol 형식).
        timeframe: OHLCV timeframe.
        leverage: 레버리지.
        position_pct_base / _max / _step: confluence-based notional sizing.
        min_rr: 최소 RR.
        fvg_min_size_pct: FVG 최소 % size.
        step_interval_sec: bot step 호출 간격.
        ohlcv_limit: fetch 봉 수.

        demo_api_key / demo_api_secret: Bybit Demo Trading 키.
        live_api_key / live_api_secret: Bybit 실매매 키 (미사용 시 빈 값).
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
    # 매매 timeframe — 5m 이상 허용. 1m 은 노이즈 과대로 제외.
    timeframe: str = Field(default="1h")

    @field_validator("timeframe")
    @classmethod
    def _validate_trade_timeframe(cls, v: str) -> str:
        if v not in TRADE_TIMEFRAMES:
            raise ValueError(
                f"timeframe '{v}' 미지원 — 허용 목록: {list(TRADE_TIMEFRAMES)}",
            )
        return v

    # Notional-based sizing — confluence_score 단계별 시드 % 박는 정책.
    # 기본 40% / score+1 마다 ↑ / 최대 90%. equity * pct / 100 = margin, leveraged.
    # ICT 정통 risk-based (qty=risk/SL_dist) 박지 않고 notional 박는 방향.
    # 단점: SL 거리 가변 → 실제 손실 폭도 가변. min_rr 2.0 / FVG / bias 필터로 보완.
    leverage: int = Field(default=20, ge=1, le=50)
    position_pct_base: float = Field(default=40.0, ge=1.0, le=100.0)
    position_pct_max: float = Field(default=90.0, ge=1.0, le=100.0)
    # confluence_score 0 → base, 1/2/3+ → base + step * score (max capped).
    # 사용자 정책: 0→40, 1→55, 2→70, 3+→90 (step=15).
    position_pct_step: float = Field(default=15.0, ge=0.0, le=50.0)
    # min_rr 1.5 — 진입 빈도 위해 기존 2.0 에서 완화. partial 1R/2R/3R 청산 구조라
    # final RR 1.5 도 평균 ~1.5R 의 실제 수익 챙길 수 있음.
    min_rr: float = Field(default=1.5, ge=1.0)
    fvg_min_size_pct: float = Field(default=0.0005, ge=0)
    step_interval_sec: int = Field(default=60, ge=10)
    ohlcv_limit: int = Field(default=1000, ge=50, le=1000)
    # setup stale threshold — FVG 이후 N 봉 안에 retest 없으면 진입 안 함.
    # 30봉 = 5m TF 에서 2.5시간 / 1h TF 에서 30시간. limit retest 시간 넉넉히 확보.
    setup_stale_bars: int = Field(default=30, ge=1, le=100)
    # disable_time_filter: True 면 Silver Bullet / Killzone 시간 윈도우 무시 (24h 매매).
    # ICT 정통은 TIME-first 라 살짝 벗어나지만 FVG/bias/sweep/min_rr 다른 필터로 깐깐.
    disable_time_filter: bool = Field(default=True)
    # multi_tf: True 면 HTF (Trade TF 위 전체) setup 추적 + LTF retrace/structure
    # shift/FVG confirm 시 진입 (ICT 정통). False 면 단일 TF 매매.
    multi_tf: bool = Field(default=False)
    multi_tf_ltf_lookback: int = Field(default=30, ge=5, le=200)
    # enable_trail: 진입 후 새 swing 형성 시 SL 이동 (ICT 정통 structure-based trail).
    enable_trail: bool = Field(default=False)
    trail_buffer_ratio: float = Field(default=0.001, ge=0.0, le=0.05)
    # use_market_entry: True 면 setup 검출 시 limit (FVG mean retest) 대신 즉시 시장가 진입.
    # 진입률 100% 보장 (limit 미체결 문제 해소), ICT 정통 retrace 진입 철학에서 살짝 벗어남.
    use_market_entry: bool = Field(default=False)
    # enable_partial_tp: False 면 partial TP1/TP2/TP3 (50/25/25%) 안 박음. 시장가 진입 시
    # setup entry vs 실제 fill price 차이로 partial TP 가 즉시 손실 fill 되는 자기-트랩
    # 회피용. trail SL 만으로 청산.
    enable_partial_tp: bool = Field(default=True)

    demo_api_key: SecretStr = Field(default=SecretStr(""))
    demo_api_secret: SecretStr = Field(default=SecretStr(""))
    live_api_key: SecretStr = Field(default=SecretStr(""))
    live_api_secret: SecretStr = Field(default=SecretStr(""))

    @property
    def active_api_key(self) -> str:
        """``run_mode``에 해당하는 API key 반환."""
        if self.run_mode is RunMode.LIVE:
            return self.live_api_key.get_secret_value()
        return self.demo_api_key.get_secret_value()

    @property
    def active_api_secret(self) -> str:
        """``run_mode``에 해당하는 API secret 반환."""
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
        """현재 모드의 API 키 보유 여부."""
        return bool(self.active_api_key and self.active_api_secret)


_singleton: IctSettings | None = None


def get_settings() -> IctSettings:
    """싱글톤 settings 반환.

    첫 호출 시 .env를 읽어 IctSettings를 만든 뒤 이후 호출은 cache 사용.
    """
    global _singleton
    if _singleton is None:
        _singleton = IctSettings()
    return _singleton


def reload_settings(env_file: str | Path | None = None) -> IctSettings:
    """싱글톤 settings를 강제로 다시 로드 — 테스트/런타임 키 갱신 등에 사용.

    Args:
        env_file: 명시적 .env 경로. ``None``이면 기본 위치에서 다시 읽음.
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
