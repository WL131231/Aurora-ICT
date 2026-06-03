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

from pydantic import Field, SecretStr, field_validator, model_validator
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
    # 2026-05-28 파트너 결정 — default 1h → 5m. ICT 정통 정합 (Silver Bullet
    # / Macro 등 사소한 setup 까지 잡기 위해 짧은 TF 가 자연스러움).
    timeframe: str = Field(default="5m")

    @field_validator("symbol")
    @classmethod
    def _validate_symbol(cls, v: str) -> str:
        """2026-05-29 PR B: 멀티 페어 — BTC + ETH 허용. 그 외는 차단.

        Bybit V5 linear perpetual ccxt unified format. ETHUSDT 추가는 파트너
        결정 (BTC/ETH 분산 진입). 다른 알트는 추후 화이트리스트로 점진 확장.
        """
        allowed = {"BTC/USDT:USDT", "ETH/USDT:USDT"}
        if v not in allowed:
            raise ValueError(
                f"symbol '{v}' 미지원 — 허용 목록: {sorted(allowed)}",
            )
        return v

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
    # min_rr — v0.4.60 정통화 시 2.0 → v0.4.61 빈도 우려로 1.5 rollback → 2026-05-27
    # 데모 (v0.4.82 22h, 1W 5L, -843 USDT, 평균이익<평균손실 비대칭) 후 **2.0 복원**.
    # 작은 이익에 큰 손실 패턴을 1:2 이상만 통과시켜 완화. 빈도↓ 기대값↑.
    min_rr: float = Field(default=2.0, ge=1.0)
    # min_confluence: B+ 등급 게이트 (#1/#8, 2026-05-25 장수 결정). HTF boost 까지 반영된
    # 최종 confluence_score (OB+1/sweep+1/macro+1~2/HTF FVG +1~3, 합 0~7) 가 이 값 미만이면
    # 진입 skip → 빈도↓·품질↑. 등급 C0~1/B2~3/B+4~5/A6+.
    # 4(B+) → 35h 가동 중 체결 3건(~2/일)으로 너무 빡빡 → 2026-05-26 **3(B 이상)** 으로 완화.
    # 데모 보고 빈도 4~5/일 맞도록 3~5 사이 튜닝 (env: AURORA_ICT_MIN_CONFLUENCE).
    # 2026-06-03: 횡보 시장에서 score 3 도 못 채워 매매 0건 (실측 setup found
    # 1749건 진입 31건). 3 → 2 추가 완화 (파트너 결정, 테스트). score 2 = "C/B"
    # 등급 — 빈도 ↑, 품질 ↓ trade-off.
    min_confluence: int = Field(default=2, ge=0, le=10)
    # FVG 최소 % size — 지난 12거래 분석 결과 작은 FVG 노이즈 비중 커서 0.0004 → 0.0006 상향.
    fvg_min_size_pct: float = Field(default=0.0006, ge=0)
    step_interval_sec: int = Field(default=60, ge=10)
    ohlcv_limit: int = Field(default=1000, ge=50, le=1000)
    # setup stale threshold — FVG 이후 N 봉 안에 retest 없으면 진입 안 함.
    # 30봉 = 5m TF 에서 2.5시간 / 1h TF 에서 30시간. limit retest 시간 넉넉히 확보.
    # 2026-06-03: 30봉 (2.5h) 가드 너무 짧아 진입 빈도 낮음 → 120봉 (10h) 완화
    # (파트너 결정). ICT 정통 (fvg.filled / ob.mitigated) 가드가 별도 작동하므로
    # 시간 가드만 풀어도 안전.
    setup_stale_bars: int = Field(default=120, ge=1, le=500)
    # disable_time_filter: True 면 Silver Bullet / Killzone 시간 윈도우 무시 (24h 매매).
    # 라이선스 정책 (model_validator):
    #   - referral: 사용자 설정 따름 (default True = 24h)
    #   - sub_*: False 강제 (Killzone+미장)
    # 2026-05-26 default False 로 변경했으나 2026-05-27 데모 (v0.4.82 22h, 1W5L) 결과
    # referral 은 24h 가 더 나음 → 2026-05-27 default True 복원. env 로 opt-in 가능
    # (AURORA_ICT_DISABLE_TIME_FILTER=false 로 referral 도 Killzone).
    disable_time_filter: bool = Field(default=True)
    # multi_tf: True 면 HTF (Trade TF 위 전체) setup 추적 + LTF retrace/structure
    # shift/FVG confirm 시 진입 (ICT 정통). False 면 단일 TF 매매.
    multi_tf: bool = Field(default=False)
    multi_tf_ltf_lookback: int = Field(default=30, ge=5, le=200)
    # enable_trail: 진입 후 새 swing 형성 시 SL 이동 (ICT 정통 structure-based trail).
    enable_trail: bool = Field(default=False)
    trail_buffer_ratio: float = Field(default=0.001, ge=0.0, le=0.05)
    # use_market_entry (#LIVE-1 fix 후 의미 변경):
    # - False (기본/권장): marketable limit entry — 현재가 바로 앞 (캔들 앞) 에 지정가 +
    #   SL/TP 동봉. 슬리피지 0. entry_limit_ttl_sec 안에 미체결이면 취소 (타점 포기).
    # - True (레거시, 비권장): 즉시 시장가 — slippage 로 TP 가 fill 만큼 밀려 목표
    #   liquidity 못 먹던 #LIVE-1 원인.
    use_market_entry: bool = Field(default=False)
    # marketable limit 미체결 TTL (초). 이 시간 지나면 pending 취소. 사용자 결정
    # 2026-05-22: 600 (10분, 5m 2봉). 시장가처럼 거의 즉시 체결되되 슬리피지 0.
    entry_limit_ttl_sec: int = Field(default=600, ge=30, le=3600)
    # min_sl_distance_pct: SL 거리가 entry 의 이 비율 미만이면 setup skip.
    # 지난 12거래 분석 결과 SL 너무 짧은 setup 손실 비중 커서 0.0005 → 0.0007 상향.
    # 2026-05-29: 새벽 3연속 SL 풀히트 (실측 SL=0.32%) 회고 — ranging 시장에서
    # 타이트 SL 이 풀히트되는 패턴 → 0.0007 (0.07%) → 0.0025 (0.25%) 상향.
    # 2026-05-30: 0.25% 가드가 너무 빡빡해 setup 의 ~75% 차단 → 매매 0건 보고.
    # 0.25% → 0.20% 완화 (파트너 결정). 0.20% 이상이면 통과, 0.25% 도 자동 통과.
    # 단기 fresh setup 살리되 노이즈에 즉시 stop 당하는 정도의 SL 은 여전히 차단.
    # 2026-06-02: 0.20% 도 여전히 setup 대부분 차단 (실측 setup found 1086건
    # 진입 0건) → 0.20% → 0.12% 추가 완화 (파트너 결정). PR #147 직전 (0.07%)
    # 보다는 여전히 보수적이지만, 진입 빈도 회복.
    # 2026-06-03: 0.12% 도 진입 거의 없음 → 0.12% → 0.10% 추가 완화
    # (파트너 결정). PR #147 직전 (0.07%) 에 더 가깝게.
    min_sl_distance_pct: float = Field(default=0.001, ge=0.0, le=0.05)
    # max_sl_distance_pct: SL 거리가 entry 의 이 비율 초과면 setup skip.
    # 0 = 비활성. default 0.005 (0.5%) — 비정상 큰 SL 차단.
    max_sl_distance_pct: float = Field(default=0.005, ge=0.0, le=0.1)
    # max_entry_distance_pct: setup.entry 가 현재가에서 이 비율 초과면 setup skip.
    # 너무 멀리 박힌 limit 은 미체결 + ttl 만료까지 대기 시간 길어 setup 변형 위험.
    # 0 = 비활성. default 0.005 (0.5%) — 파트너 결정 2026-06-03.
    max_entry_distance_pct: float = Field(default=0.005, ge=0.0, le=0.1)
    # heartbeat_interval_sec: bot loop 살아있음 INFO 로그 주기 (0 = 비활성).
    heartbeat_interval_sec: int = Field(default=900, ge=0)
    # daily_loss_limit_pct (#SAFETY-1, 2026-05-21 사용자 결정):
    # 자본 대비 % 단위 일일 손실 한도. 0 = 비활성. > 0 면 활성:
    # 누적 손실 / 시작 equity * 100 ≥ 한도 → 새 진입 skip (active position 은 SL/TP 진행).
    # Reset 시점: NY local 자정 (ICT 정통 일일 boundary 일관).
    daily_loss_limit_pct: float = Field(default=0.0, ge=0.0, le=50.0)

    # HTF EMA bias 필터 — multi_tf 와 별개의 단순 directional filter.
    # 진입 직전 htf_ema_bias_tf (기본 1h) EMA20 vs 가격 비교 → 추세 방향 setup 만 진입.
    # DEPRECATED — htf_override_mode != "off" 면 ema_bias 는 무시된다.
    # 단순 directional EMA 필터보다 HTF FVG 가중치 맵 (override) 이 더 강한 시스템.
    htf_ema_bias_enabled: bool = Field(default=True)
    htf_ema_bias_tf: str = Field(default="1h")
    htf_ema_bias_period: int = Field(default=20, ge=2, le=200)

    # --- 신규 (변경 3) HTF FVG 가중치 override ----------------------------
    # off → 사용 안 함, A → 진입 직전 차단만, C → 진입 + 봉 close 기준 flip + re-entry.
    htf_override_mode: str = Field(default="C")
    # HTF FVG 맵 빌드에 사용할 TF 들. 5m 은 LTF 라 별도 가중치 (1) 만 부여.
    htf_fvg_tfs: tuple[str, ...] = Field(
        default=("15m", "1h", "2h", "4h", "1d", "1w"),
    )

    # 2026-05-29 #HTF-LTF-CONFLICT: HTF FVG bull/bear weight 명확 우세 + LTF setup
    # 반대 방향 진입 차단. 5-29 실시간 매매 회고:
    #   #3 (bull_w=246, bear_w=218, ratio=1.128) SHORT 진입 → SL_HIT
    #   #5 (bull_w=244, bear_w=218, ratio=1.119) SHORT 진입 → SL_HIT
    # HTF override threshold 강화 (#147) 만으로 부족 — 우세하지만 threshold 미만인
    # ratio 구간이 진입 → 풀히트. 별도 가드로 ratio >= 임계 시 차단.
    # 값: 우세 비율 (1.10 = bull 이 bear 의 110% 이상이면 short 차단, 역도 마찬가지).
    # 0 = 비활성. 효과 부족 시 0.05 단위로 낮춤 (1.05 가장 엄격).
    htf_ltf_conflict_guard_ratio: float = Field(default=1.10, ge=0.0, le=2.0)

    # --- 신규 (변경 7) 실시간 flip watcher (WS + polling fallback) ---------
    # ICT 정통: FVG zone 1회 touch = mitigation 인정. 5분 봉 close 대기 X — 즉시 flip.
    # 정확성 우선: WS tick 받아도 flip 직전 REST 로 재확인 후 청산/진입 sequential.
    flip_watch_enabled: bool = Field(default=True)
    flip_watch_ws_url: str = Field(default="wss://stream.bybit.com/v5/public/linear")
    flip_watch_polling_interval_sec: float = Field(default=0.2, ge=0.05, le=5.0)
    flip_watch_ws_reconnect_max: int = Field(default=5, ge=1, le=20)
    # FVG 무효화 — touch 3회 누적 시 더 이상 mitigation 후보로 안 봄 (약화).
    htf_fvg_max_touch_count: int = Field(default=3, ge=1, le=20)

    demo_api_key: SecretStr = Field(default=SecretStr(""))
    demo_api_secret: SecretStr = Field(default=SecretStr(""))
    live_api_key: SecretStr = Field(default=SecretStr(""))
    live_api_secret: SecretStr = Field(default=SecretStr(""))

    # --- 라이선스 티어 (G-3b, 2026-05-21 합의) ---------------------------
    # launcher 가 spawn 시 ``AURORA_ICT_LICENSE_TYPE`` env 로 박음.
    # 매매 시간대 정책 자동 분기:
    #   - referral (평생): 24h 매매 (disable_time_filter=True 유지)
    #   - sub_30d/90d/365d (구독제): Killzone 시간만 (disable_time_filter=False 강제)
    # 사용자가 settings UI/.env 로 override 시도해도 라이선스 정책이 우선.
    license_type: str = Field(default="referral")

    @field_validator("license_type")
    @classmethod
    def _validate_license_type(cls, v: str) -> str:
        allowed = {"referral", "sub_30d", "sub_90d", "sub_365d"}
        if v not in allowed:
            # 잘못된 값이면 가장 제한적인 referral 로 fallback (보수적 — 봇 끊기지 X)
            return "referral"
        return v

    @model_validator(mode="after")
    def _enforce_license_tier_policy(self) -> IctSettings:
        """라이선스 티어별 매매 시간 정책 강제 (G-3b).

        구독제 (``sub_*``): ``disable_time_filter`` 를 False 로 강제 — 사용자가
        ``.env`` 로 ``AURORA_ICT_DISABLE_TIME_FILTER=true`` 박아도 무시.
        Killzone 시간대 (London/NY AM/Close/PM) 만 매매.

        레퍼럴 (``referral``): 별도 강제 X — 사용자 settings 그대로 (기본 24h).
        """
        if self.license_type.startswith("sub_"):
            self.disable_time_filter = False
        return self

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
