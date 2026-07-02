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

import re
from enum import StrEnum
from pathlib import Path

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 매매 timeframe 허용 목록 — 5m 이상.
# LTF (5m/15m/30m) 는 ICT 정통 entry TF (HTF bias + LTF refined entry).
# 1m 은 노이즈 과대로 비허용.
TRADE_TIMEFRAMES: tuple[str, ...] = ("5m", "15m", "30m", "1h", "2h", "4h", "1d", "1w")

# 2026-06-17 #ORIGO-1: 페어별 베스트 진입대기 ttl(초) override.
# 7페어 5년 백테스트 — BTC 는 1h(3600)가 최적(robust 검증), 나머지는 30분(기본).
# trail·전체ttl1h 처럼 BTC 단독 함정을 피하려 페어별로 검증된 값만 예외 부여.
PAIR_TTL_OVERRIDES: dict[str, int] = {"BTCUSDT": 3600}

# 2026-06-17 #ORIGO-MODEL: 현재 봇 모델명 — 매매 기록에 어느 모델로 매매됐는지 태그.
# 버전 프리셋이 늘면 settings 필드로 전환. 일단 단일 모델 상수.
# 2026-06-23 Origo 1.2 = 안정형 하이브리드(0.707 OTE + 횡보회피 게이트 + 분할익절
# + 횡보임계 롤링분위). 1.1(cisd+po3+OTE) 위에 시드방어형 전환 적용.
# 2026-07-02 Origo 1.3 = 진입 엣지 상향 (FST #1 자율연구, 7페어 5년 백테스트):
# min_confluence 4→5 + sl_dist_mult x3→x4 로 5년 net -139→+124 USDT 흑자 전환.
# 진입을 엄격히 걸러 이기는 판만 남김 (빈도 0.70→0.24/일 trade-off 수용).
ORIGO_MODEL_NAME = "Origo 1.3"
# 2026-06-25 #CURSUS: 투트랙 2번째 봇 = Cursus(Dual SuperTrend 추세형, dual_st).
# bot_trend_instance.CURSUS_MODEL_NAME 과 동일 문자열 유지(매매기록 model 태그 정합).
CURSUS_MODEL_NAME = "Cursus 1.0"
# 모델 선택 레지스트리 — 표시명 → 전략 id. 사용자가 model 선택 시 multi_user 가
# origo→BotIctInstance, cursus→BotTrendInstance 로 분기. UI 드롭다운도 이 목록 사용.
AVAILABLE_MODELS: dict[str, str] = {
    ORIGO_MODEL_NAME: "origo",
    CURSUS_MODEL_NAME: "cursus",
}
DEFAULT_MODEL_NAME = ORIGO_MODEL_NAME


def origo1_ttl_for_symbol(symbol: str, default_ttl: int) -> int:
    """심볼별 베스트 진입대기 ttl(초) — BTC 1h, 나머지 default(보통 30분).

    Args:
        symbol: 거래 심볼 (예 "BTCUSDT").
        default_ttl: override 없는 심볼의 기본 ttl(초).
    Returns:
        해당 심볼의 ttl(초).
    """
    return PAIR_TTL_OVERRIDES.get(symbol, default_ttl)


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
        """ccxt Bybit linear perpetual 형식 검증.

        2026-06-05 페어 확장: BTC/ETH 정적 화이트리스트 → 형식 검증으로 완화.
        실제 거래 가능 여부(거래대금 상위 N 화이트리스트)는 가동 단계
        (MultiUserBotManager.get_or_create_bot)에서 PairRegistry 로 확인한다.
        field_validator 는 클래스 메서드라 거래소 조회가 불가하므로 형식만 본다.
        ``BASE/USDT:USDT`` (BASE = 영문 대문자·숫자) 형식만 허용 — path/주입 차단.
        """
        if not re.fullmatch(r"[A-Z0-9]+/USDT:USDT", v):
            raise ValueError(
                f"symbol '{v}' 형식 오류 — 'XXX/USDT:USDT' 형식이어야 합니다.",
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
    # 2026-06-05 파트너 결정: 90→80. margin=equity*90% 면 Bybit 개시수수료+
    # 청산버퍼(남은 10%)를 못 감당해 110007 "ab not enough" 거부됨. 80% 로
    # 낮춰 ~20% 여유 확보.
    position_pct_max: float = Field(default=80.0, ge=1.0, le=100.0)
    # confluence_score 0 → base, 1/2/3+ → base + step * score (max capped).
    # 사용자 정책: 0→40, 1→55, 2→70, 3+→80 (step=15, max 80 cap).
    position_pct_step: float = Field(default=15.0, ge=0.0, le=50.0)
    # 2026-06-06 리스크 기반 sizing (파트너 결정) — True 면 고정 % notional 대신
    # 건당 리스크(equity %)를 고정하고 qty = risk금액 / SL거리 로 역산. SL 이 멀수록
    # qty 가 줄어 건당 손실(R)이 일정 → max_sl_distance 게이트 우회(아래 참조).
    # 2026-06-06 파트너 결정: 기본 ON + 공격 3~6% (매매 적던 SL 게이트 문제 해소).
    risk_based_sizing: bool = Field(default=True)
    # 건당 리스크 % (equity 대비). score 0→base, 1/2/3+→base+step*score (max cap).
    # 공격 설정: base 3.0 / step 1.5 / max 6.0 → 0:3%, 1:4.5%, 2:6%, 3+:6%.
    risk_per_trade_base: float = Field(default=3.0, gt=0.0, le=20.0)
    risk_per_trade_step: float = Field(default=1.5, ge=0.0, le=10.0)
    risk_per_trade_max: float = Field(default=6.0, gt=0.0, le=20.0)
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
    # 고RR 예외 구멍 — confluence 미달이어도 손익비(rr)가 이 값 이상이고 score>=1
    # 이면 단일신호 셋업도 진입 통과. 0=비활성. 파트너 결정 2026-06-04:
    # rr 2.5+ 1점 셋업은 confluence 게이트 우회 (손익비 좋은 단일신호 살리기).
    # 2026-06-05: 매매 빈도 너무 낮아 2.5 → 2.3 하향 (1점 셋업 진입 문턱 완화).
    # 2026-06-06: 2.3 → 3.0 상향 — 백테스트상 bypass 2.3 이 confluence 게이트를
    #   무력화해 약한 단일신호 과다 통과(net -129% vs bypass 0 의 -98%).
    # 2026-06-06(2): 3.0 → 0.0 완전 비활성 (파트너 결정 + 종합 백테스트). 가점
    #   CISD/SMT 다 켠 + 킬존 ON 조건에서도 1점&RR>=3 우회가 ETH 승률 16.7→14.3%,
    #   net -150→-193% 로 악화(1점 셋업이 순손실 덩어리). BTC 는 무영향. → 1점
    #   우회 제거, 등급2 이상만 진입(레퍼럴·구독제 전 사용자 공통).
    high_rr_bypass_min_rr: float = Field(default=0.0, ge=0.0)
    # SMT divergence (BTC↔ETH 상관) confluence 가점 — #SMT 2026-06-06.
    # 두 상관 자산이 같은 시점 swing 에서 한쪽만 새 고/저점을 박으면 '기관 흐름
    # 누설' → 못 따라온 쪽 반전 신호. setup 방향과 일치 시 confluence +1.
    # 짝 없는 알트 심볼은 자동 skip. False 면 SMT 평가 자체 안 함.
    smt_enabled: bool = Field(default=True)
    # FVG 최소 % size — 지난 12거래 분석 결과 작은 FVG 노이즈 비중 커서 0.0004 → 0.0006 상향.
    fvg_min_size_pct: float = Field(default=0.0006, ge=0)
    step_interval_sec: int = Field(default=60, ge=10)
    ohlcv_limit: int = Field(default=1000, ge=50, le=1000)
    # setup stale threshold — FVG 이후 N 봉 안에 retest 없으면 진입 안 함.
    # 30봉 = 5m TF 에서 2.5시간 / 1h TF 에서 30시간. limit retest 시간 넉넉히 확보.
    # 2026-06-03: 30봉 (2.5h) 가드 너무 짧아 진입 빈도 낮음 → 120봉 (10h) 완화
    # (파트너 결정). ICT 정통 (fvg.filled / ob.mitigated) 가드가 별도 작동하므로
    # 시간 가드만 풀어도 안전.
    # 2026-06-17: 백테스트 t6/s3 정합 — 120봉(10h)은 질 낮은 진입 양산해 5년 −8% 적자
    # 였고, 3봉(15분)으로 타이트하게 줄여야 흑자(cisd+po3 조합 시 +3.18%). cisd+po3
    # 가점이 게이트 통과를 늘려(진입 186→598건) 짧은 stale 의 빈도 감소를 보완 (파트너 결정).
    setup_stale_bars: int = Field(default=3, ge=1, le=500)
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
    # marketable limit 미체결 TTL (초). 이 시간 지나면 pending 취소 후 새 셋업 재탐색.
    # 2026-05-22: 600(10분). 2026-06-05 파트너 결정: 300(5분, 5m 1봉) — 5분 안에
    # 체결 안 되면 타점 포기하고 새로 잡기.
    # 2026-06-11 #EDGE-V2: 상한 3600→14400 — 백테스트 검증값(120분 대기)이
    # 들어갈 수 있게. 좋은 자리는 타점 되돌림을 길게 기다리는 게 체결률·성과 우위.
    # 2026-06-17: 백테스트 t6/s3 정합 — entry_ttl 30분(5m×6봉=1800초). stale 15분과
    # 짝. 좋은 타점 되돌림을 30분 기다려 체결률↑ (cisd+po3 5년 robust 흑자 구성).
    entry_limit_ttl_sec: int = Field(default=1800, ge=30, le=14400)
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
    # 0 = 비활성. 비정상 큰 SL 차단. 2026-06-04 파트너 결정: 0.5% 가 너무
    # 타이트해 turtle_soup 등 rr 좋은(2.8~3.4) 셋업이 SL 0.65~0.72% 로 다 탈락
    # → 0.75% 로 완화 (고RR 예외 게이트 #215 도달 가능하게).
    max_sl_distance_pct: float = Field(default=0.0075, ge=0.0, le=0.1)
    # 2026-06-11 #EDGE-V2: SL 거리 배수 (1.0=원본). 백테스트 10국면(5년) 검증 —
    # 넓힐수록 스탑헌트 생존으로 단조 개선. TP 는 원 RR 유지 비례 확장,
    # risk_based_sizing ON 이면 qty 가 줄어 건당 손실(R) 불변.
    sl_dist_mult: float = Field(default=1.0, ge=0.25, le=5.0)
    # 2026-06-18 #CT-SL 국면별 동적 SL: 진입 시점 "방향 정합 추세"(signed_trend
    # = 진입직전 20봉 변화율 × 방향부호)가 ct_trend_threshold 미만 = 역추세(되돌림
    # 진입)이면 sl_dist_mult 대신 sl_dist_mult_ct 사용. 0=비활성(항상 sl_dist_mult).
    # 근거: 7페어 5년 백테스트에서 역추세 분위 x4 가 전·후반 robust(되돌림이 터지면
    # 큰 폭 → 먼 TP 가 먹음), 순추세 x3 유지. net +4.0%p·승률 유지(Origo 1.1 개선).
    sl_dist_mult_ct: float = Field(default=0.0, ge=0.0, le=5.0)
    ct_trend_threshold: float = Field(default=0.0, ge=-100.0, le=100.0)
    # 2026-06-11 #SHADOW: 거른 setup 도 특징과 함께 기록(FSD-style 플라이휠).
    # 행동 영향 0, 사용자별 shadow_setups.jsonl. 오프라인 학습 데이터 축적용.
    shadow_log_enabled: bool = Field(default=True)
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
    # daily_profit_limit_pct (2026-06-10 조윤 건의):
    # 자본 대비 % 단위 일일 수익(TP) 한도. 0 = 비활성. > 0 면 활성:
    # 누적 수익 / 시작 equity * 100 ≥ 한도 → 그날 새 진입 중단 ("몇 % 먹고 종료").
    # Reset 시점: 손실 한도와 동일하게 NY local 자정.
    daily_profit_limit_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    # daily_pair_loss_limit_r (2026-06-12 파트너 결정):
    # 페어별 일일 손실 한도 — R(리스크% 1회분) 배수 단위. 이 페어의 오늘 누적
    # 손실이 R×배수에 닿으면 *그 페어만* 당일 진입 중단 (다른 페어는 계속).
    # 단일 페어 폭주(6/6: 한 페어 19연속 -33%) 차단. 기본 2R ON. 0 = 비활성.
    daily_pair_loss_limit_r: float = Field(default=2.0, ge=0.0, le=20.0)

    # HTF EMA bias 필터 — multi_tf 와 별개의 단순 directional filter.
    # 진입 직전 htf_ema_bias_tf (기본 1h) EMA20 vs 가격 비교 → 추세 방향 setup 만 진입.
    # DEPRECATED — htf_override_mode != "off" 면 ema_bias 는 무시된다.
    # 단순 directional EMA 필터보다 HTF FVG 가중치 맵 (override) 이 더 강한 시스템.
    htf_ema_bias_enabled: bool = Field(default=True)
    htf_ema_bias_tf: str = Field(default="1h")
    htf_ema_bias_period: int = Field(default=20, ge=2, le=200)
    # 2026-06-10 #ALIGN: 다중 EMA 정렬 게이트 (백테스트 10국면 검증 — 단일 EMA20
    # 보다 방향 정확도↑, 반등·상승장에서 숏 고착 완화). htf_ema_bias_enabled 이고
    # 이게 True 면 prefer_direction 과 진입 게이트를 인접 EMA 쌍(periods) 정배열/
    # 역배열 점수로 결정. |점수|>=threshold 면 그 방향만, 미만이면 진입 자제.
    htf_ema_align_enabled: bool = Field(default=True)
    htf_ema_align_periods: tuple[int, ...] = Field(
        default=(60, 120, 200, 350, 480, 620),
    )
    htf_ema_align_threshold: int = Field(default=2, ge=1, le=5)

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

        2026-06-11 #EDGE-V2 (흑자 엣지 최종, 백테스트 5년 10국면 + IN/OUT 분리
        + BTC robust 40조합 클러스터 검증):
            구독제 = 등급4 + RR2.5 + SL거리 x3.0 + 진입 대기 120분 + 킬존.
            BTC IN/OUT 흑자(+0.2/+1.4%), ETH 본전권. 빈도는 페어 수로 확장.
        강제는 전부 "최소" 방향 (사용자가 더 보수적으로 올린 값은 유지).

        2026-06-12 #FRESH-30: setup 신선도 30분 — 신호 후 30분 이내 자리만 진입.
        5페어 교차검증: 신선할수록 단조 개선, ETH 가 IN/OUT 흑자 전환, 5페어
        합산 +1.99→+2.88%. setup_stale_bars 는 *매매 TF 봉* 단위라 분→봉 환산
        (5m→6봉). 기존 기본 120봉은 5m 에서 10시간 — 검증 범위(2h) 20배 밖
        이었던 단위 불일치도 함께 해소.
        """
        if self.license_type.startswith("sub_"):
            self.disable_time_filter = False
            # 2026-06-17 #ORIGO-1: 5분봉 강제 (검증된 베스트 TF). env/사용자가 다른
            # TF 로 바꿔놔도 무시 — 라이브가 15m 로 새어 백테스트 엣지가 깨지던 문제
            # 원천 차단 (버전당 베스트 TF 고정).
            self.timeframe = "5m"
            # 2026-07-02 #ORIGO-1.3 진입 엣지 (FST #1 자율연구, 7페어 5년):
            # conf 4→5 + SL x3→x4 조합이 net -139→+124 USDT 유일 흑자 전환.
            # conf5 단독 +58, sl4 는 스탑헌트 생존으로 승률 방어. htf4 추가는
            # 과최적화(빈도 급감 +81)라 기각. 빈도 0.70→0.24/일 trade-off 수용
            # (FST 진단: net 병목은 빈도가 아니라 진입 엣지).
            if self.min_confluence < 5:
                self.min_confluence = 5
            if self.min_rr < 2.5:
                self.min_rr = 2.5
            if self.sl_dist_mult < 4.0:
                self.sl_dist_mult = 4.0
            # #CT-SL: 역추세(되돌림) 진입은 x4 (robust). 순추세/횡보는 위 x3 유지.
            self.sl_dist_mult_ct = 4.0
            self.ct_trend_threshold = 0.0
            # #ORIGO-1: ttl 30분(1800) 강제 — 7페어 5년 백테스트 최적(+9.54%).
            # BTC 만 manager 에서 1h(3600) override(페어별 베스트 ttl). 기존 2h(7200)는
            # 7페어 백테스트 손실 구간이라 폐기.
            self.entry_limit_ttl_sec = 1800
            # stale 15분(3봉 @5m) — 백테스트 t6/s3 정합 (신선한 자리만 진입).
            fresh_bars = max(1, 15 // max(1, self.timeframe_minutes))
            if self.setup_stale_bars > fresh_bars:
                self.setup_stale_bars = fresh_bars
        return self

    @property
    def timeframe_minutes(self) -> int:
        """매매 timeframe 의 분 단위 환산 (예 "5m"→5, "1h"→60). 불명이면 5."""
        tf = (self.timeframe or "").strip().lower()
        try:
            if tf.endswith("m"):
                return int(tf[:-1])
            if tf.endswith("h"):
                return int(tf[:-1]) * 60
            if tf.endswith("d"):
                return int(tf[:-1]) * 1440
            if tf.endswith("w"):
                return int(tf[:-1]) * 10080
        except ValueError:
            pass
        return 5

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
