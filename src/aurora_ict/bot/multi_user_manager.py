"""MultiUserBotManager — 사용자별 BotIctInstance 격리 관리.

파트너 결정 2026-05-28 — Aurora-ICT SaaS 전환의 핵심 lifecycle 관리자.

설계:
    - 단일 서버 프로세스 하나가 여러 사용자 봇 task 를 동시 운영.
    - 사용자 코드 → BotIctInstance 매핑 (in-memory dict).
    - 각 사용자 데이터/저장소는 ``<data_dir>/users/<code>/`` 로 분리.
    - 거래소 client 는 사용자별 settings (복호화한 api_key/secret) 로 생성.
    - 기존 ``BotManager`` (manager.py) 와 별개 — CLI/.exe 단일 사용자 호환 보존.

차이점 — BotManager 와 비교:
    - BotManager: 하나의 settings + 하나의 bot. CLI/.exe 흐름.
    - MultiUserBotManager: 사용자별 settings 동적 생성 + 사용자별 bot. FastAPI SaaS.

stop_all:
    서버 종료 (FastAPI ``lifespan`` shutdown hook) 시 호출해서 모든 사용자 봇 정지.
    aiohttp ClientSession 누수 방지 위해 client 도 명시 close.

담당: 지영민 (SaaS 전환 PR)
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import SecretStr

from aurora_ict.auth import keystore, users_db
from aurora_ict.bot.bot_ict_instance import (
    BotIctInstance,
    BotState,
    ExchangeClientProtocol,
)
from aurora_ict.bot.pair_registry import (
    EXCLUDED_PAIRS,
    FIXED_PAIRS,
    MAJOR_PAIRS,
    PairRegistry,
)
from aurora_ict.config.settings import (
    TRADE_TIMEFRAMES,
    IctSettings,
    RunMode,
    origo1_ttl_for_symbol,
)

# 페어 확장 (파트너 결정 2026-06-05 → 고정7 구도 2026-06-12):
#   - 고정 7(FIXED_PAIRS, 백테스트 검증) + 사용자 선택 3 = 총 10.
#   - 상한은 서버 부하(사용자수 × 페어수 곱셈) 방지 겸 검증 안 된 조합 억제.
#   - 알트(BTC/ETH 외)는 변동성 커서 레버리지 15배 고정.
MAX_PAIRS_PER_USER = 10
MAX_CHOICE_PAIRS = 3  # 고정 7 외 사용자 자유 선택 상한
_ALT_LEVERAGE = 15

logger = logging.getLogger(__name__)

# client factory 시그니처 — settings 받아 ExchangeClient 반환 (BotManager 와 동일).
ClientFactory = Callable[[IctSettings], Awaitable[ExchangeClientProtocol]]


# 2026-05-29 PR B: 멀티 페어 (BTC + ETH 동시 진입) 백엔드. 사용자별 단일 슬롯
# 구조 → (사용자, symbol) 복합 키 슬롯. ccxt 형식 ("BTC/USDT:USDT") 사용.
# 옛 호출 (symbol 안 주는 단일 사용자/.exe 흐름) 호환 위해 default 박음.
_DEFAULT_SYMBOL = "BTC/USDT:USDT"
SlotKey = tuple[str, str]  # (user_code, symbol)

# #SEC path traversal 방어 — user_code 가 파일 경로에 들어가므로 영숫자/_/- 만 허용.
# auth.router._CODE_PATTERN 과 동일 규칙(입력단·디렉토리 생성단 이중 검증).
_SAFE_USER_CODE_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# partial_tp_exchange 시범 적용 유저 화이트리스트 — 진입 시 거래소에 TP 2개(1.5R 50%
# Partial + swing 50% Entire) 미리 등록(봇 폴링 _maybe_partial_exit 대체). ⚠️ Bybit
# Partial 모드 소액 실측 단계라 환경변수로 지정한 유저만 ON. 미설정이면 빈 셋 → 전원 OFF.
# fly secrets set PARTIAL_TP_EXCHANGE_USERS="AICT-0Q8B-D1YU-VFRN" (콤마로 다중 지정).
_PARTIAL_TP_USERS: frozenset[str] = frozenset(
    c.strip()
    for c in os.environ.get("PARTIAL_TP_EXCHANGE_USERS", "").split(",")
    if c.strip()
)


@dataclass(slots=True)
class _UserBotSlot:
    """사용자 1명 × 1 symbol 의 봇 슬롯 — bot/client/settings 함께 보관."""

    symbol: str
    settings: IctSettings
    bot: BotIctInstance | None = None
    client: ExchangeClientProtocol | None = None


@dataclass
class MultiUserBotManager:
    """사용자별 BotIctInstance lifecycle 관리.

    Attributes:
        client_factory: settings → ExchangeClient async factory.
        db_path: users.db 경로 (api_key/secret 복호화에 사용).
        base_settings: 기본 settings — 사용자별 instance 가 이 값을 복사 후
            api_key/secret 만 덮어씀. None 이면 IctSettings() 기본.
        master_key: keystore Fernet 키. None 이면 환경변수/파일 기본 동작.
    """

    client_factory: ClientFactory
    db_path: Path | str
    base_settings: IctSettings | None = None
    master_key: bytes | None = None
    # 2026-06-08: 텔레그램 매매 알림 발송기(TelegramAlerter) — saas 가 주입.
    # None 이면 알림 비활성(콜백 미주입).
    alerter: Any = None
    # 2026-05-29 PR B: (user_code, symbol) 복합 키. 한 사용자가 BTC + ETH 등
    # 여러 페어 동시 진입 가능.
    _pair_registry: PairRegistry = field(default_factory=PairRegistry)
    # UI 페어 선택기용 시세 행 캐시(symbol/last/pct24h/volume) — 60초 TTL.
    _market_rows_cache: list[dict[str, Any]] = field(default_factory=list)
    _market_rows_at: float = 0.0
    _slots: dict[SlotKey, _UserBotSlot] = field(default_factory=dict)
    # (user_code, symbol) 별 start lock — 같은 사용자×symbol 동시 start race 방지.
    # 다른 사용자 / 다른 symbol 끼리는 병렬 가능.
    _locks: dict[SlotKey, asyncio.Lock] = field(default_factory=dict)

    def _get_lock(
        self, user_code: str, symbol: str = _DEFAULT_SYMBOL,
    ) -> asyncio.Lock:
        """(사용자, symbol) 별 asyncio.Lock — 없으면 lazy create.

        Why per-(user, symbol): 한 사용자가 BTC + ETH 동시 가동 시 둘이 서로
        막지 않게. 같은 (user, symbol) 더블 클릭만 차단.
        """
        key: SlotKey = (user_code, symbol)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def _user_data_dir(self, user_code: str) -> Path:
        """사용자별 데이터 디렉토리 — ``<data_dir>/users/<code>/``.

        Why: 사용자별 license.json / trades.db / trades_dataset 등 데이터 분리.
        한 사용자 데이터가 다른 사용자에게 노출/오염되지 않도록 격리.

        #SEC path traversal 2차 방어(defense-in-depth): user_code 는 auth 입력단
        (router._CODE_PATTERN)에서 이미 검증되지만, 디렉토리 생성 직전에 한 번 더
        형식·경로 봉쇄를 재확인한다. ``../`` 등으로 users/ 루트를 벗어나면 예외.

        Raises:
            ValueError: user_code 가 허용 문자집합을 벗어나거나, 해석된 경로가
                ``<data_dir>/users`` 하위가 아닐 때.
        """
        from aurora_ict.paths import data_dir
        if not _SAFE_USER_CODE_RE.match(user_code):
            raise ValueError(
                f"부적합한 user_code (path traversal 방어): {user_code!r}",
            )
        root = (data_dir() / "users").resolve()
        path = (root / user_code).resolve()
        if path != root and root not in path.parents:
            raise ValueError(
                f"user_code 가 users 디렉토리를 벗어남: {user_code!r}",
            )
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _build_user_settings(
        self,
        user_code: str,
        symbol: str = _DEFAULT_SYMBOL,
        *,
        force_run_mode: RunMode | None = None,
    ) -> IctSettings:
        """DB row → 사용자별 IctSettings (api_key/secret 복호화 포함).

        2026-05-29 (Live 모드 지원): 사용자가 demo / live 키를 각각 등록할 수
        있게 되어 양쪽 슬롯을 모두 DB 에서 가져옴. 현재 ``run_mode`` 의 키만
        필수 — 그 키가 없으면 ValueError (start 차단). 반대편 모드 키는
        있으면 settings 에 박고, 없으면 빈 SecretStr 로 둔다 (모드 전환 시
        ``/ict/run-mode`` 가드에서 다시 검증).

        2026-05-29 hot-fix: ``force_run_mode`` 인자 — auto_resume 가 사용자
        마지막 가동 run_mode 로 강제. base_settings.run_mode (DEMO 기본) 가
        LIVE 사용자 재가동을 막던 버그 해소.

        Args:
            user_code: 대상 사용자 라이선스 코드.
            force_run_mode: 명시 시 settings.run_mode override (auto_resume 용).

        Returns:
            base_settings 를 깊은 복사한 IctSettings 에 사용자별 demo/live
            api_key/secret 박은 것.

        Raises:
            ValueError: 사용자 미존재 또는 현재 run_mode 의 키 미등록.
        """
        user = users_db.get_user_by_code(self.db_path, user_code)
        if user is None:
            raise ValueError(f"사용자 '{user_code}' 가 DB 에 없습니다.")
        license_type = user.get("license_type", "referral")

        # base_settings 복사 — pydantic v2 model_copy 깊은 복사.
        base = self.base_settings if self.base_settings is not None else IctSettings()
        settings = base.model_copy(deep=True)
        settings.license_type = license_type
        # 2026-05-29 PR B: symbol 강제 (멀티 페어 — 슬롯별 symbol 정확히 박음).
        settings.symbol = symbol
        # 2026-06-08: 사용자별 마지막 매매 timeframe 복원 — 가동 중 TF 변경 후
        # 재시작/재가동 시 전역 base 로 회귀하던 버그(1h→15m) 해소. 미설정이면
        # base 값 유지. 잘못된 값은 무시(유효 TF 만 적용).
        try:
            _last_tf = users_db.get_last_timeframe(self.db_path, user_code)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "사용자 %s last_timeframe 조회 실패 (무시): %s", user_code, e,
            )
            _last_tf = None
        if _last_tf and _last_tf in TRADE_TIMEFRAMES:
            settings.timeframe = _last_tf
        # 페어 확장 2026-06-05: 알트(BTC/ETH 외)는 변동성 커서 레버리지 15배 고정.
        # BTC/ETH 는 사용자 설정값 유지. 거래소 max leverage 가 15 미만이면 가동 시
        # set_leverage 실패 → 실제값 보정(#LEV-1)이 받아냄.
        if symbol not in MAJOR_PAIRS:
            settings.leverage = _ALT_LEVERAGE
        # 2026-05-29 hot-fix #1: force_run_mode 명시 시 그 값으로 강제 override.
        # 사용자 마지막 가동 모드 복원 (auto_resume 흐름).
        # 2026-05-29 hot-fix #2 (파트너 보고): force_run_mode 미명시 시 DB 의
        # last_run_mode 를 폴백으로 사용. 사용자가 UI 에서 LIVE 로 전환한
        # 상태에서 START 누르면 /ict/start 핸들러가 force_run_mode 없이 호출
        # 하는데, base_settings.run_mode 가 DEMO 기본이라 DEMO 키 검사 → "DEMO
        # API 키 미등록" 에러로 가동 차단. last_run_mode 폴백으로 사용자가
        # 마지막 선택한 모드 그대로 가동.
        if force_run_mode is not None:
            settings.run_mode = force_run_mode
        else:
            try:
                last_mode_str = users_db.get_last_run_mode(
                    self.db_path, user_code,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "사용자 %s last_run_mode 조회 실패 (무시): %s",
                    user_code, e,
                )
                last_mode_str = None
            if last_mode_str == "live":
                settings.run_mode = RunMode.LIVE
            elif last_mode_str == "demo":
                settings.run_mode = RunMode.DEMO
            elif settings.run_mode is RunMode.DEMO:
                # 2026-06-06 DEMO UI 제거(실매매 전용): 첫 가동(last_run_mode 미설정) +
                # base 가 DEMO 기본일 때, live 키가 등록돼 있으면 LIVE 로 보정. 신규
                # 베타는 LIVE 키만 등록하므로 LIVE 가동 → "DEMO 키 미등록" 해소.
                # base 가 명시 LIVE 면 그대로 두고 아래 live 키 검사에 맡긴다.
                _live_keys = users_db.get_api_keys(self.db_path, user_code, "live")
                if _live_keys is not None:
                    settings.run_mode = RunMode.LIVE
        effective_mode = settings.run_mode

        # 양쪽 모드 키 로드 — 있는 것만 박는다.
        demo_pair = users_db.get_api_keys(self.db_path, user_code, "demo")
        live_pair = users_db.get_api_keys(self.db_path, user_code, "live")
        if demo_pair is not None:
            demo_key, demo_sec_enc = demo_pair
            demo_secret = keystore.decrypt_secret(demo_sec_enc, key=self.master_key)
            settings.demo_api_key = SecretStr(demo_key)
            settings.demo_api_secret = SecretStr(demo_secret)
        if live_pair is not None:
            live_key, live_sec_enc = live_pair
            live_secret = keystore.decrypt_secret(live_sec_enc, key=self.master_key)
            settings.live_api_key = SecretStr(live_key)
            settings.live_api_secret = SecretStr(live_secret)

        # 2026-05-29 hot-fix: 검사 기준을 force_run_mode 반영 후의 effective_mode 로.
        # base.run_mode 그대로 검사하면 force_run_mode 무효화됨.
        if effective_mode is RunMode.LIVE and live_pair is None:
            raise ValueError(
                f"사용자 '{user_code}' 의 LIVE 거래소 API 키가 등록되지 않았습니다.",
            )
        if effective_mode is RunMode.DEMO and demo_pair is None:
            raise ValueError(
                f"사용자 '{user_code}' 의 DEMO 거래소 API 키가 등록되지 않았습니다.",
            )

        # 모델 validator 재실행 — license_type 변경에 따른 disable_time_filter 강제.
        settings = settings.model_validate(settings.model_dump())
        return settings

    async def get_or_create_bot(
        self,
        user_code: str,
        symbol: str = _DEFAULT_SYMBOL,
        *,
        force_run_mode: RunMode | None = None,
    ) -> BotIctInstance:
        """사용자 × symbol 봇 인스턴스 가져오기 — 없으면 생성 (start 는 별도).

        2026-05-29 hot-fix: ``force_run_mode`` 인자 — auto_resume 가 사용자
        마지막 가동 모드로 강제 가동할 수 있게.
        2026-05-29 PR B: ``symbol`` 인자 — 한 사용자가 BTC + ETH 등 여러
        페어를 동시 가동할 수 있도록 (user_code, symbol) 튜플 키로 슬롯 격리.

        Args:
            user_code: 대상 사용자 라이선스 코드.
            symbol: ccxt 형식 페어 (예: ``"BTC/USDT:USDT"``, ``"ETH/USDT:USDT"``).
                기본값은 backward compat 위해 BTC.
            force_run_mode: 명시 시 IctSettings.run_mode override.

        Returns:
            BotIctInstance (생성만, state=STOPPED 일 수 있음).
        """
        key: SlotKey = (user_code, symbol)
        slot = self._slots.get(key)
        if slot is not None and slot.bot is not None:
            return slot.bot
        # 페어 확장 — 동시 가동 페어 수 상한 (신규 슬롯일 때만 카운트).
        # 고정7 구도(2026-06-12): 고정 페어는 상한 10 안에서 항상 허용,
        # 자유 선택 페어는 고정 외 MAX_CHOICE_PAIRS(3)개까지.
        existing = {s for (u, s) in self._slots if u == user_code}
        if symbol not in existing:
            if symbol in EXCLUDED_PAIRS:
                raise ValueError(
                    f"'{symbol}' 는 수익성 검증에서 탈락해 제외된 페어입니다.",
                )
            if len(existing) >= MAX_PAIRS_PER_USER:
                raise ValueError(
                    f"동시 가동 페어는 최대 {MAX_PAIRS_PER_USER}개입니다 "
                    f"(현재 {len(existing)}개 — 일부 정지 후 추가하세요).",
                )
            if symbol not in FIXED_PAIRS:
                choice = {s for s in existing if s not in FIXED_PAIRS}
                if len(choice) >= MAX_CHOICE_PAIRS:
                    raise ValueError(
                        f"선택 페어는 고정 {len(FIXED_PAIRS)}개 외 최대 "
                        f"{MAX_CHOICE_PAIRS}개입니다 (현재 선택 {len(choice)}개"
                        " — 일부 정지 후 추가하세요).",
                    )
        # 새 슬롯 — settings + client + bot 생성. settings.symbol 은
        # _build_user_settings 안에서 인자 symbol 로 강제 박힘.
        settings = self._build_user_settings(
            user_code, symbol, force_run_mode=force_run_mode,
        )
        client = await self.client_factory(settings)
        # 거래소 호출 실패 로그(retCode 10003 키 무효 등)에 어느 사용자/심볼인지
        # 식별되도록 라벨 주입 — 멀티유저 운영에서 문제 키 특정용.
        _set_label = getattr(client, "set_log_label", None)
        if callable(_set_label):
            _set_label(f"{user_code}/{symbol}")
        # 페어 확장 — 거래 가능 화이트리스트(거래대금 상위 N + 메이저) 검증.
        # 조회 실패는 통과 처리(거래소 일시 장애가 메이저 가동까지 막지 않게).
        try:
            allowed = await self._pair_registry.is_allowed(client, symbol)
        except Exception as e:  # noqa: BLE001
            logger.warning("화이트리스트 조회 실패 (%s) — 통과 처리: %s", symbol, e)
            allowed = True
        if not allowed:
            await self._safe_close_client(client)
            raise ValueError(
                f"'{symbol}' 는 거래 가능 목록(거래대금 상위 "
                f"{self._pair_registry.limit})에 없습니다.",
            )
        bot = BotIctInstance(
            client=client,
            symbol=settings.symbol,
            timeframe=settings.timeframe,
            leverage=settings.leverage,
            position_pct_base=settings.position_pct_base,
            position_pct_max=settings.position_pct_max,
            position_pct_step=settings.position_pct_step,
            risk_based_sizing=settings.risk_based_sizing,
            risk_per_trade_base=settings.risk_per_trade_base,
            risk_per_trade_step=settings.risk_per_trade_step,
            risk_per_trade_max=settings.risk_per_trade_max,
            min_rr=settings.min_rr,
            min_confluence=settings.min_confluence,
            # #TRAIL-EXCHANGE (Origo 1.4): 거래소 트레일링 — 구독제 validator 가
            # 2.0/1.5 강제, referral 은 0(off, 고정 TP 모드).
            trail_trigger_r=settings.origo_trail_trigger_r,
            trail_dist_r=settings.origo_trail_dist_r,
            high_rr_bypass_min_rr=settings.high_rr_bypass_min_rr,
            fvg_min_size_pct=settings.fvg_min_size_pct,
            smt_enabled=settings.smt_enabled,
            step_interval_sec=settings.step_interval_sec,
            ohlcv_limit=settings.ohlcv_limit,
            setup_stale_bars=settings.setup_stale_bars,
            disable_time_filter=settings.disable_time_filter,
            multi_tf=settings.multi_tf,
            multi_tf_ltf_lookback=settings.multi_tf_ltf_lookback,
            enable_trail=settings.enable_trail,
            trail_buffer_ratio=settings.trail_buffer_ratio,
            use_market_entry=settings.use_market_entry,
            # #ORIGO-1: 페어별 베스트 ttl (BTC 1h, 나머지 settings 기본 30분).
            entry_limit_ttl_sec=origo1_ttl_for_symbol(
                settings.symbol, settings.entry_limit_ttl_sec,
            ),
            min_sl_distance_pct=settings.min_sl_distance_pct,
            max_sl_distance_pct=settings.max_sl_distance_pct,
            sl_dist_mult=settings.sl_dist_mult,
            sl_dist_mult_ct=settings.sl_dist_mult_ct,
            ct_trend_threshold=settings.ct_trend_threshold,
            shadow_log_enabled=settings.shadow_log_enabled,
            max_entry_distance_pct=settings.max_entry_distance_pct,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
            htf_ltf_conflict_guard_ratio=settings.htf_ltf_conflict_guard_ratio,
            htf_ema_bias_enabled=settings.htf_ema_bias_enabled,
            htf_ema_bias_tf=settings.htf_ema_bias_tf,
            htf_ema_bias_period=settings.htf_ema_bias_period,
            htf_ema_align_enabled=settings.htf_ema_align_enabled,
            htf_ema_align_periods=settings.htf_ema_align_periods,
            htf_ema_align_threshold=settings.htf_ema_align_threshold,
            htf_override_mode=settings.htf_override_mode,
            htf_fvg_tfs=settings.htf_fvg_tfs,
            daily_loss_limit_pct=settings.daily_loss_limit_pct,
            daily_profit_limit_pct=settings.daily_profit_limit_pct,
            daily_pair_loss_limit_r=settings.daily_pair_loss_limit_r,
            # WS flip watcher — 테스트/리소스 절약 위해 multi-user 에선 기본 끔.
            # 운영 환경에서 사용자별 WS 연결 N개는 부담 — 추후 공유 stream 으로 개선.
            flip_watch_enabled=False,
            # 2026-05-29 매매 로그 격리 — 사용자별 trades.jsonl/db 디렉토리 주입.
            # _user_data_dir(code) = <data_dir>/users/<code>/. TradesStore lazy init
            # 시 이 디렉토리에 trades.jsonl / trades.db / trade_journal.log 생성.
            trades_data_dir=self._user_data_dir(user_code),
            # 2026-05-29 PR 매매기록 DEMO/LIVE 구분 (파트너 요청): 봇이 매매 시
            # 이 run_mode 가 TradeEvent.mode 에 박힘 → UI 표에 표시.
            run_mode=settings.run_mode.value,
            # 2026-06-08: 텔레그램 매매 알림 — 소유자 코드 + 알림 콜백 주입.
            # alerter 미설정(None)이면 콜백도 None → 알림 안 보냄.
            user_code=user_code,
            # 2026-06-25: Partial TP 거래소 등록 시범 — 화이트리스트 유저만 ON(소액 실측).
            partial_tp_exchange=user_code in _PARTIAL_TP_USERS,
            alert_cb=(
                self.alerter.send_trade_alert if self.alerter is not None else None
            ),
            # 2026-06-13: 약관 미동의 등 1회성 사용자 안내 (일반 텍스트).
            notify_cb=(
                self.alerter.send_user_text if self.alerter is not None else None
            ),
        )
        # 2026-06-25 #CURSUS: 사용자 선택 모델이 Cursus 면 추세형 봇으로 교체.
        # BotIctInstance 는 dataclass 생성(거래소 호출 없음 — start() 에서 호출)이라
        # 교체 비용 무시 가능. 모델 미선택(NULL)은 DEFAULT_MODEL_NAME(Origo) fallback.
        from aurora_ict.config.settings import AVAILABLE_MODELS, DEFAULT_MODEL_NAME
        _model = users_db.get_last_model(self.db_path, user_code) or DEFAULT_MODEL_NAME
        if AVAILABLE_MODELS.get(_model) == "cursus":
            from aurora_ict.bot.bot_trend_instance import BotTrendInstance
            from aurora_ict.strategy.dual_st import DualSTConfig
            bot = BotTrendInstance(
                client=client,
                symbol=settings.symbol,
                # 2026-06-26 파트너: Cursus 레버 페어별 — BTC 10x, 알트(ETH 포함) 7x.
                # base 자산 정확매칭(startswith 는 BTCDOM 등 BTC-prefix 토큰 오인).
                leverage=(10 if settings.symbol.split("/")[0] == "BTC" else 7),
                size_pct=0.9,
                cfg=DualSTConfig(trail_mult=6.0),
                # 일일 손실 한도 — Origo 와 동일하게 주입(시드 방어). 0 이면 비활성.
                daily_loss_limit_pct=settings.daily_loss_limit_pct,
                trades_data_dir=self._user_data_dir(user_code),
                user_code=user_code,
                run_mode=settings.run_mode.value,
                alert_cb=(
                    self.alerter.send_trade_alert if self.alerter is not None else None
                ),
                notify_cb=(
                    self.alerter.send_user_text if self.alerter is not None else None
                ),
            )
            logger.info("Cursus(추세형) 봇 생성 — %s %s (trail x6.0)", user_code, symbol)
        self._slots[key] = _UserBotSlot(
            symbol=symbol, settings=settings, bot=bot, client=client,
        )
        return bot

    async def _safe_close_client(self, client: object) -> None:
        """client 정리 — close 메서드 있으면 호출(슬롯 미생성 시 세션 누수 방지)."""
        close = getattr(client, "close", None)
        if close is None:
            return
        try:
            result = close()
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:  # noqa: BLE001
            logger.debug("client close 실패(무시): %s", e)

    async def list_tradable_pairs(self, user_code: str | None = None) -> list[str]:
        """거래 가능 페어 목록 — UI 페어 선택용 (거래대금 상위 N + 메이저).

        활성 슬롯 client 가 있으면 그걸로 거래소 목록을 갱신(TTL 캐시)하고, 없으면
        메이저(BTC/ETH)만 반환한다. 신규 사용자는 첫 봇 가동 후부터 전체 목록을
        받는다(슬롯 client 확보 시점).
        """
        src = None
        for (u, _s), slot in self._slots.items():
            if slot.client is None:
                continue
            src = slot.client
            if user_code is not None and u == user_code:
                break  # 해당 사용자 client 우선
        if src is None:
            return list(MAJOR_PAIRS)
        try:
            return await self._pair_registry.get_allowed(src)
        except Exception as e:  # noqa: BLE001
            logger.warning("거래 가능 페어 조회 실패: %s", e)
            return list(MAJOR_PAIRS)

    async def list_market_tickers(
        self, user_code: str | None = None, *, now: float | None = None,
    ) -> list[dict[str, Any]]:
        """UI 페어 선택기용 시세 행 — 거래대금 상위 N(symbol/last/pct24h/volume).

        활성 슬롯 client 로 거래소 시세 조회(60초 TTL 캐시). 조회 실패 시 기존
        캐시 유지, 슬롯 client 미확보 시 빈 리스트.
        """
        t = time.monotonic() if now is None else now
        if self._market_rows_cache and (t - self._market_rows_at) < 60.0:
            return self._market_rows_cache
        src = None
        for (u, _s), slot in self._slots.items():
            if slot.client is None:
                continue
            src = slot.client
            if user_code is not None and u == user_code:
                break
        if src is None:
            return self._market_rows_cache
        try:
            rows = await src.fetch_perp_tickers(self._pair_registry.limit)
        except Exception as e:  # noqa: BLE001
            logger.warning("시세 행 조회 실패: %s", e)
            return self._market_rows_cache
        if rows:
            # 2026-06-12 파트너/조윤 보고: 고정 7이 거래대금 상위 N 밖이면
            # (예: LINK 30위 밖) 피커에서 검색조차 안 됨 — 누락 고정 페어의
            # 시세를 개별 조회해 행을 보장한다 (조회 실패 시 가격 미상 행).
            have = {str(r.get("symbol")) for r in rows}
            for fixed in FIXED_PAIRS:
                if fixed in have:
                    continue
                last = None
                try:
                    fetch_ticker = getattr(src, "fetch_ticker", None)
                    if callable(fetch_ticker):
                        last = await fetch_ticker(fixed)
                except Exception as e:  # noqa: BLE001
                    logger.debug("고정 페어 시세 보강 실패 %s: %s", fixed, e)
                rows.append({
                    "symbol": fixed, "last": last,
                    "pct24h": None, "volume": None,
                })
            self._market_rows_cache = rows
            self._market_rows_at = t
        return self._market_rows_cache

    async def ensure_bot_ready(
        self, user_code: str, symbol: str = _DEFAULT_SYMBOL,
    ) -> BotIctInstance:
        """차트 lazy 로딩용 — 봇 인스턴스 생성 + prefetch 시작 (가동 X).

        2026-05-28: SaaS UX — 사용자가 START 누르기 전, 로그인 직후부터
        차트 봉/마커가 보이도록 OHLCV cache prefetch 만 미리 시작.
        ``start()`` 와 달리 거래소 leverage 설정 / 매매 loop 가동은 하지 않음 —
        봇 state 는 STOPPED 유지.
        2026-05-29 PR B: ``symbol`` 인자 — UI 가 BTC/ETH 탭 전환 시 각각 prefetch.

        Args:
            user_code: 대상 사용자 라이선스 코드.
            symbol: ccxt 형식 페어 (기본 BTC, backward compat).

        Returns:
            BotIctInstance (state=STOPPED 일 수 있음, prefetch task 진행 중).

        Raises:
            ValueError: 사용자 미존재 또는 API key 미등록 (호출자가 401/404 응답).
        """
        async with self._get_lock(user_code, symbol):
            bot = await self.get_or_create_bot(user_code, symbol)
            bot.ensure_prefetch_started()
            return bot

    async def start(
        self,
        user_code: str,
        symbol: str = _DEFAULT_SYMBOL,
        *,
        force_run_mode: RunMode | None = None,
    ) -> None:
        """사용자 × symbol 봇 기동 — get_or_create 후 BotIctInstance.start 호출.

        같은 (user, symbol) 동시 호출 race 방지를 위해 per-(user, symbol)
        lock 사용. 같은 사용자가 BTC + ETH 동시 가동 시도해도 서로 막지 않음.

        2026-05-28: 봇 가동 상태 영속화 — 성공 시 ``users_db.set_bot_running``
        으로 DB 의 ``bot_running_symbols`` JSON 배열에 symbol 추가.
        2026-05-29 hot-fix: ``force_run_mode`` 인자 — auto_resume 가 사용자
        마지막 가동 모드로 정확하게 복원할 수 있게.
        2026-05-29 PR B: ``symbol`` 인자 — 멀티 페어 가동.

        Args:
            user_code: 대상 사용자 라이선스 코드.
            symbol: ccxt 형식 페어 (기본 BTC).
            force_run_mode: 명시 시 IctSettings.run_mode override.

        Raises:
            ValueError: 사용자 미존재 / API 키 미등록 등 (build_user_settings 단계).
        """
        async with self._get_lock(user_code, symbol):
            bot = await self.get_or_create_bot(
                user_code, symbol, force_run_mode=force_run_mode,
            )
            if bot.state is BotState.RUNNING:
                logger.info(
                    "사용자 %s (symbol=%s) 봇 이미 실행 중 — re-start 무시",
                    user_code, symbol,
                )
                # 이미 가동 중이어도 DB 플래그는 보장 (운영 중 마이그레이션 직후 안전망).
                try:
                    users_db.set_bot_running(
                        self.db_path, user_code, True, symbol=symbol,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "사용자 %s/%s bot_running 영속화 실패 (무시): %s",
                        user_code, symbol, e,
                    )
                return
            slot = self._slots[(user_code, symbol)]
            # 2026-06-12 #ONEWAY: 헤지 모드 계정이면 모든 주문이 10001 거부 —
            # 봇 시작 시 해당 심볼을 원웨이로 best-effort 전환 (실패해도 진행).
            # getattr 가드 — 어댑터 외 클라이언트(테스트 Fake 등)는 skip.
            oneway_fn = getattr(slot.client, "ensure_oneway_mode", None)
            if oneway_fn is not None:
                try:
                    await oneway_fn(slot.settings.symbol)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "사용자 %s/%s 원웨이 전환 실패(무시): %s",
                        user_code, symbol, e,
                    )
            # 거래소 측 leverage 셋업 — 실패해도 봇 자체는 진행. 다만 실패 시
            # 실제 leverage 조회해서 bot.leverage 보정 (#LEV-1) — size 계산이
            # 거래소 값과 일치하게.
            # set_leverage 반환값으로 성공 판정 — aurora_adapter 가 실패 시 빈 dict
            # 또는 {"retCode": 110043} (이미 같은 값) 반환 (exception 안 던짐).
            leverage_synced = False
            try:
                result = await slot.client.set_leverage(  # type: ignore[union-attr]
                    slot.settings.symbol, slot.settings.leverage,
                )
                if isinstance(result, dict) and (
                    result.get("alreadySet") is True
                    or int(result.get("retCode", -1)) == 0
                ):
                    leverage_synced = True
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "사용자 %s/%s set_leverage 실패: %s", user_code, symbol, e,
                )
            if not leverage_synced:
                try:
                    actual = await slot.client.fetch_actual_leverage(  # type: ignore[union-attr]
                        slot.settings.symbol,
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "사용자 %s/%s fetch_actual_leverage 실패: %s",
                        user_code, symbol, e,
                    )
                    actual = None
                if actual is not None and actual != bot.leverage:
                    logger.warning(
                        "사용자 %s/%s leverage mismatch — 의도 %dx, 거래소 %dx. "
                        "size 계산 보정 (수동 변경 권장).",
                        user_code, symbol, bot.leverage, actual,
                    )
                    bot.leverage = actual
            await bot.start()
            # 봇 실제 가동 성공 직후에만 DB 영속화 — start() 가 예외 던지면 이 줄 도달 X
            # → DB 는 stale 한 0 유지 (보수적 안전).
            try:
                users_db.set_bot_running(
                    self.db_path, user_code, True, symbol=symbol,
                )
                # 2026-06-06: 선호 페어 집합에도 추가 — 전체 STOP→START 시 복원용.
                users_db.set_last_active_pair(
                    self.db_path, user_code, symbol, True,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "사용자 %s/%s bot_running 영속화 실패 (무시): %s",
                    user_code, symbol, e,
                )
            # 2026-05-29 hot-fix: 마지막 가동 run_mode 도 영속화 — fly machine
            # 재시작 후 auto_resume 가 base_settings (DEMO) 대신 이 모드로 가동.
            # LIVE 사용자가 재시작마다 DEMO 키 미등록 실패하던 버그 해소.
            # 같은 사용자 BTC/ETH 어느 페어로 가동하든 run_mode 는 사용자별 1개라
            # 마지막으로 가동한 모드로 덮어쓰는 게 맞다.
            try:
                users_db.set_last_run_mode(
                    self.db_path, user_code, slot.settings.run_mode.value,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "사용자 %s last_run_mode 영속화 실패 (무시): %s",
                    user_code, e,
                )
            logger.info(
                "MultiUserBotManager: 사용자 %s 봇 시작 (symbol=%s)",
                user_code, slot.settings.symbol,
            )

    async def stop(
        self, user_code: str, symbol: str = _DEFAULT_SYMBOL,
        *, forget_preference: bool = True,
    ) -> None:
        """사용자 × symbol 봇 정지 + client close.

        존재하지 않는 (user_code, symbol) 도 멱등 (no-op).

        2026-05-28: 봇 가동 상태 영속화 — ``users_db.set_bot_running`` 으로 DB 의
        ``bot_running_symbols`` 에서 symbol 제거. 다음 부팅 때 자동 재가동 대상
        에서 빠짐.
        2026-05-29 PR B: ``symbol`` 인자 — 사용자가 BTC 만 정지하고 ETH 는 계속
        가동 같은 부분 정지 가능.

        Args:
            user_code: 대상 사용자 라이선스 코드.
            symbol: ccxt 형식 페어 (기본 BTC).
            forget_preference: True(기본)면 선호 페어 집합에서도 제거(페어 칩으로
                끈 경우). 전체 STOP 은 False 로 호출해 선호를 유지 → START 시 복원.
        """
        key: SlotKey = (user_code, symbol)
        # 2026-06-11 리뷰 수정: start/ensure_bot_ready 와 같은 lock 으로 직렬화.
        # 무lock 이던 기존엔 START 진행 중 STOP 이 끼어들어 슬롯 pop → start 의
        # _slots[key] 접근 KeyError, 또는 DB bot_running 쓰기 순서가 꼬여
        # 사용자가 끈 봇을 auto_resume 이 되살리는 레이스가 가능했다.
        async with self._get_lock(user_code, symbol):
            slot = self._slots.get(key)
            # slot 유무와 무관하게 사용자가 명시 STOP 을 눌렀다면 DB 에서 해당 symbol
            # 제거 — 다른 fly machine 에서 가동했더라도 명시 정지 의도가 우선. DB 에
            # user row 자체가 없으면 set_bot_running 이 False 반환 (no-op).
            try:
                users_db.set_bot_running(
                    self.db_path, user_code, False, symbol=symbol,
                )
                # 2026-06-06: 페어 칩으로 끈 경우만 선호에서 제거. 전체 STOP 은 유지.
                if forget_preference:
                    users_db.set_last_active_pair(
                        self.db_path, user_code, symbol, False,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "사용자 %s/%s bot_running 영속화 실패 (무시): %s",
                    user_code, symbol, e,
                )
            if slot is None:
                return
            if slot.bot is not None:
                await slot.bot.stop()
            await self._close_client(slot)
            # 2026-06-11 버그픽스: 정지한 슬롯을 _slots 에서 제거.
            # 안 그러면 페어 카운트(start 의 existing = 사용자 슬롯 수)에 정지된
            # 슬롯이 남아 "5개 등록 → 1개 삭제 → 추가" 가 MAX_PAIRS_PER_USER 로 막힌다.
            self._slots.pop(key, None)
            logger.info(
                "MultiUserBotManager: 사용자 %s 봇 정지 (symbol=%s)",
                user_code, symbol,
            )

    async def start_preferred(
        self, user_code: str, *, force_run_mode: RunMode | None = None,
    ) -> list[str]:
        """고정 7 + 선호 페어(last_active_pairs)를 모두 가동 — 전체 START 용.

        2026-06-06: 봇 STOP→START 시 BTC 만 켜지던 문제 해소. 사용자가 마지막에
        가동했던 페어들을 그대로 복원한다.
        2026-06-12 고정7 구도: 전체 START 는 고정 페어(FIXED_PAIRS) 전부 +
        사용자가 마지막에 켰던 선택 페어(고정 외, 최대 MAX_CHOICE_PAIRS)로
        구성된다. 검증 탈락 페어(EXCLUDED_PAIRS)는 복원하지 않는다.

        Returns:
            가동된 symbol 목록.
        """
        last = users_db.get_last_active_pairs(self.db_path, user_code)
        choice = [
            s for s in last
            if s not in FIXED_PAIRS and s not in EXCLUDED_PAIRS
        ][:MAX_CHOICE_PAIRS]
        pairs = list(FIXED_PAIRS) + choice
        started: list[str] = []
        for sym in pairs:
            try:
                await self.start(user_code, sym, force_run_mode=force_run_mode)
                started.append(sym)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "start_preferred 실패 %s/%s: %s", user_code, sym, e,
                )
        return started

    async def stop_user(self, user_code: str) -> list[str]:
        """사용자의 모든 가동 페어 정지 — 전체 STOP. 선호(last_active)는 유지.

        2026-06-06: bot_running_symbols + 현재 슬롯을 합쳐 모든 페어 정지하되,
        선호 페어 집합은 안 건드려 START 시 복원되게 한다(forget_preference=False).

        Returns:
            정지된 symbol 목록.
        """
        syms = set(users_db.get_bot_running_symbols(self.db_path, user_code))
        syms |= {s for (u, s) in self._slots if u == user_code}
        stopped: list[str] = []
        for sym in syms:
            try:
                await self.stop(user_code, sym, forget_preference=False)
                stopped.append(sym)
            except Exception as e:  # noqa: BLE001
                logger.warning("stop_user 실패 %s/%s: %s", user_code, sym, e)
        return stopped

    async def _close_client(self, slot: _UserBotSlot) -> None:
        """ccxt async exchange 의 aiohttp ClientSession 명시 close.

        BotManager._close_client 와 동일 패턴 — 누수 방지.

        #LEV-7: slot.bot 도 None 박음. 이걸 안 박으면 다음 start 시
        get_or_create_bot 의 ``slot.bot is not None`` 조건에 걸려 기존 bot
        그대로 반환 → 새 client 생성 안 됨 → slot.client 영원히 None →
        모든 호출 'NoneType' AttributeError.
        """
        if slot.client is None:
            slot.bot = None
            return
        client = slot.client
        slot.client = None
        slot.bot = None
        inner = getattr(client, "_client", client)
        ex = getattr(inner, "_ex", None)
        if ex is None:
            return
        try:
            await ex.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("ccxt exchange close 실패 (무시): %s", e)

    async def status(
        self, user_code: str, symbol: str = _DEFAULT_SYMBOL,
    ) -> dict[str, Any]:
        """사용자 × symbol 봇 상태 스냅샷 (없으면 stopped + has_credentials).

        2026-05-28: UI Trade Timeframe 토글 active 표시 위해 ``timeframe`` 필드 추가.
        2026-05-29 PR B: ``symbol`` 인자 + ``running_symbols`` 필드 — UI 가 어느
        페어가 가동 중인지 표시할 수 있게.

        slot 없을 때는 base_settings.timeframe (기본 trade TF) 을 응답해
        ui_ict/app.js 의 ``b.dataset.tradeTf === s.timeframe`` 토글 매칭 가능하게.

        Args:
            user_code: 대상 사용자 라이선스 코드.
            symbol: 조회할 ccxt 형식 페어 (기본 BTC).

        Returns:
            상태 스냅샷 dict — UI 가 그대로 표시.
        """
        key: SlotKey = (user_code, symbol)
        slot = self._slots.get(key)
        # demo/live 키 등록 여부 — 슬롯 유무와 무관하게 DB 진실값 (2026-05-29).
        has_demo = users_db.has_api_keys(self.db_path, user_code, "demo")
        has_live = users_db.has_api_keys(self.db_path, user_code, "live")
        # 사용자 가동 중 symbol list — UI 가 BTC/ETH 탭별 가동 dot 표시에 사용.
        try:
            running_symbols = users_db.get_bot_running_symbols(
                self.db_path, user_code,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "사용자 %s running_symbols 조회 실패 (무시): %s", user_code, e,
            )
            running_symbols = []
        # 사용자 마지막 가동 run_mode — 슬롯 없어도 DB 에서 가져와 UI 모드 표시.
        # 이게 없으면 새로고침 후 base_settings.run_mode (DEMO) 가 잘못 표시되어
        # LIVE 사용자가 "데모로 넘어가고 STOP 에 불 켜짐" 으로 보이는 버그 (2026-05-29).
        try:
            last_mode = users_db.get_last_run_mode(self.db_path, user_code)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "사용자 %s last_run_mode 조회 실패 (무시): %s", user_code, e,
            )
            last_mode = None
        if slot is None or slot.bot is None:
            base = self.base_settings if self.base_settings is not None else IctSettings()
            # last_run_mode 우선 — 사용자가 LIVE 로 가동했었으면 새로고침 후에도 LIVE.
            run_mode = last_mode if last_mode in ("demo", "live") else base.run_mode.value
            # 현재 모드 기준 등록 여부 — UI 가 "키 등록" CTA 결정에 사용.
            has_creds = has_live if run_mode == "live" else has_demo
            # 2026-05-29: DB 에 가동 중 symbol 이 있는데 슬롯이 없으면 = fly machine
            # 재시작 직후 auto_resume 가 아직 못 살린 상태. 이 때 state="stopped" 로
            # 응답하면 UI 가 STOP 버튼을 active 로 표시해 "정지됨" 으로 오해. 대신
            # "resuming" 상태로 응답해 UI 가 복원 진행 중임을 표시할 수 있게 한다.
            # 파트너 보고 2026-05-29: "또 데모로 넘어가고 STOP 에 불들어와" 의 원인.
            this_symbol_running = symbol in running_symbols
            state_val = (
                "resuming" if this_symbol_running else BotState.STOPPED.value
            )
            return {
                "state": state_val,
                "run_mode": run_mode,
                "enabled": this_symbol_running,
                "symbol": symbol,
                "timeframe": base.timeframe,
                "has_credentials": has_creds,
                "has_demo_credentials": has_demo,
                "has_live_credentials": has_live,
                "has_active_position": False,
                "last_setup_ts_ms": 0,
                "running_symbols": running_symbols,
            }
        bot = slot.bot
        run_mode = slot.settings.run_mode.value
        return {
            "state": bot.state.value,
            "run_mode": run_mode,
            "enabled": slot.settings.enabled,
            "symbol": bot.symbol,
            "timeframe": slot.settings.timeframe,
            "has_credentials": slot.settings.has_credentials(),
            "has_demo_credentials": has_demo,
            "has_live_credentials": has_live,
            "has_active_position": bot.active_position is not None,
            "last_setup_ts_ms": bot._last_setup_ts_ms,
            "running_symbols": running_symbols,
        }

    async def get_seed_usdt(self, user_code: str) -> float | None:
        """사용자의 현재 USDT 지갑 잔고(시드) 조회 — 매매기록 화면 표시용.

        가동 중이면 해당 슬롯의 client 를 재사용(추가 연결 없음), 비가동이면
        LIVE 키로 임시 client 를 만들어 1회 조회 후 즉시 닫는다. 키 미등록·무효
        (retCode 10003)·조회 실패 시 None (UI 는 '—' 로 표시).

        Args:
            user_code: 대상 사용자 라이선스 코드.

        Returns:
            USDT total 잔고(float) 또는 None.
        """
        # 1) 가동 중 슬롯 client 재사용 — 기존 세션이라 부하 최소.
        for (u, _sym), slot in self._slots.items():
            if u == user_code and slot.client is not None:
                return await self._usdt_total(slot.client)
        # 2) 비가동 — LIVE 키가 있으면 임시 client 로 1회 조회 후 닫는다.
        if users_db.get_api_keys(self.db_path, user_code, "live") is None:
            return None
        try:
            settings = self._build_user_settings(
                user_code, _DEFAULT_SYMBOL, force_run_mode=RunMode.LIVE,
            )
            client = await self.client_factory(settings)
        except Exception:  # noqa: BLE001
            return None
        try:
            return await self._usdt_total(client)
        finally:
            await self._safe_close_client(client)

    @staticmethod
    async def _usdt_total(client: object) -> float | None:
        """client.fetch_balance() 에서 USDT total 추출. 실패/부재 시 None."""
        fetch = getattr(client, "fetch_balance", None)
        if not callable(fetch):
            return None
        try:
            bal = await fetch()
        except Exception:  # noqa: BLE001
            return None
        usdt = (bal or {}).get("USDT") or {}
        total = usdt.get("total")
        try:
            return float(total) if total is not None else None
        except (TypeError, ValueError):
            return None

    async def stop_all(self) -> None:
        """모든 사용자 × 모든 symbol 봇 정지 — 서버 종료 시 호출.

        예외 발생해도 다음 슬롯 정지 시도 (best-effort).
        2026-05-29 PR B: (user_code, symbol) 튜플 키 순회.
        """
        keys = list(self._slots.keys())
        for code, sym in keys:
            try:
                await self.stop(code, sym)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "stop_all: 사용자 %s/%s 정지 실패 — %s", code, sym, e,
                )
        logger.info(
            "MultiUserBotManager: 전체 슬롯 봇 정지 완료 (%d 슬롯)", len(keys),
        )

    def list_users(self) -> list[str]:
        """현재 슬롯에 등록된 unique 사용자 코드 목록 (디버그/관리용).

        2026-05-29 PR B: 한 사용자가 여러 symbol 슬롯을 가질 수 있어 set 으로
        중복 제거. saas.py shutdown hook 이 이걸로 사용자별 정지 순회.
        """
        return sorted({k[0] for k in self._slots.keys()})

    def list_slots(self) -> list[SlotKey]:
        """현재 (user_code, symbol) 슬롯 키 전체 목록 — 멀티 페어 디버그/관리용."""
        return list(self._slots.keys())


__all__ = [
    "ClientFactory",
    "MultiUserBotManager",
]
