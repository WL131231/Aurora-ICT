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
from aurora_ict.config.settings import IctSettings, RunMode

logger = logging.getLogger(__name__)

# client factory 시그니처 — settings 받아 ExchangeClient 반환 (BotManager 와 동일).
ClientFactory = Callable[[IctSettings], Awaitable[ExchangeClientProtocol]]


@dataclass(slots=True)
class _UserBotSlot:
    """사용자 1명의 봇 슬롯 — bot/client/settings 함께 보관."""

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
    _slots: dict[str, _UserBotSlot] = field(default_factory=dict)
    # 사용자별 start lock — 같은 사용자 동시 start race 방지.
    # 다른 사용자끼리는 병렬 가능 (격리 보장).
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def _get_lock(self, user_code: str) -> asyncio.Lock:
        """사용자별 asyncio.Lock — 없으면 lazy create.

        Why per-user: 전역 lock 박으면 사용자 A start 가 B start 를 막아 SaaS
        병렬성 무너짐. 사용자별 lock 으로 같은 user 더블 클릭만 차단.
        """
        lock = self._locks.get(user_code)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[user_code] = lock
        return lock

    def _user_data_dir(self, user_code: str) -> Path:
        """사용자별 데이터 디렉토리 — ``<data_dir>/users/<code>/``.

        Why: 사용자별 license.json / trades.db / trades_dataset 등 데이터 분리.
        한 사용자 데이터가 다른 사용자에게 노출/오염되지 않도록 격리.
        """
        from aurora_ict.paths import data_dir
        path = data_dir() / "users" / user_code
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _build_user_settings(self, user_code: str) -> IctSettings:
        """DB row → 사용자별 IctSettings (api_key/secret 복호화 포함).

        Args:
            user_code: 대상 사용자 라이선스 코드.

        Returns:
            base_settings 를 깊은 복사한 IctSettings 에 사용자별 api_key/secret 박은 것.

        Raises:
            ValueError: 사용자 미존재 또는 api_key/secret 미등록 (start 전 필수).
        """
        user = users_db.get_user_by_code(self.db_path, user_code)
        if user is None:
            raise ValueError(f"사용자 '{user_code}' 가 DB 에 없습니다.")
        api_key = user.get("api_key")
        secret_enc = user.get("api_secret_enc")
        if not api_key or not secret_enc:
            raise ValueError(
                f"사용자 '{user_code}' 의 거래소 API 키가 등록되지 않았습니다.",
            )
        # secret 복호화 — keystore master key 사용. 키 불일치 시 InvalidToken.
        api_secret = keystore.decrypt_secret(secret_enc, key=self.master_key)

        # base_settings 복사 — pydantic v2 model_copy 깊은 복사.
        base = self.base_settings if self.base_settings is not None else IctSettings()
        settings = base.model_copy(deep=True)
        # 사용자 라이선스 타입 적용 (referral / sub_*) — _enforce_license_tier_policy
        # 가 자동 호출되도록 다시 IctSettings 인스턴스 생성.
        license_type = user.get("license_type", "referral")
        # 현 run_mode 기준 양쪽 키 슬롯에 박음 (mode 전환 시 동일 키 재사용 가정).
        # SaaS 사용자는 하나의 거래소 키만 다루는 정책.
        if base.run_mode is RunMode.LIVE:
            settings.live_api_key = SecretStr(api_key)
            settings.live_api_secret = SecretStr(api_secret)
        else:
            settings.demo_api_key = SecretStr(api_key)
            settings.demo_api_secret = SecretStr(api_secret)
        settings.license_type = license_type
        # 모델 validator 재실행 — license_type 변경에 따른 disable_time_filter 강제.
        settings = settings.model_validate(settings.model_dump())
        return settings

    async def get_or_create_bot(self, user_code: str) -> BotIctInstance:
        """사용자 봇 인스턴스 가져오기 — 없으면 생성 (start 는 별도 호출).

        Args:
            user_code: 대상 사용자 라이선스 코드.

        Returns:
            BotIctInstance (생성만, state=STOPPED 일 수 있음).
        """
        slot = self._slots.get(user_code)
        if slot is not None and slot.bot is not None:
            return slot.bot
        # 새 슬롯 — settings + client + bot 생성.
        settings = self._build_user_settings(user_code)
        client = await self.client_factory(settings)
        bot = BotIctInstance(
            client=client,
            symbol=settings.symbol,
            timeframe=settings.timeframe,
            leverage=settings.leverage,
            position_pct_base=settings.position_pct_base,
            position_pct_max=settings.position_pct_max,
            position_pct_step=settings.position_pct_step,
            min_rr=settings.min_rr,
            min_confluence=settings.min_confluence,
            fvg_min_size_pct=settings.fvg_min_size_pct,
            step_interval_sec=settings.step_interval_sec,
            ohlcv_limit=settings.ohlcv_limit,
            setup_stale_bars=settings.setup_stale_bars,
            disable_time_filter=settings.disable_time_filter,
            multi_tf=settings.multi_tf,
            multi_tf_ltf_lookback=settings.multi_tf_ltf_lookback,
            enable_trail=settings.enable_trail,
            trail_buffer_ratio=settings.trail_buffer_ratio,
            use_market_entry=settings.use_market_entry,
            entry_limit_ttl_sec=settings.entry_limit_ttl_sec,
            min_sl_distance_pct=settings.min_sl_distance_pct,
            max_sl_distance_pct=settings.max_sl_distance_pct,
            heartbeat_interval_sec=settings.heartbeat_interval_sec,
            htf_ema_bias_enabled=settings.htf_ema_bias_enabled,
            htf_ema_bias_tf=settings.htf_ema_bias_tf,
            htf_ema_bias_period=settings.htf_ema_bias_period,
            htf_override_mode=settings.htf_override_mode,
            htf_fvg_tfs=settings.htf_fvg_tfs,
            daily_loss_limit_pct=settings.daily_loss_limit_pct,
            # WS flip watcher — 테스트/리소스 절약 위해 multi-user 에선 기본 끔.
            # 운영 환경에서 사용자별 WS 연결 N개는 부담 — 추후 공유 stream 으로 개선.
            flip_watch_enabled=False,
        )
        self._slots[user_code] = _UserBotSlot(
            settings=settings, bot=bot, client=client,
        )
        return bot

    async def start(self, user_code: str) -> None:
        """사용자 봇 기동 — get_or_create 후 BotIctInstance.start 호출.

        같은 사용자 동시 호출 race 방지를 위해 per-user lock 사용.

        Raises:
            ValueError: 사용자 미존재 / API 키 미등록 등 (build_user_settings 단계).
        """
        async with self._get_lock(user_code):
            bot = await self.get_or_create_bot(user_code)
            if bot.state is BotState.RUNNING:
                logger.info("사용자 %s 봇 이미 실행 중 — re-start 무시", user_code)
                return
            slot = self._slots[user_code]
            # 거래소 측 leverage 셋업 — 실패해도 봇 자체는 진행 (warning).
            try:
                await slot.client.set_leverage(  # type: ignore[union-attr]
                    slot.settings.symbol, slot.settings.leverage,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("사용자 %s set_leverage 실패: %s", user_code, e)
            await bot.start()
            logger.info(
                "MultiUserBotManager: 사용자 %s 봇 시작 (symbol=%s)",
                user_code, slot.settings.symbol,
            )

    async def stop(self, user_code: str) -> None:
        """사용자 봇 정지 + client close.

        존재하지 않는 user_code 도 멱등 (no-op).
        """
        slot = self._slots.get(user_code)
        if slot is None:
            return
        if slot.bot is not None:
            await slot.bot.stop()
        await self._close_client(slot)
        logger.info("MultiUserBotManager: 사용자 %s 봇 정지", user_code)

    async def _close_client(self, slot: _UserBotSlot) -> None:
        """ccxt async exchange 의 aiohttp ClientSession 명시 close.

        BotManager._close_client 와 동일 패턴 — 누수 방지.
        """
        if slot.client is None:
            return
        client = slot.client
        slot.client = None
        inner = getattr(client, "_client", client)
        ex = getattr(inner, "_ex", None)
        if ex is None:
            return
        try:
            await ex.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("ccxt exchange close 실패 (무시): %s", e)

    async def status(self, user_code: str) -> dict[str, Any]:
        """사용자 봇 상태 스냅샷 (없으면 stopped + has_credentials 만)."""
        slot = self._slots.get(user_code)
        if slot is None or slot.bot is None:
            # 슬롯 없음 — DB 만 봐서 credentials 여부 보고.
            user = users_db.get_user_by_code(self.db_path, user_code)
            has_creds = bool(
                user and user.get("api_key") and user.get("api_secret_enc"),
            )
            return {
                "state": BotState.STOPPED.value,
                "run_mode": "demo",
                "enabled": False,
                "symbol": "",
                "has_credentials": has_creds,
                "has_active_position": False,
                "last_setup_ts_ms": 0,
            }
        bot = slot.bot
        return {
            "state": bot.state.value,
            "run_mode": slot.settings.run_mode.value,
            "enabled": slot.settings.enabled,
            "symbol": bot.symbol,
            "has_credentials": slot.settings.has_credentials(),
            "has_active_position": bot.active_position is not None,
            "last_setup_ts_ms": bot._last_setup_ts_ms,
        }

    async def stop_all(self) -> None:
        """모든 사용자 봇 정지 — 서버 종료 (FastAPI shutdown hook) 시 호출.

        예외 발생해도 다음 사용자 정지 시도 (best-effort).
        """
        user_codes = list(self._slots.keys())
        for code in user_codes:
            try:
                await self.stop(code)
            except Exception as e:  # noqa: BLE001
                logger.warning("stop_all: 사용자 %s 정지 실패 — %s", code, e)
        logger.info("MultiUserBotManager: 전체 사용자 봇 정지 완료 (%d명)", len(user_codes))

    def list_users(self) -> list[str]:
        """현재 슬롯에 등록된 사용자 코드 목록 (디버그/관리용)."""
        return list(self._slots.keys())


__all__ = [
    "ClientFactory",
    "MultiUserBotManager",
]
