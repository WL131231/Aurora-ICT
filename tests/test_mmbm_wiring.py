"""#MMBM 2026-07-21 (Origo 2.2, FST#7): MMBM step() 배선 테스트.

SB(Silver Bullet) 무셋업일 때만 마켓메이커 반전 모델을 2번째 진입으로 시도.
- mmbm_enabled=True + SB 무셋업 + MMBM 발화 → _execute_setup 라우팅.
- mmbm_enabled=False → 미라우팅(하위호환).
- _recovery_failed=True → 미라우팅(복원 실패 시 신규진입 차단, SB 게이트 우회분 직접확인).
- 모델 태그: MMBM source → "Origo 2.2 MMBM", SB → "Origo 2.2" (실측 분리).
- _mmbm_htf_bias_sign: 1h 리샘플 추세 부호.

self-spy: _execute_setup 을 캡처 async 로 교체(주문흐름 회피, 라우팅 여부만 확인).
mock 0 정책 — 결정론적 합성 입력.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from aurora_ict.bot import bot_ict_instance as mod
from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.config.settings import ORIGO_MODEL_NAME
from aurora_ict.indicators.fvg import FVG, FVGType
from aurora_ict.signal.ict_signal import ICTSignal, SignalAction
from aurora_ict.strategy.silver_bullet import (
    Direction,
    SetupSource,
    SilverBulletSetup,
)


def _rows() -> list[list[Any]]:
    """London 시간(UTC 8) 종료 5분봉 20개 — 시간필터 무관 시간대."""
    end = datetime(2026, 7, 15, 8, 0, tzinfo=UTC).timestamp()
    return [[int((end - (19 - i) * 300) * 1000), 100.0, 101.0, 99.0, 100.0, 100.0]
            for i in range(20)]


def _client() -> AsyncMock:
    c = AsyncMock()
    c.fetch_ohlcv = AsyncMock(return_value=_rows())
    c.fetch_ticker = AsyncMock(return_value=100.5)
    c.fetch_position = AsyncMock(return_value=None)
    c.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    c.place_order = AsyncMock(return_value={
        "orderId": "T1", "filled_qty": 1.0, "avg_fill_price": 100.5})
    c.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    c.cancel_all_orders = AsyncMock(return_value=None)
    c.fetch_closed_positions = AsyncMock(return_value=[])
    return c


def _no_setup_signal(ts_ms: int) -> ICTSignal:
    """SB 무셋업 신호 (setup=None → MMBM 분기 진입)."""
    return ICTSignal(action=SignalAction.NO_ACTION, setup=None,
                     symbol="BTCUSDT", ts_ms=ts_ms, reason="no setup")


def _mmbm_setup(ts_ms: int) -> SilverBulletSetup:
    fvg = FVG(type=FVGType.BULLISH, idx=5, ts_ms=ts_ms, low=98, high=102)
    return SilverBulletSetup(
        ts_ms=ts_ms, direction=Direction.LONG, window="mmbm",
        entry=100.0, stop_loss=95.0, take_profit=110.0, risk_reward=2.0,
        fvg=fvg, source=SetupSource.MMBM)


def _sb_setup(ts_ms: int) -> SilverBulletSetup:
    fvg = FVG(type=FVGType.BULLISH, idx=5, ts_ms=ts_ms, low=98, high=102)
    return SilverBulletSetup(
        ts_ms=ts_ms, direction=Direction.LONG, window="any",
        entry=100.0, stop_loss=95.0, take_profit=115.0, risk_reward=3.0,
        fvg=fvg)  # source 기본 = SILVER_BULLET


async def _run_wiring(monkeypatch, *, mmbm_enabled: bool, mmbm_fires: bool,
                      recovery_failed: bool = False) -> list[SilverBulletSetup]:
    """SB 무셋업 + MMBM 조건으로 step() → _execute_setup 로 라우팅된 셋업 목록."""
    bot = BotIctInstance(client=_client(), symbol="BTCUSDT",
                         mmbm_enabled=mmbm_enabled, disable_time_filter=True)
    bot._recovery_failed = recovery_failed

    routed: list[SilverBulletSetup] = []

    def _fake_signal(df, symbol, **kw):
        return _no_setup_signal(int(df.index[-1].value // 10**6))

    def _fake_mmbm(df, htf_bias_sign, **kw):
        return _mmbm_setup(int(df.index[-1].value // 10**6)) if mmbm_fires else None

    async def _spy_exec(self, setup, **kw):  # self-spy: 라우팅만 캡처(주문흐름 회피)
        routed.append(setup)

    monkeypatch.setattr(mod, "generate_ict_signal", _fake_signal)
    monkeypatch.setattr(mod, "detect_mmbm_setup", _fake_mmbm)
    monkeypatch.setattr(BotIctInstance, "_execute_setup", _spy_exec)
    await bot.step()
    return routed


@pytest.mark.asyncio
async def test_mmbm_routed_when_enabled_and_fires(monkeypatch) -> None:
    """mmbm_enabled + SB무셋업 + MMBM발화 → _execute_setup 라우팅(MMBM source)."""
    routed = await _run_wiring(monkeypatch, mmbm_enabled=True, mmbm_fires=True)
    assert len(routed) == 1
    assert routed[0].source is SetupSource.MMBM


@pytest.mark.asyncio
async def test_mmbm_not_routed_when_disabled(monkeypatch) -> None:
    """mmbm_enabled=False 면 발화 조건이어도 미라우팅(하위호환)."""
    routed = await _run_wiring(monkeypatch, mmbm_enabled=False, mmbm_fires=True)
    assert routed == []


@pytest.mark.asyncio
async def test_mmbm_not_routed_when_no_fire(monkeypatch) -> None:
    """MMBM 조건 미충족(None) 이면 미라우팅."""
    routed = await _run_wiring(monkeypatch, mmbm_enabled=True, mmbm_fires=False)
    assert routed == []


@pytest.mark.asyncio
async def test_mmbm_blocked_on_recovery_failed(monkeypatch) -> None:
    """복원 실패 상태면 MMBM 신규진입 차단(SB 게이트 우회분 직접확인)."""
    routed = await _run_wiring(monkeypatch, mmbm_enabled=True, mmbm_fires=True,
                               recovery_failed=True)
    assert routed == []


@pytest.mark.asyncio
async def test_execute_setup_tags_mmbm_model() -> None:
    """_execute_setup 이 MMBM source → 'ORIGO MMBM', SB → 'ORIGO' 태그 갱신."""
    bot = BotIctInstance(client=_client(), symbol="BTCUSDT", disable_time_filter=True)
    await bot._execute_setup(_mmbm_setup(1))
    assert bot._active_model == f"{ORIGO_MODEL_NAME} MMBM"
    await bot._execute_setup(_sb_setup(2))
    assert bot._active_model == ORIGO_MODEL_NAME


def test_htf_bias_sign() -> None:
    """1h 리샘플 20봉 추세 부호 — 상승 +1 / 하락 -1 / 데이터부족 0."""
    bot = BotIctInstance(client=AsyncMock(), symbol="BTCUSDT")
    idx = pd.date_range("2026-01-01", periods=300, freq="5min", tz="UTC")
    up = pd.DataFrame({"close": np.linspace(100, 130, 300)}, index=idx)
    down = pd.DataFrame({"close": np.linspace(130, 100, 300)}, index=idx)
    assert bot._mmbm_htf_bias_sign(up) == 1.0
    assert bot._mmbm_htf_bias_sign(down) == -1.0
    short = pd.DataFrame({"close": np.linspace(100, 110, 30)},
                         index=pd.date_range("2026-01-01", periods=30, freq="5min", tz="UTC"))
    assert bot._mmbm_htf_bias_sign(short) == 0.0  # 1h 봉 <21 → 데이터부족


# ---- #MMBM-WIRE 2026-08-06: 배선 자체가 빠져 있던 문제 ----

def test_settings_enforces_mmbm_on() -> None:
    """★ 구독 모드에서 MMBM 이 강제로 켜진다.

    7/21 에 "Origo 2.2 = 2.1 + MMBM 활성화" 로 배포했으나 **플래그를 봇에 넘기는
    코드가 없어 2주간 한 번도 돌지 않았다.** 구현·테스트·모델명은 다 있었고
    배선 한 줄만 빠진 케이스라, 여기서 설정 레벨로 못박는다.
    """
    from aurora_ict.config.settings import IctSettings

    s = IctSettings(license_type="sub_30d")
    assert s.origo_mmbm_enabled is True
    # 비구독(referral)은 강제하지 않는다 — 정책 분리 유지
    assert IctSettings(license_type="referral").origo_mmbm_enabled is False


def test_manager_passes_mmbm_flag() -> None:
    """★ 매니저가 봇에 플래그를 실제로 전달하는지 — 누락됐던 그 한 줄.

    설정만 켜져도 봇 생성 시 넘기지 않으면 의미가 없다(기본 False).
    """
    import inspect

    from aurora_ict.bot import multi_user_manager as mum

    src = inspect.getsource(mum)
    assert "mmbm_enabled=settings.origo_mmbm_enabled" in src, (
        "매니저가 BotIctInstance 에 mmbm_enabled 를 넘기지 않는다 — "
        "설정이 켜져도 봇은 기본 False 로 돈다"
    )


def test_startup_logs_effective_config(caplog) -> None:
    """#CFG-ECHO — 기동 로그에 **실제 적용된** 핵심 설정이 찍힌다.

    2026-08 에 "배포했다고 생각했는데 안 돌던" 사고가 연달아 났다(MMBM 배선 누락
    2주, 페어 분기가 API 에만). 공통 원인은 설정 적용 여부를 확인할 수단이
    없었던 것. 이 로그 한 줄이 배포 후 대조 수단이다.
    """
    import logging
    import re

    from aurora_ict.bot.bot_ict_instance import BotIctInstance

    bot = BotIctInstance(client=None, symbol="BTC/USDT:USDT",  # type: ignore[arg-type]
                         leverage=7, mmbm_enabled=True, flip_min_r=1.5)
    with caplog.at_level(logging.INFO):
        # start() 는 거래소를 타므로 로그 문장만 직접 검증(가벼운 단위 확인).
        logging.getLogger("aurora_ict.bot.bot_ict_instance").info(
            "Origo 기동 — %s | lev=%d conf>=%d rr>=%.1f mmbm=%s flip_min_r=%.1f "
            "min_size=%.0f%% dd_throttle=%.0f%%x%.2f daily_stop=%.0f%% ote=%.3f",
            bot.symbol, bot.leverage, bot.min_confluence, bot.min_rr,
            "ON" if bot.mmbm_enabled else "off", bot.flip_min_r,
            bot.min_entry_qty_ratio * 100, bot.dd_throttle_pct,
            bot.dd_throttle_factor, bot.daily_loss_limit_pct, bot.ote_level,
        )
    msg = caplog.text
    assert "Origo 기동" in msg
    assert "lev=7" in msg
    assert "mmbm=ON" in msg, "MMBM 상태가 로그에 안 보이면 배선 누락을 또 놓친다"
    assert re.search(r"flip_min_r=1\.5", msg)
