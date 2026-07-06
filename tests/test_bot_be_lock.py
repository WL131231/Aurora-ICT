"""#BE-LOCK (Origo 1.5) — 이익 1R 도달 시 SL 본전 이동 검증.

MFE 실측(Origo 1.2 손절의 23%가 +20% ROI 이상 간 뒤 풀손절) 처방.
백테 BE@1R+trail2/1.5 = +278 (기준 +240), 1.0~1.25 고원 + walk-forward 통과.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
from aurora_ict.strategy.silver_bullet import Direction


def _client() -> AsyncMock:
    client = AsyncMock()
    client.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    return client


def _bot(client: AsyncMock, be: float = 1.0) -> BotIctInstance:
    bot = BotIctInstance(client=client, be_trigger_r=be)
    # LONG entry=100, SL=96 → R=4. BE 트리거 = 104.
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=96.0,
        take_profit=120.0, qty=1.0, setup_ts_ms=0,
    )
    return bot


def _df(high: float, low: float = 99.0) -> pd.DataFrame:
    idx = pd.date_range("2026-07-07", periods=1, freq="5min")
    return pd.DataFrame({"high": [high], "low": [low], "close": [high - 0.1]}, index=idx)


@pytest.mark.asyncio
async def test_be_lock_moves_sl_to_entry_at_1r() -> None:
    """이익 1R(=104) 터치 → SL 본전(100) 이동 + be_moved 셋."""
    client = _client()
    bot = _bot(client)

    await bot._maybe_be_lock(_df(high=104.5))

    kw = client.set_position_tpsl.await_args_list[-1].kwargs
    assert kw["stop_loss"] == pytest.approx(100.0)
    assert bot.active_position.stop_loss == pytest.approx(100.0)
    assert bot.active_position.be_moved is True


@pytest.mark.asyncio
async def test_be_lock_noop_below_threshold_and_once_only() -> None:
    """1R 미만이면 no-op. 이동 후 재호출도 no-op (1회만)."""
    client = _client()
    bot = _bot(client)

    await bot._maybe_be_lock(_df(high=103.0))  # 0.75R — 미달
    client.set_position_tpsl.assert_not_awaited()

    await bot._maybe_be_lock(_df(high=105.0))  # 이동
    await bot._maybe_be_lock(_df(high=110.0))  # 이미 be_moved — 추가 호출 없음
    assert client.set_position_tpsl.await_count == 1


@pytest.mark.asyncio
async def test_be_lock_off_when_zero_and_retries_on_failure() -> None:
    """be_trigger_r=0(referral) no-op. 이동 실패 시 be_moved 유지 안 됨(재시도 여지)."""
    client = _client()
    bot = _bot(client, be=0.0)
    await bot._maybe_be_lock(_df(high=110.0))
    client.set_position_tpsl.assert_not_awaited()

    client2 = _client()
    client2.set_position_tpsl = AsyncMock(return_value={})  # 실패
    bot2 = _bot(client2)
    await bot2._maybe_be_lock(_df(high=105.0))
    assert bot2.active_position.be_moved is False
    assert bot2.active_position.stop_loss == pytest.approx(96.0)  # 기존 SL 유지


def test_subscription_forces_be_trigger(monkeypatch):
    """구독제 = BE@1R 강제, referral = 0(off)."""
    import os

    from aurora_ict.config.settings import IctSettings

    for k in list(os.environ):
        if k.startswith("AURORA_ICT_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_90d")
    assert IctSettings(_env_file=None).origo_be_trigger_r == 1.0
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "referral")
    assert IctSettings(_env_file=None).origo_be_trigger_r == 0.0
