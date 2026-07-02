"""#TRAIL-EXCHANGE (Origo 1.4) — 거래소 네이티브 트레일링 무장 검증.

정합 스윕(7페어 5년, conf5/SLx4/rr2.0): 고정tp +124 → trail 2.0/1.5 +240
(RR 0.89→1.82). 무장 경로/degrade/분할익절 상호배제/입양 재무장을 검증한다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
from aurora_ict.strategy.silver_bullet import Direction


def _client(cur_price: float = 100.0) -> AsyncMock:
    client = AsyncMock()
    client.fetch_ticker = AsyncMock(return_value=cur_price)
    client.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    return client


def _bot(client: AsyncMock, trigger: float = 2.0, dist: float = 1.5) -> BotIctInstance:
    bot = BotIctInstance(client=client, trail_trigger_r=trigger, trail_dist_r=dist)
    # LONG entry=100, SL=96 → R=4. 활성가 = 100+2R = 108, 트레일 거리 = 1.5R = 6.
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=96.0,
        take_profit=108.0, qty=1.0, setup_ts_ms=0,
    )
    return bot


@pytest.mark.asyncio
async def test_arm_trailing_sets_far_tp_and_trailing_params() -> None:
    """무장 성공: TP 5R 확장 + trailingStop=1.5R 거리 + activePrice=2R."""
    client = _client(cur_price=100.0)
    bot = _bot(client)

    ok = await bot._arm_trailing()

    assert ok is True
    assert bot.active_position.trail_armed is True
    assert bot.active_position.take_profit == pytest.approx(120.0)  # 100 + 5×4
    kw = client.set_position_tpsl.await_args_list[-1].kwargs
    assert kw["take_profit"] == pytest.approx(120.0)
    assert kw["trailing_stop"] == pytest.approx(6.0)   # 1.5R = 6
    assert kw["active_price"] == pytest.approx(108.0)  # 2.0R 활성가


@pytest.mark.asyncio
async def test_arm_trailing_omits_active_price_when_already_beyond() -> None:
    """현재가가 활성가(108)를 지났으면 activePrice 생략 — 즉시 활성 (입양 케이스)."""
    client = _client(cur_price=110.0)
    bot = _bot(client)

    ok = await bot._arm_trailing()

    assert ok is True
    kw = client.set_position_tpsl.await_args_list[-1].kwargs
    assert kw["trailing_stop"] == pytest.approx(6.0)
    assert kw["active_price"] is None


@pytest.mark.asyncio
async def test_arm_trailing_failure_keeps_fixed_tp_mode() -> None:
    """무장 실패(거래소 거부) → trail_armed False, setup TP 그대로 (degrade 무해)."""
    client = _client()
    client.set_position_tpsl = AsyncMock(return_value={})  # 실패 (falsy)
    bot = _bot(client)

    ok = await bot._arm_trailing()

    assert ok is False
    assert bot.active_position.trail_armed is False
    assert bot.active_position.take_profit == pytest.approx(108.0)  # setup TP 유지


@pytest.mark.asyncio
async def test_arm_trailing_off_when_params_zero() -> None:
    """trigger/dist 0(off, referral 기본) → 무장 안 함, 호출 자체 없음."""
    client = _client()
    bot = _bot(client, trigger=0.0, dist=0.0)

    ok = await bot._arm_trailing()

    assert ok is False
    client.set_position_tpsl.assert_not_awaited()


@pytest.mark.asyncio
async def test_partial_exit_polling_skipped_when_trail_armed() -> None:
    """트레일 무장 포지션은 폴링 분할익절 skip (runner 전량 트레일)."""
    import pandas as pd

    client = _client()
    bot = _bot(client)
    bot.active_position.trail_armed = True
    bot.active_position.tp1_price = 106.0  # 분할 TP1 터치 상황
    idx = pd.date_range("2026-07-01", periods=1, freq="5min")
    df = pd.DataFrame({"high": [107.0], "low": [105.0], "close": [106.5]}, index=idx)

    await bot._maybe_partial_exit(df)

    # 부분청산 주문(place_order) 없음.
    assert not any(
        c.kwargs.get("reduce_only") for c in client.place_order.await_args_list
    )
    assert bot.active_position.partial_done is False


def test_subscription_forces_trail_params(monkeypatch):
    """구독제 = trail 2.0/1.5 강제, referral = 0(off) 유지."""
    from aurora_ict.config.settings import IctSettings

    for k in list(__import__("os").environ):
        if k.startswith("AURORA_ICT_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_90d")
    s = IctSettings(_env_file=None)
    assert s.origo_trail_trigger_r == 2.0
    assert s.origo_trail_dist_r == 1.5

    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "referral")
    s2 = IctSettings(_env_file=None)
    assert s2.origo_trail_trigger_r == 0.0
    assert s2.origo_trail_dist_r == 0.0
