"""HTF flip 경로 검증 — #FLIP-REFINE (역진입 제거·15m target 제외) + 레거시 회귀.

2026-07-02 #FLIP-REFINE (Origo 1.3, FST #2 실거래 반사실):
    * 기본 경로 = flip 은 청산(방어)까지만 — 역진입 생략 (실측 113건 net -301,
      승률 19%, 전 TF 적자).
    * flip target 은 1h(weight 4)+ 존만 — 15m 존이 TP 한참 전에 승자를 자르던
      설계 모순 해소 (반사실 @15m Δ+46R).
레거시(역진입) 경로는 ``_FLIP_REVERSE_ENABLED`` 로 보존 — 켰을 때의 SL/TP 처리
회귀(#FLIP-TP / #FLIP-SL)도 함께 검증한다.
"""

from __future__ import annotations

import pandas as pd
import pytest

import aurora_ict.bot.bot_ict_instance as bot_mod
from aurora_ict.bot.bot_ict_instance import BotIctInstance, _ActivePosition
from aurora_ict.indicators.fvg import FVGType
from aurora_ict.strategy.htf_fvg_map import HtfFvgEntry
from aurora_ict.strategy.silver_bullet import Direction
from tests.test_origo_execution_safety import OrderLedger


def _client() -> OrderLedger:
    client = OrderLedger()
    # flip 신규 진입 = SHORT. 현재가는 trigger_price 부근.
    client.price = 99.0
    client.position = dict(contracts=1.0, side="long", entryPrice=100.0)
    client.immediate_fill_fraction = 1.0
    return client


def _long_pos() -> _ActivePosition:
    return _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0,
        take_profit=115.0, qty=1.0, setup_ts_ms=0,
    )


def _target(tf: str = "1h", weight: int = 10) -> HtfFvgEntry:
    return HtfFvgEntry(
        tf=tf, weight=weight, type=FVGType.BEARISH,
        high=102.0, low=98.0, ts_ms=1000,
    )


@pytest.mark.asyncio
async def test_flip_default_closes_without_reverse_entry(tmp_path) -> None:
    """#FLIP-REFINE 기본 경로: 청산만 하고 역진입 없음 — 포지션 비움."""
    client = _client()
    bot = BotIctInstance(client=client, trades_data_dir=tmp_path)
    bot.active_position = _long_pos()

    await bot.handle_htf_flip(trigger_price=99.0, ts_ms=2000, target=_target())

    # 청산 주문 1건(reduce_only)만 — 신규 진입 주문 없음.
    assert len(client.requests) == 1
    assert client.requests[0]["reduce_only"] is True
    # 포지션 비움 + 재보호(set_position_tpsl) 호출 없음.
    assert bot.active_position is None
    assert client.protection == []


@pytest.mark.asyncio
async def test_flip_legacy_reverse_sets_take_profit_on_exchange(monkeypatch, tmp_path) -> None:
    """레거시(역진입 ON) 회귀 #FLIP-TP: SHORT flip 시 set_position_tpsl 에 TP 동봉."""
    monkeypatch.setattr(bot_mod, "_FLIP_REVERSE_ENABLED", True)
    client = _client()
    bot = BotIctInstance(client=client, trades_data_dir=tmp_path, trail_trigger_r=0)
    bot.active_position = _long_pos()

    await bot.handle_htf_flip(trigger_price=99.0, ts_ms=2000, target=_target())

    # flip 후 SHORT 신규 포지션이 살아있다.
    assert bot.active_position is not None
    assert bot.active_position.direction is Direction.SHORT
    # 거래소에 TP 가 함께 걸렸다 — set_position_tpsl take_profit != None.
    assert len(client.protection) >= 1
    kw = client.protection[-1]
    assert kw["take_profit"] is not None
    assert kw["take_profit"] == bot.active_position.take_profit
    # SHORT TP 는 entry 아래여야 한다.
    assert bot.active_position.take_profit < bot.active_position.entry


@pytest.mark.asyncio
async def test_flip_legacy_entry_order_includes_stop_loss(monkeypatch, tmp_path) -> None:
    """레거시를 켜더라도 일반 진입과 동일하게 진입 SL을 동봉한다."""
    monkeypatch.setattr(bot_mod, "_FLIP_REVERSE_ENABLED", True)
    client = _client()
    bot = BotIctInstance(client=client, trades_data_dir=tmp_path, trail_trigger_r=0)
    bot.active_position = _long_pos()

    await bot.handle_htf_flip(trigger_price=99.0, ts_ms=2000, target=_target())

    assert len(client.requests) == 2
    assert client.requests[0]["reduce_only"] is True
    assert client.requests[1]["reduce_only"] is False
    assert client.requests[1]["stop_loss"] == 102.0


@pytest.mark.asyncio
async def test_flip_target_skips_15m_zone(monkeypatch) -> None:
    """#FLIP-REFINE: target 선정은 1h(weight 4)+ 존만 — 15m(2) 은 건너뛴다.

    합산 가중치 threshold 는 15m 포함 전체로 넘더라도, "가장 가까운" 15m 존이
    target 이 되면 TP 한참 전에 승자를 자름 → 15m skip 후 첫 1h+ 존 반환.
    """
    client = _client()
    bot = BotIctInstance(client=client)
    # 합산 가중치(2+10=12)가 threshold(max(ltf×3,6)=6) 를 넘어야 후보 반환.
    near_15m = HtfFvgEntry(tf="15m", weight=2, type=FVGType.BEARISH,
                           high=101.0, low=100.5, ts_ms=1000)
    far_1h = HtfFvgEntry(tf="1h", weight=10, type=FVGType.BEARISH,
                         high=103.0, low=102.0, ts_ms=1000)

    async def _fake_map(self, _ts: int):
        return [near_15m, far_1h]

    # slots 클래스 — 클래스 레벨 교체 (외부 fetch 차단).
    monkeypatch.setattr(BotIctInstance, "_ensure_htf_fvg_map", _fake_map)
    idx = pd.date_range("2026-07-01", periods=3, freq="5min")
    ltf_df = pd.DataFrame({"close": [100.0, 100.0, 100.0]}, index=idx)

    class _Setup:
        direction = Direction.LONG

    got = await bot._evaluate_htf_override(_Setup(), ltf_df)
    assert got is far_1h  # 15m 존 skip, 1h 존 채택

    # 1h+ 존이 아예 없으면 target 미무장 (None) — 15m 4개(합산 8 > 6)로
    # threshold 는 넘겨서 "필터 때문에" None 인 것을 검증.
    async def _only_15m(self, _ts: int):
        return [
            HtfFvgEntry(tf="15m", weight=2, type=FVGType.BEARISH,
                        high=101.0 + i, low=100.5 + i, ts_ms=1000)
            for i in range(4)
        ]

    monkeypatch.setattr(BotIctInstance, "_ensure_htf_fvg_map", _only_15m)
    got2 = await bot._evaluate_htf_override(_Setup(), ltf_df)
    assert got2 is None
