"""#COND-ALIGN 2026-07-17 (Origo 2.0, FST#6): 조건부 방향정합 게이트 테스트.

약/중추세(|entry_trend|<강추세 q70)에선 진입방향이 20봉 추세와 정합일 때만,
강추세면 반전(역추세)도 허용. 자율연구: 극톱질 역추세진입만 -1.0 → 제거.

검증:
  - 약추세 + 역추세(추세와 반대 방향) → cond_align_skip
  - 약추세 + 정합(추세와 같은 방향) → 통과
  - 강추세 + 역추세 → 통과(반전 허용)
  - 게이트 off → 스킵 안 함
self-spy: _set_entry_trend 를 self-spy 하여 원하는 entry_trend_pct 주입.
_shadow_seen 키에서 verdict 확인(__slots__ 라 메서드 교체 불가).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot import bot_ict_instance as mod
from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.indicators.fvg import FVG, FVGType
from aurora_ict.signal.ict_signal import ICTSignal, SignalAction
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup


def _rows() -> list[list[Any]]:
    """London 시간(UTC 8) 종료 5분봉 20개 — NY_PM·촙게이트 무관 시간대."""
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


def _signal(direction: Direction, ts_ms: int) -> ICTSignal:
    fvg = FVG(type=FVGType.BULLISH, idx=5, ts_ms=ts_ms, low=98, high=102)
    setup = SilverBulletSetup(
        ts_ms=ts_ms, direction=direction, window="any",
        entry=100.0, stop_loss=95.0, take_profit=115.0, risk_reward=3.0, fvg=fvg)
    act = SignalAction.ENTER_LONG if direction is Direction.LONG else SignalAction.ENTER_SHORT
    return ICTSignal(action=act, setup=setup, symbol="BTCUSDT", ts_ms=ts_ms, reason="t")


async def _run(monkeypatch, direction: Direction, trend_pct: float,
               cond_align: bool = True) -> set[str]:
    """trend_pct 를 주입하고 step() 실행 → shadow verdict 집합."""
    bot = BotIctInstance(client=_client(), symbol="BTCUSDT",
                         cond_align_enabled=cond_align, regime_filter_enabled=False,
                         disable_time_filter=True, shadow_log_enabled=True)

    def _fake_signal(df, symbol, **kw):
        return _signal(direction, int(df.index[-1].value // 10**6))

    def _fake_trend(self, setup, df):  # self-spy: entry_trend_pct 강제 주입
        setup.entry_trend_pct = trend_pct

    monkeypatch.setattr(mod, "generate_ict_signal", _fake_signal)
    monkeypatch.setattr(BotIctInstance, "_set_entry_trend", _fake_trend)
    await bot.step()
    return {v for (_ts, _dir, v) in bot._shadow_seen}


# BTCUSDT 강추세 q70 = 0.401. 약추세 = 0.1, 강추세 = 0.6.


@pytest.mark.asyncio
async def test_weak_countertrend_blocked(monkeypatch) -> None:
    """약추세(0.1) + 역추세(하락추세인데 롱) → cond_align_skip."""
    # trend=-0.1(하락) 인데 LONG → signed=-0.1<0 역추세, |trend|0.1<0.401 약추세.
    verdicts = await _run(monkeypatch, Direction.LONG, -0.1)
    assert "cond_align_skip" in verdicts


@pytest.mark.asyncio
async def test_weak_aligned_passes(monkeypatch) -> None:
    """약추세(0.1) + 정합(상승추세에 롱) → 통과(cond_align 미발동)."""
    verdicts = await _run(monkeypatch, Direction.LONG, +0.1)
    assert "cond_align_skip" not in verdicts


@pytest.mark.asyncio
async def test_strong_countertrend_allowed(monkeypatch) -> None:
    """강추세(0.6>=q70 0.401) + 역추세(상승에 숏) → 반전 허용(통과)."""
    # trend=+0.6(상승) 인데 SHORT → signed=-0.6<0 역추세지만 |0.6|>=0.401 강추세.
    verdicts = await _run(monkeypatch, Direction.SHORT, +0.6)
    assert "cond_align_skip" not in verdicts


@pytest.mark.asyncio
async def test_gate_off_no_skip(monkeypatch) -> None:
    """cond_align_enabled=False 면 약추세 역추세라도 미발동(하위호환)."""
    verdicts = await _run(monkeypatch, Direction.LONG, -0.1, cond_align=False)
    assert "cond_align_skip" not in verdicts
