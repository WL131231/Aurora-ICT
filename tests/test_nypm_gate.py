"""#NYPM-GATE 2026-07-16 (FST#5): NY_PM 진입 차단 게이트 단위 테스트.

NY_PM(NY 13:30-16:00 = 02-05 KST)은 삼중검증 최악 구간(라이브 승률 10%,
5년 백테 7/7 페어 음수, 6/24 킬존연구). exclude_nypm=True 면 진입 시점이
NY_PM 이면 skip("nypm_skip"), 다른 킬존이면 게이트 통과함을 검증한다.

self-spy: generate_ict_signal 을 고정 actionable signal 로 교체(외부 mock 라이브러리
는 client 경계에만). _record_shadow verdict 를 캡처해 게이트 발동만 격리 확인.
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


def _rows_ending_at(utc_hour: int) -> list[list[Any]]:
    """마지막 봉이 주어진 UTC 시각(:00)에 닫히는 5분봉 20개."""
    # tzinfo=UTC 명시 — naive 면 로컬(KST)로 해석돼 킬존이 어긋난다.
    end = datetime(2026, 7, 15, utc_hour, 0, tzinfo=UTC).timestamp()
    rows = []
    for i in range(20):
        ts_ms = int((end - (19 - i) * 300) * 1000)
        rows.append([ts_ms, 100.0, 101.0, 99.0, 100.0, 100.0])
    return rows


def _client(utc_hour: int) -> AsyncMock:
    c = AsyncMock()
    c.fetch_ohlcv = AsyncMock(return_value=_rows_ending_at(utc_hour))
    c.fetch_ticker = AsyncMock(return_value=100.5)
    c.fetch_position = AsyncMock(return_value=None)
    c.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    c.place_order = AsyncMock(return_value={
        "orderId": "T1", "filled_qty": 1.0, "avg_fill_price": 100.5})
    c.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    c.cancel_all_orders = AsyncMock(return_value=None)
    c.fetch_closed_positions = AsyncMock(return_value=[])
    return c


def _actionable_signal(symbol: str, ts_ms: int) -> ICTSignal:
    fvg = FVG(type=FVGType.BULLISH, idx=5, ts_ms=ts_ms, low=98, high=102)
    setup = SilverBulletSetup(
        ts_ms=ts_ms, direction=Direction.LONG, window="any",
        entry=100.0, stop_loss=95.0, take_profit=115.0, risk_reward=3.0, fvg=fvg,
    )
    return ICTSignal(action=SignalAction.ENTER_LONG, setup=setup,
                     symbol=symbol, ts_ms=ts_ms, reason="test")


async def _run_capture(monkeypatch, utc_hour: int, exclude_nypm: bool) -> set[str]:
    """주어진 UTC 시각에 step() 실행, 기록된 shadow verdict 집합 반환.

    _record_shadow 는 verdict 를 self._shadow_seen 키((ts,dir,verdict))에 남긴다
    (JSONL 쓰기 전에). __slots__ 라 메서드 교체 불가 → 이 내부 기록을 직접 읽어
    게이트 발동 여부만 격리 확인. shadow_log_enabled=True 필요.
    """
    bot = BotIctInstance(client=_client(utc_hour), symbol="BTCUSDT",
                         exclude_nypm=exclude_nypm, disable_time_filter=True,
                         shadow_log_enabled=True)

    def _fake_signal(df, symbol, **kw):
        ts_ms = int(df.index[-1].value // 10**6)
        return _actionable_signal(symbol, ts_ms)

    monkeypatch.setattr(mod, "generate_ict_signal", _fake_signal)
    await bot.step()
    return {verdict for (_ts, _dir, verdict) in bot._shadow_seen}


# NY_PM = UTC 17-20시(NY EDT 13-16). Asian/London 대조군 = UTC 3 / 8.


@pytest.mark.asyncio
async def test_nypm_entry_blocked(monkeypatch) -> None:
    """진입 시점이 NY_PM(UTC 18)이면 nypm_skip 으로 차단."""
    verdicts = await _run_capture(monkeypatch, utc_hour=18, exclude_nypm=True)
    assert "nypm_skip" in verdicts


@pytest.mark.asyncio
async def test_non_nypm_not_blocked_by_gate(monkeypatch) -> None:
    """진입 시점이 London(UTC 8)이면 nypm 게이트는 통과(다른 사유는 무관)."""
    verdicts = await _run_capture(monkeypatch, utc_hour=8, exclude_nypm=True)
    assert "nypm_skip" not in verdicts


@pytest.mark.asyncio
async def test_gate_off_allows_nypm(monkeypatch) -> None:
    """exclude_nypm=False 면 NY_PM 라도 nypm 게이트 미발동(하위호환)."""
    verdicts = await _run_capture(monkeypatch, utc_hour=18, exclude_nypm=False)
    assert "nypm_skip" not in verdicts
