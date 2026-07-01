"""Cursus(BotTrendInstance) 매매 알림 발송 회귀 테스트.

2026-07-01: Cursus ``_record_trade`` 가 ``alert_cb``(텔레그램 발송)를 호출하지
않아 매매해도 알림이 안 가던 버그 수정 검증(파트너 보고). Origo 와 동일하게
ENTRY/청산(SL_HIT/TP_HIT/FLIP_CLOSE)은 발송, RECOVERED(재시작 재인식)/
SYNC_CLOSE(사후 동기화)는 재시작 소음 방지로 생략.

mock 0 정책 — 외부 mock 라이브러리 없이 캡처용 async 콜백만 주입. 거래소는
_record_trade 가 호출하지 않으므로 빈 stub 로 충분.
담당: 지영민.
"""
from __future__ import annotations

import asyncio
from pathlib import Path

from aurora_ict.bot.bot_trend_instance import BotTrendInstance
from aurora_ict.interfaces.trades_store import TradeEventType
from aurora_ict.strategy.silver_bullet import Direction


class _DummyClient:
    """_record_trade 는 거래소를 호출하지 않으므로 생성용 빈 stub."""


def _make_bot(tmp_path: Path, captured: list, code: str = "TESTCODE") -> BotTrendInstance:
    async def alert_cb(user_code, event):
        captured.append((user_code, event.event_type))

    return BotTrendInstance(
        client=_DummyClient(),
        symbol="BTC/USDT:USDT",
        trades_data_dir=tmp_path,
        user_code=code,
        alert_cb=alert_cb,
    )


def test_cursus_entry_fires_alert(tmp_path):
    """ENTRY 기록 시 alert_cb 가 사용자 코드 + 이벤트로 호출된다."""
    captured: list = []
    bot = _make_bot(tmp_path, captured)

    async def run():
        bot._record_trade(
            TradeEventType.ENTRY, direction=Direction.LONG, price=100.0, qty=1.0,
        )
        await asyncio.sleep(0)  # fire-and-forget task 실행 기회

    asyncio.run(run())
    assert captured == [("TESTCODE", TradeEventType.ENTRY)]


def test_cursus_close_fires_alert(tmp_path):
    """청산(SL_HIT) 도 알림 발송된다 — 손절 통지 필수."""
    captured: list = []
    bot = _make_bot(tmp_path, captured)

    async def run():
        bot._record_trade(
            TradeEventType.SL_HIT, direction=Direction.SHORT,
            price=90.0, qty=2.0, entry_for_pnl=100.0,
        )
        await asyncio.sleep(0)

    asyncio.run(run())
    assert captured == [("TESTCODE", TradeEventType.SL_HIT)]


def test_cursus_recovered_and_sync_close_skip_alert(tmp_path):
    """RECOVERED/SYNC_CLOSE 는 재시작 소음 방지로 발송 생략(기록은 유지)."""
    captured: list = []
    bot = _make_bot(tmp_path, captured)

    async def run():
        bot._record_trade(
            TradeEventType.RECOVERED, direction=Direction.LONG, price=100.0, qty=1.0,
        )
        bot._record_trade(
            TradeEventType.SYNC_CLOSE, direction=Direction.LONG,
            price=105.0, qty=1.0, entry_for_pnl=100.0,
        )
        await asyncio.sleep(0)

    asyncio.run(run())
    assert captured == []


def test_cursus_no_alert_cb_is_safe(tmp_path):
    """alert_cb 미주입(.exe/테스트)이어도 _record_trade 가 예외 없이 동작한다."""
    bot = BotTrendInstance(
        client=_DummyClient(), symbol="BTC/USDT:USDT",
        trades_data_dir=tmp_path, user_code="X",
    )

    async def run():
        bot._record_trade(
            TradeEventType.TP_HIT, direction=Direction.LONG,
            price=110.0, qty=1.0, entry_for_pnl=100.0,
        )
        await asyncio.sleep(0)

    asyncio.run(run())  # 예외 없이 통과하면 성공
