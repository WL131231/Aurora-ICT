"""TelegramAlerter 테스트 — 매매 알림 포맷 + 코드↔chat_id 연동 로직.

httpx 네트워크 호출(send)은 self-spy 로 교체해 외부 의존 없이 검증(mock 0 정책).
"""
from __future__ import annotations

import pytest

from aurora_ict.auth import users_db
from aurora_ict.interfaces.telegram_alerter import TelegramAlerter, format_trade
from aurora_ict.interfaces.trades_store import TradeEvent, TradeEventType


def test_format_trade_entry() -> None:
    """ENTRY 이벤트 포맷 — 라벨/심볼/방향/코드 포함."""
    ev = TradeEvent(
        ts_ms=0, event_type=TradeEventType.ENTRY, symbol="BTC/USDT:USDT",
        direction="short", price=60123.4, qty=0.05, reason="confluence=2 rr=3.2",
    )
    text = format_trade("AICT-TGTG-TGTG-TGTG", ev)
    assert "진입" in text
    assert "BTC/USDT:USDT" in text
    assert "SHORT" in text
    assert "confluence=2" in text
    assert "AICT-TGTG-TGTG-TGTG" in text


def test_format_trade_tp_with_pnl() -> None:
    """청산(TP) 이벤트 — 손익 표시(+부호)."""
    ev = TradeEvent(
        ts_ms=0, event_type=TradeEventType.TP_HIT, symbol="ETH/USDT:USDT",
        direction="long", price=1589.0, qty=1.4, pnl_usdt=13.36,
    )
    text = format_trade("AICT-TGTG-TGTG-TGTG", ev)
    assert "익절" in text
    assert "+13.36" in text


@pytest.mark.asyncio
async def test_handle_message_registers_chat_id(tmp_path) -> None:
    """유효 코드 입력 → chat_id 연동 + 완료 메시지."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    code = "AICT-TGRG-TGRG-TGRG"
    users_db.create_user(db, code)

    al = TelegramAlerter("dummytoken", db)
    sent: list[tuple[str, str]] = []

    async def _spy(chat_id: str, text: str) -> None:
        sent.append((chat_id, text))

    al.send = _spy  # type: ignore[method-assign]  # self-spy
    await al._handle_message("12345", f"내 코드 {code}")
    assert users_db.get_telegram_chat_id(db, code) == "12345"
    assert any("연동 완료" in t for _, t in sent)
    await al.aclose()


@pytest.mark.asyncio
async def test_handle_message_unknown_code(tmp_path) -> None:
    """미등록 코드 → chat_id 저장 안 함 + 거부 메시지."""
    db = tmp_path / "users.db"
    users_db.init_db(db)

    al = TelegramAlerter("dummytoken", db)
    sent: list[tuple[str, str]] = []

    async def _spy(chat_id: str, text: str) -> None:
        sent.append((chat_id, text))

    al.send = _spy  # type: ignore[method-assign]
    await al._handle_message("999", "AICT-NONE-NONE-NONE")
    assert any("등록되지 않은" in t for _, t in sent)
    await al.aclose()


@pytest.mark.asyncio
async def test_send_trade_alert_skips_when_not_linked(tmp_path) -> None:
    """chat_id 미연동 사용자는 알림 발송 안 함(send 호출 0)."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    code = "AICT-NOLK-NOLK-NOLK"
    users_db.create_user(db, code)

    al = TelegramAlerter("dummytoken", db)
    sent: list[tuple[str, str]] = []

    async def _spy(chat_id: str, text: str) -> None:
        sent.append((chat_id, text))

    al.send = _spy  # type: ignore[method-assign]
    ev = TradeEvent(
        ts_ms=0, event_type=TradeEventType.ENTRY, symbol="BTC/USDT:USDT",
        direction="long", price=60000.0, qty=0.01,
    )
    await al.send_trade_alert(code, ev)
    assert sent == []  # 미연동 → skip
    await al.aclose()
