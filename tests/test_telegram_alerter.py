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


def test_format_trade_entry_detail_from_context() -> None:
    """진입 — context_json 의 진입가/TP/SL/등급/window 상세 표시."""
    import json
    ctx = json.dumps({
        "entry": 61844.8, "sl": 62210.0, "tp": 61000.0,
        "score": 2, "rr": 3.75, "window": "turtle",
    })
    ev = TradeEvent(
        ts_ms=0, event_type=TradeEventType.ENTRY, symbol="BTC/USDT:USDT",
        direction="short", price=61844.8, qty=0.018, context_json=ctx,
    )
    text = format_trade("AICT-X-X-X", ev)
    assert "진입가" in text and "61,844" in text
    assert "목표가(TP)" in text and "61,000" in text
    assert "손절가(SL)" in text and "62,210" in text
    assert "등급 2" in text
    assert "turtle" in text


def test_format_trade_close_detail_from_context() -> None:
    """청산 — 손실 라벨 + 청산가 + 변동% + 사유."""
    import json
    ctx = json.dumps({
        "close_price": 62211.9, "move_pct": 0.59,
        "close_reason": "SL_HIT", "pnl_usd": -7.32,
    })
    ev = TradeEvent(
        ts_ms=0, event_type=TradeEventType.SL_HIT, symbol="BTC/USDT:USDT",
        direction="short", price=62211.9, qty=0.018, pnl_usdt=-7.32,
        context_json=ctx,
    )
    text = format_trade("AICT-X-X-X", ev)
    assert "손절" in text
    assert "손실" in text  # pnl < 0 → loss 라벨
    assert "-7.32" in text
    assert "청산가" in text and "62,211" in text
    assert "변동" in text and "+0.59%" in text
    assert "SL_HIT" in text


def test_format_trade_language_en() -> None:
    """영어 라벨 출력."""
    ev = TradeEvent(
        ts_ms=0, event_type=TradeEventType.TP_HIT, symbol="ETH/USDT:USDT",
        direction="long", price=1589.0, qty=1.4, pnl_usdt=13.36,
    )
    text = format_trade("AICT-X-X-X", ev, lang="en")
    assert "Take Profit" in text
    assert "Profit" in text
    assert "+13.36" in text


def test_user_prefs_roundtrip(tmp_path) -> None:
    """users_db 언어/시간대 부분 업데이트 round-trip."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    code = "AICT-PR2W-PR2W-PR2W"
    users_db.create_user(db, code)
    assert users_db.get_user_prefs(db, code) == {"language": None, "timezone": None}
    users_db.set_user_prefs(db, code, language="ja")
    assert users_db.get_user_prefs(db, code)["language"] == "ja"
    users_db.set_user_prefs(db, code, timezone="Asia/Tokyo")
    p = users_db.get_user_prefs(db, code)
    assert p["language"] == "ja" and p["timezone"] == "Asia/Tokyo"  # 부분 유지


@pytest.mark.asyncio
async def test_call_retries_then_succeeds(tmp_path) -> None:
    """_call — 첫 호출 네트워크 실패 후 재시도로 성공(알림 누락 방지)."""
    al = TelegramAlerter("dummytoken", tmp_path / "u.db")
    calls: list[int] = []

    class _Resp:
        def json(self) -> dict:
            return {"ok": True}

    async def _post(url: str, json: dict) -> _Resp:  # noqa: A002, ARG001
        calls.append(1)
        if len(calls) < 2:
            raise RuntimeError("일시 네트워크 오류")
        return _Resp()

    al._client.post = _post  # type: ignore[method-assign]  # self-spy
    r = await al._call("sendMessage", {}, retries=2)
    assert r == {"ok": True}
    assert len(calls) == 2  # 1실패 + 1성공(재시도)
    await al.aclose()


@pytest.mark.asyncio
async def test_send_trade_alert_fallback_on_format_error(tmp_path, monkeypatch) -> None:
    """format_trade 예외 시에도 fallback 메시지로 알림이 나간다(통째 누락 방지)."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    code = "AICT-FBCK-FBCK-FBCK"
    users_db.create_user(db, code)
    users_db.set_telegram_chat_id(db, code, "1")

    al = TelegramAlerter("dummytoken", db)
    sent: list[str] = []

    async def _spy(chat_id: str, text: str) -> None:
        sent.append(text)

    al.send = _spy  # type: ignore[method-assign]

    import aurora_ict.interfaces.telegram_alerter as ta

    def _boom(*a, **k):  # noqa: ANN002, ANN003, ANN202, ARG001
        raise ValueError("format 깨짐")

    monkeypatch.setattr(ta, "format_trade", _boom)
    ev = TradeEvent(
        ts_ms=0, event_type=TradeEventType.ENTRY, symbol="BTC/USDT:USDT",
        direction="short", price=1.0, qty=1.0,
    )
    await al.send_trade_alert(code, ev)
    assert sent and "매매" in sent[0] and code in sent[0]  # fallback 발송됨
    await al.aclose()


@pytest.mark.asyncio
async def test_send_trade_alert_uses_user_prefs(tmp_path) -> None:
    """send_trade_alert 가 사용자 언어 설정으로 포맷(en)."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    code = "AICT-PREF-PREF-PREF"
    users_db.create_user(db, code)
    users_db.set_telegram_chat_id(db, code, "999")
    users_db.set_user_prefs(db, code, language="en", timezone="America/New_York")

    al = TelegramAlerter("dummytoken", db)
    sent: list[tuple[str, str]] = []

    async def _spy(chat_id: str, text: str) -> None:
        sent.append((chat_id, text))

    al.send = _spy  # type: ignore[method-assign]
    ev = TradeEvent(
        ts_ms=0, event_type=TradeEventType.TP_HIT, symbol="ETH/USDT:USDT",
        direction="long", price=1589.0, qty=1.4, pnl_usdt=13.36,
    )
    await al.send_trade_alert(code, ev)
    assert sent and "Take Profit" in sent[0][1]
    await al.aclose()


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
