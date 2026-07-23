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
    """청산 — 손실 라벨 + 청산가 + 사유 (2026-06-12 파트너 포맷: ' : ' 라벨,
    빈 줄 그룹 구분, 변동% 제외, 코드는 마지막 줄)."""
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
    assert "손실 : " in text  # pnl < 0 → loss 라벨 + ' : ' 구분
    assert "-7.32" in text
    assert "청산가 : " in text and "62,211" in text
    assert "변동" not in text  # 파트너 템플릿에서 제외
    assert "사유 : " in text and "SL_HIT" in text
    assert text.rstrip().endswith("<code>AICT-X-X-X</code>")  # 코드가 마지막 줄
    assert "\n\n" in text  # 그룹 사이 빈 줄


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

    async def _spy(chat_id: str, text: str, *, keyboard: bool = False) -> None:
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

    async def _spy(chat_id: str, text: str, *, keyboard: bool = False) -> None:
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

    async def _spy(chat_id: str, text: str, *, keyboard: bool = False) -> None:
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

    async def _spy(chat_id: str, text: str, *, keyboard: bool = False) -> None:
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

    async def _spy(chat_id: str, text: str, *, keyboard: bool = False) -> None:
        sent.append((chat_id, text))

    al.send = _spy  # type: ignore[method-assign]
    ev = TradeEvent(
        ts_ms=0, event_type=TradeEventType.ENTRY, symbol="BTC/USDT:USDT",
        direction="long", price=60000.0, qty=0.01,
    )
    await al.send_trade_alert(code, ev)
    assert sent == []  # 미연동 → skip
    await al.aclose()


@pytest.mark.asyncio
async def test_relink_button_replaces_binding(tmp_path) -> None:
    """'코드 재등록' 버튼 → 새 코드 수신 시 기존 연동 해제 + 새 코드로 교체."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    old, new = "AICT-OLDC-OLDC-OLDC", "AICT-NEWC-NEWC-NEWC"
    users_db.create_user(db, old)
    users_db.create_user(db, new)

    al = TelegramAlerter("dummytoken", db)
    sent: list[str] = []

    async def _spy(chat_id: str, text: str, *, keyboard: bool = False) -> None:
        sent.append(text)

    al.send = _spy  # type: ignore[method-assign]

    # 기존 연동 → 재등록 버튼 → 새 코드.
    await al._handle_message("77", old)
    assert users_db.get_telegram_chat_id(db, old) == "77"
    await al._handle_message("77", "🔄 코드 재등록")
    assert any("재등록" in t for t in sent)
    await al._handle_message("77", new)
    assert users_db.get_telegram_chat_id(db, old) is None  # 기존 해제
    assert users_db.get_telegram_chat_id(db, new) == "77"  # 새 코드로 교체
    assert any("재등록 완료" in t for t in sent)
    await al.aclose()


@pytest.mark.asyncio
async def test_plain_link_does_not_unbind_other_codes(tmp_path) -> None:
    """재등록 버튼 없이 코드만 보내면(일반 연동) 기존 연동은 안 건드린다."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    a, b = "AICT-AAAA-AAAA-AAAA", "AICT-BBBB-BBBB-BBBB"
    users_db.create_user(db, a)
    users_db.create_user(db, b)

    al = TelegramAlerter("dummytoken", db)

    async def _spy(chat_id: str, text: str, *, keyboard: bool = False) -> None:
        pass

    al.send = _spy  # type: ignore[method-assign]
    await al._handle_message("88", a)
    await al._handle_message("88", b)
    assert users_db.get_telegram_chat_id(db, a) == "88"  # 유지
    assert users_db.get_telegram_chat_id(db, b) == "88"
    await al.aclose()


@pytest.mark.asyncio
async def test_restart_command_resets_client(tmp_path) -> None:
    """/restart — HTTP 클라이언트 재생성 + offset 리셋 + 완료 응답."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    al = TelegramAlerter("dummytoken", db)
    al._offset = 777
    old_client = al._client
    sent: list[str] = []

    async def _spy(chat_id: str, text: str, *, keyboard: bool = False) -> None:
        sent.append(text)

    al.send = _spy  # type: ignore[method-assign]
    await al._handle_message("9", "/restart")
    assert al._client is not old_client  # 새 클라이언트
    # 2026-06-12 리뷰 #1: offset 리셋 금지 — 0 이면 텔레그램이 /restart 를
    # 재배달해 재시작 무한루프. 유지돼야 한다.
    assert al._offset == 777
    assert any("재시작" in t for t in sent)
    await al.aclose()


@pytest.mark.asyncio
async def test_send_user_text_to_linked_chat(tmp_path) -> None:
    """send_user_text — 연동 chat 으로 일반 안내 발송, 미연동은 skip."""
    db = tmp_path / "users.db"
    users_db.init_db(db)
    code = "AICT-TERM-TERM-TERM"
    users_db.create_user(db, code)
    al = TelegramAlerter("dummytoken", db)
    sent: list[tuple[str, str]] = []

    async def _spy(chat_id: str, text: str, *, keyboard: bool = False) -> None:
        sent.append((chat_id, text))

    al.send = _spy  # type: ignore[method-assign]
    # 미연동 — skip
    await al.send_user_text(code, "약관 동의 필요")
    assert sent == []
    # 연동 후 발송
    users_db.set_telegram_chat_id(db, code, "55")
    await al.send_user_text(code, "약관 동의 필요")
    assert sent and sent[0][0] == "55" and "약관" in sent[0][1]
    await al.aclose()


def test_format_trade_entry_basis_from_confluences() -> None:
    """#2026-07-23: 진입 근거 구체화 — confluences 를 사람이 읽는 '근거' 라인으로."""
    import json

    from aurora_ict.interfaces.trades_store import TradeEvent, TradeEventType

    ctx = json.dumps({
        "entry": 100.0, "sl": 95.0, "tp": 115.0, "score": 5, "rr": 2.58,
        "window": "turtle",
        "confluences": ["ote", "cisd=bull", "sweep", "ob=bull@123",
                        "dol_counter_bear_-2"],
    })
    ev = TradeEvent(
        ts_ms=0, event_type=TradeEventType.ENTRY, symbol="LINK/USDT:USDT",
        direction="long", price=100.0, qty=1.0, context_json=ctx,
    )
    text = format_trade("AICT-X-X-X", ev)
    assert "근거" in text
    assert "OTE 되돌림" in text and "CISD 구조전환" in text
    assert "유동성 스윕" in text and "오더블록" in text
    assert "dol_counter" not in text          # 감점 메타 제외
    assert "등급 5" in text and "2.58" in text  # 기존 요약도 유지


def test_order_error_notify_classifies_and_throttles() -> None:
    """#ORDER-ERR-NOTIFY: 조치필요 에러 분류 + 카테고리별 쿨다운(중복 억제)."""
    from aurora_ict.bot.order_error_notify import (
        classify_order_error,
        notify_order_error,
    )

    assert classify_order_error('bybit {"retCode":110007}')[0] == "insufficient_balance"
    assert classify_order_error("Permission denied")[0] == "api_permission"
    assert classify_order_error("just a random error") is None

    sent: list[str] = []

    async def _cb(code: str, msg: str) -> None:
        sent.append(msg)

    throttle: dict[str, int] = {}
    # 이벤트루프 없는 동기 컨텍스트 — create_task RuntimeError 경로(no-op) 안전.
    notify_order_error("110007 not enough", _cb, "AICT-X", throttle)
    # 같은 카테고리 재호출 — 쿨다운으로 throttle 기록만 갱신 안 함(이미 있음).
    assert "insufficient_balance" in throttle
    notify_order_error("random", _cb, "AICT-X", throttle)  # 미분류 — no-op
    notify_order_error("110007", None, "AICT-X", throttle)  # cb 없음 — no-op
