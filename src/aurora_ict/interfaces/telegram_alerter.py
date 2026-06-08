"""텔레그램 매매 알림 봇 — 사용자별 라이선스 코드 연동 (2026-06-08, 정용우 영역).

흐름:
- 인바운드(폴링): 사용자가 봇에 자기 라이선스 코드(AICT-XXXX-XXXX-XXXX)를
  보내면 그 chat_id 를 코드에 연결 저장(users_db.set_telegram_chat_id).
- 아웃바운드(발송): 봇 매매 이벤트(_record_trade) 시 그 코드의 chat_id 를 조회해
  알림 발송. 미연동·실패는 조용히 skip — 알림이 매매를 막지 않게.

토큰은 ``AURORA_ICT_TELEGRAM_BOT_TOKEN`` env(fly secret). 미설정이면 비활성.
python-telegram-bot 대신 httpx 로 텔레그램 HTTP API 직접 호출(의존성·제어 단순).
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

import httpx

from aurora_ict.auth import users_db

logger = logging.getLogger(__name__)

_API = "https://api.telegram.org/bot{token}/{method}"

# 라이선스 코드 형식 — AICT-XXXX-XXXX-XXXX (영숫자 4자 3블록).
_CODE_RE = re.compile(
    r"\bAICT-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}\b", re.IGNORECASE,
)

# 이벤트별 (이모지, 한국어 라벨).
_EVENT_LABEL: dict[str, tuple[str, str]] = {
    "entry": ("🟢", "진입"),
    "sl_hit": ("🔴", "손절"),
    "tp_hit": ("✅", "익절"),
    "flip_open": ("🔄", "플립 진입"),
    "flip_close": ("🔁", "플립 청산"),
    "sync_close": ("🔁", "동기화 청산"),
    "manual_close": ("✋", "수동 청산"),
    "recovered": ("♻️", "포지션 복구"),
}


def _enum_val(v: Any) -> str:
    """enum 이면 .value, 아니면 그대로 — 문자열로."""
    return str(getattr(v, "value", v) if v is not None else "")


def format_trade(user_code: str, event: Any) -> str:
    """TradeEvent → 텔레그램 메시지(HTML) 텍스트."""
    etv = _enum_val(getattr(event, "event_type", "")).lower()
    emoji, label = _EVENT_LABEL.get(etv, ("•", etv or "이벤트"))
    direction = _enum_val(getattr(event, "direction", "")).upper()
    sym = getattr(event, "symbol", "") or ""
    price = getattr(event, "price", None)
    qty = getattr(event, "qty", None)
    pnl = getattr(event, "pnl_usdt", None)
    reason = getattr(event, "reason", "") or ""

    lines = [f"{emoji} <b>{label}</b>  {sym}  {direction}".rstrip()]
    if price is not None:
        row = f"가격 {price:,.2f}"
        if qty is not None:
            row += f"  수량 {qty}"
        lines.append(row)
    if pnl is not None:
        sign = "+" if pnl >= 0 else ""
        lines.append(f"손익 {sign}{pnl:,.2f} USDT")
    if reason:
        lines.append(f"<i>{reason}</i>")
    lines.append(f"<code>{user_code}</code>")
    return "\n".join(lines)


class TelegramAlerter:
    """텔레그램 매매 알림 — 폴링(코드 등록) + 발송(매매 알림).

    Args:
        token: 봇 토큰. 빈 문자열이면 비활성(enabled=False).
        db_path: users.db 경로 — chat_id 연동·조회용.
    """

    def __init__(self, token: str, db_path: Path | str) -> None:
        self.token = token or ""
        self.db_path = db_path
        self._offset = 0
        self._client = httpx.AsyncClient(timeout=30.0)

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    async def _call(self, method: str, payload: dict[str, Any]) -> dict[str, Any] | None:
        """텔레그램 API 호출 — 실패 시 None(예외 삼킴)."""
        try:
            r = await self._client.post(
                _API.format(token=self.token, method=method), json=payload,
            )
            return r.json()
        except Exception as e:  # noqa: BLE001
            logger.warning("텔레그램 %s 실패: %s", method, e)
            return None

    async def send(self, chat_id: str, text: str) -> None:
        """단일 메시지 발송 — 실패는 무시."""
        if not self.enabled:
            return
        await self._call(
            "sendMessage",
            {"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
        )

    async def send_trade_alert(self, user_code: str, event: Any) -> None:
        """매매 이벤트 → 연동된 chat_id 로 알림. 미연동·실패 시 조용히 skip."""
        if not self.enabled:
            return
        try:
            chat_id = users_db.get_telegram_chat_id(self.db_path, user_code)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s chat_id 조회 실패(알림 skip): %s", user_code, e)
            return
        if not chat_id:
            return
        await self.send(chat_id, format_trade(user_code, event))

    async def poll_loop(self) -> None:
        """getUpdates 롱폴링 — 라이선스 코드 입력 시 chat_id 연결. 무한 루프."""
        if not self.enabled:
            logger.info("텔레그램 알림 비활성(토큰 미설정) — 폴링 안 함.")
            return
        logger.info("텔레그램 매매 알림 봇 폴링 시작.")
        while True:
            try:
                await self._poll_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:  # noqa: BLE001
                logger.warning("텔레그램 폴링 오류(무시): %s", e)
                await asyncio.sleep(5)

    async def _poll_once(self) -> None:
        resp = await self._call(
            "getUpdates", {"offset": self._offset, "timeout": 25},
        )
        if not resp or not resp.get("ok"):
            await asyncio.sleep(3)
            return
        for upd in resp.get("result", []):
            self._offset = int(upd["update_id"]) + 1
            msg = upd.get("message") or upd.get("edited_message")
            if not msg:
                continue
            chat_id = str(msg["chat"]["id"])
            text = (msg.get("text") or "").strip()
            await self._handle_message(chat_id, text)

    async def _handle_message(self, chat_id: str, text: str) -> None:
        """수신 메시지 처리 — 코드면 연동, 아니면 안내."""
        match = _CODE_RE.search(text)
        if match:
            code = match.group(0).upper()
            try:
                user = users_db.get_user_by_code(self.db_path, code)
            except Exception:  # noqa: BLE001
                user = None
            if user is None:
                await self.send(
                    chat_id, f"❌ 등록되지 않은 코드입니다: <code>{code}</code>",
                )
                return
            users_db.set_telegram_chat_id(self.db_path, code, chat_id)
            await self.send(
                chat_id,
                f"✅ <b>{code}</b> 연동 완료!\n"
                "이제 이 코드의 매매가 발생하면 여기로 알림이 와요.",
            )
            return
        await self.send(
            chat_id,
            "안녕하세요 👋 Aurora-ICT 매매 알림 봇이에요.\n"
            "알림을 받으시려면 본인 <b>라이선스 코드</b>"
            "(AICT-XXXX-XXXX-XXXX)를 그대로 보내주세요.",
        )

    async def aclose(self) -> None:
        """HTTP 클라이언트 정리 — shutdown 시 호출."""
        await self._client.aclose()
