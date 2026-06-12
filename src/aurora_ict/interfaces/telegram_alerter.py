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
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import httpx

from aurora_ict.auth import users_db

logger = logging.getLogger(__name__)

# 보안: httpx 는 요청 URL(getUpdates/sendMessage — 봇 토큰 포함)을 INFO 로 로깅해
# fly 로그에 토큰이 평문 노출된다. WARNING 으로 올려 토큰 URL 이 안 찍히게 한다.
logging.getLogger("httpx").setLevel(logging.WARNING)

_API = "https://api.telegram.org/bot{token}/{method}"

# 라이선스 코드 형식 — AICT-XXXX-XXXX-XXXX (영숫자 4자 3블록).
_CODE_RE = re.compile(
    r"\bAICT-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}\b", re.IGNORECASE,
)

# 이벤트별 이모지 (언어 무관).
_EVENT_EMOJI: dict[str, str] = {
    "entry": "🟢",
    "sl_hit": "🔴",
    "tp_hit": "✅",
    "flip_open": "🔄",
    "flip_close": "🔁",
    "sync_close": "🔁",
    "manual_close": "✋",
    "recovered": "♻️",
}

# 언어별 라벨 — UI i18n(ko/en/zh/ja)과 동일 코드 체계. 미지원 언어는 ko fallback.
_LABELS: dict[str, dict[str, Any]] = {
    "ko": {
        "ev": {
            "entry": "진입", "sl_hit": "손절", "tp_hit": "익절",
            "flip_open": "플립 진입", "flip_close": "플립 청산",
            "sync_close": "동기화 청산", "manual_close": "수동 청산",
            "recovered": "포지션 복구", "event": "이벤트",
        },
        "entry_price": "진입가", "tp": "목표가(TP)", "sl": "손절가(SL)",
        "exit_price": "청산가", "qty": "수량", "grade": "등급", "rr": "손익비",
        "reason": "사유", "profit": "수익", "loss": "손실", "change": "변동",
    },
    "en": {
        "ev": {
            "entry": "Entry", "sl_hit": "Stop Loss", "tp_hit": "Take Profit",
            "flip_open": "Flip Entry", "flip_close": "Flip Close",
            "sync_close": "Sync Close", "manual_close": "Manual Close",
            "recovered": "Recovered", "event": "Event",
        },
        "entry_price": "Entry", "tp": "Target(TP)", "sl": "Stop(SL)",
        "exit_price": "Exit", "qty": "Qty", "grade": "Grade", "rr": "RR",
        "reason": "Reason", "profit": "Profit", "loss": "Loss", "change": "Change",
    },
    "zh": {
        "ev": {
            "entry": "进场", "sl_hit": "止损", "tp_hit": "止盈",
            "flip_open": "翻转进场", "flip_close": "翻转平仓",
            "sync_close": "同步平仓", "manual_close": "手动平仓",
            "recovered": "恢复仓位", "event": "事件",
        },
        "entry_price": "进场价", "tp": "目标价(TP)", "sl": "止损价(SL)",
        "exit_price": "平仓价", "qty": "数量", "grade": "等级", "rr": "盈亏比",
        "reason": "原因", "profit": "盈利", "loss": "亏损", "change": "变动",
    },
    "ja": {
        "ev": {
            "entry": "エントリー", "sl_hit": "損切り", "tp_hit": "利確",
            "flip_open": "フリップ進入", "flip_close": "フリップ決済",
            "sync_close": "同期決済", "manual_close": "手動決済",
            "recovered": "ポジション復旧", "event": "イベント",
        },
        "entry_price": "エントリー価", "tp": "目標(TP)", "sl": "損切(SL)",
        "exit_price": "決済価", "qty": "数量", "grade": "グレード", "rr": "RR",
        "reason": "理由", "profit": "利益", "loss": "損失", "change": "変動",
    },
}

_ENTRY_EVENTS = {"entry", "flip_open"}

# 2026-06-12 파트너 요청: 하단 고정 버튼 — 다른 코드로 알림을 옮길 때 사용.
_BTN_RELINK = "🔄 코드 재등록"
_MAIN_KEYBOARD = {
    "keyboard": [[{"text": _BTN_RELINK}]],
    "resize_keyboard": True,
    "is_persistent": True,
}


def _enum_val(v: Any) -> str:
    """enum 이면 .value, 아니면 그대로 — 문자열로."""
    return str(getattr(v, "value", v) if v is not None else "")


def _fmt_price(x: Any) -> str:
    """가격 자릿수 — 크기별 (BTC 6만 vs ONDO 0.36) 가독성."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if abs(v) >= 100:
        return f"{v:,.2f}"
    if abs(v) >= 1:
        return f"{v:,.4f}"
    return f"{v:.6f}"


def _fmt_time(ts_ms: Any, tz: str) -> str:
    """이벤트 시각(UTC ms) → 사용자 시간대 'MM-DD HH:MM'. 실패 시 빈 문자열."""
    try:
        dt = datetime.fromtimestamp(int(ts_ms) / 1000, ZoneInfo(tz))
        return dt.strftime("%m-%d %H:%M")
    except Exception:  # noqa: BLE001
        return ""


def format_trade(
    user_code: str, event: Any, lang: str = "ko", tz: str = "Asia/Seoul",
) -> str:
    """TradeEvent → 텔레그램 메시지(HTML). 사용자 언어/시간대로 상세 출력.

    진입(entry/flip_open): 진입가 · 목표가(TP) · 손절가(SL) · 수량 · 사유(등급/RR/window).
    청산: 손익(수익/손실) · 청산가 · 진입대비 변동% · 사유. 모두 context_json 활용.
    """
    label_set = _LABELS.get(lang) or _LABELS["ko"]
    ev_labels = label_set["ev"]
    etv = _enum_val(getattr(event, "event_type", "")).lower()
    emoji = _EVENT_EMOJI.get(etv, "•")
    label = ev_labels.get(etv, etv or ev_labels["event"])
    direction = _enum_val(getattr(event, "direction", "")).upper()
    sym = getattr(event, "symbol", "") or ""

    ctx: dict[str, Any] = {}
    raw = getattr(event, "context_json", None)
    if raw:
        try:
            ctx = json.loads(raw)
        except (ValueError, TypeError):
            ctx = {}

    head = f"{emoji} <b>{label}</b>"
    if sym:
        head += f"  {sym}"
    if direction:
        head += f"  {direction}"
    lines = [head]

    price = getattr(event, "price", None)
    if etv in _ENTRY_EVENTS:
        entry = ctx.get("entry", price)
        if entry is not None:
            lines.append(f"{label_set['entry_price']} <code>{_fmt_price(entry)}</code>")
        if isinstance(ctx.get("tp"), (int, float)):
            lines.append(f"{label_set['tp']} <code>{_fmt_price(ctx['tp'])}</code>")
        if isinstance(ctx.get("sl"), (int, float)):
            lines.append(f"{label_set['sl']} <code>{_fmt_price(ctx['sl'])}</code>")
        qty = getattr(event, "qty", None)
        if qty:
            lines.append(f"{label_set['qty']} {qty}")
        parts: list[str] = []
        if ctx.get("score") is not None:
            parts.append(f"{label_set['grade']} {ctx['score']}")
        if isinstance(ctx.get("rr"), (int, float)):
            parts.append(f"{label_set['rr']} {ctx['rr']:.2f}")
        if ctx.get("window"):
            parts.append(str(ctx["window"]))
        if parts:
            lines.append(f"{label_set['reason']} {' · '.join(parts)}")
        elif getattr(event, "reason", ""):
            # context_json 이 없으면(구식 이벤트 등) reason 원문 fallback.
            lines.append(f"{label_set['reason']} {event.reason}")
    else:
        # 2026-06-12 파트너 멘트 포맷: 수익 / 청산가+날짜 / 사유 / 코드 를
        # 빈 줄로 구분, 라벨 뒤 " : ". 변동% 줄은 템플릿에서 제외됨.
        pnl = getattr(event, "pnl_usdt", None)
        if isinstance(pnl, (int, float)):
            tag = label_set["profit"] if pnl >= 0 else label_set["loss"]
            sign = "+" if pnl >= 0 else ""
            lines.append(f"{tag} : <b>{sign}{pnl:,.2f} USDT</b>")
        lines.append("")
        cprice = ctx.get("close_price", price)
        if cprice is not None:
            lines.append(
                f"{label_set['exit_price']} : <code>{_fmt_price(cprice)}</code>",
            )
        t = _fmt_time(getattr(event, "ts_ms", 0), tz)
        if t:
            lines.append(f"<i>{t}</i>")
        reason = ctx.get("close_reason") or getattr(event, "reason", "")
        if reason:
            lines.append("")
            lines.append(f"{label_set['reason']} : <i>{reason}</i>")
        lines.append("")
        lines.append(f"<code>{user_code}</code>")
        return "\n".join(lines)

    t = _fmt_time(getattr(event, "ts_ms", 0), tz)
    lines.append(
        f"<i>{t}</i>  <code>{user_code}</code>" if t else f"<code>{user_code}</code>",
    )
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
        # '코드 재등록' 버튼을 누른 채팅 — 다음 코드 수신 시 기존 연동을 풀고
        # 새 코드로 교체한다. (메모리만 — 재시작 시 초기화돼도 무해)
        self._relink_pending: set[str] = set()

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    async def _call(
        self, method: str, payload: dict[str, Any], retries: int = 2,
    ) -> dict[str, Any] | None:
        """텔레그램 API 호출 — 네트워크/429 일시 실패 시 재시도 (알림 누락 방지).

        2026-06-10: 기존엔 1회 실패 시 영구 손실이라 rate limit(429)·타임아웃에
        알림이 사라졌다. 429 는 retry_after, 그 외 예외는 exponential backoff 로
        retries 회 재시도. 최종 실패만 WARNING.
        """
        last_err: Any = None
        for attempt in range(retries + 1):
            try:
                r = await self._client.post(
                    _API.format(token=self.token, method=method), json=payload,
                )
                data = r.json()
                if isinstance(data, dict) and not data.get("ok", True):
                    code = data.get("error_code")
                    last_err = f"error_code={code} {data.get('description', '')}"
                    # 429 rate limit — retry_after 존중(상한 30s, 채팅별 flood
                    # limit 이 10~30s 인 경우 있음). 5xx 도 일시 장애라 재시도
                    # (2026-06-11 리뷰: JSON 5xx 가 무재시도·무로그로 증발하던 것).
                    if code == 429:
                        wait = float(
                            data.get("parameters", {}).get("retry_after", 1),
                        )
                        if attempt < retries:
                            await asyncio.sleep(min(wait, 30.0))
                            continue
                    elif isinstance(code, int) and code >= 500:
                        if attempt < retries:
                            await asyncio.sleep(1.5 * (attempt + 1))
                            continue
                    # 비재시도 오류(400 등) 또는 재시도 소진 — 아래 warning 으로.
                    break
                return data
            except Exception as e:  # noqa: BLE001
                last_err = e
                if attempt < retries:
                    await asyncio.sleep(1.5 * (attempt + 1))  # backoff
                    continue
        logger.warning(
            "텔레그램 %s 실패(재시도 소진 또는 비재시도 오류): %s", method, last_err,
        )
        return None

    async def send(
        self, chat_id: str, text: str, *, keyboard: bool = False,
    ) -> None:
        """단일 메시지 발송 — 실패는 무시.

        Args:
            keyboard: True 면 하단 고정 키보드('코드 재등록') 첨부 — 봇과의
                대화성 응답에만 붙인다 (매매 알림에는 미첨부, 소음 방지).
        """
        if not self.enabled:
            return
        payload: dict[str, Any] = {
            "chat_id": chat_id, "text": text, "parse_mode": "HTML",
        }
        if keyboard:
            payload["reply_markup"] = _MAIN_KEYBOARD
        await self._call("sendMessage", payload)

    async def send_trade_alert(self, user_code: str, event: Any) -> None:
        """매매 이벤트 → 연동된 chat_id 로 알림. 미연동·실패 시 조용히 skip.

        사용자 언어/시간대(users_db.get_user_prefs)로 메시지를 포맷한다.
        미설정 시 ko / Asia/Seoul fallback.
        """
        if not self.enabled:
            return
        try:
            chat_id = users_db.get_telegram_chat_id(self.db_path, user_code)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s chat_id 조회 실패(알림 skip): %s", user_code, e)
            return
        if not chat_id:
            return
        lang, tz = "ko", "Asia/Seoul"
        try:
            prefs = users_db.get_user_prefs(self.db_path, user_code)
            lang = prefs.get("language") or "ko"
            tz = prefs.get("timezone") or "Asia/Seoul"
        except Exception as e:  # noqa: BLE001
            logger.debug("%s prefs 조회 실패(기본값 사용): %s", user_code, e)
        # format_trade 예외로 알림 통째 누락 방지 — 실패 시 최소 정보 fallback.
        try:
            text = format_trade(user_code, event, lang, tz)
        except Exception as e:  # noqa: BLE001
            logger.warning("%s format_trade 실패(fallback 발송): %s", user_code, e)
            etv = _enum_val(getattr(event, "event_type", "")).upper()
            sym = getattr(event, "symbol", "") or ""
            text = f"매매 {etv} {sym}\n<code>{user_code}</code>"
        await self.send(chat_id, text)

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

    async def restart(self) -> None:
        """봇 자체 재시작 — HTTP 클라이언트 재생성 + getUpdates 오프셋 리셋.

        2026-06-12 파트너 요청: 알림 봇이 먹통일 때 텔레그램에서 /restart 로
        즉시 복구. 폴링 루프는 그대로 두고(태스크 유지) 연결만 새로 만든다 —
        fly 프로세스 안에서 돌아 프로세스 재실행은 서버 전체 재시작이 되기 때문.
        """
        old = self._client
        self._client = httpx.AsyncClient(timeout=30.0)
        # 2026-06-12 리뷰 #1: offset 은 절대 리셋하지 않는다 — 0 으로 돌리면
        # 텔레그램이 /restart 메시지를 재배달해 재시작 무한루프(스팸)가 된다.
        try:
            await old.aclose()
        except Exception as e:  # noqa: BLE001
            logger.debug("restart: 구 클라이언트 정리 실패(무시): %s", e)
        logger.info("텔레그램 알림 봇 재시작 — 클라이언트 재생성 (offset 유지).")

    async def _handle_message(self, chat_id: str, text: str) -> None:
        """수신 메시지 처리 — 코드면 연동, 재등록 버튼이면 교체 모드, 아니면 안내."""
        # 2026-06-12 파트너: /restart — 봇 연결 자체 재시작 (먹통 복구용).
        if text.strip().lower().startswith("/restart"):
            await self.restart()
            await self.send(
                chat_id, "🔄 알림 봇 재시작 완료 — 연결을 새로 만들었어요.",
                keyboard=True,
            )
            return
        # 2026-06-12: 하단 '코드 재등록' 버튼 — 다음 코드 수신 시 이 채팅의
        # 기존 연동을 모두 풀고 새 코드로 교체.
        if _BTN_RELINK in text or text == "/relink":
            self._relink_pending.add(chat_id)
            await self.send(
                chat_id,
                "🔄 <b>코드 재등록</b>\n"
                "새 <b>라이선스 코드</b>(AICT-XXXX-XXXX-XXXX)를 보내주세요.\n"
                "기존 연동은 해제되고 새 코드로 알림이 옵니다.",
                keyboard=True,
            )
            return
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
                    keyboard=True,
                )
                return
            relinked = chat_id in self._relink_pending
            if relinked:
                # 교체 모드 — 이 채팅에 묶인 기존 코드 연동을 먼저 해제.
                self._relink_pending.discard(chat_id)
                try:
                    users_db.clear_telegram_chat_id_by_chat(self.db_path, chat_id)
                except Exception as e:  # noqa: BLE001
                    logger.warning("재등록 기존 연동 해제 실패(계속 진행): %s", e)
            users_db.set_telegram_chat_id(self.db_path, code, chat_id)
            head = "🔄 재등록 완료!" if relinked else "연동 완료!"
            await self.send(
                chat_id,
                f"✅ <b>{code}</b> {head}\n"
                "이제 이 코드의 매매가 발생하면 여기로 알림이 와요.",
                keyboard=True,
            )
            return
        await self.send(
            chat_id,
            "안녕하세요 👋 Aurora-ICT 매매 알림 봇이에요.\n"
            "알림을 받으시려면 본인 <b>라이선스 코드</b>"
            "(AICT-XXXX-XXXX-XXXX)를 그대로 보내주세요.",
            keyboard=True,
        )

    async def aclose(self) -> None:
        """HTTP 클라이언트 정리 — shutdown 시 호출."""
        await self._client.aclose()
