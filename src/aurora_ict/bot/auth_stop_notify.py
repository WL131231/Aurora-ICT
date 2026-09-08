"""거래소 키 만료/무효 자동 정지 안내 — **사용자당 1회** (2026-09-09, 장수 영역).

9/9 실측: 키 만료로 페어 6개가 동시에 자동 정지되면서 안내가 **6통** 갔다
(봇 인스턴스가 페어마다 하나라 각자 보냄). 사유는 계정 하나에 하나이므로
사용자 기준으로 한 번만 보낸다.

프로세스 전역 레지스트리 ``(user_code, kind) → 마지막 발송 시각`` 으로 묶는다.
같은 사유로 6시간 안에 다시 오면 보내지 않는다 — 키 재등록 후 다시 만료되는
건 90일 뒤라 6시간이면 충분하고, 사용자가 STOP→START 를 반복해도 도배되지 않는다.
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# 같은 (사용자, 사유) 재발송 억제 시간 — 6시간.
COOLDOWN_SEC = 6 * 60 * 60

_LAST_SENT: dict[tuple[str, str], float] = {}


def build_message(kind: str) -> str:
    """사유별 안내문(HTML). 페어를 적지 않는다 — 계정 단위 사유라서.

    Args:
        kind: ``"expired"`` | ``"invalid"``.
    """
    if kind == "expired":
        head = (
            "⚠ 거래소 API 키가 <b>만료</b>됐어요 (Bybit 는 IP 제한 없는 키를 90일 뒤 "
            "만료시킵니다).\n"
        )
    else:
        head = "⚠ 거래소 API 키가 <b>유효하지 않아요</b>(만료·삭제·오타 가능).\n"
    return (
        head
        + "가동 중인 봇을 모두 자동 정지했습니다.\n"
        "Bybit → API 관리 → 새 키 발급(선물 거래·잔고 조회 권한) → Aurora 에서 "
        "API 키 재등록 → STOP 후 START 해 주세요. 열린 포지션은 Bybit 에서 직접 "
        "확인해 주세요."
    )


async def notify_auth_stop(
    notify_cb: Any, user_code: str | None, kind: str, *, now: float | None = None,
) -> bool:
    """키 만료/무효 안내를 사용자당 1회 보낸다.

    Args:
        notify_cb: ``async (user_code, text)`` — 텔레그램 발송 콜백. None 이면 안 보냄.
        user_code: 대상 사용자. 비어 있으면 안 보냄.
        kind: ``"expired"`` | ``"invalid"``.
        now: 테스트용 현재 시각(초). 기본 ``time.monotonic()``.

    Returns:
        실제로 보냈으면 True. 쿨다운·콜백 없음·실패면 False (정지는 막지 않는다).
    """
    if notify_cb is None or not user_code:
        return False
    t = time.monotonic() if now is None else now
    key = (user_code, kind)
    last = _LAST_SENT.get(key)
    if last is not None and (t - last) < COOLDOWN_SEC:
        return False
    _LAST_SENT[key] = t            # 실패해도 기록 — 실패 시 페어마다 재시도해 도배되는 것 방지
    try:
        await notify_cb(user_code, build_message(kind))
        return True
    except Exception as e:  # noqa: BLE001 — 알림 실패가 정지를 막지 않게
        logger.warning("키 %s 안내 발송 실패 (%s): %s", kind, user_code, e)
        return False


def reset_for_tests() -> None:
    """레지스트리 초기화 — 테스트 격리용."""
    _LAST_SENT.clear()
