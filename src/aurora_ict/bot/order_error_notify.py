"""주문 실패 에러 → 사용자 조치 안내 메시지 분류 + 텔레그램(notify_cb) 발송.

담당: 지영민 (2026-07-23 파트너 요청). 잔고부족·API키 권한·약관 미동의 등 '사용자만
풀 수 있는' 거래소 거부를, 연동 텔레그램으로 해당 사용자에게 직접 안내한다.
카테고리별 쿨다운으로 매 step 도배를 방지(봇 인스턴스가 throttle dict 보유).
Origo(BotIctInstance)·Cursus(BotTrendInstance) 공용.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

# 카테고리별 재안내 최소 간격 — 같은 문제로 매 step 도배 방지 (잔고는 채워질 때까지
# 계속 실패하므로). 6시간마다 최대 1회.
_ALERT_COOLDOWN_MS = 6 * 3600 * 1000


def classify_order_error(err_s: str) -> tuple[str, str] | None:
    """거래소 주문실패 문자열 → (kind, 사용자 안내 메시지 HTML). 조치불가/불명이면 None.

    Args:
        err_s: 거래소/ccxt 예외 문자열.

    Returns:
        (분류 키, 텔레그램 안내 메시지) 또는 조치 안내 대상 아님 시 None.
    """
    s = err_s.lower()
    if "110123" in err_s or "trading terms" in s:
        return (
            "terms",
            "⚠ 주문이 거래소에서 거절되고 있어요.\nBybit 정책상 이 컨트랙트는 "
            "<b>웹/앱에서 약관 동의 1회</b>가 필요합니다. Bybit 에서 해당 페어 주문 "
            "화면을 열어 약관에 동의해 주세요 — 동의 후 봇이 자동으로 다시 매매합니다.",
        )
    if "110007" in err_s or "not enough" in s or "insufficient" in s:
        return (
            "insufficient_balance",
            "⚠ 주문이 <b>잔고 부족</b>으로 거절됐어요.\n계좌 가용 잔고를 확인해 "
            "충전하시거나, 레버리지·진입 비중을 낮춰 주세요. 잔고가 채워지면 봇이 "
            "자동으로 다시 진입합니다.",
        )
    if "10005" in err_s or "permission denied" in s or (
        "api" in s and "permission" in s
    ):
        return (
            "api_permission",
            "⚠ 거래소 API 키 <b>권한 부족</b>으로 주문/취소가 거절되고 있어요.\n"
            "Bybit API 키 설정에서 <b>거래(Trade)·포지션</b> 권한이 켜져 있는지 "
            "확인해 주세요.",
        )
    if "10003" in err_s or "api key is invalid" in s or "invalid api" in s:
        return (
            "api_key_invalid",
            "⚠ 거래소 API 키가 <b>유효하지 않아요</b>(만료·삭제·오타 가능).\n"
            "Aurora 설정에서 API 키를 다시 등록해 주세요.",
        )
    return None


async def _send(notify_cb: Any, user_code: str, msg: str) -> None:
    try:
        await notify_cb(user_code, msg)
    except Exception as e:  # noqa: BLE001
        logger.warning("주문에러 안내 발송 실패 — %s: %s", user_code, e)


def notify_order_error(
    err_s: str, notify_cb: Any, user_code: str, throttle: dict[str, int],
) -> None:
    """조치필요 에러면 카테고리별 쿨다운 후 notify_cb 로 안내(비동기 태스크 발송).

    Args:
        err_s: 거래소 예외 문자열.
        notify_cb: async (user_code, msg) 안내 콜백 (없으면 no-op).
        user_code: 대상 사용자 코드 (빈 값이면 no-op).
        throttle: {kind: last_sent_ms} — 봇 인스턴스가 보유(재시작 시 리셋 OK).
    """
    if notify_cb is None or not user_code:
        return
    hit = classify_order_error(err_s)
    if hit is None:
        return
    kind, msg = hit
    now_ms = int(time.time() * 1000)
    if now_ms - throttle.get(kind, 0) < _ALERT_COOLDOWN_MS:
        return
    throttle[kind] = now_ms
    # 이벤트 루프가 있을 때만 발송 태스크 생성(동기 테스트서 코루틴 미await 경고 방지).
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(_send(notify_cb, user_code, msg))
    task.add_done_callback(lambda t: t.exception())  # 예외 회수(경고 억제)
