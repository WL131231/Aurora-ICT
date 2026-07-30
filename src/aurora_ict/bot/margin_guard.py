"""진입 전 가용잔고(free margin) 체크 — 필요 증거금 초과 시 수량 자동 축소/skip.

담당: 지영민 (2026-07-24 파트너 지시). 배경: 봇이 여러 페어를 돌 때 한 포지션이
증거금 대부분을 먹으면 다음 페어 진입이 거래소에서 'ab not enough(잔고부족)'로
거부됐다. 진입 직전 가용잔고를 확인해 필요 증거금(=notional/leverage) 이 넘으면
가용에 맞게 수량을 축소하고, 최소주문 미달이면 0 을 반환(호출부가 skip)한다.
거래소 거부(에러·사용자 알림) 를 원천 차단. Origo·Cursus 공용.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 필요 증거금 대비 남길 버퍼 — 수수료·체결 슬리피지·유지증거금 여유(10%).
_MARGIN_BUFFER = 0.90


async def available_usdt(client: Any) -> float:
    """가용(free) USDT 잔고 — 미사용 증거금. 조회 실패/미상 시 -1(=체크 skip 신호).

    ccxt fetch_balance 의 여러 형태 대응: bal["USDT"]["free"] / bal["free"]["USDT"].
    free 가 없으면 total 로 폴백(보수적). 실패면 -1 반환해 호출부가 캡을 건너뛰게 한다.
    """
    try:
        bal = await client.fetch_balance()
    except Exception as e:  # noqa: BLE001
        logger.debug("available_usdt: fetch_balance 실패: %s", e)
        return -1.0
    if not isinstance(bal, dict):
        return -1.0
    usdt = bal.get("USDT")
    if isinstance(usdt, dict):
        free = usdt.get("free")
        if free is not None:
            return float(free)
        total = usdt.get("total")
        if total is not None:
            return float(total)
    free_map = bal.get("free")
    if isinstance(free_map, dict) and free_map.get("USDT") is not None:
        return float(free_map["USDT"])
    return -1.0


async def cap_qty_to_available(
    client: Any, symbol: str, qty: float, price: float, leverage: int,
    min_qty_ratio: float = 0.0,
) -> float:
    """가용잔고로 진입 수량 상한 — 필요증거금 > 가용이면 축소(라운딩), 최소미달이면 0.

    Args:
        client: 거래소 client (fetch_balance / round_amount).
        symbol: 심볼.
        qty: 원 진입 수량.
        price: 진입가(추정).
        leverage: 레버리지 (증거금 = notional/leverage).
        min_qty_ratio: **축소 하한** — 축소된 수량이 원 계획의 이 비율 미만이면
            진입을 포기(0 반환). 0 = 비활성(기존 동작).

            #MIN-SIZE 2026-07-30 (파트너 지시): 한 포지션이 증거금 대부분을 먹은 뒤
            남은 잔고로 **극소액 포지션을 의미 없이 또 잡는** 문제. 라이브 실측
            (Origo 581진입): 최소 notional **0.58 USDT**, 5% 분위 9.31, 일부 유저는
            진입의 **3분의 1**이 중앙값 절반 미만이었다.
            소액은 성적도 나쁘다 — notional 5분위별 ROI 최소구간 **-0.51%**(승률 38%)
            vs 최대구간 -0.25%(승률 48%)로 크기와 단조 관계. 20 USDT 미만 49건은
            pnl 합계 +1.09 로 기여가 사실상 0 이면서 건수 9% 를 차지했다.
            게다가 증거금을 묶어 더 좋은 셋업을 놓치고, 최소주문 근처면 분할익절도
            걸리지 않는다(측정되지 않는 비용).
            **절대 금액 하한은 쓰지 않는다** — 시드가 작은 유저는 정상 진입도
            수십 USDT 라 고정 하한을 걸면 거래 자체가 막힌다. 계획 대비 비율이라
            시드 크기와 무관하게 작동하고, 가용이 충분하면 영향이 없다.

    Returns:
        조정된 수량. 가용 조회 실패(-1)면 원 qty 그대로(거래소가 최종 판단).
        축소분이 최소주문 미달이거나 min_qty_ratio 미달이면 0 → 호출부가 skip.
    """
    if qty <= 0 or price <= 0 or leverage <= 0:
        return qty
    avail = await available_usdt(client)
    if avail < 0:
        return qty  # 조회 실패 — 기존 동작(거래소가 부족 시 거부)
    if avail == 0:
        logger.info("[%s] 가용잔고 0 — 진입 skip", symbol)
        return 0.0
    max_notional = avail * leverage * _MARGIN_BUFFER
    max_qty = max_notional / price
    if qty <= max_qty:
        return qty  # 여유 충분 — 원 수량 유지
    capped = max_qty
    if hasattr(client, "round_amount"):
        try:
            capped = float(client.round_amount(symbol, capped))
        except Exception:  # noqa: BLE001
            pass
    # #MIN-SIZE: 계획의 min_qty_ratio 미만으로 쪼그라들면 진입 자체를 포기.
    # 라운딩 **후** 값으로 판정한다(거래소 최소단위로 내림되어 더 작아질 수 있음).
    if min_qty_ratio > 0 and capped < qty * min_qty_ratio:
        logger.info(
            "[%s] 진입 skip — 가용 %.2f USDT 로는 계획의 %.0f%% 만 진입 가능"
            " (%.6f→%.6f, 하한 %.0f%%). 극소액 포지션 방지 (#MIN-SIZE)",
            symbol, avail, 100 * capped / qty if qty > 0 else 0.0,
            qty, capped, 100 * min_qty_ratio,
        )
        return 0.0
    logger.info(
        "[%s] 가용잔고 %.2f USDT — 진입수량 축소 %.6f→%.6f (필요증거금 초과 방지)",
        symbol, avail, qty, capped,
    )
    return capped  # 최소미달이면 round_amount 가 0 → skip
