"""진입 전 가용잔고(free margin) 체크 — 필요 증거금 초과 시 수량 자동 축소/skip.

담당: 지영민 (2026-07-24 파트너 지시). 배경: 봇이 여러 페어를 돌 때 한 포지션이
증거금 대부분을 먹으면 다음 페어 진입이 거래소에서 'ab not enough(잔고부족)'로
거부됐다. 진입 직전 가용잔고를 확인해 필요 증거금(=notional/leverage) 이 넘으면
가용에 맞게 수량을 축소하고, 최소주문 미달이면 0 을 반환(호출부가 skip)한다.
Origo·Cursus 공용. 조회 이후 다른 주문이나 가격 변동에 의한 거래소 거절은 가능하다.

2026-09-13 Codex: Bybit 통합계정의 미실현손익 누락 보정. 가용액 미확인 시
신규 진입을 보류하며, 총잔고를 가용액으로 대체하지 않는다. 청산에는 적용하지 않는다.
"""
from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# 필요 증거금 대비 남길 버퍼 — 수수료·체결 슬리피지·유지증거금 여유(10%).
_MARGIN_BUFFER = 0.90


def _finite_number(value: Any) -> float | None:
    """거래소 숫자를 유한 실수로 변환한다.

    Args:
        value: 숫자 또는 숫자 문자열.
    Returns:
        유한 실수. 누락, 불리언, 비정상 값이면 None.
    Raises:
        없음.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def parse_available_usdt(balance: Any) -> float | None:
    """일반 CCXT 응답에서 명시된 가용액만 읽는다.

    Args:
        balance: CCXT 잔고 응답. Bybit V5 원본은 전용 해석이 필요하다.
    Returns:
        0 이상 가용 USDT. 미확인 또는 잘못된 값이면 None.
    Raises:
        없음.
    """
    if not isinstance(balance, dict):
        return None
    info = balance.get("info")
    if isinstance(info, dict) and isinstance(info.get("result"), dict):
        if "list" in info["result"]:
            return None  # Bybit V5는 CCXT free 대신 계정 모드와 원본으로 해석한다.
    values = []
    usdt = balance.get("USDT")
    if isinstance(usdt, dict):
        values.append(usdt.get("free"))
    free_map = balance.get("free")
    if isinstance(free_map, dict):
        values.append(free_map.get("USDT"))
    parsed = []
    for value in values:
        if value is None or value == "":
            continue
        number = _finite_number(value)
        if number is None:
            return None
        parsed.append(number)
    return max(0.0, min(parsed)) if parsed else None


def parse_bybit_available_usdt(balance: Any, margin_mode: str | None) -> float | None:
    """Bybit 통합계정의 주문 가용액을 USDT로 해석한다.

    Args:
        balance: 원본 info.result.list가 포함된 CCXT 잔고 응답.
        margin_mode: /v5/account/info에서 확인한 증거금 모드.
    Returns:
        0 이상 가용 USDT. 필수 값이나 계정 모드를 확인하지 못하면 None.
    Raises:
        없음.
    """
    if not isinstance(balance, dict):
        return None
    info = balance.get("info")
    if (
        not isinstance(info, dict)
        or isinstance(info.get("retCode"), bool)
        or info.get("retCode") not in (0, "0")
    ):
        return None
    result = info.get("result")
    accounts = result.get("list") if isinstance(result, dict) else None
    if not isinstance(accounts, list) or len(accounts) != 1:
        return None
    account = accounts[0]
    if not isinstance(account, dict) or account.get("accountType") != "UNIFIED":
        return None
    if margin_mode not in ("REGULAR_MARGIN", "PORTFOLIO_MARGIN", "ISOLATED_MARGIN"):
        return None
    coins = account.get("coin")
    if not isinstance(coins, list):
        return None
    usdt_coins = [c for c in coins if isinstance(c, dict) and c.get("coin") == "USDT"]
    if margin_mode in ("REGULAR_MARGIN", "PORTFOLIO_MARGIN"):
        available_usd = _finite_number(account.get("totalAvailableBalance"))
        if available_usd is None:
            return None
        if available_usd <= 0:
            return 0.0
        if len(usdt_coins) != 1:
            return None
        coin = usdt_coins[0]
        equity = _finite_number(coin.get("equity"))
        usd_value = _finite_number(coin.get("usdValue"))
        if equity is None or usd_value is None or equity == 0:
            return None
        # USD와 USDT는 다른 단위다. 같은 잔고 응답의 자산 평가 비율로 환산한다.
        usd_per_usdt = _finite_number(usd_value / equity)
        if usd_per_usdt is None or usd_per_usdt <= 0:
            return None
        return _finite_number(available_usd / usd_per_usdt)

    if len(usdt_coins) != 1:
        return None
    coin = usdt_coins[0]
    wallet = _finite_number(coin.get("walletBalance"))
    deductions = [
        _finite_number(coin.get(key))
        for key in ("totalPositionIM", "totalOrderIM", "locked", "bonus")
    ]
    # 새 walletBalance에는 현물 차입이 포함될 수 있으므로 이미 빌린 금액은 제외한다.
    deductions.append(_finite_number(coin.get("spotBorrow", 0)))
    if wallet is None or any(value is None or value < 0 for value in deductions):
        return None
    available = _finite_number(wallet - sum(deductions))
    return max(0.0, available) if available is not None else None


async def available_usdt(client: Any) -> float | None:
    """진입용 가용액을 조회한다. 총잔고나 오래된 값으로 대체하지 않는다.

    Args:
        client: 거래소 어댑터 또는 fetch_balance를 제공하는 클라이언트.
    Returns:
        0 이상 가용 USDT. 조회 실패나 해석 불가 시 None.
    Raises:
        취소 예외는 호출자에게 전파한다.
    """
    try:
        # 명시된 어댑터 계약만 선택해 동적 속성 생성 객체와 혼동하지 않는다.
        if callable(getattr(type(client), "fetch_available_usdt", None)):
            value = _finite_number(await client.fetch_available_usdt())
            return max(0.0, value) if value is not None else None
        return parse_available_usdt(await client.fetch_balance())
    except Exception as e:  # noqa: BLE001
        logger.warning("진입 가용액 조회 실패 (%s) — 신규 진입 보류", type(e).__name__)
        return None


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
        조정된 수량. 가용액을 확인하지 못하거나 0 이하면 0 (신규 진입 보류).
        축소분이 최소주문 미달이거나 min_qty_ratio 미달이면 0 → 호출부가 skip.
    Raises:
        취소 예외는 호출자에게 전파한다.
    """
    numbers = [_finite_number(value) for value in (qty, price, leverage)]
    if any(value is None or value <= 0 for value in numbers):
        return 0.0
    qty, price, leverage = numbers
    avail = await available_usdt(client)
    if avail is None:
        logger.warning("[%s] 가용잔고 미확인 — 신규 진입 보류", symbol)
        return 0.0
    if avail == 0:
        logger.info("[%s] 가용잔고 0 — 진입 skip", symbol)
        return 0.0
    max_notional = avail * leverage * _MARGIN_BUFFER
    max_qty = max_notional / price
    if not math.isfinite(max_qty) or max_qty <= 0:
        return 0.0
    if qty <= max_qty:
        return qty  # 여유 충분 — 원 수량 유지
    capped = max_qty
    if hasattr(client, "round_amount"):
        try:
            capped = float(client.round_amount(symbol, capped))
        except Exception:  # noqa: BLE001
            logger.warning("[%s] 진입 수량 정렬 실패 — 신규 진입 보류", symbol)
            return 0.0
    if not math.isfinite(capped) or capped <= 0 or capped > max_qty:
        return 0.0
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
