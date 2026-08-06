"""#SIDE — 거래소 side 파싱 통일. 방향 오판은 손절을 반대편에 건다.

#CLOSE-500 을 고치다 발견: 방향 판정이 코드 곳곳에 흩어져 제각각이었고, 일부는
**인식 실패를 임의 방향으로 단정**했다.

    ccxt_client._parse_position (구):
        side = "short" if side_raw == "short" else "long"
        → "sell" 이 오면 **숏을 롱으로 읽는다**

    closed-pnl 파서 (구):
        direction = "long" if side_raw == "Sell" else "short"
        → 대문자 "Sell" 만 비교. 소문자 "sell" 이면 방향이 통째로 뒤집힌다

거래소·엔드포인트마다 표기가 다르다(ccxt position 은 long/short, Bybit 주문은
Buy/Sell, 일부 응답은 소문자). `normalize_side` 하나로 모으고, 인식 실패는
None 을 돌려 호출측이 판단을 보류하게 한다.
"""

from __future__ import annotations

import pytest

from aurora.exchange.base import normalize_side


@pytest.mark.parametrize(
    ("raw", "expect"),
    [
        # ccxt position 표기
        ("long", "long"), ("short", "short"),
        # Bybit 주문 표기 (대문자 시작)
        ("Buy", "long"), ("Sell", "short"),
        # 소문자 변형
        ("buy", "long"), ("sell", "short"),
        # 대문자·공백 섞임
        ("LONG", "long"), ("  Short  ", "short"),
        # 인식 불가 — 반드시 None (임의 방향 금지)
        ("", None), ("none", None), ("both", None), ("weird", None),
        (None, None), (0, None), ([], None),
    ],
)
def test_normalize_side(raw, expect) -> None:
    assert normalize_side(raw) is expect


def test_sell_is_not_long() -> None:
    """★ 구 버그 재현 방지 — "sell" 을 롱으로 읽으면 안 된다."""
    assert normalize_side("sell") == "short"
    assert normalize_side("Sell") == "short"


def test_buy_is_not_short() -> None:
    """반대편도 — "buy" 를 숏으로 읽으면 안 된다."""
    assert normalize_side("buy") == "long"
    assert normalize_side("Buy") == "long"


def test_unknown_never_guesses() -> None:
    """인식 실패는 None — 이걸 임의 방향으로 대체하면 사고가 난다."""
    for bad in ("", "  ", "n/a", "unknown", "flat", None):
        assert normalize_side(bad) is None


def test_bots_use_same_contract() -> None:
    """Origo·Cursus 의 방향 헬퍼가 normalize_side 와 같은 답을 낸다."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance
    from aurora_ict.bot.bot_trend_instance import BotTrendInstance
    from aurora_ict.strategy.silver_bullet import Direction

    want = {"long": Direction.LONG, "short": Direction.SHORT, None: None}
    for raw in ("long", "buy", "short", "sell", "Buy", "Sell", "", "junk"):
        exp = want[normalize_side(raw)]
        assert BotIctInstance._exchange_position_direction({"side": raw}) is exp
        assert BotTrendInstance._exchange_position_direction({"side": raw}) is exp
