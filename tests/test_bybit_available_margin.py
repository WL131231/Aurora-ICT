"""Bybit 가용 증거금의 단위와 계정 모드를 합성 잔고로 검증한다.

담당: Codex. 실계좌 정보 없이 순수 파서와 실제 CCXT 잔고 파서를 오프라인으로 실행한다.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import ccxt
import pytest

from aurora_ict.bot.margin_guard import parse_available_usdt, parse_bybit_available_usdt


def _balance(available: Any = "99", **coin_values: Any) -> dict[str, Any]:
    """평가손실과 USD 환산율을 가진 합성 UNIFIED 잔고를 만든다.

    Args:
        available: 계정 단위 USD 가용액.
        coin_values: 기본 USDT 항목에서 덮어쓸 값.

    Returns:
        CCXT 통합 잔고와 Bybit 원시 응답을 함께 가진 독립 사전.
    """
    coin = {
        "coin": "USDT",
        "walletBalance": "1000",
        "equity": "800",
        "usdValue": "792",
        "unrealisedPnl": "-200",
        "totalPositionIM": "400",
        "totalOrderIM": "300",
        "locked": "0",
        "bonus": "0",
        "availableToWithdraw": "",
    }
    coin.update(coin_values)
    return {
        "USDT": {"free": 300.0, "used": 700.0, "total": 1000.0},
        "free": {"USDT": 300.0},
        "info": {
            "retCode": 0,
            "result": {
                "list": [
                    {
                        "accountType": "UNIFIED",
                        "totalAvailableBalance": available,
                        "coin": [coin],
                    }
                ]
            },
        },
    }


@pytest.mark.parametrize(
    ("balance", "expected"),
    [
        ({"USDT": {"free": "12.5", "total": 1000}}, 12.5),
        ({"free": {"USDT": "12.5"}}, 12.5),
        ({"USDT": {"free": None, "total": 1000}, "free": {"USDT": 12.5}}, 12.5),
        ({"USDT": {"total": 1000}, "free": {"USDT": 12.5}}, 12.5),
        ({"USDT": {"free": 12}, "free": {"USDT": 8}}, 8.0),
        ({"USDT": {"free": 8}, "free": {"USDT": 12}}, 8.0),
        ({"USDT": {"free": 0}, "free": {"USDT": 12}}, 0.0),
        ({"USDT": {"free": -5}, "free": {"USDT": 12}}, 0.0),
        ({"free": {"USDT": -5}}, 0.0),
        ({"USDT": {"free": "-0.0"}}, 0.0),
    ],
)
def test_generic_free_only(balance: Any, expected: float) -> None:
    assert parse_available_usdt(balance) == expected


@pytest.mark.parametrize(
    "balance",
    [
        None,
        [],
        "100",
        {},
        {"USDT": {"total": 1000}},
        {"USDT": {"free": None, "total": 1000}},
        {"total": {"USDT": 1000}},
        {"USD": {"free": 1000}},
        {"USDT": [], "free": []},
    ],
)
def test_generic_missing_free_never_uses_total(balance: Any) -> None:
    assert parse_available_usdt(balance) is None


@pytest.mark.parametrize("value", [None, "", "bad", "NaN", "inf", "-inf", True, False, [], {}])
@pytest.mark.parametrize("nested", [True, False])
def test_generic_rejects_nonfinite_or_nonnumeric_free(value: Any, nested: bool) -> None:
    balance = {"USDT": {"free": value}} if nested else {"free": {"USDT": value}}
    assert parse_available_usdt(balance) is None


def test_generic_does_not_trust_unified_ccxt_free() -> None:
    assert parse_available_usdt(_balance()) is None


@pytest.mark.parametrize("value", ["bad", "NaN", "inf", True, False])
def test_generic_rejects_invalid_free_even_with_valid_companion(value: Any) -> None:
    balance = {"USDT": {"free": value}, "free": {"USDT": 100}}
    assert parse_available_usdt(balance) is None


@pytest.mark.parametrize("mode", ["REGULAR_MARGIN", "PORTFOLIO_MARGIN"])
def test_account_available_includes_loss_and_converts_usd_to_usdt(mode: str) -> None:
    balance = _balance()
    assert parse_bybit_available_usdt(balance, mode) == pytest.approx(100.0)
    assert balance["USDT"]["free"] == 300.0


def test_cross_available_includes_unrealised_profit() -> None:
    balance = _balance("495", equity="1200", usdValue="1188", unrealisedPnl="200")
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") == pytest.approx(500.0)


def test_cross_available_is_not_capped_at_usdt_wallet_for_multiple_collateral() -> None:
    balance = _balance("1980")
    balance["info"]["result"]["list"][0]["coin"].append(
        {"coin": "BTC", "walletBalance": "0.1", "equity": "0.1", "usdValue": "8000"}
    )
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") == pytest.approx(2000.0)


def test_portfolio_does_not_require_coin_position_or_order_margin() -> None:
    balance = _balance(totalPositionIM="", totalOrderIM="")
    assert parse_bybit_available_usdt(balance, "PORTFOLIO_MARGIN") == pytest.approx(100.0)


def test_cross_ignores_deprecated_available_to_withdraw_zero() -> None:
    balance = _balance(availableToWithdraw="0")
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") == pytest.approx(100.0)


def test_cross_negative_equity_can_still_supply_positive_conversion_rate() -> None:
    balance = _balance(equity="-800", usdValue="-792")
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") == pytest.approx(100.0)


@pytest.mark.parametrize("available", ["0", "-5", 0, -5.0])
@pytest.mark.parametrize("mode", ["REGULAR_MARGIN", "PORTFOLIO_MARGIN"])
def test_nonpositive_account_available_is_zero_without_exchange_rate(
    available: Any, mode: str,
) -> None:
    balance = _balance(available, equity="", usdValue="")
    assert parse_bybit_available_usdt(balance, mode) == 0.0


@pytest.mark.parametrize("value", [None, "", "bad", "NaN", "inf", "-inf", True, False, [], {}])
def test_cross_invalid_account_available_never_falls_back_to_ccxt(value: Any) -> None:
    assert parse_bybit_available_usdt(_balance(value), "REGULAR_MARGIN") is None


@pytest.mark.parametrize("field", ["equity", "usdValue"])
@pytest.mark.parametrize("value", [None, "", "0", "bad", "NaN", "inf", "-inf", True, False])
def test_cross_requires_valid_nonzero_conversion_components(field: str, value: Any) -> None:
    assert parse_bybit_available_usdt(_balance(**{field: value}), "REGULAR_MARGIN") is None


@pytest.mark.parametrize(
    ("equity", "usd_value"),
    [("800", "-792"), ("-800", "792"), ("1e-308", "1e308"), ("1e308", "1e-308")],
)
def test_cross_rejects_invalid_conversion_rate(equity: str, usd_value: str) -> None:
    balance = _balance(equity=equity, usdValue=usd_value)
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") is None


def test_cross_rejects_nonfinite_converted_available() -> None:
    balance = _balance("1e308", equity="1e100", usdValue="1")
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") is None


@pytest.mark.parametrize("mode", [None, "", "CROSS_MARGIN", "regular_margin", "REGULAR_MARGIN "])
def test_unknown_margin_mode_does_not_assume_cross(mode: str | None) -> None:
    assert parse_bybit_available_usdt(_balance(), mode) is None


@pytest.mark.parametrize(
    "balance",
    [
        None,
        [],
        {},
        {"USDT": {"free": 1000}},
        {"info": None},
        {"info": {"result": None}},
        {"info": {"result": {"list": None}}},
        {"info": {"result": {"list": []}}},
        {"info": {"result": {"list": {}}}},
        {"info": {"result": {"list": [None]}}},
    ],
)
def test_bybit_requires_structured_raw_unified_account(balance: Any) -> None:
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") is None


def test_bybit_rejects_non_unified_account() -> None:
    balance = _balance()
    balance["info"]["result"]["list"][0]["accountType"] = "CONTRACT"
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") is None


def test_bybit_rejects_ambiguous_unified_accounts() -> None:
    balance = _balance()
    accounts = balance["info"]["result"]["list"]
    accounts.append(deepcopy(accounts[0]))
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") is None


@pytest.mark.parametrize("ret_code", [None, 1, "110007", True, False, [], {}, "NaN", "inf"])
def test_bybit_requires_success_code(ret_code: Any) -> None:
    balance = _balance()
    balance["info"]["retCode"] = ret_code
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") is None


def test_bybit_requires_present_success_code() -> None:
    balance = _balance()
    del balance["info"]["retCode"]
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") is None


def test_bybit_accepts_string_success_code() -> None:
    balance = _balance()
    balance["info"]["retCode"] = "0"
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") == pytest.approx(100.0)


@pytest.mark.parametrize("mode", ["REGULAR_MARGIN", "PORTFOLIO_MARGIN", "ISOLATED_MARGIN"])
def test_bybit_rejects_duplicate_usdt_coins(mode: str) -> None:
    balance = _balance()
    coins = balance["info"]["result"]["list"][0]["coin"]
    coins.append(deepcopy(coins[0]))
    assert parse_bybit_available_usdt(balance, mode) is None


@pytest.mark.parametrize("coins", [None, {}, [], [None], [{"coin": "USDC"}]])
def test_positive_cross_available_requires_usdt_coin(coins: Any) -> None:
    balance = _balance()
    balance["info"]["result"]["list"][0]["coin"] = coins
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") is None


def test_isolated_deducts_only_reserved_margin_locked_and_bonus() -> None:
    balance = _balance("9999", locked="20", bonus="10", equity="", usdValue="")
    assert parse_bybit_available_usdt(balance, "ISOLATED_MARGIN") == pytest.approx(270.0)


def test_isolated_deducts_reported_spot_borrow() -> None:
    balance = _balance(spotBorrow="50")
    assert parse_bybit_available_usdt(balance, "ISOLATED_MARGIN") == pytest.approx(250.0)


@pytest.mark.parametrize("wallet", ["700", "650", "-100"])
def test_isolated_nonpositive_free_is_zero(wallet: str) -> None:
    assert parse_bybit_available_usdt(_balance(walletBalance=wallet), "ISOLATED_MARGIN") == 0.0


@pytest.mark.parametrize(
    "field", ["walletBalance", "totalPositionIM", "totalOrderIM", "locked", "bonus"]
)
def test_isolated_requires_each_formula_field(field: str) -> None:
    balance = _balance()
    del balance["info"]["result"]["list"][0]["coin"][0][field]
    assert parse_bybit_available_usdt(balance, "ISOLATED_MARGIN") is None


@pytest.mark.parametrize(
    "field", ["walletBalance", "totalPositionIM", "totalOrderIM", "locked", "bonus", "spotBorrow"]
)
@pytest.mark.parametrize("value", [None, "", "NaN", "inf", "bad", True, False])
def test_isolated_rejects_malformed_formula_fields(field: str, value: Any) -> None:
    balance = _balance(**{field: value})
    assert parse_bybit_available_usdt(balance, "ISOLATED_MARGIN") is None


@pytest.mark.parametrize("field", ["totalPositionIM", "totalOrderIM", "locked", "bonus", "spotBorrow"])
def test_isolated_rejects_negative_deductions(field: str) -> None:
    balance = _balance(**{field: "-1"})
    assert parse_bybit_available_usdt(balance, "ISOLATED_MARGIN") is None


def test_isolated_rejects_nonfinite_total_deductions() -> None:
    balance = _balance(totalPositionIM="1e308", totalOrderIM="1e308")
    assert parse_bybit_available_usdt(balance, "ISOLATED_MARGIN") is None


@pytest.mark.parametrize("available_to_withdraw", ["", "0"])
def test_actual_ccxt_parser_preserves_raw_available_for_correction(
    available_to_withdraw: str,
) -> None:
    response = _balance(availableToWithdraw=available_to_withdraw)["info"]
    exchange = ccxt.bybit()
    balance = exchange.parse_balance(response)
    assert parse_bybit_available_usdt(balance, "REGULAR_MARGIN") == pytest.approx(100.0)
    assert parse_available_usdt(balance) is None


@pytest.mark.parametrize("mode", ["REGULAR_MARGIN", "PORTFOLIO_MARGIN", "ISOLATED_MARGIN"])
def test_parsing_does_not_mutate_balance(mode: str) -> None:
    balance = _balance()
    original = deepcopy(balance)
    parse_bybit_available_usdt(balance, mode)
    assert balance == original
