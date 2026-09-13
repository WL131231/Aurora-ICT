"""진입 가용액 어댑터 계약 검증. 담당: Codex, 2026-09-13.

합성 응답을 실제 CCXT 파서와 어댑터에 전달한다. 네트워크와 주문 호출은 없다.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any

import ccxt
import pytest

from aurora_ict.bot.aurora_adapter import AuroraClientAdapter
from aurora_ict.bot.margin_guard import available_usdt, cap_qty_to_available


class WalletInput:
    """계정 모드와 잔고의 결정론적 입력 소스."""

    id = "bybit"

    def __init__(self) -> None:
        self.options: dict[str, Any] = {}
        self.calls: list[str] = []
        self.mode: Any = {"retCode": 0, "result": {"marginMode": "REGULAR_MARGIN"}}
        self.mode_error: BaseException | None = None
        self.balance_error: BaseException | None = None
        self.raw = {
            "retCode": 0,
            "result": {"list": [{
                "accountType": "UNIFIED",
                "totalAvailableBalance": "99",
                "coin": [{
                    "coin": "USDT", "walletBalance": "1000", "equity": "800",
                    "usdValue": "792", "unrealisedPnl": "-200",
                    "totalPositionIM": "400", "totalOrderIM": "300",
                    "locked": "0", "bonus": "0", "availableToWithdraw": "",
                }],
            }]},
        }

    async def load_time_difference(self) -> int:
        self.calls.append("time")
        return 0

    async def private_get_v5_account_info(self) -> Any:
        self.calls.append("mode")
        if self.mode_error is not None:
            raise self.mode_error
        return deepcopy(self.mode)

    async def fetch_balance(self) -> dict[str, Any]:
        self.calls.append("balance")
        if self.balance_error is not None:
            raise self.balance_error
        return ccxt.bybit().parse_balance(deepcopy(self.raw))


@pytest.mark.asyncio
async def test_entry_balance_uses_account_available_without_changing_display_balance() -> None:
    exchange = WalletInput()
    adapter = AuroraClientAdapter(SimpleNamespace(_ex=exchange))
    display = await adapter.fetch_balance()
    assert display["USDT"]["total"] == 1000.0
    assert display["USDT"]["free"] == 300.0
    assert await available_usdt(adapter) == pytest.approx(100.0)
    assert exchange.calls == ["balance", "time", "mode", "balance"]
    assert await cap_qty_to_available(adapter, "BTC/USDT:USDT", 20, 100, 10) == 9.0
    assert (await adapter.fetch_balance()) == display


@pytest.mark.asyncio
async def test_each_entry_refreshes_mode_and_balance() -> None:
    exchange = WalletInput()
    adapter = AuroraClientAdapter(SimpleNamespace(_ex=exchange))
    assert await available_usdt(adapter) == pytest.approx(100.0)
    exchange.raw["result"]["list"][0]["totalAvailableBalance"] = "49.5"
    assert await available_usdt(adapter) == pytest.approx(50.0)
    assert exchange.calls == ["time", "mode", "balance", "mode", "balance"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", [None, {}, {"retCode": 0}, {"retCode": False, "result": {}},
    {"retCode": 10000, "result": {"marginMode": "REGULAR_MARGIN"}},
    {"retCode": 0, "result": {"marginMode": "UNKNOWN"}},
])
async def test_unknown_mode_never_falls_back_to_ccxt_free(mode: Any) -> None:
    exchange = WalletInput()
    exchange.mode = mode
    adapter = AuroraClientAdapter(SimpleNamespace(_ex=exchange))
    assert await available_usdt(adapter) is None
    assert await cap_qty_to_available(adapter, "BTC/USDT:USDT", 1, 100, 10) == 0
    assert "balance" not in exchange.calls


@pytest.mark.asyncio
async def test_account_mode_auth_failure_is_counted() -> None:
    exchange = WalletInput()
    exchange.mode_error = ccxt.AuthenticationError("synthetic expired key")
    adapter = AuroraClientAdapter(SimpleNamespace(_ex=exchange))
    for expected in (1, 2):
        assert await available_usdt(adapter) is None
        assert adapter.auth_fail_streak == expected
    assert "balance" not in exchange.calls


@pytest.mark.asyncio
async def test_mode_success_does_not_erase_repeated_balance_auth_failure() -> None:
    exchange = WalletInput()
    exchange.balance_error = ccxt.AuthenticationError("synthetic balance authentication error")
    adapter = AuroraClientAdapter(SimpleNamespace(_ex=exchange))
    for expected in (1, 2, 3):
        assert await available_usdt(adapter) is None
        assert adapter.auth_fail_streak == expected
    exchange.balance_error = None
    assert await available_usdt(adapter) == pytest.approx(100.0)
    assert adapter.auth_fail_streak == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("where", ["mode_error", "balance_error"])
async def test_transient_query_failure_blocks_only_entry(where: str) -> None:
    exchange = WalletInput()
    setattr(exchange, where, ccxt.NetworkError("synthetic timeout"))
    adapter = AuroraClientAdapter(SimpleNamespace(_ex=exchange))
    assert await cap_qty_to_available(adapter, "BTC/USDT:USDT", 1, 100, 10) == 0
    assert adapter.auth_fail_streak == 0
    assert adapter.write_fail_streak == 0


@pytest.mark.asyncio
async def test_cancellation_propagates() -> None:
    exchange = WalletInput()
    exchange.mode_error = asyncio.CancelledError()
    adapter = AuroraClientAdapter(SimpleNamespace(_ex=exchange))
    with pytest.raises(asyncio.CancelledError):
        await available_usdt(adapter)


@pytest.mark.asyncio
async def test_missing_exchange_blocks_entry() -> None:
    adapter = AuroraClientAdapter(SimpleNamespace())
    assert await cap_qty_to_available(adapter, "BTC/USDT:USDT", 1, 100, 10) == 0
