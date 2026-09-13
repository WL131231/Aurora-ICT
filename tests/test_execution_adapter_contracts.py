"""실제 ccxt 파서/클라이언트/어댑터를 잇는 합성 거래 장부 검증.

담당: Codex. 외부 네트워크와 mock 프레임워크 없이 실패와 체결을 구분한다.
"""

from __future__ import annotations

import copy
import json
import time
from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import ccxt
import pandas as pd
import pytest

from aurora.config import settings
from aurora_ict.bot.origo_adapter import OrigoCcxtClient as CcxtClient
from aurora_ict.bot.origo_adapter import OrigoClientAdapter as AuroraClientAdapter

SYMBOL = "BTC/USDT:USDT"
MARKET = {
    "id": "BTCUSDT", "symbol": SYMBOL, "base": "BTC", "quote": "USDT", "settle": "USDT",
    "baseId": "BTC", "quoteId": "USDT", "settleId": "USDT", "type": "swap",
    "spot": False, "swap": True, "future": False, "option": False,
    "linear": True, "inverse": False, "contract": True, "contractSize": 1,
    "precision": {"amount": 0.001, "price": 0.1}, "limits": {},
}


class ExchangeLedger:
    """결정론적 주문/포지션 장부. 모든 변환은 실제 ccxt 파서에 위임한다."""

    id = "bybit"

    def __init__(self) -> None:
        """키 없이 합성 장부와 실제 파서를 만든다."""
        self.options: dict[str, Any] = {}
        self.parser = ccxt.bybit()
        self.parser.set_markets([MARKET])
        self.orders: list[dict[str, Any]] = []
        self.positions: list[dict[str, Any]] = []
        self.position_error: Exception | None = None
        self.cancel_error: Exception | None = None
        self.create_error: Exception | None = None
        self.accept_before_error = False
        self.keep_after_cancel = False
        self.hide_realtime = False
        self.repeat_cursor = False
        self.page_size = 50
        self.filled = 0.0
        self.average: float | None = None
        self.calls: list[tuple[str, Any]] = []
        self.key_response = {
            "retCode": 0,
            "result": {"readOnly": 0, "permissions": {"ContractTrade": ["Order", "Position"]}},
        }

    def market(self, symbol: str) -> dict[str, Any]:
        """검증용 심볼의 실제 파서 메타를 반환한다."""
        assert symbol == SYMBOL
        return MARKET

    def parse_order(self, raw: dict, market: dict) -> dict:
        """자료 변환을 실제 라이브러리에 위임한다."""
        return self.parser.parse_order(raw, market)

    async def load_time_difference(self) -> None:
        """외부 시간 조회 없는 합성 시간 경계."""

    async def fetch_positions(self, symbols: list[str]) -> list[dict]:
        """조회 실패는 상태 0과 구분한다."""
        self.calls.append(("positions", list(symbols)))
        if self.position_error:
            raise self.position_error
        return copy.deepcopy(self.positions)

    def add_order(self, order_id: str, tag: str, **changes: Any) -> dict:
        """Bybit 원시 형식의 지정가 주문을 장부에 추가한다."""
        raw = {
            "orderId": order_id, "orderLinkId": tag, "symbol": "BTCUSDT",
            "side": "Buy", "orderType": "Limit", "orderStatus": "New",
            "qty": "4", "price": "100", "avgPrice": "", "cumExecQty": "0",
            "leavesQty": "4", "cumExecValue": "0", "cumExecFee": "0",
            "createdTime": "1789297200000", "updatedTime": "1789297200000",
            "reduceOnly": False, "closeOnTrigger": False, "stopOrderType": "",
            "stopLoss": "", "takeProfit": "", "timeInForce": "GTC",
        }
        raw.update(changes)
        self.orders.append(raw)
        return raw

    async def create_order(self, symbol, order_type, side, qty, price, params) -> dict:
        """접수와 응답 유실을 별개의 상태로 실행한다."""
        self.calls.append(("create", dict(params)))
        if self.create_error and not self.accept_before_error:
            raise self.create_error
        raw = self.add_order(
            f"order-{len(self.orders) + 1}", params["orderLinkId"],
            side=side.title(), orderType=order_type.title(), qty=str(qty), price=str(price or 0),
            cumExecQty=str(self.filled), leavesQty=str(qty - self.filled),
            avgPrice="" if self.average is None else str(self.average),
            cumExecValue=str(self.filled * (self.average or 0)),
            orderStatus="Filled" if self.filled == qty else "New",
            reduceOnly=params.get("reduceOnly", False), stopLoss=params.get("stopLoss", ""),
        )
        if self.create_error:
            raise self.create_error
        return self.parse_order(raw, MARKET)

    def _response(self, rows: list[dict], cursor: str = "") -> dict:
        """페이지의 독립 사본을 만든다."""
        return {"retCode": 0, "result": {"list": copy.deepcopy(rows), "nextPageCursor": cursor}}

    async def private_get_v5_order_realtime(self, request: dict) -> dict:
        """대기 목록 또는 정확한 주문 ID를 조회한다."""
        self.calls.append(("realtime", dict(request)))
        if "orderId" in request or "orderLinkId" in request:
            key = "orderId" if "orderId" in request else "orderLinkId"
            return self._response([] if self.hide_realtime else [
                row for row in self.orders if row[key] == request[key]
            ])
        rows = [row for row in self.orders if row["orderStatus"] not in ("Cancelled", "Filled")]
        offset = int(request.get("cursor") or 0)
        page = rows[offset:offset + self.page_size]
        cursor = str(offset + self.page_size) if offset + self.page_size < len(rows) else ""
        if self.repeat_cursor:
            cursor = "1"
        return self._response(page, cursor)

    async def private_get_v5_order_history(self, request: dict) -> dict:
        """실시간 캐시 유실과 주문 이력 보존을 분리한다."""
        self.calls.append(("history", dict(request)))
        key = "orderId" if "orderId" in request else "orderLinkId"
        return self._response([row for row in self.orders if row[key] == request[key]])

    async def cancel_order(self, order_id: str, symbol: str) -> None:
        """취소가 실행된 경우에만 장부 상태를 바꾼다."""
        self.calls.append(("cancel", order_id))
        if self.cancel_error:
            raise self.cancel_error
        if not self.keep_after_cancel:
            next(row for row in self.orders if row["orderId"] == order_id)["orderStatus"] = "Cancelled"

    async def private_get_v5_user_query_api(self) -> dict:
        """키 원문이 없는 합성 권한 응답."""
        return copy.deepcopy(self.key_response)


@pytest.fixture
def exchange() -> ExchangeLedger:
    """운영 설정 변경 없이 테스트 프로세스의 모드만 격리한다."""
    old = settings.run_mode
    settings.run_mode = "live"
    try:
        yield ExchangeLedger()
    finally:
        settings.run_mode = old


def adapter_for(exchange: ExchangeLedger) -> AuroraClientAdapter:
    """실제 클라이언트와 어댑터를 합성 전송 경계에 연결한다."""
    client = CcxtClient(api_key="", api_secret="")
    client._ex = exchange
    client._initialized = True
    client._last_time_sync = time.monotonic()
    return AuroraClientAdapter(client)


@pytest.mark.parametrize("error_type", [ccxt.NetworkError, ccxt.RateLimitExceeded])
async def test_double_position_failure_is_unknown(exchange, error_type):
    """실제 클라이언트와 대체 조회가 실패해도 포지션 없음으로 바꾸지 않는다."""
    exchange.position_error = error_type("synthetic outage")
    adapter = adapter_for(exchange)
    with pytest.raises(error_type):
        await adapter.fetch_position(SYMBOL)
    assert [name for name, _ in exchange.calls].count("positions") == 2


async def test_slot_position_preserves_protective_fields(exchange):
    """slots 변환 후에도 SL/청산가가 남고 불필요한 이중 조회가 없다."""
    exchange.positions = [{
        "symbol": SYMBOL, "side": "long", "contracts": 4.0, "entryPrice": 100.0,
        "stopLossPrice": 98.0, "liquidationPrice": 80.0, "info": {"stopLoss": "98"},
    }]
    result = await adapter_for(exchange).fetch_position(SYMBOL)
    assert result["contracts"] == 4
    assert result["entryPrice"] == 100
    assert result["info"]["stopLoss"] == "98"
    assert result["liquidationPrice"] == 80
    assert len(exchange.calls) == 1


@pytest.mark.parametrize("row", [
    {"contracts": "NaN", "side": "long"}, {"contracts": 1, "side": "unknown"},
    {"contracts": None}, {"contracts": True}, {"contracts": -1},
    {"contracts": 1, "side": "long", "symbol": "ETH/USDT:USDT"},
])
async def test_invalid_position_never_means_flat(exchange, row):
    """수량/방향/심볼 미확인은 청산 근거가 아니다."""
    exchange.positions = [row]
    with pytest.raises(ccxt.ExchangeError):
        await adapter_for(exchange).fetch_position(SYMBOL)


async def test_empty_position_is_confirmed_flat(exchange):
    """성공한 빈 조회는 실제 무포지션으로 반환한다."""
    assert await adapter_for(exchange).fetch_position(SYMBOL) is None


async def test_order_slots_keep_id_and_do_not_invent_fill(exchange):
    """미체결 지정가의 요청 수량/가격을 체결 정보로 대체하지 않는다."""
    result = await adapter_for(exchange).place_order(SYMBOL, "buy", 4, price=100, stop_loss=98)
    assert result["id"] == result["order_id"] == "order-1"
    assert result["filled_qty"] == 0
    assert result["avg_fill_price"] is None
    assert result["remaining"] == 4
    params = next(data for name, data in exchange.calls if name == "create")
    assert params["stopLoss"] == "98"
    assert params["tpslMode"] == "Full"


async def test_reduce_fill_has_actual_quantity_and_average(exchange):
    """청산 주문도 실제 부분체결 수량/평균가를 그대로 보존한다."""
    exchange.filled, exchange.average = 1.0, 101.5
    result = await adapter_for(exchange).place_order(SYMBOL, "sell", 4, reduce_only=True, stop_loss=98)
    assert result["filled_qty"] == 1
    assert result["avg_fill_price"] == 101.5
    assert result["remaining"] == 3
    params = next(data for name, data in exchange.calls if name == "create")
    assert params["reduceOnly"] is True
    assert "stopLoss" not in params


async def test_cancel_confirms_absence_and_preserves_manual_and_protection(exchange):
    """진입 주문만 취소하며 수동 주문/봇 손절/봇 익절은 보존한다."""
    exchange.page_size = 2
    exchange.add_order("entry", "AURentry")
    exchange.add_order("manual", "manual")
    exchange.add_order("sl", "AURsl", reduceOnly=True, closeOnTrigger=True, stopOrderType="StopLoss")
    exchange.add_order("tp", "AURtp", reduceOnly="true")
    exchange.add_order("entry2", "AURentry2")
    adapter = adapter_for(exchange)
    assert await adapter.cancel_bot_orders(SYMBOL) == 2
    assert {data for name, data in exchange.calls if name == "cancel"} == {"entry", "entry2"}
    assert await adapter.cancel_bot_orders(SYMBOL) == 0
    assert next(row for row in exchange.orders if row["orderId"] == "sl")["orderStatus"] == "New"


@pytest.mark.parametrize("delay", [False, True])
async def test_cancel_failure_or_async_delay_remains_unknown(exchange, delay):
    """취소 오류와 아직 처리되지 않은 취소 접수를 모두 미확인으로 남긴다."""
    exchange.add_order("entry", "AURentry")
    if delay:
        exchange.keep_after_cancel = True
    else:
        exchange.cancel_error = ccxt.NetworkError("synthetic cancel outage")
    with pytest.raises(ccxt.BaseError):
        await adapter_for(exchange).cancel_bot_orders(SYMBOL)
    assert exchange.orders[0]["orderStatus"] == "New"


async def test_cancel_does_not_accept_truncated_pagination(exchange):
    """반복 페이지를 빈 주문 확인으로 바꾸지 않는다."""
    exchange.repeat_cursor = True
    with pytest.raises(ccxt.ExchangeError, match="반복"):
        await adapter_for(exchange).cancel_bot_orders(SYMBOL)
    assert not any(name == "cancel" for name, _ in exchange.calls)


async def test_response_loss_can_lookup_same_order_without_resubmit(exchange):
    """접수 후 응답 유실은 같은 주문을 조회해서 복구하며 추가 주문은 없다."""
    exchange.create_error = ccxt.NetworkError("synthetic response loss")
    exchange.accept_before_error = True
    adapter = adapter_for(exchange)
    with pytest.raises(ccxt.NetworkError) as captured:
        await adapter.place_order(SYMBOL, "buy", 4, price=100, stop_loss=98)
    assert getattr(captured.value, "order_definitively_rejected", False) is False
    result = await adapter.fetch_order(captured.value.order_lookup_id, SYMBOL)
    assert result["id"] == "order-1"
    assert result["filled_qty"] == 0
    assert len(exchange.orders) == 1
    assert [name for name, _ in exchange.calls].count("create") == 1


@pytest.mark.parametrize("ret_code,definitive", [
    (110007, True), ("110007", True), (110004, False), (10005, False),
])
async def test_raw_bybit_insufficient_funds_is_definitively_rejected(exchange, ret_code, definitive):
    """실제 CCXT 오류 분류와 원시 retCode를 함께 확인한 110007만 미접수로 판정한다."""
    response = {"retCode": ret_code, "retMsg": "synthetic exchange rejection", "result": {}}
    body = json.dumps(response)
    with pytest.raises(ccxt.BaseError) as parsed:
        exchange.parser.handle_errors(
            200, "OK", "https://api.bybit.com/v5/order/create", "POST",
            {}, body, response, {}, "{}",
        )
    exchange.create_error = parsed.value
    adapter = adapter_for(exchange)
    with pytest.raises(ccxt.BaseError) as captured:
        await adapter.place_order(SYMBOL, "buy", 4, price=100, stop_loss=98)
    assert getattr(captured.value, "order_definitively_rejected", False) is definitive
    assert captured.value.order_lookup_id.startswith("client:AUR")
    assert exchange.orders == []
    assert [name for name, _ in exchange.calls].count("create") == 1


@pytest.mark.parametrize("error_type,message", [
    (ccxt.InsufficientFunds, "110007 balance insufficient"),
    (ccxt.InsufficientFunds, 'bybit {"retCode":110004}'),
    (ccxt.InsufficientFunds, 'bybit {"retCode":110007.0}'),
    (ccxt.InsufficientFunds, 'another-exchange {"retCode":110007}'),
    (ccxt.NetworkError, 'bybit {"retCode":110007}'),
    (ccxt.ExchangeError, 'bybit {"retCode":110007}'),
])
async def test_rejection_text_or_untyped_error_is_not_proof(exchange, error_type, message):
    """문구 일치나 일반 예외만으로 주문이 미접수됐다고 추정하지 않는다."""
    exchange.create_error = error_type(message)
    with pytest.raises(ccxt.BaseError) as captured:
        await adapter_for(exchange).place_order(SYMBOL, "buy", 4, price=100, stop_loss=98)
    assert getattr(captured.value, "order_definitively_rejected", False) is False
    assert captured.value.order_lookup_id.startswith("client:AUR")


@pytest.mark.parametrize("price", [None, 100.0])
async def test_attached_sl_survives_actual_ccxt_request_conversion(exchange, price):
    """Origo SL 파라미터가 실제 CCXT 시장가/지정가 요청에도 보존된다."""
    await adapter_for(exchange).place_order(SYMBOL, "buy", 4, price=price, stop_loss=98)
    params = next(value for name, value in exchange.calls if name == "create")
    request = exchange.parser.create_order_request(
        SYMBOL, "market" if price is None else "limit", "buy", 4, price, params, True,
    )
    assert request["stopLoss"] == "98"
    assert request["tpslMode"] == "Full"
    assert request["slOrderType"] == "Market"
    assert request["orderLinkId"].startswith("AUR")
    assert "reduceOnly" not in request


async def test_order_history_fallback_is_exact_and_notfound_is_unknown(exchange):
    """캐시에서 사라진 주문은 정확한 ID 이력으로만 복원한다."""
    exchange.add_order("old", "AURold", orderStatus="Filled", cumExecQty="4", leavesQty="0", avgPrice="101")
    exchange.hide_realtime = True
    adapter = adapter_for(exchange)
    result = await adapter.fetch_order("old", SYMBOL)
    assert result["status"] == "closed"
    assert result["filled_qty"] == 4
    with pytest.raises(ccxt.OrderNotFound):
        await adapter.fetch_order("missing", SYMBOL)


@pytest.mark.parametrize("accepted", [False, True])
async def test_permission_error_requires_exact_order_evidence(exchange, accepted):
    """10005 뒤 다른 포지션 변화가 아니라 해당 주문 실재 여부를 검사한다."""
    exchange.create_error = ccxt.PermissionDenied("10005 synthetic denial")
    exchange.accept_before_error = accepted
    adapter = adapter_for(exchange)
    if accepted:
        result = await adapter.place_order(SYMBOL, "buy", 4, price=100, stop_loss=98)
        assert result["id"] == "order-1"
        assert adapter.write_fail_streak == 0
    else:
        with pytest.raises(ccxt.PermissionDenied):
            await adapter.place_order(SYMBOL, "buy", 4, price=100, stop_loss=98)
        assert adapter.write_fail_streak == 1


@pytest.mark.parametrize("read_only,permissions,valid", [
    (0, ["Order", "Position"], True), ("0", ["Order", "Position"], True),
    (1, ["Order", "Position"], False), (False, ["Order", "Position"], False),
    (0, ["Order"], False), (0, [], False), (None, ["Order", "Position"], False),
])
async def test_required_key_permissions(exchange, read_only, permissions, valid):
    """정지 해제에 필요한 쓰기 권한만 검사하며 키 원문은 반환하지 않는다."""
    exchange.key_response["result"] = {"readOnly": read_only, "permissions": {"ContractTrade": permissions}}
    adapter = adapter_for(exchange)
    if valid:
        assert await adapter.validate_trading_credentials() is True
    else:
        with pytest.raises(ccxt.PermissionDenied):
            await adapter.validate_trading_credentials()


async def test_public_candle_success_does_not_reset_auth_failure():
    """공개 캔들 성공 때문에 만료 키 정지가 무력화되지 않는다."""
    class CandleSource:
        async def fetch_ohlcv(self, symbol, timeframe, limit):
            """인증과 무관한 합성 공개 시세."""
            return pd.DataFrame()

    adapter = AuroraClientAdapter(CandleSource())
    adapter.auth_fail_streak = 2
    adapter.last_auth_error = "33004 expired"
    assert await adapter.fetch_ohlcv(SYMBOL, "5m", 2) == []
    assert adapter.auth_fail_streak == 2


async def test_factories_isolate_origo_without_changing_cursus_clients():
    """실제 factory의 클래스 선택만 검증한다. 주문/시세/인증 API는 호출하지 않는다."""
    from aurora.exchange.ccxt_client import CcxtClient as LegacyCcxtClient
    from aurora_ict.bot import aurora_client_factory, origo_client_factory
    from aurora_ict.bot.aurora_adapter import AuroraClientAdapter as LegacyAdapter

    config = SimpleNamespace(active_api_key="", active_api_secret="", is_demo=False)
    legacy = await aurora_client_factory(config)
    strict = await origo_client_factory(config)
    try:
        assert type(legacy) is LegacyAdapter
        assert type(legacy._client) is LegacyCcxtClient
        assert type(strict) is AuroraClientAdapter
        assert type(strict._client) is CcxtClient
        assert legacy._client._initialized is False
        assert strict._client._initialized is False
        assert not hasattr(legacy._client, "fetch_order")
        assert not hasattr(legacy, "validate_trading_credentials")
        assert callable(strict.fetch_order)
        assert callable(strict.validate_trading_credentials)
    finally:
        await legacy._client.close()
        await strict._client.close()


def test_fill_metadata_is_only_on_origo_dataclass_subclasses():
    """공통 Order/Position 필드를 바꾸지 않고 Origo만 메타데이터를 갖는다."""
    from aurora.exchange.base import Order, Position
    from aurora_ict.bot.origo_adapter import OrigoOrder, OrigoPosition

    assert "filled_qty" not in {field.name for field in fields(Order)}
    assert "raw" not in {field.name for field in fields(Position)}
    assert {"filled_qty", "avg_fill_price", "remaining", "raw"} <= {
        field.name for field in fields(OrigoOrder)
    }
    assert "raw" in {field.name for field in fields(OrigoPosition)}
    assert issubclass(OrigoOrder, Order)
    assert issubclass(OrigoPosition, Position)
