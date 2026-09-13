"""Origo 주문 확인·부분 체결·진입 이후 보호 조건을 검증한다. 담당: Codex.

실제 봇 함수와 합성 주문 장부를 사용하며 외부 네트워크나 mock은 사용하지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from ccxt.base.errors import ExchangeError, RequestTimeout

from aurora_ict.bot.bot_ict_instance import BotIctInstance, BotState, _ActivePosition
from aurora_ict.indicators.fvg import FVG, FVGType
from aurora_ict.interfaces.trades_store import TradeEventType
from aurora_ict.strategy.htf_fvg_map import HtfFvgEntry
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup

SYMBOL = "BTC/USDT:USDT"
ENTRY_TS = int(pd.Timestamp("2026-09-13T10:04:00Z").timestamp() * 1000)


class OrderLedger:
    """주문 접수·누적 체결·취소·포지션을 명시적 상태 전이로 관리한다."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.orders: dict[str, dict[str, Any]] = {}
        self.protection: list[dict[str, Any]] = []
        self.position: dict[str, Any] | None = None
        self.price: float | None = 100.0
        self.reject_entry = False
        self.entry_rejection_code = 10001
        self.definitive_rejection_marker: object = False
        self.lose_response = False
        self.cancel_fails = False
        self.position_read_fails = False
        self.order_unknown = False
        self.order_read_error: Exception | None = None
        self.immediate_fill_fraction = 0.0
        self.cancel_fill_fraction = 0.0
        self.cancel_calls = 0
        self.close_fill_fraction = 1.0
        self.close_terminal = False
        self.close_response_lost = False
        self.defer_close_position = False
        self.fail_after_close_submit = False
        self.protection_fails = False
        self.response_error_type: type[Exception] = RequestTimeout

    async def fetch_balance(self) -> dict[str, Any]:
        return {"USDT": {"total": 1000.0, "free": 1000.0}}

    async def fetch_available_usdt(self) -> float:
        return 1000.0

    async def fetch_ticker(self, symbol: str) -> float | None:
        return self.price

    async def fetch_position(self, symbol: str) -> dict[str, Any] | None:
        if self.position_read_fails:
            raise RequestTimeout("합성 포지션 조회 실패")
        return dict(self.position) if self.position is not None else None

    async def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any] | None:
        if self.order_read_error is not None:
            raise self.order_read_error
        if self.order_unknown:
            return None
        order = self.orders.get(order_id.removeprefix("client:"))
        return dict(order) if order is not None else None

    async def place_order(
        self, symbol: str, side: str, qty: float, price: float | None = None,
        reduce_only: bool = False, **params: Any,
    ) -> dict[str, Any]:
        request = dict(symbol=symbol, side=side, qty=qty, price=price, reduce_only=reduce_only, **params)
        self.requests.append(request)
        if self.reject_entry and not reduce_only:
            error = ExchangeError(f"{self.entry_rejection_code}: synthetic entry rejection")
            error.order_definitively_rejected = self.definitive_rejection_marker
            if self.entry_rejection_code == 110007:
                error.order_lookup_id = "client:rejected"
            raise error
        if reduce_only:
            assert self.position is not None
            oid = f"close{len(self.orders) + 1}"
            self.orders[oid] = dict(
                id=oid, qty=qty, status="open", filled_qty=0.0,
                avg_fill_price=0.0, reduce_only=True,
            )
            self.fill_close(oid, qty * self.close_fill_fraction)
            if self.close_terminal and self.orders[oid]["status"] == "open":
                self.orders[oid]["status"] = "canceled"
            if self.fail_after_close_submit:
                self.position_read_fails = True
            if self.close_response_lost:
                error = self.response_error_type("합성 청산 응답 유실")
                error.order_lookup_id = f"client:{oid}"
                raise error
            return dict(self.orders[oid])
        oid = str(len(self.orders) + 1)
        self.orders[oid] = dict(
            id=oid, status="open", filled_qty=0.0, avg_fill_price=0.0,
            qty=qty, remaining=qty, side=side, stop_loss=params.get("stop_loss"),
        )
        if self.immediate_fill_fraction:
            self.fill(oid, qty * self.immediate_fill_fraction)
        if self.lose_response:
            error = self.response_error_type("합성 접수 응답 유실")
            error.order_lookup_id = f"client:{oid}"
            raise error
        return dict(self.orders[oid])

    def fill(self, order_id: str, cumulative: float, average: float = 100.0) -> None:
        order = self.orders[order_id]
        assert order["filled_qty"] <= cumulative <= order["qty"]
        delta = cumulative - order["filled_qty"]
        order.update(filled_qty=cumulative, avg_fill_price=average, remaining=order["qty"] - cumulative)
        if cumulative == order["qty"]:
            order["status"] = "closed"
        if delta:
            self.position = dict(
                contracts=cumulative, side="long" if order["side"] == "buy" else "short",
                entryPrice=average, stopLossPrice=order["stop_loss"],
            )

    async def cancel_bot_orders(self, symbol: str) -> int:
        self.cancel_calls += 1
        if self.cancel_fails:
            raise RequestTimeout("합성 취소 미확인")
        count = 0
        for oid, order in self.orders.items():
            if order["status"] != "open" or order.get("reduce_only"):
                continue
            if self.cancel_fill_fraction:
                self.fill(oid, order["qty"] * self.cancel_fill_fraction)
                self.cancel_fill_fraction = 0.0
            if order["status"] == "open":
                order["status"] = "canceled"
            count += 1
        assert not any(order["status"] == "open" and not order.get("reduce_only") for order in self.orders.values())
        return count

    def fill_close(self, order_id: str, cumulative: float) -> None:
        order = self.orders[order_id]
        delta = cumulative - order["filled_qty"]
        assert 0 <= delta and cumulative <= order["qty"]
        order.update(filled_qty=cumulative, avg_fill_price=self.price)
        if cumulative == order["qty"]:
            order["status"] = "closed"
        if delta and not self.defer_close_position:
            assert self.position is not None
            self.position["contracts"] -= delta
            if self.position["contracts"] <= 1e-9:
                self.position = None

    async def set_position_tpsl(self, symbol: str, **params: Any) -> dict[str, int]:
        self.protection.append(dict(params))
        if self.position is None or self.protection_fails:
            return {}
        if params.get("stop_loss") is not None:
            self.position["stopLossPrice"] = params["stop_loss"]
        return {"retCode": 0}


def setup() -> SilverBulletSetup:
    """고정 가격의 합성 롱 진입 조건을 만든다.

    Returns:
        실제 전략 데이터 구조.
    """
    return SilverBulletSetup(
        ts_ms=ENTRY_TS, direction=Direction.LONG, window="any", entry=100.0,
        stop_loss=98.0, take_profit=110.0, risk_reward=5.0,
        fvg=FVG(type=FVGType.BULLISH, idx=5, ts_ms=ENTRY_TS, low=99.0, high=101.0),
    )


def make_bot(tmp_path: Path, ledger: OrderLedger, **kwargs: Any) -> BotIctInstance:
    """실제 봇을 임시 거래 장부와 연결한다.

    Args:
        tmp_path: 테스트 전용 기록 경로.
        ledger: 합성 거래소 상태.
        kwargs: 테스트할 설정.
    Returns:
        실제 Origo 인스턴스.
    """
    return BotIctInstance(
        client=ledger, symbol=SYMBOL, trades_data_dir=tmp_path,
        trail_trigger_r=0, trail_dist_r=0, **kwargs,
    )


def entries(bot: BotIctInstance) -> list[Any]:
    """저장된 실제 ENTRY 이벤트만 반환한다.

    Args:
        bot: 검증 대상.
    Returns:
        진입 이벤트 목록.
    """
    return [e for e in bot._trades_store.all_events() if e.event_type is TradeEventType.ENTRY]


@pytest.mark.asyncio
async def test_limit_has_attached_sl_before_any_fill(tmp_path: Path) -> None:
    ledger = OrderLedger()
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    assert ledger.requests[0]["stop_loss"] == 98.0
    assert ledger.protection == []
    assert bot._pending_entry is not None
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_sl_rejection_never_retries_without_sl(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.reject_entry = True
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    assert len(ledger.requests) == 1
    assert ledger.requests[0]["stop_loss"] == 98.0
    assert ledger.orders == {}
    assert bot._pending_entry is not None
    assert not await bot._check_pending_entry()
    assert bot._pending_entry is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RequestTimeout, ExchangeError])
async def test_unknown_submission_tracks_lookup_and_blocks_duplicate(
    tmp_path: Path, error_type: type[Exception],
) -> None:
    ledger = OrderLedger()
    ledger.lose_response = True
    ledger.response_error_type = error_type
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    assert bot._pending_entry.order_id == "client:1"
    await bot._execute_setup(setup())
    assert len(ledger.requests) == 1
    assert await bot._check_pending_entry() is False
    assert ledger.orders["1"]["status"] == "canceled"


@pytest.mark.asyncio
async def test_partial_fill_protected_and_retained_until_cancel_confirmed(tmp_path: Path) -> None:
    ledger = OrderLedger()
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    qty = ledger.orders["1"]["qty"]
    ledger.fill("1", qty / 4)
    ledger.cancel_fails = True
    assert await bot._check_pending_entry() is True
    assert bot.active_position.qty == qty / 4
    assert bot._pending_entry is not None
    assert ledger.protection[0]["stop_loss"] == 98.0
    assert sum(e.qty for e in entries(bot)) == qty / 4
    await bot._execute_setup(setup())
    assert len(ledger.requests) == 1
    ledger.cancel_fails = False
    assert await bot._check_pending_entry() is False
    assert bot._pending_entry is None
    assert sum(e.qty for e in entries(bot)) == qty / 4
    assert ledger.orders["1"]["status"] == "canceled"


@pytest.mark.asyncio
async def test_fill_during_cancel_records_only_increment(tmp_path: Path) -> None:
    ledger = OrderLedger()
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    qty = ledger.orders["1"]["qty"]
    ledger.fill("1", qty / 4)
    ledger.cancel_fill_fraction = 0.5
    assert await bot._check_pending_entry() is False
    assert bot.active_position.qty == qty / 2
    assert [e.qty for e in entries(bot)] == [qty / 4, qty / 4]


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["ttl", "cancel", "stop"])
async def test_cancel_failure_preserves_pending(tmp_path: Path, action: str) -> None:
    ledger = OrderLedger()
    ledger.cancel_fails = True
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    pending = bot._pending_entry
    if action == "ttl":
        pending.placed_ts_ms = 0
        assert await bot._check_pending_entry() is True
    elif action == "cancel":
        assert await bot.cancel_pending_entry() is False
    else:
        bot.state = BotState.RUNNING
        with pytest.raises(RuntimeError, match="취소 미확인"):
            await bot.stop()
        assert bot.state is BotState.RUNNING
        assert bot.entry_paused
    assert bot._pending_entry is pending
    assert ledger.orders["1"]["status"] == "open"
    ledger.cancel_fails = False
    assert await bot._check_pending_entry() is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["order", "position"])
async def test_unknown_exchange_state_never_clears_pending(tmp_path: Path, failure: str) -> None:
    ledger = OrderLedger()
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    bot._pending_entry.placed_ts_ms = 0
    ledger.order_unknown = failure == "order"
    ledger.position_read_fails = failure == "position"
    assert await bot._check_pending_entry() is True
    assert bot._pending_entry is not None
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_market_acceptance_without_fill_is_not_a_position(tmp_path: Path) -> None:
    ledger = OrderLedger()
    bot = make_bot(tmp_path, ledger, use_market_entry=True)
    await bot._execute_setup(setup())
    assert bot.active_position is None
    assert bot._pending_entry is not None
    assert ledger.protection == []


@pytest.mark.asyncio
async def test_immediate_partial_response_uses_actual_qty(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.immediate_fill_fraction = 0.25
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    assert bot.active_position.qty == ledger.orders["1"]["qty"] / 4
    assert bot._pending_entry.cancel_requested


def be_position(bot: BotIctInstance, ledger: OrderLedger) -> pd.DataFrame:
    """진입 전에 고점이 가능했던 합성 봉과 포지션을 배치한다.

    Args:
        bot: 검증 대상.
        ledger: 합성 장부.
    Returns:
        진입 포함 봉.
    """
    ledger.position = dict(contracts=1.0, side="long", entryPrice=100.0, stopLossPrice=98.0)
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=98.0, take_profit=110.0,
        qty=1.0, setup_ts_ms=ENTRY_TS, entry_ts_ms=ENTRY_TS, tp1_price=103.0,
    )
    return pd.DataFrame(
        [dict(open=102.0, high=104.0, low=99.9, close=100.2, volume=1.0)],
        index=pd.to_datetime(["2026-09-13T10:00:00Z"]),
    )


@pytest.mark.asyncio
async def test_pre_entry_high_cannot_trigger_be_or_partial(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.price = 100.2
    bot = make_bot(tmp_path, ledger, be_trigger_r=1.0)
    df = be_position(bot, ledger)
    await bot._maybe_be_lock(df)
    await bot._maybe_partial_exit(df)
    assert bot.active_position.stop_loss == 98.0
    assert not bot.active_position.be_moved
    assert ledger.protection == []
    assert ledger.requests == []
    ledger.price = 102.1
    await bot._maybe_be_lock(df)
    assert bot.active_position.stop_loss == 100.0


@pytest.mark.asyncio
async def test_bar_start_after_entry_may_use_extremes(tmp_path: Path) -> None:
    ledger = OrderLedger()
    bot = make_bot(tmp_path, ledger, be_trigger_r=1.0)
    df = be_position(bot, ledger)
    df.index = pd.to_datetime(["2026-09-13T10:05:00Z"])
    await bot._maybe_be_lock(df)
    assert bot.active_position.be_moved


@pytest.mark.asyncio
async def test_unknown_post_entry_price_leaves_sl_unchanged(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.price = None
    bot = make_bot(tmp_path, ledger, be_trigger_r=1.0)
    df = be_position(bot, ledger)
    await bot._maybe_be_lock(df)
    assert bot.active_position.stop_loss == 98.0


@pytest.mark.asyncio
async def test_entry_pause_does_not_disable_protection(tmp_path: Path) -> None:
    ledger = OrderLedger()
    bot = make_bot(tmp_path, ledger, entry_paused=True)
    await bot._execute_setup(setup())
    assert ledger.requests == []
    be_position(bot, ledger)
    assert await bot._ensure_protective_sl(110.0, 2.0)


@pytest.mark.asyncio
async def test_safety_stop_reason_reaches_persistence_callback(tmp_path: Path) -> None:
    reasons = []

    async def persist(reason: str) -> None:
        reasons.append(reason)

    bot = make_bot(tmp_path, OrderLedger(), safety_stop_cb=persist)
    await bot._on_auth_stop(3, "expired")
    await bot._on_write_denied_stop(3, "permission")
    assert reasons == ["auth_expired", "api_permission"]


def close_events(bot: BotIctInstance) -> list[Any]:
    """실제 청산으로 기록된 이벤트를 반환한다.

    Args:
        bot: 검증 대상.
    Returns:
        청산 이벤트 목록.
    """
    if bot._trades_store is None:
        return []
    return [e for e in bot._trades_store.all_events() if e.event_type is not TradeEventType.ENTRY]


@pytest.mark.asyncio
async def test_emergency_ack_and_partial_fill_never_clear_or_duplicate(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.close_fill_fraction = 0.0
    bot = make_bot(tmp_path, ledger)
    be_position(bot, ledger)
    await bot._emergency_close()
    await bot._emergency_close()
    assert len(ledger.requests) == 1
    assert bot.active_position.qty == 1.0
    assert close_events(bot) == []
    ledger.fill_close("close1", 0.4)
    await bot._check_pending_close()
    assert bot.active_position.qty == pytest.approx(0.6)
    assert sum(e.qty for e in close_events(bot)) == pytest.approx(0.4)
    assert bot._pending_close is not None
    ledger.fill_close("close1", 1.0)
    await bot._check_pending_close()
    assert bot.active_position is None
    assert bot._pending_close is None
    assert sum(e.qty for e in close_events(bot)) == pytest.approx(1.0)
    assert len(ledger.requests) == 1


@pytest.mark.asyncio
async def test_fill_confirmation_waits_for_position_propagation(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.defer_close_position = True
    bot = make_bot(tmp_path, ledger)
    be_position(bot, ledger)
    await bot._emergency_close()
    assert bot.active_position is not None
    assert bot._pending_close is not None
    ledger.position = None
    await bot._check_pending_close()
    assert bot.active_position is None
    assert len(ledger.requests) == 1
    assert sum(e.qty for e in close_events(bot)) == 1.0


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [RequestTimeout, ExchangeError])
async def test_lost_close_response_uses_same_lookup_without_resubmission(
    tmp_path: Path, error_type: type[Exception],
) -> None:
    ledger = OrderLedger()
    ledger.close_response_lost = True
    ledger.response_error_type = error_type
    bot = make_bot(tmp_path, ledger)
    be_position(bot, ledger)
    await bot._emergency_close()
    assert bot._pending_close.order_id == "client:close1"
    assert bot.active_position is not None
    assert close_events(bot) == []
    await bot._check_pending_close()
    assert bot.active_position is None
    assert len(ledger.requests) == 1
    assert sum(e.qty for e in close_events(bot)) == 1.0


@pytest.mark.asyncio
async def test_close_position_read_failure_preserves_intent(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.fail_after_close_submit = True
    bot = make_bot(tmp_path, ledger)
    be_position(bot, ledger)
    await bot._emergency_close()
    assert bot.active_position is not None
    assert bot._pending_close is not None
    assert close_events(bot) == []
    await bot._emergency_close()
    assert len(ledger.requests) == 1
    ledger.position_read_fails = False
    await bot._check_pending_close()
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_partial_exit_retries_only_unfilled_original_target(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.price = 103.1
    ledger.close_fill_fraction = 0.25
    ledger.close_terminal = True
    bot = make_bot(tmp_path, ledger)
    df = be_position(bot, ledger)
    await bot._maybe_partial_exit(df)
    assert bot.active_position.qty == pytest.approx(0.875)
    assert not bot.active_position.partial_done
    assert bot._pending_close.remaining == pytest.approx(0.375)
    ledger.close_fill_fraction = 1.0
    await bot._check_pending_close()
    assert [r["qty"] for r in ledger.requests] == [0.5, 0.375]
    assert bot.active_position.qty == pytest.approx(0.5)
    assert bot.active_position.partial_done
    assert bot.active_position.stop_loss == 100.0
    assert sum(e.qty for e in close_events(bot)) == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_flip_ack_is_not_a_close_and_never_immediately_retries(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.close_fill_fraction = 0.0
    bot = make_bot(tmp_path, ledger)
    be_position(bot, ledger)
    target = HtfFvgEntry(tf="1h", weight=4, type=FVGType.BEARISH, high=105.0, low=101.0, ts_ms=1)
    await bot.handle_htf_flip(103.0, ENTRY_TS, target)
    await bot.handle_htf_flip(103.0, ENTRY_TS + 1, target)
    assert bot.active_position is not None
    assert len(ledger.requests) == 1
    assert close_events(bot) == []


@pytest.mark.asyncio
async def test_stop_retains_task_when_close_order_unconfirmed(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.close_fill_fraction = 0.0
    bot = make_bot(tmp_path, ledger)
    be_position(bot, ledger)
    bot.state = BotState.RUNNING
    await bot._emergency_close()
    with pytest.raises(RuntimeError, match="청산 주문 미확인"):
        await bot.stop()
    assert bot.state is BotState.RUNNING
    assert bot.entry_paused
    assert bot._pending_close is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("sl", [0.0, float("nan"), float("inf"), 101.0])
async def test_invalid_entry_sl_never_sends_order(tmp_path: Path, sl: float) -> None:
    ledger = OrderLedger()
    bot = make_bot(tmp_path, ledger)
    signal = setup()
    signal.stop_loss = sl
    await bot._execute_setup(signal, force_qty=1.0)
    assert ledger.requests == []


@pytest.mark.asyncio
async def test_emergency_cancels_entry_remainder_and_captures_cancel_fill_race(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.immediate_fill_fraction = 0.25
    ledger.cancel_fill_fraction = 0.5
    ledger.protection_fails = True
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    assert ledger.orders["1"]["status"] == "canceled"
    assert ledger.requests[1]["reduce_only"]
    assert ledger.requests[1]["qty"] == 40.0
    assert sum(e.qty for e in entries(bot)) == 40.0
    assert sum(e.qty for e in close_events(bot)) == 40.0
    assert bot.active_position is None
    assert bot._pending_entry is None
    assert bot._pending_close is None


@pytest.mark.asyncio
async def test_emergency_waits_for_entry_cancel_before_close(tmp_path: Path) -> None:
    ledger = OrderLedger()
    ledger.immediate_fill_fraction = 0.25
    ledger.protection_fails = True
    ledger.cancel_fails = True
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    assert bot.active_position.qty == 20.0
    assert bot._pending_close is not None
    assert bot._pending_entry is not None
    assert len(ledger.requests) == 1
    ledger.cancel_fails = False
    await bot._check_pending_close()
    assert ledger.requests[1]["qty"] == 20.0
    assert bot.active_position is None
    assert bot._pending_entry is None
    assert bot._pending_close is None


@pytest.mark.asyncio
async def test_historical_entry_fill_does_not_create_flat_or_opposite_position(tmp_path: Path) -> None:
    for opposite in (False, True):
        ledger = OrderLedger()
        bot = make_bot(tmp_path / str(opposite), ledger)
        await bot._execute_setup(setup())
        ledger.fill("1", 80.0)
        ledger.position = dict(contracts=2.0, side="short", entryPrice=100.0) if opposite else None
        assert not await bot._check_pending_entry()
        assert bot.active_position is None
        assert ledger.protection == []
        assert sum(e.qty for e in entries(bot)) == 80.0
        assert bot._pending_entry is None


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", ["filled_qty", "avg_fill_price"])
async def test_terminal_close_missing_fill_details_are_refetched(tmp_path: Path, missing: str) -> None:
    ledger = OrderLedger()
    ledger.close_fill_fraction = 0.0
    bot = make_bot(tmp_path, ledger)
    be_position(bot, ledger)
    await bot._emergency_close()
    ledger.fill_close("close1", 1.0)
    confirmed = ledger.orders["close1"][missing]
    ledger.orders["close1"][missing] = None
    await bot._check_pending_close()
    assert bot._pending_close is not None
    assert bot.active_position is not None
    assert close_events(bot) == []
    ledger.orders["close1"][missing] = confirmed
    await bot._check_pending_close()
    assert bot.active_position is None
    assert bot._pending_close is None
    assert len(ledger.requests) == 1
    assert sum(e.qty for e in close_events(bot)) == 1.0


@pytest.mark.asyncio
async def test_terminal_entry_missing_filled_qty_is_not_zero_fill(tmp_path: Path) -> None:
    ledger = OrderLedger()
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    ledger.fill("1", 80.0)
    ledger.orders["1"]["filled_qty"] = None
    assert await bot._check_pending_entry()
    assert bot._pending_entry is not None
    assert bot.active_position is None
    ledger.orders["1"]["filled_qty"] = 80.0
    assert not await bot._check_pending_entry()
    assert bot.active_position.qty == 80.0
    assert sum(e.qty for e in entries(bot)) == 80.0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["missing", "exception"])
async def test_pending_close_keeps_restoring_missing_sl_without_resubmission(
    tmp_path: Path, failure: str,
) -> None:
    ledger = OrderLedger()
    ledger.close_fill_fraction = 0.0
    ledger.protection_fails = True
    bot = make_bot(tmp_path, ledger)
    be_position(bot, ledger)
    ledger.position["stopLossPrice"] = 0.0
    assert not await bot._ensure_protective_sl(110.0, 2.0)
    assert ledger.position["stopLossPrice"] == 0.0
    intent = bot._pending_close
    assert intent is not None
    ledger.protection_fails = False
    ledger.order_unknown = failure == "missing"
    ledger.order_read_error = ExchangeError("order lookup denied") if failure == "exception" else None
    await bot._check_pending_close()
    assert ledger.position["stopLossPrice"] == 98.0
    assert bot.active_position.qty == 1.0
    assert bot._pending_close is intent
    assert len(ledger.requests) == 1
    assert close_events(bot) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("marker", [True, False, 1, "true"])
async def test_entry_clears_only_with_explicit_definitive_rejection(tmp_path: Path, marker: object) -> None:
    ledger = OrderLedger()
    ledger.reject_entry = True
    ledger.entry_rejection_code = 110007
    ledger.definitive_rejection_marker = marker
    bot = make_bot(tmp_path, ledger)
    await bot._execute_setup(setup())
    if marker is True:
        assert bot._pending_entry is None
        ledger.reject_entry = False
        await bot._execute_setup(setup())
        assert len(ledger.requests) == 2
    else:
        assert bot._pending_entry.order_id == "client:rejected"
        await bot._execute_setup(setup())
        assert len(ledger.requests) == 1
