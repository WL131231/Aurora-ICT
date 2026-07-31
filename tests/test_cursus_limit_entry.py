"""#LIMIT-ENTRY — Cursus 지정가 진입(지표 라벨 좌표) 검증.

2026-07-31 개발자 변경사항 마지막 항목. 개발자가 쓰는 Pine v6 지표 정본에서
브레이크아웃 "매수 지점"은 라벨이고, 좌표가 신호봉의 **저점(롱 ▲)/고점(숏 ▼)** 이다:

    if bullBreak → label.new(bar_index, low,  '▲', style_label_up)
    if bearBreak → label.new(bar_index, high, '▼', style_label_down)

시장가로 종가를 추격하는 대신 그 되돌림 가격에 지정가를 걸어 눌림에서 잡는다.
비용도 taker 왕복 0.11% → maker 0.02% 로 줄어든다.

핵심 안전 요건 — 체결 확인이 안 되는 동안 절대 두 번 주문하지 않고, 판정 실패
(네트워크 등)에는 취소하지 않는다(실제 체결된 포지션을 무주공산으로 두지 않기).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from aurora_ict.bot.bot_trend_instance import BotTrendInstance, _PendingLimitEntry
from aurora_ict.strategy.silver_bullet import Direction

BAR_MS = 3_600_000
SIG_TS = 1_700_000_000_000


def _client() -> AsyncMock:
    c = AsyncMock()
    c.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    c.fetch_ticker = AsyncMock(return_value=100.0)
    c.place_order = AsyncMock(return_value={"orderId": "L1"})
    c.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    c.modify_stop_loss = AsyncMock(return_value={"retCode": 0})
    c.fetch_position = AsyncMock(return_value=None)
    c.cancel_bot_orders = AsyncMock(return_value=1)
    return c


def _bot(client: AsyncMock, **kw: Any) -> BotTrendInstance:
    return BotTrendInstance(client=client, limit_entry=True, **kw)


def _bar(low: float = 98.5, high: float = 101.5) -> dict[str, float]:
    """신호봉 — 종가 100 기준으로 저점 98.5 / 고점 101.5."""
    return {"ts": float(SIG_TS), "low": low, "high": high}


def _df(ts: int) -> pd.DataFrame:
    """_handle_pending_limit 이 보는 최소 DataFrame(마지막 행의 ts 만 쓴다)."""
    return pd.DataFrame({"ts": [ts], "close": [100.0]})


def _pending(direction: Direction = Direction.LONG, ttl: int = 3) -> _PendingLimitEntry:
    return _PendingLimitEntry(
        direction=direction, price=98.5, qty=1.0, stop=96.53, order_id="L1",
        signal_bar_ms=SIG_TS, expire_bar_ms=SIG_TS + ttl * BAR_MS,
    )


# ---- 주문 발행 ----

@pytest.mark.asyncio
async def test_long_places_limit_at_signal_bar_low() -> None:
    """★ 롱 = 신호봉 **저점**에 지정가 + SL 동봉. 포지션은 아직 안 생긴다."""
    c = _client()
    bot = _bot(c)

    await bot._open(Direction.LONG, price=100.0, bar=_bar())

    kw = c.place_order.await_args.kwargs
    assert kw["side"] == "buy"
    assert kw["price"] == pytest.approx(98.5)         # 종가 100 이 아니라 저점
    assert kw["stop_loss"] == pytest.approx(98.5 * 0.98)  # 지정가 기준 고정 2%
    assert bot.active_position is None                # 체결 전 = 무포지션
    assert bot._pending_limit is not None
    assert bot._pending_limit.expire_bar_ms == SIG_TS + 3 * BAR_MS


@pytest.mark.asyncio
async def test_short_places_limit_at_signal_bar_high() -> None:
    """숏 = 신호봉 **고점**에 지정가, SL 은 위."""
    c = _client()
    bot = _bot(c)

    await bot._open(Direction.SHORT, price=100.0, bar=_bar())

    kw = c.place_order.await_args.kwargs
    assert kw["side"] == "sell"
    assert kw["price"] == pytest.approx(101.5)
    assert kw["stop_loss"] == pytest.approx(101.5 * 1.02)


@pytest.mark.asyncio
async def test_market_fallback_when_no_bar() -> None:
    """봉 정보 없는 경로는 기존 시장가 — price=None 이고 즉시 포지션 생성."""
    c = _client()
    bot = _bot(c)

    await bot._open(Direction.LONG, price=100.0, bar=None)

    assert c.place_order.await_args.kwargs["price"] is None
    assert bot.active_position is not None
    assert bot._pending_limit is None


@pytest.mark.asyncio
async def test_disabled_uses_market_even_with_bar() -> None:
    """limit_entry=False 면 봉을 줘도 시장가(기존 동작 불변)."""
    c = _client()
    bot = BotTrendInstance(client=c)  # limit_entry 기본 False

    await bot._open(Direction.LONG, price=100.0, bar=_bar())

    assert c.place_order.await_args.kwargs["price"] is None
    assert bot._pending_limit is None


@pytest.mark.asyncio
async def test_bad_coordinate_skips_entry() -> None:
    """좌표 이상(0)이면 주문하지 않는다 — 시장가 폴백도 안 함(정본 이탈 방지)."""
    c = _client()
    bot = _bot(c)

    await bot._open(Direction.LONG, price=100.0, bar={"ts": float(SIG_TS), "low": 0.0,
                                                      "high": 101.5})

    assert c.place_order.await_count == 0
    assert bot._pending_limit is None


# ---- 체결 / 취소 ----

@pytest.mark.asyncio
async def test_fill_creates_position_and_clears_pending() -> None:
    """지정가 체결 → 거래소 entry 기준 포지션 확정, pending 해제."""
    c = _client()
    c.fetch_position = AsyncMock(return_value={
        "contracts": 1.0, "side": "long", "entryPrice": 98.5,
    })
    bot = _bot(c)
    bot._pending_limit = _pending()

    await bot._handle_pending_limit(_df(SIG_TS + BAR_MS), price=99.0, sig=None)

    pos = bot.active_position
    assert pos is not None
    assert pos.entry == pytest.approx(98.5)
    assert pos.stop == pytest.approx(98.5 * 0.98)                    # 고정 2%
    assert pos.tp_prices == pytest.approx([99.485, 100.47, 101.455, 102.44])
    assert bot._pending_limit is None


@pytest.mark.asyncio
async def test_pending_held_before_ttl() -> None:
    """TTL 안이고 역신호도 없으면 그대로 대기 — 취소하지 않는다."""
    c = _client()
    bot = _bot(c)
    bot._pending_limit = _pending()

    await bot._handle_pending_limit(_df(SIG_TS + 2 * BAR_MS), price=100.0, sig=None)

    assert bot._pending_limit is not None
    assert c.cancel_bot_orders.await_count == 0


@pytest.mark.asyncio
async def test_ttl_expiry_cancels() -> None:
    """TTL(3봉)을 넘긴 봉이 오면 미체결 주문 취소."""
    c = _client()
    bot = _bot(c)
    bot._pending_limit = _pending()

    await bot._handle_pending_limit(_df(SIG_TS + 4 * BAR_MS), price=100.0, sig=None)

    assert bot._pending_limit is None
    assert c.cancel_bot_orders.await_count == 1


@pytest.mark.asyncio
async def test_opposite_signal_cancels() -> None:
    """대기 중 반대 신호가 뜨면 TTL 전이라도 취소 — 추세가 뒤집혔다."""
    c = _client()
    bot = _bot(c)
    bot._pending_limit = _pending(Direction.LONG)

    await bot._handle_pending_limit(_df(SIG_TS + BAR_MS), price=100.0,
                                    sig=Direction.SHORT)

    assert bot._pending_limit is None
    assert c.cancel_bot_orders.await_count == 1


@pytest.mark.asyncio
async def test_same_signal_does_not_cancel() -> None:
    """같은 방향 신호가 재차 떠도 유지(중복 주문·불필요 취소 금지)."""
    c = _client()
    bot = _bot(c)
    bot._pending_limit = _pending(Direction.LONG)

    await bot._handle_pending_limit(_df(SIG_TS + BAR_MS), price=100.0,
                                    sig=Direction.LONG)

    assert bot._pending_limit is not None
    assert c.cancel_bot_orders.await_count == 0


@pytest.mark.asyncio
async def test_fetch_failure_keeps_pending() -> None:
    """★ 체결 확인 실패 시 취소하지 않는다 — 체결된 포지션을 무주공산으로 두지 않기."""
    c = _client()
    c.fetch_position = AsyncMock(side_effect=RuntimeError("network"))
    bot = _bot(c)
    bot._pending_limit = _pending()

    await bot._handle_pending_limit(_df(SIG_TS + 9 * BAR_MS), price=100.0, sig=None)

    assert bot._pending_limit is not None      # TTL 지났어도 유지
    assert c.cancel_bot_orders.await_count == 0


@pytest.mark.asyncio
async def test_cancel_failure_keeps_pending_for_retry() -> None:
    """취소 API 가 실패하면 pending 을 남겨 다음 step 에서 재시도."""
    c = _client()
    c.cancel_bot_orders = AsyncMock(side_effect=RuntimeError("rate limit"))
    bot = _bot(c)
    bot._pending_limit = _pending()

    await bot._handle_pending_limit(_df(SIG_TS + 4 * BAR_MS), price=100.0, sig=None)

    assert bot._pending_limit is not None


@pytest.mark.asyncio
async def test_partial_fill_cancels_remainder() -> None:
    """부분 체결 — 잔여 주문을 정리해 봇 모르게 수량이 늘지 않게 한다."""
    c = _client()
    c.fetch_position = AsyncMock(return_value={
        "contracts": 0.4, "side": "long", "entryPrice": 98.5,
    })
    bot = _bot(c)
    bot._pending_limit = _pending()

    await bot._handle_pending_limit(_df(SIG_TS + BAR_MS), price=99.0, sig=None)

    assert bot.active_position is not None
    assert bot.active_position.qty == pytest.approx(0.4)  # 실제 체결량 기준
    assert c.cancel_bot_orders.await_count == 1


def test_tf_ms_parsing() -> None:
    """TTL 계산의 기준 — timeframe 문자열 → 봉 길이(ms)."""
    c = _client()
    assert _bot(c, timeframe="1h")._tf_ms() == 3_600_000
    assert _bot(c, timeframe="15m")._tf_ms() == 900_000
    assert _bot(c, timeframe="4h")._tf_ms() == 14_400_000
    assert _bot(c, timeframe="1d")._tf_ms() == 86_400_000
    assert _bot(c, timeframe="???")._tf_ms() == 3_600_000  # 폴백
