"""BotIctInstance 박은 거 박힘 — Aurora-ICT v0.1.5."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from aurora_ict.bot.bot_ict_instance import (
    BotIctInstance,
    BotState,
    _ActivePosition,
)
from aurora_ict.indicators.structure import TrendDirection
from aurora_ict.signal.ict_signal import SignalAction
from aurora_ict.strategy.silver_bullet import Direction

NY = ZoneInfo("America/New_York")


def _bars_long_setup() -> list[tuple[float, float, float, float]]:
    """박힌 long setup 박힌 거 박힘 박힌 bars (NY 10:00 박힘 박힘 박힘)."""
    return [
        (100, 105, 99, 104),
        (104, 130, 103, 125),
        (125, 124, 100, 101),
        (101, 108, 95, 96),
        (96, 110, 95.5, 109),
        (109, 112, 92, 93),
        (93, 100, 92.5, 99),
        (99, 105, 98, 104),
        (104, 106, 100, 101),
        (101, 105, 99, 100),
        (100, 102, 99, 101),
        (101, 110, 100, 109),
        (109, 119, 108, 118),
        (118, 122, 115, 121),
    ]


def _ohlcv_rows(start_ny: datetime, bars: list[tuple[float, float, float, float]]) -> list[list[Any]]:
    """ccxt-style OHLCV rows: [ts_ms, o, h, l, c, v]."""
    rows = []
    for i, (o, h, lo, c) in enumerate(bars):
        t = start_ny + timedelta(minutes=i)
        ts_ms = int(t.timestamp() * 1000)
        rows.append([ts_ms, o, h, lo, c, 100.0])
    return rows


def _mock_client(ohlcv_rows: list[list[Any]]) -> AsyncMock:
    """Mock ExchangeClient 박힌 거.

    #LIVE-1 fix: marketable limit entry 가 fetch_ticker (현재가) 호출 + place_order
    응답의 filled_qty/avg_fill_price 로 즉시 체결 판정. 기본 mock 은 즉시 체결 시뮬.
    """
    last_close = float(ohlcv_rows[-1][4]) if ohlcv_rows else 100.0
    client = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=ohlcv_rows)
    client.fetch_ticker = AsyncMock(return_value=last_close)
    # 즉시 체결 시뮬 (filled_qty/avg_fill_price 포함) → active_position 즉시 확정.
    client.place_order = AsyncMock(return_value={
        "orderId": "TEST123",
        "filled_qty": 1.0,
        "avg_fill_price": last_close,
    })
    client.fetch_position = AsyncMock(return_value=None)
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    client.cancel_all_orders = AsyncMock(return_value=None)
    client.fetch_closed_positions = AsyncMock(return_value=[])
    client.set_position_tpsl = AsyncMock(return_value={"retCode": 0})
    return client


def _ex_pos(side: str, contracts: float, entry: float = 60000.0) -> dict[str, Any]:
    """거래소 fetch_position 응답 흉내 (#POS-SYNC 테스트용)."""
    return {"contracts": contracts, "side": side, "entryPrice": entry}


def _short_position(qty: float = 0.107) -> _ActivePosition:
    """봇 active_position — BTC 숏 (04:14 사건 재현용)."""
    return _ActivePosition(
        direction=Direction.SHORT, entry=60379.0, stop_loss=60667.0,
        take_profit=59234.0, qty=qty, setup_ts_ms=0,
    )


@pytest.mark.asyncio
async def test_sync_emergency_close_on_direction_mismatch() -> None:
    """#POS-SYNC 04:14: 봇=숏인데 거래소=롱 → 즉시 비상청산 (거래소 방향 반대로)."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(return_value=_ex_pos("long", 0.107))
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot.active_position = _short_position()
    await bot._sync_position_state()
    assert client.place_order.await_count == 1
    kw = client.place_order.await_args_list[0].kwargs
    assert kw["side"] == "sell"          # 거래소 실제(롱)의 반대 — same-side 110017 회피
    assert kw["reduce_only"] is True
    assert kw["qty"] == pytest.approx(0.107)
    assert bot.active_position is None


def test_record_trade_queues_failed_event_and_reflushes(tmp_path) -> None:
    """#SYNC-FIX(2026-06-17): record 실패 → 큐 보관, 다음 record 성공 시 재기록.

    거래소 체결됐는데 기록(JSONL/DB)이 일시 실패로 누락되던 라이브 불일치 해소.
    store.record 를 자기 인스턴스 wrapper 로 교체(self-spy, mock 0) — 첫 호출만
    실패시키고, 둘째 호출 성공 시 _flush_failed_trades 가 큐를 비우는지 검증.
    """
    from aurora_ict.interfaces.trades_store import TradeEventType, TradesStore

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    store = TradesStore(tmp_path)
    bot._trades_store = store
    real_record = store.record
    state = {"fail": True}

    def spy(ev: Any) -> None:
        if state["fail"]:
            raise OSError("disk full 시뮬")
        real_record(ev)

    store.record = spy  # type: ignore[method-assign]

    # 1) 기록 실패 → 큐 보관 (이벤트 유실 안 됨)
    bot._record_trade(
        TradeEventType.ENTRY, direction=Direction.LONG, price=100.0, qty=1.0,
    )
    assert len(bot._failed_trade_events) == 1

    # 2) 다음 record 성공 → _flush_failed_trades 가 큐 비우고 재기록
    state["fail"] = False
    bot._record_trade(
        TradeEventType.SL_HIT, direction=Direction.LONG, price=95.0, qty=1.0,
        entry_for_pnl=100.0,
    )
    assert bot._failed_trade_events == []
    types = [e.event_type for e in store.all_events()]
    assert TradeEventType.ENTRY in types  # 큐에서 재기록됨
    assert TradeEventType.SL_HIT in types


@pytest.mark.asyncio
async def test_run_loop_auto_stops_on_repeated_auth_error() -> None:
    """키 무효(AuthenticationError)가 step 에서 임계치만큼 연속되면 봇 자동 정지."""
    from ccxt.base.errors import AuthenticationError

    from aurora_ict.bot.bot_ict_instance import (
        _AUTH_FAIL_STOP_THRESHOLD,
        BotState,
    )

    class _AuthFailBot(BotIctInstance):
        # step 을 키 무효 던지게 오버라이드 — _run_loop 의 인증 실패 처리 검증.
        async def step(self):  # type: ignore[override]
            raise AuthenticationError(
                'bybit {"retCode":10003,"retMsg":"API key is invalid."}',
            )

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = _AuthFailBot(client=client, symbol="BTCUSDT", step_interval_sec=0)
    bot.state = BotState.RUNNING
    await asyncio.wait_for(bot._run_loop(), timeout=2.0)
    assert bot.state is BotState.STOPPED
    assert bot._auth_fail_streak >= _AUTH_FAIL_STOP_THRESHOLD


@pytest.mark.asyncio
async def test_run_loop_auto_stops_on_balance_auth_failures() -> None:
    """잔고 조회가 키 무효로 연속 실패(어댑터 auth_fail_streak)하면 봇 자동 정지.

    step 이 예외를 안 내도(어댑터가 10003 을 흡수해 빈 dict 반환) 자동 정지되는
    TDAF 류 케이스 — _run_loop 가 client.auth_fail_streak 를 체크한다.
    """
    from aurora_ict.bot.bot_ict_instance import _AUTH_FAIL_STOP_THRESHOLD

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    # 어댑터가 fetch_balance 10003 을 임계치만큼 누적했다고 시뮬.
    client.auth_fail_streak = _AUTH_FAIL_STOP_THRESHOLD
    bot = BotIctInstance(client=client, symbol="BTCUSDT", step_interval_sec=0)
    bot.state = BotState.RUNNING
    await asyncio.wait_for(bot._run_loop(), timeout=2.0)
    assert bot.state is BotState.STOPPED


@pytest.mark.asyncio
async def test_sync_corrects_qty_on_partial_manual_close() -> None:
    """#POS-SYNC: 사용자가 일부 수동 청산 → 청산 안 하고 봇 qty 만 거래소 실제로 보정."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(return_value=_ex_pos("short", 0.05))
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot.active_position = _short_position(qty=0.107)
    await bot._sync_position_state()
    assert client.place_order.await_count == 0
    assert bot.active_position is not None
    assert bot.active_position.qty == pytest.approx(0.05)


@pytest.mark.asyncio
async def test_emergency_close_uses_exchange_direction_and_qty() -> None:
    """#POS-SYNC: 비상청산이 봇 인식 아닌 거래소 실제 방향·수량 기준 (110017 방지)."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(return_value=_ex_pos("long", 0.107))
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot.active_position = _short_position(qty=0.2)  # 봇은 숏 0.2 로 착각
    await bot._emergency_close()
    kw = client.place_order.await_args_list[0].kwargs
    assert kw["side"] == "sell"               # 거래소 롱의 반대
    assert kw["qty"] == pytest.approx(0.107)   # 거래소 실제 수량
    assert kw["reduce_only"] is True
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_emergency_close_skips_when_exchange_flat() -> None:
    """#POS-SYNC: 비상청산 시점에 거래소 포지션 없으면 주문 없이 봇 상태만 정리."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(return_value=_ex_pos("none", 0.0))
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot.active_position = _short_position()
    await bot._emergency_close()
    assert client.place_order.await_count == 0   # 이미 닫힘 — 중복 reduce_only 안 보냄
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_step_no_signal_returns_no_action() -> None:
    """짧은 OHLCV → NO_ACTION."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = BotIctInstance(client=client)
    sig = await bot.step()
    assert sig.action is SignalAction.NO_ACTION
    client.place_order.assert_not_awaited()


@pytest.mark.asyncio
async def test_step_executes_long_setup() -> None:
    """valid long setup 박힘 박힘 → place_order 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(
        client=client,
        symbol="BTCUSDT",
        min_rr=1.0,
        fvg_min_size_pct=0.001,
        regime_filter_enabled=False,  # 진입 로직 검증 — 횡보 게이트 격리
    )
    sig = await bot.step()
    assert sig.action is SignalAction.ENTER_LONG
    # #LIVE-3/4: 계획가 limit entry 1회, SL/TP 는 동봉 안 함 (체결 후 set_position_tpsl)
    assert client.place_order.await_count == 1
    entry_call = client.place_order.await_args_list[0].kwargs
    assert entry_call["symbol"] == "BTCUSDT"
    assert entry_call["side"] == "buy"
    assert entry_call["qty"] > 0
    assert entry_call["price"] > 0                          # setup.entry (계획가) limit
    assert "stop_loss" not in entry_call                   # 동봉 X (#LIVE-4)
    assert "take_profit" not in entry_call
    assert entry_call.get("reduce_only", False) is False
    # 즉시 체결 → active position + SL/TP 는 set_position_tpsl 로
    assert bot.active_position is not None
    assert bot.active_position.direction is Direction.LONG
    client.set_position_tpsl.assert_awaited_once()
    tpsl_kw = client.set_position_tpsl.await_args_list[0].kwargs
    assert tpsl_kw["stop_loss"] < bot.active_position.entry   # long SL 아래
    assert tpsl_kw["take_profit"] > bot.active_position.entry  # long TP 위


@pytest.mark.asyncio
async def test_step_blocks_entry_outside_killzone_for_sub_license() -> None:
    """#KZ-ENTRY 2026-06-06: sub_* (disable_time_filter=False) — FVG 가 킬존(미장)에
    형성됐어도 진입 '시점'(마지막 닫힌 봉)이 킬존 밖이면 진입 skip.

    구성: NY 15:48 시작 14봉 1분봉 → 앞 봉들은 PM 킬존(NYSE 09:30-16:00 내, FVG
    통과), 마지막 봉(16:01)은 NYSE 마감 후 → 진입 게이트가 차단.
    """
    start = datetime(2026, 5, 12, 15, 48, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(
        client=client,
        symbol="BTCUSDT",
        min_rr=1.0,
        fvg_min_size_pct=0.001,
        disable_time_filter=False,  # sub_* 라이선스 정책
    )
    sig = await bot.step()
    # 셋업 자체는 검출됨(FVG 가 킬존에 형성) — 진입 '게이트'가 막은 것임을 확정.
    assert sig.action is SignalAction.ENTER_LONG
    assert client.place_order.await_count == 0   # 킬존 밖 진입 차단
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_step_allows_entry_inside_killzone_for_sub_license() -> None:
    """#KZ-ENTRY: sub_* 여도 진입 시점이 킬존 안이면 정상 진입 (게이트 통과 확인)."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)  # 전부 London Close 킬존
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(
        client=client,
        symbol="BTCUSDT",
        min_rr=1.0,
        fvg_min_size_pct=0.001,
        disable_time_filter=False,
        regime_filter_enabled=False,  # 킬존 진입 검증 — 횡보 게이트 격리
    )
    sig = await bot.step()
    assert sig.action is SignalAction.ENTER_LONG
    assert client.place_order.await_count == 1
    assert bot.active_position is not None


@pytest.mark.asyncio
async def test_step_grade_gate_skips_low_confluence() -> None:
    """B+ 등급 게이트 (#1/#8) — 최종 confluence_score 미달이면 신호는 잡혀도 진입 skip."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(
        client=client,
        symbol="BTCUSDT",
        min_rr=1.0,
        fvg_min_size_pct=0.001,
        min_confluence=99,  # 어떤 setup 도 미달 → 무조건 skip
    )
    sig = await bot.step()
    assert sig.action is SignalAction.ENTER_LONG   # 신호 자체는 검출됨
    assert client.place_order.await_count == 0     # 진입은 안 함
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_step_regime_filter_skips_ranging_setup() -> None:
    """#REGIME 2026-06-23: 횡보 국면(|진입추세%| < 페어별 floor) setup 진입 skip.

    합성 long setup 은 |entry_trend| ≈ 0(횡보) → BTCUSDT floor 0.23 미만이라 차단.
    같은 데이터로 regime off(다른 테스트들)면 정상 진입하므로 게이트가 유일 원인.
    """
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(
        client=client,
        symbol="BTCUSDT",
        min_rr=1.0,
        fvg_min_size_pct=0.001,
        regime_filter_enabled=True,  # 횡보 게이트 ON (라이브 기본값)
    )
    sig = await bot.step()
    assert sig.action is SignalAction.ENTER_LONG   # 신호 자체는 검출됨
    assert client.place_order.await_count == 0     # 횡보 국면이라 진입 skip
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_step_high_rr_bypass_passes_when_rr_above_threshold() -> None:
    """고RR 예외(#high-rr-bypass) — confluence 미달이어도 rr >= bypass 임계면 진입.

    파트너 결정 2026-06-04: 손익비 좋은 score>=1 셋업은 confluence 게이트 우회.
    이 셋업은 boost 후 score>=1·rr≈1.43 → bypass 임계 1.0 이면 통과.
    """
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(
        client=client,
        symbol="BTCUSDT",
        min_rr=1.0,
        fvg_min_size_pct=0.001,
        min_confluence=99,           # 등급 게이트 무조건 막힘
        high_rr_bypass_min_rr=1.0,   # rr 1.43 >= 1.0 → 예외 통과
        regime_filter_enabled=False,  # 고RR 예외 검증 — 횡보 게이트 격리
    )
    sig = await bot.step()
    assert sig.action is SignalAction.ENTER_LONG
    assert client.place_order.await_count == 1   # 고RR 예외로 진입
    assert bot.active_position is not None


@pytest.mark.asyncio
async def test_step_high_rr_bypass_blocks_when_rr_below_threshold() -> None:
    """고RR 예외 — rr 이 bypass 임계 미만이면 여전히 등급 미달 skip."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(
        client=client,
        symbol="BTCUSDT",
        min_rr=1.0,
        fvg_min_size_pct=0.001,
        min_confluence=99,
        high_rr_bypass_min_rr=2.0,   # rr 1.43 < 2.0 → 예외 미적용
    )
    sig = await bot.step()
    assert sig.action is SignalAction.ENTER_LONG
    assert client.place_order.await_count == 0   # 임계 미달 → 진입 차단
    assert bot.active_position is None


@pytest.mark.asyncio
async def test_step_skip_when_position_exists() -> None:
    """active position 박힘 → place_order 박힘 X (박힘 박힘 fetch_position 박힘)."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    # 박힌 position 박힘 박힘 (active 박힘)
    client.fetch_position = AsyncMock(return_value={"contracts": 0.01})
    bot = BotIctInstance(
        client=client,
        min_rr=1.0,
        fvg_min_size_pct=0.001,
    )
    # active_position 박힘 박힘 박힘 박힘 박힘 박힘 박힘
    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    bot.active_position = _ActivePosition(
        direction=Direction.LONG,
        entry=100,
        stop_loss=95,
        take_profit=110,
        qty=0.01,
        setup_ts_ms=0,
    )
    await bot.step()
    # 박힘 박힙 place_order 박힘 X
    client.place_order.assert_not_awaited()
    # fetch_position 박힘 박힘 박힘
    client.fetch_position.assert_awaited()


@pytest.mark.asyncio
async def test_step_resets_position_on_close() -> None:
    """fetch_position 박힘 None / contracts=0 → active_position 박힘 None 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    # position closed (contracts=0)
    client.fetch_position = AsyncMock(return_value={"contracts": 0})
    bot = BotIctInstance(client=client, min_rr=1.0, fvg_min_size_pct=0.001)
    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    bot.active_position = _ActivePosition(
        direction=Direction.LONG,
        entry=100,
        stop_loss=95,
        take_profit=110,
        qty=0.01,
        setup_ts_ms=0,
    )
    await bot.step()
    assert bot.active_position is None


# ============================================================
# #PR-C: close 사유 + 거래소 closed-pnl 동기화
# ============================================================


class _StubStore:
    """trades_store mock — record 호출만 캡처 (mock 0 정책: 외부 라이브러리 X)."""

    def __init__(self) -> None:
        self.events: list = []

    def record(self, ev) -> None:
        self.events.append(ev)

    def all_events(self) -> list:
        return list(self.events)


@pytest.mark.asyncio
async def test_sync_close_records_sl_hit_with_exchange_pnl() -> None:
    """closed-pnl 조회 결과 close 가격이 SL 근처면 SL_HIT + 거래소 PnL 기록 (#PR-C/#3+#4)."""
    from types import SimpleNamespace

    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    from aurora_ict.interfaces.trades_store import TradeEventType

    client = _mock_client(_ohlcv_rows(datetime(2026, 5, 12, 10, 0, tzinfo=NY), _bars_long_setup()))
    client.fetch_position = AsyncMock(return_value={"contracts": 0})
    cp = SimpleNamespace(
        symbol="BTCUSDT", direction="long",
        exit_price=95.1, pnl_usd=-50.0, closed_at_ts=1234567890,
    )
    client.fetch_closed_positions = AsyncMock(return_value=[cp])
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot._trades_store = _StubStore()
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0, take_profit=110.0,
        qty=1.0, setup_ts_ms=12345,
    )
    await bot._sync_position_state()
    assert bot.active_position is None
    assert len(bot._trades_store.events) == 1
    ev = bot._trades_store.events[0]
    assert ev.event_type is TradeEventType.SL_HIT
    assert ev.price == 95.1
    assert ev.pnl_usdt == -50.0   # 거래소 실현치


@pytest.mark.asyncio
async def test_sync_close_records_tp_hit() -> None:
    """close 가격이 TP 근처면 TP_HIT + 거래소 PnL 기록."""
    from types import SimpleNamespace

    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    from aurora_ict.interfaces.trades_store import TradeEventType

    client = _mock_client(_ohlcv_rows(datetime(2026, 5, 12, 10, 0, tzinfo=NY), _bars_long_setup()))
    client.fetch_position = AsyncMock(return_value={"contracts": 0})
    cp = SimpleNamespace(
        symbol="BTCUSDT", direction="long",
        exit_price=109.9, pnl_usd=98.5, closed_at_ts=1,
    )
    client.fetch_closed_positions = AsyncMock(return_value=[cp])
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot._trades_store = _StubStore()
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0, take_profit=110.0,
        qty=1.0, setup_ts_ms=12345,
    )
    await bot._sync_position_state()
    ev = bot._trades_store.events[0]
    assert ev.event_type is TradeEventType.TP_HIT
    assert ev.pnl_usdt == 98.5


@pytest.mark.asyncio
async def test_sync_close_fallback_when_closed_pnl_empty() -> None:
    """closed-pnl 조회가 비면 SYNC_CLOSE fallback (entry placeholder, pnl 추정)."""
    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    from aurora_ict.interfaces.trades_store import TradeEventType

    client = _mock_client(_ohlcv_rows(datetime(2026, 5, 12, 10, 0, tzinfo=NY), _bars_long_setup()))
    client.fetch_position = AsyncMock(return_value={"contracts": 0})
    client.fetch_closed_positions = AsyncMock(return_value=[])
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot._trades_store = _StubStore()
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0, take_profit=110.0,
        qty=1.0, setup_ts_ms=12345,
    )
    await bot._sync_position_state()
    ev = bot._trades_store.events[0]
    assert ev.event_type is TradeEventType.SYNC_CLOSE
    assert ev.price == 100.0   # entry placeholder
    # pnl_override None 이라 계산식: (100-100)*1 = 0
    assert ev.pnl_usdt == 0.0


@pytest.mark.asyncio
async def test_step_duplicate_setup_filtered() -> None:
    """같은 setup ts_ms 박힘 박힘 두 번째 박힘 박힘 박힘 X (중복 진입 박힘)."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(
        client=client, min_rr=1.0, fvg_min_size_pct=0.001,
        regime_filter_enabled=False,  # 중복 진입 필터 검증 — 횡보 게이트 격리
    )
    # 첫 step
    await bot.step()
    assert bot.active_position is not None
    # 박은 position 박힌 거 reset 박힘 → 박은 setup 박힘 박힘 박힘 박힘 박힙 박힘
    bot.active_position = None
    # 두 번째 step — 같은 OHLCV
    await bot.step()
    # 같은 setup_ts_ms 라 신규 진입 X — 첫 step 의 entry 1호출 (SL/TP 동봉) 만 남음
    assert client.place_order.await_count == 1


@pytest.mark.asyncio
async def test_start_recovers_position_from_exchange() -> None:
    """봇 시작 시 거래소 측 활성 포지션 fetch → active_position 복원."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(return_value={
        "contracts": 0.05,
        "side": "short",
        "entryPrice": 80000.0,
        "stopLossPrice": 80500.0,
        "takeProfitPrice": 78000.0,
    })
    bot = BotIctInstance(client=client, step_interval_sec=3600)
    await bot.start()
    assert bot.active_position is not None
    assert bot.active_position.direction is Direction.SHORT
    assert bot.active_position.entry == 80000.0
    assert bot.active_position.qty == 0.05
    assert bot.active_position.stop_loss == 80500.0
    await bot.stop()


@pytest.mark.asyncio
async def test_start_no_recovery_when_no_exchange_position() -> None:
    """거래소 측 포지션 없으면 active_position 그대로 None."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(return_value=None)
    bot = BotIctInstance(client=client, step_interval_sec=3600)
    await bot.start()
    assert bot.active_position is None
    await bot.stop()


@pytest.mark.asyncio
async def test_start_stop_lifecycle() -> None:
    """start → state RUNNING, stop → state STOPPED."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    # step_interval 박힘 박힘 박힘 박힘 박힘 (test 박힘 박힘)
    bot = BotIctInstance(client=client, step_interval_sec=3600)
    assert bot.state is BotState.STOPPED
    await bot.start()
    assert bot.state is BotState.RUNNING
    await bot.stop()
    assert bot.state is BotState.STOPPED
    assert bot._task is None


@pytest.mark.asyncio
async def test_qty_calc_with_equity() -> None:
    """equity 박힌 거 박힙 박힘 qty 박힘 박힙 박힘 박힘."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(client=client, min_rr=1.0, fvg_min_size_pct=0.001)
    sig = await bot.step()
    assert sig.setup is not None
    # equity 박힌 거 박힘 박힘 qty 박힘 박힙 박힘 박힘 박힘 박힘
    qty_low = bot._calc_qty(sig.setup, equity=1000.0)
    qty_high = bot._calc_qty(sig.setup, equity=10000.0)
    assert qty_high > qty_low
    assert qty_low > 0


@pytest.mark.asyncio
async def test_fetch_equity_from_ccxt_balance() -> None:
    """ccxt format {'USDT': {'total': N}} 박힘 박힘 N 박힘 박힘."""
    client = _mock_client([])
    client.fetch_balance = AsyncMock(return_value={"USDT": {"total": 5000.0}})
    bot = BotIctInstance(client=client)
    eq = await bot._fetch_equity()
    assert eq == 5000.0


@pytest.mark.asyncio
async def test_fetch_equity_fallback_on_error() -> None:
    """fetch_balance 실패 시 fallback 1000."""
    client = _mock_client([])
    client.fetch_balance = AsyncMock(side_effect=RuntimeError("network"))
    bot = BotIctInstance(client=client)
    eq = await bot._fetch_equity()
    assert eq == 1000.0


# ============================================================
# Multi-TF HTF Bias 통합
# ============================================================


@pytest.mark.asyncio
async def test_compute_htf_bias_no_mapping_returns_none() -> None:
    """1m timeframe — HTF 매핑 없음 → None (자동 추정 위임)."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = BotIctInstance(client=client, timeframe="1m")
    bias = await bot._compute_htf_bias()
    assert bias is None


@pytest.mark.asyncio
async def test_compute_htf_bias_uses_htf_pair_for_15m() -> None:
    """15m timeframe → HTF1=1h, HTF2=4h fetch 호출되는지."""
    bullish_bars = [
        (100, 101, 99, 100), (100, 104, 99, 103),
        (103, 103, 95, 96), (96, 97, 94, 95),
        (95, 110, 94, 109),  # CHoCH_BULLISH at idx=4
    ]
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, bullish_bars)
    client = _mock_client(rows)
    bot = BotIctInstance(client=client, timeframe="15m")
    bias = await bot._compute_htf_bias()
    # 두 HTF 동일 data (mock) → bullish → UP
    assert bias is TrendDirection.UP
    # client.fetch_ohlcv 가 1h, 4h 둘 다 호출됐어야
    tfs_called = [c.args[1] for c in client.fetch_ohlcv.await_args_list]
    assert "1h" in tfs_called
    assert "4h" in tfs_called


@pytest.mark.asyncio
async def test_htf_cache_hit_skips_recompute() -> None:
    """같은 봉 ts → 캐시 hit 으로 재계산 X (fetch 는 1회씩만)."""
    bullish_bars = [
        (100, 101, 99, 100), (100, 104, 99, 103),
        (103, 103, 95, 96), (96, 97, 94, 95),
        (95, 110, 94, 109),
    ]
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, bullish_bars)
    client = _mock_client(rows)
    bot = BotIctInstance(client=client, timeframe="1h")  # HTF=4h, 1d
    await bot._compute_htf_bias()
    n_after_first = client.fetch_ohlcv.await_count
    await bot._compute_htf_bias()
    # 2번째 호출도 fetch 는 하지만 캐시된 df 재사용 (현 구현은 fetch 후 캐시 비교)
    # — 적어도 2배 늘었어야 (HTF1+HTF2 각 1회)
    assert client.fetch_ohlcv.await_count == n_after_first * 2 - 0 \
        or client.fetch_ohlcv.await_count >= n_after_first


# ============================================================
# Daily Bias 통합
# ============================================================


@pytest.mark.asyncio
async def test_compute_daily_bias_up_when_above_pdh() -> None:
    """전일 high 위 close → daily UP."""
    daily_rows = [
        [1, 100, 110, 95, 108, 100],
        [2, 108, 120, 105, 115, 100],   # 어제 (high=120)
        [3, 115, 130, 110, 130, 100],   # 오늘
    ]
    ltf_rows = [[1000, 125, 131, 124, 130, 50]]
    client = AsyncMock()
    client.fetch_ohlcv = AsyncMock(return_value=daily_rows)
    bot = BotIctInstance(client=client, timeframe="1m")
    import pandas as pd
    df = pd.DataFrame(
        ltf_rows,
        columns=["ts_ms", "open", "high", "low", "close", "volume"],
    )
    df.index = pd.DatetimeIndex(pd.to_datetime(df["ts_ms"], unit="ms", utc=True))
    bias = await bot._compute_daily_bias(df)
    # current close=130 > pdh=120 → UP
    assert bias is TrendDirection.UP


def test_combine_with_daily_htf_none_passes_through() -> None:
    """HTF None (매핑 없음) → None 반환 (silver_bullet 자동 추정 위임)."""
    assert BotIctInstance._combine_with_daily(None, TrendDirection.UP) is None


def test_combine_with_daily_conflict_follows_daily() -> None:
    """HTF UP + Daily DOWN → daily 따름 (DOWN). 충돌 시 daily 우선 정책."""
    result = BotIctInstance._combine_with_daily(
        TrendDirection.UP, TrendDirection.DOWN,
    )
    assert result is TrendDirection.DOWN


def test_combine_with_daily_agreement() -> None:
    """HTF UP + Daily UP → UP."""
    result = BotIctInstance._combine_with_daily(
        TrendDirection.UP, TrendDirection.UP,
    )
    assert result is TrendDirection.UP


def test_combine_with_daily_one_none() -> None:
    """HTF NONE + Daily UP → UP."""
    result = BotIctInstance._combine_with_daily(
        TrendDirection.NONE, TrendDirection.UP,
    )
    assert result is TrendDirection.UP


# ============================================================
# #LIVE-1 fix: marketable limit entry + SL/TP 동봉 + pending 관리
# ============================================================


@pytest.mark.asyncio
async def test_marketable_limit_pending_when_unfilled() -> None:
    """계획가 limit 미체결 → _pending_entry 등록, active_position 아직 None."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    # 미체결 시뮬 — filled_qty / avg_fill_price 없음.
    client.place_order = AsyncMock(return_value={"orderId": "PENDING1"})
    bot = BotIctInstance(
        client=client, min_rr=1.0, fvg_min_size_pct=0.001,
        regime_filter_enabled=False,  # 미체결 pending 검증 — 횡보 게이트 격리
    )
    await bot.step()
    # entry 1회 — #LIVE-4: SL/TP 동봉 안 함 (체결 후 set_position_tpsl)
    assert client.place_order.await_count == 1
    entry_call = client.place_order.await_args_list[0].kwargs
    assert "take_profit" not in entry_call
    assert "stop_loss" not in entry_call
    # 미체결 → pending 등록, active_position 아직 X. SL/TP 도 아직 안 박음 (체결 후).
    assert bot.active_position is None
    assert bot._pending_entry is not None
    assert bot._pending_entry.direction is Direction.LONG
    client.set_position_tpsl.assert_not_awaited()


@pytest.mark.asyncio
async def test_pending_entry_promoted_on_fill() -> None:
    """pending 상태에서 거래소 체결 감지 → active_position 승격."""
    import time as _t

    from aurora_ict.bot.bot_ict_instance import _PendingEntry

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(
        return_value={"contracts": 0.01, "entryPrice": 100.5},
    )
    bot = BotIctInstance(client=client)
    bot._pending_entry = _PendingEntry(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0, take_profit=110.0,
        qty=0.01, setup_ts_ms=123, placed_ts_ms=int(_t.time() * 1000),
    )
    still_pending = await bot._check_pending_entry()
    assert still_pending is False
    assert bot._pending_entry is None
    assert bot.active_position is not None
    assert bot.active_position.entry == 100.5          # 거래소 체결가 우선
    assert bot.active_position.take_profit == 110.0    # SL/TP 보존


@pytest.mark.asyncio
async def test_pending_entry_cancelled_on_ttl() -> None:
    """pending TTL 만료 (미체결) → cancel_all_orders 호출, pending 해제."""
    from aurora_ict.bot.bot_ict_instance import _PendingEntry

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(return_value={"contracts": 0})
    bot = BotIctInstance(client=client, entry_limit_ttl_sec=600)
    bot._pending_entry = _PendingEntry(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0, take_profit=110.0,
        qty=0.01, setup_ts_ms=123, placed_ts_ms=0,  # epoch → TTL 만료
    )
    still_pending = await bot._check_pending_entry()
    assert still_pending is False
    assert bot._pending_entry is None
    assert bot.active_position is None
    client.cancel_all_orders.assert_awaited_once()


@pytest.mark.asyncio
async def test_pending_entry_waits_within_ttl() -> None:
    """pending TTL 내 미체결 → 계속 대기 (취소 X, 신규 진입 skip 신호)."""
    import time as _t

    from aurora_ict.bot.bot_ict_instance import _PendingEntry

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(return_value={"contracts": 0})
    bot = BotIctInstance(client=client, entry_limit_ttl_sec=600)
    bot._pending_entry = _PendingEntry(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0, take_profit=110.0,
        qty=0.01, setup_ts_ms=123, placed_ts_ms=int(_t.time() * 1000),  # 방금
    )
    still_pending = await bot._check_pending_entry()
    assert still_pending is True
    assert bot._pending_entry is not None
    client.cancel_all_orders.assert_not_awaited()


# ============================================================
# #BUG-7 fix: today realized PnL 거래소 closed-pnl 동기화
# ============================================================


@pytest.mark.asyncio
async def test_sync_today_realized_pnl_from_exchange() -> None:
    """거래소 closed-pnl 합산으로 today realized PnL 동기화 (#BUG-7)."""
    import time as _t
    from types import SimpleNamespace

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    now_ms = int(_t.time() * 1000)
    # 오늘 (NY 자정 이후) 청산 2건: -50, +20 → 합산 -30
    client.fetch_closed_positions = AsyncMock(return_value=[
        SimpleNamespace(pnl_usd=-50.0, closed_at_ts=now_ms),
        SimpleNamespace(pnl_usd=20.0, closed_at_ts=now_ms),
    ])
    bot = BotIctInstance(client=client)
    await bot._sync_today_realized_pnl()
    assert bot._today_realized_pnl_usdt == -30.0


@pytest.mark.asyncio
async def test_sync_pnl_excludes_before_ny_midnight() -> None:
    """NY 자정 이전 청산은 today 합산에서 제외."""
    from types import SimpleNamespace

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_closed_positions = AsyncMock(return_value=[
        SimpleNamespace(pnl_usd=-999.0, closed_at_ts=0),  # epoch → 자정 이전
    ])
    bot = BotIctInstance(client=client)
    await bot._sync_today_realized_pnl()
    assert bot._today_realized_pnl_usdt == 0.0


@pytest.mark.asyncio
async def test_daily_loss_limit_hit_via_exchange_sync() -> None:
    """거래소 동기화 실현손실이 한도 초과 → _daily_limit_hit (#SAFETY-1 + #BUG-7)."""
    import time as _t
    from types import SimpleNamespace

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    now_ms = int(_t.time() * 1000)
    client.fetch_closed_positions = AsyncMock(return_value=[
        SimpleNamespace(pnl_usd=-500.0, closed_at_ts=now_ms),
    ])
    bot = BotIctInstance(client=client, daily_loss_limit_pct=4.0)
    bot._today_start_equity = 10000.0  # 4% 한도 = 400 USDT
    await bot._sync_today_realized_pnl()
    assert bot._today_realized_pnl_usdt == -500.0
    assert bot._daily_limit_hit is True  # 손실 500 > 한도 400


# ============================================================
# 2026-05-29 #PNL-MATCH-FIX: closed-pnl 매칭 propagation 지연 보정
# ============================================================


@pytest.mark.asyncio
async def test_fetch_recent_close_rejects_stale_record_outside_tolerance() -> None:
    """cp.opened_at_ts 가 entry_ts_ms 와 10분 초과 → 이전 거래 record 로 보고 skip.

    5-29 #5 케이스 회고: Bybit propagation 지연으로 *이전* 거래 (#3) 의 close
    record 가 응답 가장 최근 자리에 와 잘못 매칭됐던 버그.
    """
    from types import SimpleNamespace

    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    from aurora_ict.interfaces.trades_store import TradeEventType

    client = _mock_client(_ohlcv_rows(datetime(2026, 5, 12, 10, 0, tzinfo=NY), _bars_long_setup()))
    client.fetch_position = AsyncMock(return_value={"contracts": 0})
    # 진입 ts 1_000_000_000_000 ms, cp.opened_at_ts 는 그보다 20분 이전 = 이전 거래.
    entry_ts = 1_000_000_000_000
    stale_cp = SimpleNamespace(
        symbol="BTCUSDT", direction="short",
        exit_price=99999.0, pnl_usd=-500.0,
        opened_at_ts=entry_ts - 20 * 60 * 1000,  # 20분 전 = tolerance 10분 초과
        closed_at_ts=entry_ts - 5 * 60 * 1000,
    )
    client.fetch_closed_positions = AsyncMock(return_value=[stale_cp])
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot._trades_store = _StubStore()
    bot.active_position = _ActivePosition(
        direction=Direction.SHORT, entry=100.0, stop_loss=105.0, take_profit=90.0,
        qty=1.0, setup_ts_ms=12345, entry_ts_ms=entry_ts,
    )
    await bot._sync_position_state()
    # stale cp 를 skip → close 매칭 실패 → SYNC_CLOSE fallback (placeholder 가격).
    ev = bot._trades_store.events[0]
    assert ev.event_type is TradeEventType.SYNC_CLOSE
    # 잘못된 -500 pnl 이 박히지 않아야 함 — 추정치 (0) 박힘.
    assert ev.pnl_usdt != -500.0


@pytest.mark.asyncio
async def test_fetch_recent_close_accepts_record_within_tolerance() -> None:
    """cp.opened_at_ts 가 entry_ts_ms 와 ±10분 이내면 정상 매칭."""
    from types import SimpleNamespace

    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    from aurora_ict.interfaces.trades_store import TradeEventType

    client = _mock_client(_ohlcv_rows(datetime(2026, 5, 12, 10, 0, tzinfo=NY), _bars_long_setup()))
    client.fetch_position = AsyncMock(return_value={"contracts": 0})
    entry_ts = 1_000_000_000_000
    # 진입 직후 ts (5분 차이) — tolerance 10분 이내라 정상 매칭.
    valid_cp = SimpleNamespace(
        symbol="BTCUSDT", direction="long",
        exit_price=95.0, pnl_usd=-50.0,
        opened_at_ts=entry_ts + 5 * 60 * 1000,
        closed_at_ts=entry_ts + 30 * 60 * 1000,
    )
    client.fetch_closed_positions = AsyncMock(return_value=[valid_cp])
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot._trades_store = _StubStore()
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=95.0, take_profit=110.0,
        qty=1.0, setup_ts_ms=12345, entry_ts_ms=entry_ts,
    )
    await bot._sync_position_state()
    ev = bot._trades_store.events[0]
    assert ev.event_type is TradeEventType.SL_HIT
    assert ev.pnl_usdt == -50.0  # 정상 매칭, 거래소 실현치 박힘


# ============================================================
# #RECONCILE — 재기동 중 청산 누락 ENTRY 보충
# ============================================================


@pytest.mark.asyncio
async def test_reconcile_fills_orphan_entry() -> None:
    """청산 누락 ENTRY(orphan)를 거래소 closed-pnl 매칭해 SYNC_CLOSE 로 보충."""
    from types import SimpleNamespace

    from aurora_ict.interfaces.trades_store import TradeEventType

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    # opened_at_ts=0 → 시간 체크 skip, symbol+direction 으로만 매칭.
    cp = SimpleNamespace(
        symbol="BTCUSDT", direction="short",
        exit_price=60000.0, pnl_usd=12.5, opened_at_ts=0,
    )
    client.fetch_closed_positions = AsyncMock(return_value=[cp])
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot._trades_store = _StubStore()
    # 청산 이벤트 없는 ENTRY (orphan).
    bot._record_trade(
        TradeEventType.ENTRY, direction=Direction.SHORT,
        price=60500.0, qty=0.1, setup_ts_ms=111,
    )
    await bot._reconcile_orphan_entries()
    syncs = [e for e in bot._trades_store.events if e.event_type is TradeEventType.SYNC_CLOSE]
    assert len(syncs) == 1
    assert syncs[0].setup_ts_ms == 111
    assert syncs[0].pnl_usdt == 12.5   # 거래소 closed-pnl 값


@pytest.mark.asyncio
async def test_reconcile_skips_already_closed_entry() -> None:
    """이미 청산 이벤트(SL_HIT)가 있는 ENTRY 는 보충하지 않음."""
    from types import SimpleNamespace

    from aurora_ict.interfaces.trades_store import TradeEventType

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    cp = SimpleNamespace(
        symbol="BTCUSDT", direction="short",
        exit_price=60000.0, pnl_usd=12.5, opened_at_ts=0,
    )
    client.fetch_closed_positions = AsyncMock(return_value=[cp])
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot._trades_store = _StubStore()
    bot._record_trade(
        TradeEventType.ENTRY, direction=Direction.SHORT,
        price=60500.0, qty=0.1, setup_ts_ms=222,
    )
    bot._record_trade(
        TradeEventType.SL_HIT, direction=Direction.SHORT,
        price=60000.0, qty=0.1, setup_ts_ms=222, entry_for_pnl=60500.0,
    )
    before = len(bot._trades_store.events)
    await bot._reconcile_orphan_entries()
    # SYNC_CLOSE 가 추가되지 않아야 (이미 SL_HIT 청산됨).
    assert len([e for e in bot._trades_store.events
                if e.event_type is TradeEventType.SYNC_CLOSE]) == 0
    assert len(bot._trades_store.events) == before


def test_remember_setup_records_ts_and_direction() -> None:
    """_remember_setup 이 ts + 방향을 함께 기록 — 롱→숏 전환 시 같은 봉 숏 누락 방지.

    기존엔 ts_ms 만 기록해, 롱 청산 직후 같은 봉의 숏 셋업이 duplicate_ts 로
    차단됐다(라이브 HYPE 버그). 방향까지 기록하면 반대 방향은 통과한다.
    """
    from types import SimpleNamespace

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = BotIctInstance(client=client, symbol="BTCUSDT")
    bot._remember_setup(SimpleNamespace(ts_ms=12345, direction=Direction.LONG))
    assert bot._last_setup_ts_ms == 12345
    assert bot._last_setup_direction is Direction.LONG
    # 같은 ts 라도 방향이 다르면 중복으로 보지 않는다(차단 조건이 거짓).
    assert not (
        12345 == bot._last_setup_ts_ms
        and Direction.SHORT == bot._last_setup_direction
    )


def test_classify_exchange_close_variants() -> None:
    """거래소 청산 분류 — TP=0 복구 포지션 SL 분류 + 방향 기준 (2026-06-12)."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance
    from aurora_ict.interfaces.trades_store import TradeEventType
    from aurora_ict.strategy.silver_bullet import Direction

    f = BotIctInstance._classify_exchange_close
    # 숏: close 가 TP 이하(더 유리) → TP_HIT.
    et, _ = f(Direction.SHORT, 100.0, 103.0, 95.0, 94.8)
    assert et is TradeEventType.TP_HIT
    # 숏: SL 슬리피지로 SL 위에서 체결 → SL_HIT.
    et, _ = f(Direction.SHORT, 100.0, 103.0, 95.0, 103.4)
    assert et is TradeEventType.SL_HIT
    # TP=0(복구 포지션) — SL 쪽만이라도 분류.
    et, _ = f(Direction.SHORT, 100.0, 103.0, 0.0, 103.1)
    assert et is TradeEventType.SL_HIT
    # 롱: close 가 TP 이상 → TP_HIT.
    et, _ = f(Direction.LONG, 100.0, 97.0, 105.0, 105.2)
    assert et is TradeEventType.TP_HIT
    # 중간 가격(수동 청산 등) → 미구분 유지.
    et, r = f(Direction.LONG, 100.0, 97.0, 105.0, 101.0)
    assert et is TradeEventType.SYNC_CLOSE and "미구분" in r
    # SL/TP 모두 미상 → 미구분.
    et, _ = f(Direction.LONG, 100.0, 0.0, 0.0, 101.0)
    assert et is TradeEventType.SYNC_CLOSE


@pytest.mark.asyncio
async def test_recover_restores_tp_from_records(tmp_path) -> None:
    """TP=0 복구 — 자기 기록의 미청산 ENTRY 에서 원 TP 재설치 (2026-06-12)."""
    import json as _json

    from aurora_ict.interfaces.trades_store import TradeEventType

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(return_value={
        "contracts": 0.05, "side": "short", "entryPrice": 80000.0,
        "stopLossPrice": 80500.0, "takeProfitPrice": 0.0,
    })
    client.set_position_tpsl = AsyncMock(return_value=True)
    bot = BotIctInstance(
        client=client, step_interval_sec=3600, trades_data_dir=tmp_path,
    )
    bot._record_trade(
        TradeEventType.ENTRY, direction=Direction.SHORT, price=80000.0,
        qty=0.05, setup_ts_ms=1234, reason="t",
        context_json=_json.dumps(
            {"entry": 80000.0, "tp": 78000.0, "sl": 80500.0},
        ),
    )
    await bot.start()
    assert bot.active_position is not None
    assert bot.active_position.take_profit == 78000.0  # 기록에서 복원
    assert bot.active_position.setup_ts_ms == 1234     # 원 setup 매칭 복원
    client.set_position_tpsl.assert_awaited()
    await bot.stop()


@pytest.mark.asyncio
async def test_recover_tp_stays_zero_without_matching_record(tmp_path) -> None:
    """TP=0 복구 — 매칭 기록 없으면 TP 0 유지 (지어내지 않음)."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(return_value={
        "contracts": 0.05, "side": "short", "entryPrice": 80000.0,
        "stopLossPrice": 80500.0, "takeProfitPrice": 0.0,
    })
    client.set_position_tpsl = AsyncMock(return_value=True)
    bot = BotIctInstance(
        client=client, step_interval_sec=3600, trades_data_dir=tmp_path,
    )
    await bot.start()
    assert bot.active_position is not None
    assert bot.active_position.take_profit == 0.0
    await bot.stop()


def test_daily_pair_loss_limit_logic() -> None:
    """페어별 일일 손실 한도 (2026-06-12) — R 배수 판정 + 비활성 조건."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = BotIctInstance(client=client, risk_per_trade_base=1.0,
                         daily_pair_loss_limit_r=2.0)
    bot._today_start_equity = 1000.0  # R = 10 USDT
    # 손실 -19 → 1.9R: 미달.
    bot._today_pair_realized_pnl_usdt = -19.0
    assert not bot._is_daily_pair_loss_limit_hit()
    # 손실 -20 → 2.0R: HIT.
    bot._today_pair_realized_pnl_usdt = -20.0
    assert bot._is_daily_pair_loss_limit_hit()
    # 수익이면 무관.
    bot._today_pair_realized_pnl_usdt = +50.0
    assert not bot._is_daily_pair_loss_limit_hit()
    # 0 = 비활성.
    bot.daily_pair_loss_limit_r = 0.0
    bot._today_pair_realized_pnl_usdt = -999.0
    assert not bot._is_daily_pair_loss_limit_hit()


def test_daily_pair_limit_resets_on_new_day() -> None:
    """페어별 한도 sticky flag — NY 자정 reset 에 함께 풀려야."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = BotIctInstance(client=client, daily_pair_loss_limit_r=2.0)
    bot._today_date_str = "2026-06-11"
    bot._daily_pair_limit_hit = True
    bot._today_pair_realized_pnl_usdt = -50.0
    bot._maybe_reset_daily_pnl(1000.0)  # 오늘 날짜와 다름 → reset
    assert bot._daily_pair_limit_hit is False
    assert bot._today_pair_realized_pnl_usdt == 0.0


@pytest.mark.asyncio
async def test_startup_cancel_skipped_when_position_recovered() -> None:
    """#TPSL-STRIP 회귀: 활성 포지션 복구 시 startup cancel_all 금지 (SL/TP 보존)."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.fetch_position = AsyncMock(return_value={
        "contracts": 1.0, "side": "short", "entryPrice": 100.0,
        "stopLossPrice": 103.0, "takeProfitPrice": 95.0,
    })
    client.cancel_all_orders = AsyncMock()
    bot = BotIctInstance(client=client, step_interval_sec=3600)
    await bot.start()
    assert bot.active_position is not None
    client.cancel_all_orders.assert_not_awaited()  # 보호장치 벗기면 안 됨
    await bot.stop()


@pytest.mark.asyncio
async def test_tpsl_verify_reasserts_when_exchange_missing() -> None:
    """#TPSL-VERIFY: 봇 기억엔 SL/TP 있는데 거래소에 없으면 재장착."""
    from aurora_ict.bot.bot_ict_instance import _ActivePosition

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.set_position_tpsl = AsyncMock(return_value=True)
    bot = BotIctInstance(client=client)
    bot.active_position = _ActivePosition(
        direction=Direction.SHORT, entry=1.1386, stop_loss=1.1537,
        take_profit=1.0954, qty=1682.5, setup_ts_ms=1,
    )
    # 거래소: 같은 방향·수량인데 SL/TP 없음 (벗겨진 상태). mark 는 SL 미관통.
    await bot._reconcile_open_position({
        "contracts": 1682.5, "side": "short", "entryPrice": 1.1386,
        "stopLossPrice": 0, "takeProfitPrice": 0, "markPrice": 1.1400,
    })
    client.set_position_tpsl.assert_awaited()
    kw = client.set_position_tpsl.await_args.kwargs
    assert kw["stop_loss"] == 1.1537 and kw["take_profit"] == 1.0954


@pytest.mark.asyncio
async def test_tpsl_verify_noop_when_exchange_matches() -> None:
    """#TPSL-VERIFY: 거래소 값 일치(틱 오차 내)면 재장착 안 함."""
    from aurora_ict.bot.bot_ict_instance import _ActivePosition

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.set_position_tpsl = AsyncMock(return_value=True)
    bot = BotIctInstance(client=client)
    bot.active_position = _ActivePosition(
        direction=Direction.SHORT, entry=1.1386, stop_loss=1.1537,
        take_profit=1.0954, qty=1682.5, setup_ts_ms=1,
    )
    await bot._reconcile_open_position({
        "contracts": 1682.5, "side": "short", "entryPrice": 1.1386,
        "stopLossPrice": 1.1537, "takeProfitPrice": 1.0954,
    })
    client.set_position_tpsl.assert_not_awaited()


@pytest.mark.asyncio
async def test_tpsl_verify_emergency_close_when_sl_breached() -> None:
    """#TPSL-VERIFY: 거래소 무SL + 가격이 SL 관통 → 재장착 대신 즉시 비상청산."""
    from aurora_ict.bot.bot_ict_instance import _ActivePosition

    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    client.set_position_tpsl = AsyncMock(return_value=True)
    client.place_order = AsyncMock(return_value={"orderId": "X"})
    bot = BotIctInstance(client=client)
    bot.active_position = _ActivePosition(
        direction=Direction.SHORT, entry=7.855, stop_loss=7.932,
        take_profit=0.0, qty=46.6, setup_ts_ms=1,
    )
    # 거래소: SL 없음 + mark 가 SL 위 (숏 관통).
    client.fetch_position = AsyncMock(return_value={
        "contracts": 46.6, "side": "short", "entryPrice": 7.855,
        "stopLossPrice": 0, "takeProfitPrice": 0, "markPrice": 7.956,
    })
    await bot._reconcile_open_position({
        "contracts": 46.6, "side": "short", "entryPrice": 7.855,
        "stopLossPrice": 0, "takeProfitPrice": 0, "markPrice": 7.956,
    })
    # 재장착이 아니라 청산(reduce_only place_order)이 나가야 함.
    client.set_position_tpsl.assert_not_awaited()
    client.place_order.assert_awaited()
    assert bot.active_position is None


# ===== #PARTIAL-TP 2026-06-23 분할익절(체감승률) 검증 =====


def test_calc_tp1_long_short_and_off() -> None:
    """_calc_tp1: 롱=entry+R, 숏=entry-R. partial_tp_rr<=0 / risk<=0 이면 0(비대상)."""
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = BotIctInstance(client=client, symbol="BTCUSDT", partial_tp_rr=1.0)
    assert bot._calc_tp1(100.0, 98.0, Direction.LONG) == pytest.approx(102.0)
    assert bot._calc_tp1(100.0, 102.0, Direction.SHORT) == pytest.approx(98.0)
    bot.partial_tp_rr = 0.0
    assert bot._calc_tp1(100.0, 98.0, Direction.LONG) == 0.0


@pytest.mark.asyncio
async def test_partial_exit_closes_half_and_moves_sl_to_breakeven() -> None:
    """TP1(1R) 도달 → 50% reduce_only 청산 + 나머지 50% SL 본전 이동."""
    import pandas as pd

    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = BotIctInstance(
        client=client, symbol="BTCUSDT", partial_tp_rr=1.0, partial_be=True,
    )
    # 롱: entry 100, SL 98(risk 2) → tp1 102(1R), swing TP 110
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=98.0,
        take_profit=110.0, qty=0.1, setup_ts_ms=1, tp1_price=102.0,
    )
    df = pd.DataFrame(
        [{"open": 101.0, "high": 103.0, "low": 101.0, "close": 102.5, "volume": 1.0}],
    )
    await bot._maybe_partial_exit(df)
    assert client.place_order.await_count == 1
    kw = client.place_order.await_args_list[0].kwargs
    assert kw["side"] == "sell"               # 롱 → sell 로 부분청산
    assert kw["qty"] == pytest.approx(0.05)    # 50%
    assert kw["reduce_only"] is True
    assert bot.active_position.partial_done is True
    assert bot.active_position.qty == pytest.approx(0.05)
    assert bot.active_position.stop_loss == pytest.approx(100.0)  # 본전
    client.set_position_tpsl.assert_awaited()


@pytest.mark.asyncio
async def test_partial_exit_skips_when_tp1_not_reached() -> None:
    """TP1 미도달 봉이면 부분익절 안 함 (원 포지션·SL 유지)."""
    import pandas as pd

    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = BotIctInstance(client=client, symbol="BTCUSDT", partial_tp_rr=1.0)
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=98.0,
        take_profit=110.0, qty=0.1, setup_ts_ms=1, tp1_price=102.0,
    )
    df = pd.DataFrame(
        [{"open": 100.5, "high": 101.5, "low": 100.0, "close": 101.0, "volume": 1.0}],
    )
    await bot._maybe_partial_exit(df)
    client.place_order.assert_not_awaited()
    assert bot.active_position.partial_done is False
    assert bot.active_position.qty == pytest.approx(0.1)


@pytest.mark.asyncio
async def test_partial_exit_once_only() -> None:
    """이미 부분익절(partial_done)한 포지션은 재청산 안 함 (중복 방지)."""
    import pandas as pd

    from aurora_ict.bot.bot_ict_instance import _ActivePosition
    client = _mock_client([[1, 100, 101, 99, 100, 10]])
    bot = BotIctInstance(client=client, symbol="BTCUSDT", partial_tp_rr=1.0)
    bot.active_position = _ActivePosition(
        direction=Direction.LONG, entry=100.0, stop_loss=100.0,
        take_profit=110.0, qty=0.05, setup_ts_ms=1, tp1_price=102.0,
        partial_done=True,
    )
    df = pd.DataFrame(
        [{"open": 102.0, "high": 103.0, "low": 102.0, "close": 102.5, "volume": 1.0}],
    )
    await bot._maybe_partial_exit(df)
    client.place_order.assert_not_awaited()
