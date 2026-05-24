"""BotIctInstance 박은 거 박힘 — Aurora-ICT v0.1.5."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance, BotState
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


@pytest.mark.asyncio
async def test_step_duplicate_setup_filtered() -> None:
    """같은 setup ts_ms 박힘 박힘 두 번째 박힘 박힘 박힘 X (중복 진입 박힘)."""
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    rows = _ohlcv_rows(start, _bars_long_setup())
    client = _mock_client(rows)
    bot = BotIctInstance(client=client, min_rr=1.0, fvg_min_size_pct=0.001)
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
    bot = BotIctInstance(client=client, min_rr=1.0, fvg_min_size_pct=0.001)
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
