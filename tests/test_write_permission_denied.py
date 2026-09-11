"""#WRITE-DENIED 2026-09-11 — 주문 권한 없는 키(10005)를 성공으로 오판하던 경로.

9/10~11 실측(TDAF): 읽기 전용 키로 재등록 → SL 등록 10005 → 비상청산 10005 →
어댑터가 '포지션 있으니 성공' 으로 처리 → 가짜 청산 기록 → 재입양 → 1분마다 반복
(4,202건). 이 파일은 (1) 어댑터가 주문 전후 상태 변화로만 성공을 인정하는지,
(2) 봇이 쓰기 거부 연속이면 정지+안내하는지, (3) 정리 도구가 안전한지 본다.

mock 0 — 결정론적 AsyncMock 만.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from ccxt.base.errors import PermissionDenied

from aurora_ict.bot import auth_stop_notify
from aurora_ict.bot.aurora_adapter import AuroraClientAdapter
from aurora_ict.bot.bot_ict_instance import (
    _WRITE_FAIL_STOP_THRESHOLD as _ORIGO_T,
    BotIctInstance,
    BotState,
)
from aurora_ict.bot.bot_trend_instance import (
    _WRITE_FAIL_STOP_THRESHOLD as _CURSUS_T,
    BotTrendInstance,
)

_DENIED = Exception(
    'bybit {"retCode":10005,"retMsg":"Permission denied, please check your API key '
    'permissions."} bybit private api uses /user/v3/private/query-api to check if you '
    "have a unified account"
)


@pytest.fixture(autouse=True)
def _reset_notify() -> None:
    auth_stop_notify.reset_for_tests()
    yield
    auth_stop_notify.reset_for_tests()


def _inner(pos_qty_seq: list[float | None], open_orders_seq: list[int]) -> AsyncMock:
    """fetch_position / fetch_open_orders 가 호출 순서대로 다른 값을 돌려주는 모형."""
    inner = AsyncMock()
    inner.place_order = AsyncMock(side_effect=_DENIED)

    def _pos(_sym):
        q = pos_qty_seq.pop(0) if pos_qty_seq else 0.0
        return None if q is None else {"contracts": q, "side": "long"}
    inner.fetch_position = AsyncMock(side_effect=_pos)
    inner._ex = AsyncMock()
    inner._ex.fetch_positions = AsyncMock(return_value=[])

    def _open(_sym):
        n = open_orders_seq.pop(0) if open_orders_seq else 0
        return [{"clientOrderId": "AUR" + "0" * 32} for _ in range(n)]
    inner._ex.fetch_open_orders = AsyncMock(side_effect=_open)
    return inner


# ── 어댑터: 10005 는 '효과' 로만 성공 ───────────────────────────────────────

@pytest.mark.asyncio
async def test_reduce_only_10005_with_position_unchanged_is_failure() -> None:
    """청산 주문 10005 + 포지션 그대로 → 실패(예외). 예전엔 '포지션 있음=성공'."""
    inner = _inner(pos_qty_seq=[0.05, 0.05], open_orders_seq=[0, 0])
    adapter = AuroraClientAdapter(inner)
    with pytest.raises(PermissionDenied):
        await adapter.place_order("BTC/USDT:USDT", "sell", 0.05, reduce_only=True)
    assert adapter.write_fail_streak == 1
    assert "10005" in (adapter.last_write_error or "")


@pytest.mark.asyncio
async def test_reduce_only_10005_with_position_reduced_is_success() -> None:
    """청산 주문 10005 인데 수량이 줄었으면 진짜 false positive → 성공."""
    inner = _inner(pos_qty_seq=[0.05, 0.0], open_orders_seq=[0, 0])
    adapter = AuroraClientAdapter(inner)
    r = await adapter.place_order("BTC/USDT:USDT", "sell", 0.05, reduce_only=True)
    assert r["info"]["ccxt_false_positive"] is True
    assert adapter.write_fail_streak == 0


@pytest.mark.asyncio
async def test_entry_10005_with_existing_position_unchanged_is_failure() -> None:
    """진입 주문 10005 + 이미 있던 포지션 수량 그대로 + 대기주문 없음 → 실패.

    9/10 새벽 가짜 entry 72건이 이 케이스였다(같은 가격 62초 간격 반복).
    """
    inner = _inner(pos_qty_seq=[0.005, 0.005], open_orders_seq=[0, 0])
    adapter = AuroraClientAdapter(inner)
    with pytest.raises(PermissionDenied):
        await adapter.place_order("BTC/USDT:USDT", "buy", 0.005, price=78650.0)
    assert adapter.write_fail_streak == 1


@pytest.mark.asyncio
async def test_entry_10005_with_new_pending_order_is_success() -> None:
    """진입 지정가 10005 인데 봇 태그 대기주문이 새로 생겼으면 성공(6월 false positive)."""
    inner = _inner(pos_qty_seq=[0.0, 0.0], open_orders_seq=[0, 1])
    adapter = AuroraClientAdapter(inner)
    r = await adapter.place_order("BTC/USDT:USDT", "buy", 0.005, price=78650.0)
    assert r["status"] == "open_uta_false_positive"
    assert adapter.write_fail_streak == 0


@pytest.mark.asyncio
async def test_entry_10005_with_position_grown_is_success() -> None:
    inner = _inner(pos_qty_seq=[0.0, 0.005], open_orders_seq=[0, 0])
    adapter = AuroraClientAdapter(inner)
    r = await adapter.place_order("BTC/USDT:USDT", "buy", 0.005)
    assert r["info"]["retCode"] == 10005
    assert adapter.write_fail_streak == 0


@pytest.mark.asyncio
async def test_write_ok_resets_streak() -> None:
    inner = _inner(pos_qty_seq=[0.05, 0.05], open_orders_seq=[0, 0])
    adapter = AuroraClientAdapter(inner)
    with pytest.raises(PermissionDenied):
        await adapter.place_order("BTC/USDT:USDT", "sell", 0.05, reduce_only=True)
    inner.place_order = AsyncMock(return_value={"orderId": "X", "filled_qty": 0.05,
                                                "avg_fill_price": 100.0})
    inner.fetch_position = AsyncMock(return_value={"contracts": 0.0})
    await adapter.place_order("BTC/USDT:USDT", "sell", 0.05, reduce_only=True)
    assert adapter.write_fail_streak == 0


@pytest.mark.asyncio
async def test_set_position_tpsl_10005_counts() -> None:
    inner = AsyncMock()
    inner._ex = AsyncMock()
    inner._ex.private_post_v5_position_trading_stop = AsyncMock(side_effect=_DENIED)
    adapter = AuroraClientAdapter(inner)
    res = await adapter.set_position_tpsl("BTC/USDT:USDT", stop_loss=75601.2)
    assert not res
    assert adapter.write_fail_streak == 1


# ── 봇: 쓰기 거부 연속이면 정지 + '권한' 안내 ─────────────────────────────

def _client_write_denied(streak: int) -> AsyncMock:
    c = AsyncMock()
    c.fetch_ohlcv = AsyncMock(return_value=[[1, 100, 101, 99, 100, 10]])
    c.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1000.0}})
    c.fetch_position = AsyncMock(return_value=None)
    c.auth_fail_streak = 0
    c.last_auth_error = None
    c.write_fail_streak = streak
    c.last_write_error = str(_DENIED)
    return c


@pytest.mark.asyncio
async def test_origo_stops_on_write_denied_and_notifies_permission() -> None:
    class _Bot(BotIctInstance):
        async def step(self):  # type: ignore[override]
            return None

    notify = AsyncMock()
    bot = _Bot(client=_client_write_denied(_ORIGO_T), symbol="BTCUSDT",
               step_interval_sec=0, notify_cb=notify, user_code="AICT-PERM-0000-0000")
    bot.state = BotState.RUNNING
    await asyncio.wait_for(bot._run_loop(), timeout=2.0)
    assert bot.state is BotState.STOPPED
    assert bot.stop_reason == "api_permission"
    notify.assert_awaited_once()
    msg = notify.await_args.args[1]
    assert "거래 권한" in msg and "손절" in msg


@pytest.mark.asyncio
async def test_cursus_stops_on_write_denied() -> None:
    class _Bot(BotTrendInstance):
        async def step(self):  # type: ignore[override]
            return None

    notify = AsyncMock()
    bot = _Bot(client=_client_write_denied(_CURSUS_T), symbol="BTC/USDT:USDT",
               step_interval_sec=0, notify_cb=notify, user_code="AICT-PERM-0000-0001")
    bot.state = BotState.RUNNING
    await asyncio.wait_for(bot._run_loop(), timeout=2.0)
    assert bot.state is BotState.STOPPED
    assert bot.stop_reason == "api_permission"
    notify.assert_awaited_once()


@pytest.mark.asyncio
async def test_cursus_emergency_close_failure_records_nothing() -> None:
    """비상청산 place_order 가 예외면 SYNC_CLOSE 를 기록하지 않는다(가짜 청산 방지)."""
    from aurora_ict.bot.bot_trend_instance import _TrendPosition
    from aurora_ict.strategy.dual_st import Direction

    recorded: list = []

    # slots dataclass 라 인스턴스 메서드 교체가 안 된다 — self-spy 는 서브클래스로.
    class _Spy(BotTrendInstance):
        def _record_trade(self, *a, **k):  # type: ignore[override]
            recorded.append((a, k))

    c = _client_write_denied(0)
    c.place_order = AsyncMock(side_effect=PermissionDenied("10005"))
    bot = _Spy(client=c, symbol="BTC/USDT:USDT")
    bot.active_position = _TrendPosition(direction=Direction.LONG, entry=100.0,
                                         qty=0.05, stop=98.0, entry_ts_ms=1)
    await bot._emergency_close("SL 적용 실패 — 무방비 방지 비상청산", price=101.0)
    assert recorded == []
    assert bot.active_position is None


# ── 정리 도구 ─────────────────────────────────────────────────────────────

def _write_jsonl(user_dir: Path, rows: list[dict]) -> None:
    user_dir.mkdir(parents=True, exist_ok=True)
    with (user_dir / "trades.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _row(ts: int, et: str, reason: str, pnl: float | None = None) -> dict:
    return {"ts_ms": ts, "event_type": et, "symbol": "BTC/USDT:USDT",
            "direction": "long", "price": 100.0, "qty": 0.05, "pnl_usdt": pnl,
            "setup_ts_ms": None, "reason": reason, "mode": "live", "model": "Cursus 1.0"}


def test_purge_dry_run_counts_only(tmp_path: Path) -> None:
    from aurora_ict.api.trades_router import purge_user_trades

    d = tmp_path / "users" / "AICT-TEST-0000-0000"
    _write_jsonl(d, [
        _row(1_000, "entry", "DualST 지정가 체결"),
        _row(2_000, "sync_close", "SL 적용 실패 — 무방비 방지 비상청산", 1.2),
        _row(3_000, "sync_close", "SL 적용 실패 — 무방비 방지 비상청산", -0.4),
        _row(4_000, "sl_hit", "SL 체결(거래소 자동청산)", -2.0),
    ])
    r = purge_user_trades(tmp_path, "AICT-TEST-0000-0000",
                          since_ms=1_500, reason_contains="무방비 방지", dry_run=True)
    assert r["ok"] and r["matched"] == 2 and r["kept"] == 2 and r["dry_run"]
    assert (d / "trades.jsonl").read_text(encoding="utf-8").count("\n") == 4


def test_purge_rewrites_with_backup_and_rebuilds_db(tmp_path: Path) -> None:
    from aurora_ict.api.trades_router import purge_user_trades

    code = "AICT-TEST-0000-0001"
    d = tmp_path / "users" / code
    _write_jsonl(d, [
        _row(1_000, "entry", "DualST 지정가 체결"),
        _row(2_000, "sync_close", "SL 적용 실패 — 무방비 방지 비상청산", 1.2),
        _row(4_000, "sl_hit", "SL 체결(거래소 자동청산)", -2.0),
    ])
    r = purge_user_trades(tmp_path, code, reason_contains="무방비 방지", dry_run=False)
    assert r["ok"] and r["matched"] == 1 and r["inserted_count"] == 2
    assert Path(r["backup"]).exists()
    lines = [json.loads(x) for x in (d / "trades.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x["event_type"] for x in lines] == ["entry", "sl_hit"]
    assert (d / "trades.db").exists()


def test_purge_refuses_without_filter(tmp_path: Path) -> None:
    from aurora_ict.api.trades_router import purge_user_trades

    r = purge_user_trades(tmp_path, "AICT-TEST-0000-0002", dry_run=False)
    assert r["ok"] is False and r["reason"] == "no_filter"
