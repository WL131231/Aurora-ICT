"""실제 장부 정리의 백업 보존 회귀 검증. 담당: Codex."""

from contextlib import closing
from pathlib import Path

from aurora_ict.api.trades_router import purge_user_trades
from aurora_ict.interfaces.trades_store import TradeEvent, TradeEventType, TradesStore


def test_consecutive_purges_preserve_each_original(tmp_path):
    """연속 정리는 서로 다른 백업에 각 실행 직전의 원본을 보존한다."""
    code = "AICT-TEST-BACK-UP01"
    user_dir = tmp_path / "users" / code
    with closing(TradesStore(user_dir)) as store:
        for ts, reason in ((1000, "first"), (2000, "second"), (3000, "keep")):
            store.record(TradeEvent(
                ts_ms=ts, event_type=TradeEventType.ENTRY, symbol="BTC/USDT:USDT",
                direction="long", price=100, qty=1, reason=reason,
            ))
    jsonl = user_dir / "trades.jsonl"
    original = jsonl.read_bytes()
    first = purge_user_trades(tmp_path, code, reason_contains="first", dry_run=False)
    intermediate = jsonl.read_bytes()
    second = purge_user_trades(tmp_path, code, reason_contains="second", dry_run=False)
    assert first["matched"] == second["matched"] == 1
    assert first["backup"] != second["backup"]
    assert Path(first["backup"]).read_bytes() == original
    assert Path(second["backup"]).read_bytes() == intermediate
    with closing(TradesStore(user_dir)) as store:
        assert [event.reason for event in store.all_events()] == ["keep"]
