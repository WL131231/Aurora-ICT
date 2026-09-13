"""실제 SQLite 커밋 잠금 뒤 중복 INSERT 방지 검증. 담당: Codex."""

from __future__ import annotations

import sqlite3

from aurora_ict.interfaces.trades_store import TradeEvent, TradeEventType, TradesStore


def test_failed_commit_rolls_back_before_next_trade(tmp_path):
    """읽기 잠금으로 세 번 commit 실패 후에도 이전 손절이 중복 확정되지 않는다."""
    store = TradesStore(tmp_path)
    reader = sqlite3.connect(store.db_path)
    try:
        store._conn.execute("PRAGMA busy_timeout=1")
        reader.execute("BEGIN")
        reader.execute("SELECT * FROM trades").fetchall()
        store.record(TradeEvent(
            ts_ms=1000, event_type=TradeEventType.SL_HIT, symbol="BTC/USDT:USDT",
            direction="long", price=90, qty=1, pnl_usdt=-10,
        ))
        assert not store._conn.in_transaction
        reader.rollback()
        store.record(TradeEvent(
            ts_ms=2000, event_type=TradeEventType.ENTRY, symbol="BTC/USDT:USDT",
            direction="long", price=100, qty=1,
        ))
        rows = reader.execute("SELECT event_type, pnl_usdt FROM trades").fetchall()
        assert rows == [("entry", None)]
        assert len(store.all_events()) == 2
        store.rebuild_sqlite()
        assert reader.execute("SELECT COUNT(*), SUM(pnl_usdt) FROM trades").fetchone() == (2, -10)
    finally:
        reader.close()
        store.close()
