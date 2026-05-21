"""TradesStore 단위 테스트 (#BUG-2 해소 검증).

- JSONL append + SQLite mirror 동기
- 봇 재시작 시뮬레이션 (TradesStore 재생성) — JSONL 보존
- SQLite 손상 시 rebuild
- 손상된 JSONL 줄 graceful 무시
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from aurora_ict.interfaces.trades_store import (
    TradeEvent,
    TradeEventType,
    TradesStore,
)


def _make_event(
    ts_ms: int = 1_700_000_000_000,
    event_type: TradeEventType = TradeEventType.ENTRY,
    price: float = 50000.0,
    qty: float = 0.01,
    pnl_usdt: float | None = None,
    setup_ts_ms: int | None = None,
    reason: str = "",
) -> TradeEvent:
    return TradeEvent(
        ts_ms=ts_ms,
        event_type=event_type,
        symbol="BTC/USDT:USDT",
        direction="long",
        price=price,
        qty=qty,
        pnl_usdt=pnl_usdt,
        setup_ts_ms=setup_ts_ms,
        reason=reason,
    )


def test_trade_event_json_roundtrip() -> None:
    """to_json_line → from_json_line 무손실."""
    ev = _make_event(pnl_usdt=12.5, setup_ts_ms=1234, reason="test")
    line = ev.to_json_line()
    assert line.endswith("\n")
    restored = TradeEvent.from_json_line(line.strip())
    assert restored == ev


def test_record_writes_jsonl_and_sqlite(tmp_path: Path) -> None:
    """record 1건 → JSONL 1줄 + SQLite 1행."""
    store = TradesStore(tmp_path)
    ev = _make_event()
    store.record(ev)

    # JSONL 1줄 확인
    lines = (tmp_path / "trades.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["event_type"] == "entry"
    assert d["price"] == 50000.0

    # SQLite 1행 확인
    conn = sqlite3.connect(tmp_path / "trades.db")
    rows = conn.execute("SELECT event_type, price, qty FROM trades").fetchall()
    conn.close()
    assert rows == [("entry", 50000.0, 0.01)]
    store.close()


def test_record_multiple_preserves_order(tmp_path: Path) -> None:
    """여러 record → JSONL 순서 + SQLite 행 순서 일치."""
    store = TradesStore(tmp_path)
    for i in range(5):
        store.record(_make_event(ts_ms=1_700_000_000_000 + i))
    events = store.all_events()
    assert [e.ts_ms for e in events] == [
        1_700_000_000_000 + i for i in range(5)
    ]
    store.close()


def test_all_events_empty_when_no_file(tmp_path: Path) -> None:
    """파일 부재 시 all_events → 빈 list."""
    store = TradesStore(tmp_path)
    assert store.all_events() == []
    store.close()


def test_corrupted_jsonl_line_skipped(tmp_path: Path) -> None:
    """JSONL 손상 줄 무시 + 나머지는 로드."""
    store = TradesStore(tmp_path)
    store.record(_make_event(ts_ms=1))
    store.close()

    # 손상 줄 + 정상 줄 추가
    with (tmp_path / "trades.jsonl").open("a", encoding="utf-8") as f:
        f.write("INVALID JSON\n")
        f.write(_make_event(ts_ms=3).to_json_line())

    store2 = TradesStore(tmp_path)
    events = store2.all_events()
    # 1, 3 만 로드 (손상 줄 무시)
    assert [e.ts_ms for e in events] == [1, 3]
    store2.close()


def test_rebuild_sqlite_from_jsonl(tmp_path: Path) -> None:
    """SQLite 정합 깨졌을 때 JSONL 기준 rebuild."""
    store = TradesStore(tmp_path)
    for i in range(3):
        store.record(_make_event(ts_ms=100 + i))
    # SQLite 직접 손상 (외부 DELETE)
    store._conn.execute("DELETE FROM trades")
    store._conn.commit()
    assert store._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 0

    # rebuild
    n = store.rebuild_sqlite()
    assert n == 3
    count = store._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    assert count == 3
    store.close()


def test_store_survives_recreate(tmp_path: Path) -> None:
    """봇 재시작 시뮬레이션 — TradesStore 재생성 후 이전 JSONL 모두 보존."""
    store1 = TradesStore(tmp_path)
    store1.record(_make_event(ts_ms=10, reason="first session"))
    store1.record(_make_event(ts_ms=20, reason="first session"))
    store1.close()

    store2 = TradesStore(tmp_path)
    events = store2.all_events()
    assert len(events) == 2
    assert all(e.reason == "first session" for e in events)

    # 새 세션 record 도 같은 JSONL 에 append
    store2.record(_make_event(ts_ms=30, reason="second session"))
    events = store2.all_events()
    assert [e.ts_ms for e in events] == [10, 20, 30]
    store2.close()


def test_pnl_field_stored(tmp_path: Path) -> None:
    """pnl_usdt 가 SQLite 에도 저장됨."""
    store = TradesStore(tmp_path)
    store.record(_make_event(
        event_type=TradeEventType.SL_HIT,
        pnl_usdt=-15.3,
    ))
    conn = sqlite3.connect(tmp_path / "trades.db")
    pnl = conn.execute("SELECT pnl_usdt FROM trades").fetchone()[0]
    conn.close()
    assert pnl == pytest.approx(-15.3)
    store.close()


def test_event_type_enum_serialization(tmp_path: Path) -> None:
    """모든 TradeEventType 값이 round-trip OK."""
    store = TradesStore(tmp_path)
    for et in TradeEventType:
        store.record(_make_event(event_type=et))
    events = store.all_events()
    types_loaded = {e.event_type for e in events}
    assert types_loaded == set(TradeEventType)
    store.close()
