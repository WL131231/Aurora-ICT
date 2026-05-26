"""Trades store — 매매 이벤트 영구 저장 (#BUG-2 해소).

설계:
    - **JSONL = source of truth** (`trades.jsonl`, append-only)
    - **SQLite = derived view** (`trades.db`, 분석 쿼리용)
    - 매 ``record(event)`` 호출 시 양쪽 동시 기록
    - SQLite 정합 깨지면 JSONL 에서 rebuild

저장 이벤트 종류 (``TradeEventType``):
    - ENTRY: 진입 체결
    - SL_HIT: 거래소 측 SL 트리거 청산
    - TP_HIT: TP reduce_only limit 체결
    - FLIP_CLOSE: HTF FVG flip 시 기존 포지션 청산
    - FLIP_OPEN: flip 직후 반대 방향 진입
    - RECOVERED: 봇 재시작 시 거래소 측 active position 복원
    - SYNC_CLOSE: ``_sync_position_state`` 가 거래소 측 closed 감지
    - MANUAL_CLOSE: API ``/ict/position/close`` 사용자 수동 청산

JSONL 1줄 = 1 TradeEvent (JSON dict). 봇 재시작 후에도 모든 이벤트 보존.
SQLite 는 분석/대시보드용 — full reconstruct 가능.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from threading import Lock

logger = logging.getLogger(__name__)


class TradeEventType(StrEnum):
    """매매 이벤트 종류."""

    ENTRY = "entry"
    SL_HIT = "sl_hit"
    TP_HIT = "tp_hit"
    FLIP_CLOSE = "flip_close"
    FLIP_OPEN = "flip_open"
    RECOVERED = "recovered"
    SYNC_CLOSE = "sync_close"
    MANUAL_CLOSE = "manual_close"


@dataclass(slots=True)
class TradeEvent:
    """매매 이벤트 1건 — JSONL 1줄 + SQLite 1행에 대응.

    Attributes:
        ts_ms: 이벤트 발생 시각 (UTC ms).
        event_type: ENTRY / SL_HIT / TP_HIT / FLIP_CLOSE / FLIP_OPEN / RECOVERED /
            SYNC_CLOSE / MANUAL_CLOSE.
        symbol: 거래 symbol (예: "BTC/USDT:USDT").
        direction: "long" / "short".
        price: 체결 또는 발생 가격.
        qty: 수량 (BTC 등 base 단위).
        pnl_usdt: 청산 이벤트 시 실현 손익 (USDT). ENTRY/RECOVERED 는 None.
        setup_ts_ms: ENTRY 와 청산 이벤트를 매칭하는 setup ts (없으면 None — RECOVERED 등).
        reason: 디버그/분석용 사유 문자열 (예: "trail SL", "flip target hit").
    """

    ts_ms: int
    event_type: TradeEventType
    symbol: str
    direction: str
    price: float
    qty: float
    pnl_usdt: float | None = None
    setup_ts_ms: int | None = None
    reason: str = ""
    # 진입 사유·confluence 세부·소스·윈도우 등 풍부한 컨텍스트(JSON 문자열).
    # ENTRY 이벤트에 채워 분석/저널에 활용. 청산 이벤트는 보통 None.
    context_json: str | None = None

    def to_json_line(self) -> str:
        """JSONL 1줄 직렬화 — trailing newline 포함."""
        d = asdict(self)
        d["event_type"] = self.event_type.value
        return json.dumps(d, ensure_ascii=False) + "\n"

    @classmethod
    def from_json_line(cls, line: str) -> TradeEvent:
        """JSONL 1줄 역직렬화."""
        d = json.loads(line)
        d["event_type"] = TradeEventType(d["event_type"])
        return cls(**d)


_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_ms INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    symbol TEXT NOT NULL,
    direction TEXT NOT NULL,
    price REAL NOT NULL,
    qty REAL NOT NULL,
    pnl_usdt REAL,
    setup_ts_ms INTEGER,
    reason TEXT NOT NULL DEFAULT '',
    context_json TEXT
);
CREATE INDEX IF NOT EXISTS idx_trades_ts ON trades(ts_ms);
CREATE INDEX IF NOT EXISTS idx_trades_setup ON trades(setup_ts_ms);
"""


def _format_journal_line(event: TradeEvent) -> str:
    """trade_journal.log 1줄 — 사람이 읽기 좋은 형식 (헤더 + 컨텍스트 들여쓰기 2줄)."""
    from datetime import datetime, timezone

    ts = datetime.fromtimestamp(event.ts_ms / 1000.0, tz=timezone.utc).astimezone()
    iso = ts.isoformat(timespec="milliseconds")
    head = (
        f"{iso} [{event.event_type.value.upper()}] {event.direction} "
        f"{event.symbol} @{event.price:.4f}"
    )
    if event.pnl_usdt is not None:
        head += f" pnl={event.pnl_usdt:+.2f}"
    if event.setup_ts_ms:
        head += f" setup_ts={event.setup_ts_ms}"
    if event.reason:
        head += f" reason={event.reason}"
    out = head + "\n"
    if event.context_json:
        try:
            ctx = json.loads(event.context_json)
        except json.JSONDecodeError:
            return out
        parts = []
        for key, fmt in (
            ("sl", "SL={:.4f}"), ("tp", "TP={:.4f}"), ("qty", "qty={:.4f}"),
            ("rr", "RR={:.2f}"), ("score", "score={}"), ("window", "window={}"),
            ("source", "source={}"), ("leverage", "lev={}x"),
        ):
            v = ctx.get(key)
            if v is not None:
                parts.append(fmt.format(v))
        if parts:
            out += "  " + " ".join(parts) + "\n"
        confs = ctx.get("confluences")
        if confs:
            out += "  confluences: " + ", ".join(confs) + "\n"
    return out


class TradesStore:
    """매매 이벤트 영구 저장 — JSONL 우선 / SQLite 거울.

    스레드 안전성:
        ``record()`` 는 내부 Lock 으로 직렬화. 봇 step / WS flip watcher 가 동시 호출
        해도 안전.

    파일 위치:
        ``data_dir / "trades.jsonl"``, ``data_dir / "trades.db"``.
        ``data_dir`` 는 호출자가 결정 (보통 ``aurora_ict.paths.data_dir()``).
    """

    def __init__(self, data_dir: Path) -> None:
        """디렉토리 보장 + SQLite 스키마 적용.

        Args:
            data_dir: trades.jsonl / trades.db 가 박힐 디렉토리.
                부재 시 ``mkdir(parents=True, exist_ok=True)`` 로 생성.
        """
        data_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = data_dir / "trades.jsonl"
        self.db_path = data_dir / "trades.db"
        # 사람이 읽기 좋은 거래 저널 (진입 사유·confluence 세부 등). JSONL/SQLite 와
        # 동일 데이터를 텍스트로 정리해 나중에 빠르게 훑어볼 수 있게 함.
        self.journal_path = data_dir / "trade_journal.log"
        self._lock = Lock()
        # SQLite 스키마 초기화 — 같은 connection 재사용 (봇 lifecycle 동안).
        # check_same_thread=False 로 record() 가 외부 스레드 (WS watcher) 에서 호출되도 OK.
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False,
        )
        self._conn.executescript(_SQLITE_SCHEMA)
        # 마이그레이션 — 기존 DB 에 context_json 컬럼 없으면 추가 (중복 시 무시).
        try:
            self._conn.execute("ALTER TABLE trades ADD COLUMN context_json TEXT")
        except sqlite3.OperationalError:
            pass  # 이미 컬럼 있음
        self._conn.commit()

    def record(self, event: TradeEvent) -> None:
        """이벤트 1건 기록 — JSONL append + SQLite insert.

        JSONL append 가 source of truth — SQLite insert 실패해도 JSONL 은 박힌다.
        SQLite 실패 시 warning 로그 + rebuild 권장 (호출자가 ``rebuild_sqlite`` 호출).

        Args:
            event: 기록할 TradeEvent.
        """
        with self._lock:
            # 1) JSONL append (atomic line write) — source of truth.
            try:
                with self.jsonl_path.open("a", encoding="utf-8") as f:
                    f.write(event.to_json_line())
            except OSError as e:
                logger.exception("trades.jsonl append 실패: %s", e)
                return  # JSONL 실패하면 SQLite/journal 도 건너뜀 — 정합 회피
            # 2) 사람이 읽기 좋은 trade_journal.log 도 append (실패해도 비치명).
            try:
                with self.journal_path.open("a", encoding="utf-8") as jf:
                    jf.write(_format_journal_line(event))
            except OSError as e:
                logger.warning("trade_journal.log append 실패: %s", e)
            # 3) SQLite insert (JSONL 성공 후).
            try:
                self._conn.execute(
                    """
                    INSERT INTO trades
                    (ts_ms, event_type, symbol, direction, price, qty,
                     pnl_usdt, setup_ts_ms, reason, context_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.ts_ms,
                        event.event_type.value,
                        event.symbol,
                        event.direction,
                        event.price,
                        event.qty,
                        event.pnl_usdt,
                        event.setup_ts_ms,
                        event.reason,
                        event.context_json,
                    ),
                )
                self._conn.commit()
            except sqlite3.Error as e:
                logger.warning(
                    "trades.db insert 실패 (JSONL 은 박힘 — rebuild 권장): %s", e,
                )

    def all_events(self) -> list[TradeEvent]:
        """JSONL 전체 이벤트 시간순 반환 — source of truth.

        Returns:
            TradeEvent list. 파일 부재 시 빈 list.
        """
        if not self.jsonl_path.exists():
            return []
        events: list[TradeEvent] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(TradeEvent.from_json_line(line))
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    logger.warning("trades.jsonl 손상 줄 무시: %s — %s", line[:80], e)
        return events

    def rebuild_sqlite(self) -> int:
        """JSONL 기준으로 SQLite 전부 재구축 — 정합 깨졌을 때 호출.

        기존 ``trades`` table 비우고 JSONL 전체를 다시 insert.

        Returns:
            insert 된 이벤트 수.
        """
        events = self.all_events()
        with self._lock:
            self._conn.execute("DELETE FROM trades")
            self._conn.executemany(
                """
                INSERT INTO trades
                (ts_ms, event_type, symbol, direction, price, qty,
                 pnl_usdt, setup_ts_ms, reason, context_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e.ts_ms,
                        e.event_type.value,
                        e.symbol,
                        e.direction,
                        e.price,
                        e.qty,
                        e.pnl_usdt,
                        e.setup_ts_ms,
                        e.reason,
                        e.context_json,
                    )
                    for e in events
                ],
            )
            self._conn.commit()
        logger.info("trades.db rebuilt from JSONL — %d rows", len(events))
        return len(events)

    def close(self) -> None:
        """SQLite connection 닫기 — 봇 stop 시 호출."""
        with self._lock:
            self._conn.close()


__all__ = [
    "TradeEvent",
    "TradeEventType",
    "TradesStore",
]
