"""Aurora-ICT interfaces — 데이터 저장 / 외부 노출 layer.

현재 모듈:
    - ``trades_store`` — 매매 이벤트 JSONL append-only + SQLite derived view.
"""

from aurora_ict.interfaces.trades_store import (
    TradeEvent,
    TradeEventType,
    TradesStore,
)

__all__ = [
    "TradeEvent",
    "TradeEventType",
    "TradesStore",
]
