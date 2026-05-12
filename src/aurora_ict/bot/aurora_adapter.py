"""AuroraClientAdapter — Aurora 측 CcxtClient 박힌 거 박은 박힘 ExchangeClientProtocol.

Aurora 측 ``CcxtClient`` 박힌 거 박힘 method 박힘 박힘 박힘 박힘:
- ``fetch_ohlcv(symbol, timeframe, limit) -> DataFrame``
- ``fetch_position(symbol) -> Position | None`` (dataclass)
- ``place_order(symbol, side, qty, price=None, ...) -> Order`` (dataclass)

Aurora-ICT 측 ``ExchangeClientProtocol`` 박힌 거 박힘 박힘 박힘:
- ``fetch_ohlcv(symbol, timeframe, limit) -> list[list[Any]]`` (raw ccxt rows)
- ``fetch_position(symbol) -> dict | None``
- ``place_order(symbol, side, qty, ...) -> dict``
- ``fetch_balance() -> dict`` (ccxt 박힌 거 박힘 박힘 박힘)

박은 박힌 거 박힘 차이 박힘 박힘 박힘 adapter 박힘 박힘 박힘 박힘. 박힌 거 박힌 거 박힘
박힘 박힘 dependency 박힌 거 박힘 박힘 박힘 — Aurora-ICT 박은 거 박힘 박힘 박힘 Aurora 박은
거 박힌 거 박힘 박힘 박힘 박힌 거 박힘 박힘 박힘 박힘.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class AuroraClientAdapter:
    """Aurora ``CcxtClient`` 박힌 거 박힌 Aurora-ICT 박은 거 박힘 박힘 박힘 박힘.

    Args:
        ccxt_client: Aurora ``CcxtClient`` instance (또는 duck-typed 박힘 박힘 박힘).
    """

    def __init__(self, ccxt_client: Any) -> None:
        self._client = ccxt_client

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int,
    ) -> list[list[Any]]:
        """DataFrame 박힌 거 박힘 박힘 ccxt raw rows 박힘 박힘 박힘.

        Aurora ``fetch_ohlcv`` 박힌 거 박힘 박힘 DataFrame 박힘 박힘 박힘 박힘 박힘 raw
        [ts_ms, o, h, l, c, v] 박힘 박힘 박힘.
        """
        df = await self._client.fetch_ohlcv(symbol, timeframe, limit)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        rows: list[list[Any]] = []
        # df.index 박힌 거 박힘 DatetimeIndex 박힘 → ms
        if isinstance(df.index, pd.DatetimeIndex):
            ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
        else:
            ts_arr = df.index.to_numpy()
        opens = df["open"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        vols = df["volume"].to_numpy() if "volume" in df.columns else [0.0] * len(df)
        for i in range(len(df)):
            rows.append([
                int(ts_arr[i]),
                float(opens[i]),
                float(highs[i]),
                float(lows[i]),
                float(closes[i]),
                float(vols[i]),
            ])
        return rows

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float | None = None,
        reduce_only: bool = False,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """Aurora place_order 박힌 거 박힘 박힘 dict 박힘 박힘.

        Aurora 박은 거 ``place_order`` signature 박힌 거 박힘 박힘 박힘:
        ``(symbol, side, qty, price=None, reduce_only=False, ...)`` → Order dataclass.

        SL/TP 박은 거 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 Bybit 박힘 박힘 conditional
        order 박힘 박힘 박힘 박힘. 박힌 거 박힌 거 박힘 ccxt ``params={"stopLoss": SL,
        "takeProfit": TP}`` 박힘 박힘 박힘 박힘 박힘. Aurora 박은 거 박은 client 박은 박은
        박은 박은 박은 박은 박은 박은 박은 (TODO: Aurora 측 SL/TP 박힘 박힘 박힘 박힘).
        """
        # Aurora place_order 박은 거 박힘 박힘 SL/TP 박은 거 박힘 박힘 — 박힌 거 박힌 거 박힘
        # 박힘 박힘 박힘 박힘 ccxt params 박힌 거 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘
        # Aurora 측 박힘 박힘 박힘 박힘 박힘 박힘 (v0.1.7 박힌 거 박힌 거 박힘 박힘 박힘 박힘
        # 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘).
        if stop_loss is not None or take_profit is not None:
            logger.warning(
                "SL/TP 박은 거 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘 "
                "(Aurora client 박은 거 박은 거 박힘 박힘 박힘 박힘 박힘): "
                "sl=%s tp=%s — 박힌 거 박힌 거 박힘 박힘 박힘 박힘 박힘 박힘 박힘 박힘.",
                stop_loss, take_profit,
            )
        order = await self._client.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            reduce_only=reduce_only,
        )
        # Aurora Order dataclass → dict 박힘
        if hasattr(order, "__dict__"):
            return dict(order.__dict__)
        if isinstance(order, dict):
            return order
        return {"raw": str(order)}

    async def fetch_position(self, symbol: str) -> dict[str, Any] | None:
        """Aurora Position dataclass → dict 박힘 박힘."""
        pos = await self._client.fetch_position(symbol)
        if pos is None:
            return None
        if hasattr(pos, "__dict__"):
            d = dict(pos.__dict__)
            # Aurora Position 박은 거 박힌 ``qty`` 박힘 → ``contracts`` 박힘 (ccxt-style)
            if "qty" in d and "contracts" not in d:
                d["contracts"] = d["qty"]
            return d
        if isinstance(pos, dict):
            return pos
        return None

    async def fetch_balance(self) -> dict[str, Any]:
        """ccxt fetch_balance 박힌 거 박힘 박힘 박힘.

        Aurora 측 client 박은 거 박힙 박힘 박힘 박힘 박힘 박힘 — ``self._client._ex`` 박힘
        박힘 ccxt async exchange 박힘 박힘 박힘 박힘 박힘 박힘. 박힌 거 박힌 거 박힘 박힘
        박힘 박힘 박힘.
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            logger.warning("Aurora client 박힌 _ex 박은 거 박힌 거 박힘 — fallback {}")
            return {}
        try:
            return await ex.fetch_balance()
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_balance 박힘: %s", e)
            return {}


__all__ = ["AuroraClientAdapter"]
