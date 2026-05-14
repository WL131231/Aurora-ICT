"""AuroraClientAdapter — Aurora 측 CcxtClient를 ExchangeClientProtocol에 맞추는 어댑터.

Aurora 측 ``CcxtClient``가 노출하는 메서드:
- ``fetch_ohlcv(symbol, timeframe, limit) -> DataFrame``
- ``fetch_position(symbol) -> Position | None`` (dataclass)
- ``place_order(symbol, side, qty, price=None, ...) -> Order`` (dataclass)

Aurora-ICT가 기대하는 ``ExchangeClientProtocol``:
- ``fetch_ohlcv(symbol, timeframe, limit) -> list[list[Any]]`` (raw ccxt rows)
- ``fetch_position(symbol) -> dict | None``
- ``place_order(symbol, side, qty, ...) -> dict``
- ``fetch_balance() -> dict`` (ccxt 표준 포맷)

두 인터페이스 사이의 형식 차이를 흡수하는 thin adapter. Aurora-ICT가 Aurora 본체에
직접 의존하지 않도록 분리한다.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class AuroraClientAdapter:
    """Aurora ``CcxtClient``를 Aurora-ICT 인터페이스로 변환.

    Args:
        ccxt_client: Aurora ``CcxtClient`` instance (또는 duck-typed 호환 객체).
    """

    # Aurora 클라이언트는 1h+ timeframe 을 대문자로만 인식 (1H/2H/4H/1D/1W).
    # 우리 UI / settings 는 소문자 사용 (ccxt 표준). 호출 시 변환.
    _TF_AURORA_MAP = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D", "1w": "1W",
    }

    def __init__(self, ccxt_client: Any) -> None:
        self._client = ccxt_client

    def _aurora_tf(self, tf: str) -> str:
        """소문자 timeframe → Aurora 대문자 포맷. 미매핑 시 원본 그대로."""
        return self._TF_AURORA_MAP.get(tf, tf)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int,
    ) -> list[list[Any]]:
        """Aurora의 DataFrame 반환을 ccxt raw rows로 변환.

        Aurora ``fetch_ohlcv``는 DataFrame을 반환하므로
        [ts_ms, o, h, l, c, v] 리스트 형태로 변환해서 돌려준다.
        """
        df = await self._client.fetch_ohlcv(symbol, self._aurora_tf(timeframe), limit)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        rows: list[list[Any]] = []
        # df.index가 DatetimeIndex면 ms로 변환
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
        """Aurora place_order 결과를 dict로 변환.

        Aurora ``place_order`` 시그니처:
        ``(symbol, side, qty, price=None, reduce_only=False, ...)`` → Order dataclass.

        SL/TP는 Bybit에서 conditional order로 처리되어야 하지만 현재 Aurora 측 client는
        ccxt ``params={"stopLoss": SL, "takeProfit": TP}`` 형태를 직접 지원하지 않는다
        (TODO: Aurora 측 SL/TP passthrough 구현).
        """
        # Aurora place_order에 SL/TP 인자가 없어 현재는 경고만 남기고 entry만 등록한다.
        # 후속으로 Aurora 측에서 ccxt params를 받아 처리해주면 여기서도 전달 가능.
        if stop_loss is not None or take_profit is not None:
            logger.warning(
                "SL/TP 전달 불가 (Aurora client 미지원): "
                "sl=%s tp=%s — entry만 등록됩니다.",
                stop_loss, take_profit,
            )
        order = await self._client.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            reduce_only=reduce_only,
        )
        # Aurora Order dataclass → dict 변환
        if hasattr(order, "__dict__"):
            return dict(order.__dict__)
        if isinstance(order, dict):
            return order
        return {"raw": str(order)}

    async def fetch_position(self, symbol: str) -> dict[str, Any] | None:
        """Aurora Position dataclass → dict 변환."""
        pos = await self._client.fetch_position(symbol)
        if pos is None:
            return None
        if hasattr(pos, "__dict__"):
            d = dict(pos.__dict__)
            # Aurora Position의 ``qty`` 필드를 ccxt-style ``contracts``로 alias 추가
            if "qty" in d and "contracts" not in d:
                d["contracts"] = d["qty"]
            return d
        if isinstance(pos, dict):
            return pos
        return None

    async def fetch_balance(self) -> dict[str, Any]:
        """ccxt fetch_balance를 그대로 호출.

        Aurora 측 client는 별도 fetch_balance를 노출하지 않으므로 내부 ``_ex``
        (ccxt async exchange)에 직접 위임한다.
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            logger.warning("Aurora client에 _ex 속성 없음 — fallback {}")
            return {}
        try:
            return await ex.fetch_balance()
        except Exception as e:  # noqa: BLE001
            logger.warning("fetch_balance 실패: %s", e)
            return {}

    async def modify_stop_loss(
        self, symbol: str, new_stop_loss: float,
    ) -> dict[str, Any]:
        """Bybit V5 set_trading_stop API 호출 — 활성 포지션 SL 수정.

        Args:
            symbol: ccxt unified symbol (e.g. "BTC/USDT:USDT").
            new_stop_loss: 새 SL 가격.

        Returns:
            Bybit API 응답 dict. 실패 시 빈 dict.
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            logger.warning("modify_stop_loss: _ex 없음 — skip")
            return {}
        # ccxt unified → Bybit raw symbol ("BTC/USDT:USDT" → "BTCUSDT").
        raw_symbol = symbol.replace("/", "").split(":")[0]
        params = {
            "category": "linear",
            "symbol": raw_symbol,
            "stopLoss": str(new_stop_loss),
            # tpsl 모드 — Full 이면 전체 포지션 SL 수정.
            "tpslMode": "Full",
            "positionIdx": 0,  # one-way mode.
        }
        try:
            result = await ex.private_post_v5_position_trading_stop(params)
        except Exception as e:  # noqa: BLE001
            logger.warning("set_trading_stop 실패 (%s, sl=%.4f): %s",
                           raw_symbol, new_stop_loss, e)
            return {}
        return dict(result) if isinstance(result, dict) else {"raw": str(result)}


__all__ = ["AuroraClientAdapter"]
