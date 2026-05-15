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
        # ccxt 의 시간 동기화 옵션 설정. Bybit V5 private API 가 timestamp 1초 이상
        # 차이 박으면 retCode 10002 거부 → 모든 호출 시 자동 보정.
        ex = getattr(ccxt_client, "_ex", None)
        if ex is not None:
            try:
                if hasattr(ex, "options") and isinstance(ex.options, dict):
                    ex.options["adjustForTimeDifference"] = True
                    ex.options["recvWindow"] = 60000
            except Exception:  # noqa: BLE001
                pass
        self._time_diff_loaded = False

    async def _ensure_time_sync(self) -> None:
        """Bybit 서버 시간과 PC 시간 차이 한 번 로드 (lazy)."""
        if self._time_diff_loaded:
            return
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            self._time_diff_loaded = True
            return
        try:
            if hasattr(ex, "load_time_difference"):
                await ex.load_time_difference()
        except Exception as e:  # noqa: BLE001
            logger.warning("load_time_difference 실패: %s", e)
        finally:
            self._time_diff_loaded = True

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
        """Aurora place_order 결과를 dict로 변환 + SL/TP passthrough.

        Aurora ``place_order`` 시그니처:
        ``(symbol, side, qty, price=None, reduce_only=False, ...)`` → Order dataclass.

        v0.4.46 — SL/TP passthrough 추가:
        - entry 주문 (reduce_only=False) + SL/TP 인자 있으면, 주문 직후 raw ccxt 의
          ``set_trading_stop`` (Bybit V5 private API) 호출해서 거래소 측 conditional
          SL/TP 설정. position 등록 후에 호출되어야 하므로 시장가 (price=None) 일 때
          가장 안정.
        - reduce_only=True (partial TP) 는 일반 limit 주문 — SL/TP 인자는 무시.
        """
        order = await self._client.place_order(
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            reduce_only=reduce_only,
        )
        order_dict: dict[str, Any]
        if hasattr(order, "__dict__"):
            order_dict = dict(order.__dict__)
        elif isinstance(order, dict):
            order_dict = order
        else:
            order_dict = {"raw": str(order)}

        # SL/TP passthrough — entry 주문이고 (reduce_only False) SL/TP 있을 때.
        if not reduce_only and (stop_loss is not None or take_profit is not None):
            await self._apply_trading_stop(
                symbol=symbol,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
        return order_dict

    async def _apply_trading_stop(
        self,
        symbol: str,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> None:
        """Bybit V5 set_trading_stop 호출 — 활성 포지션의 SL/TP 설정."""
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            logger.warning("set_trading_stop: _ex 없음 — SL/TP 적용 skip")
            return
        await self._ensure_time_sync()
        raw_symbol = symbol.replace("/", "").split(":")[0]
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": raw_symbol,
            "tpslMode": "Full",
            "positionIdx": 0,
        }
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
        try:
            await ex.private_post_v5_position_trading_stop(params)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "set_trading_stop 실패 (%s sl=%s tp=%s): %s",
                raw_symbol, stop_loss, take_profit, e,
            )

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

    async def set_leverage(
        self, symbol: str, leverage: int,
    ) -> dict[str, Any]:
        """Bybit V5 set_leverage 호출 — buy/sell leverage 동시 설정.

        Args:
            symbol: ccxt unified symbol (e.g. "BTC/USDT:USDT").
            leverage: 정수 leverage (1 ~ 50).

        Returns:
            Bybit API 응답 dict. 실패 시 빈 dict.
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            logger.warning("set_leverage: _ex 없음 — skip")
            return {}
        await self._ensure_time_sync()
        raw_symbol = symbol.replace("/", "").split(":")[0]
        params = {
            "category": "linear",
            "symbol": raw_symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        try:
            result = await ex.private_post_v5_position_set_leverage(params)
            logger.info(
                "set_leverage 완료 (%s → %dx)", raw_symbol, leverage,
            )
            return dict(result) if isinstance(result, dict) else {"raw": str(result)}
        except Exception as e:  # noqa: BLE001
            # Bybit retCode 110043: "leverage not modified" — 이미 같은 값 설정.
            # 다른 에러는 warning, 진행 자체는 안 막음.
            msg = str(e)
            if "110043" in msg or "not modified" in msg:
                logger.info(
                    "set_leverage skip (%s 이미 %dx)", raw_symbol, leverage,
                )
                return {"retCode": 110043, "alreadySet": True}
            logger.warning(
                "set_leverage 실패 (%s %dx): %s",
                raw_symbol, leverage, e,
            )
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
        await self._ensure_time_sync()
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
