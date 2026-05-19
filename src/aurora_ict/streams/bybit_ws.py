"""Bybit V5 public ticker WebSocket — 실시간 가격 스트림 (변경 7 핵심).

목적:
    HTF FVG flip 발동 판정을 5분 봉 close 대신 tick 단위로 수행. ICT 정통은
    "FVG zone 1회 touch = mitigation 인정" — wick 만 들어와도 즉시 flip.

구조:
    - ``subscribe_ticker(symbol, on_tick, ...)`` 는 async generator/coroutine.
    - websockets 라이브러리로 연결, auto-reconnect (지수 backoff), heartbeat (ping).
    - 가격 변동마다 ``on_tick(price: float, ts_ms: int)`` 호출.
    - N 회 연속 연결 실패 시 ``on_fallback()`` (있으면) 호출 → 상위에서 REST polling
      fallback 으로 스위치.

정확성 원칙:
    - 0 이하 / None / NaN 가격 무시.
    - ts_ms 가 직전 tick 보다 과거면 무시 (네트워크 reorder).
    - 봇이 멈춰선 안 되므로 모든 예외는 흡수 + 로그 + reconnect.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from typing import Awaitable, Callable

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


# 한 tick callback. async / sync 둘 다 허용 (코드 측 await 분기).
TickCallback = Callable[[float, int], Awaitable[None] | None]
FallbackCallback = Callable[[], Awaitable[None] | None]


def _bybit_symbol(symbol: str) -> str:
    """ccxt unified ("BTC/USDT:USDT") → Bybit V5 ticker symbol ("BTCUSDT")."""
    # "BTC/USDT:USDT" or "BTCUSDT" 둘 다 안전하게 처리.
    s = symbol.split(":")[0]
    return s.replace("/", "").upper()


async def subscribe_ticker(
    symbol: str,
    on_tick: TickCallback,
    *,
    ws_url: str = "wss://stream.bybit.com/v5/public/linear",
    reconnect_max: int = 5,
    on_fallback: FallbackCallback | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Bybit V5 public linear ticker subscribe.

    Args:
        symbol: ccxt unified 또는 Bybit raw symbol.
        on_tick: 각 가격 update 마다 호출. ``(price, ts_ms)``.
        ws_url: WS endpoint (settings 에서 주입).
        reconnect_max: 연속 실패 허용 횟수. 도달 시 ``on_fallback`` 호출 후 종료.
        on_fallback: WS 포기 시 호출되는 콜백 (상위에서 polling 으로 스위치).
        stop_event: 외부에서 set 하면 루프 종료.
    """
    raw_sym = _bybit_symbol(symbol)
    topic = f"tickers.{raw_sym}"
    fail_count = 0
    last_ts_ms = 0

    while True:
        if stop_event is not None and stop_event.is_set():
            return
        try:
            async with websockets.connect(
                ws_url,
                ping_interval=20,
                ping_timeout=20,
                close_timeout=5,
            ) as ws:
                await ws.send(json.dumps({"op": "subscribe", "args": [topic]}))
                fail_count = 0  # 연결 성공 → 카운터 리셋
                logger.info("Bybit WS 연결 성공 — topic=%s", topic)

                async for raw in ws:
                    if stop_event is not None and stop_event.is_set():
                        return
                    try:
                        msg = json.loads(raw)
                    except (TypeError, ValueError):
                        continue
                    # subscribe ack / pong 류 — data 없으면 skip.
                    data = msg.get("data")
                    if not isinstance(data, dict):
                        continue
                    # Bybit V5: lastPrice 가 표준 last trade 가격. markPrice 도 fallback.
                    price_str = (
                        data.get("lastPrice")
                        or data.get("markPrice")
                        or data.get("indexPrice")
                    )
                    if price_str is None:
                        continue
                    try:
                        price = float(price_str)
                    except (TypeError, ValueError):
                        continue
                    # sanity — 0 / 음수 / NaN 거름.
                    if price <= 0 or math.isnan(price) or math.isinf(price):
                        continue
                    ts_ms = int(msg.get("ts") or 0)
                    # reorder 거름.
                    if ts_ms > 0 and ts_ms < last_ts_ms:
                        continue
                    if ts_ms > 0:
                        last_ts_ms = ts_ms

                    try:
                        ret = on_tick(price, ts_ms)
                        if asyncio.iscoroutine(ret):
                            await ret
                    except Exception as e:  # noqa: BLE001
                        # callback 예외가 WS 루프를 죽이지 않게 흡수.
                        logger.exception("on_tick 콜백 예외: %s", e)
        except asyncio.CancelledError:
            raise
        except (ConnectionClosed, WebSocketException, OSError, asyncio.TimeoutError) as e:
            fail_count += 1
            backoff = min(2 ** fail_count, 30)
            logger.warning(
                "Bybit WS 연결 끊김 (%d/%d): %s — %ds 후 재시도",
                fail_count, reconnect_max, e, backoff,
            )
            if fail_count >= reconnect_max:
                logger.error(
                    "Bybit WS 재연결 한도 초과 (%d) — fallback 으로 스위치",
                    reconnect_max,
                )
                if on_fallback is not None:
                    try:
                        ret = on_fallback()
                        if asyncio.iscoroutine(ret):
                            await ret
                    except Exception as fe:  # noqa: BLE001
                        logger.exception("on_fallback 콜백 예외: %s", fe)
                return
            await asyncio.sleep(backoff)
        except Exception as e:  # noqa: BLE001
            fail_count += 1
            logger.exception("Bybit WS 예기치 못한 예외: %s", e)
            if fail_count >= reconnect_max:
                if on_fallback is not None:
                    try:
                        ret = on_fallback()
                        if asyncio.iscoroutine(ret):
                            await ret
                    except Exception as fe:  # noqa: BLE001
                        logger.exception("on_fallback 콜백 예외: %s", fe)
                return
            await asyncio.sleep(min(2 ** fail_count, 30))


__all__ = ["subscribe_ticker", "TickCallback", "FallbackCallback"]
