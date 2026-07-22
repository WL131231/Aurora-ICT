"""2026-07-22: 심볼별 공유 OHLCV 캐시 테스트.

봇별 캐시 → (symbol,tf) 전역 공유로 전환(중복제거·메모리절감). 검증:
  - 같은 심볼 두 봇이 캐시 공유(한 봇이 채우면 다른 봇이 재사용, 재fetch 안 함).
  - 다른 심볼은 격리(BTC 캐시가 ETH 로 새지 않음).
  - prefetch 는 심볼당 1회만 fetch(중복 봇은 skip).
mock 0 — 결정론적 합성 client(호출 카운트 캡처).
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.bot.shared_ohlcv_cache import SharedOhlcvCache


def _rows(symbol: str, n: int = 300) -> list[list[Any]]:
    base = 100.0 if symbol.startswith("BTC") else 50.0
    return [[i * 60_000, base, base + 1, base - 1, base, 1.0] for i in range(n)]


def _bot(symbol: str, shared: SharedOhlcvCache, fetch_counter: dict) -> BotIctInstance:
    client = AsyncMock()

    async def _fetch(sym, tf, limit):
        fetch_counter[(sym, tf)] = fetch_counter.get((sym, tf), 0) + 1
        return _rows(sym)[-limit:]

    client.fetch_ohlcv = AsyncMock(side_effect=_fetch)
    return BotIctInstance(client=client, symbol=symbol, _shared_ohlcv=shared)


def test_unit_get_set_isolated_by_symbol() -> None:
    """SharedOhlcvCache: (symbol,tf) 키 격리 — BTC/ETH 15m 서로 안 섞임."""
    c = SharedOhlcvCache()
    c.set("BTCUSDT", "15m", [[1, 1, 1, 1, 1, 1]])
    assert c.has("BTCUSDT", "15m")
    assert not c.has("ETHUSDT", "15m")
    assert c.get("ETHUSDT", "15m") is None
    assert c.get("BTCUSDT", "15m") == [[1, 1, 1, 1, 1, 1]]


@pytest.mark.asyncio
async def test_same_symbol_two_bots_share_cache() -> None:
    """같은 심볼 두 봇 — 첫 봇이 채운 캐시를 둘째가 재사용(재fetch 0)."""
    shared = SharedOhlcvCache()
    counter: dict = {}
    bot_a = _bot("BTCUSDT", shared, counter)
    bot_b = _bot("BTCUSDT", shared, counter)

    rows_a = await bot_a.get_ohlcv_cached("15m", 300)
    assert len(rows_a) > 0
    fetches_after_a = counter.get(("BTCUSDT", "15m"), 0)

    # 둘째 봇 — 공유 캐시 hit → 추가 fetch 없어야 (cache hit 경로).
    rows_b = await bot_b.get_ohlcv_cached("15m", 300)
    assert rows_b == shared.get("BTCUSDT", "15m")[-300:]
    assert counter.get(("BTCUSDT", "15m"), 0) == fetches_after_a  # 재fetch 0


@pytest.mark.asyncio
async def test_different_symbols_do_not_share() -> None:
    """다른 심볼 두 봇 — 각자 자기 심볼 데이터(격리)."""
    shared = SharedOhlcvCache()
    counter: dict = {}
    bot_btc = _bot("BTCUSDT", shared, counter)
    bot_eth = _bot("ETHUSDT", shared, counter)

    await bot_btc.get_ohlcv_cached("15m", 300)
    await bot_eth.get_ohlcv_cached("15m", 300)

    assert shared.has("BTCUSDT", "15m")
    assert shared.has("ETHUSDT", "15m")
    # 각 심볼별 값이 다름(BTC base 100 vs ETH 50)
    assert shared.get("BTCUSDT", "15m")[0][1] != shared.get("ETHUSDT", "15m")[0][1]


@pytest.mark.asyncio
async def test_prefetch_dedup_same_symbol() -> None:
    """prefetch: 같은 심볼 둘째 봇은 이미 채워진 TF skip(심볼당 1회 fetch)."""
    shared = SharedOhlcvCache()
    counter: dict = {}
    bot_a = _bot("BTCUSDT", shared, counter)
    bot_b = _bot("BTCUSDT", shared, counter)

    await bot_a._prefetch_all_ohlcv_tfs()
    fetches_a = dict(counter)
    await bot_b._prefetch_all_ohlcv_tfs()

    # 둘째 봇 prefetch 후에도 fetch 카운트 불변(전부 skip).
    assert counter == fetches_a
    # 15m·1h 각 정확히 1회만 fetch됨.
    assert counter.get(("BTCUSDT", "15m")) == 1
    assert counter.get(("BTCUSDT", "1h")) == 1
