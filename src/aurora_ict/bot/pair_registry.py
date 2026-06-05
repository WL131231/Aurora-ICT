"""PairRegistry — 거래 가능 페어 화이트리스트 (Bybit USDT perp 거래대금 상위 N).

페어를 BTC/ETH 에서 다종목으로 확장할 때, 거래 가능 목록을 거래소에서 동적으로
구성한다. ``fetch_tickers`` 는 무거우니 TTL 캐시로 호출 빈도를 제한한다. 조회
실패 시 기존 캐시를 유지해 일시적 네트워크 장애가 가동을 막지 않게 한다.

BTC/ETH(메이저)는 항상 허용 목록에 포함되도록 보장한다 — 거래소 응답 지연/실패
시에도 기존 1순위 페어는 동작해야 하기 때문.

담당: 지영민 (페어 확장 PR — 화이트리스트 동적화)
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

logger = logging.getLogger(__name__)

# 메이저 — 거래소 조회 실패해도 항상 거래 가능 목록에 포함.
MAJOR_PAIRS = ("BTC/USDT:USDT", "ETH/USDT:USDT")


class _PairSource(Protocol):
    async def list_top_usdt_perps(self, limit: int = 30) -> list[str]: ...


class PairRegistry:
    """거래대금 상위 N 페어 화이트리스트 + TTL 캐시.

    Attributes:
        limit: 상위 몇 개를 허용할지.
        ttl_sec: 캐시 유효 기간(초).
    """

    def __init__(
        self,
        client: _PairSource,
        *,
        limit: int = 30,
        ttl_sec: float = 3600.0,
    ) -> None:
        self._client = client
        self.limit = limit
        self.ttl_sec = float(ttl_sec)
        self._cache: list[str] = list(MAJOR_PAIRS)
        self._fetched_at: float | None = None

    async def get_allowed(self, *, now: float | None = None) -> list[str]:
        """거래 가능 페어 목록 반환 — 캐시 만료 시 거래소에서 갱신.

        조회 결과가 비어있으면(실패) 기존 캐시를 유지한다. 어떤 경우에도 메이저
        (BTC/ETH)는 목록에 포함된다.
        """
        t = time.monotonic() if now is None else now
        fresh = (
            self._fetched_at is not None
            and (t - self._fetched_at) < self.ttl_sec
        )
        if not fresh:
            pairs = await self._client.list_top_usdt_perps(self.limit)
            if pairs:
                merged = list(pairs)
                for major in MAJOR_PAIRS:
                    if major not in merged:
                        merged.append(major)
                self._cache = merged
                self._fetched_at = t
            else:
                logger.warning("페어 목록 갱신 실패 — 기존 캐시 유지(%d개)",
                               len(self._cache))
        return list(self._cache)

    async def is_allowed(self, symbol: str, *, now: float | None = None) -> bool:
        """symbol 이 거래 가능 화이트리스트에 있는지."""
        return symbol in await self.get_allowed(now=now)


__all__ = ["PairRegistry", "MAJOR_PAIRS"]
