"""심볼별 공유 OHLCV 캐시 — 봇 인스턴스 간 중복 제거 (메모리 절감).

담당: 지영민 (2026-07-22). 배경: 기존엔 봇 인스턴스마다 _ohlcv_cache(TF→봉) 를
보유해, 같은 심볼을 여러 유저가 돌리면 동일 시장데이터를 유저 수만큼 중복 보유했다
(80봇 = ~3GB). OHLCV 는 공개 시장데이터라 어느 유저 client 로 받아도 동일하므로,
(symbol, tf) 키로 전역 1벌만 공유하면 안전하게 중복을 제거한다(심볼당 1벌 →
고정7+선택 ~10심볼 × 2TF = 수십벌 전체, 봇 수 무관). 프리페치 네트워크 호출도
심볼당 1회로 줄어든다.

동시성: (symbol, tf) 별 asyncio.Lock 으로 같은 심볼을 동시에 시작한 봇들의 중복
fetch 를 직렬화. 저장 rows 는 교체(replace)만 하고 in-place 수정하지 않아, 다수 봇이
같은 rows 를 읽어도 안전(읽기 전용 공유).

라이프사이클: 심볼 집합이 작고 유한(고정7+선택 상한)이라 v1 은 evict 없이 유지.
심볼 pool 이 시간에 따라 커지면 refcount 기반 eviction 을 추가할 여지(TODO).
"""
from __future__ import annotations

import asyncio
from typing import Any


class SharedOhlcvCache:
    """(symbol, tf) 키 전역 OHLCV 캐시 + 키별 락.

    봇 인스턴스들이 동일 인스턴스를 참조해 시장데이터를 공유한다. 테스트는 격리를
    위해 별도 인스턴스를 주입할 수 있다.
    """

    def __init__(self) -> None:
        self._data: dict[tuple[str, str], list[list[Any]]] = {}
        self._locks: dict[tuple[str, str], asyncio.Lock] = {}

    def has(self, symbol: str, tf: str) -> bool:
        """(symbol, tf) 캐시 존재 여부."""
        return (symbol, tf) in self._data

    def get(self, symbol: str, tf: str) -> list[list[Any]] | None:
        """(symbol, tf) 캐시 반환 (없으면 None). 반환 list 는 내부 참조 —
        읽기 전용으로만 사용(교체는 set 으로)."""
        return self._data.get((symbol, tf))

    def set(self, symbol: str, tf: str, rows: list[list[Any]]) -> None:
        """(symbol, tf) 캐시 교체 저장."""
        self._data[(symbol, tf)] = rows

    def get_lock(self, symbol: str, tf: str) -> asyncio.Lock:
        """(symbol, tf) 별 락 — lazy init (이벤트 루프 안에서)."""
        key = (symbol, tf)
        lock = self._locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[key] = lock
        return lock

    def clear(self) -> None:
        """전체 비우기 (테스트/재시작용)."""
        self._data.clear()
        self._locks.clear()


# 전역 공유 인스턴스 — 모든 BotIctInstance 가 기본으로 이걸 참조(심볼별 1벌 공유).
GLOBAL_OHLCV_CACHE = SharedOhlcvCache()
