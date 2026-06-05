"""경량 in-memory rate limiter — brute-force / 자동화 남용 / app-layer DoS 완화.

외부 의존성 없이(slowapi 등 미사용) sliding-window 카운터로 IP×경로 단위 요청
빈도를 제한한다. Fly.io 운영은 단일 머신(min_machines_running=1)이라 인스턴스별
in-memory 상태로 충분하다. 다중 머신으로 확장하면 인스턴스마다 카운터가 분리되어
실효 한도가 머신 수만큼 늘어나므로, 그때는 Redis 등 공유 저장소 기반으로 교체해야
한다.

담당: 지영민 (보안 강화 PR — rate limiting 도입)
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    """Sliding-window in-memory rate limiter (thread-safe).

    ``key`` (보통 ``"<path>:<ip>"``) 별로 최근 ``window_sec`` 초 내 허용 횟수를
    ``max_hits`` 로 제한한다. uvicorn worker 가 다중 스레드로 핸들러를 돌릴 수
    있으므로 내부 상태를 ``Lock`` 으로 보호한다.

    Attributes:
        max_hits: window 내 허용 최대 요청 수.
        window_sec: 슬라이딩 윈도우 길이(초).
    """

    def __init__(self, max_hits: int, window_sec: float) -> None:
        """RateLimiter 초기화.

        Args:
            max_hits: window 내 허용 최대 요청 수 (1 이상).
            window_sec: 슬라이딩 윈도우 길이(초, 양수).

        Raises:
            ValueError: max_hits < 1 또는 window_sec <= 0.
        """
        if max_hits < 1:
            raise ValueError("max_hits 는 1 이상이어야 합니다.")
        if window_sec <= 0:
            raise ValueError("window_sec 는 양수여야 합니다.")
        self.max_hits = max_hits
        self.window_sec = float(window_sec)
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()
        # 만료 키 lazy GC 마지막 실행 시각 — 메모리 누수 방지.
        self._last_gc = 0.0

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """key 요청 1건을 평가하고 허용 여부를 반환한다.

        Args:
            key: 제한 단위 식별자 (예: ``"/auth/login:1.2.3.4"``).
            now: 테스트용 시각 주입(monotonic 초). None 이면 ``time.monotonic()``.

        Returns:
            True  — 허용(한도 내). 이 호출로 hit 1건이 적립된다.
            False — 한도 초과(차단). hit 은 적립하지 않는다.
        """
        t = time.monotonic() if now is None else now
        with self._lock:
            self._maybe_gc(t)
            window_start = t - self.window_sec
            hits = [h for h in self._hits.get(key, ()) if h > window_start]
            if len(hits) >= self.max_hits:
                # 만료분만 제거해 갱신, 새 hit 은 적립하지 않음(차단).
                self._hits[key] = hits
                return False
            hits.append(t)
            self._hits[key] = hits
            return True

    def _maybe_gc(self, now: float) -> None:
        """window 의 10배 주기로 만료 키를 일괄 제거한다(lock 보유 중 호출).

        Why: allow() 가 호출되지 않는 key 는 영원히 dict 에 남아 메모리가
        누적된다. 전수 스캔은 비싸므로 호출 빈도를 window×10 으로 제한한다.
        """
        if now - self._last_gc < self.window_sec * 10:
            return
        self._last_gc = now
        cutoff = now - self.window_sec
        dead = [k for k, v in self._hits.items() if not any(h > cutoff for h in v)]
        for k in dead:
            del self._hits[k]


__all__ = ["RateLimiter"]
