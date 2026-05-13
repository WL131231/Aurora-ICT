"""Power of 3 (AMD) — ICT 의 일/세션 phase 구조 모델.

ICT 정의:
    하루의 가격 움직임은 3단계 cycle 로 나뉜다 (Accumulation / Manipulation /
    Distribution = AMD). Institutional flow 가 이 구조를 만든다.

세션 기반 분류 (NY local time):
    - **Accumulation** (Asian, 19:00 전일 ~ 02:00) — 좁은 range, 횡보. 유동성 축적.
    - **Manipulation** (London, 02:00 ~ 07:00) — false break, Asian range sweep.
      retail trader 유인 (가짜 방향).
    - **Distribution** (NY, 07:00 ~ 16:00) — 진짜 방향. institutional position
      분배. 가장 큰 directional move.

활용:
    - Accumulation 안에서는 진입 자제 (range 안 noise).
    - Manipulation 의 sweep 끝 → reversal 기회 (가짜 break 후 반대 방향).
    - Distribution 에서 setup 채택 = 가장 높은 확률.
"""

from __future__ import annotations

from datetime import UTC, datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


class AmdPhase(StrEnum):
    """Power of 3 phase."""

    ACCUMULATION = "accumulation"   # Asian — 횡보
    MANIPULATION = "manipulation"   # London — false break
    DISTRIBUTION = "distribution"   # NY — 진짜 방향


# Phase boundaries (NY local time)
# crosses midnight 없도록 단순화 — Accumulation 은 00:00-02:00 + 19:00-23:59 둘 다.
_ACCUMULATION_LATE = (time(19, 0), time(23, 59, 59))
_ACCUMULATION_EARLY = (time(0, 0), time(2, 0))
_MANIPULATION = (time(2, 0), time(7, 0))
_DISTRIBUTION = (time(7, 0), time(16, 0))
# 16:00 ~ 19:00 NY = "afterhours" — phase 없음 (None).


def _to_ny_time(ts_ms: int) -> datetime:
    """UTC ms → NY local datetime (DST 자동)."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).astimezone(NY_TZ)


def amd_phase(ts_ms: int) -> AmdPhase | None:
    """주어진 timestamp 의 AMD phase.

    Args:
        ts_ms: UTC ms timestamp.

    Returns:
        AmdPhase 또는 None (afterhours 16:00-19:00).
    """
    ny = _to_ny_time(ts_ms).time()
    if _ACCUMULATION_EARLY[0] <= ny < _ACCUMULATION_EARLY[1]:
        return AmdPhase.ACCUMULATION
    if _MANIPULATION[0] <= ny < _MANIPULATION[1]:
        return AmdPhase.MANIPULATION
    if _DISTRIBUTION[0] <= ny < _DISTRIBUTION[1]:
        return AmdPhase.DISTRIBUTION
    if _ACCUMULATION_LATE[0] <= ny <= _ACCUMULATION_LATE[1]:
        return AmdPhase.ACCUMULATION
    return None  # 16:00 ~ 19:00 NY = afterhours


__all__ = [
    "AmdPhase",
    "amd_phase",
]
