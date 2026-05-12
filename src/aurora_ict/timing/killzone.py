"""Killzone 시간 필터 + Silver Bullet 윈도우 — ICT Time Theory.

ICT 박힌 정의 (NY EST 기준 — 표준):

| Killzone        | NY EST           | 박힌 거                              |
|-----------------|------------------|--------------------------------------|
| Asian           | 19:00 – 24:00    | range 박는 거 (low volatility)       |
| London          | 02:00 – 05:00    | 최고 directional move 박힘            |
| NY AM           | 07:00 – 10:00    | London-NY overlap (highest volume)   |
| London Close    | 10:00 – 12:00    | retrace 박힘                          |
| PM Session      | 13:30 – 16:00    | reversal                              |

Silver Bullet (3개 1시간 윈도우 — ICT 핵심 entry 시간):

| 윈도우          | NY EST           |
|-----------------|------------------|
| London SB       | 03:00 – 04:00    |
| AM SB           | 10:00 – 11:00    |
| PM SB           | 14:00 – 15:00    |

NY EST 박힌 거 = America/New_York timezone 박힘. DST (서머타임) 박힘 → 박힌 거 박힌
EDT 박힘 박힘 (3월 둘째 일요일 ~ 11월 첫째 일요일). zoneinfo 박힌 거 박혀서 자동 박힘.

박힌 거 박힌 거 = NY 박힘 박은 게 박힘 박힌 거. 그래서 박힌 timestamp 박힌 거 (UTC ms 박힘)
박은 거 박힌 거 박혀서 NY 박힌 거 박힌 거 박힘 박힘.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


class KillzoneName(StrEnum):
    """Killzone 박힌 거 이름."""

    ASIAN = "asian"
    LONDON = "london"
    NY_AM = "ny_am"
    LONDON_CLOSE = "london_close"
    PM = "pm"


@dataclass(frozen=True, slots=True)
class Killzone:
    """Killzone 1개 박힌 거.

    Attributes:
        name: Killzone 종류.
        start: NY local time 시작 (시, 분).
        end: NY local time 끝 (시, 분). end 박힘 박는 거 박는 거 박힌 시간 박힘 (exclusive).
        crosses_midnight: True 박힘 — Asian 박힘 19:00 → 박힌 24:00 박힘 박힘.
    """

    name: KillzoneName
    start: time
    end: time
    crosses_midnight: bool = False


# 표준 Killzone 박힌 거 (NY local time)
STANDARD_KILLZONES: tuple[Killzone, ...] = (
    Killzone(KillzoneName.ASIAN, time(19, 0), time(23, 59, 59), crosses_midnight=False),
    Killzone(KillzoneName.LONDON, time(2, 0), time(5, 0)),
    Killzone(KillzoneName.NY_AM, time(7, 0), time(10, 0)),
    Killzone(KillzoneName.LONDON_CLOSE, time(10, 0), time(12, 0)),
    Killzone(KillzoneName.PM, time(13, 30), time(16, 0)),
)

# Silver Bullet 윈도우 (NY local time) — 3개 1시간
SILVER_BULLET_WINDOWS: tuple[tuple[str, time, time], ...] = (
    ("london_sb", time(3, 0), time(4, 0)),
    ("am_sb", time(10, 0), time(11, 0)),
    ("pm_sb", time(14, 0), time(15, 0)),
)

# Macros — Silver Bullet 안의 더 정밀한 sub-window (20-30분).
# ICT 자료(Lumi Traders ICT Killzones cheat sheet) 기준 — 이 시간대에 setup 트리거
# 정밀도가 가장 높음 (변동성 + 진입 빈도). Silver Bullet 의 좁은 sub-window.
MACRO_WINDOWS: tuple[tuple[str, time, time], ...] = (
    # London session — Silver Bullet (03:00-04:00) 안팎의 2개 sub-window
    ("london_macro_1", time(2, 33), time(3, 0)),
    ("london_macro_2", time(4, 3),  time(4, 30)),
    # NY AM session — Silver Bullet (10:00-11:00) 전후 3개
    ("am_macro_1",     time(8, 50),  time(9, 10)),
    ("am_macro_2",     time(9, 50),  time(10, 10)),
    ("am_macro_3",     time(10, 50), time(11, 10)),
    # NY Lunch + PM
    ("lunch_macro",    time(11, 50), time(12, 10)),
    ("pm_macro_1",     time(13, 10), time(13, 40)),
    ("pm_macro_2",     time(15, 15), time(15, 45)),
)


def _to_ny_time(ts_ms: int) -> datetime:
    """UTC ms → NY local datetime (DST 자동 박힘)."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).astimezone(NY_TZ)


def _within(t: time, start: time, end: time) -> bool:
    """``start <= t < end`` 박힘. crosses midnight 박힘 X."""
    return start <= t < end


def in_killzone(ts_ms: int, killzone: Killzone) -> bool:
    """주어진 timestamp 박힌 거 박힌 killzone 박힌 거 박힘 안 박힘 박은 거."""
    ny = _to_ny_time(ts_ms).time()
    if killzone.crosses_midnight:
        # 박힘 박힘 박힌 거 박힌 거 박힙 박힘 박힘 (start ≤ t OR t < end)
        return ny >= killzone.start or ny < killzone.end
    return _within(ny, killzone.start, killzone.end)


def classify_killzone(ts_ms: int) -> KillzoneName | None:
    """주어진 timestamp 박힌 거 박힌 killzone 박힌 거 박힘 박힘.

    여러 killzone 박힌 거 박은 거 박힘 (e.g. London Close + NY AM 10:00 박힘 동시) 박힘
    박힌 거 박힌 거 박힘 박힘 — 박힌 거 박힌 거 박힘 STANDARD_KILLZONES 박힌 순서 박힘
    첫 번째 match 박힘.

    Returns:
        ``KillzoneName`` 박힙 박힌 killzone 박힌 거 박힘 박힘 ``None``.
    """
    for kz in STANDARD_KILLZONES:
        if in_killzone(ts_ms, kz):
            return kz.name
    return None


def in_silver_bullet(ts_ms: int) -> str | None:
    """Silver Bullet 윈도우 박힌 거 박힘.

    Returns:
        박힌 windows 박힌 거 이름 (``"london_sb"`` / ``"am_sb"`` / ``"pm_sb"``) 박힘,
        없으면 ``None``.
    """
    ny = _to_ny_time(ts_ms).time()
    for name, start, end in SILVER_BULLET_WINDOWS:
        if _within(ny, start, end):
            return name
    return None


def in_macro(ts_ms: int) -> str | None:
    """Macro 윈도우 (Silver Bullet 안의 정밀 sub-window).

    Returns:
        매칭되는 macro 이름 (``"london_macro_1"``, ``"am_macro_2"`` 등),
        없으면 ``None``.
    """
    ny = _to_ny_time(ts_ms).time()
    for name, start, end in MACRO_WINDOWS:
        if _within(ny, start, end):
            return name
    return None


__all__ = [
    "MACRO_WINDOWS",
    "Killzone",
    "KillzoneName",
    "NY_TZ",
    "SILVER_BULLET_WINDOWS",
    "STANDARD_KILLZONES",
    "classify_killzone",
    "in_killzone",
    "in_macro",
    "in_silver_bullet",
]
