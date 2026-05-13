"""Killzone 시간 필터 + Silver Bullet 윈도우 — ICT Time Theory.

ICT 표준 정의 (NY EST 기준):

| Killzone        | NY EST           | 특성                                  |
|-----------------|------------------|--------------------------------------|
| Asian           | 19:00 – 24:00    | range 형성 (low volatility)          |
| London          | 02:00 – 05:00    | 최고 directional move 빈도            |
| NY AM           | 07:00 – 10:00    | London-NY overlap (highest volume)   |
| London Close    | 10:00 – 12:00    | retrace 자주 발생                     |
| PM Session      | 13:30 – 16:00    | reversal                              |

Silver Bullet (3개 1시간 윈도우 — ICT 핵심 entry 시간):

| 윈도우          | NY EST           |
|-----------------|------------------|
| London SB       | 03:00 – 04:00    |
| AM SB           | 10:00 – 11:00    |
| PM SB           | 14:00 – 15:00    |

NY EST = America/New_York timezone. DST (서머타임) 기간엔 자동으로 EDT 적용
(3월 둘째 일요일 ~ 11월 첫째 일요일). zoneinfo 사용해 자동 처리.

모든 시간 기준은 NY 로컬 시간. 외부에서 들어오는 timestamp(UTC ms)를 NY local로
변환해 비교한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")


class KillzoneName(StrEnum):
    """Killzone 이름 enum."""

    ASIAN = "asian"
    LONDON = "london"
    NY_AM = "ny_am"
    LONDON_CLOSE = "london_close"
    PM = "pm"


@dataclass(frozen=True, slots=True)
class Killzone:
    """Killzone 한 건의 정의.

    Attributes:
        name: Killzone 종류.
        start: NY local time 시작 (시, 분).
        end: NY local time 끝 (시, 분). exclusive (end 시각 자체는 미포함).
        crosses_midnight: True면 24:00을 가로지름 (Asian 19:00 → 다음날 새벽 같은 경우).
    """

    name: KillzoneName
    start: time
    end: time
    crosses_midnight: bool = False


# 표준 Killzone 정의 (NY local time)
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
    """UTC ms → NY local datetime (DST 자동 반영)."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).astimezone(NY_TZ)


def _within(t: time, start: time, end: time) -> bool:
    """``start <= t < end`` 검사. midnight를 가로지르지 않는 일반 케이스."""
    return start <= t < end


def in_killzone(ts_ms: int, killzone: Killzone) -> bool:
    """주어진 timestamp가 해당 killzone 안에 있는지."""
    ny = _to_ny_time(ts_ms).time()
    if killzone.crosses_midnight:
        # midnight를 가로지르는 zone (start ≤ t OR t < end)
        return ny >= killzone.start or ny < killzone.end
    return _within(ny, killzone.start, killzone.end)


def classify_killzone(ts_ms: int) -> KillzoneName | None:
    """주어진 timestamp가 속한 killzone을 반환.

    Killzone이 겹치는 시점 (e.g. London Close + NY AM 둘 다 10:00 포함)에서는
    STANDARD_KILLZONES에 정의된 순서대로 첫 매칭을 반환한다.

    Returns:
        ``KillzoneName``. 어느 killzone에도 속하지 않으면 ``None``.
    """
    for kz in STANDARD_KILLZONES:
        if in_killzone(ts_ms, kz):
            return kz.name
    return None


def in_silver_bullet(ts_ms: int) -> str | None:
    """Silver Bullet 윈도우 여부 판정.

    Returns:
        매칭되는 윈도우 이름 (``"london_sb"`` / ``"am_sb"`` / ``"pm_sb"``).
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


# Macro 우선순위 — ICT 책 + 시장 변동성 관찰 기반.
# - high: 가장 강한 신호. 큰 변동성 + 기관 유동성 활성.
# - medium: 정상 강도. 신호 발생률 ↑, 신뢰도 보통.
# - low: 약한 macro (lunch / 후반 PM). 노이즈 ↑.
MACRO_PRIORITY: dict[str, str] = {
    # AM session — 9:50-10:10 이 ICT 의 가장 핵심 macro
    "am_macro_2":     "high",
    "am_macro_1":     "medium",
    "am_macro_3":     "medium",
    # London — 새벽 macros 도 강함
    "london_macro_1": "high",
    "london_macro_2": "medium",
    # PM — 14-15시 NY 후반은 강하나 13시는 lunch 영향
    "pm_macro_2":     "high",
    "pm_macro_1":     "medium",
    # Lunch — 노이즈 큼
    "lunch_macro":    "low",
}


def macro_priority(name: str | None) -> str | None:
    """Macro 이름 → priority ('high'/'medium'/'low').

    None 또는 미정의 macro 는 None.
    """
    if name is None:
        return None
    return MACRO_PRIORITY.get(name)


__all__ = [
    "MACRO_PRIORITY",
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
    "macro_priority",
]
