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

from dataclasses import dataclass
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
    "ClassicPo3Day",
    "ClassicPo3Type",
    "amd_phase",
    "classify_classic_po3_day",
]


# ============================================================
# Classic Po3 (그룹 2 — 가격 동작 기반 day 분류)
# ============================================================
#
# ICT 정의 (Practical Po3 + [[ict-canon-models]]):
#     기존 amd_phase 는 *시간만* 으로 분류 — 단순하지만 가격이 정통 패턴 안 따를 때
#     phase 오인식 가능. Classic Po3 는 *가격 동작 기반* 으로 일일 buy / sell day
#     분류.
#
#     - **00:00 NY local (midnight) open** = 진정한 일일 opening price ([[ict-canon-killzones]])
#     - **Classic Buy Day**: 00:00 open *아래로* manipulation (London 가 low 형성)
#       → NY 에서 open *위로* distribution (위쪽이 진짜 방향)
#     - **Classic Sell Day**: 00:00 open *위로* manipulation (London 가 high 형성)
#       → NY 에서 open *아래로* distribution (아래쪽이 진짜 방향)
#     - **Neutral / Indecision Day**: 명확한 manipulation 방향 안 보임
#
# 활용:
#     - Daily Bias 확정 자료 — PDH/PDL 돌파 방식과 보완
#     # Buy Day 면 long 매매만 채택, Sell Day 면 short 만, Neutral 이면 양방향 또는 패스


class ClassicPo3Type(StrEnum):
    """가격 동작 기반 일일 분류."""

    BUY_DAY = "buy_day"           # London 이 daily open 아래로 sweep → NY 위로
    SELL_DAY = "sell_day"         # London 이 daily open 위로 sweep → NY 아래로
    NEUTRAL = "neutral"           # 명확한 패턴 없음


@dataclass(slots=True, frozen=True)
class ClassicPo3Day:
    """하루치 Classic Po3 분류 결과.

    Attributes:
        date: NY local 날짜 (YYYY-MM-DD).
        type: BUY_DAY / SELL_DAY / NEUTRAL.
        midnight_open: 00:00 NY 의 진정한 일일 open 가격.
        london_extreme: London phase 의 manipulation 극점
            (Buy Day = low, Sell Day = high). NEUTRAL 이면 None.
    """

    date: str
    type: ClassicPo3Type
    midnight_open: float
    london_extreme: float | None


def classify_classic_po3_day(
    midnight_open: float,
    london_low: float,
    london_high: float,
    *,
    min_sweep_pct: float = 0.001,
) -> ClassicPo3Day:
    """하루의 가격 동작으로 Classic Po3 분류.

    Args:
        midnight_open: 00:00 NY 의 일일 open 가격.
        london_low: London phase (02:00-07:00) 의 최저가.
        london_high: 같은 시간대 최고가.
        min_sweep_pct: midnight_open 대비 sweep 최소 깊이 (%). 너무 작은 wick
            는 manipulation 으로 안 봄. 0.001 = 0.1%.

    Returns:
        ``ClassicPo3Day`` — date 는 별도로 채워야 (호출자 책임).

    Note:
        date 필드는 빈 문자열로 채워서 반환 — 호출자가 NY local 날짜 박을 것.
        시계열 검출 단순화를 위해 hour-level filter 는 호출자가 미리 처리한다고 가정.
    """
    threshold = midnight_open * min_sweep_pct

    swept_below = midnight_open - london_low > threshold
    swept_above = london_high - midnight_open > threshold

    if swept_below and not swept_above:
        # Buy Day: London 이 일일 open 아래로 sweep
        return ClassicPo3Day(
            date="",
            type=ClassicPo3Type.BUY_DAY,
            midnight_open=midnight_open,
            london_extreme=london_low,
        )
    if swept_above and not swept_below:
        # Sell Day
        return ClassicPo3Day(
            date="",
            type=ClassicPo3Type.SELL_DAY,
            midnight_open=midnight_open,
            london_extreme=london_high,
        )
    # 양방향 sweep 또는 sweep 없음 → 명확한 manipulation X
    return ClassicPo3Day(
        date="",
        type=ClassicPo3Type.NEUTRAL,
        midnight_open=midnight_open,
        london_extreme=None,
    )
