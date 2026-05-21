"""CBDR (Central Bank Dealer's Range) — ICT 정통 시간 박스.

ICT 정의 ([[ict-canon-killzones]]):
    - 시간: **14:00 – 20:00 NY local** (Practical 4판)
    - 측정: 그 6시간 동안의 high - low (body 기준, wick 제외)
    - 표준편차: range 의 0.5x / 1x / 1.5x / 2x / 3x 위·아래로 복제
    - 다음날 Judas swing protraction 한도 = 2-3 std dev (이 범위 안에서 sweep
      예측, 초과 시 reversal 확률 ↑)

사용:
    1. 매일 20:00 NY 시점에 CBDR 박스 확정
    2. 다음 day session 시작 시 가격 위치로 bias 분기:
       - CBDR 안: 양방향 가능, 진입 자제
       - CBDR 위 (+1 std): bullish bias
       - CBDR 아래 (-1 std): bearish bias
       - ±2 std 도달: reversal 후보
       - ±3 std 도달: extreme — 강한 reversal 또는 trend 가속

ICT 의 다른 시간 박스와 관계:
    - Asian Range (19:00-00:00) 와 시간이 살짝 겹침 (19:00-20:00)
    - CBDR 가 더 길고 (6h), Asian Range 더 좁음 (5h)
    - 둘 다 다음날 bias 자료지만 CBDR 가 더 큰 그림

crypto 24/7 적용 시 주의:
    - 정통은 forex/futures 기준 — crypto 는 weekend 도 거래
    - DST 자동 처리 (zoneinfo)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, time
from enum import StrEnum
from zoneinfo import ZoneInfo

import pandas as pd

NY_TZ = ZoneInfo("America/New_York")

# CBDR 시간 박스 — 14:00 ~ 20:00 NY local (6시간)
CBDR_START_HOUR = 14
CBDR_END_HOUR = 20

# 표준편차 배수 — 다음날 Judas swing 한도 측정용
STD_DEV_MULTIPLIERS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)


class CBDRBiasState(StrEnum):
    """다음날 가격이 CBDR 박스 대비 어디 위치 — bias 분기."""

    INSIDE = "inside"                   # CBDR 박스 안 — 양방향 가능, 진입 자제
    ABOVE_1STD = "above_1std"           # +1 std 위 — bullish bias
    ABOVE_2STD = "above_2std"           # +2 std 위 — reversal 후보
    ABOVE_3STD = "above_3std"           # +3 std 위 — extreme bullish
    BELOW_1STD = "below_1std"           # -1 std 아래 — bearish bias
    BELOW_2STD = "below_2std"           # -2 std 아래 — reversal 후보
    BELOW_3STD = "below_3std"           # -3 std 아래 — extreme bearish


@dataclass(slots=True, frozen=True)
class CBDRBox:
    """하루치 CBDR 박스.

    Attributes:
        date: NY local 기준 날짜 (이 박스가 측정된 날, 예: 2026-05-21).
              다음날 (2026-05-22) 의 bias 자료로 사용.
        high: CBDR 시간대 (14:00-20:00 NY) 의 body 최고가.
        low: 같은 시간대의 body 최저가.
        range_width: high - low.
    """

    date: str                # ISO 날짜 (YYYY-MM-DD, NY local)
    high: float
    low: float
    range_width: float

    @property
    def mid(self) -> float:
        """박스 중심선."""
        return (self.high + self.low) / 2.0

    def std_dev_level(self, multiplier: float, *, above: bool) -> float:
        """``multiplier × range`` 만큼 박스 위/아래 가격.

        Args:
            multiplier: 0.5 / 1.0 / 1.5 / 2.0 / 3.0 중 하나 (정통).
            above: True 면 박스 위 (high + N*range), False 면 아래 (low - N*range).
        """
        if above:
            return self.high + multiplier * self.range_width
        return self.low - multiplier * self.range_width


def _ts_ms_to_ny(ts_ms: int) -> datetime:
    """UTC ms → NY local datetime (DST 자동)."""
    return datetime.fromtimestamp(ts_ms / 1000, tz=UTC).astimezone(NY_TZ)


def _in_cbdr_window(ny_dt: datetime) -> bool:
    """주어진 NY local datetime 이 CBDR 시간 (14:00-20:00) 안인지.

    end_hour 가 20 이면 19:59:59.999 까지 포함 (20:00 정각은 미포함).
    """
    t = ny_dt.time()
    return time(CBDR_START_HOUR, 0) <= t < time(CBDR_END_HOUR, 0)


def detect_cbdr_boxes(df: pd.DataFrame) -> list[CBDRBox]:
    """OHLCV DataFrame 에서 일별 CBDR 박스 추출.

    Args:
        df: ``timestamp`` (UTC ms), ``open``, ``high``, ``low``, ``close`` 컬럼 보유.

    Returns:
        ``CBDRBox`` 리스트 — 날짜순. body 기준 (open/close 의 max/min) 사용.

    Note:
        ICT 는 body 기준 (wick 제외) 권장. wick 으로 박스 측정하면 단일 노이즈
        봉 하나가 박스 폭 왜곡. 본 함수는 body 만 사용.
    """
    if df.empty:
        return []

    timestamps = df["timestamp"].to_numpy()
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()

    # 일별 (NY local) 그룹화 — CBDR 시간대 봉만 모음
    buckets: dict[str, list[tuple[float, float]]] = {}
    for i in range(len(df)):
        ny_dt = _ts_ms_to_ny(int(timestamps[i]))
        if not _in_cbdr_window(ny_dt):
            continue
        date_key = ny_dt.strftime("%Y-%m-%d")
        body_high = max(float(opens[i]), float(closes[i]))
        body_low = min(float(opens[i]), float(closes[i]))
        buckets.setdefault(date_key, []).append((body_high, body_low))

    boxes: list[CBDRBox] = []
    for date_key in sorted(buckets):
        bodies = buckets[date_key]
        if not bodies:
            continue
        high = max(b[0] for b in bodies)
        low = min(b[1] for b in bodies)
        boxes.append(CBDRBox(
            date=date_key,
            high=high,
            low=low,
            range_width=high - low,
        ))
    return boxes


def classify_price_vs_cbdr(
    price: float,
    cbdr: CBDRBox,
) -> CBDRBiasState:
    """현재 가격이 CBDR 박스 대비 어디 위치인지 분류.

    Args:
        price: 평가할 가격 (보통 다음날 Asian/London 시작 가격).
        cbdr: 어제 CBDR 박스.

    Returns:
        ``CBDRBiasState`` — 다음날 bias 분기 자료.
    """
    if cbdr.low <= price <= cbdr.high:
        return CBDRBiasState.INSIDE

    if price > cbdr.high:
        # 위쪽 — 1/2/3 std 단계 분기
        if price >= cbdr.std_dev_level(3.0, above=True):
            return CBDRBiasState.ABOVE_3STD
        if price >= cbdr.std_dev_level(2.0, above=True):
            return CBDRBiasState.ABOVE_2STD
        return CBDRBiasState.ABOVE_1STD

    # price < cbdr.low
    if price <= cbdr.std_dev_level(3.0, above=False):
        return CBDRBiasState.BELOW_3STD
    if price <= cbdr.std_dev_level(2.0, above=False):
        return CBDRBiasState.BELOW_2STD
    return CBDRBiasState.BELOW_1STD


def is_within_acceptable_range(cbdr: CBDRBox, max_range_pct: float = 0.005) -> bool:
    """ICT 권장: CBDR 폭이 너무 크면 그날 패스 / scalp 만.

    정통 기준 (forex pips): <40 pips. crypto 는 % 환산 — 0.5% 가 보수적 default.

    Args:
        cbdr: CBDR 박스.
        max_range_pct: 허용 최대 폭 (mid 대비 %). 0.005 = 0.5%.

    Returns:
        True 면 정상 폭, False 면 너무 크니 진입 자제.
    """
    if cbdr.mid <= 0 or math.isnan(cbdr.range_width):
        return False
    return (cbdr.range_width / cbdr.mid) <= max_range_pct


__all__ = [
    "CBDR_END_HOUR",
    "CBDR_START_HOUR",
    "CBDRBiasState",
    "CBDRBox",
    "NY_TZ",
    "STD_DEV_MULTIPLIERS",
    "classify_price_vs_cbdr",
    "detect_cbdr_boxes",
    "is_within_acceptable_range",
]
