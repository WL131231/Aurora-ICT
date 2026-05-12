"""Killzone + Premium/Discount 박힌 거 — Aurora-ICT v0.1.2."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from aurora_ict.indicators.premium_discount import (
    DealingRange,
    PDZone,
    is_ote_zone,
    latest_dealing_range,
)
from aurora_ict.indicators.swing_points import SwingPoint, SwingType
from aurora_ict.timing.killzone import (
    STANDARD_KILLZONES,
    KillzoneName,
    classify_killzone,
    in_killzone,
    in_silver_bullet,
)

NY = ZoneInfo("America/New_York")


def _ny_to_ms(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    """NY local datetime → UTC ms."""
    dt = datetime(year, month, day, hour, minute, tzinfo=NY)
    return int(dt.timestamp() * 1000)


# ============================================================
# Killzone
# ============================================================


def test_in_killzone_london_basic() -> None:
    """NY 03:00 박힘 — London killzone (2-5am) 박힘 안 박힘."""
    ts = _ny_to_ms(2026, 5, 12, 3, 0)
    london = next(k for k in STANDARD_KILLZONES if k.name is KillzoneName.LONDON)
    assert in_killzone(ts, london) is True


def test_in_killzone_london_boundary() -> None:
    """02:00 박힘 시작 박힙 (inclusive). 05:00 박힘 끝 박힙 (exclusive)."""
    london = next(k for k in STANDARD_KILLZONES if k.name is KillzoneName.LONDON)
    assert in_killzone(_ny_to_ms(2026, 5, 12, 2, 0), london) is True
    assert in_killzone(_ny_to_ms(2026, 5, 12, 4, 59), london) is True
    assert in_killzone(_ny_to_ms(2026, 5, 12, 5, 0), london) is False


def test_in_killzone_outside() -> None:
    """NY 12:00 박힘 — London killzone 박힘 X."""
    ts = _ny_to_ms(2026, 5, 12, 12, 0)
    london = next(k for k in STANDARD_KILLZONES if k.name is KillzoneName.LONDON)
    assert in_killzone(ts, london) is False


def test_classify_killzone_london() -> None:
    """NY 03:00 → London."""
    ts = _ny_to_ms(2026, 5, 12, 3, 30)
    assert classify_killzone(ts) is KillzoneName.LONDON


def test_classify_killzone_ny_am() -> None:
    """NY 08:00 → NY_AM."""
    ts = _ny_to_ms(2026, 5, 12, 8, 30)
    assert classify_killzone(ts) is KillzoneName.NY_AM


def test_classify_killzone_pm() -> None:
    """NY 14:00 → PM."""
    ts = _ny_to_ms(2026, 5, 12, 14, 30)
    assert classify_killzone(ts) is KillzoneName.PM


def test_classify_killzone_asian() -> None:
    """NY 20:00 → Asian."""
    ts = _ny_to_ms(2026, 5, 12, 20, 0)
    assert classify_killzone(ts) is KillzoneName.ASIAN


def test_classify_killzone_none() -> None:
    """NY 06:00 박힌 거 (London 박힘 끝 + NY AM 박힘 시작 박힌 사이) → None."""
    ts = _ny_to_ms(2026, 5, 12, 6, 0)
    assert classify_killzone(ts) is None


def test_in_silver_bullet_london() -> None:
    """NY 03:30 → london_sb."""
    ts = _ny_to_ms(2026, 5, 12, 3, 30)
    assert in_silver_bullet(ts) == "london_sb"


def test_in_silver_bullet_am() -> None:
    """NY 10:30 → am_sb."""
    ts = _ny_to_ms(2026, 5, 12, 10, 30)
    assert in_silver_bullet(ts) == "am_sb"


def test_in_silver_bullet_pm() -> None:
    """NY 14:30 → pm_sb."""
    ts = _ny_to_ms(2026, 5, 12, 14, 30)
    assert in_silver_bullet(ts) == "pm_sb"


def test_in_silver_bullet_none() -> None:
    """NY 12:00 → SB 윈도우 박힘 X."""
    ts = _ny_to_ms(2026, 5, 12, 12, 0)
    assert in_silver_bullet(ts) is None


def test_dst_handling() -> None:
    """DST 박힘 박힘 — 2026-03-08 02:00 박힘 EST → EDT (서머타임 박힘 시작)."""
    # 박힌 2026-03-08 (DST 시작) NY 03:00 박힌 거 박힙 박힘 London 박힙
    ts = _ny_to_ms(2026, 3, 8, 3, 0)
    london = next(k for k in STANDARD_KILLZONES if k.name is KillzoneName.LONDON)
    assert in_killzone(ts, london) is True


# ============================================================
# Premium/Discount + Dealing Range
# ============================================================


def test_dealing_range_basic() -> None:
    """high=110 / low=90 → equilibrium=100, size=20."""
    dr = DealingRange(high=110, low=90, high_idx=10, low_idx=5)
    assert dr.equilibrium == 100.0
    assert dr.size == 20.0


def test_pd_zone_classify() -> None:
    """price=105 → premium, price=95 → discount, price=100 → equilibrium."""
    dr = DealingRange(high=110, low=90, high_idx=10, low_idx=5)
    assert dr.classify(105) is PDZone.PREMIUM
    assert dr.classify(95) is PDZone.DISCOUNT
    assert dr.classify(100) is PDZone.EQUILIBRIUM
    # tolerance_pct=0.001 박힘 → 100 ±0.1 박힘 박힙 equilibrium
    assert dr.classify(100.05) is PDZone.EQUILIBRIUM


def test_pd_fib_levels() -> None:
    """fib level 박힌 거 박힘."""
    dr = DealingRange(high=110, low=90, high_idx=10, low_idx=5)
    assert dr.fib_level(0) == 90.0
    assert dr.fib_level(0.5) == 100.0
    assert dr.fib_level(1) == 110.0
    assert dr.fib_level(0.618) == pytest.approx(102.36, rel=1e-3)


def test_latest_dealing_range_basic() -> None:
    """가장 최근 swing high + swing low 박힌 거 박힘."""
    swings = [
        SwingPoint(ts_ms=1000, type=SwingType.LOW, price=90, idx=1),
        SwingPoint(ts_ms=2000, type=SwingType.HIGH, price=105, idx=3),
        SwingPoint(ts_ms=3000, type=SwingType.LOW, price=92, idx=5),
        SwingPoint(ts_ms=4000, type=SwingType.HIGH, price=110, idx=7),
    ]
    dr = latest_dealing_range(swings)
    assert dr is not None
    assert dr.high == 110
    assert dr.low == 92
    assert dr.high_idx == 7
    assert dr.low_idx == 5


def test_latest_dealing_range_no_high() -> None:
    """swing high 박힘 X → None."""
    swings = [SwingPoint(ts_ms=1000, type=SwingType.LOW, price=90, idx=1)]
    assert latest_dealing_range(swings) is None


def test_ote_zone_long() -> None:
    """long bias → fib 0.214 ~ 0.382 박힘 박은 discount OTE."""
    dr = DealingRange(high=110, low=90, high_idx=10, low_idx=5)
    # fib 0.3 박힌 거 = 90 + 20*0.3 = 96
    assert is_ote_zone(96, dr, bias="long") is True
    # fib 0.5 박힌 거 = 100 (밖)
    assert is_ote_zone(100, dr, bias="long") is False


def test_ote_zone_short() -> None:
    """short bias → fib 0.618 ~ 0.786 박힘 박은 premium OTE."""
    dr = DealingRange(high=110, low=90, high_idx=10, low_idx=5)
    # fib 0.7 박힌 거 = 90 + 20*0.7 = 104
    assert is_ote_zone(104, dr, bias="short") is True
    assert is_ote_zone(100, dr, bias="short") is False


def test_ote_zone_invalid_bias() -> None:
    dr = DealingRange(high=110, low=90, high_idx=10, low_idx=5)
    with pytest.raises(ValueError, match="invalid bias"):
        is_ote_zone(100, dr, bias="sideways")
