"""Power of 3 (AMD) 단위 테스트."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from aurora_ict.timing.power_of_3 import AmdPhase, amd_phase

NY = ZoneInfo("America/New_York")


def _ny_ms(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    dt = datetime(year, month, day, hour, minute, tzinfo=NY)
    return int(dt.timestamp() * 1000)


def test_amd_accumulation_late_asian() -> None:
    """NY 20:00 → Asian accumulation."""
    assert amd_phase(_ny_ms(2026, 5, 12, 20, 0)) is AmdPhase.ACCUMULATION


def test_amd_accumulation_early_morning() -> None:
    """NY 01:00 → 새벽 = Asian accumulation 연속."""
    assert amd_phase(_ny_ms(2026, 5, 12, 1, 0)) is AmdPhase.ACCUMULATION


def test_amd_manipulation_london() -> None:
    """NY 03:30 → London = Manipulation."""
    assert amd_phase(_ny_ms(2026, 5, 12, 3, 30)) is AmdPhase.MANIPULATION


def test_amd_distribution_ny() -> None:
    """NY 10:00 → NY = Distribution."""
    assert amd_phase(_ny_ms(2026, 5, 12, 10, 0)) is AmdPhase.DISTRIBUTION


def test_amd_afterhours_none() -> None:
    """NY 17:00 (16:00-19:00 afterhours) → None."""
    assert amd_phase(_ny_ms(2026, 5, 12, 17, 0)) is None


def test_amd_boundary_02_00() -> None:
    """02:00 NY 정각 → Manipulation 시작 (Accumulation end exclusive)."""
    assert amd_phase(_ny_ms(2026, 5, 12, 1, 59)) is AmdPhase.ACCUMULATION
    assert amd_phase(_ny_ms(2026, 5, 12, 2, 0)) is AmdPhase.MANIPULATION


def test_amd_boundary_07_00() -> None:
    """07:00 NY 정각 → Distribution 시작."""
    assert amd_phase(_ny_ms(2026, 5, 12, 6, 59)) is AmdPhase.MANIPULATION
    assert amd_phase(_ny_ms(2026, 5, 12, 7, 0)) is AmdPhase.DISTRIBUTION


def test_amd_boundary_16_00() -> None:
    """16:00 NY 정각 → afterhours (None) 시작."""
    assert amd_phase(_ny_ms(2026, 5, 12, 15, 59)) is AmdPhase.DISTRIBUTION
    assert amd_phase(_ny_ms(2026, 5, 12, 16, 0)) is None
