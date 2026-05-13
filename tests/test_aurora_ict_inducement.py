"""Inducement (IDM) 단위 테스트."""

from __future__ import annotations

from aurora_ict.indicators.inducement import detect_inducements
from aurora_ict.indicators.swing_points import SwingPoint, SwingType


def _sw(t: SwingType, price: float, idx: int) -> SwingPoint:
    return SwingPoint(ts_ms=idx * 60_000, type=t, price=price, idx=idx, swept=False)


def test_idm_empty_inputs() -> None:
    assert detect_inducements([], []) == []
    assert detect_inducements([_sw(SwingType.HIGH, 100, 0)], []) == []


def test_idm_high_pair() -> None:
    """큰 high(idx=50, 120) 직전 작은 high(idx=45, 110) → 페어."""
    big = [_sw(SwingType.HIGH, 120, 50)]
    small = [_sw(SwingType.HIGH, 110, 45)]
    idms = detect_inducements(big, small, lookback=20)
    assert len(idms) == 1
    idm = idms[0]
    assert idm.type == "high"
    assert idm.target_idx == 50
    assert idm.idm_idx == 45
    assert idm.target_price == 120
    assert idm.idm_price == 110


def test_idm_low_pair() -> None:
    """큰 low(idx=50, 80) 직전 작은 low(idx=45, 90) → 페어 (작은 low 가 큰 low 보다 위)."""
    big = [_sw(SwingType.LOW, 80, 50)]
    small = [_sw(SwingType.LOW, 90, 45)]
    idms = detect_inducements(big, small, lookback=20)
    assert len(idms) == 1
    assert idms[0].type == "low"


def test_idm_outside_lookback_skipped() -> None:
    """lookback 밖 작은 swing 은 페어 아님."""
    big = [_sw(SwingType.HIGH, 120, 50)]
    small = [_sw(SwingType.HIGH, 110, 20)]  # idx=20, lookback=20 → 50-20=30 미달
    idms = detect_inducements(big, small, lookback=20)
    assert idms == []


def test_idm_high_skips_if_small_price_above() -> None:
    """High 페어: 작은 high.price < 큰 high.price 만 인정 (큰 거 위면 X)."""
    big = [_sw(SwingType.HIGH, 120, 50)]
    small = [_sw(SwingType.HIGH, 125, 45)]
    idms = detect_inducements(big, small, lookback=20)
    assert idms == []


def test_idm_picks_closest_when_multiple() -> None:
    """여러 작은 swing 중 큰 swing 에 가장 가까운 (idx 큰) 것 선택."""
    big = [_sw(SwingType.HIGH, 120, 50)]
    small = [
        _sw(SwingType.HIGH, 105, 35),
        _sw(SwingType.HIGH, 110, 45),  # 더 가까움
        _sw(SwingType.HIGH, 108, 40),
    ]
    idms = detect_inducements(big, small, lookback=20)
    assert len(idms) == 1
    assert idms[0].idm_idx == 45


def test_idm_direction_mismatch_skipped() -> None:
    """반대 방향 swing 은 페어 아님."""
    big = [_sw(SwingType.HIGH, 120, 50)]
    small = [_sw(SwingType.LOW, 80, 45)]
    idms = detect_inducements(big, small, lookback=20)
    assert idms == []
