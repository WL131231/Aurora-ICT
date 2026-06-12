"""tf.py 단위 테스트 — 14 케이스 (DESIGN.md §7.7 spec 정합).

mock 0 — 결정론적 합성 입력만 사용. 외부 네트워크 X.

담당: ChoYoon
"""

from __future__ import annotations

import pytest

from aurora.backtest.tf import (
    TF_VALID_AURORA,
    TF_VALID_CCXT,
    is_valid_timeframe,
    normalize_to_aurora,
    normalize_to_ccxt,
)

# ============================================================
# 정상 변환 (4)
# ============================================================


def test_normalize_to_ccxt_aurora_uppercase():
    """대문자 Aurora 포맷 → 소문자 ccxt 포맷."""
    assert normalize_to_ccxt("1H") == "1h"
    assert normalize_to_ccxt("2H") == "2h"
    assert normalize_to_ccxt("4H") == "4h"
    assert normalize_to_ccxt("1D") == "1d"
    assert normalize_to_ccxt("1W") == "1w"


def test_normalize_to_ccxt_minutes_idempotent():
    """분 단위는 양쪽 포맷 동일 — 자연 idempotent."""
    assert normalize_to_ccxt("1m") == "1m"
    assert normalize_to_ccxt("3m") == "3m"
    assert normalize_to_ccxt("5m") == "5m"
    assert normalize_to_ccxt("15m") == "15m"


def test_normalize_to_aurora_ccxt_lowercase():
    """소문자 ccxt 포맷 → 대문자 Aurora 포맷."""
    assert normalize_to_aurora("1h") == "1H"
    assert normalize_to_aurora("2h") == "2H"
    assert normalize_to_aurora("4h") == "4H"
    assert normalize_to_aurora("1d") == "1D"
    assert normalize_to_aurora("1w") == "1W"


def test_normalize_to_aurora_minutes_idempotent():
    """분 단위는 양쪽 포맷 동일 — 자연 idempotent."""
    assert normalize_to_aurora("1m") == "1m"
    assert normalize_to_aurora("3m") == "3m"
    assert normalize_to_aurora("5m") == "5m"
    assert normalize_to_aurora("15m") == "15m"


# ============================================================
# 양방향 round-trip (2)
# ============================================================


def test_round_trip_aurora_to_ccxt_to_aurora():
    """모든 Aurora TF — aurora → ccxt → aurora 라운드트립."""
    for tf in TF_VALID_AURORA:
        ccxt_form = normalize_to_ccxt(tf)
        back = normalize_to_aurora(ccxt_form)
        assert back == tf, f"round-trip 깨짐: {tf} → {ccxt_form} → {back}"


def test_round_trip_ccxt_to_aurora_to_ccxt():
    """모든 ccxt TF — ccxt → aurora → ccxt 라운드트립."""
    for tf in TF_VALID_CCXT:
        aurora_form = normalize_to_aurora(tf)
        back = normalize_to_ccxt(aurora_form)
        assert back == tf, f"round-trip 깨짐: {tf} → {aurora_form} → {back}"


# ============================================================
# Strict 검증 — 잘못된 방향 reject (2)
# ============================================================


def test_normalize_to_ccxt_rejects_ccxt_format():
    """ccxt 포맷을 normalize_to_ccxt 에 — Strict reject.

    분 단위 (1m/3m/5m/15m) 는 양쪽 동일이라 reject 안 됨 (idempotent
    자연 통과). 시간/일/주 단위만 reject 대상.
    """
    for ccxt_tf in ["1h", "2h", "4h", "1d", "1w"]:
        with pytest.raises(ValueError, match="지원하지 않는 timeframe"):
            normalize_to_ccxt(ccxt_tf)


def test_normalize_to_aurora_rejects_aurora_format():
    """Aurora 포맷을 normalize_to_aurora 에 — Strict reject."""
    for aurora_tf in ["1H", "2H", "4H", "1D", "1W"]:
        with pytest.raises(ValueError, match="지원하지 않는 timeframe"):
            normalize_to_aurora(aurora_tf)


# ============================================================
# 알려지지 않은 TF (2)
# ============================================================


def test_normalize_to_ccxt_unknown_tf_raises():
    """지원 안 하는 TF — ValueError."""
    for unknown in ["1Y", "2D", "10m"]:
        with pytest.raises(ValueError, match="지원하지 않는 timeframe"):
            normalize_to_ccxt(unknown)


def test_normalize_to_aurora_unknown_tf_raises():
    """지원 안 하는 TF — ValueError."""
    for unknown in ["1Y", "2D", "10m"]:
        with pytest.raises(ValueError, match="지원하지 않는 timeframe"):
            normalize_to_aurora(unknown)


# ============================================================
# 입력 타입·공백 검증 (3)
# ============================================================


def test_normalize_empty_string_raises():
    """빈 문자열 — ValueError("빈 timeframe 입력")."""
    with pytest.raises(ValueError, match="빈 timeframe 입력"):
        normalize_to_ccxt("")
    with pytest.raises(ValueError, match="빈 timeframe 입력"):
        normalize_to_aurora("")


def test_normalize_whitespace_raises():
    """공백 포함 — ValueError (strip 은 호출자 책임)."""
    for whitespace in [" 1H ", "1 m", "\t1H", "1H\n"]:
        with pytest.raises(ValueError, match="지원하지 않는 timeframe"):
            normalize_to_ccxt(whitespace)
        with pytest.raises(ValueError, match="지원하지 않는 timeframe"):
            normalize_to_aurora(whitespace)


def test_normalize_wrong_type_raises():
    """str 이 아닌 타입 — TypeError."""
    for wrong in [None, 60, 1.5, ["1H"], ("1H",), {"1H"}]:
        with pytest.raises(TypeError, match="timeframe 은 str 이어야 함"):
            normalize_to_ccxt(wrong)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="timeframe 은 str 이어야 함"):
            normalize_to_aurora(wrong)  # type: ignore[arg-type]


# ============================================================
# is_valid_timeframe (1)
# ============================================================


def test_is_valid_timeframe_format_modes():
    """3 가지 format 모드 검증 + 잘못된 입력 처리."""
    # "either" (기본) — 양쪽 포맷 모두 허용
    assert is_valid_timeframe("1H") is True
    assert is_valid_timeframe("1h") is True
    assert is_valid_timeframe("1m") is True   # 분 단위 양쪽 동일

    # "aurora" 모드 — Aurora 포맷만 허용
    assert is_valid_timeframe("1H", format="aurora") is True
    assert is_valid_timeframe("1h", format="aurora") is False
    assert is_valid_timeframe("1m", format="aurora") is True

    # "ccxt" 모드 — ccxt 포맷만 허용
    assert is_valid_timeframe("1h", format="ccxt") is True
    assert is_valid_timeframe("1H", format="ccxt") is False
    assert is_valid_timeframe("1m", format="ccxt") is True

    # 잘못된 입력 — raise X, False 반환
    assert is_valid_timeframe("30m") is True  # 2026-06-13 지원 추가
    assert is_valid_timeframe("") is False
    assert is_valid_timeframe(" 1H ") is False
    assert is_valid_timeframe(None) is False  # type: ignore[arg-type]
    assert is_valid_timeframe(60) is False    # type: ignore[arg-type]


def test_30m_roundtrip() -> None:
    """2026-06-13: UI 매매 TF 30m 누락 사고 — 양방향 변환 보장."""
    from aurora.backtest.tf import normalize_to_aurora, normalize_to_ccxt
    assert normalize_to_ccxt("30m") == "30m"
    assert normalize_to_aurora("30m") == "30m"
