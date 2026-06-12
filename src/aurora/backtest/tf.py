"""Timeframe 표기 변환·검증 — Aurora 포맷 ↔ ccxt 포맷.

Aurora 내부 표기 (대문자 H/D/W) 와 ccxt 라이브러리 표기 (소문자 h/d/w) 사이
양방향 변환 + 검증. PR-1 ``MultiTfAggregator.TF_MINUTES`` 와 매핑 일치 (회귀
보호 — 새 TF 추가 시 양쪽 동기화 필수).

상세 spec: ``src/aurora/backtest/DESIGN.md`` §7.

담당: ChoYoon
"""

from __future__ import annotations

from typing import Literal

# ============================================================
# 매핑 테이블 — Aurora ↔ ccxt
# ============================================================

# PR-1 MultiTfAggregator.TF_MINUTES 와 동일 키 (9 개) — 회귀 보호 필수.
_AURORA_TO_CCXT: dict[str, str] = {
    "1m":  "1m",   # 분 단위는 양쪽 동일 (자연 idempotent)
    "3m":  "3m",
    "5m":  "5m",
    "15m": "15m",
    "30m": "30m",  # 2026-06-13: UI 매매 TF 에 있는데 누락 — step 무한 실패 사고
    "1H":  "1h",   # 시간/일/주는 대소문자 변환
    "2H":  "2h",
    "4H":  "4h",
    "1D":  "1d",
    "1W":  "1w",
}
_CCXT_TO_AURORA: dict[str, str] = {v: k for k, v in _AURORA_TO_CCXT.items()}


# ============================================================
# 검증용 frozenset — 빠른 lookup + immutable
# ============================================================

TF_VALID_AURORA: frozenset[str] = frozenset(_AURORA_TO_CCXT.keys())
"""Aurora 포맷 유효 timeframe 10 개 — 외부 모듈 검증용 public 노출."""

TF_VALID_CCXT: frozenset[str] = frozenset(_CCXT_TO_AURORA.keys())
"""ccxt 포맷 유효 timeframe 9 개 — 외부 모듈 검증용 public 노출."""


# ============================================================
# 공개 함수 — 변환 (Strict)
# ============================================================


def _validate_input(tf: str) -> None:
    """공통 입력 검증 — type guard + 빈 문자열.

    ``normalize_to_ccxt`` / ``normalize_to_aurora`` 양쪽에서 호출.

    Raises:
        TypeError: ``tf`` 가 str 이 아닐 때.
        ValueError: 빈 문자열일 때.
    """
    if not isinstance(tf, str):
        raise TypeError(
            f"timeframe 은 str 이어야 함 (받은 타입: {type(tf).__name__})"
        )
    if tf == "":
        raise ValueError("빈 timeframe 입력")


def normalize_to_ccxt(tf: str) -> str:
    """Aurora 포맷 → ccxt 포맷.

    예: ``"1H"`` → ``"1h"``, ``"1m"`` → ``"1m"``, ``"1W"`` → ``"1w"``.

    분 단위 (``"1m"`` / ``"3m"`` / ``"5m"`` / ``"15m"``) 는 양쪽 동일 (자연 idempotent).

    Args:
        tf: Aurora 포맷 timeframe.

    Returns:
        ccxt 포맷 timeframe.

    Raises:
        TypeError: ``tf`` 가 str 이 아닐 때.
        ValueError: 빈 문자열 / 공백 포함 / Aurora 포맷 아닌 입력 / 지원 안 하는 TF.
    """
    _validate_input(tf)
    if tf not in _AURORA_TO_CCXT:
        raise ValueError(
            f"지원하지 않는 timeframe: {tf!r}. "
            f"Aurora 포맷만 허용: {list(_AURORA_TO_CCXT)}"
        )
    return _AURORA_TO_CCXT[tf]


def normalize_to_aurora(tf: str) -> str:
    """ccxt 포맷 → Aurora 포맷.

    예: ``"1h"`` → ``"1H"``, ``"1m"`` → ``"1m"``, ``"1w"`` → ``"1W"``.

    분 단위 (``"1m"`` / ``"3m"`` / ``"5m"`` / ``"15m"``) 는 양쪽 동일 (자연 idempotent).

    Args:
        tf: ccxt 포맷 timeframe.

    Returns:
        Aurora 포맷 timeframe.

    Raises:
        TypeError: ``tf`` 가 str 이 아닐 때.
        ValueError: 빈 문자열 / 공백 포함 / ccxt 포맷 아닌 입력 / 지원 안 하는 TF.
    """
    _validate_input(tf)
    if tf not in _CCXT_TO_AURORA:
        raise ValueError(
            f"지원하지 않는 timeframe: {tf!r}. "
            f"ccxt 포맷만 허용: {list(_CCXT_TO_AURORA)}"
        )
    return _CCXT_TO_AURORA[tf]


# ============================================================
# 공개 함수 — 검증 (raise X, bool 반환)
# ============================================================


def is_valid_timeframe(
    tf: str,
    format: Literal["aurora", "ccxt", "either"] = "either",
) -> bool:
    """포맷 검증 — raise 안 함, bool 반환.

    잘못된 타입 / 빈 / 공백 / unknown TF → ``False`` (raise X).

    Args:
        tf: 검증할 timeframe 문자열.
        format: 검증 모드.

            - ``"aurora"``: Aurora 포맷만 허용 (대문자 H/D/W).
            - ``"ccxt"``: ccxt 포맷만 허용 (소문자 h/d/w).
            - ``"either"``: 양쪽 다 허용 (기본).

    Returns:
        ``True`` 면 ``format`` 에 맞는 유효 timeframe, ``False`` 면 그 외.
    """
    if not isinstance(tf, str):
        return False
    if format == "aurora":
        return tf in TF_VALID_AURORA
    if format == "ccxt":
        return tf in TF_VALID_CCXT
    # "either"
    return tf in TF_VALID_AURORA or tf in TF_VALID_CCXT
