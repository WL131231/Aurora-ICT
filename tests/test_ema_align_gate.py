"""다중 EMA 정렬(align) 게이트 — 방향 결정 + 진입 게이트 단위 테스트.

#ALIGN (2026-06-10): 단일 EMA20 대신 인접 EMA 쌍 정배열/역배열 점수로 진입
방향을 결정. 상승추세→롱만, 하락추세→숏만, 불명확→진입 자제.

self-spy: _fetch_ohlcv_tf 를 subclass override 로 합성 추세 close 주입(mock 0).
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.strategy.silver_bullet import Direction

_N = 700  # periods 최대(620) + 여유보다 길게


def _make_bot(closes: list[float], threshold: int = 2) -> BotIctInstance:
    class _AlignBot(BotIctInstance):
        async def _fetch_ohlcv_tf(self, tf, limit):  # noqa: ANN001, ARG002
            return pd.DataFrame({"close": closes[-limit:]})

    return _AlignBot(
        client=AsyncMock(),
        htf_ema_bias_enabled=True,
        htf_ema_align_enabled=True,
        htf_ema_align_threshold=threshold,
    )


@pytest.mark.asyncio
async def test_align_uptrend_forces_long() -> None:
    """단조 상승 → 정배열(점수 +) → 롱만 허용, 숏 차단."""
    closes = [100.0 + i * 0.5 for i in range(_N)]
    bot = _make_bot(closes)
    assert await bot._compute_ema_align_score() > 0
    assert await bot._compute_htf_ema_direction() is Direction.LONG
    assert await bot._passes_ema_align_gate(Direction.LONG) is True
    assert await bot._passes_ema_align_gate(Direction.SHORT) is False


@pytest.mark.asyncio
async def test_align_downtrend_forces_short() -> None:
    """단조 하락 → 역배열(점수 -) → 숏만 허용, 롱 차단."""
    closes = [500.0 - i * 0.4 for i in range(_N)]
    bot = _make_bot(closes)
    assert await bot._compute_ema_align_score() < 0
    assert await bot._compute_htf_ema_direction() is Direction.SHORT
    assert await bot._passes_ema_align_gate(Direction.SHORT) is True
    assert await bot._passes_ema_align_gate(Direction.LONG) is False


@pytest.mark.asyncio
async def test_align_flat_undecided_blocks_both() -> None:
    """횡보(점수 0) → 추세 불명확 → 방향 None + 양방향 진입 자제."""
    closes = [100.0] * _N
    bot = _make_bot(closes)
    assert await bot._compute_ema_align_score() == 0
    assert await bot._compute_htf_ema_direction() is None  # 양방향 setup 허용
    assert await bot._passes_ema_align_gate(Direction.LONG) is False  # 게이트 자제
    assert await bot._passes_ema_align_gate(Direction.SHORT) is False


@pytest.mark.asyncio
async def test_align_threshold_respected() -> None:
    """threshold=5(완전 정배열만) — 단조 상승은 +5 라 통과."""
    closes = [100.0 + i * 0.5 for i in range(_N)]
    bot = _make_bot(closes, threshold=5)
    assert await bot._compute_ema_align_score() == 5
    assert await bot._passes_ema_align_gate(Direction.LONG) is True


@pytest.mark.asyncio
async def test_align_disabled_falls_back_to_single_ema() -> None:
    """align_enabled=False → 기존 단일 EMA20 경로 (상승→LONG)."""
    closes = [100.0 + i * 0.5 for i in range(_N)]

    class _SingleBot(BotIctInstance):
        async def _fetch_ohlcv_tf(self, tf, limit):  # noqa: ANN001, ARG002
            return pd.DataFrame({"close": closes[-limit:]})

    bot = _SingleBot(
        client=AsyncMock(),
        htf_ema_bias_enabled=True,
        htf_ema_align_enabled=False,
    )
    assert await bot._compute_htf_ema_direction() is Direction.LONG
