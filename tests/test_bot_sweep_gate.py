"""#SWEEP-GATE (Origo 1.5) — 일봉 스윕-반전 후 역방향 차단 검증.

파트너 "3번 판단"(유동성 사건 = bias 즉시 전환) 기계화. 7/1 BTC SSL 스윕-반전
후 EMA 지연으로 7/4 까지 숏 → -165 전멸 패턴 방어. 백테 K=2: +253·DD -25%.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.strategy.silver_bullet import Direction


def _daily_df(rows: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    """[(o,h,l,c), ...] → 일봉 df (마지막 행 = 오늘 미완성 봉)."""
    idx = pd.date_range("2026-06-20", periods=len(rows), freq="1D")
    return pd.DataFrame(
        {"open": [r[0] for r in rows], "high": [r[1] for r in rows],
         "low": [r[2] for r in rows], "close": [r[3] for r in rows]}, index=idx)


def _flat_days(n: int, px: float = 100.0):
    return [(px, px + 1.0, px - 1.0, px)] * n


def _patch_daily(monkeypatch, df: pd.DataFrame) -> None:
    async def _fake(self, tf: str, limit: int):
        return df

    monkeypatch.setattr(BotIctInstance, "_fetch_ohlcv_tf", _fake)


@pytest.mark.asyncio
async def test_ssl_sweep_blocks_short_allows_long(monkeypatch) -> None:
    """직전 마감일이 SSL 스윕-반전(저점 신저 + 상반부 마감) → SHORT 차단, LONG 허용."""
    # 10일 평탄(저점 99) 후 스윕일: low 97(신저) high 103, close 102(상반부).
    rows = _flat_days(11) + [(100.0, 103.0, 97.0, 102.0)] + [(102.0, 102.5, 101.5, 102.0)]
    _patch_daily(monkeypatch, _daily_df(rows))
    bot = BotIctInstance(client=AsyncMock(), sweep_gate_days=2)

    assert await bot._sweep_gate_blocked(Direction.SHORT) is True
    assert await bot._sweep_gate_blocked(Direction.LONG) is False


@pytest.mark.asyncio
async def test_bsl_sweep_blocks_long(monkeypatch) -> None:
    """고점 스윕-반락(고점 신고 + 하반부 마감) → LONG 차단."""
    rows = _flat_days(11) + [(100.0, 104.0, 99.0, 99.5)] + [(99.5, 100.0, 99.0, 99.5)]
    _patch_daily(monkeypatch, _daily_df(rows))
    bot = BotIctInstance(client=AsyncMock(), sweep_gate_days=2)

    assert await bot._sweep_gate_blocked(Direction.LONG) is True
    assert await bot._sweep_gate_blocked(Direction.SHORT) is False


@pytest.mark.asyncio
async def test_no_sweep_no_block_and_off_by_default(monkeypatch) -> None:
    """스윕 없음 → 차단 없음. sweep_gate_days=0(referral) → 항상 통과."""
    rows = _flat_days(13)
    _patch_daily(monkeypatch, _daily_df(rows))
    bot = BotIctInstance(client=AsyncMock(), sweep_gate_days=2)
    assert await bot._sweep_gate_blocked(Direction.SHORT) is False

    bot_off = BotIctInstance(client=AsyncMock(), sweep_gate_days=0)
    assert await bot_off._sweep_gate_blocked(Direction.SHORT) is False


@pytest.mark.asyncio
async def test_fetch_failure_means_no_block(monkeypatch) -> None:
    """1d fetch 실패 → 차단 없음 (기존 동작 보수 유지)."""
    async def _boom(self, tf: str, limit: int):
        raise RuntimeError("network")

    monkeypatch.setattr(BotIctInstance, "_fetch_ohlcv_tf", _boom)
    bot = BotIctInstance(client=AsyncMock(), sweep_gate_days=2)
    assert await bot._sweep_gate_blocked(Direction.SHORT) is False


def test_subscription_forces_sweep_gate(monkeypatch):
    """구독제 = 2일 강제, referral = 0(off)."""
    import os

    from aurora_ict.config.settings import IctSettings

    for k in list(os.environ):
        if k.startswith("AURORA_ICT_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_90d")
    assert IctSettings(_env_file=None).origo_sweep_gate_days == 2
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "referral")
    assert IctSettings(_env_file=None).origo_sweep_gate_days == 0
