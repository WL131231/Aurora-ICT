"""#REGIME-OTE (Origo 1.7) — 상승 국면 적응 OTE 검증.

국면 랩(2026-07-10): 상승 국면(일봉 20일 z>0.75)의 얕은 되돌림 롱은 역선택
(-96%), OTE 0.786 심화만 전/후반 동시 개선(+94%) — 조건부 적용 시 합계 +6.7%.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance


def _daily(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-06-01", periods=len(closes), freq="1D")
    return pd.DataFrame({"open": closes, "high": closes, "low": closes,
                         "close": closes}, index=idx)


def _patch_daily(monkeypatch, df: pd.DataFrame) -> None:
    async def _fake(self, tf: str, limit: int):
        return df

    monkeypatch.setattr(BotIctInstance, "_fetch_ohlcv_tf", _fake)


@pytest.mark.asyncio
async def test_up_regime_uses_deep_ote(monkeypatch) -> None:
    """강한 상승(20일 +40%, 저변동) → z>0.75 → OTE 0.786."""
    closes = list(100 * (1.017 ** np.arange(30)))  # 일 +1.7% 꾸준 — z 큼
    _patch_daily(monkeypatch, _daily(closes))
    bot = BotIctInstance(client=AsyncMock(), ote_level=0.707, ote_up_level=0.786)

    assert await bot._regime_is_up() is True
    assert await bot._effective_ote() == pytest.approx(0.786)


@pytest.mark.asyncio
async def test_flat_regime_keeps_base_ote(monkeypatch) -> None:
    """횡보(무추세) → z~0 → 기본 0.707."""
    rng = np.random.default_rng(7)
    closes = list(100 + np.cumsum(rng.normal(0, 0.5, 30)))
    _patch_daily(monkeypatch, _daily(closes))
    bot = BotIctInstance(client=AsyncMock(), ote_level=0.707, ote_up_level=0.786)

    assert await bot._regime_is_up() is False
    assert await bot._effective_ote() == pytest.approx(0.707)


@pytest.mark.asyncio
async def test_off_and_fetch_failure_keep_base(monkeypatch) -> None:
    """ote_up_level=0(referral) 또는 1d fetch 실패 → 기본 유지 (보수)."""
    closes = list(100 * (1.02 ** np.arange(30)))
    _patch_daily(monkeypatch, _daily(closes))
    bot_off = BotIctInstance(client=AsyncMock(), ote_level=0.707, ote_up_level=0.0)
    assert await bot_off._effective_ote() == pytest.approx(0.707)

    async def _boom(self, tf: str, limit: int):
        raise RuntimeError("network")

    monkeypatch.setattr(BotIctInstance, "_fetch_ohlcv_tf", _boom)
    bot = BotIctInstance(client=AsyncMock(), ote_level=0.707, ote_up_level=0.786)
    assert await bot._effective_ote() == pytest.approx(0.707)


def test_subscription_forces_up_ote(monkeypatch):
    """구독제 = 0.786 강제, referral = 0(off)."""
    import os

    from aurora_ict.config.settings import IctSettings

    for k in list(os.environ):
        if k.startswith("AURORA_ICT_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_90d")
    assert IctSettings(_env_file=None).origo_ote_up_level == pytest.approx(0.786)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "referral")
    assert IctSettings(_env_file=None).origo_ote_up_level == 0.0
