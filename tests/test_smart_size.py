"""#SMART-SIZE 2026-07-20 (FST#7): 품질 기반 사이즈 배수 단위 테스트.

볼륨·Nadaraya-Watson 중심선·RSI 3신호 방향정합 점수(0~3) → 배수 clip(0.7+q*0.2,0.4,1.4).
거래 필터 아닌 자금배분(빈도 불변). LuxAlgo 신호계열 대입 중 유일 walk-forward robust.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.indicators.fvg import FVG, FVGType
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup


def _bot() -> BotIctInstance:
    return BotIctInstance(client=AsyncMock(), symbol="BTCUSDT", smart_size_enabled=True)


def _setup(direction: Direction) -> SilverBulletSetup:
    fvg = FVG(type=FVGType.BULLISH, idx=5, ts_ms=1, low=98, high=102)
    return SilverBulletSetup(
        ts_ms=1, direction=direction, window="any",
        entry=100.0, stop_loss=95.0, take_profit=115.0, risk_reward=3.0, fvg=fvg)


def _df(closes: list[float], last_vol: float, base_vol: float = 100.0) -> pd.DataFrame:
    n = len(closes)
    vol = [base_vol] * n
    vol[-1] = last_vol
    return pd.DataFrame({
        "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes, "volume": vol,
    })


def test_all_signals_confirm_long_high_scale() -> None:
    """상승추세 + 고볼륨 + LONG → 3신호 정합(q=3) → 배수 1.3."""
    closes = list(np.linspace(90, 110, 60))  # 뚜렷한 상승 → NW중심 아래·RSI>50
    df = _df(closes, last_vol=300.0)  # 진입봉 볼륨 >> 평균
    s = _setup(Direction.LONG)
    _bot()._set_smart_size(s, df)
    assert s.smart_size_scale == pytest.approx(1.3, abs=1e-6)


def test_no_signals_long_in_downtrend_low_scale() -> None:
    """하락추세 + 저볼륨 + LONG → 정합 0(q=0) → 배수 0.7."""
    closes = list(np.linspace(110, 90, 60))  # 하락 → LONG 은 NW·RSI 역
    df = _df(closes, last_vol=10.0)  # 저볼륨
    s = _setup(Direction.LONG)
    _bot()._set_smart_size(s, df)
    assert s.smart_size_scale == pytest.approx(0.7, abs=1e-6)


def test_short_in_downtrend_high_scale() -> None:
    """하락추세 + 고볼륨 + SHORT → 3신호 정합 → 배수 1.3 (방향 반대 검증)."""
    closes = list(np.linspace(110, 90, 60))
    df = _df(closes, last_vol=300.0)
    s = _setup(Direction.SHORT)
    _bot()._set_smart_size(s, df)
    assert s.smart_size_scale == pytest.approx(1.3, abs=1e-6)


def test_disabled_keeps_neutral() -> None:
    """smart_size_enabled=False 면 1.0 중립 유지(하위호환)."""
    bot = BotIctInstance(client=AsyncMock(), symbol="BTCUSDT", smart_size_enabled=False)
    closes = list(np.linspace(90, 110, 60))
    s = _setup(Direction.LONG)
    bot._set_smart_size(s, _df(closes, last_vol=300.0))
    assert s.smart_size_scale == 1.0


def test_insufficient_data_neutral() -> None:
    """데이터 <50봉이면 1.0 유지."""
    s = _setup(Direction.LONG)
    _bot()._set_smart_size(s, _df([100.0] * 30, last_vol=300.0))
    assert s.smart_size_scale == 1.0


def test_scale_applied_to_qty() -> None:
    """배수가 실제 qty 에 반영 — scale 1.3 이면 base 대비 qty 1.3배."""
    bot = BotIctInstance(client=AsyncMock(), symbol="BTCUSDT", risk_based_sizing=True)
    s = _setup(Direction.LONG)
    s.confluence_score = 5
    s.smart_size_scale = 1.0
    base = bot._calc_qty_risk_based(s, equity=1000.0)
    s.smart_size_scale = 1.3
    boosted = bot._calc_qty_risk_based(s, equity=1000.0)
    assert boosted == pytest.approx(base * 1.3, rel=1e-6)
