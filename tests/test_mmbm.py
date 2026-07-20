"""#FST7 2026-07-20: MMBM/MMSM 셋업 감지 단위 테스트.

MMBM = HTF 추세정합 방향으로 discount/premium 에서 갓 형성된 CHoCH(반전) + FVG
지정가 진입. 검증(백테): discount·HTF정합 필수, maker 체결 시 5/6년 흑자.
mock 0 — 결정론적 합성(하락추세→급반전) 입력.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurora_ict.strategy.mmbm import detect_mmbm_setup
from aurora_ict.strategy.silver_bullet import Direction, SetupSource


def _reversal_df(n: int = 310) -> pd.DataFrame:
    """하락추세(스윙 형성→discount) 후 급반전 상승(bullish CHoCH + FVG) 합성."""
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    base = np.linspace(120, 96, n - 10)
    saw = 6 * np.sin(np.linspace(0, 10 * np.pi, n - 10))  # 굵은 스윙
    close = np.array(list(base + saw)
                     + [96, 97.5, 101, 103.5, 105, 106, 107, 108, 109, 110])
    o = close - 0.2
    h = close + 0.6
    lo = close - 0.6
    h[301] = 98.0   # bar301.high < bar303.low = bullish FVG 강제
    lo[303] = 99.5
    return pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": close, "volume": [100.0] * n},
        index=idx,
    )


def test_mmbm_long_fires_on_aligned_reversal() -> None:
    """discount + bullish CHoCH(신선) + HTF 상승정합 → MMBM long 셋업."""
    df = _reversal_df().head(308)
    s = detect_mmbm_setup(df, htf_bias_sign=1.0)
    assert s is not None
    assert s.source is SetupSource.MMBM
    assert s.direction is Direction.LONG
    assert s.window == "mmbm"
    assert s.stop_loss < s.entry < s.take_profit  # 롱 가격 순서
    assert s.risk_reward >= 2.0 - 1e-9            # 최소 2R


def test_htf_bias_against_blocks() -> None:
    """HTF 하락정합인데 bullish 반전이면 진입 자제(None) — HTF 필수조건."""
    df = _reversal_df().head(308)
    assert detect_mmbm_setup(df, htf_bias_sign=-1.0) is None


def test_stale_choch_not_fresh_none() -> None:
    """CHoCH 형성 후 시간 지나(_FRESH 초과) 신선하지 않으면 None."""
    df = _reversal_df().head(310)  # CHoCH 가 마지막서 3봉+ 전 → 신선도 실패
    assert detect_mmbm_setup(df, htf_bias_sign=1.0) is None


def test_insufficient_data_none() -> None:
    """딜링레인지 lookback(288) 미달이면 None."""
    df = _reversal_df().head(50)
    assert detect_mmbm_setup(df, htf_bias_sign=1.0) is None
