"""#RISK-LAYER (Origo 1.8) — DD 스로틀 + 일일 서킷브레이커 설정 검증.

포트폴리오 복리 시뮬(2026-07-10): 일일스탑15% + DD스로틀 25%/x0.7 조합이
기준 대비 수익↑(23.9→29.2x)·MDD↓(90→80%)·최악일↓(-46→-30%).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from aurora_ict.bot.bot_ict_instance import BotIctInstance


def _bot(tmp_path: Path, pct: float = 25.0) -> BotIctInstance:
    return BotIctInstance(
        client=AsyncMock(), trades_data_dir=tmp_path,
        dd_throttle_pct=pct, dd_throttle_factor=0.7,
    )


def test_throttle_off_below_threshold(tmp_path: Path) -> None:
    """낙폭 임계 미만 → 스케일 1.0 + peak 파일 영속."""
    bot = _bot(tmp_path)
    assert bot._dd_throttle_scale(100.0) == 1.0     # 첫 호출 — peak=100 기록
    assert bot._dd_throttle_scale(90.0) == 1.0      # 낙폭 10% < 25%
    assert (tmp_path / "peak_equity.json").exists()


def test_throttle_fires_beyond_threshold(tmp_path: Path) -> None:
    """낙폭 25% 초과 → factor(0.7). 회복해 신고점이면 다시 1.0."""
    bot = _bot(tmp_path)
    bot._dd_throttle_scale(100.0)
    assert bot._dd_throttle_scale(70.0) == pytest.approx(0.7)   # 낙폭 30%
    assert bot._dd_throttle_scale(120.0) == 1.0                 # 신고점 갱신


def test_peak_survives_restart(tmp_path: Path) -> None:
    """peak 이 파일로 영속 — 새 인스턴스(재시작)도 이전 고점 기준 낙폭 인식."""
    _bot(tmp_path)._dd_throttle_scale(200.0)        # peak 200 기록
    bot2 = _bot(tmp_path)                           # 재시작 시뮬
    assert bot2._dd_throttle_scale(140.0) == pytest.approx(0.7)  # 낙폭 30%


def test_throttle_off_and_risk_scaling(tmp_path: Path) -> None:
    """pct=0(referral) off. 리스크 기반 qty 에 factor 반영 확인."""
    bot_off = _bot(tmp_path, pct=0.0)
    assert bot_off._dd_throttle_scale(50.0) == 1.0

    from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup
    bot = _bot(tmp_path)
    bot._dd_throttle_scale(100.0)                   # peak 100
    setup = SilverBulletSetup(
        ts_ms=0, direction=Direction.LONG, window="any", entry=100.0,
        stop_loss=98.0, take_profit=110.0, risk_reward=5.0, confluence_score=5,
    )
    q_full = bot._calc_qty_risk_based(setup, 100.0)   # 낙폭 0
    q_throt = bot._calc_qty_risk_based(setup, 70.0)   # 낙폭 30% → x0.7
    # equity 도 줄었으니 (70/100)*(0.7) 배 — 스로틀 성분만 분리 검증
    assert q_throt == pytest.approx(q_full * 0.7 * 0.7, rel=1e-6)


def test_subscription_forces_risk_layer(monkeypatch):
    """구독제 = 일일한도 15% + DD 스로틀 25%/x0.7 강제. 더 타이트한 사용자값 유지."""
    import os

    from aurora_ict.config.settings import IctSettings

    for k in list(os.environ):
        if k.startswith("AURORA_ICT_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AURORA_ICT_LICENSE_TYPE", "sub_90d")
    s = IctSettings(_env_file=None)
    assert s.daily_loss_limit_pct == 15
    assert s.origo_dd_throttle_pct == pytest.approx(25.0)

    monkeypatch.setenv("AURORA_ICT_DAILY_LOSS_LIMIT_PCT", "10")
    s2 = IctSettings(_env_file=None)
    assert s2.daily_loss_limit_pct == 10  # 더 보수적(타이트) 사용자값 유지
