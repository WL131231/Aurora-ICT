"""#SHADOW 게이트 판정 기록 단위 테스트 — JSONL 기록·중복 방지·비활성.

FSD-style 데이터 플라이휠: 거른 setup 도 특징과 함께 기록 (행동 영향 0).
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.indicators.fvg import FVG, FVGType
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup


def _setup(ts: int = 111, d: Direction = Direction.LONG) -> SilverBulletSetup:
    fvg = FVG(type=FVGType.BULLISH, idx=5, ts_ms=ts, low=98, high=102)
    return SilverBulletSetup(
        ts_ms=ts, direction=d, window="am_sb",
        entry=100.0, stop_loss=95.0, take_profit=115.0, risk_reward=3.0, fvg=fvg,
    )


def test_shadow_writes_jsonl_with_features(tmp_path) -> None:
    bot = BotIctInstance(client=AsyncMock(), trades_data_dir=str(tmp_path))
    bot._last_align_score = 4
    bot._record_shadow(_setup(), "grade_skip")
    path = tmp_path / "shadow_setups.jsonl"
    assert path.exists()
    rec = json.loads(path.read_text(encoding="utf-8").strip())
    assert rec["verdict"] == "grade_skip"
    assert rec["score"] == 0
    assert rec["rr"] == 3.0
    assert rec["align_score"] == 4
    assert rec["sl_dist_pct"] == 5.0  # |100-95|/100
    assert rec["direction"] == "long"


def test_shadow_dedupes_same_setup_verdict(tmp_path) -> None:
    """같은 (ts, 방향, 판정)은 1회만 — step 반복 노이즈 방지."""
    bot = BotIctInstance(client=AsyncMock(), trades_data_dir=str(tmp_path))
    for _ in range(5):
        bot._record_shadow(_setup(), "grade_skip")
    bot._record_shadow(_setup(), "taken")  # 판정 다르면 별도 기록
    lines = (tmp_path / "shadow_setups.jsonl").read_text(
        encoding="utf-8",
    ).strip().splitlines()
    assert len(lines) == 2


def test_shadow_disabled_writes_nothing(tmp_path) -> None:
    bot = BotIctInstance(
        client=AsyncMock(), trades_data_dir=str(tmp_path),
        shadow_log_enabled=False,
    )
    bot._record_shadow(_setup(), "grade_skip")
    assert not (tmp_path / "shadow_setups.jsonl").exists()
