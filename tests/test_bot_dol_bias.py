"""DOL(Draw on Liquidity) 편향 필터 (#3 보완) 단위 테스트.

가격이 끌려갈 지배적 유동성(DOL)과 반대 방향 setup 은 confluence_score 감점
→ B+ 게이트에서 걸러짐 (오르는 장에 계속 숏 치던 문제 완화).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

import pandas as pd

from aurora_ict.bot.bot_ict_instance import BotIctInstance
from aurora_ict.strategy.silver_bullet import Direction, SilverBulletSetup

NY = ZoneInfo("America/New_York")


def _df(bars: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    rows = [{"open": o, "high": h, "low": lo, "close": c} for o, h, lo, c in bars]
    df = pd.DataFrame(rows)
    start = datetime(2026, 5, 12, 10, 0, tzinfo=NY)
    df.index = pd.DatetimeIndex([start + timedelta(minutes=i) for i in range(len(rows))])
    return df


def _setup(direction: Direction, score: int) -> SilverBulletSetup:
    return SilverBulletSetup(
        ts_ms=1,
        direction=direction,
        window="any",
        entry=100.0,
        stop_loss=99.0,
        take_profit=102.0,
        risk_reward=2.0,
        fvg=None,  # type: ignore[arg-type]
        confluence_score=score,
        confluences=[],
    )


def _bot() -> BotIctInstance:
    return BotIctInstance(client=AsyncMock())


def test_dol_penalizes_only_counter_draw_direction() -> None:
    """같은 df 에서 long/short 중 draw 와 반대 한 방향만 -2 감점 (다른 방향 그대로)."""
    # 위쪽 unswept swing high(~107) + 아래쪽 unswept swing low(~92), 마지막 close 100.
    df = _df([
        (100, 101, 99, 100),
        (100, 107, 100, 106),   # swing high 107 (현재가 위, unswept)
        (106, 106, 99, 100),
        (100, 100, 92, 93),     # swing low 92 (현재가 아래, unswept)
        (93, 98, 92, 97),
        (97, 99, 96, 98),
        (98, 100, 97, 100),     # 마지막 close 100
    ])
    bot = _bot()
    long_s = _setup(Direction.LONG, 5)
    short_s = _setup(Direction.SHORT, 5)
    bot._apply_dol_bias(long_s, df)
    bot._apply_dol_bias(short_s, df)
    # 정확히 한 방향만 5→3 감점 (draw 와 반대), 나머진 5 유지.
    assert {long_s.confluence_score, short_s.confluence_score} == {5, 3}


def test_dol_no_change_when_df_too_short() -> None:
    """봉 부족(<5) 이면 감점 없음 (guard)."""
    df = _df([(100, 101, 99, 100)] * 3)
    bot = _bot()
    s = _setup(Direction.SHORT, 5)
    bot._apply_dol_bias(s, df)
    assert s.confluence_score == 5
