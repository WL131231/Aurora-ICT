"""#SAFETY-1 일일 손실 한도 단위 테스트.

- 한도 0 → 항상 비활성
- PnL 합산 → 한도 도달 시 hit
- NY local 자정 기준 reset
- daily_loss_status 응답 형식
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from zoneinfo import ZoneInfo

from aurora_ict.bot.bot_ict_instance import BotIctInstance

NY = ZoneInfo("America/New_York")


def _bot(limit_pct: float = 0.0) -> BotIctInstance:
    client = AsyncMock()
    return BotIctInstance(client=client, daily_loss_limit_pct=limit_pct)


def test_disabled_when_limit_zero() -> None:
    """limit_pct = 0 → 항상 False (한도 비활성)."""
    bot = _bot(limit_pct=0.0)
    bot._today_start_equity = 1000.0
    bot._today_realized_pnl_usdt = -500.0  # 50% loss
    assert bot._is_daily_loss_limit_hit() is False


def test_disabled_when_start_equity_zero() -> None:
    """start_equity 미정 → False (baseline 박을 때까지 비활성)."""
    bot = _bot(limit_pct=4.0)
    bot._today_start_equity = 0.0
    bot._today_realized_pnl_usdt = -100.0
    assert bot._is_daily_loss_limit_hit() is False


def test_hit_when_loss_exceeds_limit() -> None:
    """누적 손실 / equity * 100 ≥ 한도 → True."""
    bot = _bot(limit_pct=4.0)
    bot._today_start_equity = 1000.0
    bot._today_realized_pnl_usdt = -40.0  # 정확히 4%
    assert bot._is_daily_loss_limit_hit() is True
    bot._today_realized_pnl_usdt = -39.99  # 4% 미달
    assert bot._is_daily_loss_limit_hit() is False


def test_profit_never_triggers_hit() -> None:
    """수익 (positive PnL) 은 한도 도달 안 함."""
    bot = _bot(limit_pct=4.0)
    bot._today_start_equity = 1000.0
    bot._today_realized_pnl_usdt = 500.0  # +50%
    assert bot._is_daily_loss_limit_hit() is False


def test_reset_on_new_ny_day() -> None:
    """NY local 자정 (새 날짜) → reset."""
    bot = _bot(limit_pct=4.0)
    # 어제 상태
    yesterday = (
        datetime.now(UTC).astimezone(NY).replace(year=2025, month=1, day=1)
        .strftime("%Y-%m-%d")
    )
    bot._today_date_str = yesterday
    bot._today_realized_pnl_usdt = -100.0
    bot._daily_limit_hit = True
    # _maybe_reset_daily_pnl 호출 → 오늘 날짜로 reset
    bot._maybe_reset_daily_pnl(equity_now=2000.0)
    today = datetime.now(UTC).astimezone(NY).strftime("%Y-%m-%d")
    assert bot._today_date_str == today
    assert bot._today_realized_pnl_usdt == 0.0
    assert bot._today_start_equity == 2000.0
    assert bot._daily_limit_hit is False


def test_no_reset_on_same_day() -> None:
    """같은 NY 날짜 → reset 안 함 (상태 보존)."""
    bot = _bot(limit_pct=4.0)
    today = datetime.now(UTC).astimezone(NY).strftime("%Y-%m-%d")
    bot._today_date_str = today
    bot._today_realized_pnl_usdt = -50.0
    bot._today_start_equity = 1500.0
    bot._maybe_reset_daily_pnl(equity_now=2000.0)
    assert bot._today_realized_pnl_usdt == -50.0
    assert bot._today_start_equity == 1500.0  # 변하지 않음


def test_daily_loss_status_response() -> None:
    """daily_loss_status() — UI/API 응답 dict 형식 검증."""
    bot = _bot(limit_pct=4.0)
    bot._today_start_equity = 1000.0
    bot._today_realized_pnl_usdt = -30.0
    bot._today_date_str = "2026-05-21"
    s = bot.daily_loss_status()
    assert s["limit_pct"] == 4.0
    assert s["today_pnl_usdt"] == -30.0
    assert abs(s["today_pct"] - (-3.0)) < 0.01  # -3.0% (손실)
    assert s["start_equity"] == 1000.0
    assert s["hit"] is False
    assert s["date_ny"] == "2026-05-21"
    # 2026-06-10 조윤 건의: 수익 한도 필드 동봉.
    assert s["profit_limit_pct"] == 0.0
    assert s["profit_hit"] is False


# ============================================================
# 일일 수익(TP) 한도 — 2026-06-10 조윤 건의 (손실 한도와 대칭)
# ============================================================


def _bot_profit(profit_pct: float = 0.0) -> BotIctInstance:
    return BotIctInstance(client=AsyncMock(), daily_profit_limit_pct=profit_pct)


def test_profit_disabled_when_limit_zero() -> None:
    """profit_limit_pct = 0 → 항상 False."""
    bot = _bot_profit(0.0)
    bot._today_start_equity = 1000.0
    bot._today_realized_pnl_usdt = 500.0  # +50%
    assert bot._is_daily_profit_limit_hit() is False


def test_profit_hit_when_gain_exceeds_limit() -> None:
    """누적 수익 / equity * 100 ≥ 한도 → True."""
    bot = _bot_profit(8.0)
    bot._today_start_equity = 1000.0
    bot._today_realized_pnl_usdt = 80.0  # 정확히 8%
    assert bot._is_daily_profit_limit_hit() is True
    bot._today_realized_pnl_usdt = 79.99  # 미달
    assert bot._is_daily_profit_limit_hit() is False


def test_loss_never_triggers_profit_hit() -> None:
    """손실 (negative PnL) 은 수익 한도 도달 안 함."""
    bot = _bot_profit(8.0)
    bot._today_start_equity = 1000.0
    bot._today_realized_pnl_usdt = -500.0
    assert bot._is_daily_profit_limit_hit() is False


def test_profit_hit_reset_on_new_ny_day() -> None:
    """NY 자정 reset 시 profit_hit flag 도 풀림."""
    bot = _bot_profit(8.0)
    bot._today_date_str = "2025-01-01"
    bot._daily_profit_hit = True
    bot._maybe_reset_daily_pnl(equity_now=2000.0)
    assert bot._daily_profit_hit is False


# ============================================================
# 2026-06-11 리뷰 수정 — reset 보류(잔고 실패) + sticky 한도
# ============================================================


def test_reset_held_when_equity_none() -> None:
    """잔고 조회 실패(None) 시 새 날짜여도 reset 보류 — 폴백 baseline 오염 방지."""
    bot = _bot(limit_pct=4.0)
    bot._today_date_str = "2025-01-01"  # 과거 날짜 = 새 날짜 조건 성립
    bot._today_realized_pnl_usdt = -100.0
    bot._today_start_equity = 5000.0
    bot._daily_limit_hit = True
    bot._maybe_reset_daily_pnl(equity_now=None)
    # 아무것도 안 바뀜 — 다음 성공 fetch 때 reset.
    assert bot._today_date_str == "2025-01-01"
    assert bot._today_start_equity == 5000.0
    assert bot._daily_limit_hit is True
    # 성공 fetch 오면 정상 reset.
    bot._maybe_reset_daily_pnl(equity_now=3000.0)
    assert bot._today_start_equity == 3000.0
    assert bot._daily_limit_hit is False


def test_fetch_equity_or_none_returns_none_on_failure() -> None:
    """fetch_balance 예외/형식불명 → None (1000 폴백과 구분)."""
    import asyncio
    client = AsyncMock()
    client.fetch_balance.side_effect = RuntimeError("api down")
    bot = BotIctInstance(client=client)
    assert asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        bot._fetch_equity_or_none(),
    ) is None


def test_sticky_loss_limit_blocks_after_pnl_recovers() -> None:
    """sticky: 한도 hit flag 가 서면, pnl 이 한도 아래로 회복돼도 그날 차단 유지.

    _execute_setup 게이트가 `flag or _is_hit()` 패턴이므로 flag 만 검증.
    """
    bot = _bot(limit_pct=4.0)
    bot._today_start_equity = 1000.0
    bot._today_realized_pnl_usdt = -40.0  # 4% — hit
    assert bot._is_daily_loss_limit_hit() is True
    bot._daily_limit_hit = True  # 게이트가 set
    bot._today_realized_pnl_usdt = -10.0  # 회복 — 실시간 판정은 False
    assert bot._is_daily_loss_limit_hit() is False
    assert bot._daily_limit_hit is True  # 그래도 flag 유지 → 게이트 차단
