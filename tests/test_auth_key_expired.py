"""#KEY-EXPIRED 2026-09-08 — 거래소 API 키 만료(33004)/무효(10003) 가시화.

9/8 실측: Bybit 가 IP 제한 없는 키를 90일 뒤 만료시켜 4계좌가 동시에 33004 를
받았는데, 봇은 (1) 포지션 조회 실패를 '포지션 없음'으로 읽고, (2) 잔고 전용
카운터가 0 에 머물러 자동 정지가 안 됐으며, (3) UI 는 폴백 1000.00 USDT 를
'실시간'으로 보여줬다. 이 파일은 그 세 구멍을 각각 잠근다.

mock 0 정책 — 결정론적 AsyncMock 만 쓴다(외부 호출 없음).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock

import pytest
from ccxt.base.errors import AuthenticationError

from aurora_ict.bot.aurora_adapter import AuroraClientAdapter
from aurora_ict.bot.bot_ict_instance import (
    _AUTH_FAIL_STOP_THRESHOLD as _ORIGO_THRESHOLD,
    BotIctInstance,
    BotState,
)
from aurora_ict.bot.bot_trend_instance import (
    _AUTH_FAIL_STOP_THRESHOLD as _CURSUS_THRESHOLD,
    BotTrendInstance,
)

_EXPIRED = AuthenticationError(
    'bybit {"retCode":33004,"retMsg":"Your api key has expired."}',
)
_INVALID = AuthenticationError(
    'bybit {"retCode":10003,"retMsg":"API key is invalid."}',
)


def _inner() -> AsyncMock:
    """Aurora client 모형 — _ex(ccxt) 까지 갖춘 최소 형태."""
    inner = AsyncMock()
    inner._ex = AsyncMock()
    inner._ex.fetch_balance = AsyncMock(return_value={"USDT": {"total": 1.0}})
    inner._ex.fetch_positions = AsyncMock(return_value=[])
    return inner


# ── 어댑터: 인증 실패를 세고, 삼키지 않는다 ───────────────────────────────

@pytest.mark.asyncio
async def test_fetch_position_auth_error_counts_and_raises() -> None:
    """포지션 조회 인증 실패는 None('없음')이 아니라 예외('모름')로 올라간다."""
    inner = _inner()
    inner.fetch_position = AsyncMock(side_effect=_EXPIRED)
    adapter = AuroraClientAdapter(inner)
    with pytest.raises(AuthenticationError):
        await adapter.fetch_position("BTC/USDT:USDT")
    assert adapter.auth_fail_streak == 1
    assert adapter.auth_error_kind == "expired"
    assert "expired" in (adapter.last_auth_error or "")


@pytest.mark.asyncio
async def test_fetch_ohlcv_auth_error_counts() -> None:
    """봉 조회도 키가 필요하다(10003 실측) — 같은 카운터로 센다."""
    inner = _inner()
    inner.fetch_ohlcv = AsyncMock(side_effect=_INVALID)
    adapter = AuroraClientAdapter(inner)
    with pytest.raises(AuthenticationError):
        await adapter.fetch_ohlcv("BTC/USDT:USDT", "1h", 10)
    assert adapter.auth_fail_streak == 1
    assert adapter.auth_error_kind == "invalid"


@pytest.mark.asyncio
async def test_auth_streak_resets_after_any_success() -> None:
    """키 재등록 뒤 어떤 호출이든 통하면 카운터·메시지가 지워진다."""
    inner = _inner()
    inner.fetch_position = AsyncMock(side_effect=_EXPIRED)
    adapter = AuroraClientAdapter(inner)
    for _ in range(3):
        with pytest.raises(AuthenticationError):
            await adapter.fetch_position("BTC/USDT:USDT")
    assert adapter.auth_fail_streak == 3
    await adapter.fetch_balance()          # 성공 → 리셋
    assert adapter.auth_fail_streak == 0
    assert adapter.auth_error_kind is None
    assert adapter.last_auth_error is None


@pytest.mark.asyncio
async def test_auth_fail_log_is_throttled(caplog: Any) -> None:
    """도배 방지 — 첫 1회 + 12회마다 한 번만 WARNING."""
    inner = _inner()
    inner.fetch_position = AsyncMock(side_effect=_EXPIRED)
    adapter = AuroraClientAdapter(inner)
    with caplog.at_level("WARNING"):
        for _ in range(24):
            with pytest.raises(AuthenticationError):
                await adapter.fetch_position("BTC/USDT:USDT")
    n = sum("인증 실패" in r.getMessage() for r in caplog.records)
    assert n == 3, f"24회 실패에 로그 {n}건 (기대 3: 1·12·24회차)"


# ── 봇 루프: step 이 무엇을 하든 매 바퀴 인증 상태를 본다 + 알린다 ─────────

def _client_with_expired_key(streak: int) -> AsyncMock:
    """어댑터가 이미 streak 만큼 실패를 누적한 상태를 흉내낸다."""
    c = AsyncMock()
    c.fetch_ohlcv = AsyncMock(return_value=[[1, 100, 101, 99, 100, 10]])
    c.fetch_balance = AsyncMock(return_value={})
    c.fetch_position = AsyncMock(return_value=None)
    c.auth_fail_streak = streak
    c.last_auth_error = 'bybit {"retCode":33004,"retMsg":"Your api key has expired."}'
    return c


@pytest.mark.asyncio
async def test_origo_stops_even_when_step_raises_other_error() -> None:
    """step 이 다른 예외를 내도(9/8: 포지션 조회 실패) 어댑터 카운터로 정지.

    예전엔 판정이 try 안에 있어 step 예외 시 건너뛰었다 — 카운터 13에도 RUNNING.
    """
    class _Bot(BotIctInstance):
        async def step(self):  # type: ignore[override]
            raise RuntimeError("fetch_position 실패 — 어댑터가 삼킨 인증 오류의 여파")

    notify = AsyncMock()
    bot = _Bot(
        client=_client_with_expired_key(_ORIGO_THRESHOLD), symbol="BTCUSDT",
        step_interval_sec=0, notify_cb=notify, user_code="AICT-TEST-0000-0000",
    )
    bot.state = BotState.RUNNING
    await asyncio.wait_for(bot._run_loop(), timeout=2.0)
    assert bot.state is BotState.STOPPED
    assert bot.stop_reason == "auth_expired"
    notify.assert_awaited_once()
    code, msg = notify.await_args.args
    assert code == "AICT-TEST-0000-0000"
    assert "만료" in msg and "재등록" in msg


@pytest.mark.asyncio
async def test_cursus_stops_on_adapter_auth_streak_without_balance_call() -> None:
    """Cursus 는 진입 후보 없으면 잔고를 안 본다 — 그래도 포지션 조회 실패로 정지."""
    class _Bot(BotTrendInstance):
        async def step(self):  # type: ignore[override]
            return None          # 조용히 지나가는 step (신호 없음)

    notify = AsyncMock()
    bot = _Bot(
        client=_client_with_expired_key(_CURSUS_THRESHOLD), symbol="BTC/USDT:USDT",
        step_interval_sec=0, notify_cb=notify, user_code="AICT-TEST-0000-0001",
    )
    bot.state = BotState.RUNNING
    await asyncio.wait_for(bot._run_loop(), timeout=2.0)
    assert bot.state is BotState.STOPPED
    assert bot.stop_reason == "auth_expired"
    notify.assert_awaited_once()


@pytest.mark.asyncio
async def test_cursus_auth_error_in_step_counts_not_traceback() -> None:
    """step 자체가 AuthenticationError 를 내면 카운트 후 임계치에서 정지."""
    class _Bot(BotTrendInstance):
        async def step(self):  # type: ignore[override]
            raise _INVALID

    c = _client_with_expired_key(0)
    c.last_auth_error = None
    bot = _Bot(client=c, symbol="BTC/USDT:USDT", step_interval_sec=0)
    bot.state = BotState.RUNNING
    await asyncio.wait_for(bot._run_loop(), timeout=2.0)
    assert bot.state is BotState.STOPPED
    assert bot._auth_fail_streak == _CURSUS_THRESHOLD
    assert bot.stop_reason == "auth_invalid"


@pytest.mark.asyncio
async def test_auth_stop_notify_failure_does_not_block_stop() -> None:
    """텔레그램 발송이 실패해도 정지는 된다."""
    class _Bot(BotIctInstance):
        async def step(self):  # type: ignore[override]
            return None

    notify = AsyncMock(side_effect=RuntimeError("telegram down"))
    bot = _Bot(
        client=_client_with_expired_key(_ORIGO_THRESHOLD), symbol="BTCUSDT",
        step_interval_sec=0, notify_cb=notify, user_code="AICT-TEST-0000-0002",
    )
    bot.state = BotState.RUNNING
    await asyncio.wait_for(bot._run_loop(), timeout=2.0)
    assert bot.state is BotState.STOPPED


# ── /ict/equity — 폴백 1000.00 을 사용자에게 내려주지 않는다 ──────────────

class _Cli:
    def __init__(self, kind: str | None, msg: str | None = None) -> None:
        self.auth_error_kind = kind
        self.last_auth_error = msg
        self.auth_fail_streak = 1 if kind else 0


class _Bot:
    def __init__(self, cli: _Cli, equity: float | None) -> None:
        self.client = cli
        self._eq = equity

    async def _fetch_equity_or_none(self) -> float | None:
        return self._eq

    async def _fetch_equity(self) -> float:
        return self._eq if self._eq is not None else 1000.0


@pytest.mark.asyncio
async def test_equity_payload_reports_expired_key_instead_of_fallback() -> None:
    from aurora_ict.api.app import _equity_payload

    bot = _Bot(_Cli("expired", "Your api key has expired."), None)
    r = await _equity_payload(bot, {"kind": "none", "label": "None"})
    assert r["equity"] is None
    assert r["error"] == "auth_expired"
    assert "expired" in r["message"]


@pytest.mark.asyncio
async def test_equity_payload_fetch_failed_is_none_not_1000() -> None:
    from aurora_ict.api.app import _equity_payload

    bot = _Bot(_Cli(None), None)
    r = await _equity_payload(bot, {"kind": "none", "label": "None"})
    assert r["equity"] is None and r["error"] == "fetch_failed"


@pytest.mark.asyncio
async def test_equity_payload_normal() -> None:
    from aurora_ict.api.app import _equity_payload

    bot = _Bot(_Cli(None), 123.45)
    r = await _equity_payload(bot, {"kind": "none", "label": "None"})
    assert r["equity"] == pytest.approx(123.45) and "error" not in r
