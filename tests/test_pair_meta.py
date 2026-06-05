"""페어 확장 P0 — 심볼별 거래소 메타(min_qty/lot step/max leverage) 처리 (mock 0).

- AuroraClientAdapter 의 fetch_symbol_meta / round_amount 위임 + 안전 폴백
- BotIctInstance._calc_qty 가 심볼 메타의 min_qty 를 반영 (없으면 BTC 0.001 폴백)

담당: 지영민 (페어 확장 PR — precision/min_qty)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from aurora_ict.bot.aurora_adapter import AuroraClientAdapter
from aurora_ict.bot.bot_ict_instance import BotIctInstance

# ============================================================
# 더미 client — fetch_symbol_meta / round_amount 지원
# ============================================================


class _MetaClient:
    def __init__(self, meta, rounder=None) -> None:
        self._meta = meta
        self._rounder = rounder

    async def fetch_symbol_meta(self, symbol: str):
        return self._meta

    def round_amount(self, symbol: str, amount: float) -> float:
        return self._rounder(amount) if self._rounder else amount


# ============================================================
# 1. Adapter 위임 + 폴백
# ============================================================


@pytest.mark.asyncio
async def test_adapter_fetch_symbol_meta_delegates() -> None:
    meta = {"min_qty": 0.01, "qty_step": 0.01, "max_leverage": 25.0}
    adapter = AuroraClientAdapter(_MetaClient(meta))
    assert await adapter.fetch_symbol_meta("ETH/USDT:USDT") == meta


@pytest.mark.asyncio
async def test_adapter_fetch_symbol_meta_fallback_on_error() -> None:
    class _Boom:
        async def fetch_symbol_meta(self, symbol: str):
            raise RuntimeError("net down")

    adapter = AuroraClientAdapter(_Boom())
    r = await adapter.fetch_symbol_meta("X/USDT:USDT")
    assert r == {"min_qty": None, "qty_step": None, "max_leverage": None}


@pytest.mark.asyncio
async def test_adapter_fetch_perp_tickers_delegates() -> None:
    rows = [{"symbol": "BTC/USDT:USDT", "last": 60000.0,
             "pct24h": -5.0, "volume": 8.0e9}]

    class _TickerClient:
        async def fetch_perp_tickers(self, limit: int = 30):
            return rows[:limit]

    adapter = AuroraClientAdapter(_TickerClient())
    assert await adapter.fetch_perp_tickers(30) == rows


@pytest.mark.asyncio
async def test_adapter_fetch_perp_tickers_fallback_on_error() -> None:
    class _Boom:
        async def fetch_perp_tickers(self, limit: int = 30):
            raise RuntimeError("net down")

    adapter = AuroraClientAdapter(_Boom())
    assert await adapter.fetch_perp_tickers() == []


def test_adapter_round_amount_delegates() -> None:
    adapter = AuroraClientAdapter(_MetaClient({}, rounder=lambda a: round(a, 2)))
    assert adapter.round_amount("ETH/USDT:USDT", 1.23456) == 1.23


def test_adapter_round_amount_fallback_when_unsupported() -> None:
    class _NoRound:
        pass

    adapter = AuroraClientAdapter(_NoRound())
    # round_amount 미지원 client → 원본 그대로(주문 경로 안 막음).
    assert adapter.round_amount("X/USDT:USDT", 1.2345) == 1.2345


# ============================================================
# 2. _calc_qty 가 심볼별 min_qty 반영
# ============================================================


def _bot(**kw) -> BotIctInstance:
    client = MagicMock()
    # round_amount 가 MagicMock 이면 _calc_qty 가 타입 가드로 무시 → 원본 qty 사용.
    return BotIctInstance(
        client=client, leverage=10,
        position_pct_base=40.0, position_pct_step=15.0, position_pct_max=80.0,
        **kw,
    )


def _setup(score: int = 0, entry: float = 100.0):
    s = MagicMock()
    s.confluence_score = score
    s.entry = entry
    return s


def test_calc_qty_blocks_below_symbol_min_qty() -> None:
    """심볼 min_qty 가 크면(알트) qty 미달 시 0 반환."""
    bot = _bot()
    bot._symbol_meta = {"min_qty": 1000.0}  # 비현실적으로 큰 알트 min
    # equity 100, pct 40%, lev 10 → notional 400 / entry 100 = qty 4 < 1000 → 0
    assert bot._calc_qty(_setup(), equity=100.0) == 0.0


def test_calc_qty_passes_above_symbol_min_qty() -> None:
    """심볼 min_qty 가 작으면 정상 qty 반환."""
    bot = _bot()
    bot._symbol_meta = {"min_qty": 0.01}
    qty = bot._calc_qty(_setup(), equity=100.0)
    assert qty == pytest.approx(4.0)  # 400 notional / 100 entry


def test_calc_qty_falls_back_to_btc_min_when_no_meta() -> None:
    """메타 없으면 BTC 기준 0.001 폴백 — 기존 동작 보존."""
    bot = _bot()
    bot._symbol_meta = {}
    # 아주 작은 equity → qty < 0.001 → 0
    assert bot._calc_qty(_setup(entry=100000.0), equity=1.0) == 0.0
    # 충분한 equity → 통과
    assert bot._calc_qty(_setup(entry=100.0), equity=100.0) > 0.0
