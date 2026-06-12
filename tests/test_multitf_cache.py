"""MultiTfCache 단위 테스트 — mock client + 시간 시뮬레이션.

DESIGN.md §5 spec 검증:
    - warmup: 각 TF lookback fetch
    - step: 봉 경계에서만 fetch
    - get: 캐시 read-only
    - _has_new_bar: 봉 경계 정확 검출
    - 중복 index 제거 (회귀 안전)

영역: ChoYoon (어댑터 PR 위임 받음 2026-05-03)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from aurora.exchange.data import MultiTfCache

# ============================================================
# 헬퍼 — mock client + DataFrame 생성
# ============================================================


def _make_df(start_ts_ms: int, count: int, tf_minutes: int) -> pd.DataFrame:
    """결정론적 합성 OHLCV — 시각만 정확. 가격은 100 고정."""
    timestamps = [
        pd.Timestamp(start_ts_ms + i * tf_minutes * 60_000, unit="ms", tz="UTC")
        for i in range(count)
    ]
    df = pd.DataFrame(
        {
            "open": [100.0] * count,
            "high": [101.0] * count,
            "low": [99.0] * count,
            "close": [100.5] * count,
            "volume": [10.0] * count,
        },
        index=pd.DatetimeIndex(timestamps),
    )
    return df


def _make_mock_client(fetch_responses: dict[str, pd.DataFrame] | None = None) -> Any:
    """MultiTfCache 가 사용할 ExchangeClient mock — fetch_ohlcv 만 구현."""
    client = MagicMock()
    fetch_responses = fetch_responses or {}

    async def _fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        return fetch_responses.get(timeframe, pd.DataFrame()).copy()

    client.fetch_ohlcv = AsyncMock(side_effect=_fetch_ohlcv)
    return client


# ============================================================
# 생성 / 검증
# ============================================================


def test_init_accepts_valid_timeframes():
    """TF_MINUTES 에 정의된 TF — 생성 OK."""
    client = MagicMock()
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["15m", "1H", "4H", "1D"])
    assert cache._symbol == "BTC/USDT:USDT"
    assert cache._tfs == ["15m", "1H", "4H", "1D"]


def test_init_rejects_unknown_timeframe():
    """TF_MINUTES 미정의 TF → ValueError 즉시 raise."""
    client = MagicMock()
    with pytest.raises(ValueError, match="unknown timeframe"):
        MultiTfCache(client, "BTC/USDT:USDT", ["1H", "10m"])  # 10m 미정의 (30m 은 2026-06-13 지원)


# ============================================================
# warmup
# ============================================================


@pytest.mark.asyncio
async def test_warmup_fetches_all_timeframes():
    """모든 TF 병렬 fetch — 각 TF 의 fetch_ohlcv 1회 호출."""
    df_1h = _make_df(start_ts_ms=1_700_000_000_000, count=100, tf_minutes=60)
    df_4h = _make_df(start_ts_ms=1_700_000_000_000, count=50, tf_minutes=240)
    client = _make_mock_client({"1H": df_1h, "4H": df_4h})
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["1H", "4H"])

    await cache.warmup({"1H": 100, "4H": 50})

    assert len(cache.get("1H")) == 100
    assert len(cache.get("4H")) == 50
    # 각 TF 1회 fetch
    assert client.fetch_ohlcv.call_count == 2


@pytest.mark.asyncio
async def test_warmup_default_500_bars_when_lookback_omitted():
    """lookback_per_tf 미명시 TF — limit=500 default."""
    client = _make_mock_client({"1H": pd.DataFrame()})
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["1H"])

    await cache.warmup()  # None
    call = client.fetch_ohlcv.call_args
    assert call.kwargs.get("limit") == 500 or call.args[2] == 500


@pytest.mark.asyncio
async def test_warmup_partial_lookback_fills_default():
    """lookback_per_tf 일부만 명시 — 나머지 default 500."""
    client = _make_mock_client({"1H": pd.DataFrame(), "4H": pd.DataFrame()})
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["1H", "4H"])

    await cache.warmup({"1H": 200})  # 4H 미명시

    calls = client.fetch_ohlcv.call_args_list
    # CcxtClient.fetch_ohlcv 시그니처: (symbol, timeframe, limit=500). limit 은 kwargs.
    limits_by_tf = {c.args[1]: c.kwargs.get("limit") for c in calls}
    assert limits_by_tf["1H"] == 200
    assert limits_by_tf["4H"] == 500


# ============================================================
# step — 봉 경계 검출
# ============================================================


@pytest.mark.asyncio
async def test_step_no_new_bar_skips_fetch():
    """현재 봉 진행 중 (now_ts 가 마지막 봉 + tf_ms 이내) — fetch 0회."""
    df_1h = _make_df(start_ts_ms=1_700_000_000_000, count=10, tf_minutes=60)
    client = _make_mock_client({"1H": df_1h})
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["1H"])
    await cache.warmup({"1H": 10})

    initial_calls = client.fetch_ohlcv.call_count

    # last_bar_ts = 1_700_000_000_000 + 9*60min = ...
    last_bar_ts_ms = int(df_1h.index[-1].timestamp() * 1000)
    # now_ts = last_bar_ts + 30분 (1H tf 내) → 봉 경계 X
    now_ts = last_bar_ts_ms + 30 * 60_000

    result = await cache.step(now_ts=now_ts)
    assert "1H" in result
    # warmup 이후 추가 fetch 없음
    assert client.fetch_ohlcv.call_count == initial_calls


@pytest.mark.asyncio
async def test_step_new_bar_triggers_fetch():
    """봉 경계 시점 (now_ts > last_bar + tf_ms) — fetch + append."""
    df_1h = _make_df(start_ts_ms=1_700_000_000_000, count=10, tf_minutes=60)
    last_bar_ts_ms = int(df_1h.index[-1].timestamp() * 1000)

    # warmup 후 새 봉 (last_bar + 1H + 1ms) 가 fetch 응답에 포함되도록 시뮬
    new_df_1h = _make_df(start_ts_ms=last_bar_ts_ms + 60_000 * 60, count=2, tf_minutes=60)

    # call 순서별 응답 분리: warmup 응답 vs step 응답
    fetch_calls: list[Any] = []

    async def _fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        fetch_calls.append((symbol, timeframe, limit))
        if len(fetch_calls) == 1:
            return df_1h.copy()  # warmup
        return new_df_1h.copy()  # step refresh

    client = MagicMock()
    client.fetch_ohlcv = AsyncMock(side_effect=_fetch_ohlcv)
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["1H"])
    await cache.warmup({"1H": 10})
    initial_total = len(cache.get("1H"))
    assert initial_total == 10

    # 봉 경계 명시 — last + 2*tf_ms (다음 봉 시작 후)
    now_ts = last_bar_ts_ms + 2 * 60_000 * 60
    await cache.step(now_ts=now_ts)

    # 2개의 새 봉 append 됨
    assert len(cache.get("1H")) == 12


@pytest.mark.asyncio
async def test_step_empty_cache_forces_fetch():
    """warmup 안 한 TF — _has_new_bar=True 로 fetch 강제."""
    df_1h = _make_df(start_ts_ms=1_700_000_000_000, count=5, tf_minutes=60)
    client = _make_mock_client({"1H": df_1h})
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["1H"])
    # warmup X — cache 비어있음

    await cache.step(now_ts=1_700_000_000_000)
    # fetch 호출됨 (cache 비어있어서 강제)
    assert client.fetch_ohlcv.call_count == 1
    assert len(cache.get("1H")) == 5


@pytest.mark.asyncio
async def test_step_handles_empty_fetch_response():
    """fetch 가 빈 DataFrame 반환 — cache 그대로 유지 (회귀 안전)."""
    df_1h = _make_df(start_ts_ms=1_700_000_000_000, count=10, tf_minutes=60)
    last_bar_ts_ms = int(df_1h.index[-1].timestamp() * 1000)

    fetch_calls: list[Any] = []

    async def _fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        fetch_calls.append(timeframe)
        if len(fetch_calls) == 1:
            return df_1h.copy()
        return pd.DataFrame()  # step 시 빈 응답

    client = MagicMock()
    client.fetch_ohlcv = AsyncMock(side_effect=_fetch_ohlcv)
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["1H"])
    await cache.warmup({"1H": 10})

    now_ts = last_bar_ts_ms + 2 * 60_000 * 60
    await cache.step(now_ts=now_ts)
    # cache 그대로 유지
    assert len(cache.get("1H")) == 10


@pytest.mark.asyncio
async def test_step_dedupes_overlapping_timestamps():
    """fetch 응답이 기존 cache 와 겹치는 봉 포함 — duplicate 제거 (회귀 보호)."""
    df_1h = _make_df(start_ts_ms=1_700_000_000_000, count=5, tf_minutes=60)
    last_bar_ts_ms = int(df_1h.index[-1].timestamp() * 1000)

    # 새 fetch: 마지막 3 봉 겹침 + 새 봉 2 개 (총 5)
    overlap_start = int(df_1h.index[-3].timestamp() * 1000)
    new_df_1h = _make_df(start_ts_ms=overlap_start, count=5, tf_minutes=60)

    fetch_calls: list[Any] = []

    async def _fetch_ohlcv(symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
        fetch_calls.append(timeframe)
        if len(fetch_calls) == 1:
            return df_1h.copy()
        return new_df_1h.copy()

    client = MagicMock()
    client.fetch_ohlcv = AsyncMock(side_effect=_fetch_ohlcv)
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["1H"])
    await cache.warmup({"1H": 5})

    now_ts = last_bar_ts_ms + 2 * 60_000 * 60
    await cache.step(now_ts=now_ts)
    df = cache.get("1H")
    # 5 (기존) + 2 (새 봉) = 7. 중복 3 봉 제거됨.
    assert len(df) == 7
    # 중복 index 없음
    assert df.index.is_unique


# ============================================================
# get — 캐시 read-only
# ============================================================


def test_get_raises_on_missing_timeframe():
    """warmup 안 한 TF get → KeyError."""
    client = MagicMock()
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["1H"])
    with pytest.raises(KeyError, match="cache 에 없음"):
        cache.get("1H")


@pytest.mark.asyncio
async def test_get_returns_warmed_dataframe():
    """warmup 후 get → 캐시 DataFrame."""
    df_1h = _make_df(start_ts_ms=1_700_000_000_000, count=10, tf_minutes=60)
    client = _make_mock_client({"1H": df_1h})
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["1H"])
    await cache.warmup({"1H": 10})
    result = cache.get("1H")
    assert len(result) == 10


# ============================================================
# now_ts default — time.time() fallback
# ============================================================


@pytest.mark.asyncio
async def test_step_uses_time_now_when_none():
    """now_ts=None 이면 time.time() 호출 — fetch 결정에 활용."""
    df_1h = _make_df(start_ts_ms=1_700_000_000_000, count=5, tf_minutes=60)
    client = _make_mock_client({"1H": df_1h})
    cache = MultiTfCache(client, "BTC/USDT:USDT", ["1H"])
    await cache.warmup({"1H": 5})
    initial_calls = client.fetch_ohlcv.call_count

    # 현재 시각 = 2026 (df 의 1_700_000_000_000 = 2023-11). 봉 경계 한참 지남 → fetch.
    await cache.step()  # now_ts=None
    assert client.fetch_ohlcv.call_count == initial_calls + 1
