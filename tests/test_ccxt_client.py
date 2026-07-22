"""ccxt_client.py 단위 테스트 — mock 기반, 외부 네트워크 X.

pytest-asyncio 자동 모드 (pyproject.toml asyncio_mode=auto) 활용.
ccxt async 인스턴스를 ``unittest.mock.AsyncMock`` 으로 패치.

검증 카테고리:
    - 인스턴스 생성 옵션 (Bybit perpetual options + Demo Trading)
    - paper 모드 분기 (place_order / set_leverage / cancel_all 가짜 응답)
    - fetch_ohlcv 변환 (ccxt row → DataFrame)
    - fetch_position / get_positions (ccxt position → Aurora Position)
    - get_equity (ccxt balance → Aurora Balance)
    - close() lifecycle

담당: ChoYoon (어댑터 PR 위임 받음, 2026-05-03)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from aurora.exchange.base import Balance, Order
from aurora.exchange.ccxt_client import CcxtClient

# ============================================================
# 인스턴스 생성 — DESIGN.md §3.1 옵션 검증
# ============================================================


def _make_client(exchange_id: str = "bybit", demo: bool = True) -> CcxtClient:
    """테스트용 CcxtClient — async 인스턴스를 MagicMock 으로 대체."""
    with patch("aurora.exchange.ccxt_client.ccxt_async") as mock_async:
        # ex_class = getattr(ccxt_async, exchange_id) → MagicMock
        mock_ex_instance = MagicMock()
        mock_ex_instance.fetch_ohlcv = AsyncMock(return_value=[])
        mock_ex_instance.fetch_positions = AsyncMock(return_value=[])
        mock_ex_instance.fetch_balance = AsyncMock(return_value={})
        mock_ex_instance.create_order = AsyncMock(return_value={})
        mock_ex_instance.set_leverage = AsyncMock(return_value=None)
        mock_ex_instance.cancel_all_orders = AsyncMock(return_value=None)
        mock_ex_instance.load_time_difference = AsyncMock(return_value=None)
        mock_ex_instance.close = AsyncMock(return_value=None)
        mock_ex_instance.enableDemoTrading = MagicMock(return_value=None)

        mock_ex_class = MagicMock(return_value=mock_ex_instance)
        setattr(mock_async, exchange_id, mock_ex_class)

        client = CcxtClient(
            exchange_id=exchange_id,  # type: ignore[arg-type]
            api_key="test-key",
            api_secret="test-secret",
            demo=demo,
        )
        # 테스트가 mock 호출 검증할 수 있게 노출
        client._mock_ex = mock_ex_instance        # type: ignore[attr-defined]
        client._mock_class = mock_ex_class        # type: ignore[attr-defined]
        return client


def test_constructor_bybit_perpetual_options():
    """Bybit 인스턴스 생성 시 swap + linear + recvWindow=60000 옵션 적용."""
    client = _make_client(exchange_id="bybit")
    # 인스턴스 생성 호출 인자 검증
    call_args = client._mock_class.call_args        # type: ignore[attr-defined]
    config = call_args.args[0] if call_args.args else call_args.kwargs.get("config", {})
    options = config["options"]
    assert options["defaultType"] == "swap"
    assert options["defaultSubType"] == "linear"   # USDT-margined 명시 (PR-2 패턴)
    assert options["recvWindow"] == 60000          # Windows clock skew
    assert options["adjustForTimeDifference"] is True
    assert config["enableRateLimit"] is True


def test_constructor_bybit_demo_enabled():
    """demo=True + bybit 일 때 enableDemoTrading 호출."""
    client = _make_client(exchange_id="bybit", demo=True)
    client._mock_ex.enableDemoTrading.assert_called_once_with(True)  # type: ignore[attr-defined]


def test_constructor_bybit_demo_disabled():
    """demo=False 일 때 enableDemoTrading 호출 X."""
    client = _make_client(exchange_id="bybit", demo=False)
    client._mock_ex.enableDemoTrading.assert_not_called()  # type: ignore[attr-defined]


# ============================================================
# fetch_ohlcv — DataFrame 변환
# ============================================================


@pytest.mark.asyncio
async def test_fetch_ohlcv_converts_rows_to_df():
    """ccxt row list → DatetimeIndex DataFrame 변환."""
    client = _make_client()
    client._mock_ex.fetch_ohlcv = AsyncMock(return_value=[  # type: ignore[attr-defined]
        [1700000000000, 100.0, 110.0, 95.0, 105.0, 1000.0],
        [1700003600000, 105.0, 108.0, 102.0, 107.0, 800.0],
    ])

    df = await client.fetch_ohlcv("BTC/USDT:USDT", "1H", limit=2)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 2
    assert df.iloc[0]["close"] == 105.0
    assert df.iloc[1]["volume"] == 800.0


@pytest.mark.asyncio
async def test_fetch_ohlcv_empty_response_returns_empty_df():
    """빈 응답 — 빈 DataFrame (호출자가 안전하게 처리 가능)."""
    client = _make_client()
    df = await client.fetch_ohlcv("BTC/USDT:USDT", "1H", limit=10)
    assert df.empty
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


@pytest.mark.asyncio
async def test_fetch_ohlcv_normalizes_aurora_tf_to_ccxt():
    """Aurora 포맷 timeframe → ccxt 포맷 변환 후 호출."""
    client = _make_client()
    await client.fetch_ohlcv("BTC/USDT:USDT", "1H", limit=10)
    call_args = client._mock_ex.fetch_ohlcv.call_args  # type: ignore[attr-defined]
    # ccxt 포맷 = 소문자
    assert call_args.args[1] == "1h"


# ============================================================
# Position / Balance
# ============================================================


@pytest.mark.asyncio
async def test_get_equity_parses_usdt_balance():
    """ccxt balance.USDT → Aurora Balance dataclass."""
    client = _make_client()
    client._mock_ex.fetch_balance = AsyncMock(return_value={  # type: ignore[attr-defined]
        "USDT": {"total": 8663.43, "free": 5000.0, "used": 3663.43},
    })
    balance = await client.get_equity()
    assert isinstance(balance, Balance)
    assert balance.total_usd == 8663.43
    assert balance.free_usd == 5000.0
    assert balance.used_usd == 3663.43


@pytest.mark.asyncio
async def test_get_equity_handles_missing_usdt():
    """USDT 키 없는 응답 — 0 으로 안전 fallback."""
    client = _make_client()
    client._mock_ex.fetch_balance = AsyncMock(return_value={})  # type: ignore[attr-defined]
    balance = await client.get_equity()
    assert balance.total_usd == 0.0
    assert balance.free_usd == 0.0
    assert balance.used_usd == 0.0


@pytest.mark.asyncio
async def test_get_positions_filters_zero_contracts():
    """contracts=0 인 row 는 필터링 (close 된 포지션 응답 방어).

    Note: paper 모드는 빈 리스트 즉시 반환이므로 demo 모드 명시 필요
    (CI 환경 .env 부재 → default "paper" 회귀 보호).
    """
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        client._mock_ex.fetch_positions = AsyncMock(return_value=[  # type: ignore[attr-defined]
            {"symbol": "BTC/USDT:USDT", "side": "long", "contracts": 0.5,
             "entryPrice": 50000, "leverage": 10, "unrealizedPnl": 100, "marginMode": "isolated"},
            {"symbol": "ETH/USDT:USDT", "side": "short", "contracts": 0,  # close 됨
             "entryPrice": 0, "leverage": 1, "unrealizedPnl": 0, "marginMode": "isolated"},
        ])
        positions = await client.get_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "BTC/USDT:USDT"
        assert positions[0].side == "long"
        assert positions[0].qty == 0.5


@pytest.mark.asyncio
async def test_fetch_position_returns_none_when_no_open():
    """open contract 없으면 None — demo 모드 (실 fetch_positions 호출) 검증.

    Note: paper 모드 None 반환은 별도 ``test_paper_mode_fetch_position_returns_none``.
    """
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        client._mock_ex.fetch_positions = AsyncMock(return_value=[  # type: ignore[attr-defined]
            {"symbol": "BTC/USDT:USDT", "contracts": 0},
        ])
        pos = await client.fetch_position("BTC/USDT:USDT")
        assert pos is None


# ============================================================
# Paper 모드 — DESIGN.md §3.2 / E-3
# ============================================================


@pytest.mark.asyncio
async def test_paper_mode_place_order_returns_fake():
    """paper 모드 = 실 호출 X, 가짜 'filled' Order 반환."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "paper"
        client = _make_client()
        order = await client.place_order(
            "BTC/USDT:USDT", side="buy", qty=0.1, price=None,
        )
        assert isinstance(order, Order)
        assert order.status == "filled"
        assert order.order_id.startswith("paper-")
        # 거래소 호출 검증 — create_order 안 불려야
        client._mock_ex.create_order.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_paper_mode_set_leverage_skipped():
    """paper 모드 = set_leverage 노출 X."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "paper"
        client = _make_client()
        await client.set_leverage("BTC/USDT:USDT", 10)
        client._mock_ex.set_leverage.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_paper_mode_fetch_position_returns_none():
    """paper 모드 = 항상 None (실 fetch_positions 호출 X)."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "paper"
        client = _make_client()
        result = await client.fetch_position("BTC/USDT:USDT")
        assert result is None
        client._mock_ex.fetch_positions.assert_not_called()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_paper_mode_get_positions_returns_empty():
    """paper 모드 = 빈 리스트."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "paper"
        client = _make_client()
        result = await client.get_positions()
        assert result == []


# ============================================================
# Demo / Live 모드 실제 호출
# ============================================================


@pytest.mark.asyncio
async def test_demo_mode_place_order_calls_ccxt():
    """demo 모드 = 실 create_order 호출."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        client._mock_ex.create_order = AsyncMock(return_value={  # type: ignore[attr-defined]
            "id": "12345", "symbol": "BTC/USDT:USDT", "amount": 0.1,
            "price": None, "status": "filled", "timestamp": 1700000000000,
        })
        order = await client.place_order(
            "BTC/USDT:USDT", side="buy", qty=0.1, price=None,
        )
        assert order.order_id == "12345"
        assert order.status == "filled"
        client._mock_ex.create_order.assert_called_once()  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_set_leverage_arg_order():
    """ccxt set_leverage(leverage, symbol) — 인자 순서 반대 검증 (회귀 보호)."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        await client.set_leverage("BTC/USDT:USDT", 10)
        client._mock_ex.set_leverage.assert_called_once_with(10, "BTC/USDT:USDT")  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_set_leverage_idempotent_swallows_110043():
    """Bybit retCode 110043 'leverage not modified' — silent OK (idempotent).

    Why: 봇이 매 진입마다 set_leverage 호출 → 같은 leverage 면 빈발.
    silent OK 안 하면 봇 진입 자체가 BadRequest 로 깨짐.
    """
    import ccxt as _ccxt
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        # set_leverage 가 retCode 110043 BadRequest raise 하도록 mock
        client._mock_ex.set_leverage = AsyncMock(  # type: ignore[attr-defined]
            side_effect=_ccxt.BadRequest(
                'bybit {"retCode":110043,"retMsg":"leverage not modified"}',
            ),
        )
        # 호출 자체는 raise X (silent OK)
        await client.set_leverage("BTC/USDT:USDT", 10)


@pytest.mark.asyncio
async def test_set_leverage_other_badrequest_raises():
    """다른 BadRequest (예: 110001 invalid leverage) — 그대로 raise (실 에러)."""
    import ccxt as _ccxt
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        client._mock_ex.set_leverage = AsyncMock(  # type: ignore[attr-defined]
            side_effect=_ccxt.BadRequest(
                'bybit {"retCode":110001,"retMsg":"invalid leverage"}',
            ),
        )
        with pytest.raises(_ccxt.BadRequest):
            await client.set_leverage("BTC/USDT:USDT", 999)


@pytest.mark.asyncio
async def test_place_order_reduce_only_param():
    """reduce_only=True 시 ccxt params 에 reduceOnly=True 전달."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        client._mock_ex.create_order = AsyncMock(return_value={"id": "1"})  # type: ignore[attr-defined]
        await client.place_order(
            "BTC/USDT:USDT", side="sell", qty=0.1, reduce_only=True,
        )
        call_args = client._mock_ex.create_order.call_args  # type: ignore[attr-defined]
        params = call_args.args[5]
        assert params["reduceOnly"] is True
        # 2026-07-22: 모든 봇 주문에 태그(orderLinkId) 동봉.
        assert params["orderLinkId"].startswith("AUR")


# ============================================================
# Lifecycle — _ensure_init / close
# ============================================================


@pytest.mark.asyncio
async def test_ensure_init_called_once():
    """첫 호출 시 load_time_difference, 그 후 호출은 skip."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        await client.get_equity()
        await client.get_equity()
        assert client._mock_ex.load_time_difference.call_count == 1  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ensure_init_resyncs_after_interval():
    """시각차 보정은 _TIME_SYNC_INTERVAL_SEC 경과 시 재호출 (드리프트 누적 차단).

    Why: 시작 1회 보정만 하면 Windows 시계 드리프트로 bybit +1000ms 초과 →
    InvalidNonce 폭주. 주기적 재동기로 누적 차단하는 회귀 보호.
    """
    from aurora.exchange.ccxt_client import _TIME_SYNC_INTERVAL_SEC
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings, \
         patch("aurora.exchange.ccxt_client.time") as mock_time:
        mock_settings.run_mode = "demo"
        # monotonic: 1회차 첫 보정 / 2회차 interval 미만 skip / 3회차 interval 초과 재보정
        mock_time.monotonic = MagicMock(side_effect=[
            1000.0,
            1000.0 + _TIME_SYNC_INTERVAL_SEC - 1,
            1000.0 + _TIME_SYNC_INTERVAL_SEC + 1,
        ])
        client = _make_client()
        await client.get_equity()
        await client.get_equity()
        await client.get_equity()
        assert client._mock_ex.load_time_difference.call_count == 2  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_ensure_init_resync_failure_non_fatal():
    """재동기 실패(ccxt 에러)는 비치명 — 기존 오프셋 유지하고 호출 진행.

    Why: 일시 네트워크로 시각 보정이 실패해도 매매 step 자체가 깨지면 안 됨.
    """
    import ccxt
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        client._mock_ex.load_time_difference = AsyncMock(  # type: ignore[attr-defined]
            side_effect=ccxt.NetworkError("time sync down"),
        )
        client._mock_ex.fetch_balance = AsyncMock(return_value={  # type: ignore[attr-defined]
            "USDT": {"total": 100.0, "free": 100.0, "used": 0.0},
        })
        balance = await client.get_equity()  # raise 안 하고 정상 반환
        assert balance.total_usd == 100.0


@pytest.mark.asyncio
async def test_close_calls_ccxt_close():
    """close() → ccxt async 인스턴스 close (httpx 세션 정리)."""
    client = _make_client()
    await client.close()
    client._mock_ex.close.assert_called_once()  # type: ignore[attr-defined]


# ============================================================
# Closed positions history (v0.1.23) — Bybit V5 closed-pnl
# ============================================================


@pytest.mark.asyncio
async def test_fetch_ticker_paper_returns_none():
    """v0.1.39 — paper 모드는 fetch_ticker 실 호출 X (None 반환)."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "paper"
        client = _make_client()
        result = await client.fetch_ticker("BTC/USDT:USDT")
        assert result is None


@pytest.mark.asyncio
async def test_fetch_ticker_returns_last_price():
    """v0.1.39 — ccxt fetch_ticker 응답의 'last' 필드 → float."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        client._mock_ex.fetch_ticker = AsyncMock(  # type: ignore[attr-defined]
            return_value={"last": 78845.50, "bid": 78840.0, "ask": 78850.0},
        )
        result = await client.fetch_ticker("BTC/USDT:USDT")
        assert result == 78845.50


@pytest.mark.asyncio
async def test_fetch_ticker_returns_none_on_missing_last():
    """'last' 누락 / None 시 fallback 위해 None 반환."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        client._mock_ex.fetch_ticker = AsyncMock(  # type: ignore[attr-defined]
            return_value={"bid": 78840.0, "ask": 78850.0},  # 'last' 없음
        )
        assert await client.fetch_ticker("BTC/USDT:USDT") is None


@pytest.mark.asyncio
async def test_fetch_ticker_returns_none_on_network_error():
    """v0.1.39 — 네트워크 에러 raise 안 하고 None (호출자 close fallback)."""
    import ccxt
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client()
        client._mock_ex.fetch_ticker = AsyncMock(  # type: ignore[attr-defined]
            side_effect=ccxt.NetworkError("network down"),
        )
        assert await client.fetch_ticker("BTC/USDT:USDT") is None


@pytest.mark.asyncio
async def test_fetch_closed_positions_paper_returns_empty():
    """paper 모드 = 빈 리스트 (실 호출 X)."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "paper"
        client = _make_client()
        result = await client.fetch_closed_positions(since_ms=1000, limit=50)
        assert result == []


@pytest.mark.asyncio
async def test_fetch_closed_positions_non_bybit_returns_empty():
    """bybit 외 거래소 = 빈 리스트 (TODO 미구현)."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client(exchange_id="okx")
        result = await client.fetch_closed_positions()
        assert result == []


@pytest.mark.asyncio
async def test_fetch_closed_positions_bybit_parses_records():
    """Bybit V5 closed-pnl 응답 → ClosedPosition 변환 — 핵심 필드 매핑.

    v0.1.27: Bybit ``endTime - startTime ≤ 7일`` 제약 + endTime 명시 추가.
    since_ms 미지정 = default 최근 7일 (단일 chunk fast path 검증).
    """
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings, \
         patch("aurora.exchange.ccxt_client.time") as mock_time:
        mock_settings.run_mode = "demo"
        mock_time.time = lambda: 1735100000.0  # 고정 now (chunk 계산 결정론)
        client = _make_client(exchange_id="bybit")
        # 두 record — 롱 청산 (Sell) + 숏 청산 (Buy)
        client._mock_ex.private_get_v5_position_closed_pnl = AsyncMock(  # type: ignore[attr-defined]
            return_value={
                "result": {
                    "list": [
                        {
                            "symbol": "BTCUSDT",
                            "side": "Sell",         # 롱 청산
                            "leverage": "10",
                            "closedSize": "0.01",
                            "avgEntryPrice": "60000",
                            "avgExitPrice": "61000",
                            "closedPnl": "10",
                            "createdTime": "1735000000000",
                            "updatedTime": "1735000600000",
                        },
                        {
                            "symbol": "ETHUSDT",
                            "side": "Buy",          # 숏 청산
                            "leverage": "20",
                            "closedSize": "0.5",
                            "avgEntryPrice": "3000",
                            "avgExitPrice": "2950",
                            "closedPnl": "25",
                            "createdTime": "1735000200000",
                            "updatedTime": "1735000800000",
                        },
                    ],
                    "nextPageCursor": "",  # 페이지네이션 종료
                },
            },
        )
        # since_ms = 1735100000_000 - 7일 ms — 단일 chunk fast path
        since_ms = int(1735100000.0 * 1000) - 7 * 24 * 60 * 60 * 1000
        result = await client.fetch_closed_positions(since_ms=since_ms, limit=200)
        assert len(result) == 2

        # 신→구 정렬 — closed_at_ts 큰 게 앞 (ETHUSDT 1735000800 > BTCUSDT 1735000600)
        first = result[0]
        assert first.symbol == "ETH/USDT:USDT"
        assert first.closed_at_ts == 1735000800000

        second = result[1]
        assert second.symbol == "BTC/USDT:USDT"
        assert second.direction == "long"
        assert second.leverage == 10
        assert second.qty == 0.01
        assert second.entry_price == 60000.0
        assert second.exit_price == 61000.0
        assert second.pnl_usd == 10.0
        # margin = (60000 × 0.01) / 10 = 60. ROI = 10 / 60 × 100 ≈ 16.67%
        assert abs(second.roi_pct - (10.0 / 60.0 * 100.0)) < 1e-6

        # API params: category=linear, startTime, endTime (v0.1.27 추가), limit
        call_args = client._mock_ex.private_get_v5_position_closed_pnl.call_args  # type: ignore[attr-defined]
        params = call_args.args[0] if call_args.args else call_args.kwargs.get("params", {})
        assert params["category"] == "linear"
        assert params["startTime"] == since_ms
        assert "endTime" in params  # v0.1.27 — 7일 default 윈도우 우회 fix
        assert params["endTime"] == int(1735100000.0 * 1000)
        assert params["limit"] == 200


@pytest.mark.asyncio
async def test_fetch_closed_positions_bybit_long_period_chunks():
    """30일치 fetch — 7일 chunks 자동 페이지네이션 (v0.1.27 fix).

    Bybit V5 endTime - startTime ≤ 7일 제약 우회. 30일 = 5 chunks (마지막 chunk
    < 7일). 각 chunk 마다 별도 호출 발생 verify.
    """
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings, \
         patch("aurora.exchange.ccxt_client.time") as mock_time:
        mock_settings.run_mode = "demo"
        mock_time.time = lambda: 1735100000.0
        client = _make_client(exchange_id="bybit")
        client._mock_ex.private_get_v5_position_closed_pnl = AsyncMock(  # type: ignore[attr-defined]
            return_value={"result": {"list": [], "nextPageCursor": ""}},
        )

        now_ms = int(1735100000.0 * 1000)
        thirty_days_ago = now_ms - 30 * 24 * 60 * 60 * 1000
        await client.fetch_closed_positions(since_ms=thirty_days_ago)

        # 30일 / 7일 = 4.28... → 5 chunks 호출
        assert client._mock_ex.private_get_v5_position_closed_pnl.call_count == 5  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_fetch_closed_positions_bybit_pagination_within_chunk():
    """한 chunk 안에 record 200 초과 → cursor 페이지네이션 자동 follow."""
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings, \
         patch("aurora.exchange.ccxt_client.time") as mock_time:
        mock_settings.run_mode = "demo"
        mock_time.time = lambda: 1735100000.0
        client = _make_client(exchange_id="bybit")

        # 첫 페이지 = cursor 있음, 둘째 페이지 = cursor 없음
        record_template = {
            "symbol": "BTCUSDT", "side": "Sell", "leverage": "10",
            "closedSize": "0.01", "avgEntryPrice": "60000",
            "avgExitPrice": "61000", "closedPnl": "10",
            "createdTime": "1735000000000", "updatedTime": "1735000600000",
        }
        responses = [
            {"result": {"list": [record_template], "nextPageCursor": "next_page_token"}},
            {"result": {"list": [record_template], "nextPageCursor": ""}},
        ]
        client._mock_ex.private_get_v5_position_closed_pnl = AsyncMock(  # type: ignore[attr-defined]
            side_effect=responses,
        )

        # since_ms 단일 chunk (7일 이내)
        since_ms = int(1735100000.0 * 1000) - 3 * 24 * 60 * 60 * 1000
        result = await client.fetch_closed_positions(since_ms=since_ms)

        # 두 페이지 = 2 record
        assert len(result) == 2
        # API 호출 2회 (cursor 페이지네이션)
        assert client._mock_ex.private_get_v5_position_closed_pnl.call_count == 2  # type: ignore[attr-defined]

        # 두 번째 호출에 cursor 파라미터 포함
        second_call = client._mock_ex.private_get_v5_position_closed_pnl.call_args_list[1]  # type: ignore[attr-defined]
        params = second_call.args[0] if second_call.args else second_call.kwargs.get("params", {})
        assert params.get("cursor") == "next_page_token"


@pytest.mark.asyncio
async def test_fetch_closed_positions_bybit_chunks_iterate_recent_first():
    """chunk 순회 방향 — 최근(now)부터 옛날(since_ms) 방향 (2026-05-27 fix).

    회귀 방지 — 파트너 보고: 30일 조회 시 옛날(5/8~5/11) 만 표시되고 최근
    (5/27) 거래 누락. 옛날 chunk 부터 순회하면서 limit early-exit 으로 최근
    chunks fetch 안 한 게 원인. fix 후 첫 호출의 endTime 이 now_ms 여야 함.
    """
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings, \
         patch("aurora.exchange.ccxt_client.time") as mock_time:
        mock_settings.run_mode = "demo"
        mock_time.time = lambda: 1735100000.0
        client = _make_client(exchange_id="bybit")
        client._mock_ex.private_get_v5_position_closed_pnl = AsyncMock(  # type: ignore[attr-defined]
            return_value={"result": {"list": [], "nextPageCursor": ""}},
        )

        now_ms = int(1735100000.0 * 1000)
        thirty_days_ago = now_ms - 30 * 24 * 60 * 60 * 1000
        await client.fetch_closed_positions(since_ms=thirty_days_ago)

        # 첫 호출 = 최근 chunk = (now - 7d, now)
        first_call = client._mock_ex.private_get_v5_position_closed_pnl.call_args_list[0]  # type: ignore[attr-defined]
        first_params = first_call.args[0] if first_call.args else first_call.kwargs.get("params", {})
        assert first_params["endTime"] == now_ms, "첫 chunk 의 endTime 은 now_ms 여야 함 (최근부터 순회)"
        assert first_params["startTime"] == now_ms - 7 * 24 * 60 * 60 * 1000

        # 마지막 호출 = 가장 옛날 chunk = (now - 30d, now - 28d)
        last_call = client._mock_ex.private_get_v5_position_closed_pnl.call_args_list[-1]  # type: ignore[attr-defined]
        last_params = last_call.args[0] if last_call.args else last_call.kwargs.get("params", {})
        assert last_params["startTime"] == thirty_days_ago


@pytest.mark.asyncio
async def test_fetch_closed_positions_bybit_limit_early_exit_keeps_recent():
    """limit 채워지면 옛날 chunk fetch skip — 최근 거래 보장 (2026-05-27 fix).

    버그 시나리오: 30일 조회 시 옛날 chunk 에서 limit 채워지면 최근 chunk
    안 도는 게 root cause. fix 후 — 최근 chunk 에서 채워지면 옛날은 skip
    (최근 거래 보장됨).
    """
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings, \
         patch("aurora.exchange.ccxt_client.time") as mock_time:
        mock_settings.run_mode = "demo"
        mock_time.time = lambda: 1735100000.0
        client = _make_client(exchange_id="bybit")

        # 첫(최근) chunk 가 limit 만큼 채움 → 옛날 chunks skip 되어야 함
        recent_records = [
            {
                "symbol": "BTCUSDT", "side": "Sell", "leverage": "10",
                "closedSize": "0.01", "avgEntryPrice": "60000",
                "avgExitPrice": "61000", "closedPnl": "10",
                "createdTime": "1735000000000", "updatedTime": str(1735090000000 + i),
            }
            for i in range(50)
        ]
        client._mock_ex.private_get_v5_position_closed_pnl = AsyncMock(  # type: ignore[attr-defined]
            return_value={"result": {"list": recent_records, "nextPageCursor": ""}},
        )

        now_ms = int(1735100000.0 * 1000)
        thirty_days_ago = now_ms - 30 * 24 * 60 * 60 * 1000
        result = await client.fetch_closed_positions(since_ms=thirty_days_ago, limit=50)

        # 30일이지만 첫 chunk 에서 limit 채움 → 1번만 호출 (옛날 4 chunks skip)
        assert client._mock_ex.private_get_v5_position_closed_pnl.call_count == 1, (  # type: ignore[attr-defined]
            "limit 채웠는데 다음 chunk fetch — 최근 거래만 보장하려면 early exit"
        )
        assert len(result) == 50


@pytest.mark.asyncio
async def test_fetch_closed_positions_bybit_network_error_returns_empty():
    """네트워크 에러 시 raise 안 하고 빈 리스트 (UI 안전)."""
    import ccxt
    with patch("aurora.exchange.ccxt_client.settings") as mock_settings:
        mock_settings.run_mode = "demo"
        client = _make_client(exchange_id="bybit")
        client._mock_ex.private_get_v5_position_closed_pnl = AsyncMock(  # type: ignore[attr-defined]
            side_effect=ccxt.NetworkError("network down"),
        )
        result = await client.fetch_closed_positions()
        assert result == []


# ============================================================
# 봇 주문 태그 (orderLinkId) — 2026-07-22
# 재부팅 후 고아 포지션 소유권 인식 + 유저 수동 주문 보존(선별 취소)
# ============================================================


def test_is_bot_order_and_gen_link_id() -> None:
    """_gen_bot_order_link_id 는 태그 prefix + 36자 미만, _is_bot_order 판정."""
    from aurora.exchange.ccxt_client import (
        BOT_ORDER_TAG,
        _gen_bot_order_link_id,
        _is_bot_order,
    )

    oid = _gen_bot_order_link_id()
    assert oid.startswith(BOT_ORDER_TAG)
    assert len(oid) < 36                       # Bybit orderLinkId 한도
    assert _gen_bot_order_link_id() != oid     # 고유
    # clientOrderId 필드
    assert _is_bot_order({"clientOrderId": oid}) is True
    # info.orderLinkId 폴백
    assert _is_bot_order({"info": {"orderLinkId": oid}}) is True
    # 유저 수동 주문(태그 없음)
    assert _is_bot_order({"clientOrderId": "user-123"}) is False
    assert _is_bot_order({"clientOrderId": ""}) is False
    assert _is_bot_order({}) is False


@pytest.mark.asyncio
async def test_place_order_adds_bot_tag() -> None:
    """place_order 가 create_order params 에 봇 태그 orderLinkId 부착."""
    from aurora.exchange.ccxt_client import BOT_ORDER_TAG

    client = _make_client()
    client._mock_ex.create_order = AsyncMock(return_value={  # type: ignore[attr-defined]
        "id": "o1", "symbol": "BTC/USDT:USDT", "side": "buy",
        "amount": 1.0, "price": 100.0, "status": "open", "info": {},
    })
    await client.place_order("BTC/USDT:USDT", "buy", 1.0, price=100.0)
    params = client._mock_ex.create_order.call_args.args[5]  # type: ignore[attr-defined]
    assert params["orderLinkId"].startswith(BOT_ORDER_TAG)


@pytest.mark.asyncio
async def test_cancel_bot_orders_only_cancels_tagged() -> None:
    """cancel_bot_orders 는 봇 태그 주문만 취소, 유저 수동 주문 보존."""
    from aurora.exchange.ccxt_client import _gen_bot_order_link_id

    bot_oid = _gen_bot_order_link_id()
    client = _make_client()
    client._mock_ex.fetch_open_orders = AsyncMock(return_value=[  # type: ignore[attr-defined]
        {"id": "bot1", "clientOrderId": bot_oid},
        {"id": "user1", "clientOrderId": "manual-abc"},
        {"id": "bot2", "info": {"orderLinkId": _gen_bot_order_link_id()}},
    ])
    client._mock_ex.cancel_order = AsyncMock(return_value=None)  # type: ignore[attr-defined]
    n = await client.cancel_bot_orders("BTC/USDT:USDT")
    assert n == 2  # 봇 주문 2건만
    cancelled_ids = {
        c.args[0] for c in client._mock_ex.cancel_order.call_args_list  # type: ignore[attr-defined]
    }
    assert cancelled_ids == {"bot1", "bot2"}  # user1 은 취소 안 됨


@pytest.mark.asyncio
async def test_position_opened_by_bot_matches_tag_and_entry() -> None:
    """position_opened_by_bot: 봇 태그 + 방향 + entry 1% 이내 진입주문이면 True."""
    from aurora.exchange.ccxt_client import _gen_bot_order_link_id

    client = _make_client()
    client._mock_ex.fetch_open_orders = AsyncMock(return_value=[])  # type: ignore[attr-defined]
    client._mock_ex.fetch_closed_orders = AsyncMock(return_value=[  # type: ignore[attr-defined]
        {"id": "e1", "clientOrderId": _gen_bot_order_link_id(),
         "side": "buy", "price": 80000.0, "average": 80000.0, "info": {}},
    ])
    # 매칭(long, entry 80000) → True
    assert await client.position_opened_by_bot("BTC/USDT:USDT", "long", 80000.0) is True
    # 방향 불일치(short) → False
    assert await client.position_opened_by_bot("BTC/USDT:USDT", "short", 80000.0) is False
    # entry 1% 초과 → False
    assert await client.position_opened_by_bot("BTC/USDT:USDT", "long", 90000.0) is False


@pytest.mark.asyncio
async def test_position_opened_by_bot_ignores_untagged() -> None:
    """유저 수동 주문(태그 없음)만 있으면 False (봇 것 아님)."""
    client = _make_client()
    client._mock_ex.fetch_open_orders = AsyncMock(return_value=[])  # type: ignore[attr-defined]
    client._mock_ex.fetch_closed_orders = AsyncMock(return_value=[  # type: ignore[attr-defined]
        {"id": "u1", "clientOrderId": "manual-xyz",
         "side": "buy", "price": 80000.0, "average": 80000.0, "info": {}},
    ])
    assert await client.position_opened_by_bot("BTC/USDT:USDT", "long", 80000.0) is False
