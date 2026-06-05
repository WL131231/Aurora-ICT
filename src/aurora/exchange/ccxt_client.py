"""ccxt 통합 어댑터 — Bybit Demo Trading 우선, 실거래·다른 거래소 확장 가능.

DESIGN.md §3.1 ~ §3.3 / §11 E-1 ~ E-14 정합:
    - ccxt async 인스턴스 (httpx 기반)
    - Bybit perpetual: ``defaultType='swap' + defaultSubType='linear'``
    - clock skew: ``recvWindow=60000 + adjustForTimeDifference=True + load_time_difference()``
    - Bybit Demo Trading: ``enableDemoTrading(True)`` (≠ testnet)
    - paper 모드: place_order / set_leverage / cancel_all 가짜 응답 (fetch_* 는 실 호출 OK)
    - tenacity retry: 5종 일시 장애만 재시도 (PR-2 #31 패턴 차용)
    - timeframe 변환: ``aurora.backtest.tf.normalize_to_ccxt`` 단일 점

영역: ChoYoon (위임 받음 2026-05-03, 어댑터 PR 한정)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Literal

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from aurora.backtest.tf import normalize_to_ccxt
from aurora.config import settings
from aurora.exchange.base import Balance, ClosedPosition, Order, Position

logger = logging.getLogger(__name__)

# Bybit V5 /v5/market/kline limit max — fetch_ohlcv 페이지네이션 분할 단위.
_OHLCV_MAX_PER_CALL = 1000


# tenacity retry — DESIGN.md §3.3 / E-12. PR-2 #31 _fetch_page 와 동일 정책.
# 일시 네트워크/거래소 장애만 재시도. AuthError 등은 즉시 raise (재시도 무의미).
_RETRY_TRANSIENT = retry(
    retry=retry_if_exception_type((
        ccxt.NetworkError,
        ccxt.RequestTimeout,
        ccxt.ExchangeNotAvailable,
        ccxt.RateLimitExceeded,
        ccxt.DDoSProtection,
    )),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=1, max=30),
    reraise=True,
)


# 시각 차이 재동기 주기 (초). 시작 시 1회 보정만 하면 Windows 시계가 장시간
# 가동 중 server 대비 드리프트(관측: ~11h 에 +1.3s)해 bybit 의 +1000ms 허용을
# 넘김 → 모든 서명 요청 InvalidNonce(10002) 실패 → 재시작 전까지 매매 불가.
# 따라서 주기적으로 load_time_difference 재호출해 드리프트 누적을 차단.
_TIME_SYNC_INTERVAL_SEC = 180


class CcxtClient:
    """ccxt 기반 거래소 어댑터 — Bybit Demo Trading 우선.

    호출자가 ``ExchangeClient`` Protocol 만 의존하도록 명시 상속 안 함
    (structural typing). 본 클래스 메서드 시그니처가 Protocol 정합.

    Args:
        exchange_id: 거래소 식별자 (``"bybit"`` / ``"okx"`` / ``"binance"``).
        api_key: 거래소 API 키.
        api_secret: 거래소 API 시크릿.
        passphrase: OKX 전용 (다른 거래소는 빈 문자열).
        demo: Demo Trading 모드 (Bybit 한정 — bybit.com Demo, **≠ testnet**).
            기본 False (실거래 보호). 데모 진입 시 명시 ``demo=True``.

    Lifecycle:
        ccxt async 인스턴스는 내부 httpx 세션을 보유하므로 사용 종료 시
        ``await client.close()`` 호출 필수 (asyncio 자원 누수 경고 방지).

    Example:
        >>> client = CcxtClient(
        ...     exchange_id="bybit",
        ...     api_key=settings.bybit_api_key,
        ...     api_secret=settings.bybit_api_secret,
        ...     demo=settings.bybit_demo,
        ... )
        >>> balance = await client.get_equity()
        >>> await client.close()
    """

    name: str

    def __init__(
        self,
        exchange_id: Literal["bybit", "okx", "binance"] = "bybit",
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        demo: bool = False,
    ) -> None:
        self.name = exchange_id
        self._demo = demo

        # ccxt 옵션 — DESIGN.md §3.1 검증된 조합 (2026-05-03)
        options: dict[str, Any] = {
            "defaultType": "swap",                  # Perpetual (vs spot)
            "recvWindow": 60000,                    # Windows clock skew 허용 (60초)
            "adjustForTimeDifference": True,        # 서버 시각 자동 보정
        }
        # Bybit perpetual = USDT-margined 명시 (USDC/inverse 분리, PR-2 #31 패턴)
        if exchange_id == "bybit":
            options["defaultSubType"] = "linear"
            # #LEV-4: Bybit demo 환경에서 ccxt 가 set_leverage 호출 전 자동
            # query-api (/user/v3/private/query-api) 로 UTA 체크 → demo 키 권한
            # 거부 (retCode 10005) → set_leverage 자체 실패. 아래 옵션으로
            # ccxt 가 query-api 호출 skip → set_leverage 통과 시도.
            # (Bybit V5 unified account 가 디폴트이므로 명시해도 안전.)
            options["enableUnifiedAccount"] = True
            options["enableUnifiedMargin"] = True
            options["accountType"] = "UNIFIED"

        config: dict[str, Any] = {
            "apiKey": api_key,
            "secret": api_secret,
            "enableRateLimit": True,                # ccxt 표준 — rate limit 자동 sleep
            "options": options,
        }
        # OKX 만 passphrase 사용 (ccxt 가 ``password`` 키로 받음)
        if exchange_id == "okx" and passphrase:
            config["password"] = passphrase

        # async ccxt 인스턴스 동적 생성
        ex_class = getattr(ccxt_async, exchange_id)
        self._ex = ex_class(config)

        # Bybit Demo Trading 활성화 — DESIGN.md §3.1 / E-1
        # (Demo URL = api-demo.{hostname}, ≠ testnet.bybit.com)
        if demo and exchange_id == "bybit":
            self._ex.enableDemoTrading(True)

        # 시각 차이 보정은 첫 호출 시 lazy 적용 (constructor 는 sync)
        self._initialized = False
        self._last_time_sync = 0.0  # monotonic 초 — 마지막 load_time_difference 시각

    async def _ensure_init(self) -> None:
        """첫 호출 + 주기적 시각 차이 재보정 (DESIGN.md §3.1).

        lazy init 이유: constructor 가 sync 라 ``await load_time_difference()``
        호출 불가. 첫 메서드 호출 시 보정.

        재동기 이유: 시작 1회 보정만 하면 Windows 시계가 장시간 가동 중 server
        대비 드리프트(관측 ~11h +1.3s)해 bybit +1000ms 초과 → 모든 서명 요청
        InvalidNonce(10002) → 재시작 전까지 매매 불가. ``_TIME_SYNC_INTERVAL_SEC``
        마다 재보정해 드리프트 누적을 차단한다.
        """
        now = time.monotonic()
        if self._initialized and (now - self._last_time_sync) < _TIME_SYNC_INTERVAL_SEC:
            return
        try:
            await self._ex.load_time_difference()
        except ccxt.BaseError as exc:
            # Why: 재동기 실패(일시 네트워크 등)는 비치명 — 기존 오프셋 유지하고
            # 다음 호출에서 재시도. 첫 init 실패면 _initialized 미설정이라 곧 재시도.
            logger.warning("load_time_difference 재동기 실패 (기존 오프셋 유지): %s", exc)
            return
        self._last_time_sync = now
        self._initialized = True

    async def close(self) -> None:
        """ccxt async 인스턴스 정리 — httpx 세션 close.

        호출 안 하면 asyncio 종료 시 ``Unclosed client session`` 경고.
        BotInstance lifecycle 종료 시 (또는 main.py shutdown hook) 호출 필수.
        """
        await self._ex.close()

    # ============================================================
    # OHLCV — DESIGN.md §3.3 페이지네이션 정책 (PR-2 #31 차용)
    # ============================================================

    @_RETRY_TRANSIENT
    async def _fetch_ohlcv_page(
        self,
        symbol: str,
        ccxt_tf: str,
        since_ms: int | None,
        limit: int,
    ) -> list[list[Any]]:
        """단일 페이지 fetch (tenacity retry 적용)."""
        return await self._ex.fetch_ohlcv(symbol, ccxt_tf, since=since_ms, limit=limit)

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """최근 ``limit`` 봉 OHLCV 가져오기.

        라이브 봇용 — 단일 호출 max=1000봉 (Bybit V5 kline 한도). ``limit`` 이
        1000 초과면 ``since`` 기반 옛날 방향 페이지네이션으로 자동 분할 fetch.
        max 5000봉 (안전 가드) — UI 차트 가시 범위 확장용 (2026-05-27 파트너 요청).

        Args:
            symbol: ccxt 표준 (예: ``"BTC/USDT:USDT"`` for linear perpetual).
            timeframe: Aurora 포맷 (예: ``"1H"``). ccxt 포맷으로 자동 변환.
            limit: 봉 수 (기본 500). 1000 초과는 자동 페이지네이션 (max 5000).

        Returns:
            DataFrame (DatetimeIndex UTC, columns=[open/high/low/close/volume]),
            ts 오름차순(옛날→최근). 응답 비어있으면 빈 DataFrame.
        """
        await self._ensure_init()
        ccxt_tf = normalize_to_ccxt(timeframe)
        if limit <= _OHLCV_MAX_PER_CALL:
            # fast path — 단일 호출 (기존 동작 그대로)
            page = await self._fetch_ohlcv_page(symbol, ccxt_tf, since_ms=None, limit=limit)
            return self._page_to_df(page)

        # 2026-05-27: limit > 1000 페이지네이션
        # 전략 — 첫 호출은 since=None (최신 1000봉) → cursor 를 가장 옛날 봉 ts 로,
        # 다음 호출은 since=cursor-tf_ms*1000 (그 이전 1000봉 forward) 반복.
        # 페이지 사이 1봉 겹칠 수 있어 seen ts set 으로 중복 제거.
        from aurora.backtest.replay import TF_MINUTES
        from aurora.backtest.tf import normalize_to_aurora
        aurora_tf = normalize_to_aurora(ccxt_tf)
        tf_ms = TF_MINUTES[aurora_tf] * 60_000

        seen_ts: set[int] = set()
        all_rows: list[list[Any]] = []
        cursor_ts: int | None = None
        remaining = limit
        # 2026-05-27 파트너 요청 — "거래소 시작부터 지금까지 전부".
        # max_pages 5 → 100 (100 × 1000 = 100,000봉). 거래소 history 끝나면
        # 빈 응답 / 진전 없음으로 자동 stop. Bybit BTCUSDT Perpetual 시작은
        # 2020-03 → 1H 6.17년 ≈ 54,000봉, 1D 2,250봉, 1W 322봉 모두 커버.
        max_pages = 100

        for _ in range(max_pages):
            if remaining <= 0:
                break
            page_limit = min(remaining, _OHLCV_MAX_PER_CALL)
            if cursor_ts is None:
                page = await self._fetch_ohlcv_page(symbol, ccxt_tf, since_ms=None, limit=page_limit)
            else:
                since_ms = cursor_ts - page_limit * tf_ms
                page = await self._fetch_ohlcv_page(symbol, ccxt_tf, since_ms=since_ms, limit=page_limit)
            if not page:
                break
            new_rows = [r for r in page if r[0] not in seen_ts]
            if not new_rows:
                break  # 진전 없음 — 거래소 history 끝
            for r in new_rows:
                seen_ts.add(r[0])
            new_rows.sort(key=lambda r: r[0])
            all_rows = new_rows + all_rows
            remaining -= len(new_rows)
            cursor_ts = new_rows[0][0]
        # 안전 — 전체 정렬 (페이지 prepend 시 이미 정렬이지만 방어)
        all_rows.sort(key=lambda r: r[0])
        return self._page_to_df(all_rows)

    @staticmethod
    def _page_to_df(page: list[list[Any]]) -> pd.DataFrame:
        """ccxt OHLCV row list → Aurora 표준 DataFrame.

        ccxt row: ``[ts_ms, open, high, low, close, volume]``.
        반환 DataFrame: DatetimeIndex (UTC) + 5 컬럼 (timestamp_ms 컬럼 X).
        """
        if not page:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(
            page,
            columns=["timestamp_ms", "open", "high", "low", "close", "volume"],
        )
        df.index = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
        return df[["open", "high", "low", "close", "volume"]]

    # ============================================================
    # Position / Balance
    # ============================================================

    async def fetch_position(self, symbol: str) -> Position | None:
        """단일 페어 포지션 조회 — open contract 있으면 반환, 없으면 None.

        paper 모드 = 항상 None (실 호출 X — DESIGN.md §3.2).
        """
        if settings.run_mode == "paper":
            return None
        await self._ensure_init()
        positions = await self._ex.fetch_positions([symbol])
        for raw in positions:
            if (raw.get("contracts") or 0) > 0:
                return self._parse_position(raw)
        return None

    async def get_positions(self) -> list[Position]:
        """모든 페어 포지션 — 대시보드 / multi-pair 운영용.

        contracts > 0 만 필터 (close 된 포지션 row 가 응답에 섞이는 케이스 방어).
        paper 모드 = 빈 리스트.
        """
        if settings.run_mode == "paper":
            return []
        await self._ensure_init()
        positions = await self._ex.fetch_positions()
        return [
            self._parse_position(raw)
            for raw in positions
            if (raw.get("contracts") or 0) > 0
        ]

    async def get_equity(self) -> Balance:
        """계정 자본금 (USDT 단일 자산, Phase 1).

        paper 모드도 실 fetch_balance 호출 (시드 검증 자유롭게 — DESIGN.md §3.2).
        다중 자산은 Phase 3 확장.
        """
        await self._ensure_init()
        balance = await self._ex.fetch_balance()
        usdt = balance.get("USDT", {})
        return Balance(
            total_usd=float(usdt.get("total") or 0),
            free_usd=float(usdt.get("free") or 0),
            used_usd=float(usdt.get("used") or 0),
        )

    @staticmethod
    def _parse_position(raw: dict[str, Any]) -> Position:
        """ccxt position dict → Aurora Position dataclass.

        ccxt 표준 필드 매핑 (None 안전 처리):
            - side: "long" / "short"
            - contracts: 수량 (float)
            - entryPrice / leverage / unrealizedPnl
            - marginMode: "isolated" / "cross"
        """
        side_raw = raw.get("side", "long")
        side: Literal["long", "short"] = "short" if side_raw == "short" else "long"
        margin_raw = raw.get("marginMode", "isolated")
        margin_mode: Literal["isolated", "cross"] = (
            "cross" if margin_raw == "cross" else "isolated"
        )
        return Position(
            symbol=str(raw.get("symbol") or ""),
            side=side,
            qty=float(raw.get("contracts") or 0),
            entry_price=float(raw.get("entryPrice") or 0),
            leverage=int(raw.get("leverage") or 1),
            unrealized_pnl=float(raw.get("unrealizedPnl") or 0),
            margin_mode=margin_mode,
        )

    # ============================================================
    # Order / Leverage / Cancel
    # ============================================================

    async def place_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        price: float | None = None,
        reduce_only: bool = False,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> Order:
        """주문 전송 — ``price=None`` 이면 시장가, 아니면 지정가.

        ``stop_loss`` / ``take_profit`` 가 주어지면 entry 주문에 동봉 (Bybit V5
        ``create_order`` params 의 ``stopLoss`` / ``takeProfit``). 주문 체결 시
        거래소가 포지션에 SL/TP 를 conditional 로 자동 적용 — 지정가 미체결
        주문에도 예약되어 체결 시점에 붙는다. 별도 set_trading_stop 호출 불필요.

        paper 모드 = 가짜 Order 반환 (실 호출 X). DESIGN.md §3.2 / E-3.
        """
        if settings.run_mode == "paper":
            return self._fake_order(symbol, side, qty, price)
        await self._ensure_init()
        order_type = "market" if price is None else "limit"
        params: dict[str, Any] = {}
        if reduce_only:
            params["reduceOnly"] = True
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
        raw = await self._ex.create_order(symbol, order_type, side, qty, price, params)
        return self._parse_order(raw, symbol, side, qty, price)

    async def fetch_symbol_meta(self, symbol: str) -> dict[str, float | None]:
        """심볼별 거래소 메타 — 최소수량 / lot step / 최대 레버리지.

        페어 확장(BTC·ETH → 다종목) 시 심볼마다 다른 lot size·precision·최대
        레버리지를 주문·사이징에 반영하기 위해 ccxt market 정보를 조회한다.
        markets 미로드면 1회 ``load_markets`` 후 캐시된 메타를 읽는다.

        Args:
            symbol: ccxt 통합 심볼 (예: ``"BTC/USDT:USDT"``).

        Returns:
            dict — ``min_qty`` / ``qty_step`` / ``max_leverage``. 미상장·조회
            실패 시 각 값 None (호출처가 안전 폴백하도록).
        """
        await self._ensure_init()
        try:
            if not self._ex.markets:
                await self._ex.load_markets()
            m = self._ex.market(symbol)
        except (ccxt.BaseError, KeyError, ValueError) as exc:
            logger.warning("fetch_symbol_meta 실패 (%s): %s", symbol, exc)
            return {"min_qty": None, "qty_step": None, "max_leverage": None}
        limits = m.get("limits") or {}
        amount_lim = limits.get("amount") or {}
        lev_lim = limits.get("leverage") or {}
        precision = m.get("precision") or {}
        return {
            "min_qty": amount_lim.get("min"),
            "qty_step": precision.get("amount"),
            "max_leverage": lev_lim.get("max"),
        }

    def round_amount(self, symbol: str, amount: float) -> float:
        """거래소 lot step 에 맞게 qty 를 정렬(내림)한다.

        ``amount_to_precision`` 은 markets 로드를 전제로 한다. 로드 전/미상장
        등으로 실패하면 원본 amount 를 그대로 반환해 주문 경로를 막지 않는다
        (안전 폴백 — BTC/ETH 기존 동작 회귀 방지).
        """
        try:
            return float(self._ex.amount_to_precision(symbol, amount))
        except (ccxt.BaseError, KeyError, ValueError, TypeError) as exc:
            logger.debug("round_amount 폴백 (%s, %s): %s", symbol, amount, exc)
            return amount

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """레버리지 설정 — paper 모드는 noop (로깅만).

        Idempotent 동작: Bybit 은 이미 같은 leverage 면 ``retCode 110043
        "leverage not modified"`` 로 ``BadRequest`` raise. 매 진입마다 호출하는
        패턴에서 비치명 에러라 catch + warn + return (silent OK).
        다른 retCode 는 그대로 전파 (실 에러).
        """
        if settings.run_mode == "paper":
            logger.info("paper mode: set_leverage(%s, %d) skipped", symbol, leverage)
            return
        await self._ensure_init()
        try:
            # ccxt 시그니처: set_leverage(leverage, symbol) — 인자 순서 반대 주의
            await self._ex.set_leverage(leverage, symbol)
        except ccxt.BadRequest as e:
            # Why: Bybit retCode 110043 = 이미 같은 leverage. 봇 매 진입마다 호출하는
            # 패턴에서 빈발 → silent OK. 다른 BadRequest 는 raise 보존.
            if "110043" in str(e) or "leverage not modified" in str(e):
                logger.debug(
                    "set_leverage(%s, %d): already at this leverage (110043, idempotent)",
                    symbol, leverage,
                )
                return
            raise

    async def cancel_all(self, symbol: str) -> None:
        """전체 주문 취소 (해당 페어). paper 모드는 noop."""
        if settings.run_mode == "paper":
            return
        await self._ensure_init()
        await self._ex.cancel_all_orders(symbol)

    # ============================================================
    # Real-time ticker (v0.1.39) — SL/TP 청산 폴링용 (봉 wick 즉시 반응)
    # ============================================================

    async def fetch_ticker(self, symbol: str) -> float | None:
        """현재 시장가 (last trade) 실시간 조회 — ccxt fetch_ticker 'last' 필드.

        BotInstance 가 보유 중일 때 매 step 호출 → 봉 wick 도달 시점에 SL/TP 청산.
        paper 모드 / 실패 / 'last' 누락 시 ``None`` (호출자 close fallback).
        """
        if settings.run_mode == "paper":
            return None
        await self._ensure_init()
        try:
            ticker = await self._ex.fetch_ticker(symbol)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.debug("fetch_ticker(%s) 실패 — close fallback: %s", symbol, e)
            return None
        last = ticker.get("last") if isinstance(ticker, dict) else None
        if last is None:
            return None
        try:
            return float(last)
        except (ValueError, TypeError):
            return None

    # ============================================================
    # Closed positions history (v0.1.23) — 거래소 측 거래내역 fetch
    # ============================================================

    async def fetch_closed_positions(
        self,
        since_ms: int | None = None,
        limit: int = 200,
    ) -> list[ClosedPosition]:
        """거래소 청산 포지션 history — Bybit V5 ``/v5/position/closed-pnl`` 우선.

        Bybit V5 제약 (v0.1.27 fix):
            - ``endTime - startTime ≤ 7일`` (한 호출 max 7일 윈도우)
            - ``startTime`` 만 보내면 default 7일치만 반환 (7D=16건/30D=5건/180D=0건 버그 root cause)
            - max ``limit=200`` per page, ``nextPageCursor`` 페이지네이션
            - rate limit: 600 req/5s

        흐름:
            1. ``[since_ms, now]`` 구간을 7일 chunks 로 분할
            2. 각 chunk 마다 cursor 페이지네이션 (200 record 초과 시)
            3. 모든 record 머지 + ``closed_at_ts`` 신→구 재정렬 + ``limit`` 컷

        다른 거래소(OKX/Binance/...): 본 PR 미구현 → 빈 리스트 (TODO ChoYoon).

        paper 모드 = 빈 리스트 (실 호출 X).
        """
        if settings.run_mode == "paper":
            return []
        if self.name != "bybit":
            logger.info(
                "fetch_closed_positions: %s 어댑터 미구현 — 빈 리스트 반환 (TODO)",
                self.name,
            )
            return []
        await self._ensure_init()

        now_ms = int(time.time() * 1000)
        # since_ms 미지정 = 최근 7일 default (Bybit 정책 정합)
        if since_ms is None:
            since_ms = now_ms - 7 * 24 * 60 * 60 * 1000

        seven_days_ms = 7 * 24 * 60 * 60 * 1000
        max_chunks = 30  # 안전 가드 — 30 × 7 = 210일치 (180D 토글 충분 + 여유)

        all_records: list[ClosedPosition] = []
        # 2026-05-27 버그 fix (파트너 보고: "pnl 숫자가 좀 안맞는거 같은데?"):
        # 기존 — chunk_start = since_ms 부터 옛날→최근 방향 순회. limit early-exit
        # 으로 첫(옛날) chunk 가 채워지면 break → 최근 거래가 누락됨.
        # 30D 조회 시 5/8~5/11 만 표시되고 오늘 (5/27) 거래 안 보이는 증상.
        # 수정 — 최근(now) → 옛날(since_ms) 방향으로 뒤집어 순회.
        # limit early-exit 가 "최근 N개 보장" 의미와 정합.
        since_int = int(since_ms)
        chunk_end = now_ms
        chunk_count = 0

        while chunk_end > since_int and chunk_count < max_chunks:
            chunk_start = max(chunk_end - seven_days_ms, since_int)
            cursor: str | None = None
            page_count = 0
            max_pages_per_chunk = 10  # 안전 가드 — 한 chunk 에 2000 record 이면 충분

            while page_count < max_pages_per_chunk:
                params: dict[str, Any] = {
                    "category": "linear",
                    "startTime": chunk_start,
                    "endTime": chunk_end,
                    "limit": min(limit, 200),  # Bybit max 200 per page
                }
                if cursor:
                    params["cursor"] = cursor

                try:
                    raw = await self._ex.private_get_v5_position_closed_pnl(params)
                except (ccxt.NetworkError, ccxt.ExchangeError) as e:
                    logger.warning(
                        "fetch_closed_positions(bybit) chunk %d~%d 실패: %s",
                        chunk_start, chunk_end, e,
                    )
                    break

                result = (raw or {}).get("result") or {}
                items = result.get("list") or []
                for item in items:
                    all_records.append(self._parse_closed_pnl_bybit(item))

                cursor = result.get("nextPageCursor")
                page_count += 1
                if not cursor:
                    break
                if len(all_records) >= limit:
                    break

            chunk_end = chunk_start
            chunk_count += 1
            if len(all_records) >= limit:
                break

        # chunk 간 합치면 신→구 순서 흐트러질 수 있어 재정렬 + limit cut
        all_records.sort(key=lambda r: r.closed_at_ts, reverse=True)
        return all_records[:limit]

    @staticmethod
    def _parse_closed_pnl_bybit(raw: dict[str, Any]) -> ClosedPosition:
        """Bybit V5 closed-pnl record → ``ClosedPosition`` 변환.

        Bybit 응답 키:
            - symbol: "BTCUSDT" (raw, ccxt 표준 X) → ":USDT" suffix 추가
            - side: "Sell" 면 롱 청산 = direction="long", "Buy" 면 숏 청산 = direction="short"
            - leverage: str → int
            - closedSize / avgEntryPrice / avgExitPrice / closedPnl: str → float
            - createdTime / updatedTime: str → int (ms)
        """
        raw_symbol = str(raw.get("symbol") or "")
        # "BTCUSDT" → "BTC/USDT:USDT" (linear perpetual ccxt 표준)
        if raw_symbol.endswith("USDT"):
            base = raw_symbol[:-4]
            symbol = f"{base}/USDT:USDT"
        else:
            symbol = raw_symbol

        side_raw = str(raw.get("side") or "Sell")
        direction: Literal["long", "short"] = "long" if side_raw == "Sell" else "short"

        qty = float(raw.get("closedSize") or 0)
        entry_price = float(raw.get("avgEntryPrice") or 0)
        exit_price = float(raw.get("avgExitPrice") or 0)
        leverage = int(float(raw.get("leverage") or 1))
        pnl_usd = float(raw.get("closedPnl") or 0)

        # ROI% — (pnl / margin) × 100, margin = (entry × qty) / leverage
        margin = (entry_price * qty) / max(leverage, 1)
        roi_pct = (pnl_usd / margin * 100.0) if margin > 0 else 0.0

        return ClosedPosition(
            symbol=symbol,
            direction=direction,
            leverage=leverage,
            qty=qty,
            entry_price=entry_price,
            exit_price=exit_price,
            opened_at_ts=int(float(raw.get("createdTime") or 0)),
            closed_at_ts=int(float(raw.get("updatedTime") or 0)),
            pnl_usd=pnl_usd,
            roi_pct=roi_pct,
        )

    @staticmethod
    def _parse_order(
        raw: dict[str, Any],
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        price: float | None,
    ) -> Order:
        """ccxt order dict → Aurora Order dataclass."""
        return Order(
            order_id=str(raw.get("id") or ""),
            symbol=str(raw.get("symbol") or symbol),
            side=side,
            qty=float(raw.get("amount") or qty),
            price=float(raw["price"]) if raw.get("price") is not None else price,
            status=str(raw.get("status") or ""),
            timestamp_ms=int(raw.get("timestamp") or 0),
        )

    @staticmethod
    def _fake_order(
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        price: float | None,
    ) -> Order:
        """paper 모드용 가짜 Order — 거래소 호출 없이 즉시 'filled' 응답."""
        ts_ms = int(time.time() * 1000)
        return Order(
            order_id=f"paper-{ts_ms}",
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            status="filled",
            timestamp_ms=ts_ms,
        )
