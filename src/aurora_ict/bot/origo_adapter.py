"""Origo 전용 거래소 실행 계약. 담당: Codex.

엄격한 조회/체결/취소 확인은 Origo에서만 사용한다.
공통 CcxtClient와 AuroraClientAdapter의 기존 Cursus 동작은 변경하지 않는다.
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Literal

import ccxt
import pandas as pd
from ccxt.base.errors import AuthenticationError, ExchangeError, InsufficientFunds, PermissionDenied

from aurora.config import settings
from aurora.exchange.base import Order, Position, normalize_side
from aurora.exchange.ccxt_client import CcxtClient, _gen_bot_order_link_id, _is_bot_order
from aurora_ict.bot.aurora_adapter import AuroraClientAdapter

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class OrigoOrder(Order):
    """Origo 주문의 접수 정보와 실제 체결 정보를 분리하여 보존한다."""

    filled_qty: float | None = None
    avg_fill_price: float | None = None
    remaining: float | None = None
    client_order_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(slots=True)
class OrigoPosition(Position):
    """Origo 포지션의 거래소 원본 보호 정보를 보존한다."""

    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _is_bot_entry_order(order: dict[str, Any]) -> bool:
    """보호/청산 주문을 제외한 봇 진입 주문인지 판정한다 (담당: Codex).

    Args:
        order: ccxt 표준 주문.
    Returns:
        봇 태그가 있고 청산/보호 목적이 아닐 때 True.
    Raises:
        없음.
    """
    if not _is_bot_order(order):
        return False
    info = order.get("info") or {}
    for key in ("reduceOnly", "closeOnTrigger"):
        value = order.get(key)
        if value is None:
            value = info.get(key)
        if value not in (None, False, 0, "", "false", "False"):
            return False
    return not info.get("stopOrderType") and not order.get("stopOrderType")


def _optional_order_number(value: Any) -> float | None:
    """체결 응답의 유한한 비음수 숫자만 반환한다.

    Args:
        value: 거래소 원시 값.
    Returns:
        유효한 숫자 또는 미확인 None.
    Raises:
        없음.
    """
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


class OrigoCcxtClient(CcxtClient):
    """Origo만 사용하는 엄격한 주문/포지션/진입 취소 클라이언트."""

    async def fetch_position(self, symbol: str) -> OrigoPosition | None:
        """단일 페어 포지션 조회 — open contract 있으면 반환, 없으면 None.

        paper 모드 = 항상 None (실 호출 X — DESIGN.md §3.2).
        """
        if settings.run_mode == "paper":
            return None
        await self._ensure_init()
        positions = await self._ex.fetch_positions([symbol])
        if not isinstance(positions, list):
            raise ccxt.ExchangeError("포지션 목록 미확인")
        active = []
        for raw in positions:
            if not isinstance(raw, dict) or raw.get("symbol", symbol) != symbol:
                raise ccxt.ExchangeError("포지션 응답 심볼/자료형 미확인")
            quantity = _optional_order_number(raw.get("contracts"))
            if quantity is None:
                raise ccxt.ExchangeError("포지션 응답 수량 미확인")
            if quantity > 0:
                active.append(self._parse_position(raw))
        if len(active) > 1:
            raise ccxt.ExchangeError("복수 포지션 — 원웨이 상태 미확인")
        return active[0] if active else None

    @staticmethod
    def _parse_position(raw: dict[str, Any]) -> OrigoPosition:
        """ccxt position dict → Aurora OrigoPosition dataclass.

        ccxt 표준 필드 매핑 (None 안전 처리):
            - side: "long" / "short"
            - contracts: 수량 (float)
            - entryPrice / leverage / unrealizedPnl
            - marginMode: "isolated" / "cross"
        """
        # #SIDE 2026-08-06: 예전엔 `"short" if side_raw == "short" else "long"` 이라
        # **"sell" 이 오면 숏을 롱으로** 읽었다(청산 방향이 뒤집힐 수 있는 버그).
        # normalize_side 는 long/buy · short/sell 을 모두 인식한다.
        _side = normalize_side(raw.get("side"))
        if _side is None:
            raise ccxt.ExchangeError("포지션 방향 미확인 — 임의 방향으로 관리하지 않음")
        side: Literal["long", "short"] = _side
        margin_raw = raw.get("marginMode", "isolated")
        margin_mode: Literal["isolated", "cross"] = (
            "cross" if margin_raw == "cross" else "isolated"
        )
        return OrigoPosition(
            symbol=str(raw.get("symbol") or ""),
            side=side,
            qty=float(raw.get("contracts") or 0),
            entry_price=float(raw.get("entryPrice") or 0),
            leverage=int(raw.get("leverage") or 1),
            unrealized_pnl=float(raw.get("unrealizedPnl") or 0),
            margin_mode=margin_mode,
            raw=dict(raw),
        )

    async def place_order(
        self,
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        price: float | None = None,
        reduce_only: bool = False,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> OrigoOrder:
        """주문 전송 — ``price=None`` 이면 시장가, 아니면 지정가.

        ``stop_loss`` / ``take_profit`` 가 주어지면 entry 주문에 동봉 (Bybit V5
        ``create_order`` params 의 ``stopLoss`` / ``takeProfit``). 주문 체결 시
        거래소가 포지션에 SL/TP 를 conditional 로 자동 적용 — 지정가 미체결
        주문에도 예약되어 체결 시점에 붙는다. 별도 set_trading_stop 호출 불필요.

        paper 모드 = 가짜 OrigoOrder 반환 (실 호출 X). DESIGN.md §3.2 / E-3.
        """
        if settings.run_mode == "paper":
            return self._fake_order(symbol, side, qty, price)
        await self._ensure_init()
        order_type = "market" if price is None else "limit"
        params: dict[str, Any] = {}
        # 2026-07-22: 봇 주문 식별 태그(orderLinkId). 재부팅 후 소유권 인식 +
        # 유저 수동 주문 보존(선별 취소)용. 모든 봇 주문에 부착.
        params["orderLinkId"] = _gen_bot_order_link_id()
        if reduce_only:
            params["reduceOnly"] = True
        if stop_loss is not None and not reduce_only:
            params["stopLoss"] = str(stop_loss)
            if getattr(self._ex, "id", None) == "bybit":
                params["tpslMode"] = "Full"
                params["slOrderType"] = "Market"
        if take_profit is not None and not reduce_only:
            params["takeProfit"] = str(take_profit)
        try:
            raw = await self._ex.create_order(symbol, order_type, side, qty, price, params)
        except Exception as exc:
            # 응답 유실 후에는 동일 태그로 조회만 한다. 새 UUID로 재주문하지 않는다.
            exc.order_lookup_id = "client:" + params["orderLinkId"]
            if getattr(self._ex, "id", None) == "bybit" and isinstance(exc, InsufficientFunds):
                # ccxt 4.4 create_order는 주문 응답 뒤 private 호출을 하지 않는다.
                # 명시적 110007 거절만 미접수 증거로 인정하고 일반 오류는 보존한다.
                prefix, _, body = str(exc).partition(" ")
                try:
                    response = json.loads(body) if prefix == "bybit" else None
                except (TypeError, ValueError):
                    response = None
                code = response.get("retCode") if isinstance(response, dict) else None
                if type(code) in (int, str) and code in (110007, "110007"):
                    exc.order_definitively_rejected = True
            raise
        return self._parse_order(raw, symbol, side, qty, price)

    async def fetch_order(self, order_id: str, symbol: str) -> OrigoOrder:
        """특정 주문의 접수/체결 상태를 재조회한다 (담당: Codex).

        Args:
            order_id: 거래소 ID 또는 응답 유실 때 저장한 client:AUR... 조회 키.
            symbol: ccxt 통합 심볼.
        Returns:
            실제 주문 응답. 수량/가격 미확인은 None 필드로 보존한다.
        Raises:
            ccxt.BaseError: 조회 실패 또는 주문 상태 미확인.
        """
        if not order_id:
            raise ccxt.OrderNotFound("주문 ID 미확인")
        if settings.run_mode == "paper":
            raise ccxt.OrderNotFound("paper 주문 조회 장부가 없음")
        await self._ensure_init()
        if getattr(self._ex, "id", None) == "bybit":
            market = self._ex.market(symbol)
            key = "orderLinkId" if order_id.startswith("client:") else "orderId"
            lookup = order_id.removeprefix("client:") if key == "orderLinkId" else order_id
            request = {"category": "linear", "symbol": market["id"], key: lookup}
            raw = None
            # 실시간 캐시에서 사라진 종료 주문은 거래 이력으로 다시 확인한다.
            for endpoint in (
                self._ex.private_get_v5_order_realtime,
                self._ex.private_get_v5_order_history,
            ):
                response = await endpoint(request)
                rows, _ = self._bybit_order_page(response)
                matches = [row for row in rows if row.get(key) == lookup]
                if matches:
                    if len(matches) != 1:
                        raise ccxt.ExchangeError("단일 주문 조회 결과가 중복됨")
                    raw = self._ex.parse_order(matches[0], market)
                    break
            if raw is None:
                raise ccxt.OrderNotFound("주문 미확인 — 미접수/종료로 단정하지 않음")
        else:
            if order_id.startswith("client:"):
                raise ccxt.NotSupported("이 거래소의 사용자 주문 ID 조회는 미지원")
            raw = await self._ex.fetch_order(order_id, symbol)
        side = normalize_side(raw.get("side"))
        if side is None or not raw.get("id"):
            raise ccxt.ExchangeError("주문 ID/방향 미확인")
        return self._parse_order(
            raw, symbol, "buy" if side == "long" else "sell",
            _optional_order_number(raw.get("amount")) or 0.0, None,
        )

    @staticmethod
    def _bybit_order_page(response: Any) -> tuple[list[dict[str, Any]], str]:
        """Bybit 주문 조회 응답을 검증하고 페이지를 반환한다.

        Args:
            response: 거래소 JSON 응답.
        Returns:
            주문 행과 다음 페이지 커서.
        Raises:
            ccxt.ExchangeError: 실패 응답 또는 잘못된 구조.
        """
        if (
            not isinstance(response, dict)
            or isinstance(response.get("retCode"), bool)
            or response.get("retCode") not in (0, "0")
        ):
            raise ccxt.ExchangeError("Bybit 주문 조회 응답 실패")
        result = response.get("result")
        rows = result.get("list") if isinstance(result, dict) else None
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ccxt.ExchangeError("Bybit 주문 목록 미확인")
        cursor = result.get("nextPageCursor") or ""
        if not isinstance(cursor, str):
            raise ccxt.ExchangeError("Bybit 주문 페이지 커서 미확인")
        return rows, cursor

    async def fetch_open_entry_orders(self, symbol: str) -> list[dict[str, Any]]:
        """해당 심볼의 봇 진입 주문을 누락 없이 조회한다.

        Args:
            symbol: ccxt 통합 심볼.
        Returns:
            청산/보호 주문을 제외한 봇 진입 대기 주문.
        Raises:
            ccxt.BaseError: 조회 실패, 반복 커서 또는 페이지 상한 초과.
        """
        if settings.run_mode == "paper":
            return []
        await self._ensure_init()
        if getattr(self._ex, "id", None) != "bybit":
            orders = await self._ex.fetch_open_orders(symbol)
        else:
            market = self._ex.market(symbol)
            request: dict[str, Any] = {
                "category": "linear", "symbol": market["id"], "openOnly": 0, "limit": 50,
            }
            orders = []
            cursors: set[str] = set()
            # 거래소 심볼별 상한보다 큰 1,000행. 잘린 목록을 취소 완료로 쓰지 않는다.
            for _ in range(20):
                response = await self._ex.private_get_v5_order_realtime(request)
                rows, cursor = self._bybit_order_page(response)
                orders.extend(self._ex.parse_order(row, market) for row in rows)
                if not cursor:
                    break
                if cursor in cursors:
                    raise ccxt.ExchangeError("진입 주문 목록 페이지가 반복됨")
                cursors.add(cursor)
                request["cursor"] = cursor
            else:
                raise ccxt.ExchangeError("진입 주문 목록 페이지 상한 초과")
        if not isinstance(orders, list) or any(not isinstance(o, dict) for o in orders):
            raise ccxt.ExchangeError("진입 주문 목록 미확인")
        return [order for order in orders if _is_bot_entry_order(order)]

    async def cancel_bot_orders(self, symbol: str) -> int:
        """봇 진입 주문만 취소하고 실제 대기 잔량이 없음을 확인한다.

        cancel_all(전체 취소)과 달리 미체결 주문을 조회해 봇 태그가 붙은 것만
        개별 취소. 재시작 고아 주문 청소가 유저 수동 지정가를 지우지 않게 한다.
        paper 모드는 noop.

        Returns:
            취소 접수 수. 0도 대기 주문이 없다고 확인한 결과다.
        Raises:
            ccxt.BaseError: 취소/조회 실패 또는 취소 접수 후에도 대기 주문이 남음.
        """
        if settings.run_mode == "paper":
            return 0
        await self._ensure_init()
        orders = await self.fetch_open_entry_orders(symbol)
        n = 0
        for o in orders:
            if not o.get("id"):
                raise ccxt.ExchangeError("취소할 진입 주문 ID 미확인")
            try:
                await self._ex.cancel_order(o["id"], symbol)
                n += 1
            except ccxt.OrderNotFound:
                # 취소와 체결 경합은 최종 대기 주문 조회로 판단한다.
                continue
        if await self.fetch_open_entry_orders(symbol):
            raise ccxt.ExchangeError("봇 진입 주문 취소 미확인 — 대기 상태 보존")
        return n

    @staticmethod
    def _parse_order(
        raw: dict[str, Any],
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        price: float | None,
    ) -> OrigoOrder:
        """ccxt order dict → Aurora OrigoOrder dataclass."""
        return OrigoOrder(
            order_id=str(raw.get("id") or ""),
            symbol=str(raw.get("symbol") or symbol),
            side=side,
            qty=float(raw.get("amount") or qty),
            price=float(raw["price"]) if raw.get("price") is not None else price,
            status=str(raw.get("status") or ""),
            timestamp_ms=int(raw.get("timestamp") or 0),
            filled_qty=_optional_order_number(raw.get("filled")),
            avg_fill_price=_optional_order_number(raw.get("average")),
            remaining=_optional_order_number(raw.get("remaining")),
            client_order_id=raw.get("clientOrderId"),
            raw=dict(raw),
        )

    @staticmethod
    def _fake_order(
        symbol: str,
        side: Literal["buy", "sell"],
        qty: float,
        price: float | None,
    ) -> OrigoOrder:
        """paper 모드용 가짜 OrigoOrder — 거래소 호출 없이 즉시 'filled' 응답."""
        ts_ms = int(time.time() * 1000)
        return OrigoOrder(
            order_id=f"paper-{ts_ms}",
            symbol=symbol,
            side=side,
            qty=qty,
            price=price,
            status="filled",
            timestamp_ms=ts_ms,
            filled_qty=qty,
            avg_fill_price=price,
            remaining=0.0,
        )


class OrigoClientAdapter(AuroraClientAdapter):
    """Origo 전용 응답 정규화와 실패 전파. Cursus는 기존 어댑터를 유지한다."""

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int,
    ) -> list[list[Any]]:
        """Aurora의 DataFrame 반환을 ccxt raw rows로 변환.

        Aurora ``fetch_ohlcv``는 DataFrame을 반환하므로
        [ts_ms, o, h, l, c, v] 리스트 형태로 변환해서 돌려준다.
        """
        try:
            df = await self._client.fetch_ohlcv(
                symbol, self._aurora_tf(timeframe), limit,
            )
        except AuthenticationError as e:
            self._note_auth_fail("fetch_ohlcv", e)
            raise
        # 공개 시세 조회 성공은 비공개 API 키의 유효성을 증명하지 않는다.
        if not isinstance(df, pd.DataFrame) or df.empty:
            return []
        rows: list[list[Any]] = []
        # df.index가 DatetimeIndex면 ms로 변환
        if isinstance(df.index, pd.DatetimeIndex):
            ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
        else:
            ts_arr = df.index.to_numpy()
        opens = df["open"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        closes = df["close"].to_numpy()
        vols = df["volume"].to_numpy() if "volume" in df.columns else [0.0] * len(df)
        for i in range(len(df)):
            rows.append([
                int(ts_arr[i]),
                float(opens[i]),
                float(highs[i]),
                float(lows[i]),
                float(closes[i]),
                float(vols[i]),
            ])
        return rows

    async def place_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float | None = None,
        reduce_only: bool = False,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        """Aurora place_order 결과를 dict로 변환 + SL/TP 동봉.

        Aurora ``place_order`` 시그니처:
        ``(symbol, side, qty, price=None, reduce_only=False, stop_loss, take_profit)``.

        v0.4.73 — SL/TP 동봉 방식 전환 (#LIVE-1 fix):
        - entry 주문에 ``stop_loss`` / ``take_profit`` 를 그대로 본체 place_order 로
          넘김 → Bybit V5 ``create_order`` params 의 ``stopLoss`` / ``takeProfit`` 동봉.
          체결 시 거래소가 포지션에 conditional SL/TP 자동 적용.
        - 이전 ``set_trading_stop`` 별도 호출 (포지션 API) 은 지정가 미체결 주문에
          안 박히고, TP 가 별도 reduce_only limit 으로만 박혀 포지션 카드에 안 보이던
          문제 (#LIVE-1) 를 해소. 동봉은 지정가 미체결 주문에도 예약된다.
        - reduce_only=True (청산 주문) 는 SL/TP 인자 무시.
        """
        try:
            order = await self._client.place_order(
                symbol=symbol,
                side=side,
                qty=qty,
                price=price,
                reduce_only=reduce_only,
                stop_loss=None if reduce_only else stop_loss,
                take_profit=None if reduce_only else take_profit,
            )
            self._note_write_ok()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "10005" in msg or isinstance(e, PermissionDenied):
                # 다른 주문/수동 매매의 수량 변화는 이 주문의 성공 근거가 아니다.
                lookup = getattr(e, "order_lookup_id", None)
                recovered = None
                if isinstance(lookup, str) and lookup:
                    try:
                        recovered = await self.fetch_order(lookup, symbol)
                    except Exception:  # 조회 불능은 원래 권한 오류와 함께 보류한다.
                        pass
                if recovered and recovered.get("status") in ("open", "closed", "canceled", "expired"):
                    self._note_write_ok()
                    return recovered
                self._note_write_fail("place_order", e)
                denied = PermissionDenied("주문 권한 거부(10005) — 동일 주문 접수/체결 미확인")
                if isinstance(lookup, str):
                    denied.order_lookup_id = lookup
                raise denied from e
            if isinstance(e, AuthenticationError):
                self._note_auth_fail("place_order", e)
            raise
        return self._order_dict(order)

    @staticmethod
    def _order_dict(order: Any) -> dict[str, Any]:
        """slots 자료형과 거래소 응답을 체결 정보 손실 없이 정규화한다.

        Args:
            order: 실제 Order 또는 ccxt 주문 dict.
        Returns:
            주문 ID, 접수 상태, 실제 체결 수량/평균가를 보존한 dict.
        Raises:
            ExchangeError: 구조 미확인. 요청 가격/수량을 체결로 추정하지 않는다.
        """
        if is_dataclass(order) and not isinstance(order, type):
            data = asdict(order)
        elif isinstance(order, dict):
            data = dict(order)
        elif hasattr(order, "__dict__"):
            data = dict(vars(order))
        else:
            raise ExchangeError("주문 응답 자료형 미확인")
        raw = data.pop("raw", None)
        if isinstance(raw, dict):
            data = {**raw, **data}
        order_id = data.get("id") or data.get("order_id") or data.get("orderId")
        data["id"] = data["order_id"] = str(order_id) if order_id else None
        for names in (
            ("filled_qty", "filled"), ("avg_fill_price", "average", "avgPrice"),
            ("qty", "amount"), ("remaining",),
        ):
            number = next((
                parsed for key in names
                if (parsed := _optional_order_number(data.get(key))) is not None
            ), None)
            if names[0] == "avg_fill_price" and number == 0:
                number = None
            for key in names:
                data[key] = number
        data["status"] = str(data.get("status") or "").lower()
        return data

    async def fetch_order(self, order_id: str, symbol: str) -> dict[str, Any]:
        """요청한 단일 주문의 실제 상태를 재조회한다.

        Args:
            order_id: 거래소 ID 또는 client:AUR... 조회 키.
            symbol: ccxt 통합 심볼.
        Returns:
            정규화한 주문. 조회 실패는 빈 주문으로 바꾸지 않는다.
        Raises:
            Exception: 인증/네트워크/주문 조회 실패.
        """
        try:
            result = self._order_dict(await self._client.fetch_order(order_id, symbol))
        except AuthenticationError as exc:
            self._note_auth_fail("fetch_order", exc)
            raise
        if not result.get("id"):
            raise ExchangeError("조회한 주문 ID 미확인")
        self._note_auth_ok()
        return result

    async def fetch_position(self, symbol: str) -> dict[str, Any] | None:
        """포지션 조회. Aurora client → fallback: ccxt _ex 직접 호출.

        Aurora 본진 CcxtClient.fetch_position 이 None 반환하는 경우가 있어 (v0.4.55)
        ccxt _ex.fetch_positions([symbol]) 로 직접 fetch fallback. Bybit V5 가 빈
        포지션도 contracts=0 으로 반환하므로 0인 항목은 제외.
        """
        primary_error: Exception | None = None
        try:
            pos = await self._client.fetch_position(symbol)
        except AuthenticationError as e:
            # #KEY-EXPIRED: 인증 실패를 None 으로 돌려주면 봇이 '포지션 없음'으로
            # 읽는다. 그건 사실이 아니라 '모름'이다 — 예외로 올려 호출부가 보류하게.
            self._note_auth_fail("fetch_position", e)
            raise
        except Exception as e:  # noqa: BLE001
            self._wlog("Aurora fetch_position 실패: %s — ccxt fallback", e)
            primary_error = e
            pos = None
        if pos is not None:
            result = self._position_dict(pos, symbol)
            self._note_auth_ok()
            if result is not None:
                return result

        # 2차 fallback: ccxt _ex 직접 fetch_positions
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            if primary_error is not None:
                raise primary_error
            return None
        await self._ensure_time_sync()
        try:
            positions = await ex.fetch_positions([symbol])
        except AuthenticationError as e:
            self._note_auth_fail("fetch_positions", e)
            raise
        except Exception as e:  # noqa: BLE001
            self._wlog("ccxt fetch_positions 실패: %s", e)
            raise
        if not isinstance(positions, list):
            raise ExchangeError("포지션 목록 미확인")
        active = [
            result for position in positions
            if (result := self._position_dict(position, symbol)) is not None
        ]
        if len(active) > 1:
            raise ExchangeError("단일 심볼 포지션이 복수임 — 원웨이 상태 미확인")
        self._note_auth_ok()
        return active[0] if active else None

    @staticmethod
    def _position_dict(position: Any, symbol: str) -> dict[str, Any] | None:
        """조회 성공 포지션을 검증하고 보호 정보까지 보존한다.

        Args:
            position: Position 자료형 또는 ccxt dict.
            symbol: 요청 심볼.
        Returns:
            검증한 포지션. 수량 0이 명시된 때만 None.
        Raises:
            ExchangeError: 심볼/방향/수량이 불명확한 응답.
        """
        if is_dataclass(position) and not isinstance(position, type):
            data = asdict(position)
        elif isinstance(position, dict):
            data = dict(position)
        elif hasattr(position, "__dict__"):
            data = dict(vars(position))
        else:
            raise ExchangeError("포지션 응답 자료형 미확인")
        raw = data.pop("raw", None)
        if isinstance(raw, dict):
            data = {**raw, **data}
        if data.get("symbol") and data["symbol"] != symbol:
            raise ExchangeError("요청한 심볼과 포지션 응답이 다름")
        quantity = data.get("contracts")
        if quantity is None:
            quantity = data.get("qty")
        quantity = _optional_order_number(quantity)
        if quantity is None:
            raise ExchangeError("포지션 수량 미확인")
        if quantity == 0:
            return None
        side = normalize_side(data.get("side"))
        if side is None:
            raise ExchangeError("포지션 방향 미확인")
        data["contracts"] = data["qty"] = quantity
        data["side"] = side
        if data.get("entryPrice") is None:
            data["entryPrice"] = data.get("entry_price")
        return data

    async def cancel_bot_orders(self, symbol: str) -> int:
        """봇 태그(orderLinkId) 붙은 미체결 주문만 취소 — 유저 수동 주문 보존.

        Args:
            symbol: ccxt 통합 심볼.
        Returns:
            대기 진입 주문이 없다고 확인한 뒤 취소 접수 수.
        Raises:
            Exception: 미지원/취소/조회 실패. 0으로 숨기지 않는다.
        """
        try:
            count = await self._client.cancel_bot_orders(symbol)
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ExchangeError("진입 주문 취소 결과 미확인")
            self._note_auth_ok()
            return count
        except Exception as e:  # noqa: BLE001
            if isinstance(e, AuthenticationError):
                self._note_auth_fail("cancel_bot_orders", e)
            if isinstance(e, PermissionDenied) or "10005" in str(e):
                self._note_write_fail("cancel_bot_orders", e)
            self._wlog("cancel_bot_orders 실패: %s", e)
            raise

    async def validate_trading_credentials(self) -> bool:
        """인증 정지 해제 전에 키의 선물 주문/포지션 쓰기 권한을 확인한다.

        Args:
            없음.
        Returns:
            비공개 읽기 API로 필수 권한 확인 후 True. 키/UID는 반환하지 않는다.
        Raises:
            AuthenticationError: 키 조회 실패.
            PermissionDenied: 읽기 전용 또는 선물 주문/포지션 권한 누락.
            ExchangeError: 응답 미확인 또는 미지원 거래소.
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None or getattr(ex, "id", None) != "bybit":
            raise ExchangeError("Bybit 거래 권한 확인 클라이언트 없음")
        await self._ensure_time_sync()
        try:
            response = await ex.private_get_v5_user_query_api()
        except AuthenticationError as exc:
            self._note_auth_fail("validate_trading_credentials", exc)
            raise
        if (
            not isinstance(response, dict)
            or isinstance(response.get("retCode"), bool)
            or response.get("retCode") not in (0, "0")
        ):
            raise AuthenticationError("API 키 정보 조회 실패")
        result = response.get("result")
        if not isinstance(result, dict):
            raise ExchangeError("API 키 권한 정보 미확인")
        read_only = result.get("readOnly")
        permissions = result.get("permissions")
        contract = permissions.get("ContractTrade") if isinstance(permissions, dict) else None
        if (
            isinstance(read_only, bool) or read_only not in (0, "0")
            or not isinstance(contract, list)
            or not {"Order", "Position"}.issubset(contract)
        ):
            raise PermissionDenied("Read-Write 및 Contract Orders/Positions 권한 필요")
        self._note_auth_ok()
        return True
