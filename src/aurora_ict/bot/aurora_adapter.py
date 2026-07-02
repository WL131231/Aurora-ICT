"""AuroraClientAdapter — Aurora 측 CcxtClient를 ExchangeClientProtocol에 맞추는 어댑터.

Aurora 측 ``CcxtClient``가 노출하는 메서드:
- ``fetch_ohlcv(symbol, timeframe, limit) -> DataFrame``
- ``fetch_position(symbol) -> Position | None`` (dataclass)
- ``place_order(symbol, side, qty, price=None, ...) -> Order`` (dataclass)

Aurora-ICT가 기대하는 ``ExchangeClientProtocol``:
- ``fetch_ohlcv(symbol, timeframe, limit) -> list[list[Any]]`` (raw ccxt rows)
- ``fetch_position(symbol) -> dict | None``
- ``place_order(symbol, side, qty, ...) -> dict``
- ``fetch_balance() -> dict`` (ccxt 표준 포맷)

두 인터페이스 사이의 형식 차이를 흡수하는 thin adapter. Aurora-ICT가 Aurora 본체에
직접 의존하지 않도록 분리한다.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
from ccxt.base.errors import AuthenticationError

logger = logging.getLogger(__name__)


class AuroraClientAdapter:
    """Aurora ``CcxtClient``를 Aurora-ICT 인터페이스로 변환.

    Args:
        ccxt_client: Aurora ``CcxtClient`` instance (또는 duck-typed 호환 객체).
    """

    # Aurora 클라이언트는 1h+ timeframe 을 대문자로만 인식 (1H/2H/4H/1D/1W).
    # 우리 UI / settings 는 소문자 사용 (ccxt 표준). 호출 시 변환.
    _TF_AURORA_MAP = {
        "1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
        "1h": "1H", "2h": "2H", "4h": "4H", "6h": "6H", "12h": "12H",
        "1d": "1D", "1w": "1W",
    }

    def __init__(self, ccxt_client: Any) -> None:
        self._client = ccxt_client
        # ccxt 의 시간 동기화 옵션 설정. Bybit V5 private API 가 timestamp 1초 이상
        # 차이 박으면 retCode 10002 거부 → 모든 호출 시 자동 보정.
        ex = getattr(ccxt_client, "_ex", None)
        if ex is not None:
            try:
                if hasattr(ex, "options") and isinstance(ex.options, dict):
                    ex.options["adjustForTimeDifference"] = True
                    ex.options["recvWindow"] = 60000
            except Exception:  # noqa: BLE001
                pass
        self._time_diff_loaded = False
        # 거래소 호출 실패 로그에 어느 사용자/심볼인지 식별용 라벨 (멀티유저 운영).
        # 비어있으면 기존과 동일하게 동작 — multi_user_manager 가 start 시 주입.
        self._log_label = ""
        # 거래소 인증 실패(키 무효 10003) 연속 횟수 — fetch_balance 성공 시 0 리셋.
        # 봇 _run_loop 가 임계치 도달 시 자동 정지에 사용(잔고 조회조차 안 되는 키).
        self.auth_fail_streak = 0

    def set_log_label(self, label: str) -> None:
        """거래소 호출 실패 WARNING 에 붙일 식별 라벨 설정 (예: 'AICT-XXXX/BTC').

        멀티유저 환경에서 retCode 10003(키 무효) 등 발생 시 어느 사용자인지
        로그만으로 특정하기 위함.

        Args:
            label: 사용자 코드/심볼 등 식별 문자열. None/빈 문자열이면 prefix 미부착.
        """
        self._log_label = label or ""

    def _wlog(self, msg: str, *args: Any) -> None:
        """라벨 prefix 를 붙여 WARNING 로깅. 라벨 없으면 기존과 동일.

        Args:
            msg: %-스타일 포맷 문자열.
            args: 포맷 인자.
        """
        if self._log_label:
            logger.warning("[%s] " + msg, self._log_label, *args)
        else:
            logger.warning(msg, *args)

    async def _ensure_time_sync(self) -> None:
        """Bybit 서버 시간과 PC 시간 차이 한 번 로드 (lazy)."""
        if self._time_diff_loaded:
            return
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            self._time_diff_loaded = True
            return
        try:
            if hasattr(ex, "load_time_difference"):
                await ex.load_time_difference()
        except Exception as e:  # noqa: BLE001
            self._wlog("load_time_difference 실패: %s", e)
        finally:
            self._time_diff_loaded = True

    def _aurora_tf(self, tf: str) -> str:
        """소문자 timeframe → Aurora 대문자 포맷. 미매핑 시 원본 그대로."""
        return self._TF_AURORA_MAP.get(tf, tf)

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int,
    ) -> list[list[Any]]:
        """Aurora의 DataFrame 반환을 ccxt raw rows로 변환.

        Aurora ``fetch_ohlcv``는 DataFrame을 반환하므로
        [ts_ms, o, h, l, c, v] 리스트 형태로 변환해서 돌려준다.
        """
        df = await self._client.fetch_ohlcv(symbol, self._aurora_tf(timeframe), limit)
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
        # #LEV-8 (2026-06-02): Bybit demo + ccxt 조합에서 create_order 가 raw
        # API 호출 후 query-api (/user/v3/private/query-api) UTA 체크 응답을
        # 받고 exception throw — set_leverage 와 같은 false positive.
        # 실제로는 Bybit 측에 주문이 들어가 limit pending 으로 살아있음
        # (실측 2026-06-02 KST 23:30 BTC short 1.3184 — UI pending 표시 확인).
        # ccxt exception 이 retCode 10005 + query-api 메시지면 fetch_position
        # 으로 실제 진입 확인 → 활성 포지션 또는 pending order 있으면 success
        # 처리. 없으면 진짜 실패.
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
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # #LEV-8b: query-api 메시지 없이 단순 retCode 10005 만 응답하는
            # 케이스도 false positive (실측 2026-06-03 KST 00:02 — Bybit limit
            # 들어간 상태에서 ERROR). retCode 10005 모두 false positive 처리,
            # 다음 sync_position_state 가 거래소 상태와 reconcile.
            if "10005" in msg:
                self._wlog(
                    "place_order (%s %s qty=%s): ccxt UTA 체크 false positive "
                    "(retCode 10005) — Bybit 측 실제 처리 확인 중...",
                    symbol, side, qty,
                )
                # 실제 들어갔는지 확인 — fetch_position 으로.
                pos = None
                try:
                    pos = await self.fetch_position(symbol)
                except Exception as pe:  # noqa: BLE001
                    self._wlog(
                        "place_order false positive 검증용 fetch_position 실패: %s",
                        pe,
                    )
                if pos is not None and float(pos.get("contracts") or 0) > 0:
                    self._wlog(
                        "place_order false positive 확인 — 포지션 활성. success 처리.",
                    )
                    return {
                        "id": None,
                        "symbol": symbol,
                        "side": side,
                        "qty": qty,
                        "price": price,
                        "status": "open_uta_false_positive",
                        "info": {"retCode": 10005, "ccxt_false_positive": True},
                    }
                # 포지션 없음 — pending limit 가능성 또는 진짜 실패. UI 의
                # pending 표시 (별도 `_pending_entry` 로직) 가 fetch_open_orders
                # 로 확인하므로 여기선 일단 success 응답으로 처리해 봇의 active
                # state 박음. 만약 진짜 실패면 다음 sync 에서 자동 해제.
                self._wlog(
                    "place_order false positive — 포지션 미확인 but Bybit 측 "
                    "limit 박힌 가능성 (pending). success 처리 + 다음 sync 검증.",
                )
                return {
                    "id": None,
                    "symbol": symbol,
                    "side": side,
                    "qty": qty,
                    "price": price,
                    "status": "open_uta_false_positive",
                    "info": {"retCode": 10005, "ccxt_false_positive": True},
                }
            raise
        order_dict: dict[str, Any]
        if hasattr(order, "__dict__"):
            order_dict = dict(order.__dict__)
        elif isinstance(order, dict):
            order_dict = order
        else:
            order_dict = {"raw": str(order)}

        # Entry 주문 (reduce_only False) — 실제 fill 가격 / qty 를 거래소 응답에서 추출.
        # 시장가(price=None) 진입 시 setup 가격과 실제 체결가가 다름 (slippage / spread).
        if not reduce_only:
            avg_fill, filled_qty = await self._extract_fill(order_dict, symbol)
            if avg_fill is not None:
                order_dict["avg_fill_price"] = avg_fill
            if filled_qty is not None:
                order_dict["filled_qty"] = filled_qty

        return order_dict

    async def _extract_fill(
        self, order_dict: dict[str, Any], symbol: str,
    ) -> tuple[float | None, float | None]:
        """주문 응답에서 평균 체결가 / 체결 qty 추출. 없으면 fetch_position fallback."""
        # 1차: order_dict 안 ccxt 표준 필드.
        avg_keys = ("average", "avgPrice", "avg_price", "fill_price", "price")
        qty_keys = ("filled", "filled_qty", "amount", "qty")
        avg: float | None = None
        filled: float | None = None
        for k in avg_keys:
            v = order_dict.get(k)
            if isinstance(v, (int, float)) and v > 0:
                avg = float(v)
                break
        for k in qty_keys:
            v = order_dict.get(k)
            if isinstance(v, (int, float)) and v > 0:
                filled = float(v)
                break
        if avg is not None:
            return avg, filled
        # 2차 fallback: 거래소 fetch_position 으로 entryPrice 조회 (시장가 즉시 fill).
        try:
            pos = await self.fetch_position(symbol)
        except Exception as e:  # noqa: BLE001
            logger.debug("fill 가격 fallback fetch_position 실패: %s", e)
            return None, filled
        if pos is None:
            return None, filled
        ep = pos.get("entryPrice") or pos.get("entry_price") or pos.get("averagePrice")
        if isinstance(ep, (int, float)) and ep > 0:
            avg = float(ep)
        if filled is None:
            c = pos.get("contracts") or pos.get("qty")
            if isinstance(c, (int, float)) and c > 0:
                filled = float(c)
        return avg, filled

    async def fetch_position(self, symbol: str) -> dict[str, Any] | None:
        """포지션 조회. Aurora client → fallback: ccxt _ex 직접 호출.

        Aurora 본진 CcxtClient.fetch_position 이 None 반환하는 경우가 있어 (v0.4.55)
        ccxt _ex.fetch_positions([symbol]) 로 직접 fetch fallback. Bybit V5 가 빈
        포지션도 contracts=0 으로 반환하므로 0인 항목은 제외.
        """
        # 1차: Aurora client.fetch_position
        try:
            pos = await self._client.fetch_position(symbol)
        except Exception as e:  # noqa: BLE001
            self._wlog("Aurora fetch_position 실패: %s — ccxt fallback", e)
            pos = None
        if pos is not None:
            if hasattr(pos, "__dict__"):
                d = dict(pos.__dict__)
                if "qty" in d and "contracts" not in d:
                    d["contracts"] = d["qty"]
                if float(d.get("contracts") or 0) > 0:
                    return d
            elif isinstance(pos, dict):
                if float(pos.get("contracts") or pos.get("qty") or 0) > 0:
                    return pos

        # 2차 fallback: ccxt _ex 직접 fetch_positions
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            return None
        await self._ensure_time_sync()
        try:
            positions = await ex.fetch_positions([symbol])
        except Exception as e:  # noqa: BLE001
            self._wlog("ccxt fetch_positions 실패: %s", e)
            return None
        for p in positions or []:
            contracts = float(p.get("contracts") or 0)
            if contracts > 0:
                # ccxt 표준 dict 그대로 반환 (entryPrice / side / contracts 박힘).
                if "qty" not in p:
                    p["qty"] = contracts
                return p
        return None

    async def fetch_all_positions(self) -> list[dict[str, Any]]:
        """계정 전체 열린 포지션 조회 — admin 전체 포지션 미추적 스캔용 (2026-06-12).

        봇이 추적하지 않는 포지션(수동 진입·재기동 누락 고아)도 보이게 USDT
        선물 계정 전체를 조회한다. 실패는 빈 리스트 (조회 실패가 admin 화면을
        막지 않게).

        Returns:
            계약 수 > 0 인 ccxt 표준 포지션 dict 리스트.
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            return []
        await self._ensure_time_sync()
        try:
            # Bybit V5 는 symbols=None 일 때 settleCoin 필수.
            positions = await ex.fetch_positions(None, {"settleCoin": "USDT"})
        except Exception as e:  # noqa: BLE001
            self._wlog("fetch_all_positions 실패: %s", e)
            return []
        out: list[dict[str, Any]] = []
        for p in positions or []:
            if float(p.get("contracts") or 0) > 0:
                if "qty" not in p:
                    p["qty"] = float(p.get("contracts") or 0)
                out.append(p)
        return out

    async def fetch_actual_leverage(self, symbol: str) -> int | None:
        """거래소 측 현재 leverage 조회 (포지션 무관).

        set_leverage 실패 시 fallback 으로 호출. ccxt fetch_positions 가 빈
        포지션도 leverage 필드 포함해 반환 (Bybit V5 position/list endpoint).

        #LEV-5: ccxt 가 같은 symbol 의 cross+isolated 양쪽 또는 long/short
        side 별로 여러 position dict 반환 → 첫 번째 hit 가 잘못된 항목일 수
        있음. 명시적 symbol 매칭 + 디버그 로그 추가.

        Args:
            symbol: ccxt unified symbol (예: "BTC/USDT:USDT").

        Returns:
            정수 leverage. 조회 실패 시 None.
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            return None
        try:
            await self._ensure_time_sync()
            positions = await ex.fetch_positions([symbol])
        except Exception as e:  # noqa: BLE001
            self._wlog("fetch_actual_leverage 실패 (%s): %s", symbol, e)
            return None
        # 디버그 — 응답의 모든 position 항목 가시화. WARNING level (#LEV-5)
        # 파트너 fly redirect filter 가 INFO 안 잡아 임시 WARNING. 디버그 끝나면
        # info 로 되돌리거나 제거.
        for idx, p in enumerate(positions or []):
            info = p.get("info") or {}
            self._wlog(
                "LEV_DEBUG[%s] #%d: sym=%s side=%s "
                "lev=%s marginMode=%s contracts=%s | info: lev=%s "
                "leverage_buy=%s leverage_sell=%s tradeMode=%s",
                symbol, idx, p.get("symbol"), p.get("side"),
                p.get("leverage"), p.get("marginMode"), p.get("contracts"),
                info.get("leverage"), info.get("buyLeverage"),
                info.get("sellLeverage"), info.get("tradeMode"),
            )
        # 명시적 symbol 매칭 — 다른 심볼 무시.
        for p in positions or []:
            if p.get("symbol") != symbol:
                continue
            # info dict 의 buyLeverage 가 가장 신뢰할 수 있는 raw 값 (Bybit V5).
            info = p.get("info") or {}
            for field in ("buyLeverage", "leverage"):
                raw = info.get(field)
                if raw is None:
                    continue
                try:
                    return int(float(raw))
                except (TypeError, ValueError):
                    continue
            # ccxt 표준 leverage fallback.
            lev = p.get("leverage")
            if lev is not None:
                try:
                    return int(float(lev))
                except (TypeError, ValueError):
                    continue
        return None

    async def fetch_balance(self) -> dict[str, Any]:
        """ccxt fetch_balance를 그대로 호출.

        Aurora 측 client는 별도 fetch_balance를 노출하지 않으므로 내부 ``_ex``
        (ccxt async exchange)에 직접 위임한다.
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            self._wlog("Aurora client에 _ex 속성 없음 — fallback {}")
            return {}
        try:
            result = await ex.fetch_balance()
            self.auth_fail_streak = 0  # 잔고 조회 성공 — 인증 정상, 카운터 리셋
            return result
        except AuthenticationError as e:
            # 키 무효(10003) — 별도 카운트. 봇 _run_loop 가 임계치 도달 시 자동 정지.
            self.auth_fail_streak += 1
            self._wlog(
                "fetch_balance 인증 실패(키 무효) %d회: %s",
                self.auth_fail_streak, e,
            )
            return {}
        except Exception as e:  # noqa: BLE001
            self._wlog("fetch_balance 실패: %s", e)
            return {}

    async def fetch_ticker(self, symbol: str) -> float | None:
        """현재 시장가 (last price) — marketable limit entry 가격 계산용.

        Aurora client.fetch_ticker 위임. 실패 시 None (호출처가 fallback).
        """
        try:
            return await self._client.fetch_ticker(symbol)
        except Exception as e:  # noqa: BLE001
            self._wlog("fetch_ticker 실패: %s", e)
            return None

    async def fetch_symbol_meta(self, symbol: str) -> dict[str, float | None]:
        """심볼별 거래소 메타(min_qty / qty_step / max_leverage) 위임.

        페어 확장 시 봇이 가동 직후 자기 심볼 메타를 캐시해 사이징·precision 에
        쓴다. 실패 시 각 값 None (호출처 안전 폴백).
        """
        try:
            return await self._client.fetch_symbol_meta(symbol)
        except Exception as e:  # noqa: BLE001
            self._wlog("fetch_symbol_meta 실패 (%s): %s", symbol, e)
            return {"min_qty": None, "qty_step": None, "max_leverage": None}

    def round_amount(self, symbol: str, amount: float) -> float:
        """거래소 lot step 에 맞춘 qty 정렬 위임. 실패 시 원본 반환(안전 폴백)."""
        try:
            return self._client.round_amount(symbol, amount)
        except Exception as e:  # noqa: BLE001
            logger.debug("round_amount 폴백 (%s): %s", symbol, e)
            return amount

    async def list_top_usdt_perps(self, limit: int = 30) -> list[str]:
        """거래대금 상위 USDT perp 심볼 목록 위임. 실패 시 빈 리스트."""
        try:
            return await self._client.list_top_usdt_perps(limit)
        except Exception as e:  # noqa: BLE001
            self._wlog("list_top_usdt_perps 위임 실패: %s", e)
            return []

    async def fetch_perp_tickers(self, limit: int = 30) -> list[dict]:
        """거래대금 상위 USDT perp 시세 행(symbol/last/pct24h/volume) 위임."""
        try:
            return await self._client.fetch_perp_tickers(limit)
        except Exception as e:  # noqa: BLE001
            self._wlog("fetch_perp_tickers 위임 실패: %s", e)
            return []

    async def cancel_all_orders(self, symbol: str) -> None:
        """해당 페어 미체결 주문 전체 취소 — pending limit entry TTL 만료 시 사용.

        Aurora client.cancel_all 위임. 미체결 entry limit 만 대상 (체결된 포지션의
        conditional SL/TP 는 주문이 아니라 포지션 속성이라 영향 없음).
        """
        try:
            await self._client.cancel_all(symbol)
        except Exception as e:  # noqa: BLE001
            self._wlog("cancel_all_orders 실패: %s", e)

    async def fetch_closed_positions(
        self, since_ms: int | None = None, limit: int = 200,
    ) -> list[Any]:
        """거래소 청산 history (Bybit V5 closed-pnl) — today 실현손익 거래소 동기화용.

        Aurora client.fetch_closed_positions 위임 (ClosedPosition list, pnl_usd 포함).
        실패 시 빈 list (호출처가 기존 값 유지).
        """
        try:
            return await self._client.fetch_closed_positions(
                since_ms=since_ms, limit=limit,
            )
        except Exception as e:  # noqa: BLE001
            self._wlog("fetch_closed_positions 실패: %s", e)
            return []

    async def set_leverage(
        self, symbol: str, leverage: int,
    ) -> dict[str, Any]:
        """Bybit V5 set_leverage 호출 — buy/sell leverage 동시 설정.

        Args:
            symbol: ccxt unified symbol (e.g. "BTC/USDT:USDT").
            leverage: 정수 leverage (1 ~ 50).

        Returns:
            Bybit API 응답 dict. 실패 시 빈 dict.
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            self._wlog("set_leverage: _ex 없음 — skip")
            return {}
        await self._ensure_time_sync()
        raw_symbol = symbol.replace("/", "").split(":")[0]
        params = {
            "category": "linear",
            "symbol": raw_symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        try:
            result = await ex.private_post_v5_position_set_leverage(params)
            logger.info(
                "set_leverage 완료 (%s → %dx)", raw_symbol, leverage,
            )
            return dict(result) if isinstance(result, dict) else {"raw": str(result)}
        except Exception as e:  # noqa: BLE001
            # Bybit retCode 110043: "leverage not modified" — 이미 같은 값 설정.
            # #LEV-6: Bybit demo + ccxt 조합에서 retCode 10005 (query-api 권한
            # 거부) 가 false positive 로 발생. 실제로는 set_leverage 가 Bybit
            # 측에서 성공 처리됨 (거래소 화면 변경 확인). ccxt 의 UTA 체크
            # endpoint 권한 거부는 실제 set_leverage 결과와 무관 → success 처리.
            msg = str(e)
            if "110043" in msg or "not modified" in msg:
                logger.info(
                    "set_leverage skip (%s 이미 %dx)", raw_symbol, leverage,
                )
                return {"retCode": 110043, "alreadySet": True}
            if "10005" in msg and "query-api" in msg:
                self._wlog(
                    "set_leverage (%s → %dx): ccxt UTA 체크 false positive "
                    "(retCode 10005) — Bybit 측 실제로는 성공 처리됨.",
                    raw_symbol, leverage,
                )
                return {"retCode": 0, "alreadySet": False}
            self._wlog(
                "set_leverage 실패 (%s %dx): %s",
                raw_symbol, leverage, e,
            )
            return {}

    async def ensure_oneway_mode(self, symbol: str) -> dict[str, Any]:
        """Bybit V5 switch-mode 로 해당 심볼을 원웨이(단방향) 모드로 강제.

        2026-06-12 #ONEWAY: 봇은 positionIdx=0(원웨이) 고정인데 계정이 헤지
        모드면 모든 주문이 retCode 10001 로 거부 — "봇 출범 후 매매 0건"
        사용자의 유력 원인. 봇 시작 시 best-effort 로 호출해 원천 차단.
        해당 심볼만 전환(mode=0) — 다른 심볼의 헤지 사용엔 영향 없음.

        Returns:
            Bybit 응답 dict. 이미 원웨이(110025)면 ``alreadySet=True``,
            실패는 빈 dict (봇 진행 안 막음).
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            self._wlog("ensure_oneway_mode: _ex 없음 — skip")
            return {}
        await self._ensure_time_sync()
        raw_symbol = symbol.replace("/", "").split(":")[0]
        params = {"category": "linear", "symbol": raw_symbol, "mode": 0}
        try:
            result = await ex.private_post_v5_position_switch_mode(params)
            logger.info("원웨이 모드 전환 완료 (%s)", raw_symbol)
            return dict(result) if isinstance(result, dict) else {"raw": str(result)}
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            # 110025: position mode not modified — 이미 원웨이.
            if "110025" in msg or "not modified" in msg.lower():
                return {"retCode": 110025, "alreadySet": True}
            if "10005" in msg and "query-api" in msg:
                # set_leverage 와 동일한 ccxt UTA 체크 false positive.
                return {"retCode": 0, "alreadySet": False}
            self._wlog("ensure_oneway_mode 실패 (%s): %s", raw_symbol, e)
            return {}

    async def modify_stop_loss(
        self, symbol: str, new_stop_loss: float,
    ) -> dict[str, Any]:
        """Bybit V5 set_trading_stop API 호출 — 활성 포지션 SL 수정.

        Args:
            symbol: ccxt unified symbol (e.g. "BTC/USDT:USDT").
            new_stop_loss: 새 SL 가격.

        Returns:
            Bybit API 응답 dict. 실패 시 빈 dict.
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            self._wlog("modify_stop_loss: _ex 없음 — skip")
            return {}
        await self._ensure_time_sync()
        # ccxt unified → Bybit raw symbol ("BTC/USDT:USDT" → "BTCUSDT").
        raw_symbol = symbol.replace("/", "").split(":")[0]
        params = {
            "category": "linear",
            "symbol": raw_symbol,
            "stopLoss": str(new_stop_loss),
            # tpsl 모드 — Full 이면 전체 포지션 SL 수정.
            "tpslMode": "Full",
            "positionIdx": 0,  # one-way mode.
        }
        try:
            result = await ex.private_post_v5_position_trading_stop(params)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "34040" in msg or "not modified" in msg:
                logger.debug(
                    "modify_stop_loss skip (%s sl=%.4f 이미 적용)",
                    raw_symbol, new_stop_loss,
                )
                return {"retCode": 34040, "alreadySet": True}
            self._wlog("set_trading_stop 실패 (%s, sl=%.4f): %s",
                           raw_symbol, new_stop_loss, e)
            return {}
        return dict(result) if isinstance(result, dict) else {"raw": str(result)}

    async def set_position_tpsl(
        self,
        symbol: str,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        tp_size: float | None = None,
        tpsl_mode: str = "Full",
        trailing_stop: float | None = None,
        active_price: float | None = None,
    ) -> dict[str, Any]:
        """Bybit V5 set_trading_stop — 활성 포지션에 SL/TP conditional 동시 설정.

        #PARTIAL-TP-ORDER 2026-06-24: tpsl_mode="Partial" + tp_size 면 부분 TP(포지션
        일부 수량)를 position-attached 로 박는다 — 진입 시 1.5R/swing 두 TP 를 거래소에
        미리 등록해 봇 폴링 의존 제거. 포지션 닫히면 거래소가 자동 취소(파트너 지적).
        ⚠️ 부분 TP 2개 처리·자동취소는 거래소 실동작이라 소액 실측 검증 필수(백테 불가).

        #LIVE-4 fix: limit entry 주문에 SL/TP 동봉하면 Bybit 가 주문 시점 현재가 기준
        검증 (10001 "StopLoss should greater/lower base_price") 으로 거부 — 계획가가
        현재가에서 벗어나면 방향 위반. 그래서 entry 는 SL/TP 없이 limit 으로 넣고,
        체결되어 포지션이 생긴 뒤 이 메서드로 conditional SL/TP 를 박는다. 체결 시점엔
        가격=계획가라 SL/TP 방향이 유효하다.

        Returns:
            Bybit API 응답 dict. 실패 시 빈 dict (호출처가 warning 처리).
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            self._wlog("set_position_tpsl: _ex 없음 — skip")
            return {}
        await self._ensure_time_sync()
        raw_symbol = symbol.replace("/", "").split(":")[0]
        params: dict[str, Any] = {
            "category": "linear",
            "symbol": raw_symbol,
            "tpslMode": tpsl_mode,  # "Full"(전체) or "Partial"(tp_size 부분 TP)
            "positionIdx": 0,
        }
        if stop_loss is not None:
            params["stopLoss"] = str(stop_loss)
        if take_profit is not None:
            params["takeProfit"] = str(take_profit)
        # Partial 모드: 부분 TP 수량(tpSize) — 이 수량만 TP 체결(나머지 수량은 유지).
        # 진입 시 1.5R(50%)·swing(50%) 두 TP 를 부분으로 박는 용도.
        if tpsl_mode == "Partial" and tp_size is not None:
            params["tpSize"] = str(tp_size)
        # #TRAIL-EXCHANGE 2026-07-02 (Origo 1.4): Bybit 네이티브 트레일링 스탑.
        # trailingStop = 추적 거리(가격 절대값), activePrice = 활성화 가격 —
        # 가격이 activePrice 도달 후 거래소가 tick 단위로 스탑을 끌어올림(봇 무관).
        # activePrice 생략 시 즉시 활성(입양 시 이미 활성가 지난 포지션용).
        if trailing_stop is not None:
            params["trailingStop"] = str(trailing_stop)
            if active_price is not None:
                params["activePrice"] = str(active_price)
        try:
            result = await ex.private_post_v5_position_trading_stop(params)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "34040" in msg or "not modified" in msg:
                logger.debug(
                    "set_position_tpsl skip (%s sl=%s tp=%s 이미 적용)",
                    raw_symbol, stop_loss, take_profit,
                )
                return {"retCode": 34040, "alreadySet": True}
            self._wlog(
                "set_position_tpsl 실패 (%s sl=%s tp=%s): %s",
                raw_symbol, stop_loss, take_profit, e,
            )
            return {}
        # #TPSL-RETCODE fix: ccxt 가 예외를 안 던지고 retCode!=0 인 에러 바디를 200 OK
        # 로 그대로 돌려주는 경우, 호출처는 truthy 판정만 하므로 실패를 SL 적용 성공으로
        # 오인 → 무SL 포지션 잔존. retCode 를 명시 검사해 0(정상)·34040(이미 적용) 외엔
        # falsy({}) 반환 → 상위 _ensure_protective_sl 의 재시도·비상청산 경로가 작동.
        # 2026-06-06 #TPSL-RETCODE-STR fix: Bybit/ccxt 가 retCode 를 문자열 '0' 으로
        # 돌려주는 경우가 있어 int 비교(0)에서 타입 불일치로 정상(OK)을 이상으로 오판
        # → 멀쩡한 SL 을 실패 처리 → 비상청산. int 로 정규화해 비교한다.
        result_dict = dict(result) if isinstance(result, dict) else {"raw": str(result)}
        ret_code_raw = result_dict.get("retCode")
        if ret_code_raw is not None:
            try:
                ret_code = int(ret_code_raw)
            except (TypeError, ValueError):
                ret_code = -1  # 파싱 불가 = 이상 취급
            if ret_code not in (0, 34040):
                self._wlog(
                    "set_position_tpsl retCode 이상 (%s sl=%s tp=%s): %s",
                    raw_symbol, stop_loss, take_profit, result_dict,
                )
                return {}
        return result_dict


__all__ = ["AuroraClientAdapter"]
