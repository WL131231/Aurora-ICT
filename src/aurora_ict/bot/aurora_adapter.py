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
from dataclasses import asdict, is_dataclass
from typing import Any

import pandas as pd
from ccxt.base.errors import AuthenticationError, ExchangeError, PermissionDenied

from aurora.exchange.base import normalize_side
from aurora.exchange.ccxt_client import _optional_order_number
from aurora_ict.bot.margin_guard import parse_available_usdt, parse_bybit_available_usdt

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
        # 거래소 인증 실패 연속 횟수 — 잔고·포지션·봉 조회 어느 것이든 성공하면 0.
        # 봇 _run_loop 가 임계치 도달 시 자동 정지에 사용.
        # 2026-09-08 #KEY-EXPIRED: 예전엔 fetch_balance 만 셌다. Cursus 는 진입
        # 후보가 있을 때만 잔고를 조회해서, 키가 만료돼도(33004) 포지션 조회만
        # 매 step 실패하며 카운터가 0 에 머물렀다 → 봇이 '눈 감은 채' RUNNING.
        self.auth_fail_streak = 0
        # 마지막 인증 실패 메시지(거래소 retMsg). UI/상태 API 가 사용자에게 보여준다.
        self.last_auth_error: str | None = None
        self._auth_log_n = 0
        # #WRITE-DENIED 2026-09-11: 주문·SL 등 **쓰기** 호출이 10005(Permission
        # denied)로 연속 거부된 횟수. 읽기는 되는데 쓰기만 막힌 키(읽기 전용 키)를
        # 잡기 위한 별도 카운터 — auth_fail_streak 는 읽기 성공에 리셋돼 못 잡는다.
        self.write_fail_streak = 0
        self.last_write_error: str | None = None

    def set_log_label(self, label: str) -> None:
        """거래소 호출 실패 WARNING 에 붙일 식별 라벨 설정 (예: 'AICT-XXXX/BTC').

        멀티유저 환경에서 retCode 10003(키 무효) 등 발생 시 어느 사용자인지
        로그만으로 특정하기 위함.

        Args:
            label: 사용자 코드/심볼 등 식별 문자열. None/빈 문자열이면 prefix 미부착.
        """
        self._log_label = label or ""

    @property
    def auth_error_kind(self) -> str | None:
        """현재 인증 실패 종류 — 'expired'(33004 만료) · 'invalid'(10003 등) · None.

        Returns:
            연속 실패가 0 이면 None. 메시지에 expired 가 있으면 'expired'.
        """
        if self.auth_fail_streak <= 0:
            return None
        msg = (self.last_auth_error or "").lower()
        return "expired" if "expired" in msg else "invalid"

    def _note_auth_fail(self, where: str, e: BaseException) -> None:
        """인증 실패 1건 기록 — 카운터·메시지 갱신 + 도배 방지 로그.

        매 step 마다 WARNING 을 찍으면 5초 간격으로 로그가 도배된다(9/8 실측
        분당 12줄). 첫 1회와 이후 12회마다 한 번만 남긴다.

        Args:
            where: 실패한 호출 이름(로그용).
            e: ccxt AuthenticationError.
        """
        self.auth_fail_streak += 1
        self.last_auth_error = str(e)[:200]
        self._auth_log_n += 1
        if self._auth_log_n == 1 or self._auth_log_n % 12 == 0:
            self._wlog(
                "%s 인증 실패 %d회 (키 %s): %s",
                where, self.auth_fail_streak,
                "만료" if self.auth_error_kind == "expired" else "무효", e,
            )

    def _note_auth_ok(self) -> None:
        """인증이 통한 호출 뒤 카운터 리셋(키 재등록 후 정상 복귀 감지)."""
        if self.auth_fail_streak:
            self._wlog("거래소 인증 복구 — 실패 %d회 후 정상", self.auth_fail_streak)
        self.auth_fail_streak = 0
        self.last_auth_error = None
        self._auth_log_n = 0

    def _note_write_fail(self, where: str, e: BaseException) -> None:
        """쓰기 호출 권한 거부(10005) 1건 기록.

        Args:
            where: 실패한 호출 이름.
            e: 거래소 예외.
        """
        self.write_fail_streak += 1
        self.last_write_error = str(e)[:200]
        self._wlog("%s 권한 거부 %d회 (API 키 거래 권한 없음): %s",
                   where, self.write_fail_streak, e)

    def _note_write_ok(self) -> None:
        """쓰기 호출이 실제로 통했을 때 카운터 리셋."""
        self.write_fail_streak = 0
        self.last_write_error = None

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
            self._note_auth_ok()
            return result
        except AuthenticationError as e:
            # 키 무효/만료 — 카운트만 하고 빈 dict. 봇 _run_loop 가 임계치에서 자동 정지.
            self._note_auth_fail("fetch_balance", e)
            return {}
        except Exception as e:  # noqa: BLE001
            self._wlog("fetch_balance 실패: %s", e)
            return {}

    async def fetch_available_usdt(self) -> float | None:
        """신규 진입에 쓸 가용 증거금만 조회한다 (담당: Codex, 2026-09-13).

        Args:
            없음.
        Returns:
            가용 USDT. 조회 실패, 계정 모드 또는 환산 정보 미확인 시 None.
        Raises:
            취소 예외는 호출자에게 전파한다.
        """
        ex = getattr(self._client, "_ex", None)
        if ex is None:
            return None
        if getattr(ex, "id", None) != "bybit":
            return parse_available_usdt(await self.fetch_balance())
        try:
            await self._ensure_time_sync()
            account_info = await ex.private_get_v5_account_info()
            result = account_info.get("result") if isinstance(account_info, dict) else None
            if (
                not isinstance(result, dict)
                or isinstance(account_info.get("retCode"), bool)
                or account_info.get("retCode") not in (0, "0")
            ):
                self._wlog("계정 모드 응답 이상 — 신규 진입 보류")
                return None
            mode = result.get("marginMode")
            if mode not in ("REGULAR_MARGIN", "PORTFOLIO_MARGIN", "ISOLATED_MARGIN"):
                self._wlog("계정 모드 미확인 — 신규 진입 보류")
                return None
            # 계정 정보 성공만으로 인증 실패 누적을 초기화하지 않는다.
            # 총잔고/UI용 fetch_balance는 유지하고, 진입 직전 최신 응답만 해석한다.
            balance = await self.fetch_balance()
            available = parse_bybit_available_usdt(balance, mode)
            if available is None:
                self._wlog("가용 증거금 해석 불가 (모드=%s) — 신규 진입 보류", mode)
            else:
                logger.info(
                    "[%s] 진입 가용 증거금 %.8f USDT (Bybit 모드=%s)",
                    self._log_label, available, mode,
                )
            return available
        except AuthenticationError as e:
            self._note_auth_fail("fetch_available_usdt", e)
            return None
        except Exception as e:  # noqa: BLE001
            self._wlog("가용 증거금 조회 실패 (%s) — 신규 진입 보류", type(e).__name__)
            return None

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

    async def position_opened_by_bot(
        self, symbol: str, side: str, entry_price: float, qty: float = 0.0,
    ) -> bool:
        """고아 포지션이 봇 것인지 판정 (주문 이력의 봇 태그 대조) — client 위임.

        하위 client 미지원/실패 시 False (보수적 — 확실히 봇 것일 때만 채택).
        """
        try:
            return await self._client.position_opened_by_bot(
                symbol, side, entry_price, qty,
            )
        except AttributeError:
            return False
        except Exception as e:  # noqa: BLE001
            self._wlog("position_opened_by_bot 실패: %s", e)
            return False

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
            self._note_write_ok()
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "34040" in msg or "not modified" in msg:
                logger.debug(
                    "modify_stop_loss skip (%s sl=%.4f 이미 적용)",
                    raw_symbol, new_stop_loss,
                )
                return {"retCode": 34040, "alreadySet": True}
            if "10005" in str(e):
                self._note_write_fail("modify_stop_loss", e)
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
            self._note_write_ok()
        except Exception as e:  # noqa: BLE001
            if "10005" in str(e):
                self._note_write_fail("set_position_tpsl", e)
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
