"""거래소 추상 인터페이스 — 모든 어댑터가 따라야 할 프로토콜.

본 모듈은 거래소 어댑터(ccxt_client.py 등)가 구현해야 할 ``ExchangeClient``
Protocol 과 도메인 dataclass (``Order`` / ``Position`` / ``Balance``) 를 정의.

설계 원칙:
    - 거래소 차이 (Bybit / OKX / Binance) 는 어댑터가 흡수
    - 호출자(``BotInstance`` / ``Executor``) 는 Protocol 만 의존 (테스트 mock 용이)
    - 모든 메서드 ``async`` (실 거래소 호출은 I/O bound)

담당: ChoYoon (exchange 영역) — 어댑터 PR 한정 위임 받음 (DESIGN.md §10, 2026-05-03)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import pandas as pd


def normalize_side(raw: object) -> Literal["long", "short"] | None:
    """거래소 응답의 side 값을 방향으로 정규화 — 인식 실패 시 **None**.

    #CLOSE-500 후속 2026-08-06: 방향 판정이 코드 곳곳에 흩어져 제각각이었고,
    일부는 **인식 실패를 임의 방향으로 단정**했다(예: ``"short" if s=="short"
    else "long"`` → ``"sell"`` 이 오면 숏을 롱으로 읽는다). 방향을 잘못 잡으면
    손절을 반대편에 걸거나 엉뚱한 방향으로 청산 주문을 낸다.

    거래소·엔드포인트마다 표기가 다르다:
        ccxt position   long / short
        Bybit 주문      Buy / Sell
        일부 응답       buy / sell (소문자)

    Args:
        raw: side 필드 값(문자열이 아니어도 안전).

    Returns:
        ``"long"`` / ``"short"``, 또는 인식 실패 시 ``None``.
        **None 을 임의 방향으로 대체하지 말 것** — 호출측이 판단을 보류해야 한다.
    """
    s = str(raw or "").strip().lower()
    if s in ("long", "buy"):
        return "long"
    if s in ("short", "sell"):
        return "short"
    return None


@dataclass(slots=True)
class Order:
    """주문 결과."""

    order_id: str
    symbol: str
    side: Literal["buy", "sell"]
    qty: float
    price: float | None  # None이면 시장가
    status: str
    timestamp_ms: int


@dataclass(slots=True)
class Position:
    """포지션 정보."""

    symbol: str
    side: Literal["long", "short"]
    qty: float
    entry_price: float
    leverage: int
    unrealized_pnl: float
    margin_mode: Literal["isolated", "cross"]


@dataclass(slots=True)
class ClosedPosition:
    """거래소가 매칭해서 돌려준 청산 포지션 한 record (v0.1.23).

    봇 자기 거래내역 (``ClosedTrade``) 와 별개 — 거래소 history API 가 직접 매칭한 결과.
    Bybit V5 ``/v5/position/closed-pnl`` 응답 형식 정합 + 어댑터별 변환 흡수.

    Phase 1 = 거래소 메타 (entry/exit/leverage/pnl) 만. ``triggered_by`` 등 Aurora 메타는 X.
    """

    symbol: str                   # ccxt 표준 (예: "BTC/USDT:USDT")
    direction: Literal["long", "short"]
    leverage: int
    qty: float                    # 청산 수량
    entry_price: float
    exit_price: float
    opened_at_ts: int             # ms (포지션 생성 시각 — createdTime)
    closed_at_ts: int             # ms (포지션 청산 시각 — updatedTime)
    pnl_usd: float                # realized PnL
    roi_pct: float                # (pnl / margin) × 100, margin = (entry × qty) / leverage
    # v0.1.65: 수수료 (USDT). 거래소 응답에 fee 정확치 박혀있으면 추출, 없으면 0.
    # 봇 자기 record (ClosedTrade) 는 시장가 taker 시뮬레이션. 외부 record 는
    # ccxt 어댑터가 채울 책임 (Phase 3 — 일단 default 0).
    fee_usd: float = 0.0


@dataclass(slots=True)
class Balance:
    """계정 자본금 — DESIGN.md §4 (옵션 a 어댑터 PR 신설).

    Phase 1 = USDT 단일 자산 기준. 다중 자산은 Phase 3 에서 확장.

    Attributes:
        total_usd: 전체 자본금 (USDT). ``free_usd + used_usd`` 와 일치.
        free_usd:  사용 가능 (마진 안 묶인 자본).
        used_usd:  현재 묶여있는 마진 (열린 포지션의 마진 합).
    """

    total_usd: float
    free_usd: float
    used_usd: float


class ExchangeClient(Protocol):
    """거래소 어댑터 인터페이스."""

    name: str  # "bybit", "okx", "binance"

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> pd.DataFrame:
        """OHLCV 캔들 가져오기."""
        ...

    async def fetch_position(self, symbol: str) -> Position | None:
        """단일 페어 포지션 조회."""
        ...

    async def get_positions(self) -> list[Position]:
        """모든 페어 포지션 조회 — 대시보드 / multi-pair 운영용.

        DESIGN.md §4 옵션 a 어댑터 PR 신설. 빈 리스트 반환 가능.
        """
        ...

    async def get_equity(self) -> Balance:
        """계정 자본금 조회 — 대시보드 잔고 표시 + 포지션 사이즈 계산 입력.

        DESIGN.md §4 옵션 a 어댑터 PR 신설. ``run_mode='paper'`` 시
        config 의 가짜 시드 반환 (어댑터 구현 측 결정).
        """
        ...

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
        """주문 전송 — ``stop_loss`` / ``take_profit`` 동봉 시 체결 포지션에 적용."""
        ...

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """레버리지 설정."""
        ...

    async def cancel_all(self, symbol: str) -> None:
        """전체 주문 취소."""
        ...

    async def fetch_ticker(self, symbol: str) -> float | None:
        """현재 시장가 (last trade price) 실시간 조회 (v0.1.39).

        SL/TP 청산 폴링용 — 봉 닫힘 close 보다 빠른 반응 (wick 시점 청산 가능).
        Aurora ``BotInstance._step`` 가 보유 중일 때 매 1초 호출.

        Returns:
            ``float`` 마지막 거래 가격 / ``None`` 호출 실패 (호출자가 close fallback).
        """
        ...

    async def fetch_closed_positions(
        self,
        since_ms: int | None = None,
        limit: int = 200,
    ) -> list[ClosedPosition]:
        """거래소 측 청산 포지션 history 조회 (v0.1.23).

        UI `거래내역 (P&L)` 표 / 결과 통계가 봇 시작 전 / 외부 거래까지 포함하도록.
        DESIGN.md §3.x 미정의 — Bybit 우선 구현, 다른 거래소는 stub 가능.

        Args:
            since_ms: 시작 시각 (ms epoch). None 이면 거래소 기본 (보통 최근 7일).
            limit: 한 페이지 최대 record 수. Bybit V5 = 50~200, 기본 200.

        Returns:
            신→구 정렬 (최근 청산 먼저). 미구현 어댑터는 빈 리스트.
        """
        ...
