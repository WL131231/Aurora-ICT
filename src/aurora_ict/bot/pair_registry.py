"""PairRegistry — 거래 가능 페어 화이트리스트 (Bybit USDT perp 거래대금 상위 N).

페어를 BTC/ETH 에서 다종목으로 확장할 때, 거래 가능 목록을 거래소에서 동적으로
구성한다. ``fetch_tickers`` 는 무거우니 TTL 캐시로 호출 빈도를 제한한다. 조회
실패 시 기존 캐시를 유지해 일시적 네트워크 장애가 가동을 막지 않게 한다.

BTC/ETH(메이저)는 항상 허용 목록에 포함되도록 보장한다 — 거래소 응답 지연/실패
시에도 기존 1순위 페어는 동작해야 하기 때문.

조회용 client(source)는 메서드 인자로 받는다. MultiUserBotManager 는 사용자별
client 를 동적 생성하므로, registry 가 특정 client 에 묶이지 않고 호출 시점의
client 로 조회하되 캐시는 공유한다(전부 같은 거래소).

담당: 지영민 (페어 확장 PR — 화이트리스트 동적화)
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

logger = logging.getLogger(__name__)

# 메이저 — 거래소 조회 실패해도 항상 거래 가능 목록에 포함.
# (레버리지 정책 분기에도 사용 — 알트는 _ALT_LEVERAGE 고정)
MAJOR_PAIRS = ("BTC/USDT:USDT", "ETH/USDT:USDT")

# 고정 페어 7 — 2026-06-11~12 흑자엣지 v2 백테스트로 확정 (IN +2.40 / OUT +0.96,
# ~2.1회/일). 서비스가 항상 가동하는 기본 포트폴리오. 사용자는 이 외에
# MAX_CHOICE_PAIRS(3)개까지 추가 선택 가능 (총 10).
FIXED_PAIRS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "LINK/USDT:USDT",
    "HYPE/USDT:USDT",
)

# Cursus(DualST 추세형) 전용 고정 페어 — 2026-07-31 개발자 지정.
# "종목 변경 해야해요 트론 필수 / LINK 제외" → FIXED_PAIRS 에서 LINK↔TRX 교체.
# ⚠️ FIXED_PAIRS 는 Origo 백테로 확정된 목록이라 **공유하면 안 된다** — 같이 고치면
#    Origo 페어까지 바뀐다. 모델별로 분리해 `fixed_pairs_for_model()` 로 조회한다.
# 백테 근거(5년 1h, 라이브 정합 엔진): TRX -1,941% vs LINK -3,571% → 교체가 +1,630%
#    개선. 단 TRX 자체도 적자라 "덜 나쁜 페어로 교체"이지 흑자 전환은 아니다.
CURSUS_FIXED_PAIRS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "TRX/USDT:USDT",
    "HYPE/USDT:USDT",
)


def fixed_pairs_for_model(model_name: str | None) -> tuple[str, ...]:
    """모델별 고정 페어 목록.

    Args:
        model_name: 사용자 선택 모델명(예: "Origo 2.3" / "Cursus 1.0"). None 이면 기본.

    Returns:
        Cursus 계열이면 CURSUS_FIXED_PAIRS, 그 외 FIXED_PAIRS.
    """
    from aurora_ict.config.settings import AVAILABLE_MODELS  # 순환 import 회피

    if model_name and AVAILABLE_MODELS.get(model_name) == "cursus":
        return CURSUS_FIXED_PAIRS
    return FIXED_PAIRS


# 추천 선택 페어 — 피커 상단 고정 + '추천' 배지용. 순서 = 순위.
# 2026-07-28 Origo 2.2 현행 설정(NY_PM·cond_align 포함) 23종 재스캔으로 갱신
# (파트너 승인: 추천 배지만, 고정 리스트 불변). 통과 기준 = 양반기 흑자 + 연도
# robust:
#   TIA(+9.1, 승률 74%, net/MDD 6.1, 4년 전부 흑자) · NEAR(+6.4, 5년 전부 흑자,
#   6/12·7/20 스캔에 이어 3회째 통과) · DOT(+4.3, net/MDD 3.8, 6년 전부 흑자).
#   FIL(H1 -2.1 반기 비일관)·ARB(n=9 표본부족)·ENA(-4.9 사망)는 현행 설정에서
#   탈락 → 제거. 교훈 재확인: 게이트 구성이 바뀌면 페어 적합성도 재검증.
RECOMMENDED_PAIRS = (
    "TIA/USDT:USDT",
    "NEAR/USDT:USDT",
    "DOT/USDT:USDT",
)

# 제외 페어 — 백테스트에서 IN/OUT 양면 마이너스로 검증 탈락. 화이트리스트에서
# 빼고, 직접 가동 요청도 차단한다.
#   BNB: 11페어 검증 (2026-06-12)
#   UNI(-0.78/-0.57)·APT(-0.60/-0.20)·TON(-0.23/-0.46): 신규 8페어 검증 (2026-06-12)
#   SUI(-0.36/-1.62, n139): 2차 9국면 검증에서 양면 마이너스 확정 (2026-06-12)
EXCLUDED_PAIRS = frozenset({
    "BNB/USDT:USDT",
    "UNI/USDT:USDT",
    "APT/USDT:USDT",
    "TON/USDT:USDT",
    "SUI/USDT:USDT",
})


class PairSource(Protocol):
    async def list_top_usdt_perps(self, limit: int = 30) -> list[str]: ...


class PairRegistry:
    """거래대금 상위 N 페어 화이트리스트 + TTL 캐시.

    Attributes:
        limit: 상위 몇 개를 허용할지.
        ttl_sec: 캐시 유효 기간(초).
    """

    def __init__(self, *, limit: int = 30, ttl_sec: float = 3600.0) -> None:
        self.limit = limit
        self.ttl_sec = float(ttl_sec)
        # 첫 조회 전 폴백 — 고정 7 은 거래소 응답 없이도 항상 가동 가능해야 함.
        self._cache: list[str] = list(FIXED_PAIRS)
        self._fetched_at: float | None = None

    async def get_allowed(
        self, source: PairSource, *, now: float | None = None,
    ) -> list[str]:
        """거래 가능 페어 목록 반환 — 캐시 만료 시 source 에서 갱신.

        조회 결과가 비어있으면(실패) 기존 캐시를 유지한다. 어떤 경우에도 고정
        페어(FIXED_PAIRS)는 목록에 포함되고, 제외 페어(EXCLUDED_PAIRS — 검증
        탈락)는 거래대금 상위권이어도 목록에서 빠진다.
        """
        t = time.monotonic() if now is None else now
        fresh = (
            self._fetched_at is not None
            and (t - self._fetched_at) < self.ttl_sec
        )
        if not fresh:
            pairs = await source.list_top_usdt_perps(self.limit)
            if pairs:
                merged = [p for p in pairs if p not in EXCLUDED_PAIRS]
                for fixed in FIXED_PAIRS:
                    if fixed not in merged:
                        merged.append(fixed)
                self._cache = merged
                self._fetched_at = t
            else:
                logger.warning("페어 목록 갱신 실패 — 기존 캐시 유지(%d개)",
                               len(self._cache))
        return list(self._cache)

    async def is_allowed(
        self, source: PairSource, symbol: str, *, now: float | None = None,
    ) -> bool:
        """symbol 이 거래 가능 화이트리스트에 있는지."""
        return symbol in await self.get_allowed(source, now=now)


__all__ = [
    "EXCLUDED_PAIRS",
    "FIXED_PAIRS",
    "MAJOR_PAIRS",
    "RECOMMENDED_PAIRS",
    "PairRegistry",
    "PairSource",
]
