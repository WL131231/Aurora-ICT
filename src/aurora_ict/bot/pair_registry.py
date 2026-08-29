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

# Origo 고정 페어 — 2026-08-06 **7개 → BTC·ETH 2개로 축소** (#PAIR-MAJOR).
#
# 근거: 낙폭 해부에서 최대 낙폭 85.6%(2년 지속) 구간을 뜯어보니 알트가 원인이었다.
#   · 최악 손실 10건 중 LINK 가 5건. 정상 구간엔 LINK 5건인데 낙폭 구간엔 10건.
#   · **알트 5개만 돌리면 5년에 0.86배** — 원금이 줄고 파산확률 19%.
# 복리 시뮬(7x · 동시보유 반영 · DD 스로틀 · 거래 복원추출):
#     조합            거래   자산(중앙)  낙폭    최악5%   파산
#     기존 7페어      126     2.43배   85.6%   0.20배   6.2%
#     BTC+ETH          45    10.07배   36.7%   1.80배   0.0%
#     LINK 만 제외     108     8.27배   65.3%   0.97배   0.5%
#     알트만(5)         81     0.86배   85.7%   0.18배  19.0%
# "2개라서 좋은 것"이 아니다 — 2페어 조합 21개를 전수 비교했더니 상위 3개가 전부
# BTC 포함(BTC+XRP 9.11 / BTC+ETH 8.89 / BTC+DOGE 6.96)이고 하위 3개는 전부 알트
# 조합(SOL+HYPE 0.62 / SOL+LINK 0.38 / LINK+HYPE 0.37)이었다. **메이저라서**다.
#
# 빈도 감소(126→45건)는 MMBM 2번째 진입모델 배선(#MMBM-WIRE)으로 상쇄한다
# — 결합 시 월 15.3건으로 오히려 기존보다 늘어난다.
# ── 2026-08-15 #PAIR7: 7페어 복원 (2026-07 축소 결정 뒤집음) ──
# 7월에 알트를 뺀 근거는 "낙폭 85.6% 의 주범"이었는데, 그 판단에 두 가지 왜곡이
# 있었다:
#   ① turtle_soup 이 알트에서 특히 나빴다(495건 -0.109R). 그걸 알트 탓으로 돌렸다.
#      → 8/11 turtle 제거(#421) 후 알트 5페어는 1,902건 **+0.066R** 로 양수다.
#   ② 백테가 라이브의 **2배**를 걸고 있었다(size_pct 0.9 × lev 7 = 6.3배 고정 vs
#      라이브 실측 3.16배). 낙폭이 과장됐다. → 8/10 사이징 이식으로 교정.
# 둘을 고치고 다시 재니 7페어가 우세하다(리스크 x0.5 기준):
#     BTC+ETH  월 14건 · 자산 14.12x · 낙폭 48.3% · 파산 0.1%
#     7페어    월 46건 · 자산 10.69x · 낙폭 54.2% · 파산 0.0%
# 빈도 3.3배(일 1.5회 — 파트너 요구 "일 1~3회")를 얻고 위험은 현행 수준이다.
# 검증 5관문 통과: 시간분리 뒤40%(+0.046R) · 구간 5/5 흑자 · 페어 7/7 흑자 ·
# 리스크배수 0.35~0.70 안정 · 최악 1년 파산 없음. 연도별로도 6/6 흑자
# (2021 +0.180 / 2022 +0.083 / 2023 +0.097 / 2024 +0.020 / 2025 +0.076 / 2026 +0.210).
FIXED_PAIRS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "LINK/USDT:USDT",
    "HYPE/USDT:USDT",
)

# 과거 고정 페어였다가 빠진 것들 — 자동 정합(reconcile_fixed_pairs)의 정리 대상.
# 이 목록이 없으면 "양쪽 모델 어디에도 없는" 페어(예: Origo 축소 후의 LINK)가
# 정리 대상에서 누락돼 계속 돌아간다.
# ⚠️ 정리는 **포지션이 없을 때만** 이뤄진다 — 열린 거래는 SL/TP 로 끝난 뒤 빠진다
#    (강제 청산 안 함. 손실을 확정시키고 되돌릴 수 없기 때문).
# [08-15] #PAIR7 로 전부 FIXED_PAIRS 에 복귀해 비어 있다. 다음에 페어를 뺄 때
# 여기에 넣으면 자동 정합이 정리한다(포지션 없을 때만).
LEGACY_FIXED_PAIRS: tuple[str, ...] = ()

# Cursus(DualST 추세형) 전용 고정 페어 — 2026-07-31 개발자 지정.
# "종목 변경 해야해요 트론 필수 / LINK 제외" → FIXED_PAIRS 에서 LINK↔TRX 교체.
# ⚠️ FIXED_PAIRS 는 Origo 백테로 확정된 목록이라 **공유하면 안 된다** — 같이 고치면
#    Origo 페어까지 바뀐다. 모델별로 분리해 `fixed_pairs_for_model()` 로 조회한다.
# 백테 근거(5년 1h, 라이브 정합 엔진): TRX -1,941% vs LINK -3,571% → 교체가 +1,630%
#    개선. 단 TRX 자체도 적자라 "덜 나쁜 페어로 교체"이지 흑자 전환은 아니다.
# [08-29] HYPE 제외 — 파트너 지시. Cursus 는 6페어로 운용한다.
# LEGACY_FIXED_PAIRS 에 넣을 필요는 없다 — 정합(reconcile_fixed_pairs)이
# `FIXED_PAIRS + CURSUS_FIXED_PAIRS + LEGACY` 를 합쳐 "내 모델 목록에 없는 것"을
# 정리 대상으로 잡는데, HYPE 는 Origo 쪽 FIXED_PAIRS 에 그대로 남아 있어 이미
# 그 합집합에 든다. 나중에 **양쪽 모두에서** 뺄 때는 LEGACY 에 넣어야 한다.
# ⚠️ 정리는 포지션이 없을 때만 이뤄진다 — 열린 HYPE 거래는 SL/TP 로 끝난 뒤
#    빠진다(강제 청산 안 함. 손실을 확정시키고 되돌릴 수 없기 때문).
CURSUS_FIXED_PAIRS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "TRX/USDT:USDT",
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
        # 첫 조회 전 폴백 — 고정 페어는 거래소 응답 없이도 항상 가동 가능해야 함.
        # 두 모델 모두 포함(#CURSUS-PAIRS): Origo 목록만 넣으면 조회 전 TRX 가동이
        # 거부된다.
        self._cache: list[str] = list(dict.fromkeys(
            (*FIXED_PAIRS, *CURSUS_FIXED_PAIRS, *LEGACY_FIXED_PAIRS),
        ))
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
                # #CURSUS-PAIRS 2026-07-31: **두 모델의 고정 페어 모두** 보장한다.
                # FIXED_PAIRS 만 보던 탓에 Cursus 전용 TRX 가 화이트리스트에 없어
                # 가동이 거부됐다("거래대금 상위 30 에 없습니다" — 라이브 실측).
                # 고정 페어는 정의상 거래대금 순위와 무관하게 허용돼야 한다.
                # #PAIR-MAJOR 2026-08-06: 레거시(고정에서 빠진 알트)도 보장한다.
                # 고정에서 뺀 것은 "자동으로 안 켜질 뿐" 금지가 아니다 — 사용자가
                # 직접 선택하는 길은 열어둔다(LINK 는 거래대금 30위 밖이라 보장이
                # 없으면 선택조차 거부된다).
                for fixed in (*FIXED_PAIRS, *CURSUS_FIXED_PAIRS, *LEGACY_FIXED_PAIRS):
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
    "CURSUS_FIXED_PAIRS",
    "LEGACY_FIXED_PAIRS",
    "EXCLUDED_PAIRS",
    "FIXED_PAIRS",
    "MAJOR_PAIRS",
    "RECOMMENDED_PAIRS",
    "PairRegistry",
    "PairSource",
    "fixed_pairs_for_model",
]
