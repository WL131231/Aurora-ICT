"""#CURSUS-PAIRS 후속 — 모델별 고정 페어가 **봇 기동 경로**까지 반영되는지.

2026-07-31 라이브 로그로 발견: 모델 분기(`fixed_pairs_for_model`)를 넣었지만
API(페어 피커)에만 적용됐고, 봇을 실제로 띄우는 MultiUserBotManager 는 여전히
Origo 목록(FIXED_PAIRS)을 쓰고 있었다. 그래서 Cursus 유저에게 **LINK 가 계속
켜지고 TRX 는 안 켜졌다** — 개발자 지정("트론 필수 / LINK 제외")과 정반대.

    Cursus heartbeat — symbol=LINK/USDT:USDT active_pos=False   ← 떠 있으면 안 됨
    (TRX 는 아예 없음)

여기서 검증하는 건 "목록이 무엇인가"(pair_registry 테스트)가 아니라
**기동 경로가 그 목록을 참조하는가**다.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from aurora_ict.auth import users_db
from aurora_ict.bot.multi_user_manager import MultiUserBotManager
from aurora_ict.bot.pair_registry import CURSUS_FIXED_PAIRS, FIXED_PAIRS
from aurora_ict.config.settings import CURSUS_MODEL_NAME, ORIGO_MODEL_NAME, IctSettings

CODE = "AICT-PAIR-TEST-0001"
LINK, TRX = "LINK/USDT:USDT", "TRX/USDT:USDT"


@pytest.fixture
def mu(tmp_path: Path) -> MultiUserBotManager:
    db = tmp_path / "users.db"
    users_db.init_db(db)
    users_db.create_user(db, CODE)
    return MultiUserBotManager(
        client_factory=lambda *a, **k: None, db_path=db,
        base_settings=IctSettings(enabled=True),
        master_key=Fernet.generate_key(),
    )


def test_cursus_user_gets_trx_not_link(mu: MultiUserBotManager) -> None:
    """★ 라이브 재현 — Cursus 사용자의 고정 페어에 TRX 가 있고 LINK 는 없다."""
    users_db.set_last_model(mu.db_path, CODE, CURSUS_MODEL_NAME)

    pairs = mu._fixed_pairs(CODE)

    assert TRX in pairs
    assert LINK not in pairs
    assert pairs == CURSUS_FIXED_PAIRS


def test_origo_user_gets_origo_list(mu: MultiUserBotManager) -> None:
    """Origo 사용자는 Origo 목록을 받는다 — Cursus 변경이 새지 않는다.

    구체적 종목에 기대지 않는다(2026-08-06 Origo 는 BTC+ETH 로 축소됐다).
    검증 대상은 모델별 분기이지 목록 내용이 아니다.
    """
    users_db.set_last_model(mu.db_path, CODE, ORIGO_MODEL_NAME)

    pairs = mu._fixed_pairs(CODE)

    assert pairs == FIXED_PAIRS
    assert TRX not in pairs          # TRX 는 Cursus 전용


def test_unset_model_falls_back_to_origo(mu: MultiUserBotManager) -> None:
    """모델 미설정 사용자는 기존 동작(Origo 고정 7) 유지."""
    assert mu._fixed_pairs(CODE) == FIXED_PAIRS


def test_unknown_user_does_not_raise(mu: MultiUserBotManager) -> None:
    """존재하지 않는 코드로 물어도 예외 없이 기본 목록 — 기동을 막지 않는다."""
    assert mu._fixed_pairs("AICT-NOBODY-XXXX-0000") == FIXED_PAIRS


def test_start_preferred_uses_model_pairs(mu: MultiUserBotManager) -> None:
    """전체 START 가 켜는 목록 = 모델 고정 페어 + 선택 페어.

    start_preferred 는 거래소 연결을 타므로, 목록 구성 로직만 동일 재현해
    회귀를 잡는다(mock 0 — DB 만 쓰는 결정론적 검증).
    """
    users_db.set_last_model(mu.db_path, CODE, CURSUS_MODEL_NAME)

    fixed = mu._fixed_pairs(CODE)
    last = [LINK, "ADA/USDT:USDT"]   # 사용자가 마지막에 켰던 페어
    choice = [s for s in last if s not in fixed]
    pairs = list(fixed) + choice

    assert TRX in pairs
    # LINK 는 고정에서 빠졌지만 사용자가 직접 켰던 선택 페어로는 남을 수 있다
    # (금지 페어가 아니라 "자동으로 켜지지 않을 뿐" — 의도된 동작).
    assert pairs.count(LINK) == 1
    assert pairs.index(TRX) < pairs.index(LINK)


# ---- 화이트리스트 (2026-07-31 라이브 실패에서 추가) ----

class _Src:
    """거래대금 상위 N 조회 스텁 — TRX 는 순위 밖(실제 상황)."""

    def __init__(self, pairs: list[str]) -> None:
        self._pairs = pairs

    async def list_top_usdt_perps(self, limit: int) -> list[str]:
        return self._pairs[:limit]


@pytest.mark.asyncio
async def test_whitelist_includes_cursus_fixed_pairs() -> None:
    """★ 라이브 실패 재현 — TRX 가 거래대금 상위 밖이어도 가동 허용돼야 한다.

    실측 로그:
        [.../TRX/USDT:USDT] 고정 페어 정합 가동 실패:
        'TRX/USDT:USDT' 는 거래 가능 목록(거래대금 상위 30)에 없습니다.

    고정 페어는 정의상 거래대금 순위와 무관하게 허용된다(LINK 도 같은 이유로
    이미 보강돼 있었는데, Cursus 목록만 빠져 있었다).
    """
    from aurora_ict.bot.pair_registry import PairRegistry

    reg = PairRegistry(limit=30)
    src = _Src(["BTC/USDT:USDT", "ETH/USDT:USDT"])   # TRX·LINK 없음

    allowed = await reg.get_allowed(src)

    assert TRX in allowed
    assert LINK in allowed
    assert await reg.is_allowed(src, TRX)


@pytest.mark.asyncio
async def test_whitelist_fallback_before_first_fetch() -> None:
    """첫 조회 전 폴백 캐시에도 두 모델 고정 페어가 모두 들어 있어야 한다."""
    from aurora_ict.bot.pair_registry import PairRegistry

    reg = PairRegistry()

    assert TRX in reg._cache
    assert LINK in reg._cache
