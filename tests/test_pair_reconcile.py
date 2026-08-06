"""#CURSUS-PAIRS — 고정 페어 자동 정합(사용자 조작 없이 LINK→TRX 수렴).

개발자 페어 변경 후, 이미 떠 있는 슬롯은 사용자가 STOP→START 하기 전까지 옛
목록 그대로였다(모델 전환도 심볼 목록은 유지한 채 봇만 재생성한다). 10분 주기
작업이 이를 수렴시킨다.

여기서 가장 중요한 건 **정지하면 안 되는 경우를 정지하지 않는 것**이다. 포지션이
열린 봇을 끄면 SL/TP 를 관리할 주체가 사라져 무방비로 남는다 — 자동화가 만들 수
있는 최악의 사고다. 그래서 판정 실패(네트워크 오류)조차 "정지 금지"로 취급한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import pytest
from cryptography.fernet import Fernet

from aurora_ict.auth import users_db
from aurora_ict.bot.multi_user_manager import MultiUserBotManager
from aurora_ict.bot.pair_registry import CURSUS_FIXED_PAIRS, FIXED_PAIRS
from aurora_ict.config.settings import CURSUS_MODEL_NAME, IctSettings

CODE = "AICT-RECO-TEST-0001"
LINK, TRX, BTC = "LINK/USDT:USDT", "TRX/USDT:USDT", "BTC/USDT:USDT"

# Cursus 에만 있는 고정 페어 수 = 정합이 새로 켜는 대상 수.
# 하드코딩하지 않는다 — 2026-08-06 Origo 가 7→2 로 축소되며 1개에서 5개가 됐다.
_CURSUS_ONLY = len(set(CURSUS_FIXED_PAIRS) - set(FIXED_PAIRS))


@dataclass
class _Bot:
    active_position: Any = None
    _pending_limit: Any = None
    _pending_entry: Any = None


@dataclass
class _Slot:
    symbol: str
    client: Any
    bot: Any = None
    settings: Any = None


@dataclass
class _Rec:
    """정지/가동 호출 기록 — self-spy(외부 mock 라이브러리 없이)."""

    stopped: list[tuple[str, str]] = field(default_factory=list)
    started: list[tuple[str, str]] = field(default_factory=list)
    migrated: set[str] = field(default_factory=set)


def _client(contracts: float = 0.0, *, fail: bool = False) -> AsyncMock:
    c = AsyncMock()
    if fail:
        c.fetch_position = AsyncMock(side_effect=RuntimeError("network"))
    else:
        c.fetch_position = AsyncMock(return_value={"contracts": contracts,
                                                   "side": "long"})
    return c


@pytest.fixture
def mu(tmp_path: Path) -> tuple[MultiUserBotManager, _Rec]:
    db = tmp_path / "users.db"
    users_db.init_db(db)
    users_db.create_user(db, CODE)
    users_db.set_last_model(db, CODE, CURSUS_MODEL_NAME)
    m = MultiUserBotManager(
        client_factory=lambda *a, **k: None, db_path=db,
        base_settings=IctSettings(enabled=True),
        master_key=Fernet.generate_key(),
    )
    rec = _Rec()

    async def _stop(user_code: str, symbol: str = BTC, **_kw: Any) -> None:
        rec.stopped.append((user_code, symbol))
        m._slots.pop((user_code, symbol), None)

    async def _start(user_code: str, symbol: str = BTC, **_kw: Any) -> None:
        rec.started.append((user_code, symbol))
        m._slots[(user_code, symbol)] = _Slot(symbol, _client())

    m.stop = _stop        # type: ignore[method-assign]
    m.start = _start      # type: ignore[method-assign]

    # 마이그레이션 마커를 인메모리로 격리 — 실제 사용자 데이터 디렉토리에 파일을
    # 쓰면 테스트가 결정론적이지 않다(첫 실행만 통과하고 이후 skip 됨).
    done: set[str] = set()
    m._pair_migration_done = lambda u: u in done        # type: ignore[method-assign]
    m._mark_pair_migration_done = done.add              # type: ignore[method-assign]
    rec.migrated = done                                 # type: ignore[attr-defined]
    return m, rec


def _put(m: MultiUserBotManager, sym: str, client: AsyncMock,
         bot: Any = None) -> None:
    m._slots[(CODE, sym)] = _Slot(sym, client, bot)


@pytest.mark.asyncio
async def test_idle_legacy_pair_swapped(mu: tuple) -> None:
    """★ 라이브 케이스 — 포지션 없는 LINK 는 정지되고 그 자리에 TRX 가 뜬다."""
    m, rec = mu
    _put(m, LINK, _client(0.0), _Bot())

    st = await m.reconcile_fixed_pairs()

    assert (CODE, LINK) in rec.stopped
    assert (CODE, TRX) in rec.started
    assert st == {"stopped": 1, "started": _CURSUS_ONLY, "held": 0}


@pytest.mark.asyncio
async def test_open_position_is_not_stopped(mu: tuple) -> None:
    """★ 포지션이 열린 봇은 절대 정지하지 않는다 — 무방비 포지션 방지."""
    m, rec = mu
    _put(m, LINK, _client(0.0), _Bot(active_position=object()))

    st = await m.reconcile_fixed_pairs()

    assert rec.stopped == []
    assert st["held"] == 1
    # 새 페어(TRX) 가동은 잔재 정지와 무관하게 진행된다 — 켜지 못할 이유가 없고,
    # 정지 성공에 연동하면 정지만 되고 가동이 실패한 경우 영영 안 켜진다.
    assert (CODE, TRX) in rec.started


@pytest.mark.asyncio
async def test_exchange_position_without_bot_state(mu: tuple) -> None:
    """봇은 모르지만 거래소에 포지션이 있으면(입양 전·수동) 역시 보류."""
    m, rec = mu
    _put(m, LINK, _client(0.5), _Bot())

    st = await m.reconcile_fixed_pairs()

    assert rec.stopped == []
    assert st["held"] == 1


@pytest.mark.asyncio
async def test_pending_limit_order_is_not_stopped(mu: tuple) -> None:
    """지정가 진입 대기 중이면 거래소에 주문이 걸려 있다 — 보류."""
    m, rec = mu
    _put(m, LINK, _client(0.0), _Bot(_pending_limit=object()))

    st = await m.reconcile_fixed_pairs()

    assert rec.stopped == []
    assert st["held"] == 1


@pytest.mark.asyncio
async def test_position_check_failure_is_not_stopped(mu: tuple) -> None:
    """★ 판정 실패(네트워크)도 '정지 금지' 로 취급 — 껐는데 포지션이 있었으면 최악."""
    m, rec = mu
    _put(m, LINK, _client(fail=True), _Bot())

    st = await m.reconcile_fixed_pairs()

    assert rec.stopped == []
    assert st["held"] == 1


@pytest.mark.asyncio
async def test_user_chosen_pair_untouched(mu: tuple) -> None:
    """사용자가 직접 고른 선택 페어(ADA)는 정리 대상이 아니다."""
    m, rec = mu
    _put(m, "ADA/USDT:USDT", _client(0.0), _Bot())

    st = await m.reconcile_fixed_pairs()

    assert rec.stopped == []          # ADA 는 정리 대상이 아니다
    assert st["stopped"] == 0
    assert st["held"] == 0


@pytest.mark.asyncio
async def test_current_fixed_pair_untouched(mu: tuple) -> None:
    """현재 모델의 고정 페어는 당연히 유지."""
    m, rec = mu
    _put(m, BTC, _client(0.0), _Bot())
    _put(m, TRX, _client(0.0), _Bot())

    st = await m.reconcile_fixed_pairs()

    assert rec.stopped == []
    assert st["stopped"] == 0


@pytest.mark.asyncio
async def test_idempotent(mu: tuple) -> None:
    """정합 후 다시 돌려도 아무 일도 일어나지 않는다(10분마다 도는 작업)."""
    m, rec = mu
    _put(m, LINK, _client(0.0), _Bot())
    await m.reconcile_fixed_pairs()
    rec.stopped.clear()
    rec.started.clear()

    st = await m.reconcile_fixed_pairs()

    assert rec.stopped == []
    assert rec.started == []
    assert st == {"stopped": 0, "started": 0, "held": 0}


@pytest.mark.asyncio
async def test_new_pair_started_even_without_legacy(mu: tuple) -> None:
    """★ 7/31 라이브 함정 — 잔재가 이미 정리됐어도 새 페어는 켜야 한다.

    LINK 정지는 성공했는데 TRX 가동만 실패했고(화이트리스트 거부), 다음 주기엔
    정리할 잔재가 없어 교체가 영영 트리거되지 않았다. 그래서 가동을 정지 성공에
    연동하지 않는다.
    """
    m, rec = mu
    _put(m, BTC, _client(0.0), _Bot())   # 잔재(LINK) 없음

    st = await m.reconcile_fixed_pairs()

    assert (CODE, TRX) in rec.started
    assert st == {"stopped": 0, "started": _CURSUS_ONLY, "held": 0}


@pytest.mark.asyncio
async def test_new_pair_started_only_once(mu: tuple) -> None:
    """가동은 사용자당 1회 — 이후 사용자가 그 페어를 끄면 다시 켜지 않는다."""
    m, rec = mu
    _put(m, BTC, _client(0.0), _Bot())
    await m.reconcile_fixed_pairs()
    assert CODE in rec.migrated
    m._slots.pop((CODE, TRX), None)      # 사용자가 TRX 를 껐다고 가정
    rec.started.clear()

    st = await m.reconcile_fixed_pairs()

    assert rec.started == []
    assert st["started"] == 0
