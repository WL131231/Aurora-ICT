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
from aurora_ict.config.settings import CURSUS_MODEL_NAME, IctSettings

CODE = "AICT-RECO-TEST-0001"
LINK, TRX, BTC = "LINK/USDT:USDT", "TRX/USDT:USDT", "BTC/USDT:USDT"


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
    assert st == {"stopped": 1, "started": 1, "held": 0}


@pytest.mark.asyncio
async def test_open_position_is_not_stopped(mu: tuple) -> None:
    """★ 포지션이 열린 봇은 절대 정지하지 않는다 — 무방비 포지션 방지."""
    m, rec = mu
    _put(m, LINK, _client(0.0), _Bot(active_position=object()))

    st = await m.reconcile_fixed_pairs()

    assert rec.stopped == []
    assert rec.started == []          # 1:1 교체이므로 새 페어도 안 켬
    assert st["held"] == 1


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

    assert rec.stopped == []
    assert st == {"stopped": 0, "started": 0, "held": 0}


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
