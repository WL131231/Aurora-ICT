"""#MODEL-SWITCH-POS — 포지션 보유 중 모델 전환은 사용자에게 되묻는다.

파트너 지적(2026-08-01): 모델을 바꾸면 새 모델 봇이 옛 포지션을 입양하는데,
모델마다 청산 구조가 달라 **SL/TP 를 자기 방식으로 덮어쓴다**.

    Origo  SL = ATR×4 구조 기반 · TP = 2R + 부분익절
    Cursus SL = 고정 2%        · TP = 1/2/3/4% 4분할

SL 이 더 타이트해지면 그 자리에서 즉시 손절되고, 느슨해지면 계획보다 큰 손실을
떠안는다. 그래서 포지션이 열려 있으면 전환을 **보류하고 선택을 받는다**:

    defer  포지션 없는 페어만 즉시 전환. 나머지는 거래가 끝나면 자동 전환.
    close  지금 시장가 청산하고 전부 전환.

검증의 핵심은 "선택 없이는 아무것도 바뀌지 않는다"와 "유예된 페어의 봇을 건드리지
않는다"이다. 둘 중 하나라도 깨지면 사용자 포지션이 계획과 다르게 청산된다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from cryptography.fernet import Fernet

from aurora_ict.auth import users_db
from aurora_ict.bot.multi_user_manager import MultiUserBotManager
from aurora_ict.config.settings import CURSUS_MODEL_NAME, ORIGO_MODEL_NAME, IctSettings

CODE = "AICT-MSW1-MSW1-MSW1"
BTC, ETH = "BTC/USDT:USDT", "ETH/USDT:USDT"


@dataclass
class _State:
    value: str = "running"


@dataclass
class _Pos:
    direction: Any = None
    qty: float = 0.5
    entry: float = 100.0
    stop_loss: float = 98.0
    take_profit: float = 104.0


@dataclass
class _OrigoBot:
    """Origo 봇 스텁 — BotTrendInstance 가 아니므로 모델 판정상 origo."""

    active_position: Any = None
    _pending_limit: Any = None
    _pending_entry: Any = None
    state: _State = field(default_factory=_State)
    closed: list[str] = field(default_factory=list)

    async def _emergency_close(self, reason: str, price: float | None = None) -> None:
        self.closed.append(reason)
        self.active_position = None


@dataclass
class _Slot:
    symbol: str
    client: Any
    bot: Any = None
    settings: Any = None


class _Client:
    """포지션 조회 스텁 — contracts 로 노출 유무를 결정론적으로 준다."""

    def __init__(self, contracts: float = 0.0) -> None:
        self.contracts = contracts

    async def fetch_position(self, symbol: str) -> dict[str, Any]:
        return {"contracts": self.contracts, "side": "long"}


@pytest.fixture
def mu(tmp_path: Path) -> MultiUserBotManager:
    db = tmp_path / "users.db"
    users_db.init_db(db)
    users_db.create_user(db, CODE)
    users_db.set_last_model(db, CODE, ORIGO_MODEL_NAME)
    return MultiUserBotManager(
        client_factory=lambda *a, **k: None, db_path=db,
        base_settings=IctSettings(enabled=True),
        master_key=Fernet.generate_key(),
    )


# ---- 포지션 판정 (전환을 막는 근거) ----

@pytest.mark.asyncio
async def test_open_position_blocks(mu: MultiUserBotManager) -> None:
    """★ 봇이 포지션을 들고 있으면 '노출 있음' — 전환 확인 창의 트리거."""
    mu._slots[(CODE, BTC)] = _Slot(BTC, _Client(), _OrigoBot(active_position=_Pos()))

    assert await mu._has_live_exposure(CODE, BTC) is True


@pytest.mark.asyncio
async def test_exchange_position_blocks(mu: MultiUserBotManager) -> None:
    """봇은 모르지만 거래소에 포지션이 있으면 역시 막는다."""
    mu._slots[(CODE, BTC)] = _Slot(BTC, _Client(0.7), _OrigoBot())

    assert await mu._has_live_exposure(CODE, BTC) is True


@pytest.mark.asyncio
async def test_idle_slot_is_switchable(mu: MultiUserBotManager) -> None:
    """비어 있는 슬롯은 즉시 전환 가능."""
    mu._slots[(CODE, BTC)] = _Slot(BTC, _Client(), _OrigoBot())

    assert await mu._has_live_exposure(CODE, BTC) is False


# ---- 유예 전환 완료 (reconcile_models) ----

@pytest.mark.asyncio
async def test_deferred_switch_completes_when_flat(mu: MultiUserBotManager) -> None:
    """★ 거래가 끝나면 사용자가 아무것도 안 해도 새 모델로 교체된다."""
    users_db.set_last_model(mu.db_path, CODE, CURSUS_MODEL_NAME)
    mu._slots[(CODE, BTC)] = _Slot(BTC, _Client(), _OrigoBot())   # 포지션 없음
    calls: list[tuple[str, str]] = []

    async def _stop(user_code: str, symbol: str = BTC, **_kw: Any) -> None:
        calls.append(("stop", symbol))
        mu._slots.pop((user_code, symbol), None)

    async def _start(user_code: str, symbol: str = BTC, **_kw: Any) -> None:
        calls.append(("start", symbol))

    mu.stop = _stop        # type: ignore[method-assign]
    mu.start = _start      # type: ignore[method-assign]

    st = await mu.reconcile_models()

    assert st == {"switched": 1, "held": 0}
    assert calls == [("stop", BTC), ("start", BTC)]


@pytest.mark.asyncio
async def test_deferred_switch_waits_while_position_open(
    mu: MultiUserBotManager,
) -> None:
    """★ 거래 중인 페어는 절대 건드리지 않는다 — 옛 모델이 끝까지 관리."""
    users_db.set_last_model(mu.db_path, CODE, CURSUS_MODEL_NAME)
    mu._slots[(CODE, BTC)] = _Slot(BTC, _Client(), _OrigoBot(active_position=_Pos()))
    calls: list[str] = []

    async def _stop(user_code: str, symbol: str = BTC, **_kw: Any) -> None:
        calls.append(symbol)

    mu.stop = _stop        # type: ignore[method-assign]

    st = await mu.reconcile_models()

    assert st == {"switched": 0, "held": 1}
    assert calls == []


@pytest.mark.asyncio
async def test_no_switch_when_model_matches(mu: MultiUserBotManager) -> None:
    """이미 선택 모델이면 아무 일도 하지 않는다(주기 작업이라 멱등해야 한다)."""
    mu._slots[(CODE, BTC)] = _Slot(BTC, _Client(), _OrigoBot())   # Origo 봇 = 선택 모델
    calls: list[str] = []

    async def _stop(user_code: str, symbol: str = BTC, **_kw: Any) -> None:
        calls.append(symbol)

    mu.stop = _stop        # type: ignore[method-assign]

    st = await mu.reconcile_models()

    assert st == {"switched": 0, "held": 0}
    assert calls == []


@pytest.mark.asyncio
async def test_mixed_symbols_partial_switch(mu: MultiUserBotManager) -> None:
    """한 사용자 안에서 페어별로 갈린다 — 비어 있는 것만 전환."""
    users_db.set_last_model(mu.db_path, CODE, CURSUS_MODEL_NAME)
    mu._slots[(CODE, BTC)] = _Slot(BTC, _Client(), _OrigoBot(active_position=_Pos()))
    mu._slots[(CODE, ETH)] = _Slot(ETH, _Client(), _OrigoBot())
    stopped: list[str] = []

    async def _stop(user_code: str, symbol: str = BTC, **_kw: Any) -> None:
        stopped.append(symbol)
        mu._slots.pop((user_code, symbol), None)

    async def _start(user_code: str, symbol: str = BTC, **_kw: Any) -> None:
        return None

    mu.stop = _stop        # type: ignore[method-assign]
    mu.start = _start      # type: ignore[method-assign]

    st = await mu.reconcile_models()

    assert stopped == [ETH]           # BTC 는 포지션 때문에 유지
    assert st == {"switched": 1, "held": 1}


@pytest.mark.asyncio
async def test_stopped_bot_switches_without_restart(mu: MultiUserBotManager) -> None:
    """정지 상태 봇은 슬롯만 교체하고 다시 켜지 않는다(사용자가 끈 것을 존중)."""
    users_db.set_last_model(mu.db_path, CODE, CURSUS_MODEL_NAME)
    bot = _OrigoBot()
    bot.state = _State("stopped")
    mu._slots[(CODE, BTC)] = _Slot(BTC, _Client(), bot)
    started: list[str] = []

    async def _stop(user_code: str, symbol: str = BTC, **_kw: Any) -> None:
        mu._slots.pop((user_code, symbol), None)

    async def _start(user_code: str, symbol: str = BTC, **_kw: Any) -> None:
        started.append(symbol)

    mu.stop = _stop        # type: ignore[method-assign]
    mu.start = _start      # type: ignore[method-assign]

    st = await mu.reconcile_models()

    assert st["switched"] == 1
    assert started == []
