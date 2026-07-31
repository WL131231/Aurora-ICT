"""#PARTIAL-ORPHAN — 부분 익절된 봇 포지션을 고아로 오인하지 않는다.

2026-07-31 라이브 사고: Cursus 4분할 TP 로 TP1(25%)만 체결된 포지션 3건이
admin 화면에서 **미추적**으로 방치됐다(SL 은 살아 있었으나 TP2~4·래더 트레일 정지).

실측 수량 — 진입 → TP1 청산 → 잔량:
    AICT-4H1M   564   → 141 → 423
    AICT-RG5R  8568   → 2142 → 6426
    AICT-SRKA    57   → 14  → 43

원인 두 갈래(둘 다 실패해야 "유저 수동 포지션" 으로 간주된다):
  ① `_find_unclosed_entry_event` 의 청산 판정이 TP_HIT 하나만 보고 종료 처리 →
     미청산 ENTRY 후보에서 빠짐.
  ② `position_opened_by_bot` 의 수량 대조가 **양방향 ±10%** 라, 25% 익절된
     잔량(564→423)을 "수량 불일치" 로 배제.
방아쇠는 모델 전환(Cursus→Origo)이었으나, 같은 봇으로 재시작만 해도 재현된다.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from aurora_ict.bot.bot_ict_instance import fully_closed_setups
from aurora_ict.interfaces.trades_store import TradeEventType


@dataclass
class _Ev:
    """TradeEvent 최소 스텁 — 결정론적 입력(mock 0 정책)."""

    symbol: str
    event_type: TradeEventType
    setup_ts_ms: int
    qty: float = 0.0
    direction: str = "short"
    ts_ms: int = 0
    price: float = 0.0


SYM = "DOGE/USDT:USDT"


def test_partial_tp_not_treated_as_closed() -> None:
    """★ 라이브 재현 — 진입 564 / TP1 141 청산 → 아직 종료 아님."""
    events = [
        _Ev(SYM, TradeEventType.ENTRY, 1000, qty=564.0),
        _Ev(SYM, TradeEventType.TP_HIT, 1000, qty=141.0),
    ]
    assert 1000 not in fully_closed_setups(events, SYM)


def test_all_four_tps_is_closed() -> None:
    """4분할이 모두 체결되면 종료로 판정."""
    events = [
        _Ev(SYM, TradeEventType.ENTRY, 1000, qty=564.0),
        *[_Ev(SYM, TradeEventType.TP_HIT, 1000, qty=141.0) for _ in range(4)],
    ]
    assert 1000 in fully_closed_setups(events, SYM)


def test_sl_hit_full_qty_is_closed() -> None:
    """부분 익절 후 잔량 전체가 SL 로 나가면 종료."""
    events = [
        _Ev(SYM, TradeEventType.ENTRY, 1000, qty=564.0),
        _Ev(SYM, TradeEventType.TP_HIT, 1000, qty=141.0),
        _Ev(SYM, TradeEventType.SL_HIT, 1000, qty=423.0),
    ]
    assert 1000 in fully_closed_setups(events, SYM)


def test_legacy_event_without_qty_is_closed() -> None:
    """수량 미기록 구 이벤트는 보수적으로 종료 처리(오채택 방지)."""
    events = [
        _Ev(SYM, TradeEventType.ENTRY, 1000, qty=564.0),
        _Ev(SYM, TradeEventType.FLIP_CLOSE, 1000, qty=0.0),
    ]
    assert 1000 in fully_closed_setups(events, SYM)


def test_other_symbol_isolated() -> None:
    """다른 심볼 이벤트가 섞여도 영향 없음."""
    events = [
        _Ev(SYM, TradeEventType.ENTRY, 1000, qty=564.0),
        _Ev("BTC/USDT:USDT", TradeEventType.SL_HIT, 1000, qty=564.0),
    ]
    assert 1000 not in fully_closed_setups(events, SYM)


def test_90pct_threshold() -> None:
    """수수료 등으로 잔량이 미세하게 남아도(>=90% 청산) 종료로 본다."""
    events = [
        _Ev(SYM, TradeEventType.ENTRY, 1000, qty=100.0),
        _Ev(SYM, TradeEventType.TP_HIT, 1000, qty=91.0),
    ]
    assert 1000 in fully_closed_setups(events, SYM)
    events2 = [
        _Ev(SYM, TradeEventType.ENTRY, 2000, qty=100.0),
        _Ev(SYM, TradeEventType.TP_HIT, 2000, qty=89.0),
    ]
    assert 2000 not in fully_closed_setups(events2, SYM)


@pytest.mark.parametrize(
    ("entry_filled", "current_qty", "expect"),
    [
        (564.0, 423.0, True),    # ★ 라이브 케이스 — 25% 익절된 잔량
        (8568.0, 6426.0, True),
        (57.0, 43.0, True),
        (564.0, 564.0, True),    # 미청산 원본
        (564.0, 700.0, False),   # 현재가 체결량 초과 = 다른 진입 합산 → 배제
    ],
)
def test_tag_match_qty_rule(entry_filled: float, current_qty: float,
                            expect: bool) -> None:
    """태그 매칭 수량 규칙 — 봇 포지션은 부분 청산으로 **줄어들 수만** 있다.

    ccxt_client.position_opened_by_bot 의 판정식과 동일:
        배제 조건 = qty > filled * 1.1
    """
    excluded = current_qty > entry_filled * 1.1
    assert (not excluded) is expect
