"""#PAIR-DROP — 가동 불가 페어를 자동 해제하고 사용자에게 알린다.

2026-08-07 라이브 실측: 한 사용자가 예전에 고른 BANK·BILL 이 거래대금 상위 30 밖으로
밀려 화이트리스트에서 빠졌다. 재가동은 **정책상 영원히 실패**하는데 10분마다 재시도
하며 경고만 쌓였다.

    bot 자동 재가동 실패 — AICT-.../BANK/USDT:USDT:
      'BANK/USDT:USDT' 는 거래 가능 목록(거래대금 상위 30)에 없습니다.

파트너 지시: "안내창 띄우고 자동 페어 선택 취소로".

가장 중요한 것은 **일시 장애와 정책 거부를 구분**하는 것이다. API 키 만료나 거래소
응답 실패로 페어를 해제해 버리면 사용자가 고른 종목이 조용히 사라진다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aurora_ict.auth import users_db

CODE = "AICT-DROP-DROP-DROP"
SYM = "BANK/USDT:USDT"


class _Alerter:
    """텔레그램 알림 스텁 — 보낸 메시지를 기록한다."""

    def __init__(self, fail: bool = False) -> None:
        self.sent: list[tuple[str, str]] = []
        self.fail = fail

    async def send_user_text(self, user_code: str, text: str) -> None:
        if self.fail:
            raise RuntimeError("telegram down")
        self.sent.append((user_code, text))


class _Mu:
    def __init__(self, alerter: Any = None) -> None:
        self.alerter = alerter


@pytest.fixture
def db(tmp_path: Path) -> Path:
    p = tmp_path / "users.db"
    users_db.init_db(p)
    users_db.create_user(p, CODE)
    users_db.set_bot_running(p, CODE, True, symbol=SYM)
    users_db.set_last_active_pair(p, CODE, SYM, True)
    return p


def _drop(db_path: Path, msg: str) -> None:
    """saas 의 정리 로직과 동일 — 정책 거부일 때만 해제."""
    permanent = "제외된 페어" in msg or "거래 가능 목록" in msg
    if permanent:
        users_db.set_bot_running(db_path, CODE, False, symbol=SYM)
        users_db.set_last_active_pair(db_path, CODE, SYM, False)


def test_whitelist_drop_clears_both_flags(db: Path) -> None:
    """★ 화이트리스트 탈락 — 가동 플래그와 **선호**에서 모두 빠진다.

    선호까지 빼야 전체 START 때 같은 실패가 반복되지 않는다.
    """
    _drop(db, "'BANK/USDT:USDT' 는 거래 가능 목록(거래대금 상위 30)에 없습니다.")

    assert SYM not in users_db.get_bot_running_symbols(db, CODE)
    assert SYM not in users_db.get_last_active_pairs(db, CODE)


def test_excluded_pair_also_dropped(db: Path) -> None:
    """검증 탈락(EXCLUDED) 페어도 같은 처리 — 기존 동작 유지."""
    _drop(db, "'BNB/USDT:USDT' 는 제외된 페어입니다.")

    assert SYM not in users_db.get_bot_running_symbols(db, CODE)


@pytest.mark.parametrize("msg", [
    "API 키가 등록되지 않았습니다.",
    "bybit {'retCode':10006,'retMsg':'Too many visits.'}",
    "서버 전체 동시 가동 한도에 도달했습니다.",
    "동시 가동 페어는 최대 5개입니다",
])
def test_transient_failure_keeps_selection(db: Path, msg: str) -> None:
    """★ 일시 장애로는 해제하지 않는다 — 사용자가 고른 종목이 사라지면 안 된다."""
    _drop(db, msg)

    assert SYM in users_db.get_bot_running_symbols(db, CODE)
    assert SYM in users_db.get_last_active_pairs(db, CODE)


@pytest.mark.asyncio
async def test_notify_explains_reason_and_recovery() -> None:
    """안내 문구에 **이유와 되돌리는 법**이 들어간다 — 조용히 빼면 혼란이다."""
    from aurora_ict.saas import _notify_pair_dropped

    al = _Alerter()
    await _notify_pair_dropped(
        _Mu(al), CODE, SYM, "'BANK/USDT:USDT' 는 거래 가능 목록에 없습니다.",
    )

    assert len(al.sent) == 1
    code, text = al.sent[0]
    assert code == CODE
    assert "BANK" in text
    assert "거래대금" in text          # 이유
    assert "다시 추가" in text          # 되돌리는 법
    assert "다른 종목은 정상" in text    # 전체 장애가 아님을 명시


@pytest.mark.asyncio
async def test_notify_failure_is_swallowed() -> None:
    """알림이 실패해도 예외가 새지 않는다 — 정리는 이미 끝났다."""
    from aurora_ict.saas import _notify_pair_dropped

    await _notify_pair_dropped(_Mu(_Alerter(fail=True)), CODE, SYM, "거래 가능 목록")


@pytest.mark.asyncio
async def test_no_alerter_is_safe() -> None:
    """알림 수단이 없는 배포(단일 사용자 등)에서도 안전."""
    from aurora_ict.saas import _notify_pair_dropped

    await _notify_pair_dropped(_Mu(None), CODE, SYM, "거래 가능 목록")
