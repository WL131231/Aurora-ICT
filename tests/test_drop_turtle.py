"""#DROP-TURTLE 2026-08-11 — turtle_soup 진입 소스 제거.

5년·7심볼·3,478건 실측에서 **유일하게 적자가 확정된 소스**였다.
본표본 BTC+ETH 245건 −0.248R [−0.390 ~ −0.100] · 홀드아웃 알트5 495건 −0.109R,
롱·숏 양쪽 모두 음수. 제거 시 본표본 +0.062→+0.152R / 홀드아웃 +0.030→+0.066R.

3단 검증을 통과했다:
  ① 홀드아웃 재현       순열 p=0.0034 (탐색에 안 쓴 알트 5페어)
  ② 플라시보           무작위로 같은 수를 빼면 +0.062R, turtle 을 빼면 +0.153R
                       → p=0.0000. "나쁜 거래를 빼서 좋아진 것"이 아니다
  ③ 다중비교 보정       소스 4종 동시 검정 → 문턱 0.0125, 실측 p=0.0000/0.0026
  ④ 워크포워드         앞 절반으로 판단 → 뒤 절반에서만 검정 p=0.0002/0.0073

여기서 고정하는 것은 **기본값이 꺼져 있고, 켜면 되돌아온다**는 두 가지다.
지표 검출(detect_turtle_soup_setups) 자체는 남겨둔다 — UI 마커와 연구용으로 쓰인다.
"""

from __future__ import annotations

import pandas as pd
import pytest

from aurora_ict.config.settings import IctSettings
from aurora_ict.strategy.silver_bullet import build_extra_source_setups


def _df(n: int = 300) -> pd.DataFrame:
    """결정론적 합성 봉 — 스윕 후 되돌림이 반복돼 turtle 이 잡히는 형태."""
    rows = []
    base = 100.0
    for i in range(n):
        # 20봉 주기로 저점을 뚫었다가 되돌아오는 패턴
        dip = -3.0 if i % 20 == 19 else 0.0
        o = base + (i % 5) * 0.2
        h = o + 1.0
        lo = o - 1.0 + dip
        c = o + (0.5 if i % 2 else -0.5)
        rows.append({"open": o, "high": h, "low": lo, "close": c, "volume": 100.0})
        base += 0.05
    df = pd.DataFrame(rows)
    df.index = pd.to_datetime(
        [1735689600000 + i * 300_000 for i in range(n)], unit="ms", utc=True,
    )
    return df


def test_turtle_off_removes_only_turtle_setups() -> None:
    """★ 끄면 turtle 소스만 사라지고 다른 소스는 그대로."""
    df = _df()
    on = build_extra_source_setups(df, min_rr=1.0, enable_turtle_soup=True)
    off = build_extra_source_setups(df, min_rr=1.0, enable_turtle_soup=False)

    def srcs(xs: list) -> set[str]:
        return {str(getattr(s.source, "value", s.source)) for s in xs}

    assert "turtle_soup" not in srcs(off)
    # turtle 을 뺀 나머지는 보존된다 (다른 소스가 잡힌 경우)
    assert srcs(off) == srcs(on) - {"turtle_soup"}
    assert len(off) <= len(on)


def test_turtle_flag_is_reversible() -> None:
    """켜면 되돌아온다 — 배포 후 롤백이 플래그 하나여야 한다."""
    df = _df()
    on = build_extra_source_setups(df, min_rr=1.0, enable_turtle_soup=True)
    off = build_extra_source_setups(df, min_rr=1.0, enable_turtle_soup=False)
    again = build_extra_source_setups(df, min_rr=1.0, enable_turtle_soup=True)
    assert len(again) == len(on)
    assert len(again) >= len(off)


def test_settings_default_off() -> None:
    """settings 기본값이 꺼져 있다."""
    assert IctSettings().origo_turtle_soup_enabled is False


@pytest.mark.parametrize("tier", ["sub_30d", "sub_90d", "sub_365d"])
def test_subscription_forces_off(tier: str) -> None:
    """★ 구독 정책에서 강제로 꺼진다 — 사용자가 켜도 무시."""
    s = IctSettings(license_type=tier, origo_turtle_soup_enabled=True)
    assert s.origo_turtle_soup_enabled is False


def test_referral_keeps_user_choice() -> None:
    """무료(referral)는 사용자 설정을 존중 — 강제는 구독 정책에만 건다."""
    s = IctSettings(license_type="referral", origo_turtle_soup_enabled=True)
    assert s.origo_turtle_soup_enabled is True


def test_bot_default_off() -> None:
    """봇 인스턴스 기본값도 꺼져 있다 (매니저 배선 누락 시 안전측)."""
    from aurora_ict.bot.bot_ict_instance import BotIctInstance

    # dataclass(slots=True) 라 클래스 속성이 descriptor 다 — 기본값을 직접 본다.
    assert BotIctInstance.__dataclass_fields__["turtle_soup_enabled"].default is False
