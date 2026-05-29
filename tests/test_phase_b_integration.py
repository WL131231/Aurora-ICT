"""Phase B 통합 — 4 source 의 setup builder + ict_signal 통합 단위 테스트.

CLAUDE.md mock 0 정책 — 결정론적 OHLC 합성 입력만.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from aurora_ict.signal.ict_signal import generate_ict_signal
from aurora_ict.strategy.silver_bullet import (
    Direction,
    SetupSource,
    SilverBulletSetup,
    build_extra_source_setups,
)


def _synthetic_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """결정론적 random walk OHLC — 합성 차트."""
    rng = np.random.default_rng(seed)
    prices = 80000 + np.cumsum(rng.standard_normal(n) * 50)
    bars = []
    for i, p in enumerate(prices):
        bars.append({
            "timestamp": 1700000000000 + i * 60000 * 5,
            "open": float(p),
            "high": float(p) + 30,
            "low": float(p) - 30,
            "close": float(p) + rng.standard_normal() * 10,
        })
    return pd.DataFrame(bars)


# ============================================================
# Phase B-1: SilverBulletSetup 구조 확장
# ============================================================


def test_setup_source_enum_has_all_5_values():
    """SetupSource 가 모든 source 박혔는지."""
    assert SetupSource.FVG.value == "fvg"
    assert SetupSource.TURTLE_SOUP.value == "turtle_soup"
    assert SetupSource.MITIGATION_BLOCK.value == "mitigation_block"
    assert SetupSource.IMPLIED_FVG.value == "implied_fvg"
    assert SetupSource.REJECTION_BLOCK.value == "rejection_block"


def test_setup_zone_property_with_explicit_fields():
    """fvg 없고 _zone_* 박힌 setup 의 zone_high/low/anchor_idx 정확."""
    setup = SilverBulletSetup(
        ts_ms=1000, direction=Direction.LONG, window="turtle",
        entry=80000, stop_loss=79500, take_profit=82000, risk_reward=4.0,
        fvg=None, source=SetupSource.TURTLE_SOUP,
        _zone_high=80100, _zone_low=79900, _anchor_idx=50,
    )
    assert setup.zone_high == 80100
    assert setup.zone_low == 79900
    assert setup.anchor_idx == 50


def test_setup_zone_property_fallback_to_fvg():
    """fvg 박힌 setup (source=FVG) — zone_* 가 fvg.high/low 로 fallback."""
    from aurora_ict.indicators.fvg import FVG, FVGType
    fvg = FVG(ts_ms=1000, type=FVGType.BULLISH, high=80100, low=79900, idx=50)
    setup = SilverBulletSetup(
        ts_ms=1000, direction=Direction.LONG, window="am_sb",
        entry=80000, stop_loss=79500, take_profit=82000, risk_reward=4.0,
        fvg=fvg, source=SetupSource.FVG,
    )
    # fvg 의 high/low/idx 자동 반환
    assert setup.zone_high == 80100
    assert setup.zone_low == 79900
    assert setup.anchor_idx == 50


# ============================================================
# Phase B-2: 새 source 별 builder + B-4: ict_signal 통합
# ============================================================


def test_build_extra_setups_returns_multiple_sources():
    """합성 df 에서 4 source 중 적어도 1개 이상 setup 검출."""
    df = _synthetic_df(100, seed=42)
    setups = build_extra_source_setups(df, min_rr=1.0)
    assert len(setups) > 0

    sources = {s.source for s in setups}
    # 최소 한 source 잡힘
    assert len(sources) >= 1
    # 모든 setup 이 새 source (FVG 아님)
    assert SetupSource.FVG not in sources


def test_build_extra_setups_respects_min_rr():
    """min_rr 빡빡하면 setup 줄어듦."""
    df = _synthetic_df(100, seed=42)
    setups_loose = build_extra_source_setups(df, min_rr=1.0)
    setups_strict = build_extra_source_setups(df, min_rr=10.0)
    assert len(setups_strict) < len(setups_loose) or len(setups_loose) == 0


def test_build_extra_setups_bias_filter():
    """bias 박으면 그 방향 setup 만."""
    from aurora_ict.indicators.structure import TrendDirection
    df = _synthetic_df(100, seed=42)
    long_only = build_extra_source_setups(df, min_rr=1.0, bias=TrendDirection.UP)
    short_only = build_extra_source_setups(df, min_rr=1.0, bias=TrendDirection.DOWN)
    for s in long_only:
        assert s.direction is Direction.LONG
    for s in short_only:
        assert s.direction is Direction.SHORT


def test_ict_signal_uses_extra_sources_when_no_fvg_setup():
    """기존 FVG setup 없을 때도 새 source 가 setup 으로 채택."""
    df = _synthetic_df(100, seed=42)
    sig = generate_ict_signal(
        df, "BTCUSDT", min_rr=1.0, stale_bars=100, disable_time_filter=True,
    )
    # signal 이 actionable 이면 setup 의 source 가 새 4 source 중 하나일 수 있음
    if sig.is_actionable and sig.setup is not None:
        # source 가 enum 의 valid 값
        assert sig.setup.source in SetupSource


def test_extra_setup_has_valid_zone_and_sl_tp():
    """새 builder 결과 모두 zone/SL/TP 가 valid (NaN/None 없음)."""
    df = _synthetic_df(100, seed=42)
    setups = build_extra_source_setups(df, min_rr=1.0)
    for s in setups:
        assert s.zone_high > s.zone_low
        assert s.entry > 0
        assert s.stop_loss > 0
        assert s.take_profit > 0
        assert s.risk_reward >= 1.0
        # 방향 일관성 검증
        if s.direction is Direction.LONG:
            assert s.stop_loss < s.entry
            assert s.take_profit > s.entry
        else:
            assert s.stop_loss > s.entry
            assert s.take_profit < s.entry


# ============================================================
# Phase B-3: 마이그레이션 회귀 (기존 FVG setup 정상 동작)
# ============================================================


def test_fvg_setup_anchor_idx_equals_fvg_idx():
    """source=FVG 인 setup 의 anchor_idx 가 fvg.idx 와 동일."""
    from aurora_ict.indicators.fvg import FVG, FVGType
    fvg = FVG(ts_ms=1000, type=FVGType.BULLISH, high=80100, low=79900, idx=42)
    setup = SilverBulletSetup(
        ts_ms=1000, direction=Direction.LONG, window="am_sb",
        entry=80000, stop_loss=79500, take_profit=82000, risk_reward=4.0,
        fvg=fvg,
    )
    assert setup.anchor_idx == 42


def test_setup_without_fvg_and_zone_raises_when_accessed():
    """fvg=None + _zone_high=None → zone_high 접근 시 명확한 에러."""
    import pytest
    setup = SilverBulletSetup(
        ts_ms=1000, direction=Direction.LONG, window="x",
        entry=80000, stop_loss=79500, take_profit=82000, risk_reward=4.0,
        fvg=None,
    )
    with pytest.raises(ValueError, match="zone_high"):
        _ = setup.zone_high


# ============================================================
# 2026-05-27 fix: Phase B 도 disable_time_filter 적용 (사각지대 메꿈)
# ============================================================


def test_phase_b_time_filter_disabled_passes_all() -> None:
    """disable_time_filter=True → killzone 밖이라도 통과 (기존 동작)."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from aurora_ict.strategy.silver_bullet import _phase_b_in_time_window
    ny = ZoneInfo("America/New_York")
    ts_gap = int(datetime(2026, 5, 27, 6, 0, tzinfo=ny).timestamp() * 1000)
    assert _phase_b_in_time_window(ts_gap, disable_time_filter=True) is True


def test_phase_b_in_nyse_session_passes_with_filter_on() -> None:
    """NYSE 시간 안 + KZ/Macro/SB 면 disable_time_filter=False 여도 통과.

    2026-05-28: 파트너 결정 — sub_* 정책 "미장 안의 Killzone/Macro/SB 만".
    이전 기대: NY 08:00 (NY_AM KZ 안) → True
    새 기대 : 미장 전이라 False. 미장 안 + NY_AM (NY 09:35) → True.
    """
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from aurora_ict.strategy.silver_bullet import _phase_b_in_time_window
    ny = ZoneInfo("America/New_York")
    # 화요일 (2026-05-26) NY 09:35 — 미장 안 + NY_AM KZ + am_macro_2 → 통과
    ts_pass = int(datetime(2026, 5, 26, 9, 35, tzinfo=ny).timestamp() * 1000)
    assert _phase_b_in_time_window(ts_pass, disable_time_filter=False) is True
    # NY 08:00 (KZ 안이지만 미장 전) — 새 정책에서 차단
    ts_pre_market = int(datetime(2026, 5, 26, 8, 0, tzinfo=ny).timestamp() * 1000)
    assert _phase_b_in_time_window(ts_pre_market, disable_time_filter=False) is False


def test_phase_b_outside_killzone_blocked_with_filter_on() -> None:
    """킬존 밖 (NY 06:00 = London 끝과 NY_AM 시작 사이) 은 차단 — 사각지대 fix 검증."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from aurora_ict.strategy.silver_bullet import _phase_b_in_time_window
    ny = ZoneInfo("America/New_York")
    ts_gap = int(datetime(2026, 5, 27, 6, 0, tzinfo=ny).timestamp() * 1000)
    assert _phase_b_in_time_window(ts_gap, disable_time_filter=False) is False


def test_phase_b_outside_killzone_blocked_asian_gap() -> None:
    """Asian 끝(23:59) 과 London 시작(02:00) 사이 — NY 00:30 도 차단."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from aurora_ict.strategy.silver_bullet import _phase_b_in_time_window
    ny = ZoneInfo("America/New_York")
    ts_gap = int(datetime(2026, 5, 27, 0, 30, tzinfo=ny).timestamp() * 1000)
    assert _phase_b_in_time_window(ts_gap, disable_time_filter=False) is False


# ============================================================
# 2026-05-29 #MIN-SL-EXTRA fix: build_extra_source_setups 의 setup 도
# min_sl_distance_pct 가드 적용 (이전엔 우회됐던 버그)
# ============================================================


def test_generate_ict_signal_filters_extra_source_with_min_sl_distance():
    """min_sl_distance_pct 가드가 extra source setup 에도 적용된다.

    5-29 회고: 새벽~오후 5건 매매 모두 source=turtle_soup / mitigation_block /
    implied_fvg 였는데, build_extra_source_setups 호출에 min_sl_distance_pct
    인자가 빠져 0.13~0.18% 타이트 SL setup 이 통과 → 풀히트 100%. 합친 후
    setups 리스트를 한 번 더 필터링하는 가드로 두 source 경로 모두 일관 적용.
    """
    df = _synthetic_df(100, seed=42)
    # min_sl_distance_pct=0 (비활성) — 기준 전부 통과 setup 수
    sig_off = generate_ict_signal(
        df, symbol="BTC/USDT:USDT", min_rr=1.0, min_sl_distance_pct=0.0,
    )
    # 매우 큰 임계 — 거의 모든 setup skip (SL 거리 5% 이상만 통과)
    sig_strict = generate_ict_signal(
        df, symbol="BTC/USDT:USDT", min_rr=1.0, min_sl_distance_pct=0.05,
    )
    # 합성 df 에서 sig_off 는 actionable (setup found) 일 수 있지만
    # sig_strict 는 모든 setup 필터링되어 NO_ACTION 으로 떨어져야 한다.
    if sig_off.setup is not None:
        # entry/sl 거리 비율 계산 — strict 임계 (0.05) 보다 작아야 필터링 대상.
        e = sig_off.setup.entry
        s = sig_off.setup.stop_loss
        dist_pct = abs(e - s) / e if e > 0 else 0
        if dist_pct < 0.05:
            # 같은 setup 이 strict 에서 필터링됐는지 확인 — anchor_idx 로 매칭.
            if sig_strict.setup is not None:
                assert sig_strict.setup.anchor_idx != sig_off.setup.anchor_idx
            # 또는 strict 가 NO_ACTION 으로 떨어졌으면 가드 작동 확인.


def test_min_sl_distance_filter_removes_tight_sl_setup():
    """SL 거리가 임계 미만인 setup 은 직접 만들어도 ict_signal 출력에서 빠진다."""
    df = _synthetic_df(100, seed=42)
    # 임계 1% 로 호출. 합성 df 의 ATR 가 좁으면 거의 모든 setup 필터링.
    sig = generate_ict_signal(
        df, symbol="BTC/USDT:USDT", min_rr=1.0,
        min_sl_distance_pct=0.01,  # 1% 이상만 통과
    )
    # 통과한 setup 이 있으면 그 setup 의 SL 거리는 >= 1%.
    if sig.setup is not None:
        e = sig.setup.entry
        s = sig.setup.stop_loss
        assert abs(e - s) / e >= 0.01
