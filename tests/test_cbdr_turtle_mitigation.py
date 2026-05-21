"""그룹 1 신규 indicators 단위 테스트 — CBDR / Turtle Soup / Mitigation Block.

CLAUDE.md mock 0 정책 — 결정론적 합성 OHLCV 입력만 (외부 거래소 호출 X).
"""
from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from aurora_ict.indicators.cbdr import (
    CBDR_END_HOUR,
    CBDR_START_HOUR,
    CBDRBiasState,
    CBDRBox,
    classify_price_vs_cbdr,
    detect_cbdr_boxes,
    is_within_acceptable_range,
)
from aurora_ict.indicators.mitigation_block import (
    MitigationBlock,
    detect_mitigation_blocks,
    filter_retested,
)
from aurora_ict.indicators.order_block import OrderBlock, OrderBlockType
from aurora_ict.indicators.turtle_soup import (
    TurtleSoupDirection,
    detect_turtle_soup_setups,
)

NY_TZ = ZoneInfo("America/New_York")


def _ny_ts_ms(year: int, month: int, day: int, hour: int, minute: int = 0) -> int:
    """NY local 시각 → UTC ms."""
    dt = datetime(year, month, day, hour, minute, tzinfo=NY_TZ)
    return int(dt.astimezone(UTC).timestamp() * 1000)


def _bar(ts_ms: int, o: float, h: float, lo: float, c: float) -> dict:
    return {"timestamp": ts_ms, "open": o, "high": h, "low": lo, "close": c}


# ============================================================
# CBDR
# ============================================================


def test_cbdr_window_constants():
    """CBDR 시간: 14:00-20:00 NY local (정통)."""
    assert CBDR_START_HOUR == 14
    assert CBDR_END_HOUR == 20


def test_detect_cbdr_single_day():
    """하루치 CBDR 박스 — body 기준 (wick 제외) 검증."""
    bars = []
    # 2026-05-21 NY local 14:00-19:00 (1h 봉 6개) — body high 100, body low 90
    bars.append(_bar(_ny_ts_ms(2026, 5, 21, 14), 95, 105, 92, 100))
    bars.append(_bar(_ny_ts_ms(2026, 5, 21, 15), 100, 102, 96, 98))
    bars.append(_bar(_ny_ts_ms(2026, 5, 21, 16), 98, 99, 88, 90))    # body low 90
    bars.append(_bar(_ny_ts_ms(2026, 5, 21, 17), 90, 95, 89, 94))
    bars.append(_bar(_ny_ts_ms(2026, 5, 21, 18), 94, 96, 92, 95))
    bars.append(_bar(_ny_ts_ms(2026, 5, 21, 19), 95, 98, 93, 97))
    # 20:00 봉은 CBDR 미포함 (end_hour exclusive)
    bars.append(_bar(_ny_ts_ms(2026, 5, 21, 20), 97, 150, 50, 130))

    df = pd.DataFrame(bars)
    boxes = detect_cbdr_boxes(df)
    assert len(boxes) == 1
    box = boxes[0]
    assert box.date == "2026-05-21"
    # body high = max(open, close) 중 max = 100 (14시 봉의 close)
    assert box.high == 100.0
    # body low = min(open, close) 중 min = 90 (16시 봉의 close)
    assert box.low == 90.0
    assert box.range_width == 10.0
    assert box.mid == 95.0


def test_detect_cbdr_skips_non_window_hours():
    """CBDR 시간 밖 (13시, 20시 이후) 봉은 무시."""
    bars = [
        _bar(_ny_ts_ms(2026, 5, 21, 13), 100, 200, 50, 150),  # 13시 — 무시
        _bar(_ny_ts_ms(2026, 5, 21, 14), 95, 105, 92, 100),
        _bar(_ny_ts_ms(2026, 5, 21, 20), 200, 300, 100, 250),  # 20시 — 무시
    ]
    df = pd.DataFrame(bars)
    boxes = detect_cbdr_boxes(df)
    # 14시 봉만 들어감 — body high=100, low=95
    assert len(boxes) == 1
    assert boxes[0].high == 100.0
    assert boxes[0].low == 95.0


def test_detect_cbdr_empty_df():
    """빈 df → 빈 리스트."""
    assert detect_cbdr_boxes(pd.DataFrame()) == []


def test_cbdr_std_dev_level():
    """표준편차 배수 계산 — high + N*range / low - N*range."""
    box = CBDRBox(date="2026-05-21", high=100, low=90, range_width=10)
    assert box.std_dev_level(1.0, above=True) == 110.0
    assert box.std_dev_level(2.0, above=True) == 120.0
    assert box.std_dev_level(3.0, above=True) == 130.0
    assert box.std_dev_level(1.0, above=False) == 80.0
    assert box.std_dev_level(2.0, above=False) == 70.0


def test_classify_price_vs_cbdr_inside():
    box = CBDRBox(date="2026-05-21", high=100, low=90, range_width=10)
    assert classify_price_vs_cbdr(95, box) is CBDRBiasState.INSIDE
    assert classify_price_vs_cbdr(100, box) is CBDRBiasState.INSIDE
    assert classify_price_vs_cbdr(90, box) is CBDRBiasState.INSIDE


def test_classify_price_vs_cbdr_above_levels():
    """가격이 박스 위 — 1/2/3 std 단계 분기."""
    box = CBDRBox(date="2026-05-21", high=100, low=90, range_width=10)
    # +1 std = 110, +2 std = 120, +3 std = 130
    assert classify_price_vs_cbdr(105, box) is CBDRBiasState.ABOVE_1STD
    assert classify_price_vs_cbdr(115, box) is CBDRBiasState.ABOVE_1STD  # 110 < 115 < 120
    assert classify_price_vs_cbdr(125, box) is CBDRBiasState.ABOVE_2STD  # 120 <= 125 < 130
    assert classify_price_vs_cbdr(135, box) is CBDRBiasState.ABOVE_3STD


def test_classify_price_vs_cbdr_below_levels():
    box = CBDRBox(date="2026-05-21", high=100, low=90, range_width=10)
    # -1 std = 80, -2 std = 70, -3 std = 60
    assert classify_price_vs_cbdr(85, box) is CBDRBiasState.BELOW_1STD
    assert classify_price_vs_cbdr(75, box) is CBDRBiasState.BELOW_1STD
    assert classify_price_vs_cbdr(65, box) is CBDRBiasState.BELOW_2STD
    assert classify_price_vs_cbdr(55, box) is CBDRBiasState.BELOW_3STD


def test_is_within_acceptable_range():
    # 폭 5 / mid 100 → 5% — 0.5% 초과
    box_wide = CBDRBox(date="2026-05-21", high=102.5, low=97.5, range_width=5)
    assert is_within_acceptable_range(box_wide, max_range_pct=0.005) is False

    # 폭 0.3 / mid 100 → 0.3% — OK
    box_narrow = CBDRBox(date="2026-05-21", high=100.15, low=99.85, range_width=0.3)
    assert is_within_acceptable_range(box_narrow, max_range_pct=0.005) is True


# ============================================================
# Turtle Soup
# ============================================================


def test_turtle_soup_short_setup():
    """SHORT setup: 직전 N-bar high 를 wick 으로만 갱신 + close 안쪽."""
    # 20봉 high = 100, 21번째 봉이 wick 으로 105 찍고 close 95
    bars = []
    for _i in range(20):
        bars.append({"open": 90, "high": 100, "low": 85, "close": 95})
    # 21번째 봉 — SHORT setup
    bars.append({"open": 99, "high": 105, "low": 94, "close": 95})  # high>100, close<100

    df = pd.DataFrame(bars)
    setups = detect_turtle_soup_setups(df, lookback=20)
    assert len(setups) == 1
    s = setups[0]
    assert s.direction is TurtleSoupDirection.SHORT
    assert s.sweep_idx == 20
    assert s.sweep_price == 100.0
    assert s.wick_extreme == 105.0
    assert s.opposite_target == 85.0


def test_turtle_soup_long_setup():
    """LONG setup: N-bar low 를 wick 으로만 갱신 + close 안쪽."""
    bars = []
    for _i in range(20):
        bars.append({"open": 95, "high": 100, "low": 90, "close": 95})
    # 21번째 봉 — LONG setup (low<90, close>90)
    bars.append({"open": 91, "high": 96, "low": 85, "close": 95})

    df = pd.DataFrame(bars)
    setups = detect_turtle_soup_setups(df, lookback=20)
    assert len(setups) == 1
    s = setups[0]
    assert s.direction is TurtleSoupDirection.LONG
    assert s.sweep_idx == 20
    assert s.sweep_price == 90.0
    assert s.wick_extreme == 85.0


def test_turtle_soup_close_outside_rejected_when_strict():
    """require_close_inside=True (정통) — close 가 swing 밖이면 sweep 인정 X."""
    bars = []
    for _i in range(20):
        bars.append({"open": 95, "high": 100, "low": 90, "close": 95})
    # close 도 swing 밖 (105) → 진짜 돌파, sweep 아님
    bars.append({"open": 99, "high": 110, "low": 95, "close": 105})

    df = pd.DataFrame(bars)
    setups_strict = detect_turtle_soup_setups(df, lookback=20, require_close_inside=True)
    assert setups_strict == []

    setups_loose = detect_turtle_soup_setups(df, lookback=20, require_close_inside=False)
    assert len(setups_loose) == 1


def test_turtle_soup_empty_or_short_df():
    """빈/짧은 df → 빈 리스트."""
    assert detect_turtle_soup_setups(pd.DataFrame()) == []
    short_df = pd.DataFrame([{"open": 100, "high": 100, "low": 100, "close": 100}] * 5)
    assert detect_turtle_soup_setups(short_df, lookback=20) == []


def test_turtle_soup_no_sweep():
    """N-bar swing 갱신 없으면 빈 리스트."""
    bars = [{"open": 95, "high": 100, "low": 90, "close": 95}] * 25
    df = pd.DataFrame(bars)
    assert detect_turtle_soup_setups(df, lookback=20) == []


# ============================================================
# Mitigation Block
# ============================================================


def _ob(idx: int, ob_type: OrderBlockType, high: float, low: float,
        mitigated: bool = False) -> OrderBlock:
    """Test OB factory."""
    return OrderBlock(
        ts_ms=1000 * idx,
        type=ob_type,
        open=low,
        high=high,
        low=low,
        close=high,
        idx=idx,
        displacement_idx=idx + 1,
        mitigated=mitigated,
    )


def test_mitigation_block_skips_unmitigated_obs():
    """mitigated=False 면 Mitigation Block 후보 X."""
    obs = [_ob(0, OrderBlockType.BULLISH, high=100, low=90, mitigated=False)]
    df = pd.DataFrame([{"high": 105, "low": 95, "close": 100}] * 10)
    blocks = detect_mitigation_blocks(obs, df)
    assert blocks == []


def test_mitigation_block_bullish_full_cycle():
    """Bullish OB: mitigated → departure (zone 위로 close) → retest (wick zone 진입)."""
    # OB zone: 90-100, OB 봉 idx=0
    obs = [_ob(0, OrderBlockType.BULLISH, high=100, low=90, mitigated=True)]

    bars = [
        {"high": 100, "low": 90, "close": 95},     # 0: OB 봉 자체
        {"high": 98, "low": 92, "close": 95},      # 1: mitigation (wick 침범)
        {"high": 110, "low": 100, "close": 108},   # 2: departure (close > 100.1)
        {"high": 112, "low": 105, "close": 110},   # 3: 멀어짐
        {"high": 105, "low": 95, "close": 100},    # 4: retest (wick zone 진입)
    ]
    df = pd.DataFrame(bars)
    blocks = detect_mitigation_blocks(obs, df, departure_buffer_pct=0.001)
    assert len(blocks) == 1
    b = blocks[0]
    assert b.type is OrderBlockType.BULLISH
    assert b.origin_ob_idx == 0
    assert b.mitigation_idx == 1
    assert b.departure_idx == 2
    assert b.retest_idx == 4
    assert b.high == 100.0
    assert b.low == 90.0


def test_mitigation_block_bearish_full_cycle():
    """Bearish OB: 위쪽 zone, departure 는 아래로 close."""
    obs = [_ob(0, OrderBlockType.BEARISH, high=100, low=90, mitigated=True)]
    bars = [
        {"high": 100, "low": 90, "close": 95},     # 0: OB 봉
        {"high": 99, "low": 91, "close": 95},      # 1: mitigation
        {"high": 88, "low": 78, "close": 82},      # 2: departure (close < 89.9)
        {"high": 85, "low": 75, "close": 80},      # 3: 멀어짐
        {"high": 95, "low": 87, "close": 89},      # 4: retest
    ]
    df = pd.DataFrame(bars)
    blocks = detect_mitigation_blocks(obs, df)
    assert len(blocks) == 1
    assert blocks[0].type is OrderBlockType.BEARISH
    assert blocks[0].retest_idx == 4


def test_mitigation_block_no_departure_means_no_block():
    """가격이 mitigation 후 zone 안 머무름 → Mitigation Block 후보 X."""
    obs = [_ob(0, OrderBlockType.BULLISH, high=100, low=90, mitigated=True)]
    bars = [
        {"high": 100, "low": 90, "close": 95},
        {"high": 98, "low": 92, "close": 95},      # mitigation
        {"high": 99, "low": 93, "close": 96},      # zone 안에 머무름
        {"high": 98, "low": 91, "close": 94},
    ]
    df = pd.DataFrame(bars)
    blocks = detect_mitigation_blocks(obs, df)
    assert blocks == []


def test_mitigation_block_departure_no_retest_yet():
    """departure 후 아직 retest 안 됐으면 retest_idx=None (잠재 후보)."""
    obs = [_ob(0, OrderBlockType.BULLISH, high=100, low=90, mitigated=True)]
    bars = [
        {"high": 100, "low": 90, "close": 95},
        {"high": 98, "low": 92, "close": 95},      # mitigation
        {"high": 110, "low": 100, "close": 108},   # departure
        {"high": 115, "low": 109, "close": 112},   # 계속 멀어짐, retest X
    ]
    df = pd.DataFrame(bars)
    blocks = detect_mitigation_blocks(obs, df)
    assert len(blocks) == 1
    assert blocks[0].retest_idx is None
    assert filter_retested(blocks) == []


def test_filter_retested_returns_only_completed():
    """filter_retested — retest 완료된 것만 반환."""
    blocks = [
        MitigationBlock(OrderBlockType.BULLISH, 0, 1, 2, retest_idx=4, high=100, low=90),
        MitigationBlock(OrderBlockType.BEARISH, 5, 6, 7, retest_idx=None, high=110, low=100),
    ]
    filtered = filter_retested(blocks)
    assert len(filtered) == 1
    assert filtered[0].retest_idx == 4
