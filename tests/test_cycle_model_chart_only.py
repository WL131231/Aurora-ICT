"""Cycle(순환매) 모델 — 차트 전용 배선 검증.

2026-08-18. Cycle 은 모델 목록에는 있지만 매매 배선이 없다. 목록에만 올린 채
두면 모델 분기가 전부 ``== "cursus"`` 이분법이라 Cycle 을 골라도 Origo 가 대신
매매한다. 실제로 MMBM 이 매니저 배선 누락으로 2주간 꺼진 채였던 전례가 있어,
"조용히 다른 모델이 도는" 상태를 테스트로 막는다.
"""
from __future__ import annotations

import pandas as pd
import pytest

from aurora_ict.config.settings import (
    AVAILABLE_MODELS,
    CHART_ONLY_MODELS,
    CURSUS_MODEL_NAME,
    CYCLE_MODEL_NAME,
    ORIGO_MODEL_NAME,
    is_chart_only_model,
)
from aurora_ict.indicators import cycle_levels


def test_cycle_is_registered_in_model_list() -> None:
    """UI 드롭다운이 AVAILABLE_MODELS 를 그대로 쓰므로 목록에 있어야 보인다."""
    assert CYCLE_MODEL_NAME in AVAILABLE_MODELS
    assert AVAILABLE_MODELS[CYCLE_MODEL_NAME] == "cycle"


def test_cycle_is_chart_only_but_others_are_not() -> None:
    """Cycle 만 기동 차단 대상 — 기존 두 모델은 영향을 받으면 안 된다."""
    assert is_chart_only_model(CYCLE_MODEL_NAME)
    assert not is_chart_only_model(ORIGO_MODEL_NAME)
    assert not is_chart_only_model(CURSUS_MODEL_NAME)
    assert not is_chart_only_model(None)
    assert not is_chart_only_model("없는 모델")


def test_chart_only_ids_are_unwired() -> None:
    """차트 전용 id 는 매매 분기가 아는 id(origo/cursus)와 겹치면 안 된다.

    겹치면 차단이 아니라 그 모델의 매매를 막아버리는 사고가 된다.
    """
    assert CHART_ONLY_MODELS.isdisjoint({"origo", "cursus"})


def _sample_frame(n: int = 400) -> pd.DataFrame:
    """구름·매물대·2468 이 모두 잡히도록 1K 대역을 오르내리는 봉을 만든다."""
    base = 90_000.0
    rows = []
    for i in range(n):
        # 600 폭으로 왕복 — 2468(+200/+400/+600/+800) 대역을 반복해서 스친다.
        mid = base + (i % 40) * 30.0
        rows.append({
            "high": mid + 40.0,
            "low": mid - 40.0,
            "close": mid + (10.0 if i % 2 else -10.0),
            "volume": 100.0 + (i % 7),
        })
    return pd.DataFrame(rows)


def test_find_touches_shape_and_cooldown() -> None:
    """터치 마커는 차트용이라 (a) 형식이 맞고 (b) 봉마다 도배되면 안 된다."""
    df = _sample_frame()
    touches = cycle_levels.find_touches(df)
    assert touches, "왕복 표본에서 터치가 하나도 안 나오면 판정이 죽은 것"
    for t in touches:
        assert set(t) == {"idx", "price", "kind", "source", "count"}
        assert t["kind"] in ("support", "resistance")
        assert 1 <= t["count"] <= 3, "겹침은 출처 3종 기준이라 3 을 넘을 수 없다"
        assert 0 <= t["idx"] < len(df)
    # 쿨다운(20봉)이 걸려 있으니 전체 봉의 절반을 넘게 찍히면 안 된다.
    assert len(touches) < len(df) / 2

    idxs = [t["idx"] for t in touches]
    assert idxs == sorted(idxs), "차트 마커는 시간 오름차순이어야 한다"


def test_find_touches_short_frame_is_empty() -> None:
    """표본이 구름 계산에 못 미치면 조용히 빈 목록 — 예외로 차트를 깨지 않는다."""
    assert cycle_levels.find_touches(_sample_frame(30)) == []


@pytest.mark.parametrize("direction,expect", [("up", (200, 400)), ("down", (600, 800))])
def test_levels_2468_direction_filter(direction: str, expect: tuple[int, int]) -> None:
    """PDF 원문 — 상방이면 2·4(+200/+400), 하방이면 6·8(+600/+800) 만 본다."""
    levels = cycle_levels.levels_2468(90_500.0, direction)
    offsets = {int(round(v)) % 1000 for v in levels}
    assert offsets == set(expect)
