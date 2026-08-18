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


def _sample_frame(n: int = 400, freq: str = "3min") -> pd.DataFrame:
    """구름·매물대·2468 이 모두 잡히도록 1K 대역을 오르내리는 봉을 만든다.

    상위 TF 구름은 리샘플로 만들므로 DatetimeIndex 가 필수다 — 인덱스가 없으면
    ``base_tf_minutes`` 가 0 을 돌려 구름이 통째로 빠진다.
    """
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
    df = pd.DataFrame(rows)
    df.index = pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC")
    return df


def test_find_touches_shape_and_cooldown() -> None:
    """터치 마커는 차트용이라 (a) 형식이 맞고 (b) 봉마다 도배되면 안 된다."""
    df = _sample_frame()
    touches = cycle_levels.find_touches(df)
    assert touches, "왕복 표본에서 터치가 하나도 안 나오면 판정이 죽은 것"
    for t in touches:
        assert set(t) == {"idx", "price", "kind", "source", "tf", "count"}
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


def test_chart_markers_bundle_carries_cycle_touches() -> None:
    """UI 는 /ict/markers 응답만 보고 별을 그린다 — 필드가 빠지면 조용히 안 나온다."""
    from aurora_ict.api.markers import ChartMarkers, to_chart_markers

    assert "cycle_touches" in ChartMarkers().to_dict()

    df = _sample_frame()
    df["open"] = df["close"]
    payload = to_chart_markers(df).to_dict()["cycle_touches"]
    assert payload, "표본에 터치가 있는데 번들에 안 실렸다"
    first = payload[0]
    assert set(first) == {"ts", "price", "kind", "source", "tf", "count"}
    # 프론트가 ms → 초로 나누므로 ms 단위여야 한다(초로 주면 1970년에 찍힌다).
    assert first["ts"] > 10**12


def test_multi_tf_clouds_only_integer_multiples_with_enough_bars() -> None:
    """상위 TF 는 (a) 차트 TF 의 정수배 (b) 표본 충분 일 때만 생긴다.

    3분봉으로 5분봉은 만들 수 없고, 표본이 짧으면 1D·1W 는 구름 자체가 안 나온다 —
    그런데도 넣으면 forward-fill 이 상수 하나를 전 구간에 깔아 가짜 레벨이 된다.
    """
    df = _sample_frame(600, freq="3min")
    tfs = set(cycle_levels.multi_tf_clouds(df))
    assert "5m" not in tfs, "3분봉에서 5분봉은 만들 수 없다"
    assert "1W" not in tfs, "표본이 모자라면 빠져야 한다"
    assert tfs <= {"3m", "15m", "1h", "2h", "4h", "1D"}
    for label in tfs:
        assert cycle_levels.TF_MINUTES[label] % 3 == 0


def test_cloud_series_is_chart_ready() -> None:
    """구름 시계열은 프론트가 바로 setData 할 수 있어야 한다(ms · NaN 제거)."""
    cloud = cycle_levels.cloud_series(_sample_frame(600))
    assert cloud["a"] and cloud["b"]
    pt = cloud["a"][0]
    assert set(pt) == {"time", "value"}
    assert pt["time"] > 10**12, "ms 단위여야 한다"
    assert all(p["value"] == p["value"] for p in cloud["a"]), "NaN 이 남았다"
