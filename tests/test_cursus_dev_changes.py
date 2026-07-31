"""Cursus 개발자 변경사항 (2026-07-31) — 하이켄아시 · 페어 분리 · 노출 축소.

개발자 전달(파트너 경유):
    ① 1시간봉 기준 (현행 동일)
    ② 차트는 하이켄아시 — "눌림 가격대에 들어갈 수 있음, 지정가"
    ③ 종목 변경 — TRX 필수 추가, LINK 제외
    ④ 레버리지 10배 (기존 BTC 10x / 알트 7x)
    ⑤ 종목당 10% 진입 (기존 90%)

지정가(②의 후반부)는 "매수 지점" 기준 회신 대기라 **미적용** — 시장가 유지.

백테 근거(5년 7페어 1h, 라이브 정합 엔진):
    하이켄아시 → 거래 8,151→5,591건(-31%), net -23,113%→-16,018%.
        단 승률 48%·RR 0.90→0.91 로 **신호 품질은 불변** — 개선분은 전부 비용 절감.
    TRX vs LINK → -1,941% vs -3,571% (교체가 +1,630% 개선)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurora_ict.bot.pair_registry import (
    CURSUS_FIXED_PAIRS,
    FIXED_PAIRS,
    fixed_pairs_for_model,
)
from aurora_ict.strategy.dual_st import DualSTConfig, compute_signals, heikin_ashi


def _df(n: int = 300, seed: int = 0) -> pd.DataFrame:
    """결정론적 합성 OHLCV (mock 0 정책 — 외부 호출 없음)."""
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 1.0, n))
    return pd.DataFrame(
        {"open": c, "high": c + rng.uniform(0.2, 1.5, n),
         "low": c - rng.uniform(0.2, 1.5, n), "close": c,
         "volume": rng.uniform(10, 100, n)},
        index=idx,
    )


# ---- ③ 페어 분리 --------------------------------------------------------

def test_cursus_pairs_swap_link_to_trx() -> None:
    """Cursus 목록 = TRX 포함 · LINK 제외. 나머지 6종은 Origo 와 동일."""
    assert "TRX/USDT:USDT" in CURSUS_FIXED_PAIRS
    assert "LINK/USDT:USDT" not in CURSUS_FIXED_PAIRS
    assert set(CURSUS_FIXED_PAIRS) - {"TRX/USDT:USDT"} == \
        set(FIXED_PAIRS) - {"LINK/USDT:USDT"}


def test_origo_pairs_unchanged() -> None:
    """★ Origo 목록은 건드리지 않는다 — 백테로 확정된 7페어 유지."""
    assert "LINK/USDT:USDT" in FIXED_PAIRS
    assert "TRX/USDT:USDT" not in FIXED_PAIRS


def test_fixed_pairs_for_model_routing() -> None:
    """모델명으로 목록이 갈린다. 미지정/미등록은 Origo(기본)."""
    assert fixed_pairs_for_model("Cursus 1.0") == CURSUS_FIXED_PAIRS
    assert fixed_pairs_for_model("Origo 2.3") == FIXED_PAIRS
    assert fixed_pairs_for_model(None) == FIXED_PAIRS
    assert fixed_pairs_for_model("존재하지않는모델") == FIXED_PAIRS


# ---- ② 하이켄아시 -------------------------------------------------------

def test_heikin_ashi_formula() -> None:
    """HA 정의 검증 — close=(O+H+L+C)/4, open=직전 두 값 평균, high/low 는 포함관계."""
    df = _df(50)
    ha = heikin_ashi(df)
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    assert ha["close"].to_numpy() == pytest.approx((o + h + lo + c) / 4.0)
    assert ha["open"].iloc[0] == pytest.approx((o[0] + c[0]) / 2.0)
    assert ha["open"].iloc[1] == pytest.approx(
        (ha["open"].iloc[0] + ha["close"].iloc[0]) / 2.0)
    # high/low 는 실제 고저와 HA 몸통을 모두 포함
    assert (ha["high"] >= ha[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (ha["low"] <= ha[["open", "close"]].min(axis=1) + 1e-9).all()


def test_heikin_ashi_preserves_index_and_volume() -> None:
    df = _df(30)
    ha = heikin_ashi(df)
    assert ha.index.equals(df.index)
    assert ha["volume"].to_numpy() == pytest.approx(df["volume"].to_numpy())


def test_signals_ohlc_stays_real_price() -> None:
    """★ 핵심 — HA 를 켜도 반환 OHLC 는 **실제 가격**이어야 한다.

    진입가·손절·익절이 이 값에서 계산되므로, HA 값(계산값이라 실제 체결가가
    아닐 수 있음)이 새어 들어가면 체결 불가능한 주문이 나간다.
    """
    df = _df(200)
    out = compute_signals(df, DualSTConfig(use_heikin_ashi=True))
    for col in ("open", "high", "low", "close"):
        assert out[col].to_numpy() == pytest.approx(df[col].to_numpy())


def test_heikin_ashi_changes_signals() -> None:
    """HA 는 ST·정렬 판정을 바꾼다 — 켰을 때 신호가 달라져야 옵션이 작동한 것."""
    df = _df(600, seed=7)
    plain = compute_signals(df, DualSTConfig(use_heikin_ashi=False))
    ha = compute_signals(df, DualSTConfig(use_heikin_ashi=True))
    assert not plain["st1"].equals(ha["st1"])          # ST 라인 자체가 달라짐
    assert not plain["buy_sig"].equals(ha["buy_sig"]) or \
        not plain["sell_sig"].equals(ha["sell_sig"])   # 신호 시점도 달라짐


def test_default_is_plain_candle() -> None:
    """기본값은 원본(실제 캔들) — 하위 호환."""
    assert DualSTConfig().use_heikin_ashi is False
    df = _df(200)
    a = compute_signals(df, DualSTConfig())
    b = compute_signals(df, DualSTConfig(use_heikin_ashi=False))
    assert a["st1"].equals(b["st1"])
