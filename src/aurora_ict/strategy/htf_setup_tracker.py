"""HtfSetupTracker — Trade TF 위 모든 HTF 의 activie setup 추적.

ICT 정통 multi-TF entry framework:
1. HTF (Trade TF 위 모든 TF) 에서 setup 검출
2. LTF (사용자 trade TF) 에서 retrace + structure shift confirm → 진입

이 모듈은 (1) 만 담당. LTF entry confirmation 은 별도 (LtfEntryConfirmer).

각 HTF 별로 최신 setup 한 건만 보관. 새 setup 등장 시 교체, SL hit 시 제거.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from aurora_ict.strategy.multi_tf_bias import compute_bias_from_df
from aurora_ict.strategy.silver_bullet import (
    Direction,
    SilverBulletSetup,
    detect_silver_bullet_setups,
)

# Trade TF → 추적할 HTF 전체 list (Trade TF 위 모든 단계).
HTF_HIERARCHY: dict[str, tuple[str, ...]] = {
    "5m":  ("15m", "30m", "1h", "2h", "4h", "1d", "1w"),
    "15m": ("30m", "1h", "2h", "4h", "1d", "1w"),
    "30m": ("1h", "2h", "4h", "1d", "1w"),
    "1h":  ("2h", "4h", "1d", "1w"),
    "2h":  ("4h", "1d", "1w"),
    "4h":  ("1d", "1w"),
    "1d":  ("1w",),
    "1w":  (),
}


@dataclass(slots=True)
class HtfActiveSetup:
    """HTF 한 건의 활성 setup."""

    htf_tf: str
    setup: SilverBulletSetup

    def contains_price(self, price: float) -> bool:
        """현재 가격이 이 setup 의 zone 안에 있는지.

        v0.4.71+: setup.zone_low/high property 사용 — FVG / Turtle / Mitigation /
        Implied / Rejection 모든 source 일관 처리.
        """
        return self.setup.zone_low <= price <= self.setup.zone_high


@dataclass(slots=True)
class HtfSetupTracker:
    """각 HTF 별 최신 active setup 추적.

    Attributes:
        trade_tf: 사용자 선택 trade TF (LTF). HTF list 매핑 기준.
        min_rr: HTF setup 최소 RR.
        fvg_min_size_pct: HTF FVG 최소 % size.
    """

    trade_tf: str
    min_rr: float = 1.5
    fvg_min_size_pct: float = 0.0005
    _active: dict[str, HtfActiveSetup] = field(default_factory=dict)

    def htf_list(self) -> tuple[str, ...]:
        """이 trade_tf 의 추적 대상 HTF list."""
        return HTF_HIERARCHY.get(self.trade_tf, ())

    def update_htf(
        self, htf_tf: str, df: pd.DataFrame,
    ) -> SilverBulletSetup | None:
        """HTF 한 건 fetch 결과로 latest setup 갱신.

        Args:
            htf_tf: HTF timeframe ("15m"/"1h"/...).
            df: 해당 HTF OHLCV DataFrame.

        Returns:
            새로 등장한 setup. 같은 setup 이거나 없으면 None.
        """
        bias = compute_bias_from_df(df)
        setups = detect_silver_bullet_setups(
            df,
            bias=bias,
            min_rr=self.min_rr,
            fvg_min_size_pct=self.fvg_min_size_pct,
            disable_time_filter=True,
        )
        if not setups:
            return None
        latest = setups[-1]
        prev = self._active.get(htf_tf)
        if prev is not None and prev.setup.ts_ms == latest.ts_ms:
            return None
        self._active[htf_tf] = HtfActiveSetup(htf_tf=htf_tf, setup=latest)
        return latest

    def invalidate_if_sl_hit(self, current_price: float) -> list[str]:
        """현재 가격이 active setup 의 SL 을 침범했으면 그 setup 제거.

        Args:
            current_price: 현재 가격 (last close).

        Returns:
            제거된 HTF list.
        """
        removed: list[str] = []
        for htf, active in list(self._active.items()):
            s = active.setup
            if s.direction is Direction.LONG and current_price <= s.stop_loss:
                del self._active[htf]
                removed.append(htf)
            elif s.direction is Direction.SHORT and current_price >= s.stop_loss:
                del self._active[htf]
                removed.append(htf)
        return removed

    def invalidate_if_tp_hit(self, current_price: float) -> list[str]:
        """현재 가격이 active setup 의 TP final 도달했으면 그 setup 제거 (완성).

        Args:
            current_price: 현재 가격 (last close).

        Returns:
            제거된 HTF list.
        """
        removed: list[str] = []
        for htf, active in list(self._active.items()):
            s = active.setup
            if s.direction is Direction.LONG and current_price >= s.take_profit:
                del self._active[htf]
                removed.append(htf)
            elif s.direction is Direction.SHORT and current_price <= s.take_profit:
                del self._active[htf]
                removed.append(htf)
        return removed

    def get_active_setups(self) -> dict[str, HtfActiveSetup]:
        """현재 활성 setup dict 반환 (사본)."""
        return dict(self._active)

    def setups_containing_price(self, price: float) -> list[HtfActiveSetup]:
        """현재 가격이 FVG zone 안에 들어있는 HTF setup list.

        같은 방향 setup 만 통과 (LONG zone 안에 SHORT 가격이면 무효).
        """
        return [a for a in self._active.values() if a.contains_price(price)]


__all__ = [
    "HTF_HIERARCHY",
    "HtfActiveSetup",
    "HtfSetupTracker",
]
