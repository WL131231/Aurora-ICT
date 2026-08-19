"""FVG (Fair Value Gap) detector — Aurora-ICT v0.1.0 첫 indicator.

ICT 핵심 PD-Array. 3봉 패턴 기반 imbalance:
- **Bullish FVG (BISI — Buyside Imbalance, Sellside Inefficiency)**: 1봉 high < 3봉 low
- **Bearish FVG (SIBI — Sellside Imbalance, Buyside Inefficiency)**: 1봉 low > 3봉 high

중간 (2번째) 봉은 보통 큰 displacement candle (장대봉). 이 봉의 wick이 1봉/3봉과
overlap되지 않아야 valid FVG — 즉 1봉 wick와 3봉 wick 사이에 빈 gap이 남는 패턴.

가격은 FVG 영역을 다시 매워줄 확률이 높다 (mean reversion = IPDA의 fair value
re-balance). 그래서 ICT 진입 시점 = FVG retest 발생 순간.

Inverse FVG (IFVG): FVG가 깨졌을 때 (close가 FVG 너머로) 발생. 첫 momentum
shift 신호.

Key threshold:
- **High** = FVG top edge (bullish 측 3봉 low / bearish 측 1봉 low)
- **Low** = FVG bottom edge (bullish 측 1봉 high / bearish 측 3봉 high)
- **Mean Threshold (50%)** = (high + low) / 2 — Consequent Encroachment (C.E)

진입 시 보통 mean threshold까지의 retest 후 reaction을 노린다.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

# ── 연구용 전역 스위치 (#FVG-BODY 2026-08-16) ──────────────────────────────
# ICT 정통은 FVG 를 **중간봉의 큰 몸통(displacement)** 으로 식별한다.
#   - Practical 4th p.37: "먼저 몸통이 가장 큰 캔들을 찾아라. 그 다음 직전 캔들의
#     고가와 다음 캔들의 저가를 표시하라"
#   - Bible p.11: 코끼리 발(자금 크기)이 물을 밀어낸다 = displacement
#   - Trading-Strategy p.5: "Displacement 는 거의 예외 없이 MSS 와 FVG 를 낳는다"
# 우리 봇은 갭 폭(min_size_pct)만 걸러 왔다 — 원인이 아니라 결과의 부산물을 본 것.
# 이 값이 설정되면 detect_fvgs 의 **모든 호출부**(silver_bullet / mmbm /
# ltf_entry_confirmer / htf_fvg_map / markers)에 몸통 필터가 함께 걸린다.
# replay 가 cfg.fvg_body_mult 로 세팅하므로 연구 스크립트는 cfg 만 바꾸면 된다.
_RESEARCH_BODY_MULT: float | None = None


class FVGType(StrEnum):
    """FVG 방향."""

    BULLISH = "bullish"  # BISI
    BEARISH = "bearish"  # SIBI


@dataclass(slots=True)
class FVG:
    """FVG 한 건의 데이터.

    Attributes:
        ts_ms: 중간 (displacement) 봉 open time (ms).
        type: bullish (BISI) / bearish (SIBI).
        high: FVG 상단 가격 (gap 위 edge).
        low: FVG 하단 가격 (gap 아래 edge).
        idx: DataFrame 상 중간 봉 index.
        filled: 추후 가격이 mean threshold까지 retest했는지.
        invalidated: close가 FVG 영역 너머로 갔는지 (IFVG 발생).
    """

    ts_ms: int
    type: FVGType
    high: float
    low: float
    idx: int
    filled: bool = False
    invalidated: bool = False

    @property
    def mean_threshold(self) -> float:
        """Consequent Encroachment (50% 라인) — 진입 시 가장 sensitive한 level."""
        return (self.high + self.low) / 2.0

    def ote_threshold(self, level: float) -> float:
        """OTE 되돌림 레벨 진입가. level=0.5=mean(CE), 0.62~0.79=더 깊은 진입.

        bullish FVG(롱)는 가격이 위로 갔다 gap 으로 되돌림 → level 클수록 low(깊은)
        쪽. bearish 는 high(깊은) 쪽. level=0.5 면 mean_threshold 와 동일.
        """
        if self.type is FVGType.BULLISH:
            return self.high - level * (self.high - self.low)
        return self.low + level * (self.high - self.low)

    @property
    def size(self) -> float:
        """FVG 폭 (가격 단위)."""
        return self.high - self.low


def detect_fvgs(
    df: pd.DataFrame,
    min_size: float | None = None,
    min_size_pct: float | None = None,
    *,
    auto_threshold: bool = False,
    auto_threshold_multiplier: float = 2.0,
    require_displacement: bool = True,
    body_mult: float | None = None,
) -> list[FVG]:
    """3봉 패턴 FVG 검출.

    Args:
        df: OHLCV DataFrame — index = timestamp (ms 또는 datetime), columns =
            ``open / high / low / close`` (volume optional). 최소 3 row 필요.
        min_size: 절대 가격 단위 최소 gap (None = 필터 미적용).
        min_size_pct: 중간 봉 close 대비 % 최소 gap (예: 0.001 = 0.1%).
            ``min_size``와 함께 지정하면 둘 다 만족해야 통과.
        auto_threshold: ``True``면 LuxAlgo SMC 표준 동적 threshold 적용 —
            ``ta.cum(abs(barDeltaPercent))/bar_index * auto_threshold_multiplier``.
            ``min_size_pct``를 동적값으로 덮어씀.
        auto_threshold_multiplier: auto_threshold 배수 (기본 2.0, LuxAlgo SMC).
        require_displacement: ``True``면 중간 봉 close 의 displacement 확인 —
            Bullish: ``close[i] > high[i-1]`` / Bearish: ``close[i] < low[i-1]``.
            False positive 감소 (LuxAlgo FVG indicator 표준).

    Returns:
        FVG list — 시간순. 빈 list = 미검출.

    Notes:
        - df.index가 DatetimeIndex면 ``ts_ms = index.astype(int) // 10**6``.
        - df.index가 int (ms)면 그대로 사용.
    """
    if len(df) < 3:
        return []

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df missing columns: {missing}")

    # ts_ms array 추출
    if isinstance(df.index, pd.DatetimeIndex):
        ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
    else:
        ts_arr = df.index.to_numpy()

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    # auto_threshold: cumulative mean of |(close-open)/open| * multiplier
    auto_thr: float | None = None
    if auto_threshold:
        opens = df["open"].to_numpy()
        # bar delta % = abs(close - open) / open, 누적 평균
        valid = opens != 0
        delta_pct = (closes - opens) / opens
        delta_pct = delta_pct * valid  # 0-division 가드
        cum_mean = float(abs(delta_pct).mean()) if len(delta_pct) > 0 else 0.0
        auto_thr = cum_mean * auto_threshold_multiplier

    # #FVG-BODY — 중간봉 몸통(displacement) 임계. 인자 > 연구 전역 순으로 적용.
    # 임계 = **그 시점까지의** 누적 평균 x 배수 — LuxAlgo 의
    # ta.cum(|barDeltaPercent|)/bar_index 와 같은 인과적 계산이다.
    # Why: 위 auto_threshold 는 df 전체 평균이라 백테에서 미래 봉이 임계에 섞인다.
    #      같은 실수를 반복하지 않으려고 별도 인자로 분리했다.
    _bm = body_mult if body_mult is not None else _RESEARCH_BODY_MULT
    body_pct = None
    body_thr = None
    if _bm:
        _opens = df["open"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            body_pct = np.abs(closes - _opens) / np.where(_opens != 0, _opens, np.nan)
        _cum = np.nancumsum(body_pct) / np.arange(1, len(body_pct) + 1)
        body_thr = _cum * _bm

    fvgs: list[FVG] = []
    for i in range(1, len(df) - 1):
        # #FVG-BODY — displacement 미달이면 방향 판정 전에 탈락 (ICT 정석 1단계)
        if body_thr is not None and not (body_pct[i] >= body_thr[i]):
            continue
        # Bullish FVG: 1봉 high < 3봉 low → 위쪽으로 gap
        if highs[i - 1] < lows[i + 1]:
            # 중간 봉 displacement: close 가 1봉 high 위
            if require_displacement and closes[i] <= highs[i - 1]:
                continue
            gap_low = float(highs[i - 1])
            gap_high = float(lows[i + 1])
            gap_size = gap_high - gap_low
            if not _passes_size(gap_size, closes[i], min_size, min_size_pct):
                continue
            if auto_thr is not None and closes[i] > 0:
                if gap_size / closes[i] < auto_thr:
                    continue
            fvgs.append(FVG(
                ts_ms=int(ts_arr[i]),
                type=FVGType.BULLISH,
                high=gap_high,
                low=gap_low,
                idx=i,
            ))
        # Bearish FVG: 1봉 low > 3봉 high → 아래쪽으로 gap
        elif lows[i - 1] > highs[i + 1]:
            if require_displacement and closes[i] >= lows[i - 1]:
                continue
            gap_high = float(lows[i - 1])
            gap_low = float(highs[i + 1])
            gap_size = gap_high - gap_low
            if not _passes_size(gap_size, closes[i], min_size, min_size_pct):
                continue
            if auto_thr is not None and closes[i] > 0:
                if gap_size / closes[i] < auto_thr:
                    continue
            fvgs.append(FVG(
                ts_ms=int(ts_arr[i]),
                type=FVGType.BEARISH,
                high=gap_high,
                low=gap_low,
                idx=i,
            ))

    return fvgs


def _passes_size(
    gap_size: float,
    mid_close: float,
    min_size: float | None,
    min_size_pct: float | None,
) -> bool:
    """절대/% 최소 size 필터 통과 검사."""
    if min_size is not None and gap_size < min_size:
        return False
    if min_size_pct is not None:
        if mid_close <= 0 or gap_size / mid_close < min_size_pct:
            return False
    return True


def mark_filled_and_invalidated(
    fvgs: list[FVG],
    df: pd.DataFrame,
) -> None:
    """FVG list에 대해 이후 가격 동작을 검사하고 flag를 갱신.

    각 FVG 이후 봉을 순회하며:
    - **filled**: 가격이 mean_threshold에 닿음 (50% retest).
      - bullish: low ≤ mean_threshold
      - bearish: high ≥ mean_threshold
    - **invalidated**: close가 FVG 영역 너머로 이탈.
      - bullish: close < low (FVG 깨짐)
      - bearish: close > high

    in-place mutation.
    """
    if not fvgs:
        return

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()

    for fvg in fvgs:
        # 중간 봉 다음다음(idx+2) 봉부터 검사
        for j in range(fvg.idx + 2, len(df)):
            if fvg.type is FVGType.BULLISH:
                if not fvg.filled and lows[j] <= fvg.mean_threshold:
                    fvg.filled = True
                if closes[j] < fvg.low:
                    fvg.invalidated = True
                    break
            else:  # BEARISH
                if not fvg.filled and highs[j] >= fvg.mean_threshold:
                    fvg.filled = True
                if closes[j] > fvg.high:
                    fvg.invalidated = True
                    break


@dataclass(slots=True)
class IFVG:
    """Inversion FVG — invalidated 된 FVG 의 반전 entry zone.

    ICT 핵심: FVG 가 깨지면 (close 가 영역 너머로) 그 FVG 는 IFVG 로 전환.
    방향이 반전 — 원래 bullish FVG 깨지면 bearish IFVG, vice versa.

    IFVG 영역은 원래 FVG 와 동일 (high/low). 가격이 retest 하러 돌아오면
    반전 진입 기회.

    Attributes:
        ts_ms: invalidation 봉 (close 가 FVG 너머로 간 봉) open time.
        type: 반전된 새 방향 (origin 의 반대).
        high: 원래 FVG 의 high.
        low: 원래 FVG 의 low.
        invalidated_idx: invalidation 봉 idx.
        origin_fvg_idx: 원래 FVG 봉 idx.
    """

    ts_ms: int
    type: FVGType
    high: float
    low: float
    invalidated_idx: int
    origin_fvg_idx: int

    @property
    def mean_threshold(self) -> float:
        return (self.high + self.low) / 2.0

    def ote_threshold(self, level: float) -> float:
        """OTE 되돌림 레벨 진입가 (FVG.ote_threshold 와 동일 규칙)."""
        if self.type is FVGType.BULLISH:
            return self.high - level * (self.high - self.low)
        return self.low + level * (self.high - self.low)

    @property
    def size(self) -> float:
        return self.high - self.low


def detect_ifvgs(
    fvgs: list[FVG],
    df: pd.DataFrame,
) -> list[IFVG]:
    """Invalidated FVG 들을 IFVG (반전) 로 변환.

    선행 조건: ``fvgs`` 는 ``mark_filled_and_invalidated`` 가 적용된 상태.

    각 invalidated FVG 마다:
    - 처음 close 가 FVG 너머로 간 봉 (invalidation 봉) 을 찾아 ts/idx 기록
    - type 을 반전 (bullish FVG → bearish IFVG, bearish FVG → bullish IFVG)

    Args:
        fvgs: ``detect_fvgs`` + ``mark_filled_and_invalidated`` 결과.
        df: 동일 OHLCV DataFrame.

    Returns:
        IFVG list — 시간순 (invalidation 봉 ts 기준).
    """
    if not fvgs or len(df) < 3:
        return []

    if isinstance(df.index, pd.DatetimeIndex):
        ts_arr = (df.index.astype("int64") // 10**6).to_numpy()
    else:
        ts_arr = df.index.to_numpy()

    closes = df["close"].to_numpy()
    ifvgs: list[IFVG] = []

    for fvg in fvgs:
        if not fvg.invalidated:
            continue
        # 첫 invalidation 봉 찾기 (close 가 FVG 너머로 간 첫 시점)
        inv_idx = -1
        for j in range(fvg.idx + 2, len(df)):
            if fvg.type is FVGType.BULLISH:
                if closes[j] < fvg.low:
                    inv_idx = j
                    break
            elif closes[j] > fvg.high:
                inv_idx = j
                break
        if inv_idx < 0:
            continue
        inverted_type = (
            FVGType.BEARISH if fvg.type is FVGType.BULLISH else FVGType.BULLISH
        )
        ifvgs.append(IFVG(
            ts_ms=int(ts_arr[inv_idx]),
            type=inverted_type,
            high=fvg.high,
            low=fvg.low,
            invalidated_idx=inv_idx,
            origin_fvg_idx=fvg.idx,
        ))

    ifvgs.sort(key=lambda f: f.ts_ms)
    return ifvgs


__all__ = [
    "FVG",
    "IFVG",
    "FVGType",
    "detect_fvgs",
    "detect_ifvgs",
    "mark_filled_and_invalidated",
]
