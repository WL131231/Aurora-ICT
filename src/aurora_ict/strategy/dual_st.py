"""Dual SuperTrend 추세형 전략 — 투트랙 2번째 봇(추세형)의 신호 레이어.

설계 배경:
    파트너 결정(2026-06-25): Origo(ICT 되돌림 단타) 외에 추세형 봇 1개를 투트랙으로
    추가. 외부 개발자 제공 ``매매기법.py`` 의 Dual SuperTrend 를 베이스로 하되, 백테
    (``Aurora-ICT-research/scripts/dst_trend_bt.py``, 1h 7페어 5년 + walk-forward 검증)
    로 확정한 "우리 원칙" 버전이다.

전략 요지:
    - 진입: close 가 ST1(ATR14×2.0) & ST2(ATR14×3.0) **둘 다 위 = 롱 / 둘 다 아래 = 숏**.
      정렬이 새로 발생한 봉(돌파)에서만 진입. 양방향.
    - 청산: 별도 트레일 ST(기본 ×6, ``trail_mult``)를 추세 방향으로만 끌어올리다 가격이
      깨면 청산. 반대 신호가 뜨면 REVERSE(청산 후 역진입). **고정 TP 없음** — ST 라인이
      곧 트레일 스탑이라 손실=타이트 / 수익=추세 끝까지 → RR 비대칭.
    - 원본의 4단계 분할TP(1~4%)는 백테에서 RR<1 적자라 **제거**. 트레일링 중심이 핵심.

이 모듈은 **신호 계산만** 담당한다. 실제 진입/청산/트레일 스탑 갱신은 봇
(``BotTrendInstance``)이 거래소 인프라와 함께 수행한다. 백테(dst_trend_bt.py)와
지표·신호 로직을 100% 동일하게 유지해 백테↔실거래 정합을 보장한다.

담당: 지영민 (투트랙 추세형 봇).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from aurora_ict.strategy.silver_bullet import Direction


@dataclass(slots=True)
class DualSTConfig:
    """Dual SuperTrend 전략 설정.

    Attributes:
        atr_period: ATR 기간 (ST1/ST2/트레일 공통). 표준 14.
        st1_mult: 진입 판정용 ST1 ATR 배수. 표준 2.0 (타이트).
        st2_mult: 진입 판정용 ST2 ATR 배수. 표준 3.0 (넓음). 둘 다 정렬 시 진입.
        trail_mult: 청산 트레일 ST ATR 배수. 백테 균형점 6.0(보수) / 8.0(공격).
            좁으면 휩쏘에 자주 잘려 RR↓, 넓으면 큰 추세 끝까지 먹어 RR↑.
    """

    atr_period: int = 14
    st1_mult: float = 2.0
    st2_mult: float = 3.0
    trail_mult: float = 6.0


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Wilder ATR — (high-low, |high-prevclose|, |low-prevclose|) 의 ewm 평균.

    Args:
        df: OHLCV DataFrame (high/low/close 컬럼 필수).
        period: ATR 기간.

    Returns:
        ATR Series (df.index 정합). 앞 period-1 개는 NaN.
    """
    h, low, c = df["high"], df["low"], df["close"]
    pc = c.shift(1)
    tr = pd.concat([(h - low), (h - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def supertrend_line(df: pd.DataFrame, mult: float, period: int) -> pd.Series:
    """SuperTrend 라인 — dir -1=상승추세(라인=하단밴드) / 1=하락추세(라인=상단밴드).

    매매기법.py 차용 + 백테(dst_trend_bt.py) 검증 로직과 동일. 밴드는 추세 방향으로만
    조여(stair-step) 라인이 가격을 트레일한다.

    Args:
        df: OHLCV DataFrame.
        mult: ATR 배수 (밴드 폭).
        period: ATR 기간.

    Returns:
        SuperTrend 라인 Series (df.index 정합).
    """
    hl2 = (df["high"] + df["low"]) / 2.0
    a = atr(df, period)
    upper = (hl2 + mult * a).values
    lower = (hl2 - mult * a).values
    close = df["close"].values
    av = a.values
    n = len(df)
    up = upper.copy()
    lo = lower.copy()
    st = np.full(n, np.nan)
    dir_ = np.ones(n)
    for i in range(1, n):
        if np.isnan(av[i]):
            continue
        if not np.isnan(lo[i - 1]):
            lo[i] = lo[i] if (lo[i] > lo[i - 1] or close[i - 1] < lo[i - 1]) else lo[i - 1]
        if not np.isnan(up[i - 1]):
            up[i] = up[i] if (up[i] < up[i - 1] or close[i - 1] > up[i - 1]) else up[i - 1]
        if np.isnan(st[i - 1]) or st[i - 1] == up[i - 1]:
            dir_[i] = -1 if close[i] > up[i] else 1
        else:
            dir_[i] = 1 if close[i] < lo[i] else -1
        st[i] = lo[i] if dir_[i] == -1 else up[i]
    return pd.Series(st, index=df.index)


def compute_signals(df: pd.DataFrame, cfg: DualSTConfig) -> pd.DataFrame:
    """ST1/ST2 정렬 돌파 신호 + 트레일 ST 컬럼 추가.

    Args:
        df: OHLCV DataFrame (시간 오름차순).
        cfg: 전략 설정.

    Returns:
        입력 복사본에 st1/st2/trail/buy_sig/sell_sig 컬럼 추가.
        buy_sig: ST1·ST2 둘 다 위 정렬이 직전 봉 대비 새로 발생(롱 진입).
        sell_sig: 둘 다 아래 정렬 새로 발생(숏 진입).
        trail: 청산 트레일 스탑 기준 ST 라인(trail_mult).
    """
    out = df.copy()
    out["st1"] = supertrend_line(out, cfg.st1_mult, cfg.atr_period)
    out["st2"] = supertrend_line(out, cfg.st2_mult, cfg.atr_period)
    out["trail"] = supertrend_line(out, cfg.trail_mult, cfg.atr_period)
    src = out["close"]
    bull = (src > out["st1"]) & (src > out["st2"])
    bear = (src < out["st1"]) & (src < out["st2"])
    out["buy_sig"] = bull & ~bull.shift(1, fill_value=False)
    out["sell_sig"] = bear & ~bear.shift(1, fill_value=False)
    return out


def latest_signal(df: pd.DataFrame, cfg: DualSTConfig) -> Direction | None:
    """가장 최근(마감) 봉의 진입 신호 — 봉 종가 확정 후 봇이 호출.

    Args:
        df: OHLCV DataFrame (마지막 행 = 직전 마감 봉).
        cfg: 전략 설정.

    Returns:
        Direction.LONG(롱 진입) / Direction.SHORT(숏 진입) / None(신호 없음).
    """
    sig = compute_signals(df, cfg)
    row = sig.iloc[-1]
    if bool(row["buy_sig"]):
        return Direction.LONG
    if bool(row["sell_sig"]):
        return Direction.SHORT
    return None


def latest_trail_stop(df: pd.DataFrame, cfg: DualSTConfig) -> float:
    """가장 최근 봉의 트레일 ST 라인값 — 봇의 트레일 스탑 갱신 기준.

    봇은 이 값을 추세 방향으로만 끌어(롱=max, 숏=min) 트레일 스탑을 운용하고,
    가격이 스탑을 깨면 청산한다.

    Args:
        df: OHLCV DataFrame.
        cfg: 전략 설정.

    Returns:
        트레일 ST 라인 최신값. 계산 불가(표본 부족)면 NaN.
    """
    trail = supertrend_line(df, cfg.trail_mult, cfg.atr_period)
    return float(trail.iloc[-1])
