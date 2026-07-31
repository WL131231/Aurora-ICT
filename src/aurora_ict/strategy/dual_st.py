"""Dual SuperTrend 추세형 전략 — 투트랙 2번째 봇(추세형)의 신호 레이어.

설계 배경:
    파트너 결정(2026-06-25): Origo(ICT 되돌림 단타) 외에 추세형 봇 1개를 투트랙으로
    추가. 외부 개발자 제공 ``매매기법.py`` 의 Dual SuperTrend 를 베이스로 하되, 백테
    (``Aurora-ICT-research/scripts/dst_trend_bt.py``, 1h 7페어 5년 + walk-forward 검증)
    로 확정한 "우리 원칙" 버전이다.

전략 요지 (2026-07-07 원본 정합 — 파트너 지시 "매매기법.py 그대로, 변형 금지"):
    - 진입: close 가 ST1(ATR14×2.0) & ST2(ATR14×3.0) **둘 다 위 = 롱 / 둘 다 아래 = 숏**.
      정렬이 새로 발생한 봉(돌파)에서만 진입. 양방향. 마감봉 기준.
    - 청산(원본 엔진): **고정 SL 2%** + **4분할 TP 1/2/3/4% ×25%** + **TP 래더 트레일**
      (TP2 체결 후 SL→TP1, TP3 체결 후 SL→TP2, TP4 전량 종료). 반대 신호 REVERSE.
    - (이력) 2026-06-25~07-07 은 트레일 ST(×6) 청산의 "우리원칙" 변형이 라이브에 있었음
      — 2026-07-07 원본 파일 수령 후 원본 엔진으로 복원. ST×6 은 연구용으로만 보존.

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
    """Dual SuperTrend 전략 설정 — 원본 ``매매기법.py`` 정합 (2026-07-07 파트너 지시).

    Attributes:
        atr_period: ATR 기간 (ST1/ST2 공통). 원본 14.
        st1_mult: 진입 판정용 ST1 ATR 배수. 원본 2.0 (타이트).
        st2_mult: 진입 판정용 ST2 ATR 배수. 원본 3.0 (넓음). 둘 다 정렬 시 진입.
        sl_pct: 고정 SL — 진입가 대비 %. 원본 2%.
        trail_trigger_target: TP 래더 트레일 시작 TP 번호. 원본 2 —
            TP2 체결 후 SL→TP1, TP3 체결 후 SL→TP2 로 계단식 이동.
        trail_enabled: 래더 트레일 on/off. 원본 True.
        trail_mult: (deprecated — 라이브 미사용) 과거 우리원칙 버전의 트레일 ST
            배수. 백테 연구 스크립트 호환용으로만 유지.
    """

    atr_period: int = 14
    st1_mult: float = 2.0
    st2_mult: float = 3.0
    sl_pct: float = 0.02
    trail_trigger_target: int = 2
    trail_enabled: bool = True
    trail_mult: float = 6.0
    # #HEIKIN-ASHI 2026-07-31 (개발자 변경사항): ST 계산·신호 판정을 하이켄아시
    # 캔들로. False = 원본(실제 캔들).
    # 백테(5년 7페어): 거래 8,151→5,591건(-31%), net -23,113%→-16,018%.
    # ⚠️ 승률 48%·RR 0.90→0.91 로 **신호 품질은 개선되지 않는다** — 개선분은 전부
    #    거래 감소에 따른 비용 절감이다(거래당 gross 0.497% vs 비용 3.33% 구조).
    #    즉 이 옵션은 "좋은 신호를 고르는" 게 아니라 "노이즈 진입을 줄이는" 장치다.
    # 체결·손익은 항상 **실제 캔들 가격**으로 이뤄진다(HA 는 신호 판정에만 사용).
    use_heikin_ashi: bool = False


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    """하이켄아시 캔들 변환 — 신호 판정 전용.

        HA_close = (O+H+L+C)/4
        HA_open  = (직전 HA_open + 직전 HA_close)/2   (첫 봉은 (O+C)/2)
        HA_high  = max(H, HA_open, HA_close)
        HA_low   = min(L, HA_open, HA_close)

    평활 효과로 추세가 매끄러워져 노이즈 반전(=휩쏘 진입)이 줄어든다.

    ⚠️ HA_open/HA_close 는 **계산값이라 실제로 거래되지 않은 가격**일 수 있다.
       따라서 이 결과는 신호 판정에만 쓰고, 진입가·손절·익절은 실제 캔들 가격을
       기준으로 삼아야 한다.

    Args:
        df: OHLCV DataFrame (시간 오름차순, open/high/low/close 필수).

    Returns:
        같은 인덱스의 HA 캔들 DataFrame. volume 이 있으면 그대로 옮긴다.
    """
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    ha_c = (o + h + low + c) / 4.0
    ha_o = np.empty(n, dtype=float)
    if n:
        ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    out = pd.DataFrame(
        {
            "open": ha_o,
            "high": np.maximum.reduce([h, ha_o, ha_c]),
            "low": np.minimum.reduce([low, ha_o, ha_c]),
            "close": ha_c,
        },
        index=df.index,
    )
    if "volume" in df:
        out["volume"] = df["volume"].to_numpy()
    return out


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
    # #HEIKIN-ASHI: ST·정렬 판정만 HA 캔들로. 반환되는 OHLC 는 **실제 캔들**이라
    # 호출부(봇/백테)의 진입가·손절·익절 계산은 실가격 기준으로 유지된다.
    src_df = heikin_ashi(df) if cfg.use_heikin_ashi else out
    out["st1"] = supertrend_line(src_df, cfg.st1_mult, cfg.atr_period)
    out["st2"] = supertrend_line(src_df, cfg.st2_mult, cfg.atr_period)
    out["trail"] = supertrend_line(src_df, cfg.trail_mult, cfg.atr_period)
    src = src_df["close"]
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
