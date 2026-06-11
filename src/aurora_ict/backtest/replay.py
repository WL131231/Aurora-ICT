"""ICT 봇 백테스트 — 슬라이딩 윈도우 진입 시뮬 + PnL 집계.

1차 범위 (#BACKTEST 2026-06-06):
    detect_silver_bullet_setups 를 과거 1m OHLCV 에 슬라이딩 적용 → 가장 최근
    setup 을 stale/게이트(min_rr·min_confluence·high_rr_bypass) 통과 시 진입 →
    다음 봉들에서 SL/TP 도달 시뮬 → 슬리피지·수수료 반영 net PnL 집계.

    confluence_score 는 silver_bullet 내부(OB/macro/sweep/bias)만 반영. bot 레벨
    boost (HTF FVG / SMT / CISD) 는 2차에서 합산 예정 — 1차는 baseline.

가정/단순화:
    - 동시 포지션 1개 (청산 후 다음 진입).
    - 진입 = setup.entry(계획 limit) 즉시 체결 가정 + 진입 슬리피지 (불리 방향).
    - SL/TP = 진입 후 봉의 high/low 도달 판정. 같은 봉 동시 도달 시 SL 우선(보수).
    - 미청산 시 마지막 봉 close 강제 청산("eod").
    - size_pct 고정 (상대 비교용). risk-based sizing 은 2차.

담당: 지영민 (#BACKTEST 2026-06-06)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct
from aurora_ict.indicators.cisd import CisdType, detect_cisd
from aurora_ict.indicators.smt import SmtType, detect_smt_divergence
from aurora_ict.indicators.swing_points import detect_swing_points
from aurora_ict.strategy.silver_bullet import Direction, detect_silver_bullet_setups


@dataclass(slots=True)
class BacktestConfig:
    """백테스트 파라미터 — sweep 대상 + 시뮬/비용 설정.

    게이트/검출 파라미터는 봇 settings 와 같은 의미. 시뮬 파라미터는 백테스트 전용.
    """

    # --- 검출/게이트 (sweep 대상) ---
    min_rr: float = 2.0
    min_confluence: int = 2
    high_rr_bypass_min_rr: float = 0.0  # 라이브 settings 일치 (2026-06-06 3.0→0.0 비활성)
    setup_stale_bars: int = 120
    fvg_min_size_pct: float = 0.0006
    min_sl_distance_pct: float = 0.001
    disable_time_filter: bool = True  # 백테스트 기본 24h (시간 필터 영향 분리)
    expand_to_killzone: bool = False
    # --- bot 레벨 boost 반영 (2차-②) — confluence_score 에 가산 후 게이트 ---
    apply_cisd: bool = False  # CISD 방향 일치 시 +1 (단일 심볼)
    apply_smt: bool = False   # SMT divergence 방향 일치 시 +1 (corr_df 필요)
    # --- 시뮬/비용 ---
    window: int = 500  # 슬라이딩 lookback 봉 수 (detect 입력 길이)
    entry_ttl_bars: int = 5  # limit 체결 대기 봉 수 (entry_limit_ttl_sec 300s / 60s @1m)
    leverage: float = 20.0
    size_pct: float = 0.3  # 시드 대비 포지션 (고정 — 상대 비교용)
    # 같은 봉 SL/TP 동시 도달 처리:
    #   False(기본) = 봉 경로 휴리스틱(bullish 봉 저점 먼저 / bearish 봉 고점 먼저)
    #   True = 무조건 SL 우선(worst-case 보수) — win 과소평가
    sl_priority: bool = False
    # --- 2026-06-10 #SHORT-BIAS: 실시간 봇 HTF EMA bias 게이트 재현 ---
    # 실거래는 _passes_htf_ema_bias(1h EMA[period] vs 현재가)로 방향을 거른다.
    # 백테스트엔 그게 없어 양방향 진입 → 실거래(숏 91%)와 동작 불일치였다.
    #   "off"    = 게이트 없음(양방향, 기존 백테스트 동작)
    #   "strict" = 1h close>EMA→롱만 / close<EMA→숏만 (실시간 봇 현행)
    #   "band"   = EMA ±band_pct 완충대 안이면 양방향 허용(되돌림 whipsaw 회피)
    #   "align"  = 다중 EMA 정배열/역배열 점수 게이트 (조윤 EMA 가중치 아이디어)
    htf_ema_bias: str = "off"
    htf_ema_period: int = 20
    htf_ema_band_pct: float = 0.0  # "band" 모드 완충 폭 (예: 0.003 = 0.3%)
    # "align" 모드 — 인접 EMA 쌍 정렬 점수. 60>120>200>... 정배열이면 +1씩(롱),
    # 역배열이면 -1씩(숏). 범위 -(N-1)~+(N-1). |score|>=threshold 면 그 방향만
    # 진입, 미만이면 추세 불명확 → 진입 자제(되돌림/횡보 whipsaw 회피).
    htf_align_periods: tuple[int, ...] = (60, 120, 200, 350, 480, 620)
    htf_align_threshold: int = 1
    # 2026-06-10 조윤 동적 전환: 보유 중 EMA 정렬 점수가 보유방향과 반대로
    # |score|>=flip_threshold 강반전하면 SL/TP 기다리지 않고 그 봉 close 에서
    # 즉시 청산(outcome="flip"). 반등 조기 포착 — 다음 봉부터 재진입(게이트 적용).
    htf_align_flip: bool = False
    htf_align_flip_threshold: int = 3
    # 2026-06-11 TF 플립: run_backtest_multitf 전용. 보유 중(진입~청산 사이 매 1m
    # 봉)에 현재 보유 setup 의 TF 보다 *더 높은 TF* 에서 새 유효 setup(stale 아님 +
    # conf 게이트 + EMA/align 방향 게이트 통과)이 뜨면, 그 봉 close 에서 기존 포지션을
    # 즉시 청산(outcome="tf_flip")하고 그 HTF setup 으로 전환 진입을 시도한다(체결되면
    # 그 SL/TP 로 재보유, 더 높은 TF 뜨면 또 플립 — 반복). 방향 무관(HTF 가 더 중요).
    # False 면 기존 정적 우선 진입과 100% 동일(회귀 없음). run_backtest 엔 영향 없음.
    tf_flip: bool = False
    # 2026-06-10 흑자 탐색: SL 거리(entry~setup.stop_loss)를 mult 배 (1.0=원본,
    # >1 넓힘=stop-hunt 회피·손실 큼, <1 좁힘). tp_keeps_rr=True 면 TP 를 원 RR
    # 유지하게 재계산(SL 비례), False 면 TP 고정(RR 변동).
    sl_dist_mult: float = 1.0
    tp_keeps_rr: bool = True
    # 2026-06-12 #LIQ-CAP 재현: 라이브 hotfix 와 동일 — 확장 SL 거리를
    # 청산 거리(entry/leverage)의 80% 로 캡, 캡<원본이면 확장 포기(원본 유지).
    sl_liq_cap: bool = False


@dataclass(slots=True)
class Trade:
    """체결 1건 기록."""

    entry_idx: int
    exit_idx: int
    direction: str
    entry: float
    exit_price: float
    outcome: str  # "tp" / "sl" / "eod"
    raw_pnl_pct: float
    net_pnl_pct: float
    confluence_score: int
    # 멀티 TF 백테스트(run_backtest_multitf)에서 이 setup 이 나온 TF 라벨("5m"/"15m"/"1h").
    # 단일 TF run_backtest 는 비워 둠(None) — 기존 동작/시그니처 불변.
    source_tf: str | None = None


@dataclass(slots=True)
class BacktestResult:
    """집계 결과."""

    config: BacktestConfig
    trades: list[Trade] = field(default_factory=list)
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    total_net_pnl_pct: float = 0.0
    avg_net_pnl_pct: float = 0.0
    long_count: int = 0
    short_count: int = 0

    def summary(self) -> str:
        """한 줄 요약 (sweep 표 출력용)."""
        return (
            f"trades={self.n_trades} win={self.win_rate:.1%} "
            f"net={self.total_net_pnl_pct:+.2%} avg={self.avg_net_pnl_pct:+.3%} "
            f"L/S={self.long_count}/{self.short_count}"
        )


def load_ohlcv_parquet(path: str | Path) -> pd.DataFrame:
    """1m OHLCV parquet → DatetimeIndex(UTC) DataFrame.

    detect_silver_bullet_setups 가 fvg.ts_ms (macro confluence 등)에 index 를
    쓰므로 ts 기반 index 가 필요. columns = open/high/low/close/volume.
    """
    df = pd.read_parquet(path)
    if "timestamp" in df.columns:
        df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    cols = ["open", "high", "low", "close", "volume"]
    return df[[c for c in cols if c in df.columns]]


def _boost_score(
    base_score: int, direction: Direction, window: pd.DataFrame,
    corr_window: pd.DataFrame | None, cfg: BacktestConfig,
) -> int:
    """bot 레벨 boost(CISD/SMT)를 base confluence_score 에 가산 — bot step() 재현.

    HTF FVG boost 는 멀티 TF map 필요라 1차 boost 범위에서 제외 (별도 단계).
    """
    score = base_score
    if cfg.apply_cisd:
        cisd = detect_cisd(window)
        if cisd is not None:
            want = CisdType.BULLISH if direction is Direction.LONG else CisdType.BEARISH
            if cisd is want:
                score += 1
    if cfg.apply_smt and corr_window is not None and len(corr_window) > 0:
        swings = detect_swing_points(window)
        if len(swings) >= 2:
            events = detect_smt_divergence(swings, corr_window)
            if events:
                want = SmtType.BULLISH if direction is Direction.LONG else SmtType.BEARISH
                if events[-1].type is want:
                    score += 1
    return score


def resample_ohlcv(df: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    """1m OHLCV → 상위 TF 집계 (라이브 trade TF 5m 일치용).

    DatetimeIndex 기준 OHLC 표준 집계 (open=first, high=max, low=min, close=last).
    봉이 안 채워진 구간(NaN)은 제거.

    Args:
        df: DatetimeIndex OHLCV.
        rule: pandas resample rule (예 "5min", "15min").
    """
    agg = {
        "open": df["open"].resample(rule).first(),
        "high": df["high"].resample(rule).max(),
        "low": df["low"].resample(rule).min(),
        "close": df["close"].resample(rule).last(),
    }
    if "volume" in df.columns:
        agg["volume"] = df["volume"].resample(rule).sum()
    return pd.DataFrame(agg).dropna()


def _gate_pass(score: int, rr: float, cfg: BacktestConfig) -> bool:
    """봇 step() 의 confluence 게이트 재현 — min_confluence 또는 고RR 예외."""
    if score >= cfg.min_confluence:
        return True
    return (
        cfg.high_rr_bypass_min_rr > 0
        and score >= 1
        and rr >= cfg.high_rr_bypass_min_rr
    )


def _simulate_fill(
    highs, lows, signal_idx: int, direction: Direction,
    limit: float, ttl_bars: int,
) -> int | None:
    """limit(setup.entry) 가격이 ttl_bars 봉 내 닿으면 체결 봉 idx, 안 닿으면 None.

    ICT 는 retrace limit 진입 — 신호 후 가격이 FVG mean(entry) 까지 되돌아와야
    체결. TTL 안에 안 닿으면 타점 포기(미체결). bot 의 marketable limit + TTL 재현.

    - LONG: 가격이 내려와 low <= entry 면 체결
    - SHORT: 가격이 올라와 high >= entry 면 체결
    """
    n = len(highs)
    end = min(signal_idx + 1 + ttl_bars, n)
    for j in range(signal_idx + 1, end):
        if direction is Direction.LONG and float(lows[j]) <= limit:
            return j
        if direction is Direction.SHORT and float(highs[j]) >= limit:
            return j
    return None


def _simulate_exit(
    opens, highs, lows, closes, entry_idx: int, direction: Direction,
    sl: float, tp: float, cfg: BacktestConfig,
    align_score: Any = None,
    flip_check: Any = None,
) -> tuple[int, float, str]:
    """진입 후 봉들에서 SL/TP 먼저 닿는 지점 → (exit_idx, exit_price, outcome).

    같은 봉에 SL·TP 둘 다 도달 시 봉 내 경로를 추정:
    bullish 봉(close>=open)은 보통 open→저점→고점→close 경로라 **저점 먼저**,
    bearish 봉은 open→고점→저점→close 라 **고점 먼저** 형성됐다고 가정한다.
    sl_priority=True 면 무조건 SL(worst-case 보수).

    align_score 가 주어지고 cfg.htf_align_flip=True 면, 보유 중 EMA 정렬 점수가
    보유방향과 반대로 |score|>=flip_threshold 강반전한 봉의 close 에서 즉시 청산
    (outcome="flip") — 조윤 동적 전환. SL/TP 우선, 미도달 시 flip 판정.

    flip_check 가 주어지면(run_backtest_multitf 의 TF 플립 전용), 매 봉 SL/TP·align
    flip 미도달 시 ``flip_check(j)`` 를 호출한다. 그게 truthy(=더 높은 TF setup 발생)
    면 그 봉 close 에서 즉시 청산(outcome="tf_flip"). SL/TP 가 항상 우선. flip_check 는
    j 시점까지 닫힌 HTF 봉만 보므로 look-ahead 없음(호출부 _tf_setup_at 책임).
    run_backtest 는 flip_check 를 넘기지 않으므로 동작 불변.
    """
    n = len(highs)
    use_flip = cfg.htf_align_flip and align_score is not None
    flip_t = cfg.htf_align_flip_threshold
    for j in range(entry_idx + 1, n):
        hi, lo = float(highs[j]), float(lows[j])
        if direction is Direction.LONG:
            hit_sl, hit_tp = lo <= sl, hi >= tp
        else:
            hit_sl, hit_tp = hi >= sl, lo <= tp
        if hit_sl and hit_tp:
            if cfg.sl_priority:
                return j, sl, "sl"
            bar_up = float(closes[j]) >= float(opens[j])
            # LONG: SL=저점쪽 → bullish 봉이면 SL 먼저. SHORT: SL=고점쪽 → bearish 봉이면 SL 먼저.
            sl_first = bar_up if direction is Direction.LONG else not bar_up
            return (j, sl, "sl") if sl_first else (j, tp, "tp")
        if hit_sl:
            return j, sl, "sl"
        if hit_tp:
            return j, tp, "tp"
        # SL/TP 미도달 — EMA 정렬 점수 강반전 시 flip 청산 (조윤 동적 전환).
        if use_flip:
            sc = align_score[j]
            if sc == sc:  # NaN 아니면
                flipped = (
                    (direction is Direction.LONG and sc <= -flip_t)
                    or (direction is Direction.SHORT and sc >= flip_t)
                )
                if flipped:
                    return j, float(closes[j]), "flip"
        # SL/TP·align flip 미도달 — 더 높은 TF setup 발생 시 TF 플립 청산.
        if flip_check is not None and flip_check(j):
            return j, float(closes[j]), "tf_flip"
    return n - 1, float(closes[-1]), "eod"


def _precompute_htf_ema(df: pd.DataFrame, period: int) -> pd.Series:
    """각 1m 봉 시점의 '직전까지 확정된 1h 봉들'로 계산한 EMA 시리즈.

    look-ahead 방지: 진입 시점 i 의 1h EMA 는 그 시점에 이미 '닫힌' 1h 봉들만
    사용(진행 중 1h 봉 제외). 호출부에서 현재 1m close 와 비교 → 실시간
    ``_passes_htf_ema_bias``(닫힌 EMA vs 현재가)와 같은 의미.

    Args:
        df: 1m OHLCV (DatetimeIndex UTC).
        period: EMA 기간 (1h 봉 기준).

    Returns:
        df.index 에 정렬된 EMA 시리즈 (초기 구간은 NaN — 게이트 skip).
    """
    h1 = df["close"].resample("1h").last()
    ema_h1 = h1.ewm(span=period, adjust=False).mean()
    # 직전 확정봉 EMA (현재 진행 1h 봉은 아직 안 닫혔으니 shift 로 제외).
    ema_valid = ema_h1.shift(1)
    return ema_valid.reindex(df.index, method="ffill")


def _precompute_align_score(
    df: pd.DataFrame, periods: tuple[int, ...],
) -> pd.Series:
    """각 1m 봉 시점의 다중 EMA 정렬 점수 (조윤 EMA 가중치 아이디어).

    1h 봉 기준 EMA[periods] 를 계산하고, 인접 쌍이 정배열(짧은>긴)이면 +1,
    역배열이면 -1 누적. 전부 정배열이면 +(N-1)(강한 상승추세), 전부 역배열이면
    -(N-1)(강한 하락). 0 근처면 추세 불명확(되돌림/횡보).

    look-ahead 방지: _precompute_htf_ema 와 동일하게 직전 확정 1h 봉 기준(shift).

    Args:
        df: 1m OHLCV (DatetimeIndex UTC).
        periods: EMA 기간들 (짧은→긴 순서, 1h 봉 기준).

    Returns:
        df.index 에 정렬된 점수 시리즈 (초기 구간 NaN — 게이트 skip).
    """
    h1 = df["close"].resample("1h").last()
    emas = [h1.ewm(span=p, adjust=False).mean() for p in periods]
    score = pd.Series(0.0, index=h1.index)
    for a, b in zip(emas[:-1], emas[1:], strict=False):
        score = score + (a > b).astype(int) - (a < b).astype(int)
    # EMA 미성숙 구간 무효 처리 — ewm(adjust=False) 는 첫 봉부터 비-NaN 이라
    # notna() 마스크는 무효였다(2026-06-11 리뷰). 가장 긴 period 만큼 1h 봉이
    # 쌓이기 전엔 NaN → 게이트 skip(방향 강제 안 함). 라이브는 거래소에서
    # pmax+50 봉을 받아 항상 성숙 상태이므로, 슬라이스 초기 구간은 보수적으로
    # 게이트를 끄는 쪽이 잘못된 방향 강제보다 낫다.
    pmax = max(periods)
    mature = pd.Series(range(len(h1)), index=h1.index) >= pmax
    score = score.where(mature)
    return score.shift(1).reindex(df.index, method="ffill")


def run_backtest(
    df: pd.DataFrame, cfg: BacktestConfig, corr_df: pd.DataFrame | None = None,
) -> BacktestResult:
    """슬라이딩 윈도우 백테스트 실행.

    Args:
        df: 1m OHLCV (DatetimeIndex, columns open/high/low/close). load_ohlcv_parquet 산출.
        cfg: 파라미터.
        corr_df: SMT boost 용 상관 심볼 OHLCV (apply_smt=True 시 필요, 같은 기간 정렬).

    Returns:
        BacktestResult — per-trade 기록 + 집계.
    """
    n = len(df)
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    # #SHORT-BIAS: HTF EMA bias 게이트용 1h EMA (off 면 계산 skip).
    htf_ema = (
        _precompute_htf_ema(df, cfg.htf_ema_period).to_numpy()
        if cfg.htf_ema_bias in ("strict", "band") else None
    )
    # align 점수 — 진입 게이트("align") 또는 동적 flip 청산 중 하나라도 쓰면 계산.
    htf_align = (
        _precompute_align_score(df, cfg.htf_align_periods).to_numpy()
        if (cfg.htf_ema_bias == "align" or cfg.htf_align_flip) else None
    )
    trades: list[Trade] = []
    i = cfg.window
    while i < n - 1:
        window = df.iloc[i - cfg.window : i + 1]
        setups = detect_silver_bullet_setups(
            window,
            min_rr=cfg.min_rr,
            fvg_min_size_pct=cfg.fvg_min_size_pct,
            min_confluence=0,  # 게이트는 하니스(_gate_pass)에서 — bot step() 재현
            expand_to_killzone=cfg.expand_to_killzone,
            disable_time_filter=cfg.disable_time_filter,
            min_sl_distance_pct=cfg.min_sl_distance_pct,
        )
        if not setups:
            i += 1
            continue
        setup = setups[-1]
        bars_since = len(window) - 1 - setup.anchor_idx
        if bars_since > cfg.setup_stale_bars:
            i += 1
            continue
        # bot 레벨 boost(CISD/SMT) 반영 — confluence_score 가산 후 게이트 (bot step 재현).
        corr_window = (
            corr_df.iloc[i - cfg.window : i + 1] if corr_df is not None else None
        )
        score = _boost_score(
            setup.confluence_score, setup.direction, window, corr_window, cfg,
        )
        if not _gate_pass(score, setup.risk_reward, cfg):
            i += 1
            continue
        # #SHORT-BIAS: HTF EMA bias 방향 게이트 (실시간 _passes_htf_ema_bias 재현).
        if htf_ema is not None:
            ema_v = htf_ema[i]
            if ema_v == ema_v:  # NaN 아니면 (초기 구간 NaN 은 게이트 skip = 허용)
                cl = float(closes[i])
                band = ema_v * cfg.htf_ema_band_pct
                is_long = setup.direction is Direction.LONG
                if cfg.htf_ema_bias == "strict":
                    blocked = (cl > ema_v and not is_long) or (cl < ema_v and is_long)
                else:  # "band" — 완충대 밖에서만 방향 강제, 안이면 양방향 허용
                    blocked = (
                        (cl > ema_v + band and not is_long)
                        or (cl < ema_v - band and is_long)
                    )
                if blocked:
                    i += 1
                    continue
        # "align" — 다중 EMA 정렬 점수 게이트 (조윤 EMA 가중치).
        if htf_align is not None and cfg.htf_ema_bias == "align":
            sc = htf_align[i]
            if sc == sc:  # NaN 아니면
                is_long = setup.direction is Direction.LONG
                t = cfg.htf_align_threshold
                if sc >= t:
                    blocked = not is_long       # 상승추세 → 롱만
                elif sc <= -t:
                    blocked = is_long           # 하락추세 → 숏만
                else:
                    blocked = True              # 추세 불명확 → 진입 자제
                if blocked:
                    i += 1
                    continue
        # 체결 시뮬 — limit(setup.entry)에 ttl 봉 내 가격이 닿아야 체결. 미체결이면 skip.
        fill_idx = _simulate_fill(
            highs, lows, i, setup.direction, setup.entry, cfg.entry_ttl_bars,
        )
        if fill_idx is None:
            i += 1
            continue  # 타점 미도달 — 미체결(타점 포기)
        d_val = setup.direction.value
        # limit 체결이라 entry 슬리피지 0 (계획가 그대로). 청산만 슬리피지(시장가 발동).
        entry = setup.entry
        # 흑자 탐색: SL 거리 변형 (stop-hunt 회피 vs 손실크기 트레이드오프).
        sl, tp = setup.stop_loss, setup.take_profit
        if cfg.sl_dist_mult != 1.0:
            risk = abs(entry - sl)
            if risk > 0:
                rr0 = abs(tp - entry) / risk
                new_risk = risk * cfg.sl_dist_mult
                if cfg.sl_liq_cap and cfg.leverage > 0:
                    cap = entry * 0.8 / cfg.leverage
                    if new_risk > cap:
                        new_risk = max(risk, cap)  # 라이브 #LIQ-CAP 동일
                if setup.direction is Direction.LONG:
                    sl = entry - new_risk
                    tp = entry + new_risk * rr0 if cfg.tp_keeps_rr else tp
                else:
                    sl = entry + new_risk
                    tp = entry - new_risk * rr0 if cfg.tp_keeps_rr else tp
        exit_idx, exit_raw, outcome = _simulate_exit(
            opens, highs, lows, closes, fill_idx, setup.direction,
            sl, tp, cfg,
            align_score=htf_align if cfg.htf_align_flip else None,
        )
        exit_slip = slip_pct(
            float(highs[exit_idx]), float(lows[exit_idx]), float(closes[exit_idx]),
        )
        exit_price = apply_slippage(exit_raw, d_val, "exit", exit_slip)
        sign = 1.0 if setup.direction is Direction.LONG else -1.0
        raw_pnl_pct = (exit_price - entry) / entry * sign
        net_pnl_pct, _ = apply_costs(raw_pnl_pct, cfg.size_pct, cfg.leverage)
        trades.append(Trade(
            entry_idx=fill_idx, exit_idx=exit_idx, direction=d_val,
            entry=entry, exit_price=exit_price, outcome=outcome,
            raw_pnl_pct=raw_pnl_pct, net_pnl_pct=net_pnl_pct,
            confluence_score=score,
        ))
        i = exit_idx + 1  # 청산 후 다음 봉부터 재탐색 (동시 포지션 1개)
    return _aggregate(cfg, trades)


def _aggregate(cfg: BacktestConfig, trades: list[Trade]) -> BacktestResult:
    n_trades = len(trades)
    n_wins = sum(1 for t in trades if t.net_pnl_pct > 0)
    total = sum(t.net_pnl_pct for t in trades)
    longs = sum(1 for t in trades if t.direction == "long")
    return BacktestResult(
        config=cfg,
        trades=trades,
        n_trades=n_trades,
        n_wins=n_wins,
        win_rate=(n_wins / n_trades) if n_trades else 0.0,
        total_net_pnl_pct=total,
        avg_net_pnl_pct=(total / n_trades) if n_trades else 0.0,
        long_count=longs,
        short_count=n_trades - longs,
    )


def _detect_params(cfg: BacktestConfig) -> dict[str, Any]:
    """detect_silver_bullet_setups 결과에 영향 주는 cfg 필드만 추출.

    타임라인 캐시(build_setup_timeline)의 유효 조건 = 이 필드들이 전부 같을 것.
    conf 게이트(min_confluence)·sl_dist_mult·entry_ttl_bars·align threshold 등은
    detect 와 무관하므로 여기 포함하지 않는다 (그래서 조합 간 타임라인 공유 가능).

    Args:
        cfg: 백테스트 파라미터.

    Returns:
        detect 입력 인자에 대응하는 cfg 필드 dict (검증/기록용).
    """
    return {
        "min_rr": cfg.min_rr,
        "fvg_min_size_pct": cfg.fvg_min_size_pct,
        "expand_to_killzone": cfg.expand_to_killzone,
        "disable_time_filter": cfg.disable_time_filter,
        "min_sl_distance_pct": cfg.min_sl_distance_pct,
        "window": cfg.window,
    }


class SetupTimeline(list):
    """build_setup_timeline 산출 컨테이너 — 평범한 list + detect 파라미터 메타.

    ``timeline[i]`` = i 번째 1m 봉 시점의 detect 결과:
        - ``None``  — setup 없음 (또는 i 가 슬라이딩 범위 밖),
        - ``(setup, bars_since)`` — 가장 최근 setup + 경과 봉 수(stale 판정은
          재생 시 cfg.setup_stale_bars 와 비교해야 하므로 숫자로 보관).

    ``detect_params`` 속성에 빌드 당시 detect 인자(cfg 필드)를 기록해 둬서
    run_backtest_from_timeline 이 cfg 불일치(특히 min_rr)를 잡아낼 수 있다.
    """

    detect_params: dict[str, Any]


def build_setup_timeline(df: pd.DataFrame, cfg: BacktestConfig) -> list:
    """파라미터 그리드 가속용 setup 타임라인 빌드 — detect 결과 캐시.

    run_backtest 와 동일한 슬라이딩(각 i, cfg.window <= i < n-1)으로
    detect_silver_bullet_setups 를 1회씩 돌려 그 결과만 저장한다. 백테스트
    비용의 거의 전부가 이 detect 라서, conf 게이트·SL 배수·entry ttl·align
    threshold 처럼 detect 와 무관한 파라미터만 다른 조합들은 이 타임라인
    하나를 공유해 run_backtest_from_timeline 으로 고속 재생할 수 있다.

    주의: min_rr / fvg_min_size_pct / expand_to_killzone / disable_time_filter /
    min_sl_distance_pct / window 는 detect 인자라서 **이 중 하나라도 다르면
    타임라인을 따로 만들어야 한다** (예: rr sweep 은 rr 별로 빌드).

    Args:
        df: 1m OHLCV (DatetimeIndex UTC). load_ohlcv_parquet 산출.
        cfg: 백테스트 파라미터 — detect 관련 필드(min_rr 등)만 사용.

    Returns:
        길이 len(df) 의 SetupTimeline(list 서브클래스). timeline[i] = None 또는
        (setup, bars_since). detect_params 속성에 빌드 cfg 의 detect 필드 기록.
    """
    n = len(df)
    timeline = SetupTimeline([None] * n)
    timeline.detect_params = _detect_params(cfg)
    for i in range(cfg.window, n - 1):
        window = df.iloc[i - cfg.window : i + 1]
        setups = detect_silver_bullet_setups(
            window,
            min_rr=cfg.min_rr,
            fvg_min_size_pct=cfg.fvg_min_size_pct,
            min_confluence=0,  # 게이트는 재생(_gate_pass)에서 — run_backtest 와 동일
            expand_to_killzone=cfg.expand_to_killzone,
            disable_time_filter=cfg.disable_time_filter,
            min_sl_distance_pct=cfg.min_sl_distance_pct,
        )
        if setups:
            setup = setups[-1]
            timeline[i] = (setup, len(window) - 1 - setup.anchor_idx)
    return timeline


def run_backtest_from_timeline(
    df: pd.DataFrame,
    timeline: list,
    cfg: BacktestConfig,
    corr_df: pd.DataFrame | None = None,
) -> BacktestResult:
    """타임라인 캐시 재생 백테스트 — run_backtest 와 결과 동일, detect 만 생략.

    run_backtest 의 detect 호출을 timeline[i] 조회로 대체하고 나머지(stale 검사 →
    boost → _gate_pass → HTF EMA/align 게이트 → _simulate_fill → sl_dist_mult →
    _simulate_exit → 비용 → i=exit_idx+1 점프)는 한 줄 한 줄 동일 로직이다.
    같은 df·cfg 면 run_backtest(df, cfg) 와 Trade 리스트가 필드 단위로 일치한다.

    _boost_score 는 window 슬라이스가 필요해서 apply_cisd/apply_smt 가 켜진
    경우에만 그때 슬라이스해 호출하고, 둘 다 꺼져 있으면(no-op) skip 한다
    (run_backtest 와 score 결과 동일 — 최적화일 뿐).

    Args:
        df: 타임라인 빌드에 쓴 것과 **같은** 1m OHLCV (DatetimeIndex UTC).
        timeline: build_setup_timeline(df, build_cfg) 산출. build_cfg 의 detect
            필드(min_rr 등)가 cfg 와 일치해야 함 — detect_params 메타가 있으면
            불일치 시 ValueError.
        cfg: 재생 파라미터. detect 무관 필드(min_confluence, setup_stale_bars,
            sl_dist_mult, entry_ttl_bars, htf_* 등)는 자유롭게 바꿔도 됨.
        corr_df: SMT boost 용 상관 심볼 OHLCV (apply_smt=True 시 필요).

    Returns:
        BacktestResult — run_backtest 와 동일 집계.

    Raises:
        ValueError: 타임라인 길이가 df 와 다르거나, detect_params 메타가 cfg 의
            detect 필드와 불일치할 때 (잘못된 캐시 재사용 방지).
    """
    n = len(df)
    if len(timeline) != n:
        raise ValueError(
            f"timeline 길이({len(timeline)}) != df 길이({n}) — 같은 df 로 빌드한 "
            "타임라인을 쓸 것",
        )
    built = getattr(timeline, "detect_params", None)
    if built is not None:
        want = _detect_params(cfg)
        if built != want:
            raise ValueError(
                f"detect 파라미터 불일치 — 타임라인 빌드 {built} vs cfg {want}. "
                "min_rr 등 detect 인자가 다르면 타임라인을 따로 빌드해야 함",
            )
    opens = df["open"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    closes = df["close"].to_numpy()
    # #SHORT-BIAS: HTF EMA bias 게이트용 1h EMA (off 면 계산 skip) — run_backtest 동일.
    htf_ema = (
        _precompute_htf_ema(df, cfg.htf_ema_period).to_numpy()
        if cfg.htf_ema_bias in ("strict", "band") else None
    )
    htf_align = (
        _precompute_align_score(df, cfg.htf_align_periods).to_numpy()
        if (cfg.htf_ema_bias == "align" or cfg.htf_align_flip) else None
    )
    need_boost = cfg.apply_cisd or cfg.apply_smt  # 꺼져 있으면 _boost_score no-op → skip
    trades: list[Trade] = []
    i = cfg.window
    while i < n - 1:
        item = timeline[i]  # run_backtest 의 detect 호출 대체 (캐시 조회)
        if item is None:
            i += 1
            continue
        setup, bars_since = item
        if bars_since > cfg.setup_stale_bars:
            i += 1
            continue
        # bot 레벨 boost(CISD/SMT) — 켜진 경우만 window 슬라이스 (run_backtest 동일 의미).
        if need_boost:
            window = df.iloc[i - cfg.window : i + 1]
            corr_window = (
                corr_df.iloc[i - cfg.window : i + 1] if corr_df is not None else None
            )
            score = _boost_score(
                setup.confluence_score, setup.direction, window, corr_window, cfg,
            )
        else:
            score = setup.confluence_score
        if not _gate_pass(score, setup.risk_reward, cfg):
            i += 1
            continue
        # #SHORT-BIAS: HTF EMA bias 방향 게이트 (run_backtest 와 동일).
        if htf_ema is not None:
            ema_v = htf_ema[i]
            if ema_v == ema_v:  # NaN 아니면 (초기 구간 NaN 은 게이트 skip = 허용)
                cl = float(closes[i])
                band = ema_v * cfg.htf_ema_band_pct
                is_long = setup.direction is Direction.LONG
                if cfg.htf_ema_bias == "strict":
                    blocked = (cl > ema_v and not is_long) or (cl < ema_v and is_long)
                else:  # "band" — 완충대 밖에서만 방향 강제, 안이면 양방향 허용
                    blocked = (
                        (cl > ema_v + band and not is_long)
                        or (cl < ema_v - band and is_long)
                    )
                if blocked:
                    i += 1
                    continue
        # "align" — 다중 EMA 정렬 점수 게이트 (run_backtest 와 동일).
        if htf_align is not None and cfg.htf_ema_bias == "align":
            sc = htf_align[i]
            if sc == sc:  # NaN 아니면
                is_long = setup.direction is Direction.LONG
                t = cfg.htf_align_threshold
                if sc >= t:
                    blocked = not is_long       # 상승추세 → 롱만
                elif sc <= -t:
                    blocked = is_long           # 하락추세 → 숏만
                else:
                    blocked = True              # 추세 불명확 → 진입 자제
                if blocked:
                    i += 1
                    continue
        # 체결 시뮬 — limit(setup.entry)에 ttl 봉 내 가격이 닿아야 체결. 미체결이면 skip.
        fill_idx = _simulate_fill(
            highs, lows, i, setup.direction, setup.entry, cfg.entry_ttl_bars,
        )
        if fill_idx is None:
            i += 1
            continue  # 타점 미도달 — 미체결(타점 포기)
        d_val = setup.direction.value
        # limit 체결이라 entry 슬리피지 0 (계획가 그대로). 청산만 슬리피지(시장가 발동).
        entry = setup.entry
        # 흑자 탐색: SL 거리 변형 (stop-hunt 회피 vs 손실크기 트레이드오프).
        sl, tp = setup.stop_loss, setup.take_profit
        if cfg.sl_dist_mult != 1.0:
            risk = abs(entry - sl)
            if risk > 0:
                rr0 = abs(tp - entry) / risk
                new_risk = risk * cfg.sl_dist_mult
                if cfg.sl_liq_cap and cfg.leverage > 0:
                    cap = entry * 0.8 / cfg.leverage
                    if new_risk > cap:
                        new_risk = max(risk, cap)  # 라이브 #LIQ-CAP 동일
                if setup.direction is Direction.LONG:
                    sl = entry - new_risk
                    tp = entry + new_risk * rr0 if cfg.tp_keeps_rr else tp
                else:
                    sl = entry + new_risk
                    tp = entry - new_risk * rr0 if cfg.tp_keeps_rr else tp
        exit_idx, exit_raw, outcome = _simulate_exit(
            opens, highs, lows, closes, fill_idx, setup.direction,
            sl, tp, cfg,
            align_score=htf_align if cfg.htf_align_flip else None,
        )
        exit_slip = slip_pct(
            float(highs[exit_idx]), float(lows[exit_idx]), float(closes[exit_idx]),
        )
        exit_price = apply_slippage(exit_raw, d_val, "exit", exit_slip)
        sign = 1.0 if setup.direction is Direction.LONG else -1.0
        raw_pnl_pct = (exit_price - entry) / entry * sign
        net_pnl_pct, _ = apply_costs(raw_pnl_pct, cfg.size_pct, cfg.leverage)
        trades.append(Trade(
            entry_idx=fill_idx, exit_idx=exit_idx, direction=d_val,
            entry=entry, exit_price=exit_price, outcome=outcome,
            raw_pnl_pct=raw_pnl_pct, net_pnl_pct=net_pnl_pct,
            confluence_score=score,
        ))
        i = exit_idx + 1  # 청산 후 다음 봉부터 재탐색 (동시 포지션 1개)
    return _aggregate(cfg, trades)


# 멀티 TF 기본 스펙: (TF 라벨, resample rule, ttl_bars=체결 대기 1m 봉 수).
# ttl_bars 를 TF 봉 길이(분)와 같게 둬서 "그 TF 한 봉 동안 retrace 체결을 기다린다".
# (파트너 핵심 아이디어 — 높은 TF setup 은 더 오래 기다려 준다.)
_DEFAULT_TF_SPECS: tuple[tuple[str, str, int], ...] = (
    ("5m", "5min", 5),
    ("15m", "15min", 15),
    ("1h", "1h", 60),
)


def _latest_closed_tf_idx(
    tf_index: pd.DatetimeIndex, tf_delta: pd.Timedelta, now_ts: pd.Timestamp,
) -> int:
    """now_ts(현재 1m 봉 시각) 시점에 '이미 닫힌' 마지막 TF 봉의 위치(없으면 -1).

    look-ahead 방지의 핵심: TF 봉 라벨(open time) L 은 [L, L+tf_delta) 구간을 덮고
    L+tf_delta 에 비로소 '확정(닫힘)'된다. 따라서 현재 1m 시각 now_ts 에서 사용
    가능한 마지막 TF 봉은 ``L + tf_delta <= now_ts`` 를 만족하는 마지막 봉이다.
    진행 중(아직 안 닫힌) TF 봉은 절대 보지 않는다 → 미래 봉 참조 없음.

    Args:
        tf_index: 해당 TF DataFrame 의 DatetimeIndex (봉 open time, 오름차순).
        tf_delta: TF 한 봉 길이 (예 5min → Timedelta("5min")).
        now_ts: 현재 1m 봉의 timestamp.

    Returns:
        닫힌 마지막 TF 봉의 정수 위치. 아직 닫힌 봉이 없으면 -1.
    """
    # 닫힘 시각 = open_time + tf_delta. 그 값이 now_ts 이하인 마지막 봉을 이진탐색.
    # searchsorted(now_ts - tf_delta, "right") - 1 == (open_time <= now_ts - tf_delta)
    # 인 마지막 봉 == (open_time + tf_delta <= now_ts) 인 마지막 봉.
    cutoff = now_ts - tf_delta
    return int(tf_index.searchsorted(cutoff, side="right")) - 1


def run_backtest_multitf(
    df_1m: pd.DataFrame,
    cfg: BacktestConfig,
    tf_specs: list[tuple[str, str, int]] | None = None,
    corr_df: pd.DataFrame | None = None,
) -> BacktestResult:
    """멀티 TF setup 백테스트 — 높은 TF 우선 진입 + TF별 체결 대기.

    단일 TF(5m) setup 만 잡는 ``run_backtest`` 와 비교용. 여러 TF(기본 5m·15m·1h)
    에서 각각 ``detect_silver_bullet_setups`` 를 돌리고, 매 1m 시점마다 **높은
    TF(1h>15m>5m) 우선**으로 유효(stale 아님 + 게이트 통과) setup 을 찾아 진입한다.
    1h setup 이 있으면 그걸, 없으면 15m, 그것도 없으면 5m 을 쓴다.

    게이트/비용/시뮬은 ``run_backtest`` 와 **완전히 동일**:
        - confluence 게이트 ``_gate_pass`` (boost 는 1m 윈도우 기준 _boost_score),
        - HTF EMA bias 게이트 (``htf_ema_bias`` strict/band/align, 1m 기준 precompute),
        - SL 거리 변형 ``sl_dist_mult`` / ``tp_keeps_rr``,
        - 체결 ``_simulate_fill`` / 청산 ``_simulate_exit`` (둘 다 1m 봉 경로),
        - 슬리피지/수수료 (apply_slippage / apply_costs).
    차이는 (1) setup 을 어느 TF 에서 잡는가, (2) **진입 시 그 setup 이 나온 TF 의
    ttl_bars 로 체결 대기**한다는 점뿐.

    TF 플립 (``cfg.tf_flip=True``):
        위 정적 우선 진입은 그대로 두고, **보유 중**(진입~청산 사이 매 1m 봉) 추가로
        '현재 보유 setup 의 TF 보다 더 높은 TF' 에서 새 유효 setup(stale 아님 + conf
        게이트 + EMA/align 방향 게이트 통과)이 뜨는지 본다. 뜨면 그 봉 close 에서 기존
        포지션을 즉시 청산(``outcome="tf_flip"``)하고 그 HTF setup 으로 **전환 진입**한다
        (그 HTF ttl_bars 안에 ``_simulate_fill`` 로 체결되면 그 SL/TP 로 재보유 → 또 더
        높은 TF 뜨면 또 플립, 반복; 타점 미도달이면 청산만 하고 다음 1m 부터 재탐색).
        방향 무관(HTF 가 더 중요 — 같든 다르든 전환). 같은 HTF setup 으로의 중복 플립은
        ``entered_ts``(ts_ms) 로 막는다. 5m 보유→15m·1h 신호면 플립, 15m 보유→1h 신호면
        플립, 1h 보유 중엔 더 높은 TF 가 없어 플립 없음. ``tf_flip=False`` 면 기존 정적
        진입과 **100% 동일**(회귀 없음). ``run_backtest`` 는 영향받지 않는다.

    look-ahead 방지:
        - 각 TF setup 은 그 시점까지 **닫힌** TF 봉들로만 검출한다
          (``_latest_closed_tf_idx``: open_time+tf_delta <= 현재 1m 시각). 진행 중
          TF 봉은 절대 포함하지 않는다.
        - 체결/청산은 1m 배열에서 진입 시점 **이후** 봉만 전방 탐색.
        - HTF EMA / align 점수도 기존 함수가 직전 확정 1h 봉(shift)만 쓴다.
        - TF 플립도 동일: 보유 중 j 봉의 HTF 판정은 ``_tf_setup_at``(닫힌 HTF 봉만)을
          j 시점으로 재사용하고, 전환 체결은 j(=청산 봉) **다음** 봉부터 전방 탐색한다.

    성능:
        매 1m 마다 3개 TF 를 전부 re-detect 하면 5년치를 못 돌린다. TF별로 **그 TF
        봉이 새로 닫혔을 때만** detect 하고 그 결과를 캐시한다(``_tf_cache``). 1m 시각이
        같은 TF 봉 구간 안이면 캐시 재사용 → detect 호출 수가 1m 수가 아니라 TF 봉
        수로 줄어든다.

    Args:
        df_1m: 1m OHLCV (DatetimeIndex UTC). ``load_ohlcv_parquet`` 산출.
        cfg: 파라미터 (``run_backtest`` 와 공유). ``cfg.entry_ttl_bars`` 는 무시되고
            TF별 ``tf_specs`` 의 ttl_bars 가 쓰인다. ``cfg.window`` 는 각 TF detect 의
            lookback 봉 수로 그대로 쓴다.
        tf_specs: ``[(라벨, resample_rule, ttl_bars=분), ...]``. 기본 5m/15m/1h.
            **낮은→높은 TF 순으로 줄 것** (내부에서 진입은 높은 TF 부터 시도).
        corr_df: SMT boost 용 상관 심볼 1m OHLCV (apply_smt=True 시).

    Returns:
        BacktestResult — ``run_backtest`` 와 동일 집계. 각 Trade.source_tf 에 진입
        TF 라벨이 기록된다.
    """
    if tf_specs is None:
        tf_specs = list(_DEFAULT_TF_SPECS)
    # 진입은 높은 TF(긴 봉) 우선 — ttl_bars(분) 큰 순. 입력 순서 무관하게 정렬.
    specs_hi_to_lo = sorted(tf_specs, key=lambda s: s[2], reverse=True)

    n = len(df_1m)
    opens = df_1m["open"].to_numpy()
    highs = df_1m["high"].to_numpy()
    lows = df_1m["low"].to_numpy()
    closes = df_1m["close"].to_numpy()
    index_1m = df_1m.index

    # #SHORT-BIAS 게이트용 1m 기준 precompute (run_backtest 와 동일 — look-ahead 안전).
    htf_ema = (
        _precompute_htf_ema(df_1m, cfg.htf_ema_period).to_numpy()
        if cfg.htf_ema_bias in ("strict", "band") else None
    )
    htf_align = (
        _precompute_align_score(df_1m, cfg.htf_align_periods).to_numpy()
        if (cfg.htf_ema_bias == "align" or cfg.htf_align_flip) else None
    )

    # TF별 리샘플 + 봉 길이 Timedelta + detect 결과 캐시 준비.
    tf_frames: dict[str, dict[str, Any]] = {}
    for label, rule, ttl in specs_hi_to_lo:
        tf_df = resample_ohlcv(df_1m, rule)
        tf_frames[label] = {
            "df": tf_df,
            "delta": pd.Timedelta(rule),
            "ttl": ttl,
            "last_detect_pos": -2,        # 마지막으로 detect 한 닫힌 TF 봉 위치
            "cached_setup": None,         # 캐시된 (setup, score) — 게이트 전 raw
        }

    def info_rule(label: str) -> str:
        """label → resample rule (corr_df TF 정렬용)."""
        for lb, rule, _ in specs_hi_to_lo:
            if lb == label:
                return rule
        return "5min"

    def _tf_setup_at(label: str, now_ts: pd.Timestamp, i_1m: int):
        """해당 TF 에서 now_ts 시점에 유효한 setup (stale·게이트 통과) 또는 None.

        그 TF 봉이 새로 닫혔을 때만 detect 하고 캐시. 캐시된 setup 에 stale 검사 +
        boost + _gate_pass + EMA/align 방향 게이트를 (현재 1m 시점 기준) 적용한다.
        """
        info = tf_frames[label]
        tf_df: pd.DataFrame = info["df"]
        tf_index: pd.DatetimeIndex = tf_df.index
        closed_pos = _latest_closed_tf_idx(tf_index, info["delta"], now_ts)
        if closed_pos < 0:
            return None
        # 새 TF 봉이 닫혔을 때만 detect — 같은 봉 구간이면 캐시 재사용 (성능).
        if closed_pos != info["last_detect_pos"]:
            info["last_detect_pos"] = closed_pos
            lo = max(0, closed_pos + 1 - cfg.window)
            window = tf_df.iloc[lo : closed_pos + 1]  # 닫힌 봉까지만 (look-ahead 차단)
            setups = detect_silver_bullet_setups(
                window,
                min_rr=cfg.min_rr,
                fvg_min_size_pct=cfg.fvg_min_size_pct,
                min_confluence=0,
                expand_to_killzone=cfg.expand_to_killzone,
                disable_time_filter=cfg.disable_time_filter,
                min_sl_distance_pct=cfg.min_sl_distance_pct,
            )
            if not setups:
                info["cached_setup"] = None
            else:
                setup = setups[-1]
                bars_since = len(window) - 1 - setup.anchor_idx
                # 2026-06-11 리뷰 수정: stale 단위를 1m 환산 — bars_since 는 TF 봉
                # 수라 1h TF 면 120봉=120시간(5일)이나 유효해 HTF 쏠림 왜곡.
                # run_backtest(1m 기준 120봉=2시간)와 같은 시간 의미로 비교.
                tf_min = max(1, int(info["delta"].total_seconds() // 60))
                if bars_since * tf_min > cfg.setup_stale_bars:
                    info["cached_setup"] = None
                else:
                    # boost 는 그 TF 윈도우 기준 (run_backtest 와 동일 의미).
                    corr_win = None
                    if corr_df is not None:
                        corr_tf = resample_ohlcv(corr_df, info_rule(label))
                        corr_win = corr_tf.reindex(window.index)
                    score = _boost_score(
                        setup.confluence_score, setup.direction, window, corr_win, cfg,
                    )
                    info["cached_setup"] = (setup, score)
        cached = info["cached_setup"]
        if cached is None:
            return None
        setup, score = cached
        if not _gate_pass(score, setup.risk_reward, cfg):
            return None
        # HTF EMA bias 방향 게이트 (현재 1m 시점 — run_backtest 와 동일).
        is_long = setup.direction is Direction.LONG
        if htf_ema is not None:
            ema_v = htf_ema[i_1m]
            if ema_v == ema_v:
                cl = float(closes[i_1m])
                band = ema_v * cfg.htf_ema_band_pct
                if cfg.htf_ema_bias == "strict":
                    blocked = (cl > ema_v and not is_long) or (cl < ema_v and is_long)
                else:  # band
                    blocked = (
                        (cl > ema_v + band and not is_long)
                        or (cl < ema_v - band and is_long)
                    )
                if blocked:
                    return None
        if htf_align is not None and cfg.htf_ema_bias == "align":
            sc = htf_align[i_1m]
            if sc == sc:
                t = cfg.htf_align_threshold
                if sc >= t:
                    blocked = not is_long
                elif sc <= -t:
                    blocked = is_long
                else:
                    blocked = True
                if blocked:
                    return None
        return setup, score

    # 진입 확정 setup 의 재진입 방지(ts_ms) — 정적 진입·플립 전환 공통.
    # (TF 우선순위는 ttl_bars 크기로 직접 비교 — 큰 ttl = 더 높은 TF.)
    entered_ts: dict[str, int] = {}

    def _flip_target_at(holding_ttl: int, j: int):
        """j 봉 시점에 '현재 보유 TF 보다 더 높은 TF' 의 유효 setup → (label, ttl, res).

        TF 플립용. holding_ttl(보유 TF 의 ttl_bars)보다 큰 TF 만, 높은 TF 우선으로
        스캔해 첫 유효 setup 을 반환(없으면 None). look-ahead 방지: _tf_setup_at 이
        j 시점까지 '닫힌' HTF 봉만 보고(_latest_closed_tf_idx), EMA/align 게이트도
        j 시점 1m 값으로 평가한다. 방향 무관 — HTF 면 같은 방향이든 반대든 전환.
        같은 HTF setup(ts_ms)으로의 중복 플립은 entered_ts 로 막는다.
        """
        now_j = index_1m[j]
        for label, _rule, ttl in specs_hi_to_lo:
            if ttl <= holding_ttl:  # 더 높은 TF 만 (같거나 낮은 TF 는 플립 대상 아님)
                continue
            res = _tf_setup_at(label, now_j, j)
            if res is None:
                continue
            setup, _ = res
            sid = getattr(setup, "ts_ms", None)
            if sid is not None and entered_ts.get(label) == sid:
                continue  # 이미 그 setup 으로 진입/플립함 — 중복 방지
            return label, ttl, res
        return None

    def _exec_entry(setup, score: int, ttl: int, label: str, i_1m: int):
        """i_1m 시점 setup 을 체결·청산까지 시뮬해 Trade 1건 + 다음 탐색 위치 반환.

        반환: (trade 또는 None, next_i, flip_to). 체결 실패 시 (None, i_1m+1, None).
        cfg.tf_flip=True 면 보유 중 더 높은 TF setup 발생 시 outcome="tf_flip" 으로
        조기 청산하고, 그 HTF 전환 정보를 flip_to=(label,ttl,res) 로 돌려준다(호출부가
        전환 진입을 이어서 시도). flip_to 가 있으면 next_i 는 의미 없음(전환 우선).
        """
        sid = getattr(setup, "ts_ms", None)
        fill_idx = _simulate_fill(
            highs, lows, i_1m, setup.direction, setup.entry, ttl,
        )
        if fill_idx is None:
            return None, i_1m + 1, None
        if sid is not None:
            entered_ts[label] = sid  # 진입 확정 — 이 setup 재진입/중복플립 차단
        d_val = setup.direction.value
        entry = setup.entry
        sl, tp = setup.stop_loss, setup.take_profit
        if cfg.sl_dist_mult != 1.0:
            risk = abs(entry - sl)
            if risk > 0:
                rr0 = abs(tp - entry) / risk
                new_risk = risk * cfg.sl_dist_mult
                if cfg.sl_liq_cap and cfg.leverage > 0:
                    cap = entry * 0.8 / cfg.leverage
                    if new_risk > cap:
                        new_risk = max(risk, cap)  # 라이브 #LIQ-CAP 동일
                if setup.direction is Direction.LONG:
                    sl = entry - new_risk
                    tp = entry + new_risk * rr0 if cfg.tp_keeps_rr else tp
                else:
                    sl = entry + new_risk
                    tp = entry - new_risk * rr0 if cfg.tp_keeps_rr else tp
        # TF 플립 활성 시, 보유 중 매 봉 더 높은 TF setup 을 탐지하는 flip_check 구성.
        flip_box: dict[str, Any] = {}

        def _flip_check(jbar: int) -> bool:
            ft = _flip_target_at(ttl, jbar)
            if ft is None:
                return False
            flip_box["target"] = ft  # 전환 정보 저장 — _simulate_exit 가 이 봉서 청산
            return True

        exit_idx, exit_raw, outcome = _simulate_exit(
            opens, highs, lows, closes, fill_idx, setup.direction,
            sl, tp, cfg,
            align_score=htf_align if cfg.htf_align_flip else None,
            flip_check=_flip_check if cfg.tf_flip else None,
        )
        exit_slip = slip_pct(
            float(highs[exit_idx]), float(lows[exit_idx]), float(closes[exit_idx]),
        )
        exit_price = apply_slippage(exit_raw, d_val, "exit", exit_slip)
        sign = 1.0 if setup.direction is Direction.LONG else -1.0
        raw_pnl_pct = (exit_price - entry) / entry * sign
        net_pnl_pct, _ = apply_costs(raw_pnl_pct, cfg.size_pct, cfg.leverage)
        trade = Trade(
            entry_idx=fill_idx, exit_idx=exit_idx, direction=d_val,
            entry=entry, exit_price=exit_price, outcome=outcome,
            raw_pnl_pct=raw_pnl_pct, net_pnl_pct=net_pnl_pct,
            confluence_score=score, source_tf=label,
        )
        flip_to = flip_box.get("target") if outcome == "tf_flip" else None
        return trade, exit_idx + 1, flip_to

    trades: list[Trade] = []
    i = cfg.window
    while i < n - 1:
        now_ts = index_1m[i]
        chosen = None
        chosen_label = None
        chosen_ttl = 0
        # 높은 TF 우선 — 첫 유효 setup 에서 멈춤 (정적 우선 진입, flip on/off 공통).
        # 2026-06-11 리뷰 수정: 이미 진입한 setup(dedup)은 *그 TF 만* 건너뛰고
        # 하위 TF 로 폴백 — 기존엔 dedup 이 루프 밖이라 1h setup 하나가 캐시에
        # 살아있는 동안 하위 TF 신규 setup 까지 전부 차단됐다(거래 가뭄 왜곡).
        for label, _rule, ttl in specs_hi_to_lo:
            res = _tf_setup_at(label, now_ts, i)
            if res is None:
                continue
            sid_cand = getattr(res[0], "ts_ms", None)
            if sid_cand is not None and entered_ts.get(label) == sid_cand:
                continue  # 이 TF 의 같은 setup 재진입 방지 — 다음(하위) TF 폴백
            chosen, _score = res
            chosen_label = label
            chosen_ttl = ttl
            break
        if chosen is None:
            i += 1
            continue
        # 진입 → 보유 → 청산. cfg.tf_flip=True 면 보유 중 HTF setup 발생 시
        # outcome="tf_flip" 으로 조기청산하고 그 HTF setup 으로 전환 진입을 이어간다
        # (체결되면 또 보유 — 더 높은 TF 뜨면 또 플립, 반복 가능). 미체결이면 그냥
        # 정적 청산만 하고 다음 1m 부터 재탐색.
        cur_setup, cur_score, cur_ttl, cur_label = chosen, _score, chosen_ttl, chosen_label
        cur_i = i
        while True:
            trade, next_i, flip_to = _exec_entry(
                cur_setup, cur_score, cur_ttl, cur_label, cur_i,
            )
            if trade is not None:
                trades.append(trade)
            if flip_to is None:
                i = next_i  # 정적 청산(tp/sl/eod) 또는 미체결 — 다음 봉부터 재탐색
                break
            # tf_flip — 전환 진입을 그 청산 봉(exit_idx)에서 시도. 체결 안 되면
            # _exec_entry 가 미체결 처리 → flip_to=None 으로 루프 종료.
            flip_label, flip_ttl, (flip_setup, flip_score) = flip_to
            cur_setup, cur_score, cur_ttl, cur_label = (
                flip_setup, flip_score, flip_ttl, flip_label,
            )
            cur_i = trade.exit_idx  # 전환 setup 의 체결 대기는 청산 봉 다음부터
    return _aggregate(cfg, trades)
