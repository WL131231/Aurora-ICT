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
    high_rr_bypass_min_rr: float = 2.3
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
) -> tuple[int, float, str]:
    """진입 후 봉들에서 SL/TP 먼저 닿는 지점 → (exit_idx, exit_price, outcome).

    같은 봉에 SL·TP 둘 다 도달 시 봉 내 경로를 추정:
    bullish 봉(close>=open)은 보통 open→저점→고점→close 경로라 **저점 먼저**,
    bearish 봉은 open→고점→저점→close 라 **고점 먼저** 형성됐다고 가정한다.
    sl_priority=True 면 무조건 SL(worst-case 보수).
    """
    n = len(highs)
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
    return n - 1, float(closes[-1]), "eod"


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
        exit_idx, exit_raw, outcome = _simulate_exit(
            opens, highs, lows, closes, fill_idx, setup.direction,
            setup.stop_loss, setup.take_profit, cfg,
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
