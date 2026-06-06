"""Aurora-ICT 백테스트 하니스.

generate_ict_signal / detect_silver_bullet_setups 를 과거 OHLCV 에 슬라이딩
적용해 진입→SL/TP 시뮬→PnL 집계. 파라미터 sweep 으로 최적 설정 탐색.

aurora/backtest (ChoYoon, Aurora 본체) 와 별개 — ICT 신호 전용. 비용 모델
(슬리피지/수수료)만 aurora.backtest.cost 재사용.

담당: 지영민 (#BACKTEST 2026-06-06)
"""

from aurora_ict.backtest.replay import (
    BacktestConfig,
    BacktestResult,
    Trade,
    load_ohlcv_parquet,
    run_backtest,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "Trade",
    "load_ohlcv_parquet",
    "run_backtest",
]
