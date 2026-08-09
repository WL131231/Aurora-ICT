"""멀티 TF 2차 — 최종설정(#EDGE-V2) 기반 TF 조합 재검증.

1차 멀티 TF 검증(multitf_search)은 구설정(conf3+RR2.0, SLx1)이었다. 최종설정
(conf4+RR2.5+SLx3+캡+신선도30)에서 "선택 TF 를 바닥으로 그 이상 TF 도 매매"
구도가 단일 5m 보다 나은지(파트너 질문)를 조합별로 잰다.

ttl 은 TF 비례(5m=120분 기준 ×24봉): 5m→120, 15m→360, 1h→1440 — 라이브
구독제의 '2시간 대기'를 TF 봉 수로 환산한 동일 비율. ttl 값이 TF 우선순위
키를 겸하므로(클수록 높은 TF) 비례 스킴이 정렬도 자연히 맞는다.

사용: python scripts/multitf_round2.py BTCUSDT  (페어당 프로세스 — 병렬)
"""
from __future__ import annotations

import collections
import sys

import pandas as pd

from aurora_ict.backtest.replay import (
    BacktestConfig,
    run_backtest,
    run_backtest_multitf,
)

SYM = sys.argv[1] if len(sys.argv) > 1 else "BTCUSDT"

IN_PHASES = [
    ("2021-11", "2021-11-10", "2021-12-22"),
    ("2023-01", "2023-01-05", "2023-02-16"),
    ("2024-02", "2024-01-25", "2024-03-08"),
]
OUT_PHASES = [
    ("2024-08", "2024-08-01", "2024-09-12"),
    ("2026", "2026-04-28", "2026-06-10"),
]

# (라벨, resample, ttl=분) — 라이브 정합: 어느 TF 든 체결 대기 2시간 고정
# (구독제 validator 가 entry_limit_ttl 7200s 를 TF 무관 강제). +1분 차등은
# 우선순위 정렬 키(ttl 큰 쪽 = 높은 TF)용 — 대기시간 차이는 무시 수준.
# 1차(TF 비례 ttl=360/1440)는 미체결 상위 TF 주문이 하루를 잠가 빈도가
# 42→9 로 붕괴 — 무효 판정 후 재설계.
TF = {
    "5m": ("5m", "5min", 120),
    "15m": ("15m", "15min", 121),
    "1h": ("1h", "1h", 122),
    "4h": ("4h", "4h", 123),
}
COMBOS = [
    ("5m(멀티엔진)", ["5m"]),
    ("5m+15m", ["5m", "15m"]),
    ("5m+15m+1h", ["5m", "15m", "1h"]),
    ("15m+1h", ["15m", "1h"]),
    ("5m~4h", ["5m", "15m", "1h", "4h"]),
]

CFG = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, entry_ttl_bars=120, setup_stale_bars=30, sl_liq_cap=True,
)


def load_slice(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{SYM}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def _agg(phases: list, specs) -> tuple[int, float, float, dict]:
    n = w = 0
    net = 0.0
    tfc: collections.Counter = collections.Counter()
    for _pn, s, e in phases:
        df = load_slice(s, e)
        if len(df) < 5000:
            continue
        if specs is None:
            bt = run_backtest(df, BacktestConfig(**CFG))
        else:
            bt = run_backtest_multitf(df, BacktestConfig(**CFG), tf_specs=specs)
        n += bt.n_trades
        net += bt.total_net_pnl_pct
        w += bt.n_wins
        tfc.update(getattr(t, "source_tf", None) for t in bt.trades)
    wr = w / n * 100 if n else 0.0
    return n, net, wr, {k: v for k, v in tfc.items() if k}


def main() -> int:
    print(f"##### {SYM} — 멀티 TF 2차 (최종설정) #####", flush=True)
    runs = [("단일 5m (현행)", None)] + [
        (name, [TF[k] for k in keys]) for name, keys in COMBOS
    ]
    for name, specs in runs:
        ni, neti, wri, tfi = _agg(IN_PHASES, specs)
        no, neto, wro, tfo = _agg(OUT_PHASES, specs)
        tf_all = collections.Counter(tfi)
        tf_all.update(tfo)
        print(
            f"  {name:14s} IN n={ni:3d} w={wri:4.1f}% net={neti:+6.2f}% | "
            f"OUT n={no:3d} w={wro:4.1f}% net={neto:+6.2f}%"
            + (f"  TF분포={dict(tf_all)}" if tf_all else ""),
            flush=True,
        )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
