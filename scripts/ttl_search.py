"""흑자 엣지 탐색 5단계 — 미체결 대기(entry_ttl) 변형 (조윤? 파트너 아이디어).

TF 클수록 봉이 느려 되돌림 타점에 늦게 닿음 → 5분 고정이면 큰 TF 에서 미체결
다발. ttl 늘리면 체결률↑(타점 더 잡음) vs 신선도↓ 트레이드오프 → net 으로 판단.
기준: align T2 + conf3 + 시간필터 + SLx2. IN/OUT 분리.
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest

SYMBOLS = sys.argv[1:] or ["BTCUSDT", "ETHUSDT"]

IN_PHASES = [
    ("2021-11 하락", "2021-11-10", "2021-12-22"),
    ("2023-01 반등", "2023-01-05", "2023-02-16"),
    ("2024-02 강상승", "2024-01-25", "2024-03-08"),
]
OUT_PHASES = [
    ("2024-08 급락반등", "2024-08-01", "2024-09-12"),
    ("2026 실거래", "2026-04-28", "2026-06-10"),
]
TTLS = [5, 15, 30, 60, 120]  # 1m봉 수 = 분 (5분~2시간)


def load_slice(sym: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def _run_set(sym: str, phases: list, ttl: int) -> tuple[int, float, int, int]:
    n = w = lo = 0
    net = 0.0
    for _pn, s, e in phases:
        df = load_slice(sym, s, e)
        cfg = BacktestConfig(
            htf_ema_bias="align", htf_align_threshold=2,
            min_confluence=3, min_rr=2.0,
            disable_time_filter=False, sl_dist_mult=2.0,
            entry_ttl_bars=ttl,
        )
        bt = run_backtest(df, cfg)
        n += bt.n_trades
        net += bt.total_net_pnl_pct
        w += bt.n_wins
        lo += bt.n_trades - bt.n_wins
    return n, net, w, lo


def main() -> int:
    for sym in SYMBOLS:
        print(f"##### {sym} — entry_ttl sweep (IN vs OUT) #####", flush=True)
        for ttl in TTLS:
            ni, neti, wi, loi = _run_set(sym, IN_PHASES, ttl)
            no, neto, wo, loo = _run_set(sym, OUT_PHASES, ttl)
            wri = wi / (wi + loi) * 100 if (wi + loi) else 0
            wro = wo / (wo + loo) * 100 if (wo + loo) else 0
            print(
                f"  ttl={ttl:3d}분 IN n={ni:3d} w={wri:4.1f}% net={neti:+6.2f}%  "
                f"| OUT n={no:3d} w={wro:4.1f}% net={neto:+6.2f}%",
                flush=True,
            )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
