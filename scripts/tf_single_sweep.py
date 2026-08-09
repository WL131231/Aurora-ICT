"""단일 매매 TF 스윕 — 5m/15m/30m/1h 각각으로 고정7 IN/OUT 흑자 검증.

파트너 질문(2026-06-13): TF 선택기를 없애고 5m 고정할지. 5m 외 단일 TF 가
흑자인지 확인. 1m parquet 을 각 TF 로 리샘플해 run_backtest (최종설정 동일).

사용: python scripts/tf_single_sweep.py <SYM>  (페어당 프로세스 — 병렬)
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest

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
TFS = [("5m","5min"),("15m","15min"),("30m","30min"),("1h","1h"),
       ("2h","2h"),("4h","4h"),("1d","1d"),("1w","1W")]

CFG = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, entry_ttl_bars=120, setup_stale_bars=30, sl_liq_cap=True,
)


def _resample(df1m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """1m → TF 봉 OHLCV (라이브 봇이 보는 매매 TF 봉 재현)."""
    o = df1m["open"].resample(rule).first()
    h = df1m["high"].resample(rule).max()
    lo = df1m["low"].resample(rule).min()
    c = df1m["close"].resample(rule).last()
    v = df1m["volume"].resample(rule).sum()
    out = pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "volume": v},
    ).dropna()
    return out


def _load_1m(start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{SYM}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def main() -> int:
    print(f"##### {SYM} — 단일 TF 스윕 (최종설정) #####", flush=True)
    for tf_label, rule in TFS:
        tot = {"in": [0, 0.0, 0], "out": [0, 0.0, 0]}
        for grp, phases in (("in", IN_PHASES), ("out", OUT_PHASES)):
            for _pn, s, e in phases:
                d1 = _load_1m(s, e)
                if len(d1) < 5000:
                    continue
                df = _resample(d1, rule)
                if len(df) < 700:  # align 최장 EMA(620) + 여유
                    continue
                bt = run_backtest(df, BacktestConfig(**CFG))
                tot[grp][0] += bt.n_trades
                tot[grp][1] += bt.total_net_pnl_pct
                tot[grp][2] += bt.n_wins
        ni, neti, wi = tot["in"]
        no, neto, wo = tot["out"]
        wri = wi / ni * 100 if ni else 0.0
        wro = wo / no * 100 if no else 0.0
        print(
            f"  {tf_label:4s} IN n={ni:3d} w={wri:4.1f}% net={neti:+6.2f}% | "
            f"OUT n={no:3d} w={wro:4.1f}% net={neto:+6.2f}%",
            flush=True,
        )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
