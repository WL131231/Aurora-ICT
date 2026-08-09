"""#OTE 연구 — FVG 진입 깊이(0.5 CE → 0.62~0.79 더 깊은 ICT OTE)가 RR·승률을
개선하나. 고정7 IN/OUT. 깊은 진입 = RR↑(entry 가 SL 에 근접) + 체결률↓(그 깊이까지
가격이 안 올 수 있음) 트레이드오프의 순효과 측정.

파트너 질문(2026-06-15): ICT OTE(0.62~0.79) 정통 진입이 현행 50% mean 보다 나은가.
오버피팅 체크 = IN(추세)/OUT(반등·횡보) 둘 다.

사용: PYTHONPATH=src python scripts/ote_search.py [SYM]   (SYM 생략 시 고정7 전체)
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest

FIXED7 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
PAIRS = [sys.argv[1]] if len(sys.argv) > 1 else FIXED7

IN_PHASES = [
    ("2021-11", "2021-11-10", "2021-12-22"),
    ("2023-01", "2023-01-05", "2023-02-16"),
    ("2024-02", "2024-01-25", "2024-03-08"),
]
OUT_PHASES = [
    ("2024-08", "2024-08-01", "2024-09-12"),
    ("2026", "2026-04-28", "2026-06-10"),
]

BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, entry_ttl_bars=120, setup_stale_bars=30, sl_liq_cap=True,
)

VARIANTS = {
    "ote_0.50(base)": dict(ote_level=0.50),
    "ote_0.62":       dict(ote_level=0.62),
    "ote_0.705":      dict(ote_level=0.705),
    "ote_0.786":      dict(ote_level=0.786),
}


def _resample(d: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    o = d["open"].resample(rule).first()
    h = d["high"].resample(rule).max()
    lo = d["low"].resample(rule).min()
    c = d["close"].resample(rule).last()
    v = d["volume"].resample(rule).sum()
    return pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "volume": v},
    ).dropna()


def _load(sym: str, s: str, e: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[s:e]


def run_phase(sym: str, phases, vcfg: dict):
    """한 심볼·페이즈군 누적 (n, net%, wins)."""
    n = 0
    net = 0.0
    w = 0
    for _pn, s, e in phases:
        d1 = _load(sym, s, e)
        if len(d1) < 5000:
            continue
        df = _resample(d1)
        if len(df) < 700:
            continue
        bt = run_backtest(df, BacktestConfig(**{**BASE, **vcfg}))
        n += bt.n_trades
        net += bt.total_net_pnl_pct
        w += bt.n_wins
    return n, net, w


def main() -> int:
    print(f"페어: {','.join(PAIRS)}", flush=True)
    for vname, vcfg in VARIANTS.items():
        print(f"\n##### {vname} #####", flush=True)
        tin = [0, 0.0, 0]
        tout = [0, 0.0, 0]
        for sym in PAIRS:
            for _grp, phases, tot in (("IN", IN_PHASES, tin), ("OUT", OUT_PHASES, tout)):
                n, net, w = run_phase(sym, phases, vcfg)
                tot[0] += n
                tot[1] += net
                tot[2] += w
        for label, tot in (("IN ", tin), ("OUT", tout)):
            n, net, w = tot
            wr = w / n * 100 if n else 0.0
            print(f"  {label}: n={n:4d} w={wr:4.1f}% net={net:+7.2f}%", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
