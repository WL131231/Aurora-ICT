"""연구5: EMA align vs MSS 구조 게이트 — "EMA 둘지/뺄지/대체할지/병행할지"를
고정7 IN/OUT 숫자로 결론. 파트너 질문(2026-06-15): EMA 와 ICT 구조신호 충돌 처리.

5변형:
  A_align       : 현행 EMA align T2 (baseline)
  B_off         : 게이트 제거 (양방향 — EMA 빼면?)
  C_align+flip  : EMA align + MSS 조기청산 (계층 분담)
  D_mss_gate    : EMA off + MSS 구조 방향 게이트 (EMA 대체)
  E_combo       : EMA align AND MSS 게이트 + MSS flip (병행 최강)

가설: EMA(느린 큰 방향)+MSS(빠른 전환)는 대체 아닌 역할분담 → C/E 우위 예상.
오버피팅 체크 = IN(추세)/OUT(반등·횡보) 둘 다.

사용: PYTHONPATH=src python scripts/ema_vs_mss_search.py [SYM]
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

# 게이트/청산 외 공통 라이브 설정.
COMMON = dict(
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, entry_ttl_bars=120, setup_stale_bars=30, sl_liq_cap=True,
)

VARIANTS = {
    "A_align(현행)":       dict(htf_ema_bias="align", htf_align_threshold=2),
    "B_EMA_off":           dict(htf_ema_bias="off"),
    "C_align+mss_flip":    dict(htf_ema_bias="align", htf_align_threshold=2, mss_flip=True),
    "D_mss_gate(EMA대체)": dict(htf_ema_bias="off", mss_bias_gate=True),
    "E_align+gate+flip":   dict(
        htf_ema_bias="align", htf_align_threshold=2,
        mss_bias_gate=True, mss_flip=True,
    ),
    "F_align+mss_fill":    dict(
        htf_ema_bias="align", htf_align_threshold=2,
        align_mss_fill=True, mss_flip=True,
    ),
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
        bt = run_backtest(df, BacktestConfig(**{**COMMON, **vcfg}))
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
