"""#MSS-FLIP 연구 — 보유 중 보유방향 반대 CHoCH(구조전환) 조기청산이 반등 손실을
줄이나. baseline(현행 라이브 align T2) vs +mss_flip(swing 감도별)을 고정7 IN/OUT 으로
비교. EMA flip(후행)보다 빠른 구조 기반 컷이 통하는지 정량 검증.

파트너 질문(2026-06-15): align 숏게이트가 반등을 못 따라가 숏 연쇄손절(오늘 XRP -52)
→ MSS(CHoCH)로 조기 컷하면 줄어드나. 오버피팅 체크 = IN(추세)/OUT(반등·횡보) 둘 다.

사용: PYTHONPATH=src python scripts/mss_flip_search.py [SYM]   (SYM 생략 시 고정7 전체)
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

# 현행 라이브 설정 (align T2 + min_conf4 + min_rr2.5 + sl_mult3 + ttl120 + stale30 + liq_cap).
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, entry_ttl_bars=120, setup_stale_bars=30, sl_liq_cap=True,
)

VARIANTS = {
    "baseline":  {},
    "mss_1/1":   dict(mss_flip=True, mss_swing_left=1, mss_swing_right=1),
    "mss_2/2":   dict(mss_flip=True, mss_swing_left=2, mss_swing_right=2),
    "mss_3/3":   dict(mss_flip=True, mss_swing_left=3, mss_swing_right=3),
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
    """한 심볼·페이즈군 누적 (n, net%, wins, mss_flip 수)."""
    n = 0
    net = 0.0
    w = 0
    flips = 0
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
        flips += sum(1 for t in bt.trades if t.outcome == "mss_flip")
    return n, net, w, flips


def main() -> int:
    print(f"페어: {','.join(PAIRS)}", flush=True)
    for vname, vcfg in VARIANTS.items():
        print(f"\n##### {vname} #####", flush=True)
        tin = [0, 0.0, 0, 0]
        tout = [0, 0.0, 0, 0]
        for sym in PAIRS:
            for _grp, phases, tot in (("IN", IN_PHASES, tin), ("OUT", OUT_PHASES, tout)):
                n, net, w, f = run_phase(sym, phases, vcfg)
                tot[0] += n
                tot[1] += net
                tot[2] += w
                tot[3] += f
        for label, tot in (("IN ", tin), ("OUT", tout)):
            n, net, w, f = tot
            wr = w / n * 100 if n else 0.0
            print(
                f"  {label}: n={n:4d} w={wr:4.1f}% net={net:+7.2f}% mss_flips={f}",
                flush=True,
            )
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
