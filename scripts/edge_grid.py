"""자율 흑자 탐색 — 종합 그리드 (conf×rr×SL×ttl), IN/OUT·BTC/ETH 4중 검증.

파트너 요청(2026-06-11): 모든 핵심 수치를 조금씩 조정하며 흑자 전환 조합을
자동 탐색. robust = IN·OUT·BTC·ETH 4개 모두 흑자(과적합 방지).
인자로 그리드 범위를 받아 세분화 라운드에 재사용.
"""
from __future__ import annotations

import sys

import pandas as pd

from aurora_ict.backtest.replay import BacktestConfig, run_backtest

IN_PHASES = [
    ("2021-11 하락", "2021-11-10", "2021-12-22"),
    ("2023-01 반등", "2023-01-05", "2023-02-16"),
    ("2024-02 강상승", "2024-01-25", "2024-03-08"),
]
OUT_PHASES = [
    ("2024-08 급락반등", "2024-08-01", "2024-09-12"),
    ("2026 실거래", "2026-04-28", "2026-06-10"),
]

# 기본 그리드 (세분화 라운드는 이 리스트만 바꿔 재실행).
CONFS = [3, 4]
RRS = [2.0, 2.5]
SLS = [1.0, 1.5, 2.0, 2.5]
TTLS = [5, 15, 30, 60]
MIN_TRADES = 25  # 이보다 적으면 통계 불신(과적합) → 표시만, robust 제외


def load_slice(sym: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]].loc[start:end]


def _net(sym: str, phases: list, c: int, r: float, sl: float, ttl: int) -> tuple[int, float]:
    n = 0
    net = 0.0
    for _pn, s, e in phases:
        df = load_slice(sym, s, e)
        bt = run_backtest(df, BacktestConfig(
            htf_ema_bias="align", htf_align_threshold=2,
            min_confluence=c, min_rr=r, disable_time_filter=False,
            sl_dist_mult=sl, entry_ttl_bars=ttl,
        ))
        n += bt.n_trades
        net += bt.total_net_pnl_pct
    return n, net


def main() -> int:
    results = []
    for c in CONFS:
        for r in RRS:
            for sl in SLS:
                for ttl in TTLS:
                    row = {"c": c, "r": r, "sl": sl, "ttl": ttl}
                    ok = True
                    for sym in ("BTCUSDT", "ETHUSDT"):
                        ni, neti = _net(sym, IN_PHASES, c, r, sl, ttl)
                        no, neto = _net(sym, OUT_PHASES, c, r, sl, ttl)
                        row[f"{sym}_in"] = neti
                        row[f"{sym}_out"] = neto
                        row[f"{sym}_n"] = ni + no
                        if not (neti > 0 and neto > 0 and ni + no >= MIN_TRADES):
                            ok = False
                    row["robust"] = ok
                    results.append(row)
                    tag = " <ROBUST흑자>" if ok else ""
                    print(
                        f"  c{c} r{r} sl{sl} ttl{ttl:3d}: "
                        f"BTC {row['BTCUSDT_in']:+.2f}/{row['BTCUSDT_out']:+.2f} "
                        f"ETH {row['ETHUSDT_in']:+.2f}/{row['ETHUSDT_out']:+.2f}"
                        f" n={row['BTCUSDT_n']}/{row['ETHUSDT_n']}{tag}",
                        flush=True,
                    )
    # 요약: OUT 합 기준 상위 + robust
    print("\n##### TOP (BTC_out+ETH_out 합 기준) #####", flush=True)
    results.sort(key=lambda x: -(x["BTCUSDT_out"] + x["ETHUSDT_out"]))
    for row in results[:8]:
        tag = " <ROBUST>" if row["robust"] else ""
        print(
            f"  c{row['c']} r{row['r']} sl{row['sl']} ttl{row['ttl']}: "
            f"out합={row['BTCUSDT_out'] + row['ETHUSDT_out']:+.2f}"
            f" (BTC {row['BTCUSDT_out']:+.2f} ETH {row['ETHUSDT_out']:+.2f}){tag}",
            flush=True,
        )
    robust = [r for r in results if r["robust"]]
    print(f"\n##### ROBUST 흑자(4중 양수) {len(robust)}개 #####", flush=True)
    for row in robust:
        print(f"  c{row['c']} r{row['r']} sl{row['sl']} ttl{row['ttl']}", flush=True)
    print("DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
