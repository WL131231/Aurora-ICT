"""고정7 포트폴리오 동시 드로다운 시뮬 — '최악의 하루' 안전 점검.

7페어(BTC·ETH·SOL·XRP·DOGE·LINK·HYPE)가 BTC 상관으로 같은 날 함께 손실 날 때
계좌 단위 일일 손익 분포가 일일 손실 한도·리스크% 대비 적정한지 잰다.

모드:
    dump  <SYM>  — 페어 1개 최종설정 백테스트 → 트레이드(진입/청산 시각·손익%)
                   를 data/tmp_trades_<SYM>.json 으로 덤프 (페어당 프로세스).
    agg          — 덤프 7개를 합산: 일일 합산 손익 분포(최악일·분위수·한도 초과
                   일수), 페어 동시 보유 분포, 동시·동방향 비율.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

FIXED = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]

PHASES = [
    ("2021-11", "2021-11-10", "2021-12-22"),
    ("2023-01", "2023-01-05", "2023-02-16"),
    ("2024-02", "2024-01-25", "2024-03-08"),
    ("2024-08", "2024-08-01", "2024-09-12"),
    ("2026", "2026-04-28", "2026-06-10"),
]

CFG = dict(
    htf_ema_bias="align", htf_align_threshold=2,
    min_confluence=4, min_rr=2.5, disable_time_filter=False,
    sl_dist_mult=3.0, entry_ttl_bars=120, setup_stale_bars=30, sl_liq_cap=True,
)


def dump(sym: str) -> None:
    from aurora_ict.backtest.replay import BacktestConfig, run_backtest

    out = []
    for pname, s, e in PHASES:
        p = Path(f"data/{sym}_1m_full.parquet")
        df = pd.read_parquet(p)
        df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
        df = df[["open", "high", "low", "close", "volume"]].loc[s:e]
        if len(df) < 5000:
            continue
        bt = run_backtest(df, BacktestConfig(**CFG))
        for t in bt.trades:
            out.append({
                "phase": pname,
                "entry_ts": int(df.index[t.entry_idx].timestamp()),
                "exit_ts": int(df.index[min(t.exit_idx, len(df) - 1)].timestamp()),
                "direction": t.direction,
                "net_pnl_pct": t.net_pnl_pct,
            })
    Path(f"data/tmp_trades_{sym}.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"{sym}: {len(out)} trades dumped", flush=True)


def agg() -> None:
    trades = []
    for sym in FIXED:
        p = Path(f"data/tmp_trades_{sym}.json")
        if not p.exists():
            print(f"!! {sym} 덤프 없음 — skip", flush=True)
            continue
        for t in json.loads(p.read_text(encoding="utf-8")):
            t["sym"] = sym
            trades.append(t)
    print(f"합산 트레이드 {len(trades)}건 (7페어 × 5국면)")

    # 1) 일일 합산 손익 (청산일 기준, 계좌 % 단위 합).
    daily: dict[str, float] = defaultdict(float)
    for t in trades:
        day = pd.Timestamp(t["exit_ts"], unit="s", tz="UTC").strftime("%Y-%m-%d")
        daily[day] += t["net_pnl_pct"]
    vals = sorted(daily.values())
    n = len(vals)
    worst5 = sorted(daily.items(), key=lambda kv: kv[1])[:5]
    print(f"\n[일일 합산 손익] 거래일 {n}일")
    print(f"  최악일 5: " + ", ".join(f"{d} {v:+.2f}%" for d, v in worst5))
    for th in (-1.0, -2.0, -3.0, -5.0):
        c = sum(1 for v in vals if v <= th)
        print(f"  {th:+.0f}% 이하: {c}일 ({c / n * 100:.1f}%)" if n else "")
    if n:
        import math
        p5 = vals[max(0, math.ceil(n * 0.05) - 1)]
        print(f"  p5(하위 5% 분위): {p5:+.2f}% | 최고일: {vals[-1]:+.2f}%")

    # 2) 동시 보유 분포 — 분 단위 이벤트 스윕.
    events = []
    for t in trades:
        events.append((t["entry_ts"], 1, t["direction"]))
        events.append((t["exit_ts"], -1, t["direction"]))
    events.sort()
    cur = 0
    cur_dir: Counter = Counter()
    conc: Counter = Counter()
    max_same_dir = 0
    for _ts, d, dirn in events:
        cur += d
        cur_dir[dirn] += d
        conc[cur] += 1
        max_same_dir = max(max_same_dir, cur_dir["long"], cur_dir["short"])
    print(f"\n[동시 보유] 최대 동시 포지션: {max(conc) if conc else 0}페어"
          f" | 최대 동방향 동시: {max_same_dir}페어")

    # 3) 같은 날 다중 손실 — 한 날에 손실 트레이드 몇 개까지 겹치나.
    day_loss: dict[str, int] = defaultdict(int)
    for t in trades:
        if t["net_pnl_pct"] < 0:
            day = pd.Timestamp(t["exit_ts"], unit="s", tz="UTC").strftime("%Y-%m-%d")
            day_loss[day] += 1
    if day_loss:
        mx = max(day_loss.items(), key=lambda kv: kv[1])
        print(f"[같은 날 손실 트레이드] 최대 {mx[1]}건 ({mx[0]})")
    print("DONE", flush=True)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "dump":
        dump(sys.argv[2])
    else:
        agg()
