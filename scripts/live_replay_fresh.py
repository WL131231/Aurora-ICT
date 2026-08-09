"""라이브 손실 기간 정밀 재현 — 갱신 데이터(_1m, 6/3~6/18)로 라이브와 동기간 대조.

라이브 실측: net-218.2, 승29%, 진입 숏91%/롱9% (6/10~18 손실 누적).
이 기간을 Origo 1.1 게이트 백테스트로 돌려:
  - 백테 흑자+양방향 → 실행 갭(체결·청산·슬리피지·게이트차이)
  - 백테 손실+숏편향 → 시장 국면(약추세) 문제
방향 분포(롱/숏)가 라이브 숏91%와 얼마나 다른지가 prefer_direction 게이트차 단서.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/live_replay_fresh.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _resample  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def _load_1m(sym):
    df = pd.read_parquet(f"data/{sym}_1m.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]]


def main() -> int:
    print("=== 라이브 손실 기간 정밀 재현 (갱신 6/3~6/18) ===", flush=True)
    tn = tnet = tw = tl = ts = 0
    lines = []
    for sym in PAIRS:
        df5 = _resample(_load_1m(sym))
        ttl = 12 if sym == "BTCUSDT" else 6
        bt = run_backtest(df5, BacktestConfig(**{**BASE, "entry_ttl_bars": ttl}))
        n = len(bt.trades)
        net = sum(t.net_pnl_pct for t in bt.trades)
        nw = sum(1 for t in bt.trades if t.net_pnl_pct > 0)
        nl = sum(1 for t in bt.trades if t.direction == "long")
        ns = n - nl
        d = (df5.index[-1] - df5.index[0]).total_seconds() / 86400
        lines.append(
            f"{sym:<9} net{net:+6.1f} 승{nw}/{n}({100 * nw / n if n else 0:.0f}%) "
            f"롱{nl}/숏{ns} {n / d if d else 0:.2f}회/일"
        )
        tn += n; tnet += net; tw += nw; tl += nl; ts += ns
        print(f"  {sym} done net{net:+.0f} {n}건", flush=True)

    wr = (tw / tn * 100) if tn else 0.0
    lp = (tl / tn * 100) if tn else 0.0
    out = ["", "=== 갱신 데이터 15일 7페어 백테스트 ==="] + lines
    out.append(f"\n합계: net{tnet:+.1f} 승{tw}/{tn}({wr:.0f}%) 롱{tl}({lp:.0f}%)/숏{ts} {tn / 15:.2f}회/일")
    out.append("\n=== 라이브 실측 대조 ===")
    out.append("  라이브:    net-218.2  승29%  진입 숏91%/롱9%")
    out.append(f"  백테(동기간): net{tnet:+.0f}  승{wr:.0f}%  롱{lp:.0f}%/숏{100 - lp:.0f}%")
    verdict = []
    if tnet > 0:
        verdict.append("백테 흑자 → 같은 기간 엣지 작동. 라이브 손실 = 실행 갭(체결/청산/슬리피지/게이트).")
    else:
        verdict.append("백테도 손실 → 시장 국면(약추세) 문제. 엣지 맞고 추세장 회복 가능성.")
    if lp >= 20 and tn:
        verdict.append(f"백테 롱 {lp:.0f}% vs 라이브 롱 9% → 라이브 prefer_direction 이 롱을 과도 차단(숏편향) 의심.")
    else:
        verdict.append("백테도 숏편향 → 시장이 실제 하락/횡보. 방향은 정상.")
    out.append("\n→ " + "\n→ ".join(verdict))

    txt = "\n".join(out)
    with open("live_replay_fresh_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print(txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 live_replay_fresh_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
