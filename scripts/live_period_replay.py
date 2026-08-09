"""라이브 기간 백테스트 재현 — 최근 ~20일 7페어를 Origo 1.1 게이트로 돌려
라이브 손실(숏91%·net-218·승29%)이 국면 문제인지 실행 갭인지 가른다.

- 백테스트도 최근 20일 손실+숏편향 → 시장 국면(약추세) 문제, 엣지는 맞고 기다림
- 백테스트는 흑자+양방향인데 라이브만 손실 → 실행 갭(체결·청산·슬리피지)

방향 분포·net·승률·빈도를 라이브와 직접 대조. 데이터 마지막 날짜도 출력(최신성).
Origo 1.1 게이트: cisd+po3·conf4·rr2.5·ttl6(BTC12)·sl x3·킬존·size0.9.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/live_period_replay.py
"""
from __future__ import annotations

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)
DAYS = 20


def main() -> int:
    print(f"=== 라이브 기간 재현 (최근 {DAYS}일, Origo 1.1 게이트) ===", flush=True)
    tot_net = tot_n = tot_win = tot_long = tot_short = 0
    last_dt = None
    lines = []
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        last = df5.index[-1]
        last_dt = max(last_dt, last) if last_dt else last
        recent = df5[df5.index >= last - pd.Timedelta(days=DAYS)]
        if len(recent) < 600:
            lines.append(f"{sym:<9} 데이터 부족({len(recent)}봉)")
            continue
        ttl = 12 if sym == "BTCUSDT" else 6
        bt = run_backtest(recent, BacktestConfig(**{**BASE, "entry_ttl_bars": ttl}))
        n = len(bt.trades)
        net = sum(t.net_pnl_pct for t in bt.trades)
        nwin = sum(1 for t in bt.trades if t.net_pnl_pct > 0)
        nl = sum(1 for t in bt.trades if t.direction == "long")
        ns = n - nl
        d = (recent.index[-1] - recent.index[0]).total_seconds() / 86400
        lines.append(
            f"{sym:<9} net{net:+6.1f} 승{nwin}/{n}({100 * nwin / n if n else 0:.0f}%) "
            f"롱{nl}/숏{ns} {n / d if d else 0:.2f}회/일"
        )
        tot_net += net; tot_n += n; tot_win += nwin; tot_long += nl; tot_short += ns
        print(f"  {sym} done (net{net:+.0f}, {n}건)", flush=True)

    out = ["", f"데이터 최신: {last_dt}", "", f"=== 최근 {DAYS}일 7페어 백테스트 ==="]
    out += lines
    wr = (tot_win / tot_n * 100) if tot_n else 0.0
    lp = (tot_long / tot_n * 100) if tot_n else 0.0
    out.append(f"\n합계: net{tot_net:+.1f} 승{tot_win}/{tot_n}({wr:.0f}%) 롱{tot_long}({lp:.0f}%)/숏{tot_short} {tot_n / DAYS:.2f}회/일")
    out.append("\n=== 라이브 실측(본인, 전체 청산) 대조 ===")
    out.append("  라이브: net-218.2 승12/42(29%) 진입 숏91%/롱9%")
    out.append(f"  백테스트(최근{DAYS}일): net{tot_net:+.0f} 승률{wr:.0f}% 롱{lp:.0f}%")
    if tot_net > 0:
        out.append("\n→ 백테스트 흑자 = 같은 기간 엣지 작동. 라이브 손실은 실행 갭(체결·청산·슬리피지·표본) 의심.")
    else:
        out.append("\n→ 백테스트도 손실 = 시장 국면(약추세) 문제. 엣지는 맞고 추세장 오면 회복 가능성.")

    txt = "\n".join(out)
    with open("live_period_replay_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print(txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 live_period_replay_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
