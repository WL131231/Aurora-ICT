"""페어 확장 빈도 — 고정7 + 검증 알트별 1일 거래수·net·승률 (Origo 1 게이트).

구독제 목표 1일 2~4회. 고정7 만으론 1일 2.7회(6/17 진단). 검증 알트
NEAR·ENA·FIL 추가 시 빈도 충족 + net 유지되는지 페어별 실측. Origo 1 운영
게이트 그대로: cisd+po3, ttl6(BTC만 12=1h), sl x3, conf4, rr2.5, size 0.9.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/pair_freq.py
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest  # noqa: E402

FIXED7 = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
CAND = ["NEARUSDT", "ENAUSDT", "FILUSDT", "ARBUSDT", "BCHUSDT"]
ALL = FIXED7 + CAND
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def _one(df5, cfg):
    bt = run_backtest(df5, cfg)
    n = len(bt.trades)
    net = sum(t.net_pnl_pct for t in bt.trades)
    nwin = sum(1 for t in bt.trades if t.net_pnl_pct > 0)
    return net, (nwin / n * 100) if n else 0.0, n


def _pair(sym):
    """한 페어: Origo 1(cisd+po3 ON) vs OFF 두 게이트 → 빈도·net·승률 비교.

    cisd+po3 가 빈도를 얼마나 희생하는지(net 대비)를 페어별로 직접 측정.
    반환: (sym, days, on(net,wr,freq,n), off(net,wr,freq,n)).
    """
    df5 = _resample(_load_full(sym))
    if len(df5) < 1400:
        return (sym, 1.0, (0.0, 0.0, 0.0, 0), (0.0, 0.0, 0.0, 0))
    days = len(df5) / 288.0
    ttl = 12 if sym == "BTCUSDT" else 6  # BTC만 1h, 나머지 30m
    on_net, on_wr, on_n = _one(df5, BacktestConfig(**{**BASE, "entry_ttl_bars": ttl}))
    off_net, off_wr, off_n = _one(
        df5, BacktestConfig(**{**BASE, "entry_ttl_bars": ttl, "apply_cisd": False, "apply_po3": False})
    )
    return (
        sym, days,
        (on_net, on_wr, on_n / days, on_n),
        (off_net, off_wr, off_n / days, off_n),
    )


def main() -> int:
    with Pool(min(6, len(ALL))) as p:
        rows = p.map(_pair, ALL)
    by = {r[0]: r for r in rows}

    lines = ["===== 페어 확장 빈도 + cisd+po3 ON/OFF 비교 (5년) ====="]
    lines.append("각 페어: [ON=cisd+po3] net/승률/1일빈도  vs  [OFF] net/승률/1일빈도")

    def block(title, syms):
        out = [f"\n[{title}]"]
        on_net = on_freq = off_net = off_freq = 0.0
        for s in syms:
            _, _, on, off = by[s]
            out.append(
                f"{s:<10} ON {on[0]:+7.1f}/{on[1]:4.1f}%/{on[2]:.2f}회   "
                f"OFF {off[0]:+7.1f}/{off[1]:4.1f}%/{off[2]:.2f}회"
            )
            on_net += on[0]; on_freq += on[2]
            off_net += off[0]; off_freq += off[2]
        out.append(
            f"{'합계':<10} ON {on_net:+7.1f}/    /{on_freq:.2f}회   "
            f"OFF {off_net:+7.1f}/    /{off_freq:.2f}회"
        )
        return out, on_net, on_freq, off_net, off_freq

    f_lines, f_on_net, f_on_freq, f_off_net, f_off_freq = block("고정7", FIXED7)
    lines += f_lines
    c_lines, *_ = block("검증 알트 후보", CAND)
    lines += c_lines

    lines.append("\n[고정7 + 후보 누적 빈도 — ON(net좋은순 추가)]")
    cand_sorted = sorted(CAND, key=lambda s: by[s][2][0], reverse=True)
    cum_freq, cum_net = f_on_freq, f_on_net
    lines.append(f"  고정7              : {cum_freq:.2f}회/일  net{cum_net:+.0f}")
    for s in cand_sorted:
        _, _, on, _ = by[s]
        cum_freq += on[2]; cum_net += on[0]
        mark = " ✅2~4회" if 2.0 <= cum_freq <= 4.0 else ""
        lines.append(f"  +{s:<14}: {cum_freq:.2f}회/일  net{cum_net:+.0f}  (단독 net{on[0]:+.0f}/{on[1]:.0f}%){mark}")

    lines.append(f"\n[빈도 trade-off 요약 (고정7)]")
    lines.append(f"  ON (cisd+po3): {f_on_freq:.2f}회/일  net{f_on_net:+.0f}")
    lines.append(f"  OFF          : {f_off_freq:.2f}회/일  net{f_off_net:+.0f}")
    if f_on_freq > 0:
        lines.append(f"  → OFF 가 빈도 {f_off_freq / f_on_freq:.1f}배, net {f_off_net:+.0f} vs {f_on_net:+.0f}")

    txt = "\n".join(lines)
    with open("pair_freq_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 pair_freq_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
