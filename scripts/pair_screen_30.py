"""페어 전수 스크리닝 — 30종 각 Origo 1.1 게이트 net·승률·빈도. 빈도 처방용.

빈도 연구 1차 결론: 게이트/시간필터로는 빈도 못 올림(net 박살). 빈도의 답은
net 흑자 페어를 최대한 확보(페어 수 비례 빈도↑). 보유 30종 전수로 net 흑자
페어 풀을 만들고, 목표 1일 2~4회에 필요한 페어 수를 산정. Origo 1.1 게이트
그대로: cisd+po3·conf4·rr2.5·ttl6(BTC12)·sl x3·킬존·size0.9.

Pool 병렬(단독 실행이라 경합 없음) + 진행로그. timeline 빌드 무거워 시간 걸림.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/pair_screen_30.py
"""
from __future__ import annotations

import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest  # noqa: E402

# 고정7 + 검증 알트 + 미검증 후보 (보유 데이터 30종)
PAIRS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT",
    "NEARUSDT", "ENAUSDT", "FILUSDT", "ARBUSDT", "BCHUSDT", "AVAXUSDT", "LTCUSDT",
    "ADAUSDT", "TRXUSDT", "DOTUSDT", "ATOMUSDT", "AAVEUSDT", "ETCUSDT", "OPUSDT",
    "INJUSDT", "SEIUSDT", "TIAUSDT", "WLDUSDT", "WIFUSDT",
]
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
)


def _pair(sym):
    """한 페어: Origo 1.1 게이트 백테스트 → (sym, net, 승률, 1일빈도, n)."""
    try:
        df5 = _resample(_load_full(sym))
    except Exception:
        return (sym, 0.0, 0.0, 0.0, 0)
    if len(df5) < 1400:
        return (sym, 0.0, 0.0, 0.0, 0)
    days = len(df5) / 288.0
    ttl = 12 if sym == "BTCUSDT" else 6
    bt = run_backtest(df5, BacktestConfig(**{**BASE, "entry_ttl_bars": ttl}))
    n = len(bt.trades)
    net = sum(t.net_pnl_pct for t in bt.trades)
    nwin = sum(1 for t in bt.trades if t.net_pnl_pct > 0)
    return (sym, net, (nwin / n * 100) if n else 0.0, n / days if days else 0.0, n)


def main() -> int:
    rows = []
    with Pool(6) as p:
        for r in p.imap_unordered(_pair, PAIRS):
            rows.append(r)
            print(f"  [{len(rows)}/{len(PAIRS)}] {r[0]} net{r[1]:+.0f} {r[3]:.2f}회", flush=True)

    rows.sort(key=lambda r: r[1], reverse=True)  # net 내림차순
    lines = ["===== 페어 전수 스크리닝 (Origo 1.1 게이트, 5년, net 순) ====="]
    lines.append(f"{'페어':<10} {'net%':>8} {'승률':>6} {'1일빈도':>8} {'거래':>6}")
    pos = [r for r in rows if r[1] > 0]
    for sym, net, wr, freq, n in rows:
        mark = "" if net > 0 else "  ✗적자"
        lines.append(f"{sym:<10} {net:+8.1f} {wr:5.1f}% {freq:7.2f}회 {n:5d}{mark}")

    lines.append(f"\n[net 흑자 페어 {len(pos)}종 누적 빈도 (net 순 추가)]")
    cum_f = cum_net = 0.0
    for sym, net, wr, freq, n in pos:
        cum_f += freq
        cum_net += net
        tag = " ✅2~4회" if 2.0 <= cum_f <= 4.0 else ("" if cum_f < 2.0 else " >4")
        lines.append(f"  +{sym:<10}: 누적 {cum_f:.2f}회/일  net{cum_net:+.0f}  (단독 {net:+.0f}/{wr:.0f}%){tag}")
    lines.append(f"\n흑자 페어 전체: {cum_f:.2f}회/일, net{cum_net:+.0f}")
    need = "달성" if cum_f >= 2.0 else f"부족({cum_f:.2f}<2.0) — 진입 source 추가 등 추가 레버 필요"
    lines.append(f"목표 1일 2~4회: {need}")

    txt = "\n".join(lines)
    with open("pair_screen_30_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 pair_screen_30_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
