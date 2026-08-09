"""ttl sweep — cisd+po3 위에서 진입대기(ttl)별 net + 빈도 트레이드오프.

핵심 질문(2026-06-17): ttl 30분=흑자but빈도1일0.33회 / ttl 2h(라이브)=빈도1일2.7회.
ttl 을 키우면 빈도↑ 하는데 net 이 흑자를 유지하는가? = 빈도+흑자 동시 가능 지점.
trail 은 7페어 net 손해로 기각 → trail 없이 ttl 만.

ttl 은 replay 단계 파라미터(timeline 재사용) → timeline 1개 + ttl 변형 재생.
BTC 단독 서치(파트너 지시). n→1일 환산(7페어 합산 추정 ×7).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/ttl_sweep_5y.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import run_parallel  # noqa: E402

SYM = ["BTCUSDT"]
DAYS_5Y = 1825
N_PAIRS = 7  # 7페어 합산 환산

# 라이브 구독제 정합 (trail 없음 — 기각). ttl 만 변형.
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=4, min_rr=2.5, sl_dist_mult=3.0,
    setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False,
    size_pct=0.9,
)

# ttl 봉수 (5분봉): 6=30분, 12=1h, 18=1.5h, 24=2h(라이브), 36=3h
VARIANTS = {
    "ttl6_30m":  dict(entry_ttl_bars=6),
    "ttl12_1h":  dict(entry_ttl_bars=12),
    "ttl18_90m": dict(entry_ttl_bars=18),
    "ttl24_2h":  dict(entry_ttl_bars=24),
    "ttl36_3h":  dict(entry_ttl_bars=36),
}


def main() -> int:
    totals, per_sym = run_parallel(SYM, BASE, VARIANTS, nproc=5)
    rows = []
    for vn, (n, net, w) in totals.items():
        wr = w / n * 100 if n else 0.0
        perday = n / DAYS_5Y * N_PAIRS  # 7페어 합산 1일 추정
        rows.append((vn, n, wr, net, perday))
    rows.sort(key=lambda r: int(r[0].split("_")[0][3:]))  # ttl 봉수 순

    lines = ["===== ttl sweep BTC 5년 (cisd+po3, trail 없음) ====="]
    lines.append("  변형        거래   승률    net      1일추정(7페어)")
    for vn, n, wr, net, pd in rows:
        flag = " ←빈도목표" if 2.0 <= pd <= 4.5 else ""
        lines.append(f"  {vn:11s} n={n:4d} {wr:5.1f}% {net:+7.2f}%  ~{pd:.2f}/일{flag}")
    lines.append("\n※ 빈도목표=1일 2~4회. net>0 이면서 빈도목표 드는 ttl = 빈도+흑자 동시.")

    txt = "\n".join(lines)
    with open("ttl_sweep_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 ttl_sweep_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
