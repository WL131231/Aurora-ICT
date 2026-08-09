"""빈도 타깃 그리드 — 진입완화(근거 2~3개) + 트레일링으로 흑자 유지 가능한지.

파트너 목표(2026-06-17): 구독제(흑자엣지 mc4) 위에서 빈도를 1일 2~4회로.
mc4 는 빈도 부족(1일 ~0.5회) → mc2/mc3 완화로 빈도↑, 단 흑자는 trail 로 보강.
핵심 질문: 근거 완화로 빈도 1일 2~4회 달성 + trail 로 흑자/승률 유지 되는가?

빈도 환산: 1일 2~4회 = 7페어 합산. BTC 단독 ≈ 1/7 → BTC 5년 약 520~1040건 목표.
timeline 의존(min_rr·dol)만 재빌드, mc·sl·trail 재생 재사용. BTC 서치.

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/grid_mc2_5y.py
"""
from __future__ import annotations

import itertools
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _run_pair  # noqa: E402

SYM = "BTCUSDT"
DAYS_5Y = 1825
N_PAIRS = 7  # 7페어 합산 환산용
FREQ_LO = FREQ_HI = None  # 아래서 계산

COMMON = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    entry_ttl_bars=6, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False,
    size_pct=0.9,
)

# timeline 의존 축
MIN_RR = [2.0, 2.5]
DOL = [False, True]
# 재생 재사용 축 — mc2/mc3 = 빈도 레버
MIN_CONF = [2, 3]
SL_DIST = [3.0, 3.5]
EXIT = {
    "base":      {},
    "t0.5_d1.0": dict(trail_trigger=0.5, trail_dist=1.0),
    "t1.0_d0.5": dict(trail_trigger=1.0, trail_dist=0.5),
    "t1.0_d1.0": dict(trail_trigger=1.0, trail_dist=1.0),
    "t1.0_d1.5": dict(trail_trigger=1.0, trail_dist=1.5),
    "t1.5_d0.5": dict(trail_trigger=1.5, trail_dist=0.5),
    "t1.5_d1.0": dict(trail_trigger=1.5, trail_dist=1.0),
    "t1.5_d1.5": dict(trail_trigger=1.5, trail_dist=1.5),
}


def _build_variants():
    v = {}
    for mc, sl in itertools.product(MIN_CONF, SL_DIST):
        for en, ex in EXIT.items():
            v[f"mc{mc}|sl{sl}|{en}"] = {"min_confluence": mc, "sl_dist_mult": sl, **ex}
    return v


def _grid_worker(payload):
    label, base, variants = payload
    _, out = _run_pair((SYM, base, variants))
    return (label, out)


def main() -> int:
    # BTC 1일 환산: 7페어 합산 1일 2~4회 → BTC 단독 1일 2/7~4/7회 → 5년 건수
    lo = round(2 / N_PAIRS * DAYS_5Y)   # ≈ 521
    hi = round(4 / N_PAIRS * DAYS_5Y)   # ≈ 1043
    variants = _build_variants()
    timelines = []
    for rr, dol in itertools.product(MIN_RR, DOL):
        base = {**COMMON, "min_rr": rr, "apply_dol": dol}
        timelines.append(((rr, dol), base))

    payloads = [(lbl, base, variants) for lbl, base in timelines]
    n_tl = len(payloads)
    print(f"timeline {n_tl} × 재생 {len(variants)} = {n_tl * len(variants)} 조합 | "
          f"빈도타깃 BTC 5년 {lo}~{hi}건(=1일2~4회) 서치...", flush=True)

    results = []
    with Pool(6) as p:
        for i, (label, out) in enumerate(p.imap_unordered(_grid_worker, payloads), 1):
            results.append((label, out))
            rr, dol = label
            print(f"[{i}/{n_tl}] rr{rr} dol{int(dol)} 완료", flush=True)

    rows = []
    for (rr, dol), out in results:
        if out is None:
            continue
        for vn, (n, net, w) in out.items():
            mc_s, sl_s, en = vn.split("|")
            wr = w / n * 100 if n else 0.0
            perday = n / DAYS_5Y * N_PAIRS  # 7페어 합산 1일 추정
            rows.append((int(mc_s[2:]), rr, float(sl_s[2:]), dol, en, n, wr, net, perday))

    # 빈도 타깃(1일 2~4회) 충족 + net>0 만, net 내림차순
    lines = [f"===== 빈도타깃(1일 2~4회) 충족 & net>0 | net 내림차순 ====="]
    inband = [r for r in rows if 2.0 <= r[8] <= 4.5 and r[7] > 0]
    inband.sort(key=lambda r: -r[7])
    if not inband:
        lines.append("  (1일 2~4회 구간 & 흑자 조합 없음 — mc 더 완화 or 페어확장 필요)")
    for mc, rr, sl, dol, en, n, wr, net, pd in inband[:40]:
        lines.append(f"  mc{mc} rr{rr} sl{sl} dol{int(dol)} {en:10s} "
                     f"n={n:5d}(~{pd:.1f}/일) win={wr:5.1f}% net={net:+7.2f}%")
    # 승률 60%+ & 빈도충족
    lines.append("\n===== 빈도충족 & 승률 60%+ (net 내림차순) — 3목표 동시 =====")
    triple = [r for r in inband if r[6] >= 60.0]
    if not triple:
        lines.append("  (빈도+승률60%+ 동시 없음 — 트레이드오프 잔존)")
    for mc, rr, sl, dol, en, n, wr, net, pd in triple[:20]:
        lines.append(f"  mc{mc} rr{rr} sl{sl} dol{int(dol)} {en:10s} "
                     f"n={n:5d}(~{pd:.1f}/일) win={wr:5.1f}% net={net:+7.2f}%")
    # 전체 net top + 빈도 범위 참고
    lines.append("\n===== 전체 net 내림차순 TOP20 (빈도 무관) =====")
    rows.sort(key=lambda r: -r[7])
    for mc, rr, sl, dol, en, n, wr, net, pd in rows[:20]:
        lines.append(f"  mc{mc} rr{rr} sl{sl} dol{int(dol)} {en:10s} "
                     f"n={n:5d}(~{pd:.1f}/일) win={wr:5.1f}% net={net:+7.2f}%")

    txt = "\n".join(lines)
    with open("grid_mc2_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 grid_mc2_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
