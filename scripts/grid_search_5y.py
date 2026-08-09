"""대규모 그리드 서치 BTC 단독 — timeline 캐시 최적화판 (파트너: 수십~수백 조합).

핵심 최적화(2026-06-17): timeline 은 검출 캐시. detect 의존 축(min_rr·dol)만
timeline 재빌드, 나머지(min_confluence·sl_dist_mult·trail)는 재생 재사용.
→ 6 timeline(min_rr 3 × dol 2) × 72 재생(min_conf 3 × sl_dist 3 × exit 8) = 432 조합.
   timeline 빌드 54→6 (약 9배 가속). 6 timeline 을 6코어 1배치 병렬.

파트너 지시: 서치는 BTC → 상위 후보만 7페어 robust. size 0.9 운영정합.
목표 3개: 승률 60%+ / 매매빈도↑(min_conf 완화) / net 유지(trail 보존).

사용: PYTHONIOENCODING=utf-8 PYTHONPATH=src python scripts/grid_search_5y.py
"""
from __future__ import annotations

import itertools
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _run_pair  # noqa: E402  (top-level, multiprocessing picklable)

SYM = "BTCUSDT"

COMMON = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    entry_ttl_bars=6, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False,
    size_pct=0.9,  # 운영 정합 (score 4+ → 90%)
)

# timeline 의존 축 (재빌드 필요)
MIN_RR = [2.0, 2.5, 3.0]
DOL = [False, True]
# 재생 재사용 축 (timeline 1개로 전부 커버)
MIN_CONF = [3, 4, 5]
SL_DIST = [2.5, 3.0, 3.5]
EXIT = {
    "base":      {},
    "t0.5_d1.0": dict(trail_trigger=0.5, trail_dist=1.0),
    "t1.0_d0.5": dict(trail_trigger=1.0, trail_dist=0.5),
    "t1.0_d1.0": dict(trail_trigger=1.0, trail_dist=1.0),
    "t1.0_d1.5": dict(trail_trigger=1.0, trail_dist=1.5),
    "t1.0_d2.0": dict(trail_trigger=1.0, trail_dist=2.0),
    "t1.5_d1.0": dict(trail_trigger=1.5, trail_dist=1.0),
    "t1.5_d1.5": dict(trail_trigger=1.5, trail_dist=1.5),
}


def _build_variants():
    """timeline 1개 위에서 재생할 72 변형 (min_conf × sl_dist × exit)."""
    v = {}
    for mc, sl in itertools.product(MIN_CONF, SL_DIST):
        for en, ex in EXIT.items():
            v[f"mc{mc}|sl{sl}|{en}"] = {"min_confluence": mc, "sl_dist_mult": sl, **ex}
    return v


def _grid_worker(payload):
    """timeline 1조합(min_rr,dol): 1회 빌드 + 72 재생. (multiprocessing top-level)"""
    label, base, variants = payload
    _, out = _run_pair((SYM, base, variants))
    return (label, out)


def main() -> int:
    variants = _build_variants()
    timelines = []
    for rr, dol in itertools.product(MIN_RR, DOL):
        base = {**COMMON, "min_rr": rr, "apply_dol": dol}
        timelines.append(((rr, dol), base))

    payloads = [(lbl, base, variants) for lbl, base in timelines]
    n_tl = len(payloads)
    print(f"timeline {n_tl}개 × 재생 {len(variants)} = {n_tl * len(variants)} 조합 "
          f"BTC 서치 시작...", flush=True)

    results = []
    with Pool(6) as p:
        for i, (label, out) in enumerate(p.imap_unordered(_grid_worker, payloads), 1):
            results.append((label, out))
            rr, dol = label
            print(f"[{i}/{n_tl}] rr{rr} dol{int(dol)} timeline 완료 (재생 {len(variants)})",
                  flush=True)

    rows = []
    for (rr, dol), out in results:
        if out is None:
            continue
        for vn, (n, net, w) in out.items():
            mc_s, sl_s, en = vn.split("|")
            wr = w / n * 100 if n else 0.0
            rows.append((int(mc_s[2:]), rr, float(sl_s[2:]), dol, en, n, wr, net))

    rows.sort(key=lambda r: -r[7])
    lines = [f"===== 그리드 BTC 5년 ({len(rows)}조합) | net 내림차순 TOP30 ====="]
    for mc, rr, sl, dol, en, n, wr, net in rows[:30]:
        lines.append(f"  mc{mc} rr{rr} sl{sl} dol{int(dol)} {en:10s} "
                     f"n={n:5d} win={wr:5.1f}% net={net:+7.2f}%")
    lines.append("\n===== 승률 60%+ 만 (net 내림차순) — 파트너 타깃 =====")
    hits = [r for r in rows if r[6] >= 60.0]
    if not hits:
        lines.append("  (60%+ 없음 — trail dist 더 타이트(d0.5) 필요할 수)")
    for mc, rr, sl, dol, en, n, wr, net in hits[:40]:
        lines.append(f"  mc{mc} rr{rr} sl{sl} dol{int(dol)} {en:10s} "
                     f"n={n:5d} win={wr:5.1f}% net={net:+7.2f}%")
    lines.append("\n===== 빈도 상위 15 (n 내림차순, net>0만) =====")
    freq = sorted([r for r in rows if r[7] > 0], key=lambda r: -r[5])[:15]
    for mc, rr, sl, dol, en, n, wr, net in freq:
        lines.append(f"  mc{mc} rr{rr} sl{sl} dol{int(dol)} {en:10s} "
                     f"n={n:5d} win={wr:5.1f}% net={net:+7.2f}%")

    txt = "\n".join(lines)
    with open("grid_search_5y_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    try:
        print("\n" + txt + "\nDONE", flush=True)
    except UnicodeEncodeError:
        print("(결과는 grid_search_5y_result.txt)\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
