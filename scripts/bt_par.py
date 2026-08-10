"""페어 병렬 5년 백테스트 헬퍼 — 페어별 독립 timeline 빌드를 멀티프로세스로 분산.

6물리코어 풀가동: 프로세스당 BLAS 1스레드로 묶고(오버서브스크립션 방지) 페어를 동시에.
모든 5년 스크립트가 run_parallel() 하나로 통일 (파트너 2026-06-16: 항상 병렬로).

사용:
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bt_par import run_parallel
    if __name__ == "__main__":
        totals, per_sym = run_parallel(PAIRS, BASE, VARIANTS, nproc=6)
"""
from __future__ import annotations

import os

# BLAS 스레드 × 프로세스 = 논리코어(12) 균형. 1로 너무 조이면 detect 가 느려져
# 역효과(7페어 64분 경험). 2스레드 × 6프로세스 = 12 로 detect 속도 + 병렬 둘 다.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import hashlib  # noqa: E402
import pickle  # noqa: E402

import pandas as pd  # noqa: E402
from multiprocessing import Pool  # noqa: E402

from aurora_ict.backtest.replay import (  # noqa: E402
    BacktestConfig,
    build_setup_timeline,
    run_backtest_from_timeline,
)

# 2026-06-22 #TL-CACHE: timeline 빌드(매 봉 9종 ICT detect)가 검증 병목 — 페어당
# 수 분, 캐시 없어 ote/페어/검증마다 재빌드(며칠 누적). detect 인자+df범위 키로
# pickle 캐시 → 같은 설정 재빌드 즉시화. 반복 검증·분할진입 검증 가속.
_TL_CACHE_DIR = "data/tl_cache"


def cached_setup_timeline(df, cfg, sym: str):
    """build_setup_timeline 캐시 — detect 인자+df범위 키로 pickle 재사용.

    키 = (sym, ote_level, min_rr, fvg_min_size_pct, expand_to_killzone,
    disable_time_filter, min_sl_distance_pct, window, df 길이·시작·끝). 이 중
    하나라도 다르면 다른 timeline → 재빌드. 같으면 캐시 hit(즉시).

    2026-08-08 정합 이식으로 키가 3개 늘었다(phase_b_sources / ote_up_level /
    prefer_direction_select). 셋업 후보 집합·진입가·타임라인 형태(dir_items)를
    바꾸므로 반드시 별도 캐시여야 한다. 키 접두어 v2 로 구 캐시와 분리.
    """
    parts = (
        "v2",
        sym, round(cfg.ote_level, 4), round(cfg.min_rr, 3),
        round(cfg.fvg_min_size_pct, 6), bool(cfg.expand_to_killzone),
        bool(cfg.disable_time_filter), round(cfg.min_sl_distance_pct, 6),
        int(cfg.window), len(df), int(df.index[0].value), int(df.index[-1].value),
        bool(cfg.phase_b_sources), round(cfg.ote_up_level, 4),
        bool(cfg.prefer_direction_select),
    )
    # [08-09] #KZ-WIDE — 미장 게이트를 끄면 셋업 집합이 달라지므로 키를 나눈다.
    # 단 **기본값(True)일 때는 키를 건드리지 않는다** — 그래야 기존 캐시가
    # 그대로 유효하다. 처음엔 무조건 키에 넣었다가 5년 타임라인 전체가
    # 재빌드(페어당 2시간)되는 걸 보고 조건부로 바꿨다.
    if not getattr(cfg, "nyse_gate", True):
        parts = parts + ("kzwide",)
    # [08-10] #FVG-REUSE — 창/FVG 재사용 제한이 바뀌면 셋업 집합이 달라진다.
    _wo = bool(getattr(cfg, "window_once", True))
    _mf = int(getattr(cfg, "max_per_fvg", 0))
    if not _wo or _mf:
        parts = parts + (f"wo{int(_wo)}_mf{_mf}",)
    key = hashlib.md5(repr(parts).encode()).hexdigest()[:16]
    os.makedirs(_TL_CACHE_DIR, exist_ok=True)
    path = os.path.join(_TL_CACHE_DIR, f"{sym}_{key}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    tl = build_setup_timeline(df, cfg)
    with open(path, "wb") as f:
        pickle.dump(tl, f)
    return tl


def _resample(d: pd.DataFrame, rule: str = "5min") -> pd.DataFrame:
    o = d["open"].resample(rule).first()
    h = d["high"].resample(rule).max()
    lo = d["low"].resample(rule).min()
    c = d["close"].resample(rule).last()
    v = d["volume"].resample(rule).sum()
    return pd.DataFrame(
        {"open": o, "high": h, "low": lo, "close": c, "volume": v},
    ).dropna()


def _load_full(sym: str) -> pd.DataFrame:
    df = pd.read_parquet(f"data/{sym}_1m_full.parquet")
    df.index = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], unit="ms", utc=True))
    return df[["open", "high", "low", "close", "volume"]]


def _run_pair(payload):
    """한 페어: timeline 1회 빌드 → 모든 변형 재생. (multiprocessing top-level)"""
    sym, base, variants = payload
    df5 = _resample(_load_full(sym))
    if len(df5) < 700:
        return (sym, None)
    tl = build_setup_timeline(df5, BacktestConfig(**base))
    out = {}
    for vn, vc in variants.items():
        bt = run_backtest_from_timeline(df5, tl, BacktestConfig(**{**base, **vc}))
        out[vn] = (bt.n_trades, bt.total_net_pnl_pct, bt.n_wins)
    return (sym, out)


def _run_pair_detect(payload):
    """한 페어, detect 변형용 — 변형마다 timeline 빌드(min_rr/ote_level 등 detect 변경).

    variants = {vname: 완전한 cfg dict}. (multiprocessing top-level)
    """
    sym, variants = payload
    df5 = _resample(_load_full(sym))
    if len(df5) < 700:
        return (sym, None)
    out = {}
    for vn, cfgd in variants.items():
        cfg = BacktestConfig(**cfgd)
        tl = build_setup_timeline(df5, cfg)
        bt = run_backtest_from_timeline(df5, tl, cfg)
        out[vn] = (bt.n_trades, bt.total_net_pnl_pct, bt.n_wins)
    return (sym, out)


def run_parallel_detect(pairs, variants, nproc: int = 6):
    """detect 변형 페어 병렬 (변형마다 timeline). variants = {vname: 완전 cfg dict}."""
    payloads = [(s, variants) for s in pairs]
    with Pool(min(nproc, len(pairs))) as p:
        results = p.map(_run_pair_detect, payloads)
    totals = {vn: [0, 0.0, 0] for vn in variants}
    per_sym = {vn: {} for vn in variants}
    for sym, out in results:
        if out is None:
            continue
        for vn, (n, net, w) in out.items():
            totals[vn][0] += n
            totals[vn][1] += net
            totals[vn][2] += w
            per_sym[vn][sym] = net
    return totals, per_sym


def run_parallel(pairs, base, variants, nproc: int = 6):
    """페어 병렬 실행. 반환: (totals, per_sym).

    totals = {vname: [n, net합, wins합]}, per_sym = {vname: {sym: net}}.
    """
    payloads = [(s, base, variants) for s in pairs]
    with Pool(min(nproc, len(pairs))) as p:
        results = p.map(_run_pair, payloads)
    totals = {vn: [0, 0.0, 0] for vn in variants}
    per_sym = {vn: {} for vn in variants}
    for sym, out in results:
        if out is None:
            continue
        for vn, (n, net, w) in out.items():
            totals[vn][0] += n
            totals[vn][1] += net
            totals[vn][2] += w
            per_sym[vn][sym] = net
    return totals, per_sym


def fmt(title, totals, per_sym, pairs, base_key="base"):
    """결과 텍스트 포맷 (base 대비 + 페어별)."""
    base_net = totals.get(base_key, [0, 0.0, 0])[1]
    rows = []
    for vn, t in totals.items():
        n, net, w = t
        wr = w / n * 100 if n else 0.0
        rows.append((vn, n, wr, net, net - base_net))
    rows.sort(key=lambda r: -r[3])
    lines = [f"===== {title} | base={base_net:+.2f}% ====="]
    for vn, n, wr, net, d in rows:
        mark = " <<" if net > base_net and vn != base_key else ""
        lines.append(
            f"  {vn:20s} n={n:5d} w={wr:4.1f}% net={net:+7.2f}% (vs base {d:+5.2f}){mark}"
        )
    lines.append("----- 페어별 net% -----")
    for sym in pairs:
        row = " ".join(f"{vn[:9]}={per_sym[vn].get(sym, 0):+5.1f}" for vn in totals)
        lines.append(f"  {sym:9s} {row}")
    return "\n".join(lines)
