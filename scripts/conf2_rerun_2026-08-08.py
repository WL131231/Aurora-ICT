"""#PARITY-CONF 2026-08-08 — confluence 후보 **정합 기준선 재실행** (사후 필터 아님).

## 왜 재실행인가

2026-08-07/08 캠페인은 min_confluence=0 거래 JSON 에 **사후 필터**를 걸어 후보를
골랐다. replay 는 동시 포지션 1개라 게이트를 바꾸면 슬롯 점유가 달라지고, 앞쪽
거래가 뒤쪽 거래를 밀어낸다 — 사후 필터 성적은 진짜 재실행과 다르다(그 캠페인도
L1 로 명시했다). 여기서는 모든 후보를 replay 재생 루프 안 게이트
(cfg.require_items / min_confluence / htf_fvg_support)로 **다시 돌린다**.

## 사전 등록 변형 (결과 보기 전에 고정 — 다중비교 분모)

    BASE      현행 라이브 (min_conf=5, HTF boost on)          ← 비교 기준선
    POOL      min_conf=0 (게이트만 개방)                       ← 모집단(기여도 분석용)
    [후보 — 2026-08-08 캠페인 산출물 재판정]
    MH_BIAS   macro_high AND bias (문턱 없음)                  ← 배포 후보 원안
    MH_BIAS5  macro_high AND bias + 문턱5                      ← 후보 보수판
    MH        macro_high 단독                                  ← 후보 보조
    [HTF 축 — 라이브 진입의 93% 에 붙는 항목이라 최우선]
    NOHTF5/4/3/2   htf_fvg_support=False + 문턱 5/4/3/2       ← 빈도 정합 비교용
    HTF3      문턱5 + htf 가중 +3 요구
    HTFLT3    문턱5 + htf +3 아닌 것만 (0~2)
    [Phase 축 — 정합 이식으로 새로 생긴 축]
    PHASEB    문턱5 + Phase B(turtle/implied/mitigation/rejection) 만
    PHASEA    문턱5 + Phase A(FVG/IFVG) 만
    NOTURTLE  문턱5 + turtle_soup 제외
    NODOL     문턱5 + DOL 역방향 -2 감점 해제 (사실상 컷 게이트라 항목 기여도)

## 산출물

    data/conf2/runs_main.json      BTC+ETH  (본 표본)
    data/conf2/runs_holdout.json   SOL·XRP·DOGE·LINK·HYPE (파라미터 재선택 금지)
"""

from __future__ import annotations

import json
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAIN = ["BTCUSDT", "ETHUSDT"]
HOLDOUT = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
OUT_DIR = "data/conf2"

# 변형명 → live_cfg extra. **결과 보기 전 고정** (Bonferroni 분모).
VARIANTS: dict[str, dict] = {
    "BASE": {},
    "POOL": {"min_confluence": 0},
    "MH_BIAS": {"min_confluence": 0, "require_items": ("macro_high", "bias")},
    "MH_BIAS5": {"require_items": ("macro_high", "bias")},
    "MH": {"min_confluence": 0, "require_items": ("macro_high",)},
    "NOHTF5": {"htf_fvg_support": False},
    "NOHTF4": {"htf_fvg_support": False, "min_confluence": 4},
    "NOHTF3": {"htf_fvg_support": False, "min_confluence": 3},
    "NOHTF2": {"htf_fvg_support": False, "min_confluence": 2},
    "HTF3": {"require_items": ("htf>=3",)},
    "HTFLT3": {"require_items": ("!htf>=3",)},
    "PHASEB": {"require_items": ("phase_b",)},
    "PHASEA": {"require_items": ("phase_a",)},
    "NOTURTLE": {"require_items": ("!turtle_soup",)},
    # DOL 역방향 -2 는 점수를 문턱 아래로 떨어뜨려 **사실상 컷 게이트**로 작동한다.
    # 항목 기여도의 일부이므로 같이 재실행해 둔다(정합 이식으로 부호가 바뀐 항목).
    "NODOL": {"apply_dol_counter": False},
}
# 기준선/모집단은 검정 대상이 아니다 (다중비교 분모에서 제외).
NOT_TESTED = ("BASE", "POOL")


def rec_of(t, df5, sym: str) -> dict:
    """Trade → 항목 플래그 레코드 (conf2_extract 와 동일 스키마)."""
    conf, bs = list(t.confluences), list(t.boosts)
    htf_b, htf_w, macro_pri = 0, 0.0, "none"
    f = {k: False for k in ("ob", "sweep", "bias", "turtle_soup", "implied_fvg",
                            "mitigation_block", "rejection_block", "cisd", "po3",
                            "ote", "smt", "dol_counter")}
    for c in conf:
        if c.startswith("ob="):
            f["ob"] = True
        elif c.startswith("macro_high="):
            macro_pri = "high"
        elif c.startswith("macro_low="):
            macro_pri = "low"
        elif c.startswith("macro="):
            macro_pri = "normal"
        elif c.startswith("bias="):
            f["bias"] = True
        elif c in f:
            f[c] = True
    for b in bs:
        if b.startswith("htf_support_weight="):
            htf_b = int(b.rsplit("+", 1)[1])
            htf_w = float(b.split("=", 1)[1].split("_boost")[0])
        elif b.startswith("dol_counter"):
            f["dol_counter"] = True
        elif b in f:
            f[b] = True
    risk = abs(t.entry - t.entry_sl) / t.entry if (t.entry_sl > 0 and t.entry > 0) else 0.0
    phase_b = any(f[k] for k in ("turtle_soup", "implied_fvg",
                                 "mitigation_block", "rejection_block"))
    return {
        "sym": sym,
        "ts": df5.index[t.entry_idx].isoformat(),
        "exit_ts": df5.index[t.exit_idx].isoformat(),
        "dir": t.direction,
        "score": int(t.confluence_score),
        "base": int(t.base_score),
        "htf_boost": htf_b,
        "htf_w": htf_w,
        "macro_pri": macro_pri,
        "phase": "B" if phase_b else "A",
        "risk_pct": risk,
        "r": (float(t.raw_pnl_pct) / risk) if risk > 0 else float("nan"),
        "raw": float(t.raw_pnl_pct),
        "net": float(t.net_pnl_pct),
        "outcome": t.outcome,
        "flags": f,
    }


def _run_pair(payload) -> tuple[str, dict]:
    """한 페어: timeline 1회 로드 → 전 변형 재생. (multiprocessing top-level)"""
    sym, bars, names = payload
    from bt_par import _load_full, _resample, cached_setup_timeline
    from live_parity import live_cfg

    from aurora_ict.backtest.replay import run_backtest_from_timeline
    df5 = _resample(_load_full(sym))
    if bars > 0:
        df5 = df5.iloc[-bars:]
    tl = cached_setup_timeline(df5, live_cfg(sym), sym)
    out: dict = {"trades": {}, "meta": {}}
    for vn in names:
        t0 = time.time()
        bt = run_backtest_from_timeline(df5, tl, live_cfg(sym, VARIANTS[vn]))
        out["trades"][vn] = [rec_of(t, df5, sym) for t in bt.trades]
        out["meta"][vn] = {"sec": time.time() - t0, "n": len(bt.trades)}
    out["meta"]["_pair"] = {
        "bars": len(df5), "months": (df5.index[-1] - df5.index[0]).days / 30.44,
        "start": df5.index[0].isoformat(), "end": df5.index[-1].isoformat(),
        "setup_bars": sum(1 for x in tl if x is not None),
    }
    return sym, out


def main() -> int:
    bars, groups, nproc = 0, {"main": MAIN, "holdout": HOLDOUT}, 4
    names = list(VARIANTS)
    out_name = None   # --out= 로 그룹명(=파일명) 강제. 홀드아웃을 타임라인 준비
                      # 순서에 맞춰 배치로 쪼개 돌린 뒤 합치려고 추가(2026-08-08).
    for a in sys.argv[1:]:
        if a.startswith("--bars="):
            bars = int(a.split("=")[1])
        elif a.startswith("--out="):
            out_name = a.split("=")[1]
        elif a.startswith("--pairs="):
            groups = {"dev": a.split("=")[1].split(",")}
        elif a.startswith("--group="):
            g = a.split("=")[1]
            groups = {g: {"main": MAIN, "holdout": HOLDOUT}[g]}
        elif a.startswith("--nproc="):
            nproc = int(a.split("=")[1])
    os.makedirs(OUT_DIR, exist_ok=True)
    if out_name:
        groups = {out_name: next(iter(groups.values()))}
    for gname, syms in groups.items():
        agg: dict = {"variants": {k: list(v.items()) for k, v in VARIANTS.items()},
                     "not_tested": list(NOT_TESTED), "trades": {}, "meta": {}}
        payloads = [(s, bars, names) for s in syms]
        with Pool(min(nproc, len(syms)), maxtasksperchild=1) as p:
            for sym, res in p.imap_unordered(_run_pair, payloads):
                for vn in names:
                    agg["trades"].setdefault(vn, []).extend(res["trades"][vn])
                agg["meta"][sym] = res["meta"]
                m = res["meta"]["_pair"]
                print(f"[{sym}] {m['months']:.1f}개월 · " + " ".join(
                    f"{vn}={res['meta'][vn]['n']}" for vn in names), flush=True)
        sfx = f"_{bars}" if bars else ""
        path = os.path.join(OUT_DIR, f"runs_{gname}{sfx}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False)
        print(f"저장 → {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
