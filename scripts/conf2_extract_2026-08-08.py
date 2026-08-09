"""#PARITY-CONF 2026-08-08 — 정합 기준선에서 confluence 항목 데이터 **재추출**.

## 왜 다시 뽑나

2026-08-07/08 confluence 캠페인(data/conf/*.json)은 **라이브와 다른 봇**에서 뽑은
데이터였다. 그 백테엔 Phase B 4소스(라이브 진입의 68%)도, HTF supporting boost
(라이브 진입의 93%)도, ote/dol 부호도 없었다. 정합 이식(replay.py 2026-08-08)
이후 기준선이 바뀌었으므로 항목 데이터부터 다시 뽑는다.

## 뽑는 것 (페어별 2회 재생 — timeline 은 캐시라 detect 재계산 없음)

    CUR   min_confluence=5   현행 라이브 그대로. **후보 비교의 기준선**.
    POOL  min_confluence=0   점수 게이트만 푼 모집단. 항목 기여도 분석용.

## 항목 (라이브 매매기록 태그 이름 기준)

    htf_support_weight  boosts "htf_support_weight=W_boost+N"  → htf_boost 0~3
    po3_distribution    boosts "po3"
    turtle_soup         confluences "turtle_soup"      (Phase B 소스)
    implied_fvg         confluences "implied_fvg"      (Phase B 소스)
    bias / sweep / ob   confluences (Phase A 가점)
    macro               macro_high(+2) / macro(+1) / macro_low(+0)
    cisd / ote / smt    boosts
    dol_counter         boosts (역방향 -2 **감점**)

## [한계] — 결론에 안고 갈 것

L1. **POOL 은 CUR 의 상위집합이 아니다.** replay 는 동시 포지션 1개라 문턱을
    낮추면 앞쪽 저점수 거래가 슬롯을 먹어 뒤쪽 고점수 거래를 밀어낸다. POOL 위
    사후 필터는 **스크리닝**이고, 후보 판정은 require_items 로 **재실행**해야 한다
    (conf2_rerun). 이 스크립트가 CUR∩POOL 겹침률을 실측해 같이 남긴다.
L2. live_parity.GAPS 의 미이식(미완성봉 / MMBM / smart_size / margin guard /
    daily_pair_loss_limit / detect 입력 1000봉 / bias 주입)은 그대로다.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from live_parity import live_cfg  # noqa: E402

from aurora_ict.backtest.replay import run_backtest_from_timeline  # noqa: E402

MAIN = ["BTCUSDT", "ETHUSDT"]
HOLDOUT = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
OUT_DIR = "data/conf2"


def rec_of(t, df5, sym: str) -> dict:
    """Trade → 항목 플래그가 붙은 레코드."""
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
        "htf_w": htf_w,      # 지지 HTF FVG 가중치 합 (TF_WEIGHT 5m=1…1w=40)
        "macro_pri": macro_pri,
        "phase": "B" if phase_b else "A",
        "risk_pct": risk,
        "r": (float(t.raw_pnl_pct) / risk) if risk > 0 else float("nan"),
        "raw": float(t.raw_pnl_pct),
        "net": float(t.net_pnl_pct),
        "outcome": t.outcome,
        "flags": f,
    }


def run_pair(sym: str, bars: int = 0) -> dict:
    """한 페어 CUR/POOL 두 재생. Returns: {"cur":[...], "pool":[...], "meta":{}}"""
    df5 = _resample(_load_full(sym))
    if bars > 0:
        df5 = df5.iloc[-bars:]
    t0 = time.time()
    cfg = live_cfg(sym)
    tl = cached_setup_timeline(df5, cfg, sym)
    out = {}
    for name, mc in (("cur", 5), ("pool", 0)):
        c = live_cfg(sym, {"min_confluence": mc})
        bt = run_backtest_from_timeline(df5, tl, c)
        out[name] = [rec_of(t, df5, sym) for t in bt.trades]
    months = (df5.index[-1] - df5.index[0]).days / 30.44
    cur_ts = {r["ts"] for r in out["cur"]}
    pool_ts = {r["ts"] for r in out["pool"]}
    out["meta"] = {
        "sym": sym, "bars": len(df5), "months": months,
        "start": df5.index[0].isoformat(), "end": df5.index[-1].isoformat(),
        "setup_bars": sum(1 for x in tl if x is not None),
        "overlap_cur_in_pool": (len(cur_ts & pool_ts) / len(cur_ts)) if cur_ts else 0.0,
        "sec": time.time() - t0,
    }
    return out


def main() -> int:
    bars, groups = 0, {"main": MAIN, "holdout": HOLDOUT}
    for a in sys.argv[1:]:
        if a.startswith("--bars="):
            bars = int(a.split("=")[1])
        elif a.startswith("--pairs="):
            groups = {"main": a.split("=")[1].split(",")}
        elif a.startswith("--group="):
            groups = {a.split("=")[1]: {"main": MAIN, "holdout": HOLDOUT}[a.split("=")[1]]}
    os.makedirs(OUT_DIR, exist_ok=True)
    for gname, syms in groups.items():
        agg: dict = {"cur": [], "pool": [], "meta": {}}
        for sym in syms:
            r = run_pair(sym, bars)
            agg["cur"] += r["cur"]
            agg["pool"] += r["pool"]
            agg["meta"][sym] = r["meta"]
            m = r["meta"]
            print(f"{sym:<10} CUR {len(r['cur']):>5}건 ({len(r['cur']) / m['months']:.2f}/월) "
                  f"POOL {len(r['pool']):>6}건  겹침 {100 * m['overlap_cur_in_pool']:.0f}%  "
                  f"{m['months']:.1f}개월 {m['sec']:.0f}s", flush=True)
        sfx = f"_{bars}" if bars else ""
        path = os.path.join(OUT_DIR, f"{gname}{sfx}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(agg, f, ensure_ascii=False)
        print(f"저장 → {path}  CUR {len(agg['cur'])} / POOL {len(agg['pool'])}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
