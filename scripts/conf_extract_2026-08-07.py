"""#AUTONOMOUS 2026-08-07 — confluence **항목별 플래그** 추출 (조합 실험 기반 데이터).

## 왜

파트너 질문: "5등급 진입인데 4등급으로 낮추되 1+2+3+4 가 아니라 1+2+4+5 /
1+3+4+5 식으로 **섞어보자**. 진입 조건이 문제인지, 조건이 4개인 게 문제인지."

지금까지 백테는 confluence 를 **합산 점수 하나**로만 들고 다녔다(Trade.confluence_score).
그래서 "어떤 항목 조합이 좋은가"를 물어볼 수가 없었다. 이 스크립트는 거래 하나하나에
"그때 무슨 항목이 켜져 있었는지"를 붙여 JSON 으로 떨군다. 그러면 이후 조합 실험을
detect 재계산 없이 돌릴 수 있다.

## 백테에서의 confluence 실제 구성 (★파트너 브리핑과 다르다 — 확인 필요)

브리핑은 라이브 봇 기준 5항목(OB·Macro·Sweep·Bias·HTF)이었으나,
**live_parity 백테(replay.py)의 실제 구성은 6항목이고 HTF 가 빠져 있다**:

    ob        미체결 오더블록 근접                     +1   (silver_bullet)
    macro     high +2 / normal +1 / low +0(기록만)      (silver_bullet)
    sweep     같은 방향 유동성 스윕                     +1   (silver_bullet)
    bias      상위TF/일봉 추세와 방향 일치              +1   (silver_bullet)
    cisd      CISD 방향 일치                            +1   (replay _boost_score, LIVE_BASE on)
    po3       AMD distribution 국면                     +1   (replay _boost_score, LIVE_BASE on)
    ── htf    HTF FVG 지지 가중치 +1~+3 → **백테 미구현** (bot_ict_instance 전용)

따라서 백테 최대는 7점(ob1+macro2+sweep1+bias1+cisd1+po31)이고, 라이브 봇은 여기에
htf +0~3 이 더 붙는다. 이 차이는 결론에 반드시 명시할 것.

htf 는 점수에 못 넣는 대신 **참고값(htf_w)** 으로 따로 계산해 붙인다 —
flip_ab_backtest.build_fvg_zones 로 HTF FVG 존을 만들고, 라이브
find_supporting_htf_fvg 의 "같은 방향 + 가격 반대편" 조건을 진입 봉에 적용한다.
근사인 이유는 아래 [한계] 참조.

## 방법

1. min_confluence=0 으로 run_live_parity 를 돌려 셋업을 최대한 확보.
   (min_confluence 는 타임라인 캐시 키에 없어서 재계산 없이 즉시 돈다)
2. 각 거래에 항목 플래그 부착 (replay.py 연구 사본에 Trade.confluences /
   base_score / boosts 필드를 추가해 매칭 없이 **직접** 받아온다).
3. 저장: data/conf/trades_main.json (BTC+ETH), trades_holdout.json (5페어).
4. 검산: flags 로 재계산한 점수 == 원래 confluence_score 인가.

## [한계] — 이후 분석에서 반드시 안고 갈 것

L1. **conf0 거래집합은 conf5 거래집합의 상위집합이 아니다.** replay 는 동시
    포지션 1개라 진입 후 `i = exit_idx + 1` 로 점프한다. 문턱을 낮추면 앞쪽에서
    저품질 거래가 먼저 자리를 차지해 뒤쪽 고품질 거래를 **밀어낸다**. 즉 이
    데이터에 사후 필터를 걸어 "1+2+4+5 조합"을 재현해도 진짜 재실행과 다르다.
    본 실험은 스크리닝용이고, 유망 조합은 _gate_pass 를 고쳐 **재실행**해야 한다.
    (이 스크립트가 conf5 실행도 같이 돌려 겹침률을 실측해 보고한다.)
L2. htf_w 는 근사 — (a) 라이브는 셋업 감지 봉에서 계산하는데 여기선 체결 봉
    기준, (b) touch_count 를 5m 해상도로 근사, (c) 15m 이상 TF 만 사용.
L3. live_parity.GAPS 의 기존 미구현 항목(ote_up 0.786·sweep_gate·smart_size·
    dd_throttle·daily_loss_limit)은 그대로 남아 있다.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flip_ab_backtest import build_fvg_zones  # noqa: E402
from live_parity import run_live_parity  # noqa: E402

from aurora_ict.indicators.fvg import FVGType  # noqa: E402

MAIN = ["BTCUSDT", "ETHUSDT"]
HOLDOUT = ["SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
OUT_DIR = "data/conf"

# 항목별 점수 가중치 — silver_bullet.py:428~462 + replay._boost_score 실측.
# macro 만 우선순위별로 다르다(high=2 / normal=1 / low=0).
W_OB, W_SWEEP, W_BIAS, W_CISD, W_PO3 = 1, 1, 1, 1, 1
W_MACRO = {"high": 2, "normal": 1, "low": 0, "none": 0}


def parse_flags(t) -> dict:
    """Trade.confluences + Trade.boosts → 항목 플래그 dict.

    Args:
        t: replay.Trade (연구 사본 — confluences/base_score/boosts 필드 보유).

    Returns:
        {ob, macro, macro_pri, sweep, bias, cisd, po3} 플래그 dict.
        macro_pri 는 "high"/"normal"/"low"/"none".
    """
    ob = sweep = bias = False
    macro_pri = "none"
    for c in t.confluences:
        if c.startswith("ob="):
            ob = True
        elif c.startswith("macro_high="):
            macro_pri = "high"
        elif c.startswith("macro_low="):
            macro_pri = "low"
        elif c.startswith("macro="):
            macro_pri = "normal"
        elif c == "sweep":
            sweep = True
        elif c.startswith("bias="):
            bias = True
    bs = set(t.boosts)
    return dict(
        ob=ob, macro=(macro_pri != "none"), macro_pri=macro_pri,
        sweep=sweep, bias=bias, cisd=("cisd" in bs), po3=("po3" in bs),
    )


def recompute_score(f: dict) -> int:
    """플래그로 confluence_score 를 되짚어 계산 — 검산용."""
    return (
        W_OB * f["ob"]
        + W_MACRO[f["macro_pri"]]
        + W_SWEEP * f["sweep"]
        + W_BIAS * f["bias"]
        + W_CISD * f["cisd"]
        + W_PO3 * f["po3"]
    )


def htf_support_weight(zones: list[dict], idx: int, direction: int,
                       price: float) -> int:
    """진입 봉에서 **같은 방향** HTF FVG 지지 가중치 합 (근사).

    라이브 find_supporting_htf_fvg 이식 — 롱이면 가격 **아래** bullish FVG,
    숏이면 가격 **위** bearish FVG 만 지지로 인정하고 weight 를 합산한다.
    활성 구간(c5 <= idx < t5)은 flip_ab_backtest.build_fvg_zones 판정을 그대로 쓴다
    (filled / invalidated / touch3 / fetch limit 200봉 수명).

    Args:
        zones: build_fvg_zones 산출.
        idx: 5m 인덱스 (체결 봉).
        direction: 1=롱 / -1=숏.
        price: 그 봉 종가.

    Returns:
        가중치 합 (0 이면 지지 없음).
    """
    want = FVGType.BULLISH if direction == 1 else FVGType.BEARISH
    tot = 0
    for z in zones:
        if z["typ"] is not want:
            continue
        if not (z["c5"] <= idx < z["t5"]):
            continue
        if direction == 1 and z["hi"] >= price:
            continue
        if direction == -1 and z["lo"] <= price:
            continue
        tot += z["w"]
    return tot


def boost_of(total_w: int) -> int:
    """가중치 합 → 계단식 boost. bot_ict_instance._apply_htf_supporting_boost 동일."""
    if total_w >= 20:
        return 3
    if total_w >= 10:
        return 2
    if total_w >= 2:
        return 1
    return 0


def r_of(t) -> float:
    """거래 1건의 R 배수 = raw 손익 / 초기 위험폭. entry_sl 없으면 nan."""
    risk = abs(t.entry - t.entry_sl)
    if risk <= 0 or t.entry <= 0:
        return float("nan")
    return float(t.raw_pnl_pct) / (risk / t.entry)


def extract(sym: str, zones_cache: dict) -> tuple[list[dict], dict]:
    """한 페어 추출. 반환: (거래 레코드 리스트, 진단 dict)."""
    df5, trades, gstats = run_live_parity(sym, {"min_confluence": 0})
    closes = df5["close"].to_numpy()
    zones = zones_cache.get(sym)
    if zones is None:
        zones = build_fvg_zones(df5)
        zones_cache[sym] = zones

    recs, mismatch = [], 0
    for t in trades:
        f = parse_flags(t)
        calc = recompute_score(f)
        if calc != int(t.confluence_score):
            mismatch += 1
        d = 1 if t.direction == "long" else -1
        w = htf_support_weight(zones, t.entry_idx, d, float(closes[t.entry_idx]))
        recs.append({
            "sym": sym,
            "ts": df5.index[t.entry_idx].isoformat(),
            "idx": int(t.entry_idx),
            "ex": t.outcome,
            "raw": float(t.raw_pnl_pct),
            "net": float(t.net_pnl_pct),
            "r": r_of(t),
            "dir": t.direction,
            "score": int(t.confluence_score),
            "score_calc": calc,
            "flags": {
                "ob": f["ob"], "macro": f["macro"], "macro_pri": f["macro_pri"],
                "sweep": f["sweep"], "bias": f["bias"],
                "cisd": f["cisd"], "po3": f["po3"],
                # htf 는 점수에 미반영(백테 미구현) — 참고 컬럼.
                "htf_w": int(w), "htf_boost": boost_of(w),
            },
        })
    return recs, {"gates": gstats, "mismatch": mismatch}


def overlap_check(sym: str, recs: list[dict]) -> tuple[int, int]:
    """conf5(현행) 실행 거래가 conf0 집합에 얼마나 남아 있는지 — L1 정량화.

    Returns:
        (conf5 거래수, 그중 conf0 집합에 같은 entry_idx 로 존재하는 수).
    """
    _, t5, _ = run_live_parity(sym, {"min_confluence": 5})
    have = {r["idx"] for r in recs if r["sym"] == sym}
    hit = sum(1 for t in t5 if t.entry_idx in have)
    return len(t5), hit


def summarize(name: str, recs: list[dict]) -> None:
    """페어별 거래수·항목 출현율·점수 분포 출력."""
    print(f"\n{'=' * 78}\n[{name}] 총 {len(recs)}건", flush=True)
    if not recs:
        return
    by_sym: dict[str, list] = {}
    for r in recs:
        by_sym.setdefault(r["sym"], []).append(r)
    print(f"  {'페어':<10} {'n':>5} {'월빈도':>7} {'ΣR':>8} {'평균R':>7} {'승률':>6}",
          flush=True)
    for s, rs in by_sym.items():
        ts = sorted(pd.Timestamp(r["ts"]) for r in rs)
        months = max((ts[-1] - ts[0]).days / 30.44, 1e-9)
        rr = [r["r"] for r in rs if r["r"] == r["r"]]
        wr = 100 * sum(1 for r in rs if r["raw"] > 0) / len(rs)
        print(f"  {s.replace('USDT', ''):<10} {len(rs):>5} {len(rs) / months:>7.2f} "
              f"{np.sum(rr):>8.1f} {np.mean(rr):>7.3f} {wr:>5.0f}%", flush=True)

    print("\n  [항목별 출현율]", flush=True)
    keys = ["ob", "macro", "sweep", "bias", "cisd", "po3"]
    for k in keys:
        n = sum(1 for r in recs if r["flags"][k])
        print(f"    {k:<10} {n:>5}건  {100 * n / len(recs):>5.1f}%", flush=True)
    pri = Counter(r["flags"]["macro_pri"] for r in recs)
    print("    macro 우선순위: " + " ".join(
        f"{k}={pri.get(k, 0)}" for k in ("high", "normal", "low", "none")), flush=True)
    hb = Counter(r["flags"]["htf_boost"] for r in recs)
    print("    htf_boost(참고·점수미반영): " + " ".join(
        f"+{k}={hb.get(k, 0)}" for k in (0, 1, 2, 3)), flush=True)

    print("\n  [점수 분포]", flush=True)
    sc = Counter(r["score"] for r in recs)
    for k in sorted(sc):
        rs = [r for r in recs if r["score"] == k]
        rr = [r["r"] for r in rs if r["r"] == r["r"]]
        note = "  ※표본부족(<30)" if len(rs) < 30 else ""
        print(f"    score={k}: {sc[k]:>5}건 ({100 * sc[k] / len(recs):>4.1f}%) "
              f"ΣR={np.sum(rr):>8.1f} 평균R={np.mean(rr):>+6.3f}{note}", flush=True)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    zones_cache: dict = {}
    out: dict[str, list[dict]] = {}
    diag: dict[str, dict] = {}
    for name, pairs in (("main", MAIN), ("holdout", HOLDOUT)):
        recs: list[dict] = []
        for sym in pairs:
            print(f"[추출] {sym} ...", flush=True)
            r, d = extract(sym, zones_cache)
            recs += r
            diag[sym] = d
            print(f"   → {len(r)}건 (게이트 {d['gates']}) 검산불일치 {d['mismatch']}",
                  flush=True)
        recs.sort(key=lambda r: (r["ts"], r["sym"]))
        out[name] = recs
        path = os.path.join(OUT_DIR, f"trades_{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(recs, f, ensure_ascii=False)
        print(f"   저장 → {path}", flush=True)

    for name in ("main", "holdout"):
        summarize(name, out[name])

    print(f"\n{'=' * 78}\n[검산] flags 재계산 점수 vs 원본 confluence_score", flush=True)
    tot = bad = 0
    for name in ("main", "holdout"):
        for r in out[name]:
            tot += 1
            if r["score"] != r["score_calc"]:
                bad += 1
    print(f"  전체 {tot}건 중 불일치 {bad}건 ({100 * bad / max(tot, 1):.3f}%)", flush=True)
    if bad:
        ex = [r for n in ("main", "holdout") for r in out[n]
              if r["score"] != r["score_calc"]][:5]
        for r in ex:
            print(f"    예시 {r['sym']} {r['ts']} 원본={r['score']} "
                  f"재계산={r['score_calc']} flags={r['flags']}", flush=True)

    print(f"\n{'=' * 78}\n[L1 겹침 검사] conf5 거래가 conf0 집합에 남아 있는 비율",
          flush=True)
    print("  (동시포지션 1개 제약 때문에 문턱을 낮추면 거래가 서로를 밀어낸다)",
          flush=True)
    all_recs = out["main"] + out["holdout"]
    for sym in MAIN + HOLDOUT:
        n5, hit = overlap_check(sym, all_recs)
        pct = 100 * hit / n5 if n5 else 0.0
        print(f"    {sym.replace('USDT', ''):<8} conf5 {n5:>4}건 중 conf0 집합에 "
              f"{hit:>4}건 존재 ({pct:>5.1f}%)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
