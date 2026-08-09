"""#AUTONOMOUS 2026-08-07: 피보나치 되돌림 레벨 1개를 돌려 거래를 JSON 으로 저장.

파트너: "0.5 / 0.618 / 0.707 / 0.786 / 0.886 되돌림 잡는거. 국면마다 다르게 쓰는 것도".

레벨마다 셋업 타임라인을 새로 계산해야 한다(bt_par 캐시 키에 ote_level 이 들어간다)
— 그래서 레벨별로 프로세스를 나눠 병렬 실행하고, 결과를 파일로 모아 분석한다.

현행 라이브: ote_level 0.707, 단 상승 국면(일봉 20일 z>0.75)에서는 0.786
(#REGIME-OTE, 2026-07-10). 이 조건부 규칙이 여전히 최선인지가 이번 연구의 축이다.

저장 필드 — 이후 분석(국면·검증)이 재계산 없이 쓸 수 있게:
    ts(진입ms) · ex(청산ms) · raw · r(R배수) · dir · sym · outcome
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import PAIRS, run_live_parity  # noqa: E402

OUT = Path("data/fib")


def run(level: float, pairs: list[str]) -> dict:
    rows = []
    stats = {}
    for sym in pairs:
        df5, kept, gate = run_live_parity(sym, {"ote_level": level})
        stats[sym] = {k: int(v) for k, v in gate.items()}
        idx = df5.index
        for t in kept:
            sl = float(getattr(t, "entry_sl", 0.0) or 0.0)
            risk = abs(float(t.entry) - sl)
            rows.append({
                "sym": sym,
                "ts": int(idx[t.entry_idx].value // 10**6),
                "ex": int(idx[min(t.exit_idx, len(idx) - 1)].value // 10**6),
                "raw": float(t.raw_pnl_pct),
                "r": (float(t.raw_pnl_pct) * float(t.entry) / risk) if risk > 0 else 0.0,
                "dir": str(getattr(t.direction, "value", t.direction)).lower(),
                "outcome": str(getattr(t, "outcome", "")),
                "entry": float(t.entry),
            })
    rows.sort(key=lambda r: r["ts"])
    return {"level": level, "n": len(rows), "gate": stats, "trades": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=float, required=True)
    ap.add_argument("--pairs", default="BTCUSDT,ETHUSDT",
                    help="쉼표 구분. 기본은 현행 고정 페어(BTC+ETH).")
    a = ap.parse_args()
    pairs = [p.strip() for p in a.pairs.split(",") if p.strip()] or PAIRS
    res = run(a.level, pairs)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"lv{a.level:.3f}.json"
    p.write_text(json.dumps(res), encoding="utf-8")
    rs = [t["r"] for t in res["trades"]]
    avg = sum(rs) / len(rs) if rs else 0.0
    wr = 100.0 * sum(1 for x in rs if x > 0) / len(rs) if rs else 0.0
    print(f"level={a.level:.3f} pairs={','.join(pairs)} n={res['n']} "
          f"건당={avg:+.3f}R 승률={wr:.0f}% → {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
