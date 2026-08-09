"""conf2_rerun 배치 산출물 합치기 — 홀드아웃을 나눠 돌린 뒤 하나로.

왜: 홀드아웃 5페어의 정합(v2) 타임라인이 동시에 준비되지 않는다(페어당 빌드
2~3시간, 순차 완료). 준비된 페어부터 돌리고 마지막에 합쳐야 대기가 줄어든다.
합치는 건 페어별 독립 백테라 **거래 목록 concat + meta 병합**이면 충분하다
(페어 간 상호작용은 백테에 없다 — 복리 시뮬만 합산 시점에 계산된다).

사용: python scripts/conf2_merge_runs_2026-08-08.py holdout_a holdout_b -> holdout
"""

from __future__ import annotations

import json
import os
import sys

OUT_DIR = "data/conf2"


def main() -> int:
    args = sys.argv[1:]
    if "->" in args:
        k = args.index("->")
        srcs, dst = args[:k], args[k + 1]
    else:
        srcs, dst = args[:-1], args[-1]
    agg: dict = {}
    for s in srcs:
        p = os.path.join(OUT_DIR, f"runs_{s}.json")
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if not agg:
            agg = {"variants": d["variants"], "not_tested": d["not_tested"],
                   "trades": {}, "meta": {}}
        if d["variants"] != agg["variants"]:
            raise SystemExit(f"변형 목록 불일치 — {p} 는 다른 사전등록으로 돌았다")
        for vn, tr in d["trades"].items():
            agg["trades"].setdefault(vn, []).extend(tr)
        agg["meta"].update(d["meta"])
        print(f"  + {p}: {len(d['meta'])}페어 · "
              f"BASE {len(d['trades']['BASE'])}건", flush=True)
    path = os.path.join(OUT_DIR, f"runs_{dst}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False)
    print(f"저장 → {path} ({len(agg['meta'])}페어 · "
          f"BASE {len(agg['trades']['BASE'])}건)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
