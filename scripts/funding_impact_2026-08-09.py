"""#AUTONOMOUS 2026-08-09: 펀딩비 실측 — 백테가 빠뜨린 비용이 얼마인가.

파트너 질문: "실매매랑 백테스트 오차범위 몇%정도 될까".

## 발견
백테는 수수료(0.04%×2)와 슬리피지(0.02~0.05%×2)는 반영하지만 **펀딩비가 없다**
(replay.py · cost.py · live_parity.py 어디에도 funding 문자열이 없다).

무기한 선물은 8시간마다 정산한다. SB 는 단타처럼 보이지만 실제 보유는
**중앙 13.3시간 · 평균 30.4시간**이고 **62% 가 8시간을 넘는다** — 거래당 평균 3.8회
정산을 겪는다. 기댓값이 −0.067R 로 손익분기 경계인 전략에서는 무시할 수 없다.

## 방법
`data/{SYM}_funding.parquet` 의 실제 펀딩률을 거래 보유구간에 맞춰 합산한다.
방향을 지킨다 — **롱은 양수 펀딩을 내고 숏은 받는다**(음수면 반대).
명목 기준 비용을 손절폭(risk_pct)으로 나눠 R 로 환산해야 기댓값과 같은 단위가 된다.

레버리지는 무관하다: 펀딩도 명목, 손익도 명목 기준이라 R 환산에서 상쇄된다.
"""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

SRC = (("본표본 BTC+ETH", "data/conf2/runs_main.json"),
       ("홀드아웃 알트5", "data/conf2/runs_holdout.json"))


def load_funding(sym: str) -> pd.DataFrame | None:
    try:
        d = pd.read_parquet(f"data/{sym}_funding.parquet")
    except Exception:  # noqa: BLE001
        return None
    # ts 컬럼 이름이 구현마다 다를 수 있어 관대하게 찾는다.
    tcol = next((c for c in d.columns if "time" in c.lower() or c.lower() in ("ts", "index")), None)
    rcol = next((c for c in d.columns if "rate" in c.lower() or "funding" in c.lower()), None)
    if rcol is None:
        return None
    if tcol is None:
        if not isinstance(d.index, pd.DatetimeIndex):
            return None
        idx = d.index
    else:
        idx = pd.to_datetime(d[tcol], unit="ms", utc=True) if np.issubdtype(
            d[tcol].dtype, np.number) else pd.to_datetime(d[tcol], utc=True)
    out = pd.DataFrame({"rate": d[rcol].to_numpy(float)}, index=pd.DatetimeIndex(idx))
    return out.sort_index()


def main() -> int:
    print("=== 펀딩비 실측 — 백테 미반영분", flush=True)

    fund: dict[str, pd.DataFrame] = {}
    for name, path in SRC:
        with open(path, encoding="utf-8") as f:
            rows = [t for t in json.load(f)["trades"]["BASE"]
                    if t.get("r") is not None and t["r"] == t["r"]]

        syms = sorted({t["sym"] for t in rows})
        for s in syms:
            if s not in fund:
                fund[s] = load_funding(s)

        costs, applied, miss = [], 0, 0
        for t in rows:
            fd = fund.get(t["sym"])
            if fd is None or fd.empty:
                miss += 1
                continue
            a = dt.datetime.fromisoformat(t["ts"])
            b = dt.datetime.fromisoformat(t["exit_ts"])
            seg = fd.loc[(fd.index > a) & (fd.index <= b), "rate"]
            if seg.empty:
                costs.append(0.0)
                continue
            applied += 1
            # 롱은 양수 펀딩을 지불(비용 +), 숏은 수취(비용 −)
            sign = 1.0 if t["dir"] == "long" else -1.0
            paid = float(seg.sum()) * sign          # 명목 대비 비율
            rp = float(t["risk_pct"]) or np.nan
            costs.append(paid / rp if rp == rp and rp > 0 else 0.0)

        if not costs:
            print(f"\n  [{name}] 펀딩 데이터 없음 (미보유 {miss}건)", flush=True)
            continue

        c = np.array(costs)
        r = np.array([t["r"] for t in rows if fund.get(t["sym"]) is not None
                      and not fund[t["sym"]].empty])
        n = min(len(c), len(r))
        c, r = c[:n], r[:n]
        print(f"\n  [{name}] 거래 {n}건 (펀딩 구간 있는 것 {applied}건"
              f"{f' · 데이터 없는 심볼 {miss}건 제외' if miss else ''})", flush=True)
        print(f"    펀딩 비용  건당 평균 {c.mean():+.4f}R · 중앙 {np.median(c):+.4f}R"
              f" · 최대 {c.max():+.3f}R", flush=True)
        print(f"    기댓값     {r.mean():+.4f}R  →  펀딩 반영 후 {(r - c).mean():+.4f}R"
              f"  (차이 {-c.mean():+.4f}R)", flush=True)
        for d_, lab in (("long", "롱"), ("short", "숏")):
            m = np.array([t["dir"] == d_ for t in rows
                          if fund.get(t["sym"]) is not None
                          and not fund[t["sym"]].empty][:n])
            if m.sum() < 10:
                continue
            print(f"      {lab}  {m.sum():>4}건  펀딩 {c[m].mean():+.4f}R"
                  f"  ({'지불' if c[m].mean() > 0 else '수취'})", flush=True)

    print("\n  결론 기준 — 이 크기가 기댓값 대비 유의미하면 replay 에 이식해야 한다.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
