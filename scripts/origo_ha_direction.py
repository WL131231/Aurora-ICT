"""#AUTONOMOUS 2026-08-02: Origo 방향 판정에 하이켄아시가 정보를 갖는가 (탐색).

7/31 실험은 HA 로 **FVG 를 탐지**했다가 셋업이 354→47건으로 무너졌다. HA 가 정의상
봉 사이 갭을 메우기 때문이고, 애초에 안 맞는 조합이었다(HA 는 추세를 보는 도구).

파트너 지적("하이켄아시를 그래도 제대로 보는 경우에서 봐서 비교해봐야지")에 따라
**추세 판정 자리**에 넣어본다. Origo 에는 이미 그 자리가 있다 — HTF EMA align 게이트
(다중 EMA 정렬 점수로 롱/숏 방향을 강제). FVG 탐지는 실제 캔들 그대로 두므로
셋업이 깎이지 않고, 순수하게 "방향 판정이 나아지나"만 잰다.

⚠️ 이것은 **탐색**이다. 라이브의 align 은 진입 방향을 강제하므로, 사후에 "HA 방향과
   다른 거래 제거"로 재현하는 것은 근사다(방향 강제가 없었다면 다른 셋업이 잡혔을
   수 있다). 정보가 확인되면 그때 replay 에 옵션으로 제대로 구현한다. 여기서 답할
   질문은 하나 — **HA 방향이 성적과 상관이 있는가**.

기준선은 현행 라이브 정합(align 켜진 상태, 126건). 여기에 HA 방향 일치/불일치로
쪼개서 성적 차이를 본다. 차이가 없으면 이 축은 거기서 끝이다.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_parity import PAIRS, run_live_parity, stat  # noqa: E402


def ha_frame(df: pd.DataFrame) -> pd.DataFrame:
    """하이켄아시 변환 — 프로덕션 dual_st.heikin_ashi 와 같은 식."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    n = len(df)
    ha_c = (o + h + lo + c) / 4.0
    ha_o = np.empty(n, dtype=float)
    if n:
        ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, n):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    return pd.DataFrame({"ha_open": ha_o, "ha_close": ha_c}, index=df.index)


def ha_direction(df5: pd.DataFrame, tf: str) -> pd.Series:
    """상위 TF 하이켄아시 방향 — +1(상승) / -1(하락). 5분봉 인덱스로 정렬.

    HA 몸통 색(close > open)이 추세 방향이다. 상위 TF 로 리샘플해 계산한 뒤
    5분봉에 forward-fill 하되 **한 칸 shift** 해서 미완결 봉을 보지 않는다.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    htf = df5.resample(tf).agg(agg).dropna()
    ha = ha_frame(htf)
    d = np.where(ha["ha_close"].to_numpy() > ha["ha_open"].to_numpy(), 1, -1)
    s = pd.Series(d, index=htf.index).shift(1)      # 완결봉만
    return s.reindex(df5.index, method="ffill")


def main() -> int:
    print("=== Origo × 하이켄아시 방향 판정 (탐색) ===", flush=True)
    print("  FVG 탐지는 실제 캔들 그대로 · 방향만 HA 로 본다", flush=True)
    print("  기준선 = 현행 라이브 정합(align 게이트 켜진 상태)\n", flush=True)

    for tf in ("1h", "4h"):
        rows_all: list[tuple[pd.Timestamp, float, str]] = []
        rows_ok: list[tuple[pd.Timestamp, float, str]] = []
        rows_no: list[tuple[pd.Timestamp, float, str]] = []
        for sym in PAIRS:
            df5, kept, _ = run_live_parity(sym)
            hd = ha_direction(df5, tf)
            for t in kept:
                ts = df5.index[t.entry_idx]
                net = float(t.net_pnl_pct)
                rows_all.append((ts, net, sym))
                d = hd.get(ts, np.nan)
                if not np.isfinite(d):
                    continue
                long_ = str(getattr(t.direction, "value", t.direction)).lower() == "long"
                match = (d > 0) == long_
                (rows_ok if match else rows_no).append((ts, net, sym))

        s_all, s_ok, s_no = stat(rows_all), stat(rows_ok), stat(rows_no)
        print(f"[HTF = {tf}]", flush=True)
        for lab, s in (("전체(기준선)", s_all), ("HA 방향 일치", s_ok),
                       ("HA 방향 불일치", s_no)):
            if s is None:
                print(f"  {lab:<16} 표본부족", flush=True)
                continue
            per = s["net"] / max(s["n"], 1)
            print(f"  {lab:<16} n={s['n']:4d} net={s['net']:+9.1f}% "
                  f"건당={per:+6.2f}% 승률={s['wr']:3.0f}% RR={s['rr']:4.2f} "
                  f"MDD={s['mdd']:6.1f}", flush=True)
        if s_ok and s_no:
            print(f"  → 건당 차이 {s_ok['net'] / max(s_ok['n'], 1) - s_no['net'] / max(s_no['n'], 1):+.2f}%p"
                  f"  (일치 {s_ok['n']}건 / 불일치 {s_no['n']}건)", flush=True)
        print("", flush=True)

    print("판정 기준: 일치/불일치 건당 차이가 작으면 HA 방향은 정보가 없다 → 기각.",
          flush=True)
    print("           크면 replay 에 방향 옵션으로 구현해 정식 배터리를 돌린다.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
