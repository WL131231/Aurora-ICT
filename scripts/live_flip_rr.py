"""#AUTONOMOUS 2026-07-30: flip 청산의 **R 단위** 반사실 — 홀드했으면 TP 먼저? SL 먼저?

1차 반사실(live_flip_counterfactual.py): flip 후 4h 유리 +0.70% / 불리 +0.49%,
유리>불리 79% → 조기 절단 시사. 그러나 ①"유리 최대" 는 완벽 타이밍 가정 ②불리
0.49%(20x=9.8%)도 커서 홀드 시 SL 위험 ③n=29 소표본. 확정 불가.

정밀화: **R 단위 + 선착순**.
  1) 진입가 확보 — setup_ts_ms 로 ENTRY 이벤트 매칭.
  2) SL 폭(R) 추정 — sl_hit 건의 |청산가-진입가|/진입가 분포에서 페어별 중앙값.
     (CSV 에 SL 가격 컬럼이 없어 실측 손절 거리로 역산. 라이브 sl_dist_mult=4.0×ATR)
  3) flip 이 **몇 R 에서** 나갔는지 계산 = (청산가-진입가)/진입가/SL% .
  4) flip 시점 이후 경로에서 **2R TP 와 SL 중 어느 쪽이 먼저** 닿는지 실제 가격으로 판정.
     → TP 선착 비율이 높으면 조기 절단 확정(타겟 늦출 근거).
     → SL 선착이 많으면 flip 이 실제로 이익을 지키고 있는 것(유지).
  5) 기대값 비교: 현행(flip 즉시 청산) vs 홀드(2R TP / SL) — R 단위 EV.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from live_flip_counterfactual import CSV, load_recent, to_bt_symbol  # noqa: E402

HOLD_BARS = 288        # 5m×288 = 24h 추적


def main() -> int:
    d = pd.read_csv(CSV)
    d["ts"] = pd.to_datetime(d.ts_ms, unit="ms", utc=True)
    o = d[d.model == "Origo 2.2"].copy()
    o["bt_sym"] = o.symbol.map(to_bt_symbol)

    # ENTRY 매칭 — setup_ts_ms 기준
    ent = o[o.event_type == "entry"].set_index("setup_ts_ms")
    print(f"ENTRY {len(ent)}건 / 청산 매칭 시도", flush=True)

    # SL 폭 역산 — sl_hit 의 |청산가-진입가|/진입가
    sl_rows = []
    for r in o[o.event_type == "sl_hit"].itertuples():
        if r.setup_ts_ms not in ent.index:
            continue
        e = ent.loc[r.setup_ts_ms]
        ep = float(e.price if np.isscalar(e.price) else e.price.iloc[0])
        if ep <= 0:
            continue
        sl_rows.append(dict(sym=r.bt_sym, slpct=abs(float(r.price) - ep) / ep * 100))
    S = pd.DataFrame(sl_rows)
    if S.empty:
        print("SL 역산 실패 — ENTRY 매칭 0", flush=True)
        return 1
    sl_med = S.groupby("sym").slpct.median().to_dict()
    glob_med = float(S.slpct.median())
    print(f"\n=== SL 폭 역산 (sl_hit {len(S)}건) ===", flush=True)
    print(f"  전체 중앙값 {glob_med:.3f}%  |  " +
          " ".join(f"{k.replace('USDT', '')}:{v:.2f}%" for k, v in sl_med.items()), flush=True)

    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for r in o[o.event_type == "flip_close"].itertuples():
        if r.setup_ts_ms not in ent.index:
            continue
        e = ent.loc[r.setup_ts_ms]
        ep = float(e.price if np.isscalar(e.price) else e.price.iloc[0])
        if ep <= 0:
            continue
        sym = r.bt_sym
        slp = sl_med.get(sym, glob_med) / 100.0     # SL 폭(비율)
        if slp <= 0:
            continue
        sgn = 1.0 if str(r.direction).lower() == "long" else -1.0
        fp = float(r.price)
        r_at_flip = ((fp - ep) / ep * sgn) / slp    # flip 시점 이익(R)
        # 홀드 시뮬 — flip 시각부터 2R TP vs SL 선착
        if sym not in cache:
            try:
                cache[sym] = load_recent(sym)
            except Exception:  # noqa: BLE001
                cache[sym] = pd.DataFrame()
        df = cache[sym]
        if df.empty:
            continue
        i0 = df.index.searchsorted(r.ts)
        seg = df.iloc[i0:i0 + HOLD_BARS]
        if seg.empty:
            continue
        tp_px = ep * (1 + sgn * 2.0 * slp)          # 2R
        sl_px = ep * (1 - sgn * 1.0 * slp)
        first = "미도달"
        for row in seg.itertuples():
            if sgn > 0:
                if row.low <= sl_px:
                    first = "SL"; break
                if row.high >= tp_px:
                    first = "TP"; break
            else:
                if row.high >= sl_px:
                    first = "SL"; break
                if row.low <= tp_px:
                    first = "TP"; break
        rows.append(dict(sym=sym, pnl=r.pnl_usdt, r_flip=r_at_flip, first=first,
                         ts=r.ts))
    F = pd.DataFrame(rows)
    print(f"\n=== flip {len(F)}건 R 분석 ===", flush=True)
    if F.empty:
        return 1
    print(f"  flip 시점 이익: 평균 {F.r_flip.mean():+.2f}R  중앙값 {F.r_flip.median():+.2f}R  "
          f"(고정 TP = 2.0R)", flush=True)
    print(f"  분포: <0.5R {100 * (F.r_flip < 0.5).mean():.0f}% / "
          f"0.5~1R {100 * ((F.r_flip >= 0.5) & (F.r_flip < 1)).mean():.0f}% / "
          f"1~2R {100 * ((F.r_flip >= 1) & (F.r_flip < 2)).mean():.0f}% / "
          f">=2R {100 * (F.r_flip >= 2).mean():.0f}%", flush=True)

    print("\n=== 홀드했다면 (flip 시각 이후 24h, 2R TP vs SL 선착) ===", flush=True)
    vc = F["first"].value_counts()
    for k in ("TP", "SL", "미도달"):
        n = int(vc.get(k, 0))
        print(f"  {k:<6} {n:3d}건 ({100 * n / len(F):.0f}%)", flush=True)

    # 기대값 — 현행(flip 즉시) vs 홀드
    ev_now = F.r_flip.mean()
    tp_n = int(vc.get("TP", 0)); sl_n = int(vc.get("SL", 0)); nd_n = int(vc.get("미도달", 0))
    # 미도달은 flip 시점 이익 유지로 근사(보수적)
    ev_hold = (tp_n * 2.0 + sl_n * -1.0 + nd_n * F[F["first"] == "미도달"].r_flip.mean()
               ) / len(F) if len(F) else 0.0
    print(f"\n  현행(flip 즉시 청산) 기대값: {ev_now:+.2f}R", flush=True)
    print(f"  홀드(2R TP / SL) 기대값   : {ev_hold:+.2f}R", flush=True)
    print(f"  → {'홀드 우세 (타겟 늦출 근거)' if ev_hold > ev_now else 'flip 유지가 우세'}",
          flush=True)

    print("\n=== 심볼별 ===", flush=True)
    g = F.groupby("sym").agg(n=("pnl", "size"), r_flip=("r_flip", "mean"),
                             tp=("first", lambda x: (x == "TP").sum()),
                             sl=("first", lambda x: (x == "SL").sum())).round(2)
    print(g.to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
