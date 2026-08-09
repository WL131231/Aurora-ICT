"""#AUTONOMOUS 2026-07-30: 라이브 flip_close 반사실 분석 — "안 나갔으면 어땠나".

발견(라이브 Origo 2.2, 100청산):
  tp_hit  14건 평균 **+7.51**   ← 고정 TP(2R) 도달
  flip_close 30건 평균 **+3.51** ← 반대방향 HTF FVG 도달 청산
  sl_hit  53건 평균 -5.05
→ 익절의 2/3 가 작은 문(flip)으로 나간다. 실현 RR 0.94 (손익분기 1.17).

⚠️ 백테로 처방을 찾을 수 없다: replay.py 에 **HTF FVG flip 경로가 미구현**
   (liq_tp/swing_tp/liquidity 0회). 라이브 익절 2/3 를 만드는 경로가 백테에 없어
   백테 튜닝은 다른 물건을 만지는 것이 된다(7/27 MMBM "배포≠검증 구성" 교훈).

대안 = **실거래 반사실**: flip 으로 청산한 실제 시각·가격·방향을 기준으로, 그 이후
가격이 유리하게 얼마나 더 갔는지 OHLCV 로 직접 추적한다.
  · fwd_max  : 청산 후 H봉 내 최대 유리 이동%  (조기 절단 여부)
  · fwd_min  : 청산 후 H봉 내 최대 불리 이동%  (flip 이 손실을 막아준 정도)
  · 비교군   : 같은 기간 tp_hit·sl_hit 건도 같은 방식으로 측정
판정: flip 후 유리 이동이 크고 불리 이동이 작으면 **조기 절단** → 타겟을 늦출 근거.
      불리 이동이 크면 flip 이 실제로 손실을 막고 있는 것 → 유지.
"""
from __future__ import annotations

import sys

import pandas as pd

CSV = "data/live/fst_snapshots/fst_2026-07-30_all_users.csv"
HORIZONS = (12, 48, 144)      # 5m 봉 기준 = 1h / 4h / 12h
# 기존 data/*_1m_full.parquet 은 6/18 에서 끝나 7월 라이브를 못 덮는다 →
# data/recent/ (5/21~7/30, fetch_ohlcv.py 로 신규 수집) 사용.
RECENT = "data/recent/{sym}_1m_recent.parquet"


def to_bt_symbol(s: str) -> str:
    return s.split("/")[0] + "USDT"


def load_recent(sym: str) -> pd.DataFrame:
    """최근 1m parquet → 5m 리샘플 (UTC 인덱스)."""
    d = pd.read_parquet(RECENT.format(sym=sym))
    d["ts"] = pd.to_datetime(d["timestamp"], unit="ms", utc=True)
    d = d.set_index("ts").sort_index()
    return d.resample("5min").agg({"open": "first", "high": "max",
                                  "low": "min", "close": "last"}).dropna()


def main() -> int:
    d = pd.read_csv(CSV)
    d["ts"] = pd.to_datetime(d.ts_ms, unit="ms", utc=True)
    o = d[d.model == "Origo 2.2"].copy()
    ev = o[o.event_type.isin(["flip_close", "tp_hit", "sl_hit"])].copy()
    ev["bt_sym"] = ev.symbol.map(to_bt_symbol)

    cache: dict[str, pd.DataFrame] = {}
    rows = []
    skipped: dict[str, int] = {}
    for r in ev.itertuples():
        sym = r.bt_sym
        if sym not in cache:
            try:
                cache[sym] = load_recent(sym)
            except Exception:  # noqa: BLE001
                cache[sym] = pd.DataFrame()
        df = cache[sym]
        if df.empty:
            skipped[sym] = skipped.get(sym, 0) + 1
            continue
        idx = df.index.searchsorted(r.ts)
        if idx >= len(df) - max(HORIZONS):
            skipped[sym + "(기간밖)"] = skipped.get(sym + "(기간밖)", 0) + 1
            continue
        px = float(r.price)
        if px <= 0:
            continue
        # 포지션 방향 — 롱이면 이후 상승이 유리
        sgn = 1.0 if str(r.direction).lower() == "long" else -1.0
        rec = dict(ev=r.event_type, sym=sym, pnl=r.pnl_usdt, ts=r.ts)
        for hz in HORIZONS:
            seg = df.iloc[idx:idx + hz]
            if seg.empty:
                continue
            hi = float(seg["high"].max()); lo = float(seg["low"].min())
            fav = (hi - px) / px * 100 if sgn > 0 else (px - lo) / px * 100
            adv = (px - lo) / px * 100 if sgn > 0 else (hi - px) / px * 100
            rec[f"fav{hz}"] = fav
            rec[f"adv{hz}"] = adv
        rows.append(rec)
    R = pd.DataFrame(rows)
    if skipped:
        print(f"제외: {skipped}", flush=True)
    print(f"\n분석 대상 {len(R)}건 (Origo 2.2 flip/tp/sl)\n", flush=True)
    if R.empty:
        print("대상 0건 — 중단", flush=True)
        return 1

    print("=== 청산 후 가격 경로 (방향 정규화, % — 레버리지 미적용) ===", flush=True)
    hdr = f"  {'경로':<12} {'n':>4}"
    for hz in HORIZONS:
        hdr += f" {'유리' + str(hz):>9} {'불리' + str(hz):>9}"
    print(hdr, flush=True)
    for evname in ("flip_close", "tp_hit", "sl_hit"):
        s = R[R.ev == evname]
        if s.empty:
            continue
        line = f"  {evname:<12} {len(s):>4}"
        for hz in HORIZONS:
            fc, ac = f"fav{hz}", f"adv{hz}"
            line += f" {s[fc].mean():>8.2f}% {s[ac].mean():>8.2f}%"
        print(line, flush=True)

    print("\n=== flip_close 상세 — 조기 절단 판정 ===", flush=True)
    f = R[R.ev == "flip_close"]
    if not f.empty:
        for hz in HORIZONS:
            fav, adv = f[f"fav{hz}"], f[f"adv{hz}"]
            better = (fav > adv).mean() * 100
            print(f"  +{hz}봉({hz * 5 / 60:.0f}h): 유리 평균 {fav.mean():+.2f}% / "
                  f"불리 평균 {adv.mean():+.2f}% / 유리>불리 비율 {better:.0f}%", flush=True)
        print("\n  해석: 유리 >> 불리 면 flip 이 이익을 조기 절단(타겟 늦출 근거).", flush=True)
        print("        불리 >= 유리 면 flip 이 손실을 막고 있음(유지).", flush=True)
        # 청산 후 20배 레버리지 환산 기대 손익 (참고)
        print("\n  [20x 환산 참고] flip 후 계속 들고 있었다면 최대 유리:", flush=True)
        for hz in HORIZONS:
            print(f"    +{hz}봉: 평균 {f[f'fav{hz}'].mean() * 20:+.1f}% (시드 대비)", flush=True)

    print("\n=== 심볼별 flip 후 유리 이동 (48봉=4h) ===", flush=True)
    if not f.empty and "fav48" in f:
        g = f.groupby("sym").agg(n=("pnl", "size"), pnl=("pnl", "sum"),
                                 fav=("fav48", "mean"), adv=("adv48", "mean")).round(2)
        print(g.sort_values("fav", ascending=False).to_string(), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
