"""#AUTONOMOUS 2026-07-30: 라이브 횡보 판정기 **교체** 실험 — q33 추세크기 → CSI (파트너 지시).

파트너 정정(핵심): 앞선 7/27 "수비 게이트 기각" 은 기존 게이트 **위에 얹는** 추가
게이트(ADX·CHOP·ER·BBW) 실험이었다. 파트너 의도는 다르다 —
**이미 라이브에 있는 횡보 판정기를 더 정확한 것으로 교체**하는 것.

현행 라이브(bot_ict_instance.py `regime_filter_enabled`, 6/23 배포):
    판정 = |entry_trend_pct| (진입 직전 20봉 추세 크기) < 페어별 q33(롤링 33분위) → skip
    이 게이트 자체는 net 흑자의 필수조건으로 검증됨(없애면 7페어 적자, q33 넣으면 +18).
    그러나 판정기는 "20봉 추세가 작다" 는 거친 근사.
CSI(chop_state_index.py): 재료 8종 복합 로지스틱, 횡보 **인식** 정밀도 61~63.7%.
→ 게이트 골격은 그대로 두고 **판정기만 교체**하면 같은 회피 로직이 더 정확해지는가?
   (이건 csi_applications.py 4갈래[사이즈·연속손실·TP단축·볼밴]에 없던 새 축.)

변형:
  G0 게이트 OFF            (대조군 — 회피 안 함)
  G1 현행 q33 추세크기      (라이브 기준선)
  G2 CSI > thr             (교체)
  G3 q33 AND CSI           (둘 다 횡보일 때만 skip = 보수적)
  G4 q33 OR  CSI           (하나라도 횡보면 skip = 적극적)

⚠️ look-ahead 방어: CSI 는 앞 70% 로 학습된 모델이다. 전 구간 적용은 학습구간이
   낙관 편향이 된다 → **[전구간]과 [검증구간(뒤 30%)]을 따로** 출력. 정직한 판정은 후자.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from chop_state_index import csi_series, fit_csi, load_1h  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
# 라이브 정합 — 단 regime_filter 는 **끈다**(전 거래를 뽑아 사후에 각 판정기로 필터).
BASE = dict(
    htf_ema_bias="align", htf_align_threshold=2, sl_liq_cap=True,
    min_confluence=5, sl_dist_mult=4.0, setup_stale_bars=3,
    apply_cisd=True, apply_po3=True, disable_time_filter=False, size_pct=0.9,
    ote_level=0.707, min_rr=2.0, tp_rr_override=0.0, entry_ttl_bars=6,
    trail_trigger=2.0, trail_dist=1.5, partial_tp_rr=1.5, partial_be=True,
)


def collect(sym: str, model):
    """게이트 OFF 상태 전 거래 + 각 거래의 (진입추세크기, CSI값, 시각)."""
    df5 = _resample(_load_full(sym))
    cfg = BacktestConfig(**BASE)
    tl = cached_setup_timeline(df5, cfg, sym)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    csi = csi_series(load_1h(sym), model).reindex(df5.index, method="ffill")
    rows = []
    for t in bt.trades:
        ts = df5.index[t.entry_idx]
        trend = abs(getattr(t, "entry_trend_pct", np.nan) or np.nan)
        rows.append(dict(ts=ts, pnl=t.net_pnl_pct, trend=trend,
                         csi=float(csi.iloc[t.entry_idx]) if t.entry_idx < len(csi) else np.nan,
                         sym=sym))
    return pd.DataFrame(rows)


def summarize(d: pd.DataFrame, label: str) -> dict:
    if d.empty:
        return dict(label=label, n=0, net=0.0)
    net = d.pnl.sum()
    wr = 100 * (d.pnl > 0).mean()
    ys = d.groupby(d.ts.dt.year).pnl.sum()
    half = len(d) // 2
    ds = d.sort_values("ts")
    h1 = ds.pnl.iloc[:half].sum(); h2 = ds.pnl.iloc[half:].sum()
    syms = d.groupby("sym").pnl.sum()
    return dict(label=label, n=len(d), net=net, wr=wr, h1=h1, h2=h2,
                ypos=int((ys > 0).sum()), ytot=len(ys),
                spos=int((syms > 0).sum()), stot=len(syms),
                ys=" ".join(f"{y}:{v:+.1f}" for y, v in ys.items()))


def line(s: dict) -> str:
    if s["n"] == 0:
        return f"  {s['label']:<22} 거래 없음"
    return (f"  {s['label']:<22} n={s['n']:4d} net={s['net']:+8.1f} 승률={s['wr']:3.0f}% "
            f"H1={s['h1']:+7.1f} H2={s['h2']:+7.1f} 연도{s['ypos']}/{s['ytot']} "
            f"페어{s['spos']}/{s['stot']}")


def report(D: pd.DataFrame, title: str, thrs=(0.5, 0.6, 0.7)):
    print(f"\n\n########## {title} (거래 {len(D)}건) ##########", flush=True)
    # 페어별 q33 — 라이브의 롤링 33분위를 근사(페어별 전체 분위)
    q33 = D.groupby("sym").trend.quantile(1 / 3).to_dict()
    D = D.copy()
    D["is_chop_q33"] = D.apply(
        lambda r: (not np.isnan(r.trend)) and r.trend < q33.get(r.sym, 0.0), axis=1)
    print(line(summarize(D, "G0 게이트 OFF")), flush=True)
    keep_q33 = D[~D.is_chop_q33]
    print(line(summarize(keep_q33, "G1 현행 q33(기준선)")), flush=True)
    skip_q33 = D[D.is_chop_q33]
    print(f"     └ q33 이 걸러낸 거래: n={len(skip_q33)} net={skip_q33.pnl.sum():+.1f}", flush=True)
    for thr in thrs:
        chop_csi = D.csi >= thr
        keep = D[~chop_csi]
        skip = D[chop_csi]
        print(line(summarize(keep, f"G2 CSI>={thr} 교체")), flush=True)
        print(f"     └ CSI 가 걸러낸 거래: n={len(skip)} net={skip.pnl.sum():+.1f}", flush=True)
        both = D[~(D.is_chop_q33 & chop_csi)]
        print(line(summarize(both, f"G3 q33 AND CSI>={thr}")), flush=True)
        either = D[~(D.is_chop_q33 | chop_csi)]
        print(line(summarize(either, f"G4 q33 OR CSI>={thr}")), flush=True)


def main() -> int:
    print("CSI 모델 학습(앞 70%)...", flush=True)
    model = fit_csi(PAIRS)
    ds = []
    for sym in PAIRS:
        try:
            d = collect(sym, model)
            ds.append(d)
            print(f"  {sym}: {len(d)}건", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {sym} 실패: {type(e).__name__} {e}", flush=True)
    D = pd.concat(ds, ignore_index=True).dropna(subset=["csi"])
    if D.empty:
        print("거래 없음 — 중단", flush=True)
        return 1
    print(f"\n진입추세 결측: {D.trend.isna().mean() * 100:.0f}%", flush=True)

    report(D, "[전 구간] ⚠️ CSI 학습구간 포함 — 낙관 편향 있음(참고용)")
    # 검증 구간 = 각 페어 시계열의 뒤 30% (CSI 학습에 쓰이지 않은 구간)
    cut = D.ts.quantile(0.7)
    print(f"\n검증 구간 경계: {cut}", flush=True)
    report(D[D.ts >= cut], "[검증 구간 뒤30%] ★정직한 판정")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
