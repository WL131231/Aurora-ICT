"""#AUTONOMOUS 2026-07-30: Origo 2.3(flip 1.5R 배포판)이 **횡보에서 어떤가** 재확인.

파트너 질문: "현행 로직이 횡보 자체에서는 어떤지 다시 한 번 봐보자."

왜 다시 보나 — 그동안 "횡보가 문제"를 전제로 몇 주를 팠으나 오늘 라이브 실측에서
손실 원인은 **flip 조기 절단**(0.61R)으로 특정됐고, 그걸 고쳐 배포했다(#FLIP-MIN-R).
전제가 바뀌었으니 횡보 성적도 새 조건에서 다시 재야 한다.

이전 측정과 다른 점 (전부 오늘 확보):
  · **라이브 정합 하니스**(live_parity.py) — regime_filter·cond_align·nypm 포함.
    그전 국면 분해(7/29)는 이 게이트들이 없는 상태였다.
  · **flip 최소 1.5R 적용** — 배포판 그대로.
  · `entry_sl` 실측 필드 — R 단위 계산 정확.

두 가지 '횡보' 를 따로 본다(스케일이 60배 달라 서로 다른 현상):
  ① CSI  — 12시간 스케일 인식(재료 8종 로지스틱, 정밀도 63.7%)
  ② BTC 30일 변화율 ±15% — 장기 추세 부재 국면
그리고 **regime_filter 가 걸러낸 거래의 성적**도 함께 본다(게이트가 제 일을 하는가).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chop_state_index import csi_series, fit_csi, load_1h  # noqa: E402
from fib_regime_mtf import btc_regime  # noqa: E402
from flip_ab_backtest import build_fvg_zones  # noqa: E402
from flip_verdict import apply_flip  # noqa: E402
from live_parity import (  # noqa: E402
    LIVE_FLIP_MIN_R, PAIRS, gate_cond_align, gate_nypm, gate_regime, line,
    run_live_parity, stat,
)


def summarize(rows, label: str, min_n: int = 15) -> None:
    if len(rows) < min_n:
        print(f"  {label:<22} n={len(rows):3d} 표본부족", flush=True)
        return
    s = stat(rows)
    be = (100 - s["wr"]) / max(s["wr"], 1e-9)
    verdict = "흑자권" if s["rr"] > be else "적자구조"
    print(f"  {label:<22} n={s['n']:3d} net={s['net']:+8.1f}% 승률={s['wr']:3.0f}% "
          f"RR={s['rr']:4.2f} 분기={be:4.2f} {verdict}", flush=True)


def main() -> int:
    print("CSI 모델 학습(앞 70%)...", flush=True)
    model = fit_csi(PAIRS)
    reg_map = btc_regime("1h")

    kept_rows: list = []      # (ts, net, sym, csi, reg)
    skip_rows: list = []      # regime_filter/cond_align/nypm 이 걸러낸 것
    for sym in PAIRS:
        df5, kept, st = run_live_parity(sym)
        zones = build_fvg_zones(df5)
        # 배포판 그대로 — flip 최소 1.5R
        res, _ = apply_flip(df5, kept, zones, mode="minr", min_w=4,
                            min_r=LIVE_FLIP_MIN_R, sym=sym)
        csi = csi_series(load_1h(sym), model).reindex(df5.index, method="ffill").shift(1)
        reg = reg_map.reindex(df5.index, method="ffill")
        for (ts, net, s), t in zip(res, kept):
            i = t.entry_idx
            cv = float(csi.iloc[i]) if i < len(csi) and not pd.isna(csi.iloc[i]) else np.nan
            rv = reg.iloc[i] if i < len(reg) else "횡보"
            kept_rows.append((ts, net, s, cv, rv if isinstance(rv, str) else "횡보"))
        print(f"  {sym}: 통과 {len(kept)}건 (게이트 차단 "
              f"regime {st['regime']}·cond {st['cond_align']}·nypm {st['nypm']})", flush=True)

    base = [(r[0], r[1], r[2]) for r in kept_rows]
    print("\n\n===== 전체 (Origo 2.3 배포판 = 정합 게이트 + flip 1.5R) =====", flush=True)
    print("  " + line(stat(base)), flush=True)

    print("\n\n===== ① CSI 구간별 (12시간 스케일 횡보 인식) =====", flush=True)
    have = [r for r in kept_rows if not np.isnan(r[3])]
    if have:
        qs = np.quantile([r[3] for r in have], [1 / 3, 2 / 3])
        print(f"  (3분위 경계 CSI {qs[0]:.3f} / {qs[1]:.3f})", flush=True)
        for lab, sel in (
            ("추세측 (CSI 하위33%)", lambda c: c < qs[0]),
            ("중간", lambda c: qs[0] <= c < qs[1]),
            ("횡보측 (CSI 상위33%)", lambda c: c >= qs[1]),
        ):
            rows = [(r[0], r[1], r[2]) for r in have if sel(r[3])]
            summarize(rows, lab)
        # 절대 임계도 — 라이브 게이트 후보로 쓰였던 값들
        for thr in (0.5, 0.6):
            rows = [(r[0], r[1], r[2]) for r in have if r[3] >= thr]
            summarize(rows, f"CSI >= {thr} (횡보 인식)")

    print("\n\n===== ② BTC 30일 국면별 =====", flush=True)
    for rg in ("상승", "횡보", "하락"):
        rows = [(r[0], r[1], r[2]) for r in kept_rows if r[4] == rg]
        summarize(rows, f"{rg}장")

    print("\n\n===== ③ 게이트가 걸러낸 거래는 실제로 나빴나 =====", flush=True)
    print("  (regime_filter/cond_align/nypm 이 차단한 거래를 통과시켰다면?)", flush=True)
    for sym in PAIRS:
        pass
    blocked: list = []
    for sym in PAIRS:
        df5, kept, _ = run_live_parity(sym)
        # 게이트 전 전체 trade 를 다시 얻어 차단분만 추출
        from live_parity import LIVE_BASE  # noqa: PLC0415
        from bt_par import cached_setup_timeline  # noqa: PLC0415
        from aurora_ict.backtest.replay import (  # noqa: PLC0415
            BacktestConfig, run_backtest_from_timeline,
        )
        cfg = BacktestConfig(**LIVE_BASE)
        bt = run_backtest_from_timeline(df5, cached_setup_timeline(df5, cfg, sym), cfg)
        kept_idx = {t.entry_idx for t in kept}
        for t in bt.trades:
            if t.entry_idx in kept_idx:
                continue
            ts = df5.index[t.entry_idx]
            why = ("regime" if not gate_regime(t, sym)
                   else "cond_align" if not gate_cond_align(t, sym)
                   else "nypm" if not gate_nypm(ts) else "기타")
            blocked.append((ts, t.net_pnl_pct, sym, why))
    for why in ("regime", "cond_align", "nypm"):
        rows = [(b[0], b[1], b[2]) for b in blocked if b[3] == why]
        summarize(rows, f"{why} 차단분")
    allb = [(b[0], b[1], b[2]) for b in blocked]
    summarize(allb, "차단 전체")
    print("\n  → 차단분 net 이 음수여야 게이트가 제 일을 하는 것.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
