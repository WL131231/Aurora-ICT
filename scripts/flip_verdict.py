"""#AUTONOMOUS 2026-07-30: flip 처방 **최종 재판정** — 라이브 정합 하니스 위에서.

## 왜 재판정인가
오늘 1~5차 flip A/B(F1 +254.9% 최고, 처방 전부 악화)는 **게이트 없는 하니스**에서
돌린 것이라 무효였다. 전면 감사 결과 regime_filter·cond_align·exclude_nypm 등
10개 라이브 기능이 공용 하니스에 미구현이었다([[live_parity]] 참조).

라이브 정합 하니스(live_parity.py)로 잡은 기준선:
    Origo 2.2 flip 없음 → n=126 net **+1703%** 승률 **47%** RR **1.94** (흑자권)
    라이브 실측         → 승률 **46%** RR **0.94** (적자)
    승률은 일치, RR 만 2배 차이 → 차이는 **청산 경로 하나(HTF FVG flip)**.

## 이 스크립트의 검증 게이트
flip 을 넣었을 때 백테 RR 이 **라이브 0.94 근처를 재현**해야 한다.
재현되면 그 하니스로 낸 처방 판정을 신뢰할 수 있다. 안 되면 판정 보류.

시나리오: F0 flip없음 / F1 라이브정합(1h+) / F2 4h+ / A 부분청산 / B 최소R
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flip_ab_backtest import _dir_of, _sl_of, build_fvg_zones, flip_target_at  # noqa: E402
from live_parity import (  # noqa: E402
    LIVE_BASE, PAIRS, line, parity_report, run_live_parity, stat,
)

from aurora.backtest.cost import apply_costs  # noqa: E402

LIVE_WR, LIVE_RR = 46.0, 0.94
LEVERAGE = 20.0        # replay BacktestConfig.leverage 기본값과 동일


def apply_flip(df5, trades, zones, mode: str, min_w: int = 4, min_r: float = 0.0,
               partial: float = 0.0, sym: str = ""):
    """라이브 정합 trade 목록에 flip 청산을 사후 적용. 반환: (결과, 발동R 목록)."""
    h5 = df5["high"].to_numpy(); l5 = df5["low"].to_numpy()
    out, fired = [], []
    for t in trades:
        ts = df5.index[t.entry_idx]
        base = t.net_pnl_pct
        if mode == "off":
            out.append((ts, base, sym)); continue
        d = _dir_of(t)
        entry = float(t.entry)
        z = flip_target_at(zones, t.entry_idx, d, entry, min_w)
        if z is None:
            out.append((ts, base, sym)); continue
        lo_i, hi_i = t.entry_idx + 1, min(t.exit_idx, len(df5) - 1)
        if hi_i <= lo_i:
            out.append((ts, base, sym)); continue
        seg_hi = h5[lo_i:hi_i + 1]; seg_lo = l5[lo_i:hi_i + 1]
        hit = np.flatnonzero((seg_lo <= z["hi"]) & (seg_hi >= z["lo"]))
        if not hit.size:
            out.append((ts, base, sym)); continue
        # FlipWatcher = 틱 감시, zone 경계 1회 touch 즉시 flip(wick 포함) → edge 체결.
        fpx = z["lo"] if d == 1 else z["hi"]
        raw = (fpx - entry) / entry * d
        risk = abs(entry - _sl_of(t, entry, d))
        r_at = (raw * entry / risk) if risk > 0 else 0.0
        # ⚠️ 단위 주의 — raw 는 **가격 변동 비율**(0.0084=0.84%)이고 net_pnl_pct 는
        # **레버리지·사이즈 반영 시드 대비**(0.30=30%)다. 그냥 더하면 flip 손익이
        # 사실상 0 이 되어 시나리오 차이가 0.8% 로 뭉개진다(7/30 1차 판정 오류).
        # replay 와 동일한 apply_costs 로 환산해야 정합.
        fnet, _ = apply_costs(raw, LIVE_BASE["size_pct"], LEVERAGE)
        fired.append(r_at)
        if mode == "flip":
            out.append((ts, fnet, sym))
        elif mode == "minr":
            out.append((ts, fnet if r_at >= min_r else base, sym))
        elif mode == "partial":
            out.append((ts, partial * fnet + (1 - partial) * base, sym))
        else:
            out.append((ts, base, sym))
    return out, fired


SCEN = [
    ("F0 flip 없음", dict(mode="off")),
    ("F1 라이브정합(1h+)", dict(mode="flip", min_w=4)),
    ("F2 4h+ 존만", dict(mode="flip", min_w=10)),
    ("A 부분청산 50%", dict(mode="partial", min_w=4, partial=0.5)),
    ("A 부분청산 30%", dict(mode="partial", min_w=4, partial=0.3)),
    ("B 최소 0.5R", dict(mode="minr", min_w=4, min_r=0.5)),
    ("B 최소 1.0R", dict(mode="minr", min_w=4, min_r=1.0)),
    ("B 최소 1.5R", dict(mode="minr", min_w=4, min_r=1.5)),
]


def main() -> int:
    agg: dict[str, list] = {n: [] for n, _ in SCEN}
    fired_f1: list[float] = []
    n_total = 0
    for sym in PAIRS:
        df5, kept, st = run_live_parity(sym)
        zones = build_fvg_zones(df5)
        n_total += len(kept)
        for name, kw in SCEN:
            res, fired = apply_flip(df5, kept, zones, sym=sym, **kw)
            agg[name].extend(res)
            if name == "F1 라이브정합(1h+)":
                fired_f1.extend(fired)
        print(f"  {sym}: 정합통과 {len(kept)}건 / zone {len(zones)}", flush=True)

    print("\n\n===== ★ 정합 검증 (이게 안 맞으면 아래 판정 무효) =====", flush=True)
    f1 = stat(agg["F1 라이브정합(1h+)"])
    if fired_f1:
        fa = np.array(fired_f1)
        print(f"  flip 발동률   백테 {100 * len(fa) / max(n_total, 1):.0f}%  vs 라이브 30%",
              flush=True)
        print(f"  발동 평균 R   백테 {fa.mean():+.2f}R  vs 라이브 +0.61R", flush=True)
    if f1:
        print(f"  승률          백테 {f1['wr']:.0f}%  vs 라이브 {LIVE_WR:.0f}%", flush=True)
        print(f"  RR            백테 {f1['rr']:.2f}  vs 라이브 {LIVE_RR:.2f}"
              f"   ← 핵심 지표", flush=True)
        ok = abs(f1["rr"] - LIVE_RR) < 0.35
        print(f"  → {'✅ 재현 성공 — 판정 유효' if ok else '⚠️ 미재현 — 판정 보류'}", flush=True)

    print("\n\n===== 시나리오 비교 (라이브 정합 하니스) =====", flush=True)
    base_net = stat(agg["F1 라이브정합(1h+)"])["net"]
    for name, _ in SCEN:
        s = stat(agg[name])
        mark = ""
        if s and name != "F1 라이브정합(1h+)":
            d = s["net"] - base_net
            mark = f"  현행대비 {d:+.1f}%" + ("  ★개선" if d > 0 else "")
        print(f"  {name:<20} {line(s)}{mark}", flush=True)
    print("\n  ※ 현행 라이브 = F1. F1 대비 net 개선 + 연도·페어 분산 유지가 배포 후보.",
          flush=True)
    parity_report(stat(agg["F1 라이브정합(1h+)"]), {"wr": LIVE_WR, "rr": LIVE_RR})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
