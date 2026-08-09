"""#AUTONOMOUS 2026-07-31: Origo 를 하이켄아시 캔들로 돌리면? (파트너 호기심).

Cursus 개발자가 하이켄아시를 제안한 김에 Origo 에도 적용해본다.
Origo 는 ICT FVG 되돌림 진입이라 HA 로 바꾸면 **검출되는 FVG 자체가 달라진다**
(HA 는 평활돼 갭이 줄고, 대신 추세 구간의 연속 몸통이 뚜렷해진다).

## 방법 — 신호는 HA, 체결은 실가격
`cached_setup_timeline(HA_df)` 로 셋업(FVG·스윙·CHoCH)을 만들고,
`run_backtest_from_timeline(실제_df, tl)` 로 **실제 가격에서 체결**시킨다.
timeline 만 HA 기준이고 손익은 실가격이라 정직하다.

## ⚠️ 구조적 한계 (결론에 반드시 명시)
HA_open/HA_close 는 **계산값이라 실제로 거래되지 않은 가격**일 수 있다.
Origo 는 setup.entry/stop_loss/take_profit 이 그 캔들 좌표에서 나오므로,
HA 기준 진입가에 지정가를 걸면 **실제로 그 가격에 체결되지 않을 위험**이 있다.
(HA_high/HA_low 는 실제 고저를 포함해 그나마 안전하지만 open/close 는 아니다.)
따라서 이 실험은 "HA 신호가 더 나은가" 를 보는 **탐색**이며, 좋게 나와도
그대로 배포할 수 없다 — 체결 가능한 좌표로 재설계가 필요하다.

비교: 현행(실제 캔들) vs HA 캔들. 라이브 정합 게이트(regime·cond_align·nypm)
      + flip 1.5R 을 양쪽 동일 적용해 신호 차이만 분리한다.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from cursus_dev_changes import heikin_ashi  # noqa: E402
from flip_ab_backtest import build_fvg_zones  # noqa: E402
from flip_verdict import apply_flip  # noqa: E402
from live_parity import (  # noqa: E402
    LIVE_BASE, LIVE_FLIP_MIN_R, PAIRS, gate_cond_align, gate_nypm, gate_regime,
    line, stat,
)

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402


def run(use_ha: bool):
    rows = []
    per: dict[str, float] = {}
    for sym in PAIRS:
        df5 = _resample(_load_full(sym))
        cfg = BacktestConfig(**LIVE_BASE)
        # 신호(FVG·스윙·CHoCH) 생성용 캔들 — HA 또는 실제
        sig_df = heikin_ashi(df5) if use_ha else df5
        tl = cached_setup_timeline(sig_df, cfg, sym + ("_HA" if use_ha else ""))
        # 체결·손익은 **항상 실제 캔들**로
        bt = run_backtest_from_timeline(df5, tl, cfg)
        kept = []
        for t in bt.trades:
            ts = df5.index[t.entry_idx]
            if not gate_regime(t, sym) or not gate_cond_align(t, sym) or not gate_nypm(ts):
                continue
            kept.append(t)
        if not kept:
            per[sym] = 0.0
            continue
        zones = build_fvg_zones(df5)
        res, _ = apply_flip(df5, kept, zones, mode="minr", min_w=4,
                            min_r=LIVE_FLIP_MIN_R, sym=sym)
        rows.extend(res)
        per[sym] = sum(p for _, p, _ in res) * 100
        print(f"  {sym}: 원본 {len(bt.trades)} → 게이트통과 {len(kept)}건", flush=True)
    return stat(rows), per


def main() -> int:
    print("=== Origo × 하이켄아시 (신호만 HA, 체결은 실가격) ===\n", flush=True)
    print("[현행 — 실제 캔들]", flush=True)
    s_cur, per_cur = run(use_ha=False)
    print(f"  {line(s_cur)}\n", flush=True)

    print("[하이켄아시 캔들]", flush=True)
    s_ha, per_ha = run(use_ha=True)
    print(f"  {line(s_ha)}\n", flush=True)

    print("\n===== 비교 =====", flush=True)
    for lab, s in (("현행(실제 캔들)", s_cur), ("하이켄아시", s_ha)):
        if s is None:
            print(f"  {lab:<16} 거래 없음", flush=True)
            continue
        be = (100 - s["wr"]) / max(s["wr"], 1e-9)
        print(f"  {lab:<16} n={s['n']:4d} net={s['net']:+8.1f}% 승률={s['wr']:3.0f}% "
              f"RR={s['rr']:4.2f} 분기={be:4.2f} MDD={s['mdd']:6.1f} "
              f"연도{s['ypos']}/{s['ytot']} 페어{s['spos']}/{s['stot']}", flush=True)
    if s_cur and s_ha:
        print(f"\n  → 차이 {s_ha['net'] - s_cur['net']:+.1f}% "
              f"(거래 {s_ha['n'] - s_cur['n']:+d}건)", flush=True)

    print("\n[페어별]", flush=True)
    print(f"  {'페어':<8}{'현행':>10}{'HA':>10}{'차이':>10}", flush=True)
    for sym in PAIRS:
        c, h = per_cur.get(sym, 0.0), per_ha.get(sym, 0.0)
        print(f"  {sym.replace('USDT', ''):<8}{c:>+10.1f}{h:>+10.1f}{h - c:>+10.1f}",
              flush=True)

    print("\n\n⚠️ 한계: HA_open/close 는 계산값이라 **실제 체결되지 않은 가격**일 수 있다.", flush=True)
    print("   Origo 는 진입가가 캔들 좌표에서 나오므로, 좋게 나와도 그대로 배포 불가 —", flush=True)
    print("   체결 가능한 좌표로 재설계가 필요하다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
