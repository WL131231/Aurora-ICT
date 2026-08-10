"""#FVG-REUSE 2026-08-10: 창당 1회 제한을 풀고 FVG 재사용 제한으로 — 파트너 결정.

## 왜
`detect_silver_bullet_setups` 는 **하루에 창(killzone/macro/SB) 하나당 셋업 1건**만
채택한다. 2026-05-12 첫 커밋부터 있었고 커밋 메시지에 "윈도우/일 1회" 한 줄이 전부다
— **근거 기록이 없다.** 정통 SB 의 "하루 세 발" 개념을 그대로 옮긴 것으로 보이는데,
지금은 킬존·매크로까지 창을 넓혀놨으므로 그 전제가 맞지 않는다.

빈도(2페어 월 6.8건)가 우리 최대 약점인데 검증되지 않은 상한이 걸려 있었다.
8/9 킬존 확대 검증에서 시간대를 넓혔는데 거래가 **오히려 줄어든**(402→380) 것도
이 제한 때문이다 — 이른 시간대가 창을 먼저 소진해 자리만 옮겨갔다.

## 파트너 제안
"FVG 는 사실상 1회용이거나 닿을수록 힘이 약해지니까, 같은 FVG 는 2번까지만."
창 제한 대신 **FVG 재사용 횟수**로 바꾸는 것 — 개념적으로도 더 맞다.

## 변형 (사전등록)
    BASE   현행 — 창당 1회
    FVG1   창 제한 해제 + 같은 FVG 1회      (대조군: 창→FVG 로 축만 교체)
    FVG2   창 제한 해제 + 같은 FVG 2회      ★ 파트너 안
    FREE   창 제한 해제 + 무제한            (상한 확인용 — 같은 자리 반복 위험)

## 범위
1차는 **최근 12만 봉(약 14개월)** 으로 방향만 본다. 전체 5년은 변형당 2시간이라
유망한 것만 뒤에 돌린다. 판정은 그때 4관문으로.
"""

from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from live_parity import live_cfg  # noqa: E402

from aurora_ict.backtest.replay import run_backtest_from_timeline  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT"]
BARS = 120_000          # 최근 약 14개월
VARIANTS = (
    ("BASE (창당 1회)", {}),
    ("FVG1 (FVG 1회)", {"window_once": False, "max_per_fvg": 1}),
    ("FVG2 (FVG 2회)", {"window_once": False, "max_per_fvg": 2}),
    ("FREE (무제한)", {"window_once": False}),
)


def run(sym: str, extra: dict):
    df = _resample(_load_full(sym)).tail(BARS)
    cfg = live_cfg(sym, extra or None)
    tl = cached_setup_timeline(df, cfg, sym)
    bt = run_backtest_from_timeline(df, tl, cfg)
    months = (df.index[-1] - df.index[0]).days / 30.4
    rows = []
    for t in bt.trades:
        risk = abs(float(t.entry) - float(getattr(t, "entry_sl", 0.0) or 0.0))
        if risk <= 0 or t.entry <= 0:
            continue
        rows.append({
            "r": float(t.raw_pnl_pct) * float(t.entry) / risk,
            "mmbm": "mmbm" in tuple(t.confluences),
            "dir": str(getattr(t.direction, "value", t.direction)).lower(),
        })
    return rows, months


def main() -> int:
    print(f"=== FVG 재사용 제한 — 최근 {BARS:,}봉 (방향 확인용)", flush=True)
    print("  창당 1회 제한은 2026-05-12 첫 커밋부터 근거 없이 있었다.", flush=True)
    print(f"\n  {'변형':<18}{'거래':>6}{'월빈도':>8}{'건당R':>9}{'승률':>7}"
          f"{'SB':>7}{'MMBM':>7}   {'SB 건당R':>10}", flush=True)

    for name, extra in VARIANTS:
        t0 = time.time()
        allr, months = [], 0.0
        for sym in PAIRS:
            try:
                rows, m = run(sym, extra)
            except Exception as e:  # noqa: BLE001
                print(f"  {name:<18}실패 — {type(e).__name__}: {str(e)[:50]}", flush=True)
                allr = []
                break
            allr += rows
            months = max(months, m)
        if not allr:
            continue
        r = np.array([x["r"] for x in allr])
        sb = np.array([x["r"] for x in allr if not x["mmbm"]])
        mm = [x for x in allr if x["mmbm"]]
        print(f"  {name:<18}{len(r):>6}{len(r) / max(months, 1e-9):>8.2f}"
              f"{r.mean():>+9.3f}{100 * (r > 0).mean():>6.0f}%"
              f"{len(sb):>7}{len(mm):>7}"
              f"{(sb.mean() if len(sb) else float('nan')):>+10.3f}"
              f"   ({time.time() - t0:.0f}초)", flush=True)

    print("\n  판정 — 거래가 늘면서 건당 R 이 유지/개선돼야 의미가 있다.", flush=True)
    print("  거래만 늘고 건당 R 이 떨어지면 같은 자리를 반복해 먹는 것일 뿐이다.",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
