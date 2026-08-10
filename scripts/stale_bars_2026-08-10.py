"""#STALE 2026-08-10: 진입 대기창(setup_stale_bars) 재검증 — 진짜 빈도 병목.

## 왜 이게 남은 축인가
오늘 확인한 것들:
  · 창당 1회 제한을 풀어도 거래가 4% 만 는다(247→257)
  · FVG 재사용 1회/2회/무제한이 **결과가 완전히 동일** — 같은 FVG 를 두 번 쓰는
    경우가 아예 없다
둘 다 원인이 하나로 모인다. **셋업이 잡히고 3봉(15분) 안에 가격이 되돌아와야
진입**하는데 그게 대부분 일어나지 않는다. 다른 제한들은 이 앞에서 이미 무력하다.

## 이 값의 내력
2026-06-17 에 120봉(10시간) → **3봉(15분)** 으로 조였다. 근거는
"120봉은 질 낮은 진입 양산해 5년 −8% 적자, 3봉이어야 흑자(cisd+po3 조합 시 +3.18%)".
그런데 그 백테는 8/8 에 **라이브의 1/7 만 보고 있었음**이 드러난 그 백테이고,
근거로 든 `cisd` 는 8/9 재판정에서 홀드아웃 소멸(+0.906R → +0.135R)했다.
즉 **이 값의 근거는 두 겹으로 무너져 있다.**

## 변형 (사전등록 격자)
    3(현행) · 6 · 12 · 24 · 48 · 120
15분 / 30분 / 1시간 / 2시간 / 4시간 / 10시간에 해당한다.

## 판정
거래가 늘면서 건당 R 이 유지/개선돼야 한다. **늘어난 거래만 따로** 봐서
(증분 건당 R) 추가분이 손해면 그 지점에서 끊는다 — 총량만 보면 좋은 거래가
나쁜 거래를 가려준다. 유망하면 전체 5년 + 홀드아웃 4관문으로 확정한다.
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
BARS = 120_000                       # 최근 약 14개월 (방향 확인용)
GRID = (3, 6, 12, 24, 48, 120)       # 사전등록 — 사후에 넓히지 않는다


def run(sym: str, stale: int):
    df = _resample(_load_full(sym)).tail(BARS)
    cfg = live_cfg(sym, {"setup_stale_bars": stale})
    # stale 은 timeline 빌드가 아니라 재생 단계 게이트라 **캐시가 공유된다**
    tl = cached_setup_timeline(df, cfg, sym)
    bt = run_backtest_from_timeline(df, tl, cfg)
    months = (df.index[-1] - df.index[0]).days / 30.4
    out = []
    for t in bt.trades:
        risk = abs(float(t.entry) - float(getattr(t, "entry_sl", 0.0) or 0.0))
        if risk <= 0 or t.entry <= 0:
            continue
        out.append({
            "key": (sym, int(t.entry_idx)),
            "r": float(t.raw_pnl_pct) * float(t.entry) / risk,
            "mmbm": "mmbm" in tuple(t.confluences),
        })
    return out, months


def main() -> int:
    print(f"=== 진입 대기창(stale) 재검증 — 최근 {BARS:,}봉", flush=True)
    print("  현행 3봉(15분)은 2026-06-17 에 정해졌고, 그 근거가 된 백테와"
          " cisd 가 모두 무너졌다.", flush=True)
    print(f"\n  {'대기창':<14}{'거래':>6}{'월빈도':>8}{'건당R':>9}{'승률':>7}"
          f"{'SB':>6}{'SB건당R':>10}   {'증분(현행 대비)':<22}", flush=True)

    base_keys: set = set()
    base_stat = None
    for stale in GRID:
        t0 = time.time()
        allr, months = [], 0.0
        for sym in PAIRS:
            rows, m = run(sym, stale)
            allr += rows
            months = max(months, m)
        r = np.array([x["r"] for x in allr])
        sb = np.array([x["r"] for x in allr if not x["mmbm"]])
        keys = {x["key"] for x in allr}

        inc = ""
        if base_stat is None:
            base_keys, base_stat = keys, r.mean()
        else:
            add = [x["r"] for x in allr if x["key"] not in base_keys]
            if add:
                a = np.array(add)
                inc = f"+{len(a)}건 · 건당 {a.mean():+.3f}R"
            else:
                inc = "추가 없음"

        lab = f"{stale}봉({stale * 5}분)" if stale * 5 < 60 else f"{stale}봉({stale * 5 // 60}시간)"
        print(f"  {lab:<14}{len(r):>6}{len(r) / max(months, 1e-9):>8.2f}"
              f"{r.mean():>+9.3f}{100 * (r > 0).mean():>6.0f}%"
              f"{len(sb):>6}{(sb.mean() if len(sb) else float('nan')):>+10.3f}"
              f"   {inc:<22} ({time.time() - t0:.0f}초)", flush=True)

    print("\n  판정 — 증분 건당 R 이 양수여야 그 지점까지 넓힐 값어치가 있다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
