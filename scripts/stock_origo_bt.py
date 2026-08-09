"""#AUTONOMOUS 2026-08-06: Origo(ICT FVG 되돌림) 를 주식에 적용.

파트너 요청 — Cursus 와 같은 질문("주식에서도 통하나")을 Origo 에도.

## 구조적으로 안 맞는 부분을 먼저 밝힌다

Origo 는 **24시간 시장 전제**로 만들어졌다. 주식에 옮기면 두 축이 깨진다:

① **킬존(런던·뉴욕 세션)** — 주식은 장중만 거래되고, 미국 주식은 NY 세션과 겹치지만
   한국 주식은 아시아 시간대다. 세션 필터를 그대로 두면 한국 종목은 진입이 거의
   0 이 되어 비교가 무의미하다 → `disable_time_filter=True` 로 **끄고** 돌린다.
   즉 여기서 재는 것은 "ICT 시간론"이 아니라 **FVG 되돌림 구조 자체의 엣지**다.
② **오버나이트 갭** — FVG(캔들 사이 빈 구간)는 주식에서 매일 장 시작에 인위적으로
   생긴다. 크립토의 FVG(유동성 공백)와 의미가 다르다. 게다가 SL 이 갭으로 건너뛰면
   그 가격에 못 팔린다 → SL 청산 건은 **다음 봉 시가로 재계산**한다(정직).

## 그 외 반영
- 레버리지 1배(주식 현물). replay 의 net_pnl_pct 는 20x·size 0.9 기준이라 쓰지 않고
  raw_pnl_pct 에서 다시 계산한다.
- 시장별 비용(미국 제로커미션 / 한국 거래세 포함).
- 롱/숏 분리 — 한국 개인은 공매도가 사실상 막혀 있어 롱만 실현 가능.
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import cached_setup_timeline  # noqa: E402
from live_parity import LIVE_BASE  # noqa: E402
from stock_cursus_bt import COST, line, stat  # noqa: E402
from stock_fetch import MARKETS, fetch  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402


def run_symbol(df: pd.DataFrame, sym: str, cost: float, *,
               gap_fill: bool = True, side_only: str | None = None):
    """Origo 셋업·체결을 주식 df 로 돌리고 1배 기준 손익 산출."""
    kw = dict(LIVE_BASE)
    kw["disable_time_filter"] = True     # 킬존 무의미 — 구조 엣지만 측정
    cfg = BacktestConfig(**kw)
    tl = cached_setup_timeline(df, cfg, f"STK_{sym}")
    bt = run_backtest_from_timeline(df, tl, cfg)
    o = df["open"].to_numpy(float)
    years = df.index.year.to_numpy()

    out = []
    for t in bt.trades:
        long_ = str(getattr(t.direction, "value", t.direction)).lower() == "long"
        if side_only is not None and (side_only == "long") != long_:
            continue
        sgn = 1.0 if long_ else -1.0
        raw = float(t.raw_pnl_pct)
        # 갭 — SL 청산인데 시가가 이미 SL 너머면 그 가격에 못 팔았다.
        if gap_fill and t.outcome == "sl":
            sl_px = float(getattr(t, "entry_sl", 0.0) or 0.0)
            ex = int(t.exit_idx)
            if sl_px > 0 and 0 <= ex < len(o):
                gapped = (o[ex] < sl_px) if long_ else (o[ex] > sl_px)
                if gapped:
                    raw = (o[ex] - float(t.entry)) / float(t.entry) * sgn
        out.append((raw - cost, int(years[int(t.exit_idx)]),
                    "long" if long_ else "short"))
    return out


def run_market(mkt: str, tickers: list[str], interval: str, **kw):
    allt = []
    for t in tickers:
        try:
            df = fetch(t, interval=interval,
                       period="730d" if interval == "1h" else "10y")
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 300:
            continue
        try:
            allt += run_symbol(df, f"{t}_{interval}", COST[mkt], **kw)
        except Exception as e:  # noqa: BLE001
            print(f"    ({t} 실패: {str(e)[:60]})", flush=True)
    return allt


def main() -> int:
    print("=== Origo(ICT FVG 되돌림) × 주식 ===", flush=True)
    print("  레버리지 1x · 시간필터 OFF(킬존 무의미) · 갭은 시가 체결", flush=True)
    print("  ⚠️ 재는 것은 'ICT 시간론' 이 아니라 FVG 되돌림 구조 자체의 엣지", flush=True)
    hdr = f"  {'':<22}{'n':>6}{'net':>11}{'건당':>7}{'승률':>5}{'RR':>6}{'MDD':>9}"

    for interval, label in (("1d", "일봉 10년"), ("1h", "1시간봉 ~3년")):
        print(f"\n### {label}", flush=True)
        print(hdr, flush=True)
        for mkt, ts in MARKETS.items():
            print(line(mkt, stat(run_market(mkt, ts, interval))), flush=True)

        print("\n  [롱/숏 분리]", flush=True)
        print(hdr, flush=True)
        for mkt, ts in MARKETS.items():
            for so in ("long", "short"):
                print(line(f"{mkt} {so}",
                           stat(run_market(mkt, ts, interval, side_only=so))),
                      flush=True)

        print("\n  [갭 효과]", flush=True)
        print(hdr, flush=True)
        for mkt, ts in MARKETS.items():
            a = stat(run_market(mkt, ts, interval, gap_fill=False))
            b = stat(run_market(mkt, ts, interval, gap_fill=True))
            print(line(f"{mkt} 갭무시", a), flush=True)
            print(line(f"{mkt} 갭반영", b), flush=True)
            if a and b:
                print(f"     → 갭이 {b['net'] - a['net']:+.1f}%p", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
