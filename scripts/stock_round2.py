"""#AUTONOMOUS 2026-08-06 (2차): ICT 정통 × 15분 / Cursus 롱 전용 × 1시간 — 주식.

파트너 지시:
  - "ICT 는 아예 정통으로" → **시간 필터(킬존) 켠다**. 1차에서는 껐었다(주식 세션과
    안 맞는다고 판단). 정통 ICT 는 원래 미국 선물·주식에서 나온 방법론이므로
    미국 주식에는 오히려 본향이다.
  - "Cursus 는 롱 전용" → 숏 전면 배제(1차에서 전 시장 큰 적자 확인).
  - "일봉으로 보면 롱이 거의 무조건 이기니까 15분/1시간 단타로" → 정확한 지적이다.
    일봉 10년에서는 Buy & Hold 가 압도했는데, 그건 전략이 나빠서라기보다 **보유
    기간이 길수록 우상향 편향이 커지기 때문**이다. 짧은 TF 에서는 그 편향이 줄어
    전략 자체의 엣지를 더 정직하게 볼 수 있다.

## 데이터 제약 (결론에 반드시 반영)
야후 인트라데이 한도 — **15분봉은 약 3개월(1,500봉)**, 1시간봉은 약 3년.
크립토 라이브 정합이 5년에 126건이었음을 감안하면 15분봉 3개월로는 **판정이 불가**하고
진입이 성립하는지만 본다. 실질 판정은 1시간봉 3년이다.

## 정통 킬존과 한국 시장
ICT 킬존은 런던·뉴욕 세션 기준이다. 한국 장(00:00~06:30 UTC)은 그 시간대에 열리지
않으므로 **정통 킬존을 그대로 적용하면 한국 종목은 진입이 원천적으로 0** 이다.
이건 전략의 실패가 아니라 적용 대상이 아니라는 뜻 — 결과를 그렇게 읽어야 한다.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from bt_par import cached_setup_timeline  # noqa: E402
from live_parity import LIVE_BASE  # noqa: E402
from stock_cursus_bt import COST, simulate, stat  # noqa: E402
from stock_fetch import MARKETS, fetch  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

RNG = np.random.default_rng(20260806)
N_PL = 300
PERIOD = {"15m": "60d", "1h": "730d", "1d": "10y"}


def origo_trades(df, sym: str, cost: float, *, canon: bool = True):
    """Origo 롱 진입 — canon=True 면 정통(킬존 ON). SL 청산은 갭(시가) 반영."""
    kw = dict(LIVE_BASE)
    kw["disable_time_filter"] = not canon
    cfg = BacktestConfig(**kw)
    tl = cached_setup_timeline(df, cfg, f"R2_{sym}_{'C' if canon else 'N'}")
    bt = run_backtest_from_timeline(df, tl, cfg)
    o = df["open"].to_numpy(float)
    out = []
    for t in bt.trades:
        if str(getattr(t.direction, "value", t.direction)).lower() != "long":
            continue
        raw = float(t.raw_pnl_pct)
        if t.outcome == "sl":
            sl = float(getattr(t, "entry_sl", 0.0) or 0.0)
            ex = int(t.exit_idx)
            if sl > 0 and 0 <= ex < len(o) and o[ex] < sl:
                raw = (o[ex] - float(t.entry)) / float(t.entry)
        out.append((raw - cost, 0, "long", int(t.exit_idx) - int(t.entry_idx)))
    return out


def placebo_total(frames, cost: float, n_total: int, holds: np.ndarray) -> np.ndarray:
    """무작위 롱 — 전략과 같은 총 거래 수·보유 기간 분포."""
    if not frames or n_total <= 0 or len(holds) == 0:
        return np.zeros(0)
    per = max(1, n_total // len(frames))
    pl = np.zeros(N_PL)
    for df in frames:
        c = df["close"].to_numpy(float)
        n = len(c)
        if n < 30:
            continue
        for j in range(N_PL):
            h = RNG.choice(holds, size=per)
            st = RNG.integers(0, n - 2, size=per)
            en = np.minimum(st + h, n - 1)
            pl[j] += float(np.sum((c[en] - c[st]) / c[st] - cost)) * 100.0
    return pl


def evaluate(tag: str, mkt: str, tickers, interval: str, engine: str):
    frames, allt = [], []
    bars = tot = 0
    for t in tickers:
        try:
            df = fetch(t, interval=interval, period=PERIOD[interval])
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 300:
            continue
        frames.append(df)
        tot += len(df)
        try:
            tr = (origo_trades(df, f"{t}_{interval}", COST[mkt]) if engine == "origo"
                  else simulate(df, COST[mkt], side_only="long"))
        except Exception as e:  # noqa: BLE001
            print(f"    ({t} 실패: {str(e)[:50]})", flush=True)
            continue
        allt += tr
        bars += sum(x[3] for x in tr)
    s = stat(allt)
    if s is None:
        print(f"  {tag:<20} 진입 {len(allt):>4}건 — 판정 불가(표본부족)", flush=True)
        return
    hs = [x[3] for x in allt if x[3] > 0]
    pl = placebo_total(frames, COST[mkt], len(allt), np.array(hs if hs else [5]))
    p = float((pl >= s["net"]).mean()) if len(pl) else float("nan")
    # Buy & Hold — 같은 구간 종목별 보유 수익 합
    bh = sum((d["close"].to_numpy(float)[-1] / d["close"].to_numpy(float)[0] - 1) * 100.0
             for d in frames) - COST[mkt] * 100.0 * len(frames)
    expo = 100.0 * bars / max(tot, 1)
    per_s = s["net"] / max(bars, 1)
    per_b = bh / max(tot, 1)
    print(f"  {tag:<20}{s['n']:>5}{s['net']:>+9.1f}%{s['wr']:>5.0f}%{expo:>6.1f}%"
          f"{bh:>+9.1f}%{per_s:>+9.4f}%{per_b:>+9.4f}%{np.median(pl):>+9.1f}%{p:>7.3f}",
          flush=True)


def main() -> int:
    print("=== 2차: ICT 정통(킬존 ON) × 15분 / Cursus 롱 × 1시간 — 주식 ===", flush=True)
    print("  레버리지 1x · 갭 시가체결 · 비용 시장별 · 롱 전용", flush=True)
    hdr = (f"  {'':<20}{'n':>5}{'net':>10}{'승률':>5}{'노출':>6}{'B&H':>10}"
           f"{'봉당전략':>10}{'봉당B&H':>9}{'무작위':>10}{'p':>7}")

    print("\n### Origo 정통(킬존 ON) × 15분봉 3개월", flush=True)
    print("  ※ 한국 장(00~06:30 UTC)은 ICT 킬존에 열리지 않는다 — 0건이 정상", flush=True)
    print(hdr, flush=True)
    for mkt, ts in MARKETS.items():
        evaluate(mkt, mkt, ts, "15m", "origo")

    print("\n### Origo 정통(킬존 ON) × 1시간봉 3년  [실질 판정]", flush=True)
    print(hdr, flush=True)
    for mkt, ts in MARKETS.items():
        evaluate(mkt, mkt, ts, "1h", "origo")

    print("\n### Origo 킬존 OFF × 1시간봉 3년  [대조 — 시간필터 기여도]", flush=True)
    print(hdr, flush=True)
    for mkt, ts in MARKETS.items():
        frames, allt = [], []
        for t in ts:
            try:
                df = fetch(t, interval="1h", period="730d")
            except Exception:  # noqa: BLE001
                continue
            if df is None or len(df) < 300:
                continue
            frames.append(df)
            try:
                allt += origo_trades(df, f"{t}_1h", COST[mkt], canon=False)
            except Exception:  # noqa: BLE001
                continue
        s = stat(allt)
        if s is None:
            print(f"  {mkt:<20}{len(allt):>5}  판정불가(표본부족)", flush=True)
        else:
            print(f"  {mkt:<20}{s['n']:>5}{s['net']:>+9.1f}%{s['wr']:>5.0f}%"
                  f"  RR {s['rr']:.2f}", flush=True)

    print("\n### Cursus 롱 전용 × 1시간봉 3년", flush=True)
    print(hdr, flush=True)
    for mkt, ts in MARKETS.items():
        evaluate(mkt, mkt, ts, "1h", "cursus")

    print("\n  판정 — p<0.05(무작위 롱 이김) + 봉당전략 > 봉당B&H 여야 통과", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
