"""#AUTONOMOUS 2026-08-06: Origo 완화판 × 주식 롱 — 구조를 낮추면 뭐라도 나오나.

⚠️ **이것은 Origo 가 아니다.** 라이브 설정(min_confluence 5)에서는 주식 진입이
**0 건**이다(AAPL 1h 5,073봉 / 일봉 10년 / 삼성전자 전부 0). confluence 게이트가
전부 막는다 — Origo 의 점수 체계는 24시간 시장의 구조(킬존 스윕·세션 인수인계·
HTF 정렬)를 전제로 하는데 주식엔 그 구조가 없다.

여기서는 진입 조건을 크게 풀어(confluence 1 · min_rr 0.5 · TTL 30) **FVG 되돌림이라는
뼈대만 남겼을 때** 뭐라도 나오는지 본다. 나와도 그건 새 전략 설계의 출발점이지
"Origo 가 주식에서 통한다"는 뜻이 아니다.

대조는 Cursus 때와 같다 — 무작위 롱(같은 거래 수·보유 기간)을 이기는가.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
from bt_par import cached_setup_timeline  # noqa: E402
from live_parity import LIVE_BASE  # noqa: E402
from stock_cursus_bt import COST, stat  # noqa: E402
from stock_fetch import MARKETS, fetch  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

RNG = np.random.default_rng(20260806)
N_PL = 300
OVER = {"min_confluence": 1, "min_rr": 0.5,
        "entry_ttl_bars": 30, "setup_stale_bars": 20}


def run_symbol(df, sym: str, cost: float):
    """롱 진입만 수집 — SL 청산은 갭(시가) 반영."""
    kw = dict(LIVE_BASE)
    kw["disable_time_filter"] = True
    kw.update(OVER)
    cfg = BacktestConfig(**kw)
    tl = cached_setup_timeline(df, cfg, f"RLX_{sym}")
    bt = run_backtest_from_timeline(df, tl, cfg)
    o = df["open"].to_numpy(float)
    years = df.index.year.to_numpy()
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
        out.append((raw - cost, int(years[int(t.exit_idx)]), "long",
                    int(t.exit_idx) - int(t.entry_idx)))
    return out


def main() -> int:
    print("=== Origo 완화판(conf 1 · rr 0.5 · ttl30) × 주식 롱 · 일봉 10년 ===",
          flush=True)
    print("  ⚠️ 라이브 설정(conf 5)에서는 진입 **0 건** — 이건 뼈대만 남긴 탐색이다.",
          flush=True)
    print(f"  {'시장':<10}{'n':>6}{'net':>10}{'건당':>8}{'승률':>6}"
          f"{'노출%':>7}{'무작위롱':>11}{'p':>7}", flush=True)

    for mkt, tickers in MARKETS.items():
        allt: list[tuple] = []
        bars = tot = 0
        frames = []
        for t in tickers:
            try:
                df = fetch(t, interval="1d", period="10y")
            except Exception:  # noqa: BLE001
                continue
            if df is None or len(df) < 300:
                continue
            frames.append(df)
            tr = run_symbol(df, f"{t}_1d", COST[mkt])
            allt += tr
            bars += sum(x[3] for x in tr)
            tot += len(df)
        s = stat(allt)
        if s is None:
            print(f"  {mkt:<10} 표본부족 ({len(allt)}건)", flush=True)
            continue
        hs = [x[3] for x in allt if x[3] > 0]
        holds = np.array(hs if hs else [5], dtype=int)
        per_sym = max(1, len(allt) // max(len(frames), 1))
        pl = np.zeros(N_PL)
        for df in frames:
            c = df["close"].to_numpy(float)
            n = len(c)
            for j in range(N_PL):
                h = RNG.choice(holds, size=per_sym)
                st = RNG.integers(0, n - 2, size=per_sym)
                en = np.minimum(st + h, n - 1)
                pl[j] += float(np.sum((c[en] - c[st]) / c[st] - COST[mkt])) * 100.0
        p = float((pl >= s["net"]).mean())
        print(f"  {mkt:<10}{s['n']:>6}{s['net']:>+9.1f}%{s['per']:>+7.2f}%"
              f"{s['wr']:>5.0f}%{100 * bars / max(tot, 1):>6.1f}%"
              f"{np.median(pl):>+10.1f}%{p:>7.3f}", flush=True)
    print("  판정 — p<0.05 여야 무작위 롱보다 낫다고 말할 수 있다.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
