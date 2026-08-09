"""#AUTONOMOUS 2026-08-06 (3차): confluence 문턱만 낮추면? — 파트너 제안.

파트너: "5점을 채우는 게 아니라, 하나만 나와도 진입 — 이건 어때".

앞선 완화판은 confluence 말고도 min_rr·TTL·킬존을 **동시에** 풀어서, 무엇이 기여했는지
분리되지 않았다. 여기서는 **confluence 문턱만** 1~5 로 스윕하고 나머지는 정통 그대로
둔다(킬존 ON · min_rr 2.0 · TTL 6 · stale 3).

ICT 관점에서도 근거가 있는 변형이다 — 단일 PD array(FVG 하나)만 보고 들어가는 방식은
실제로 쓰인다. 점수 합산은 Aurora 가 추가한 필터층이지 ICT 정통 요구가 아니다.

대상은 나스닥. 한국 장은 ICT 킬존 시간에 열리지 않아 문턱과 무관하게 0 건이다.
판정은 동일 — 무작위 롱(같은 거래 수·보유기간)을 이기고, 봉당으로 B&H 를 이겨야 한다.
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
from stock_fetch import NASDAQ, fetch  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

RNG = np.random.default_rng(20260806)
N_PL = 300


def trades_for(df, sym: str, conf: int, cost: float):
    """정통 설정에서 min_confluence 만 바꿔 롱 진입 수집."""
    kw = dict(LIVE_BASE)
    kw["disable_time_filter"] = False      # 킬존 ON — 정통
    kw["min_confluence"] = conf
    cfg = BacktestConfig(**kw)
    tl = cached_setup_timeline(df, cfg, f"CF_{sym}")   # conf 무관 — 캐시 공유
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


def main() -> int:
    cost = COST["NASDAQ"]
    print("=== confluence 문턱만 스윕 (킬존 ON · rr 2.0 · TTL 6 유지) — 나스닥 롱 ===",
          flush=True)
    print("  ※ 한국 장은 ICT 킬존 시간에 열리지 않아 문턱과 무관하게 0건", flush=True)

    for interval, period, label in (("1h", "730d", "1시간봉 3년"),
                                    ("15m", "60d", "15분봉 3개월")):
        frames = []
        for t in NASDAQ:
            try:
                df = fetch(t, interval=interval, period=period)
            except Exception:  # noqa: BLE001
                continue
            if df is not None and len(df) >= 300:
                frames.append((t, df))
        if not frames:
            continue
        tot = sum(len(d) for _, d in frames)
        bh = sum((d["close"].to_numpy(float)[-1] / d["close"].to_numpy(float)[0] - 1)
                 * 100.0 for _, d in frames) - cost * 100.0 * len(frames)
        print(f"\n### {label}  (종목 {len(frames)} · B&H {bh:+.1f}% · "
              f"봉당 B&H {bh / max(tot, 1):+.4f}%)", flush=True)
        print(f"  {'conf':<6}{'n':>5}{'net':>10}{'승률':>6}{'RR':>6}{'노출':>7}"
              f"{'봉당전략':>10}{'무작위':>10}{'p':>7}", flush=True)

        for conf in (1, 2, 3, 4, 5):
            allt, bars = [], 0
            for t, df in frames:
                try:
                    tr = trades_for(df, f"{t}_{interval}", conf, cost)
                except Exception as e:  # noqa: BLE001
                    print(f"    ({t} 실패 {str(e)[:40]})", flush=True)
                    continue
                allt += tr
                bars += sum(x[3] for x in tr)
            s = stat(allt)
            if s is None:
                print(f"  {conf:<6}{len(allt):>5}  판정불가(표본부족)", flush=True)
                continue
            hs = [x[3] for x in allt if x[3] > 0]
            holds = np.array(hs if hs else [5], dtype=int)
            per = max(1, len(allt) // len(frames))
            pl = np.zeros(N_PL)
            for _, df in frames:
                c = df["close"].to_numpy(float)
                n = len(c)
                for j in range(N_PL):
                    h = RNG.choice(holds, size=per)
                    st = RNG.integers(0, n - 2, size=per)
                    en = np.minimum(st + h, n - 1)
                    pl[j] += float(np.sum((c[en] - c[st]) / c[st] - cost)) * 100.0
            p = float((pl >= s["net"]).mean())
            print(f"  {conf:<6}{s['n']:>5}{s['net']:>+9.1f}%{s['wr']:>5.0f}%"
                  f"{s['rr']:>6.2f}{100 * bars / max(tot, 1):>6.1f}%"
                  f"{s['net'] / max(bars, 1):>+9.4f}%{np.median(pl):>+9.1f}%{p:>7.3f}",
                  flush=True)
    print("\n  판정 — p<0.05 + 봉당전략 > 봉당B&H 여야 통과", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
