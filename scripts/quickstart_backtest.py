"""외부 개발자용 최소 실행 예제 — 5분이면 끝나는 백테 1회.

이 파일 하나만 돌려보면 하니스가 어떻게 생겼는지 감이 온다.
전체 5년 × 전 페어는 몇 시간짜리라 여기서는 **최근 N봉만** 자른다.

    cd Aurora-ICT-research
    PYTHONPATH=src python scripts/quickstart_backtest.py
    PYTHONPATH=src python scripts/quickstart_backtest.py --symbol ETHUSDT --bars 40000

핵심 규칙 하나만 기억하면 된다 — **설정을 직접 만들지 말고 `live_cfg()` 를 쓸 것.**
라이브 봇이 쓰는 42개 설정이 거기 모여 있고, 하나라도 빠지면 백테가 라이브와
다른 봇이 된다. 실제로 그렇게 두 달을 흘려보낸 적이 있다(2026-08-08 발견,
라이브 진입의 93%에 붙는 가점이 백테에 아예 없었다).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from live_parity import GAPS, LIVE_BASE, live_cfg  # noqa: E402

from aurora_ict.backtest.replay import run_backtest_from_timeline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Aurora-ICT 백테 최소 예제")
    ap.add_argument("--symbol", default="BTCUSDT", help="data/{SYMBOL}_1m_full.parquet 필요")
    ap.add_argument("--bars", type=int, default=20000, help="5분봉 개수 (20000 ≈ 10주)")
    a = ap.parse_args()

    print(f"=== Aurora-ICT 백테 최소 예제 — {a.symbol} 최근 {a.bars}봉(5m)")

    # ① 데이터 — 1분봉 parquet 을 5분봉으로 리샘플. 없으면 fetch_ohlcv.py 로 받는다.
    t0 = time.time()
    df = _resample(_load_full(a.symbol)).tail(a.bars)
    print(f"  ① 데이터   {len(df)}봉  {df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}"
          f"  ({time.time() - t0:.1f}초)")

    # ② 설정 — 반드시 live_cfg(). 직접 BacktestConfig(...) 를 만들면 정합이 깨진다.
    cfg = live_cfg(a.symbol)
    print(f"  ② 설정     live_cfg() — {len(LIVE_BASE)}개 라이브 설정 적용")
    print(f"             문턱={cfg.min_confluence} · 최소RR={cfg.min_rr} · "
          f"레버리지={getattr(cfg, 'leverage', '?')} · stale={cfg.setup_stale_bars}봉")

    # ③ 셋업 타임라인 — 무거운 계산은 여기 한 번뿐이고 디스크에 캐시된다.
    #    같은 (심볼, 설정) 조합은 두 번째부터 즉시 로드된다.
    t0 = time.time()
    tl = cached_setup_timeline(df, cfg, f"QS_{a.symbol}_{a.bars}")
    print(f"  ③ 타임라인 캐시 완료 ({time.time() - t0:.1f}초) — 재실행은 즉시")

    # ④ 체결 시뮬 — 타임라인 위에서 게이트/진입/청산만 돌린다. 설정 스윕이 싼 이유.
    t0 = time.time()
    bt = run_backtest_from_timeline(df, tl, cfg)
    print(f"  ④ 체결     {len(bt.trades)}건 ({time.time() - t0:.1f}초)")

    if not bt.trades:
        print("\n  거래 0건 — 구간이 짧거나 그 기간에 조건이 안 나왔다. --bars 를 늘려보라.")
        return 0

    # ⑤ 결과 — R 배수(위험 1단위당 손익)로 본다. % 수익률은 레버리지·사이징에
    #    좌우돼 비교가 안 되지만 R 은 설정이 달라도 비교가 된다.
    rs = []
    for t in bt.trades:
        risk = abs(float(t.entry) - float(getattr(t, "entry_sl", 0.0) or 0.0))
        if risk > 0:
            rs.append(float(t.raw_pnl_pct) * float(t.entry) / risk)
    if rs:
        wins = [r for r in rs if r > 0]
        months = (df.index[-1] - df.index[0]).days / 30.4
        print(f"\n  건당 {sum(rs) / len(rs):+.3f}R · 승률 {100 * len(wins) / len(rs):.0f}%"
              f" · 빈도 {len(rs) / max(months, 1e-9):.2f}건/월")

    print(f"\n  ⚠️ 아직 라이브와 다른 부분이 {len(GAPS)}개 남아 있다 (live_parity.GAPS):")
    for k in list(GAPS)[:3]:
        print(f"      · {k} — {GAPS[k][:60]}")
    print("     이 축들을 건드리는 결론은 지금 낼 수 없다.")

    print("\n  다음 단계 — 이 숫자만으로 '엣지 발견'이라고 하면 안 된다.")
    print("  순열검정 · 플라시보 대조 · 홀드아웃까지 통과해야 한다(문서 참조).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
