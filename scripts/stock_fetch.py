"""#AUTONOMOUS 2026-08-06: 주식 1h OHLCV 수집 — 나스닥/코스피/코스닥 대표주.

파트너 요청: "주식에서도 통할지 궁금한데".

yfinance 1h 는 최대 730일(2년)까지만 준다. 크립토 백테(5년)보다 짧으므로 결론의
신뢰도는 그만큼 낮다 — 보고에 반드시 명시한다.

종목은 각 시장 대표 대형주로 잡았다(시총 순위는 수시로 바뀌므로 "상위 10 정확 재현"이
아니라 **대표성 있는 대형주 10**이다). 데이터가 안 받아지는 종목은 자동 제외한다.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

CACHE = Path("data/stocks")

NASDAQ = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "AVGO", "TSLA", "COST", "NFLX"]
KOSPI = ["005930.KS", "000660.KS", "373220.KS", "207940.KS", "005380.KS",
         "000270.KS", "068270.KS", "105560.KS", "005490.KS", "035420.KS"]
KOSDAQ = ["247540.KQ", "086520.KQ", "028300.KQ", "196170.KQ", "068760.KQ",
          "214150.KQ", "145020.KQ", "328130.KQ", "277810.KQ", "091990.KQ"]
MARKETS = {"NASDAQ": NASDAQ, "KOSPI": KOSPI, "KOSDAQ": KOSDAQ}


def fetch(ticker: str, *, force: bool = False, interval: str = "1h",
          period: str = "730d") -> pd.DataFrame | None:
    """OHLCV — 캐시 우선. 컬럼 open/high/low/close/volume, UTC 인덱스.

    1h 는 야후가 최근 730일까지만 준다(10년 불가). 장기는 1d 로 받는다.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{ticker.replace('.', '_')}_{interval}.parquet"
    if p.exists() and not force:
        return pd.read_parquet(p)
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval,
                     progress=False, auto_adjust=False)
    if df is None or len(df) == 0:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                            "Close": "close", "Volume": "volume"})
    df = df[["open", "high", "low", "close", "volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df.to_parquet(p)
    return df


def main() -> int:
    force = "--force" in sys.argv
    total = 0
    for mkt, tickers in MARKETS.items():
        print(f"\n[{mkt}]", flush=True)
        for t in tickers:
            try:
                df = fetch(t, force=force)
            except Exception as e:  # noqa: BLE001
                print(f"  {t:<12} 실패 — {e}", flush=True)
                continue
            if df is None or len(df) < 200:
                print(f"  {t:<12} 데이터 부족 ({0 if df is None else len(df)}봉) — 제외",
                      flush=True)
                continue
            total += 1
            print(f"  {t:<12} {len(df):>5}봉  {df.index[0].date()} ~ {df.index[-1].date()}",
                  flush=True)
    print(f"\n사용 가능 종목 {total}개", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
