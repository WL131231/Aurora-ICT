"""#AUTONOMOUS 2026-07-27: 펀딩비 히스토리 수집 — B트랙(펀딩 신호 연구) 데이터.

Bybit 공개 API(fetch_funding_rate_history, 인증 불요)로 고정7 페어 5년 펀딩비
(8h 주기)를 받아 parquet 캐시. 이후 펀딩 극단(롤링 분위) → Origo 방향 가중/역방향
신호 연구에 사용.
"""
from __future__ import annotations

import time

import ccxt
import pandas as pd

PAIRS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
         "DOGE/USDT:USDT", "LINK/USDT:USDT", "HYPE/USDT:USDT"]


def fetch_all(ex: ccxt.bybit, sym: str) -> pd.DataFrame:
    rows = []
    since = ex.parse8601("2021-01-01T00:00:00Z")
    while True:
        batch = ex.fetch_funding_rate_history(sym, since=since, limit=200)
        if not batch:
            break
        rows += [(b["timestamp"], b["fundingRate"]) for b in batch]
        nxt = batch[-1]["timestamp"] + 1
        if nxt <= since:
            break
        since = nxt
        if len(batch) < 200:
            break
        time.sleep(0.12)  # rate limit 예의
    df = pd.DataFrame(rows, columns=["ts_ms", "rate"]).drop_duplicates("ts_ms")
    df.index = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df[["rate"]]


def main() -> int:
    ex = ccxt.bybit({"options": {"defaultType": "swap"}})
    for sym in PAIRS:
        base = sym.split("/")[0]
        try:
            df = fetch_all(ex, sym)
            df.to_parquet(f"data/{base}USDT_funding.parquet")
            print(f"{base}: {len(df)}개 ({df.index.min()} ~ {df.index.max()})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"{base}: 실패 {e}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
