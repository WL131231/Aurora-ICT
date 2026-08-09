"""실제 라이브 entry_trend 역산 — CSV 진입 시점 OHLCV로 |진입추세%| 복원 → 페어별 q33.

파트너(6/23): 실 거래 데이터로 횡보 임계 학습. CSV entry 체결시각(ts_ms) 직전 20봉
변화율(_set_entry_trend 라이브 로직 동일)을 OHLCV 로 역산 → 실제 라이브 |entry_trend|
분포 → q33. 백테 하드코딩 q33 과 비교해 실데이터 기반 보정값 산출(롤링 seed 도 됨).
"""
from __future__ import annotations

import sys

import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402

P = r"C:\Users\지영민\Downloads\trades_all_users (1).csv"
SYMS = ["BTCUSDT", "ETHUSDT", "DOGEUSDT", "HYPEUSDT", "XRPUSDT", "SOLUSDT", "LINKUSDT"]
HARD = {"BTCUSDT": 0.230, "ETHUSDT": 0.268, "SOLUSDT": 0.396, "XRPUSDT": 0.271,
        "DOGEUSDT": 0.275, "LINKUSDT": 0.315, "HYPEUSDT": 0.527}

df = pd.read_csv(P)
ent = df[(df["event_type"] == "entry") & (df["mode"] == "live")].copy()

lines = ["===== 실제 라이브 entry_trend 역산 (페어별 q33) =====",
         f"{'페어':<10} {'표본':>5} {'실제q33':>8} {'하드코딩':>8} {'차이':>8}"]
seed = {}
for sym in SYMS:
    try:
        o = _resample(_load_full(sym))
    except Exception as e:
        lines.append(f"{sym:<10} 로드실패 {e}")
        continue
    closes = o["close"].values
    idx = o.index
    csv_sym = sym[:-4] + "/USDT:USDT"  # BTCUSDT -> BTC/USDT:USDT
    sub = ent[ent["symbol"] == csv_sym]
    trends = []
    for ts in sub["ts_ms"]:
        t = pd.to_datetime(int(ts), unit="ms", utc=True)
        pos = idx.searchsorted(t) - 1  # 진입 직전(마지막 닫힌) 봉
        if pos < 20 or pos >= len(closes):
            continue
        past = closes[pos - 20]
        if past <= 0:
            continue
        trends.append(abs((closes[pos] - past) / past * 100.0))
    if len(trends) >= 5:
        trends.sort()
        q33 = trends[len(trends) // 3]
        seed[sym] = trends
        hard = HARD[sym]
        lines.append(f"{sym:<10} {len(trends):5d} {q33:8.3f} {hard:8.3f} {q33 - hard:+8.3f}")
    else:
        lines.append(f"{sym:<10} {len(trends):5d}  (표본부족 — 하드코딩 유지)")

lines.append("\n※ 실제q33 > 하드코딩이면 라이브 변동성이 백테보다 커 게이트가 더 엄격해야 함(반대도).")
lines.append("  이 분포를 롤링 seed 로 봇 _trend_history 에 미리 주입하면 배포 직후부터 실데이터 기반.")

txt = "\n".join(lines)
with open("reverse_entry_trend_result.txt", "w", encoding="utf-8") as f:
    f.write(txt + "\n")
# seed 데이터 저장 (롤링 주입용)
import json
with open("regime_seed.json", "w", encoding="utf-8") as f:
    json.dump({k: [round(x, 4) for x in v] for k, v in seed.items()}, f)
try:
    print(txt + "\nDONE")
except UnicodeEncodeError:
    print("(결과는 reverse_entry_trend_result.txt)\nDONE")
