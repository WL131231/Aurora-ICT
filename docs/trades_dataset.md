# Aurora-ICT 거래 학습/복기 Dataset

봇이 매 청산 시점에 자동 생성하는 per-trade JSON sidecar 파일 형식.
나중에 학습(ML 분류기/RL), 복기(equity curve 재구성, drawdown 분석), 룰 튜닝
등에 활용 가능.

## 저장 위치
```
<data_dir>/trades_dataset/<exit_iso>__<direction>__<classification>.json
```

- `<data_dir>` — 봇이 사용하는 데이터 디렉토리
  - Windows: `%APPDATA%/Aurora-ICT/` (또는 PyInstaller frozen 시 .exe 옆)
  - macOS: `~/Library/Application Support/Aurora-ICT/`
  - dev: 현재 작업 디렉토리 또는 `aurora_ict.paths.data_dir()` 가 결정
- `<exit_iso>` — 청산 시각 UTC ISO basic (예: `20260528T023250Z`)
- `<direction>` — `long` / `short`
- `<classification>` — `SL_HIT` / `TP_HIT` / `SYNC_CLOSE` / `FLIP_OPEN` / `MANUAL_CLOSE` 등 (특수문자 `_` 로 sanitize)

예시 파일명: `20260528T023250Z__short__TP_HIT.json`

## 필드 명세 (`version: "v1"`)

```jsonc
{
  "version": "v1",
  "bot_version": "0.4.96",           // 그 시점 봇 버전
  "symbol": "BTC/USDT:USDT",
  "direction": "short",
  "classification": "TP_HIT",        // exit.classification 과 동일

  "entry": {
    "ts_ms": 1748397600000,
    "iso_utc": "2026-05-28T02:00:00+00:00",
    "iso_kst": "2026-05-28T11:00:00+09:00",
    "iso_ny":  "2026-05-27T22:00:00-04:00",
    "entry_px": 74495.0,
    "sl": 74668.61,
    "tp": 74104.7,
    "qty": 0.931,
    "setup_ts_ms": 1748397300000,    // setup 검출 시각 (체결 직전 봉)
    "rr_target": 2.25,                // |TP-entry| / |SL-entry|
    "context": {                      // _build_entry_context_json 의 unpack
      "source": "silver_bullet" | "turtle" | "rejection" | "mitigation" | "flip" | ...,
      "window": "ny_pm_sb" | "am_macro_1" | ...,
      "killzone": "ny_pm",
      "confluence_score": 5,
      "confluences": ["OB", "sweep", "macro", "HTF_FVG_2H"],
      "ltf_weight": 2,
      "htf_flip_target": { "tf": "2H", "type": "bearish", "weight": 6 } | null,
      // ... 봇이 진입 시 채운 모든 컨텍스트
    },
    "equity_at_entry_usdt": 3892.54
  },

  "exit": {
    "ts_ms": 1748401956000,
    "iso_utc": "2026-05-28T02:32:36+00:00",
    "iso_kst": "2026-05-28T11:32:36+09:00",
    "iso_ny":  "2026-05-27T22:32:36-04:00",
    "close_px": 74160.0,
    "classification": "TP_HIT",
    "pnl_usdt": 280.15,               // 거래소 closed-pnl API 의 net (fees/funding 반영)
    "pnl_pct_on_equity_entry": 7.20,  // pnl / equity_at_entry × 100
    "duration_sec": 4356,
    "equity_at_exit_usdt": 4172.69,
    "from_closed_pnl_api": true       // false 면 placeholder (조회 실패)
  },

  "ohlcv_snapshot": {
    "1h_entry50_to_exit5": [
      [ts_ms, open, high, low, close, volume],
      ...  // entry 50봉 전 ~ exit + 5봉 (1H 기준 약 55~60 캔들)
    ],
    "5m_entry50_to_exit5": [          // 5m 기준 같은 window
      [ts_ms, open, high, low, close, volume],
      ...
    ]
  },

  "settings_snapshot": {
    "min_rr": 2.0,
    "min_confluence": 3,
    "disable_time_filter": true,
    "leverage": 20,
    "position_pct_base": 40.0,
    "position_pct_max": 90.0,
    "position_pct_step": 15.0,
    "fvg_min_size_pct": 0.0005,
    "trail_buffer_ratio": 0.001,
    "timeframe": "5m"
  }
}
```

## 활용 예시

### 1) 분류기 학습 (TP vs SL 예측)
```python
import json
from pathlib import Path
import pandas as pd

dataset_dir = Path("~/AppData/Roaming/Aurora-ICT/trades_dataset").expanduser()
rows = []
for fp in dataset_dir.glob("*.json"):
    d = json.loads(fp.read_text(encoding="utf-8"))
    rows.append({
        "ts": d["entry"]["ts_ms"],
        "direction": d["direction"],
        "classification": d["classification"],
        "rr_target": d["entry"]["rr_target"],
        "confluence_score": d["entry"]["context"].get("confluence_score"),
        "source": d["entry"]["context"].get("source"),
        "killzone": d["entry"]["context"].get("killzone"),
        "duration_sec": d["exit"]["duration_sec"],
        "pnl_usdt": d["exit"]["pnl_usdt"],
        "won": d["classification"] == "TP_HIT",
    })

df = pd.DataFrame(rows)
print(df.groupby("source")["won"].agg(["mean", "count"]))
print(df.groupby("killzone")["pnl_usdt"].sum())
```

### 2) Equity Curve 재구성
```python
df = df.sort_values("ts")
df["cum_pnl"] = df["pnl_usdt"].cumsum()
df["equity"] = 3892.54 + df["cum_pnl"]
df.plot(x="ts", y="equity")
```

### 3) MFE/MAE 사후 계산 (OHLCV snapshot 활용)
```python
# entry 후 첫 N봉의 high/low 로 max favorable / adverse excursion 산출
candles_1h = d["ohlcv_snapshot"]["1h_entry50_to_exit5"]
entry_ts = d["entry"]["ts_ms"]
post_entry = [c for c in candles_1h if c[0] >= entry_ts]
if d["direction"] == "short":
    mfe = d["entry"]["entry_px"] - min(c[3] for c in post_entry)  # 가격 최저
    mae = max(c[2] for c in post_entry) - d["entry"]["entry_px"]  # 가격 최고
else:
    mfe = max(c[2] for c in post_entry) - d["entry"]["entry_px"]
    mae = d["entry"]["entry_px"] - min(c[3] for c in post_entry)
```

### 4) 룰 튜닝 — min_rr / min_confluence 영향 분석
```python
# 어떤 confluence_score 에서 승률 가장 높은가?
df.groupby("confluence_score")["won"].agg(["mean", "count"])
```

## 데이터 보존
- 봇은 절대 dataset 파일을 삭제/덮어쓰지 않음 (파일명 unique — exit_ts UTC 기준)
- 누적될수록 학습 가능한 sample 수 증가
- 보조 백업 권장: trades.db (SQLite) 도 같이 보존 (raw event list)

## 버전 관리
- 본 명세는 `version: "v1"` — 향후 필드 추가 시 `v2` 등으로 bump
- 기존 v1 파일은 그대로 보존 (마이그레이션 X — 시간순 학습 데이터)

담당: 장수 / 2026-05-28 추가 (PR — feat/trade-dataset-export)
