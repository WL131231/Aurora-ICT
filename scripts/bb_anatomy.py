"""#AUTONOMOUS 2026-07-28: 볼린저 원신호 해부 — 비용 0 gross 엣지 존재 여부 (파트너: 횡보 대응).

배경: 6/27 BB 평균회귀 기각은 **한 형태만**(이탈 즉시 진입·중앙선TP·ATR SL·1h) 봤다.
타사 EA 해부(BOLINGER_PROFIT_SYSTEM)에서 확인한 진짜 진입 규칙은 "이탈 후 **복귀**"
였고, 우리는 그 축·TP 종류·%B·TF 를 안 봤다. 배포 판단 전에 **원신호에 gross 엣지가
있는지**부터 비용 없이 확인한다(없으면 어떤 변형도 무의미 — 즉시 종결).

측정(진입/청산 로직 없이 사건 후 수익률 분포만):
  사건 A: 종가가 하단밴드 이탈(<lower) — 즉시 롱 관점
  사건 B: 이탈 후 **복귀**(직전 이탈 & 현재 종가 >= lower) — EA 방식
  사건 C: %B 극단(<0.05) / D: 밴드 접촉(저가<=lower<종가)
각 사건 후 +6/12/24/48봉 수익률(방향 정규화: 롱관점=+, 숏관점 반전) 평균·중앙값·승률·t값.
국면 분해: 전체 / ADX<20(횡보) / BBW 하위33%(수축). TF: 5m·15m·1h.
비용 0 — 순수 신호 검정.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from chop_gate_bakeoff import adx14, bbw20, roll_q  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
HORIZONS = (6, 12, 24, 48)
BB_N, BB_K = 20, 2.0


def bb(c: np.ndarray, n: int = BB_N, k: float = BB_K):
    s = pd.Series(c)
    ma = s.rolling(n).mean()
    sd = s.rolling(n).std()
    return (ma - k * sd).to_numpy(), ma.to_numpy(), (ma + k * sd).to_numpy()


def collect(sym: str, tf: str) -> pd.DataFrame:
    df = _resample(_load_full(sym))
    if tf not in ("5m", "5min"):
        df = df.resample(tf).agg({"open": "first", "high": "max",
                                  "low": "min", "close": "last"}).dropna()
    c = df["close"].to_numpy(); h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    n = len(c)
    lower, mid, upper = bb(c)
    width = (upper - lower) / np.maximum(mid, 1e-12)
    pctb = (c - lower) / np.maximum(upper - lower, 1e-12)
    adx = adx14(h, lo, c)
    bbw_q33 = roll_q(width, 24 * 90 if tf == "1h" else (96 * 90 if tf.startswith("15") else 288 * 90), 0.33)
    out = []
    for i in range(BB_N + 2, n - max(HORIZONS) - 1):
        if np.isnan(lower[i]) or np.isnan(adx[i]):
            continue
        # 사건 판정 (직전 완결봉 i 기준, 진입은 i+1 시가 가정 — 인과)
        ev = None
        side = 0
        if c[i] < lower[i]:
            ev, side = "A_이탈(하)", 1
        elif c[i] > upper[i]:
            ev, side = "A_이탈(상)", -1
        elif c[i - 1] < lower[i - 1] and c[i] >= lower[i]:
            ev, side = "B_복귀(하)", 1
        elif c[i - 1] > upper[i - 1] and c[i] <= upper[i]:
            ev, side = "B_복귀(상)", -1
        elif pctb[i] < 0.05:
            ev, side = "C_%B극단(하)", 1
        elif pctb[i] > 0.95:
            ev, side = "C_%B극단(상)", -1
        if ev is None:
            continue
        base = df["open"].to_numpy()[i + 1] if "open" in df else c[i]
        row = dict(sym=sym, tf=tf, ev=ev.split("_")[0], evf=ev, side=side,
                   adx=adx[i], narrow=(not np.isnan(bbw_q33[i])) and width[i] < bbw_q33[i])
        for hz in HORIZONS:
            j = i + hz
            if j >= n:
                row[f"r{hz}"] = np.nan
            else:
                row[f"r{hz}"] = (c[j] - base) / base * 100 * side
        out.append(row)
    return pd.DataFrame(out)


def summarize(df: pd.DataFrame, label: str) -> None:
    if df.empty:
        print(f"  {label:<22} n=0", flush=True)
        return
    parts = []
    for hz in HORIZONS:
        col = df[f"r{hz}"].dropna()
        if len(col) < 20:
            parts.append(f"h{hz}: n<20")
            continue
        t = col.mean() / (col.std() / np.sqrt(len(col)) + 1e-12)
        parts.append(f"h{hz}:{col.mean():+.3f}%(승{100 * (col > 0).mean():.0f}% t{t:+.1f})")
    print(f"  {label:<22} n={len(df):5d} " + "  ".join(parts), flush=True)


def main() -> int:
    tfs = sys.argv[1:] or ["15min", "1h"]
    for tf in tfs:
        print(f"\n===== TF {tf} =====", flush=True)
        allp = []
        for sym in PAIRS:
            try:
                allp.append(collect(sym, tf))
            except Exception as e:  # noqa: BLE001
                print(f"  {sym} 실패: {e}", flush=True)
        if not allp:
            continue
        d = pd.concat(allp, ignore_index=True)
        for ev in ("A", "B", "C"):
            sub = d[d.ev == ev]
            print(f" [{ev}]", flush=True)
            summarize(sub, "전체")
            summarize(sub[sub.adx < 20], "ADX<20(횡보)")
            summarize(sub[sub.narrow], "BBW수축(하위33%)")
            summarize(sub[(sub.adx < 20) & sub.narrow], "횡보+수축")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
