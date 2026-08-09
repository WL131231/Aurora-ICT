"""#AUTONOMOUS 2026-07-29: 분출(횡보 이탈) 탐지 — 파트너 구상 핵심.

파트너 구상: "횡보 상태면 매매 STOP → 분출 탐지에 올인 → 분출 방향으로 진입".
전제(확보됨): 횡보 **인식** 정밀도 63.7%(멀티TF AND, 1.93x). 예측은 불가.

[1단계] 분출 신호 자체의 gross 검정 — 진입/청산 로직 없이, 신호 후 방향성 수익률.
  전제 상태: CSI(1h·12봉) >= thr 인 "횡보 중"에서만 신호 탐색.
  분출 후보(전부 직전 완결봉 — 인과):
    B1 박스이탈    : 최근 24봉 고/저 돌파 (Donchian 이탈)
    B2 밴드이탈    : 종가가 BB(20,2) 밖 (스퀴즈 후 확장)
    B3 볼륨급증    : 볼륨 > 20봉평균×2 & 방향봉(종가-시가 부호)
    B4 ATR급증     : 현재봉 범위 > ATR14×2 (변동성 점화)
    B5 밴드확장    : BBW 가 5봉 전 대비 +30% 이상 (스퀴즈 해제)
    B6 복합(B1&B3): 박스이탈 + 볼륨 확인
    B7 복합(B5&B1): 밴드확장 + 박스이탈
  측정: 신호 후 +6/12/24/48봉(5m) 방향 정규화 수익률 평균·승률·t값.
        비교군 = 같은 횡보 상태의 무신호 시점(기저).
[2단계] 매매화 — gross 통과 신호만 TP/SL 붙여 실거래 시뮬(비용 0.08/0.11%).
TF: 5m(체결·신호) + 1h(상태). 페어 7종 5년.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from chop_state_index import csi_series, fit_csi, load_1h  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
HZ = (6, 12, 24, 48)
CSI_THR = 0.5


def build(sym: str, model) -> pd.DataFrame:
    df = _resample(_load_full(sym))
    o = df["open"].to_numpy(); c = df["close"].to_numpy()
    h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    v = df["volume"].to_numpy() if "volume" in df else np.ones(len(c))
    n = len(c)
    s = pd.Series(c)
    ma = s.rolling(20).mean().to_numpy(); sd = s.rolling(20).std().to_numpy()
    up = ma + 2 * sd; dn = ma - 2 * sd
    bbw = (4 * sd) / np.maximum(ma, 1e-12)
    bbw_chg = pd.Series(bbw).pct_change(5).to_numpy()
    don_hi = pd.Series(h).rolling(24).max().shift(1).to_numpy()
    don_lo = pd.Series(lo).rolling(24).min().shift(1).to_numpy()
    volma = pd.Series(v).rolling(20).mean().to_numpy()
    tr = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    rng = h - lo
    csi = csi_series(load_1h(sym), model).reindex(df.index, method="ffill").to_numpy()
    # 신호 판정 (봉 i 종가 확정 → 진입은 i+1 시가)
    sig = {}
    dirn = np.sign(c - o)
    sig["B1 박스이탈"] = np.where(c > don_hi, 1, np.where(c < don_lo, -1, 0))
    sig["B2 밴드이탈"] = np.where(c > up, 1, np.where(c < dn, -1, 0))
    big_v = v > volma * 2
    sig["B3 볼륨급증"] = np.where(big_v & (dirn > 0), 1, np.where(big_v & (dirn < 0), -1, 0))
    big_r = rng > atr * 2
    sig["B4 ATR급증"] = np.where(big_r & (dirn > 0), 1, np.where(big_r & (dirn < 0), -1, 0))
    exp_ = bbw_chg > 0.30
    sig["B5 밴드확장"] = np.where(exp_ & (dirn > 0), 1, np.where(exp_ & (dirn < 0), -1, 0))
    sig["B6 박스+볼륨"] = np.where((sig["B1 박스이탈"] != 0) & big_v, sig["B1 박스이탈"], 0)
    sig["B7 확장+박스"] = np.where((sig["B1 박스이탈"] != 0) & exp_, sig["B1 박스이탈"], 0)
    rows = []
    base_px = np.concatenate([o[1:], [np.nan]])   # i+1 시가
    for name, arr in sig.items():
        idx = np.flatnonzero((arr != 0) & (csi >= CSI_THR))
        for i in idx:
            if i + max(HZ) >= n or np.isnan(base_px[i]):
                continue
            d = arr[i]
            r = dict(sym=sym, sig=name, ts=df.index[i], d=d)
            for hz in HZ:
                r[f"r{hz}"] = (c[i + hz] - base_px[i]) / base_px[i] * 100 * d
            rows.append(r)
    # 기저: 횡보 상태의 무신호 시점 표본(랜덤 방향)
    none_idx = np.flatnonzero((csi >= CSI_THR) & (sig["B1 박스이탈"] == 0)
                              & (sig["B4 ATR급증"] == 0))
    rng_ = np.random.default_rng(0)
    pick = rng_.choice(none_idx[none_idx + max(HZ) < n], size=min(3000, len(none_idx)),
                       replace=False) if len(none_idx) > 10 else []
    for i in pick:
        d = 1 if rng_.random() > 0.5 else -1
        if np.isnan(base_px[i]):
            continue
        r = dict(sym=sym, sig="기저(무신호)", ts=df.index[i], d=d)
        for hz in HZ:
            r[f"r{hz}"] = (c[i + hz] - base_px[i]) / base_px[i] * 100 * d
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> int:
    model = fit_csi(PAIRS)
    ds = []
    for sym in PAIRS:
        try:
            ds.append(build(sym, model))
        except Exception as e:  # noqa: BLE001
            print(f"{sym} 실패: {e}", flush=True)
    d = pd.concat(ds, ignore_index=True)
    print(f"총 신호 {len(d):,} (CSI>={CSI_THR} 횡보 상태 한정)\n", flush=True)
    print(f"{'신호':<14} {'n':>7} " + "  ".join(f"{'h' + str(z):>22}" for z in HZ), flush=True)
    for name in ["기저(무신호)", "B1 박스이탈", "B2 밴드이탈", "B3 볼륨급증",
                 "B4 ATR급증", "B5 밴드확장", "B6 박스+볼륨", "B7 확장+박스"]:
        sub = d[d.sig == name]
        if sub.empty:
            continue
        parts = []
        for hz in HZ:
            col = sub[f"r{hz}"].dropna()
            if len(col) < 30:
                parts.append(f"{'n<30':>22}")
                continue
            t = col.mean() / (col.std() / np.sqrt(len(col)) + 1e-12)
            parts.append(f"{col.mean():+.3f}%(승{100 * (col > 0).mean():.0f}% t{t:+.1f})".rjust(22))
        print(f"{name:<14} {len(sub):7,} " + "  ".join(parts), flush=True)
    print("\n→ 기저 대비 평균·t 가 유의하게 큰 신호만 2단계(매매화) 진행", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
