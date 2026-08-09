"""#AUTONOMOUS 2026-07-28: BTC 볼린저 다방면 매트릭스 (파트너: 여러 로직 섞어 전수).

원신호 해부(bb_anatomy)는 7페어 합산이었고 "비용 미만" 판정. BTC 만 떼서 매매 형태로
전수 조합한다 — 파트너 지시: 밴드 닿을 때 / 이탈 후 복귀 / 섞기 등.

진입 규칙(롱 기준, 숏 미러 — 전부 직전 완결봉 신호 → 다음 봉 진입, 인과):
  E1 touch      : 저가<=하단 (밴드 터치, 종가 무관)
  E2 breach     : 종가<하단 (이탈 — 6/27 형태)
  E3 reentry    : 직전 이탈 & 종가>=하단 (복귀 — EA 형태)
  E4 reentry_sto: 복귀 + 스토캐스틱(5,3,3) 과매도(<20) 확인
  E5 touch_sto  : 터치 + 스토 과매도
  E6 double     : 2봉 내 2회 이탈 후 복귀 (이중 바닥형)
  E7 reentry_vol: 복귀 + 이탈봉 볼륨 > 20봉평균 (관심 확인)
청산 TP: 중앙선(mid) / 반대밴드(opp) / 1R / 2R      SL: 밴드 밖 ATR×{1.0,1.5} / 고정 0.5%
국면: 전체 / ADX<20 / BBW 수축 하위33% / 횡보+수축
TF: 5m, 15min, 1h.  비용: 왕복 0.11%(시장가) 및 0.08%(maker 지정가) 양쪽.
판정: 5년 net>0 + 양반기 흑자 + 연도 다수 흑자.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from chop_gate_bakeoff import adx14, roll_q  # noqa: E402

SYM = "BTCUSDT"
BB_N, BB_K = 20, 2.0
HOLD_MAX = 96          # 최대 보유 봉


def prep(tf: str):
    df = _resample(_load_full(SYM))
    if tf not in ("5m", "5min"):
        df = df.resample(tf).agg({"open": "first", "high": "max",
                                  "low": "min", "close": "last",
                                  "volume": "sum"}).dropna()
    c = df["close"].to_numpy(); h = df["high"].to_numpy()
    lo = df["low"].to_numpy(); o = df["open"].to_numpy()
    v = df["volume"].to_numpy() if "volume" in df else np.ones(len(c))
    s = pd.Series(c)
    mid = s.rolling(BB_N).mean().to_numpy()
    sd = s.rolling(BB_N).std().to_numpy()
    up = mid + BB_K * sd; dn = mid - BB_K * sd
    width = (up - dn) / np.maximum(mid, 1e-12)
    bars_day = {"5m": 288, "5min": 288, "15min": 96, "1h": 24}[tf]
    narrow_thr = roll_q(width, bars_day * 90, 0.33)
    adx = adx14(h, lo, c)
    # 스토캐스틱(5,3,3)
    lowmin = pd.Series(lo).rolling(5).min().to_numpy()
    highmax = pd.Series(h).rolling(5).max().to_numpy()
    k_raw = 100 * (c - lowmin) / np.maximum(highmax - lowmin, 1e-12)
    k = pd.Series(k_raw).rolling(3).mean().to_numpy()
    # ATR
    tr = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    atr = pd.Series(tr).rolling(14).mean().to_numpy()
    volma = pd.Series(v).rolling(20).mean().to_numpy()
    return df, dict(c=c, h=h, lo=lo, o=o, v=v, mid=mid, up=up, dn=dn, width=width,
                    narrow_thr=narrow_thr, adx=adx, k=k, atr=atr, volma=volma)


def signals(d: dict, rule: str) -> np.ndarray:
    """봉별 신호 (+1 롱, -1 숏, 0 없음) — 봉 i 종가 확정 기준."""
    c, h, lo, up, dn, k, v, volma = (d[x] for x in
                                     ("c", "h", "lo", "up", "dn", "k", "v", "volma"))
    n = len(c)
    sig = np.zeros(n)
    br_dn = c < dn      # 하단 이탈
    br_up = c > up
    for i in range(BB_N + 3, n):
        if np.isnan(dn[i]) or np.isnan(k[i]):
            continue
        s = 0
        if rule == "E1":       # 터치
            if lo[i] <= dn[i]:
                s = 1
            elif h[i] >= up[i]:
                s = -1
        elif rule == "E2":     # 이탈
            if br_dn[i]:
                s = 1
            elif br_up[i]:
                s = -1
        elif rule == "E3":     # 복귀
            if br_dn[i - 1] and not br_dn[i]:
                s = 1
            elif br_up[i - 1] and not br_up[i]:
                s = -1
        elif rule == "E4":     # 복귀 + 스토
            if br_dn[i - 1] and not br_dn[i] and k[i] < 20:
                s = 1
            elif br_up[i - 1] and not br_up[i] and k[i] > 80:
                s = -1
        elif rule == "E5":     # 터치 + 스토
            if lo[i] <= dn[i] and k[i] < 20:
                s = 1
            elif h[i] >= up[i] and k[i] > 80:
                s = -1
        elif rule == "E6":     # 이중 이탈 후 복귀
            if (br_dn[i - 1] or br_dn[i - 2]) and not br_dn[i] and br_dn[i - 3]:
                s = 1
            elif (br_up[i - 1] or br_up[i - 2]) and not br_up[i] and br_up[i - 3]:
                s = -1
        elif rule == "E7":     # 복귀 + 볼륨
            if br_dn[i - 1] and not br_dn[i] and v[i - 1] > volma[i - 1]:
                s = 1
            elif br_up[i - 1] and not br_up[i] and v[i - 1] > volma[i - 1]:
                s = -1
        sig[i] = s
    return sig


def run(d: dict, idx, sig: np.ndarray, tp_kind: str, sl_kind: str,
        regime: str, cost: float) -> list[tuple[pd.Timestamp, float]]:
    c, h, lo, o, mid, up, dn = (d[x] for x in ("c", "h", "lo", "o", "mid", "up", "dn"))
    adx, width, nthr, atr = (d[x] for x in ("adx", "width", "narrow_thr", "atr"))
    n = len(c)
    trades = []
    i = BB_N + 4
    while i < n - 1:
        s = sig[i]
        if s == 0:
            i += 1
            continue
        # 국면 필터 (신호봉 기준)
        if regime == "chop" and not (adx[i] < 20):
            i += 1; continue
        if regime == "narrow" and not ((not np.isnan(nthr[i])) and width[i] < nthr[i]):
            i += 1; continue
        if regime == "both" and not (adx[i] < 20 and (not np.isnan(nthr[i])) and width[i] < nthr[i]):
            i += 1; continue
        j0 = i + 1
        entry = o[j0]
        a = atr[i] if not np.isnan(atr[i]) else entry * 0.005
        if s == 1:
            sl = {"atr1": dn[i] - a, "atr15": dn[i] - 1.5 * a,
                  "pct": entry * 0.995}[sl_kind]
            risk = entry - sl
            if risk <= 0:
                i += 1; continue
            tp = {"mid": mid[i], "opp": up[i], "1R": entry + risk, "2R": entry + 2 * risk}[tp_kind]
            if tp <= entry:
                i += 1; continue
        else:
            sl = {"atr1": up[i] + a, "atr15": up[i] + 1.5 * a,
                  "pct": entry * 1.005}[sl_kind]
            risk = sl - entry
            if risk <= 0:
                i += 1; continue
            tp = {"mid": mid[i], "opp": dn[i], "1R": entry - risk, "2R": entry - 2 * risk}[tp_kind]
            if tp >= entry:
                i += 1; continue
        netv = 0.0
        exit_j = min(j0 + HOLD_MAX, n - 1)
        for j in range(j0, exit_j + 1):
            if s == 1:
                if lo[j] <= sl:
                    netv = (sl - entry) / entry; break
                if h[j] >= tp:
                    netv = (tp - entry) / entry; break
            else:
                if h[j] >= sl:
                    netv = (entry - sl) / entry; break
                if lo[j] <= tp:
                    netv = (entry - tp) / entry; break
        else:
            netv = ((c[exit_j] - entry) / entry) * s
        trades.append((idx[j0], (netv - cost) * 100))
        i = j0 + 1
    return trades


def stat(tr) -> tuple[str, bool]:
    if len(tr) < 30:
        return f"n={len(tr):4d} (표본부족)", False
    tr = sorted(tr)
    nets = [p for _, p in tr]
    net = sum(nets)
    w = sum(1 for p in nets if p > 0)
    half = len(tr) // 2
    h1 = sum(p for _, p in tr[:half]); h2 = sum(p for _, p in tr[half:])
    ys: dict[int, float] = {}
    for t, p in tr:
        ys[t.year] = ys.get(t.year, 0.0) + p
    ypos = sum(1 for v in ys.values() if v > 0)
    ok = net > 0 and h1 > 0 and h2 > 0 and ypos >= len(ys) - 1
    yearly = " ".join(f"{y}:{v:+.0f}" for y, v in sorted(ys.items()))
    return (f"n={len(tr):4d} net={net:+8.1f}% 승률={100 * w / len(tr):3.0f}% "
            f"H1={h1:+7.1f} H2={h2:+7.1f} [{yearly}]"), ok


def main() -> int:
    tfs = sys.argv[1:] or ["5m", "15min", "1h"]
    rules = ["E1", "E2", "E3", "E4", "E5", "E6", "E7"]
    tps = ["mid", "opp", "1R", "2R"]
    sls = ["atr1", "atr15", "pct"]
    regimes = ["all", "chop", "narrow", "both"]
    winners = []
    for tf in tfs:
        df, d = prep(tf)
        idx = df.index
        print(f"\n########## TF {tf} ##########", flush=True)
        for rule in rules:
            sig = signals(d, rule)
            for tp in tps:
                for sl in sls:
                    for rg in regimes:
                        for cost, cname in ((0.0011, "taker"), (0.0008, "maker")):
                            tr = run(d, idx, sig, tp, sl, rg, cost)
                            line, ok = stat(tr)
                            tag = f"{rule} tp={tp:<3} sl={sl:<5} {rg:<6} {cname}"
                            if ok:
                                winners.append((tf, tag, line))
                                print(f"★{tag:<38} {line}", flush=True)
    print("\n\n===== ★ 통과 조합 =====", flush=True)
    if not winners:
        print("  없음 — BTC 볼린저 전 조합 불합격", flush=True)
    for tf, tag, line in winners:
        print(f"  [{tf}] {tag} {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
