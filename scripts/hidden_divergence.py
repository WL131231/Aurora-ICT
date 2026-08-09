"""#AUTONOMOUS 2026-07-29: RSI **히든 다이버전스** = 추세 지속 신호 검증 (파트너 지시 ①).

Ditomasso 자료([[ref_elliott_fib_ditomasso]]): 다이버전스 3종
  · 일반 : 가격 HH + RSI LH (약세 반전) / 가격 LL + RSI HL (강세 반전)
  · 히든 : 가격 HL + RSI LL (**강세 지속**) / 가격 LH + RSI HH (**약세 지속**)
우리 봇은 RSI 다이버전스를 쓰지만 **일반(반전)만** 본다. 히든은 방향이 반대(지속)라
아직 안 쓰는 재료 = 진짜 갭.

[1단계] gross 검정 — 매매 없이 신호 후 방향 수익률.
  일반/히든 × 강세/약세 4종 + 기저(무작위). h6/12/24/48.
  히든이 참이면 **히든 강세 후 상승 / 히든 약세 후 하락** 이 기저보다 유의해야.
검증(7/29 확립 5종 전부 적용):
  · 플라시보(무작위 시점)  · 국면×방향 매트릭스  · TF 재현(15min·1h)
  · 같은 봉 배제(gross 라 해당 없음 — 진입은 신호 다음봉 시가)  · 롱/숏 분리
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from fib_regime_mtf import btc_regime  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
HZ = (6, 12, 24, 48)
SW = 5          # 스윙 판정 좌우 봉수
LOOKBACK = 60   # 직전 스윙과 비교할 최대 거리


def rsi(c, n=14):
    d = np.diff(c, prepend=c[0])
    up = pd.Series(np.where(d > 0, d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    dn = pd.Series(np.where(d < 0, -d, 0.0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / np.maximum(dn, 1e-12)
    return (100 - 100 / (1 + rs)).to_numpy()


def swings(h, lo, k=SW):
    """확정 스윙(좌우 k봉) — 인과 위해 신호는 i+k 에서만 사용 가능."""
    n = len(h)
    hi_idx, lo_idx = [], []
    for i in range(k, n - k):
        if h[i] == max(h[i - k:i + k + 1]):
            hi_idx.append(i)
        if lo[i] == min(lo[i - k:i + k + 1]):
            lo_idx.append(i)
    return hi_idx, lo_idx


def find_divs(c, h, lo, r):
    """반환: {(종류, 방향): [확정봉 idx]} — 확정봉 = 스윙 확정 시점(i+SW)."""
    hi_idx, lo_idx = swings(h, lo)
    out = {("일반", 1): [], ("일반", -1): [], ("히든", 1): [], ("히든", -1): []}
    # 약세 계열 — 고점끼리 비교
    for a, b in zip(hi_idx, hi_idx[1:]):
        if b - a > LOOKBACK:
            continue
        conf = b + SW
        if h[b] > h[a] and r[b] < r[a]:
            out[("일반", -1)].append(conf)      # 가격 HH + RSI LH = 약세 반전
        if h[b] < h[a] and r[b] > r[a]:
            out[("히든", -1)].append(conf)      # 가격 LH + RSI HH = 약세 지속
    # 강세 계열 — 저점끼리 비교
    for a, b in zip(lo_idx, lo_idx[1:]):
        if b - a > LOOKBACK:
            continue
        conf = b + SW
        if lo[b] < lo[a] and r[b] > r[a]:
            out[("일반", 1)].append(conf)       # 가격 LL + RSI HL = 강세 반전
        if lo[b] > lo[a] and r[b] < r[a]:
            out[("히든", 1)].append(conf)       # 가격 HL + RSI LL = 강세 지속
    return out


def build(tf: str):
    rows = []
    reg_map = btc_regime("1h")
    for sym in PAIRS:
        df = _resample(_load_full(sym)).resample(tf).agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        o = df["open"].to_numpy(); c = df["close"].to_numpy()
        h = df["high"].to_numpy(); lo = df["low"].to_numpy()
        n = len(c)
        r = rsi(c)
        reg = reg_map.reindex(df.index, method="ffill").to_numpy()
        divs = find_divs(c, h, lo, r)
        for (kind, d), idxs in divs.items():
            for i in idxs:
                if i + 1 >= n or i + max(HZ) >= n:
                    continue
                base = o[i + 1]                 # 확정 다음봉 시가(인과)
                if base <= 0:
                    continue
                row = dict(sym=sym, kind=kind, d=d, ts=df.index[i],
                           reg=reg[i] if isinstance(reg[i], str) else "횡보")
                for hz in HZ:
                    row[f"r{hz}"] = (c[i + hz] - base) / base * 100 * d
                rows.append(row)
        # 기저 — 무작위 시점, 방향 무작위
        rng = np.random.default_rng(abs(hash(sym)) % 2**31)
        pick = rng.choice(np.arange(60, n - max(HZ) - 2), size=min(2500, max(n - 200, 10)),
                          replace=False)
        for i in pick:
            base = o[i + 1]
            if base <= 0:
                continue
            dd = 1 if rng.random() > 0.5 else -1
            row = dict(sym=sym, kind="기저", d=dd, ts=df.index[i],
                       reg=reg[i] if isinstance(reg[i], str) else "횡보")
            for hz in HZ:
                row[f"r{hz}"] = (c[i + hz] - base) / base * 100 * dd
            rows.append(row)
    return pd.DataFrame(rows)


def report(D, tf):
    print(f"\n\n============ TF {tf} ============", flush=True)
    print(f"{'신호':<16} {'n':>7} " + "  ".join(f"{'h' + str(z):>23}" for z in HZ), flush=True)
    for kind, d, nm in (("기저", None, "기저(무작위)"),
                        ("일반", 1, "일반·강세(반전)"), ("일반", -1, "일반·약세(반전)"),
                        ("히든", 1, "히든·강세(지속)"), ("히든", -1, "히든·약세(지속)")):
        sub = D[D.kind == kind] if d is None else D[(D.kind == kind) & (D.d == d)]
        if len(sub) < 50:
            continue
        parts = []
        for hz in HZ:
            col = sub[f"r{hz}"].dropna()
            t = col.mean() / (col.std() / np.sqrt(len(col)) + 1e-12)
            parts.append(f"{col.mean():+.3f}%(승{100 * (col > 0).mean():.0f}% t{t:+.1f})".rjust(23))
        print(f"{nm:<16} {len(sub):7,} " + "  ".join(parts), flush=True)

    # 국면 × 방향 — **같은 국면·같은 방향의 기저**와 나란히. (기저는 방향 무작위라
    # 롱/숏을 갈라야 "상승장 히든강세" 를 "상승장 아무 롱" 과 비교할 수 있다.)
    print("\n[국면 × 방향] h24 — 신호 vs 같은 조건 기저", flush=True)
    print(f"  {'국면':<6} {'방향':<6} {'n':>6} {'신호평균':>10} {'기저n':>6} {'기저평균':>10} "
          f"{'차이':>9} {'t(차이)':>8}", flush=True)
    for rg in ("상승", "횡보", "하락"):
        for d, dn in ((1, "강세"), (-1, "약세")):
            s = D[(D.kind == "히든") & (D.d == d) & (D.reg == rg)]["r24"].dropna()
            b = D[(D.kind == "기저") & (D.d == d) & (D.reg == rg)]["r24"].dropna()
            if len(s) < 40 or len(b) < 40:
                continue
            diff = s.mean() - b.mean()
            se = np.sqrt(s.var() / len(s) + b.var() / len(b)) + 1e-12
            print(f"  {rg:<6} {dn:<6} {len(s):6d} {s.mean():+9.3f}% {len(b):6d} "
                  f"{b.mean():+9.3f}% {diff:+8.3f}% {diff / se:+7.1f}", flush=True)


def main() -> int:
    for tf in ("1h", "15min"):
        report(build(tf), tf)
    print("\n→ 히든이 기저 대비 유의(+)하고 **두 TF 모두** 같은 부호면 2단계 매매화", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
