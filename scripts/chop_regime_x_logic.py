"""#AUTONOMOUS 2026-07-28: 횡보 판정 3종 × 매매로직 2종 매트릭스 (파트너 지시).

파트너: 횡보 판정을 ①ADX ②볼린저 스퀴즈 ③Choppiness Index 로 구분하고, 그 구간에서만
(a) 볼린저 매매 (b) Origo 를 TP/SL 짧게 — 두 로직을 돌려 비교.

횡보 판정(전부 직전 완결봉, 인과):
  R1 ADX14 < {18, 20, 25}
  R2 BB 스퀴즈 = BBW < 롤링90일 분위 {20%, 33%}  (밴드 수축)
  R3 CHOP14 > {55, 61.8}
  R0 전체(판정 없음) — 기준선
로직A 볼린저: 매트릭스 상위 진입규칙(E3 복귀 / E7 복귀+볼륨) × TP{1R,2R,mid} × SL{atr1,atr15}
로직B Origo-단축: 라이브게이트 거래를 그대로 쓰되 **TP/SL 을 짧게 재시뮬** —
  진입가·방향은 동일, TP = entry ± k×ATR14(k∈{0.5,1.0}), SL = entry ∓ k×ATR(대칭 1:1),
  즉 "횡보라 목표를 짧게 잡고 빨리 먹는" 변형. 원래 R 기반 TP/SL 은 base 로 비교.
TF: 볼린저 1h(매트릭스 승자 TF), Origo 는 5m 진입 그대로.
비용: maker 0.08% / taker 0.11% 양쪽.
판정: net>0 + 양반기 흑자 + 연도 다수 흑자 → ★. 승자는 이후 배터리(이웃·전이·TF).
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import btc_bb_matrix as BBM  # noqa: E402
from bt_par import _load_full, _resample, cached_setup_timeline  # noqa: E402
from chop_gate_bakeoff import BASE, adx14, chop14, roll_q  # noqa: E402

from aurora_ict.backtest.replay import BacktestConfig, run_backtest_from_timeline  # noqa: E402

SYM = "BTCUSDT"


# ---------- 공통: 국면 마스크 ----------
def regime_masks(h, lo, c, bars_day: int) -> dict[str, np.ndarray]:
    adx = adx14(h, lo, c)
    chop = chop14(h, lo, c)
    s = pd.Series(c)
    mid = s.rolling(20).mean().to_numpy()
    sd = s.rolling(20).std().to_numpy()
    bbw = (4.0 * sd) / np.maximum(mid, 1e-12)
    q20 = roll_q(bbw, bars_day * 90, 0.20)
    q33 = roll_q(bbw, bars_day * 90, 0.33)
    return {
        "R0_전체": np.ones(len(c), dtype=bool),
        "R1_ADX<18": adx < 18,
        "R1_ADX<20": adx < 20,
        "R1_ADX<25": adx < 25,
        "R2_스퀴즈q20": bbw < q20,
        "R2_스퀴즈q33": bbw < q33,
        "R3_CHOP>55": chop > 55,
        "R3_CHOP>61.8": chop > 61.8,
    }


def stat(tr) -> tuple[str, bool]:
    if len(tr) < 25:
        return f"n={len(tr):4d} (표본부족)", False
    tr = sorted(tr)
    nets = [p for _, p in tr]
    net = sum(nets)
    w = sum(1 for p in nets if p > 0)
    half = len(tr) // 2
    h1 = sum(p for _, p in tr[:half]); h2 = sum(p for _, p in tr[half:])
    eq = pk = mdd = 0.0
    for _, p in tr:
        eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    ys: dict[int, float] = {}
    for t, p in tr:
        ys[t.year] = ys.get(t.year, 0.0) + p
    ypos = sum(1 for v in ys.values() if v > 0)
    ok = net > 0 and h1 > 0 and h2 > 0 and ypos >= len(ys) - 1
    yearly = " ".join(f"{y}:{v:+.0f}" for y, v in sorted(ys.items()))
    return (f"n={len(tr):4d} net={net:+7.1f}% 승률={100 * w / len(tr):3.0f}% "
            f"H1={h1:+6.1f} H2={h2:+6.1f} MDD={mdd:5.1f} [{yearly}]"), ok


# ---------- 로직A: 볼린저 × 국면 ----------
def logic_a() -> list[tuple[str, str, bool]]:
    print("\n########## 로직A: 볼린저(1h) × 횡보판정 ##########", flush=True)
    BBM.SYM = SYM
    df, d = BBM.prep("1h")
    idx = df.index
    masks = regime_masks(d["h"], d["lo"], d["c"], 24)
    res = []
    for rule in ("E3", "E7"):
        sig_raw = BBM.signals(d, rule)
        for rname, m in masks.items():
            sig = np.where(m, sig_raw, 0.0)
            for tp in ("1R", "2R", "mid"):
                for sl in ("atr1", "atr15"):
                    for cost, cn in ((0.0008, "maker"), (0.0011, "taker")):
                        tr = BBM.run(d, idx, sig, tp, sl, "all", cost)
                        line, ok = stat(tr)
                        tag = f"{rule} {rname:<12} tp={tp:<3} sl={sl:<5} {cn}"
                        if ok:
                            print(f"★{tag:<44} {line}", flush=True)
                            res.append((tag, line, ok))
    return res


# ---------- 로직B: Origo 단축 TP/SL × 국면 ----------
def logic_b() -> list[tuple[str, str, bool]]:
    print("\n########## 로직B: Origo(5m) 단축TP/SL × 횡보판정 ##########", flush=True)
    df5 = _resample(_load_full(SYM))
    cfg = BacktestConfig(**BASE)
    tl = cached_setup_timeline(df5, cfg, SYM)
    bt = run_backtest_from_timeline(df5, tl, cfg)
    c = df5["close"].to_numpy(); h = df5["high"].to_numpy(); lo = df5["low"].to_numpy()
    n = len(c)
    tr14 = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    atr = pd.Series(tr14).rolling(14).mean().to_numpy()
    masks = regime_masks(h, lo, c, 288)
    # 라이브게이트 통과 거래만
    trs = [t for t in bt.trades if not (17 <= df5.index[t.entry_idx].hour < 21)]
    mags = [abs(t.entry_trend_pct) for t in trs]
    q70 = np.percentile(mags, 70) if mags else 0.0
    kept = []
    for t in trs:
        sgn = 1.0 if t.direction == "long" else -1.0
        if abs(t.entry_trend_pct) < q70 and t.entry_trend_pct * sgn < 0:
            continue
        kept.append(t)
    res = []
    for rname, m in masks.items():
        # base(원래 TP/SL) — 국면별 성적
        base_tr = [(df5.index[t.entry_idx], t.net_pnl_pct) for t in kept if m[t.entry_idx]]
        line, ok = stat(base_tr)
        tag = f"{rname:<12} base(원TP/SL)"
        print(f"{'★' if ok else ' '}{tag:<32} {line}", flush=True)
        if ok:
            res.append((tag, line, ok))
        # 단축 TP/SL 재시뮬
        for k in (0.5, 1.0, 1.5):
            for cost, cn in ((0.0008, "maker"), (0.0011, "taker")):
                out = []
                for t in kept:
                    i = t.entry_idx
                    if not m[i] or np.isnan(atr[i]):
                        continue
                    d = 1 if t.direction == "long" else -1
                    entry = t.entry
                    a = atr[i]
                    tp = entry + d * k * a
                    sl = entry - d * k * a
                    netv = 0.0
                    for j in range(i + 1, min(i + 1 + 288, n)):
                        if d == 1:
                            if lo[j] <= sl:
                                netv = (sl - entry) / entry; break
                            if h[j] >= tp:
                                netv = (tp - entry) / entry; break
                        else:
                            if h[j] >= sl:
                                netv = (entry - sl) / entry; break
                            if lo[j] <= tp:
                                netv = (entry - tp) / entry; break
                    out.append((df5.index[i], (netv - cost) * 100))
                line, ok = stat(out)
                tag = f"{rname:<12} 단축TP/SL {k}×ATR {cn}"
                if ok:
                    print(f"★{tag:<44} {line}", flush=True)
                    res.append((tag, line, ok))
    return res


def main() -> int:
    a = logic_a()
    b = logic_b()
    print("\n\n===== ★ 통과 요약 =====", flush=True)
    if not a and not b:
        print("  없음 — 전 조합 불합격", flush=True)
    for tag, line, _ in a + b:
        print(f"  {tag} | {line}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
