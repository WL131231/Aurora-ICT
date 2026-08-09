"""#AUTONOMOUS 2026-08-01: Cursus + CSI 횡보 진입 게이트 — 현행 라이브 정합 검증.

파트너 지시: "cursus 에 횡보장 진입 차단 게이트 넣어야되거든? 그건 우리 csi 로 가자".

⚠️ 7/30 의 `cursus_csi_gate.py` 결과는 **무효**다 — 그 하니스는 트레일 변형본
(trail_mult 3.0)이었고, 라이브는 원본 4분할 TP 엔진이다. 게다가 그 뒤로 라이브가
크게 바뀌었다(지정가 진입 · 하이켄아시 · 10x/10% · TRX↔LINK). 하니스 정합 규칙에
따라 **현행 라이브와 같은 조건**에서 다시 측정한다.

CSI 는 프로덕션 모듈(`aurora_ict.indicators.chop_state`)의 **하드코딩 계수를 그대로**
쓴다. 재학습하면 그 자체가 룩어헤드이고, 배포될 코드와 다른 것을 재는 셈이 된다.

게이트: CSI(직전 완결봉) >= thr 이면 신규 진입 skip. 보유 포지션은 그대로 관리.
판정: net 개선 + 연도 일관 + 롱/숏 동시 + 부트스트랩. 임계값 절벽이면 과최적으로 기각.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
sys.path.insert(0, "../Aurora-ICT/src")
import cursus_limit_battery as B  # noqa: E402
import dst_trend_bt_clamped as DST  # noqa: E402
from cursus_dev_changes import NEW_PAIRS, heikin_ashi  # noqa: E402

from aurora_ict.indicators.chop_state import (  # noqa: E402
    _B as CSI_B,
)
from aurora_ict.indicators.chop_state import (
    _MU as CSI_MU,
)
from aurora_ict.indicators.chop_state import (
    _SG as CSI_SG,
)
from aurora_ict.indicators.chop_state import (
    _W as CSI_W,
)
from aurora_ict.indicators.chop_state import (
    adx14,
    chop14,
)

RNG = np.random.default_rng(20260801)


def csi_series(df: pd.DataFrame) -> np.ndarray:
    """CSI 시계열 — `chop_state.compute_csi` 의 재료·계수를 그대로 벡터화.

    compute_csi 는 마지막 봉 하나만 반환하므로 백테용으로 전 구간을 계산한다.
    재료 정의가 어긋나면 검증이 무의미하므로 **같은 식**을 쓴다(4h ADX 는 1h 로
    대체 — 라이브도 df4h 미주입 시 같은 경로를 탄다).

    Returns:
        길이 len(df) 의 횡보 확률(계산 불가 구간은 NaN).
    """
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    v = df["volume"].to_numpy(float) if "volume" in df else np.ones(len(c))
    s = pd.Series(c)
    ma = s.rolling(20).mean().to_numpy()
    sd = s.rolling(20).std().to_numpy()
    bbw = (4.0 * sd) / np.maximum(ma, 1e-12)
    bbw_slope = pd.Series(bbw).pct_change(5).to_numpy()
    volr = v / np.maximum(pd.Series(v).rolling(20).mean().to_numpy(), 1e-12)
    adx = adx14(h, lo, c)
    chop = chop14(h, lo, c)
    idx = df.index
    x = np.column_stack([
        adx, chop, bbw, bbw_slope, volr,
        idx.hour.to_numpy(float), idx.dayofweek.to_numpy(float), adx,
    ])
    z = (x - CSI_MU) / CSI_SG
    out = 1.0 / (1.0 + np.exp(-np.clip(z @ CSI_W + CSI_B, -30.0, 30.0)))
    out[~np.isfinite(x).all(axis=1)] = np.nan
    return out


def simulate(sym: str, *, thr: float | None, entry: str = "limit",
             ttl: int = 3, side_only: str | None = None):
    """B.simulate 와 동일 엔진 + CSI 진입 게이트.

    Args:
        thr: 이 값 이상이면 진입 차단. None 이면 게이트 off(기준선).
    """
    df = DST._load_1h(sym)
    sig = DST._signals(heikin_ashi(df))
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c_arr = df["close"].to_numpy(float)
    o_arr = df["open"].to_numpy(float)
    years = df.index.year.to_numpy()
    buy = np.concatenate([[False], sig["buy_sig"].to_numpy()[:-1]])
    sell = np.concatenate([[False], sig["sell_sig"].to_numpy()[:-1]])
    # 직전 완결봉 CSI — 라이브도 진행 중 봉을 넣지 않는다(미래 정보 차단).
    csi = np.concatenate([[np.nan], csi_series(df)[:-1]])
    n = len(c_arr)

    trades: list[tuple[float, int, str]] = []
    n_sig = n_fill = n_block = 0
    side = None
    e_px = stop = 0.0
    tps: list[float] = []
    hits = 0
    remain = 1.0
    acc = 0.0
    fee_in = 0.0
    i = 1
    while i < n:
        if side is not None:
            sgn = 1.0 if side == "long" else -1.0
            rev = bool(sell[i]) if side == "long" else bool(buy[i])
            if (lo[i] <= stop) if side == "long" else (h[i] >= stop):
                raw = (stop - e_px) / e_px * sgn
                acc += (raw - fee_in - B.FEE_TAKER) * B.SIZE_PCT * remain * B.LEVERAGE
                trades.append((acc, int(years[i]), side))
                side = None; acc = 0.0; remain = 1.0; hits = 0
            else:
                done = False
                while hits < len(tps):
                    tp = tps[hits]
                    if not ((h[i] >= tp) if side == "long" else (lo[i] <= tp)):
                        break
                    hits += 1
                    raw = (tp - e_px) / e_px * sgn
                    frac = remain if hits >= len(tps) else B.TP_FRAC
                    acc += (raw - fee_in - B.FEE_MAKER) * B.SIZE_PCT * frac * B.LEVERAGE
                    remain = 0.0 if hits >= len(tps) else max(remain - B.TP_FRAC, 0.0)
                    if hits >= len(tps):
                        trades.append((acc, int(years[i]), side))
                        side = None; acc = 0.0; remain = 1.0; hits = 0
                        done = True
                        break
                    if hits >= 2:
                        lad = tps[hits - 2]
                        if (side == "long" and lad > stop) or (side == "short" and lad < stop):
                            stop = lad
                if not done and side is not None and rev:
                    raw = (o_arr[i] - e_px) / e_px * sgn
                    acc += (raw - fee_in - B.FEE_TAKER) * B.SIZE_PCT * remain * B.LEVERAGE
                    trades.append((acc, int(years[i]), side))
                    side = None; acc = 0.0; remain = 1.0; hits = 0

        if side is None and (buy[i] or sell[i]):
            d = 1 if buy[i] else -1
            want = "long" if d == 1 else "short"
            if side_only is not None and want != side_only:
                i += 1
                continue
            n_sig += 1
            # ── CSI 횡보 게이트 ── (NaN = 계산 불가 → 게이트 off, 라이브와 동일)
            if thr is not None and np.isfinite(csi[i]) and csi[i] >= thr:
                n_block += 1
                i += 1
                continue
            sb = i - 1
            if entry == "market":
                px, fill_i = c_arr[sb], i
            else:
                px = lo[sb] if d == 1 else h[sb]
                if not np.isfinite(px) or px <= 0:
                    i += 1
                    continue
                fill_i = None
                for j in range(i, min(i + ttl + 1, n)):
                    if (d == 1 and lo[j] <= px) or (d == -1 and h[j] >= px):
                        fill_i = j
                        break
                if fill_i is None:
                    i += 1
                    continue
            n_fill += 1
            fee_in = B.FEE_MAKER if entry == "limit" else B.FEE_TAKER
            e_px = px
            side = want
            sgn = float(d)
            stop = e_px * (1 - sgn * B.SL_PCT)
            tps = [e_px * (1 + sgn * p) for p in B.TP_PCTS]
            hits = 0; remain = 1.0; acc = 0.0
            i = fill_i + 1
            continue
        i += 1
    return trades, n_sig, n_block


def run(pairs, **kw):
    allt: list[tuple[float, int, str]] = []
    s = b = 0
    for sym in pairs:
        t, ns, nb = simulate(sym, **kw)
        allt += t
        s += ns
        b += nb
    return allt, 100 * b / max(s, 1)


def main() -> int:
    print("=== Cursus + CSI 횡보 진입 게이트 (현행 라이브 정합) ===", flush=True)
    print("  엔진: 지정가 라벨 TTL3 · HA 신호 · 10x/10% · 4분할TP · 원본 SL 2%",
          flush=True)
    print("  CSI: 프로덕션 계수 그대로(chop_state.py) · 직전 완결봉", flush=True)

    base, _ = run(NEW_PAIRS, thr=None)
    B.head("① 임계값 스캔 (차단률과 net)")
    print(B.line("게이트 없음(기준선)", B.stat(base)), flush=True)
    best = None
    for thr in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75):
        t, blk = run(NEW_PAIRS, thr=thr)
        s = B.stat(t)
        if s is None:
            continue
        print(B.line(f"CSI >= {thr:.2f} 차단", s, blk), flush=True)
        if best is None or s["net"] > best[1]["net"]:
            best = (thr, s, t)
    if best is None:
        print("  표본부족 — 판정 불가", flush=True)
        return 1
    thr, s_best, t_best = best
    print(f"\n  → net 최대: CSI >= {thr:.2f}", flush=True)

    B.head("② 연도 일관성 (기준선 vs 최적 임계값)")
    for y in sorted({x[1] for x in base}):
        sb_ = B.stat([x for x in base if x[1] == y])
        sg_ = B.stat([x for x in t_best if x[1] == y])
        a = f"{sb_['net']:+.1f}%" if sb_ else "n/a"
        g = f"{sg_['net']:+.1f}%" if sg_ else "n/a"
        mark = ""
        if sb_ and sg_:
            mark = "  개선" if sg_["net"] > sb_["net"] else "  ←악화"
        print(f"  {y}   게이트없음 {a:>10}   CSI게이트 {g:>10}{mark}", flush=True)

    B.head("③ 롱/숏 분리")
    for so in ("long", "short"):
        t0, _ = run(NEW_PAIRS, thr=None, side_only=so)
        t1, blk = run(NEW_PAIRS, thr=thr, side_only=so)
        print(B.line(f"{so} 게이트없음", B.stat(t0)), flush=True)
        print(B.line(f"{so} CSI게이트", B.stat(t1), blk), flush=True)

    B.head("④ 페어별 (기준선 → 게이트)")
    for sym in NEW_PAIRS:
        t0, _ = run([sym], thr=None)
        t1, _ = run([sym], thr=thr)
        s0, s1 = B.stat(t0), B.stat(t1)
        if s0 is None or s1 is None:
            print(f"  {sym:<26} 표본부족", flush=True)
            continue
        mark = "개선" if s1["net"] > s0["net"] else "←악화"
        print(f"  {sym:<26}{s0['net']:>+10.1f}% → {s1['net']:>+10.1f}%   {mark}",
              flush=True)

    print("\n=== ⑤ 부트스트랩 (복원추출 5000회) ===", flush=True)
    for tag, tr in (("게이트없음", base), (f"CSI>={thr:.2f}", t_best)):
        v = np.array([x[0] for x in tr], float)
        boot = 100 * RNG.choice(v, size=(5000, len(v)), replace=True).sum(axis=1)
        p5, p50, p95 = np.percentile(boot, [5, 50, 95])
        print(f"  {tag:<12} 중앙 {p50:>+9.1f}%  5% {p5:>+9.1f}%  95% {p95:>+9.1f}%"
              f"   흑자확률 {100 * (boot > 0).mean():>5.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
