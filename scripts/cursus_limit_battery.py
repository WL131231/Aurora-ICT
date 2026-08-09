"""#AUTONOMOUS 2026-07-31: Cursus 지정가 진입(라벨 좌표) 검증 배터리.

개발자 확인으로 매수 지점이 **신호봉 저점(롱)/고점(숏)** 으로 확정됐다. 배포는
선행했고(파트너 지시 "라이브 적용 먼저"), 이 스크립트는 그 결정이 실제로 옳은지
사후 측정한다. 기각 가능한 형태로 돌린다 — 좋게 나오길 바라고 짜지 않는다.

배터리(7/30 확립 표준 + 지정가 고유 축):
  ① 시장가 기준선 대비 — 같은 신호·같은 비용모델에서 진입 방식만 교체
  ② 체결 규칙 민감도 — touch(≤) vs strict(<) vs deep(px 아래로 1tick 더)
     ⚠️ 지정가 백테 최대 함정. "저점을 정확히 찍고 반등" 은 실제로는 큐 뒤에
        서서 체결 안 되는 경우가 많다. strict/deep 에서 무너지면 신기루다.
  ③ 연도 일관성 — 특정 연도 몰빵이면 기각
  ④ 페어별
  ⑤ 롱/숏 분리 — 한쪽만 흑자면 방향 편향
  ⑥ TTL 민감도 — 3봉이 절벽 위 봉우리면 과최적
  ⑦ 부트스트랩 — 거래 복원추출 5000회, net 5% 분위가 0 위인지

비용은 진입/청산을 분리해 실제 수수료 구조를 반영한다(이것이 지정가의 존재 이유):
    진입  limit=maker 0.02% / market=taker 0.055%
    청산  TP=지정가 maker 0.02% / SL·REVERSE=시장가 taker 0.055%
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts")
import dst_trend_bt_clamped as DST  # noqa: E402
from cursus_dev_changes import NEW_PAIRS, heikin_ashi  # noqa: E402

FEE_MAKER = 0.0002    # Bybit VIP0 지정가
FEE_TAKER = 0.00055   # Bybit VIP0 시장가
LEVERAGE, SIZE_PCT = 10.0, 0.1   # 배포된 개발자 설정
SL_PCT = 0.02
TP_PCTS = (0.01, 0.02, 0.03, 0.04)
TP_FRAC = 0.25
RNG = np.random.default_rng(20260731)


def simulate(sym: str, *, entry: str, fill: str = "touch", ttl: int = 3,
             side_only: str | None = None):
    """한 심볼 시뮬 — (net%, year, side) 거래 리스트.

    Args:
        sym: 심볼.
        entry: "limit"(라벨 좌표 지정가) / "market"(신호봉 종가 시장가 = 현행).
        fill: 지정가 체결 규칙 — "touch"(lo<=px) / "strict"(lo<px) / "deep"(0.05% 초과).
        ttl: 미체결 대기 봉 수.
        side_only: "long"/"short" 면 그 방향만.

    Returns:
        (거래 리스트, 신호 수, 체결 수).
    """
    df = DST._load_1h(sym)
    sig = DST._signals(heikin_ashi(df))          # 신호는 HA(배포 설정)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c_arr = df["close"].to_numpy(float)
    o_arr = df["open"].to_numpy(float)
    years = df.index.year.to_numpy()
    # buy_sig 는 신호봉 자신에 서지만 봇은 봉 마감 후에야 인지한다 → 1봉 지연.
    buy = np.concatenate([[False], sig["buy_sig"].to_numpy()[:-1]])
    sell = np.concatenate([[False], sig["sell_sig"].to_numpy()[:-1]])
    n = len(c_arr)

    trades: list[tuple[float, int, str]] = []
    n_sig = n_fill = 0
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
                acc += (raw - fee_in - FEE_TAKER) * SIZE_PCT * remain * LEVERAGE
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
                    frac = remain if hits >= len(tps) else TP_FRAC
                    acc += (raw - fee_in - FEE_MAKER) * SIZE_PCT * frac * LEVERAGE
                    remain = 0.0 if hits >= len(tps) else max(remain - TP_FRAC, 0.0)
                    if hits >= len(tps):
                        trades.append((acc, int(years[i]), side))
                        side = None; acc = 0.0; remain = 1.0; hits = 0
                        done = True
                        break
                    if hits >= 2:                      # 래더 트레일
                        lad = tps[hits - 2]
                        if (side == "long" and lad > stop) or (side == "short" and lad < stop):
                            stop = lad
                if not done and side is not None and rev:
                    raw = (o_arr[i] - e_px) / e_px * sgn
                    acc += (raw - fee_in - FEE_TAKER) * SIZE_PCT * remain * LEVERAGE
                    trades.append((acc, int(years[i]), side))
                    side = None; acc = 0.0; remain = 1.0; hits = 0

        if side is None and (buy[i] or sell[i]):
            d = 1 if buy[i] else -1
            want = "long" if d == 1 else "short"
            if side_only is not None and want != side_only:
                i += 1
                continue
            n_sig += 1
            sb = i - 1                     # 신호봉(라벨이 그려지는 봉)
            if entry == "market":
                px, fill_i = c_arr[sb], i  # 현행 — 인지 직후 종가 추격
            else:
                px = lo[sb] if d == 1 else h[sb]
                if not np.isfinite(px) or px <= 0:
                    i += 1
                    continue
                edge = {"touch": 0.0, "strict": -1e-9, "deep": 0.0005}[fill]
                trig = px * (1 - d * edge)   # deep 은 px 를 더 뚫어야 체결로 인정
                fill_i = None
                for j in range(i, min(i + ttl + 1, n)):
                    if (d == 1 and lo[j] <= trig) or (d == -1 and h[j] >= trig):
                        fill_i = j
                        break
                if fill_i is None:
                    i += 1
                    continue
            n_fill += 1
            fee_in = FEE_MAKER if entry == "limit" else FEE_TAKER
            e_px = px
            side = want
            sgn = float(d)
            stop = e_px * (1 - sgn * SL_PCT)
            tps = [e_px * (1 + sgn * p) for p in TP_PCTS]
            hits = 0; remain = 1.0; acc = 0.0
            i = fill_i + 1                 # 같은 봉 청산 배제
            continue
        i += 1
    return trades, n_sig, n_fill


def run(pairs, **kw):
    """여러 페어 합산 — (거래, 체결률%)."""
    allt: list[tuple[float, int, str]] = []
    s = f = 0
    for sym in pairs:
        t, ns, nf = simulate(sym, **kw)
        allt += t
        s += ns
        f += nf
    return allt, 100 * f / max(s, 1)


def stat(trades) -> dict | None:
    if len(trades) < 10:
        return None
    v = np.array([t[0] for t in trades], float)
    win = v[v > 0]
    loss = v[v <= 0]
    dd = np.maximum.accumulate(np.cumsum(v)) - np.cumsum(v)
    return {
        "n": len(v), "net": 100 * v.sum(), "wr": 100 * len(win) / len(v),
        "rr": (win.mean() / abs(loss.mean())) if len(win) and len(loss) else float("nan"),
        "mdd": 100 * dd.max(),
    }


def line(tag: str, s: dict | None, fr: float | None = None) -> str:
    if s is None:
        return f"  {tag:<26} 표본부족"
    be = (100 - s["wr"]) / max(s["wr"], 1e-9)
    mark = "★" if s["rr"] > be and s["net"] > 0 else " "
    frs = f"{fr:>5.0f}%" if fr is not None else "     "
    return (f"  {tag:<26}{s['n']:>6}{frs}{s['net']:>+10.1f}%{s['wr']:>5.0f}%"
            f"{s['rr']:>6.2f}{be:>6.2f}{s['mdd']:>9.1f}% {mark}")


def head(t: str) -> None:
    print(f"\n=== {t} ===", flush=True)
    print(f"  {'':<26}{'n':>6}{'체결':>6}{'net':>11}{'승률':>5}{'RR':>6}{'BE':>6}{'MDD':>10}",
          flush=True)


def main() -> int:
    print("=== Cursus 지정가(라벨 좌표) 검증 배터리 ===", flush=True)
    print(f"  비용: 진입 maker {FEE_MAKER:.2%}/taker {FEE_TAKER:.3%} · "
          f"청산 TP maker / SL·REV taker", flush=True)
    print(f"  설정: {LEVERAGE:.0f}x · size {SIZE_PCT:.0%} · HA 신호 · 페어 {len(NEW_PAIRS)}종",
          flush=True)

    head("① 기준선 — 진입 방식 교체")
    mkt, _ = run(NEW_PAIRS, entry="market")
    print(line("시장가(현행)", stat(mkt)), flush=True)
    lim, fr = run(NEW_PAIRS, entry="limit", ttl=3)
    print(line("지정가 라벨 TTL3", stat(lim), fr), flush=True)

    head("② 체결 규칙 민감도 (지정가 최대 함정)")
    for f_ in ("touch", "strict", "deep"):
        t, fr_ = run(NEW_PAIRS, entry="limit", fill=f_, ttl=3)
        print(line(f"fill={f_}", stat(t), fr_), flush=True)

    head("③ 연도 일관성")
    yrs = sorted({t[1] for t in lim} | {t[1] for t in mkt})
    for y in yrs:
        sm = stat([t for t in mkt if t[1] == y])
        sl_ = stat([t for t in lim if t[1] == y])
        a = f"{sm['net']:+.1f}%" if sm else "n/a"
        b = f"{sl_['net']:+.1f}%" if sl_ else "n/a"
        flag = "" if (sl_ and sl_["net"] > 0) else "  ←적자"
        print(f"  {y}   시장가 {a:>10}   지정가 {b:>10}{flag}", flush=True)

    head("④ 페어별 (지정가)")
    for sym in NEW_PAIRS:
        t, fr_ = run([sym], entry="limit", ttl=3)
        print(line(sym, stat(t), fr_), flush=True)

    head("⑤ 롱/숏 분리")
    for so in ("long", "short"):
        t, fr_ = run(NEW_PAIRS, entry="limit", ttl=3, side_only=so)
        print(line(f"지정가 {so}", stat(t), fr_), flush=True)
        t2, _ = run(NEW_PAIRS, entry="market", side_only=so)
        print(line(f"시장가 {so}", stat(t2)), flush=True)

    head("⑥ TTL 민감도")
    for ttl in (1, 2, 3, 4, 6, 12):
        t, fr_ = run(NEW_PAIRS, entry="limit", ttl=ttl)
        print(line(f"TTL {ttl}봉", stat(t), fr_), flush=True)

    print("\n=== ⑦ 부트스트랩 (거래 복원추출 5000회) ===", flush=True)
    for tag, tr in (("시장가", mkt), ("지정가", lim)):
        v = np.array([x[0] for x in tr], float)
        if len(v) < 10:
            print(f"  {tag}: 표본부족", flush=True)
            continue
        boot = 100 * RNG.choice(v, size=(5000, len(v)), replace=True).sum(axis=1)
        p5, p50, p95 = np.percentile(boot, [5, 50, 95])
        pos = 100 * (boot > 0).mean()
        print(f"  {tag:<6} 중앙 {p50:>+9.1f}%  5% {p5:>+9.1f}%  95% {p95:>+9.1f}%"
              f"   흑자확률 {pos:>5.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
