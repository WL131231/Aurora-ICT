"""#AUTONOMOUS 2026-08-02: Cursus 하이켄아시 재검증 — 지정가 전환 후에도 유효한가.

배경: HA 는 개발자 제안으로 7/31 배포됐고, 당시 근거는 **시장가 하니스** 숫자였다
(거래 8,151→5,591건 -31%, net -23,113%→-16,018%). 그 직후 진입이 지정가로 바뀌면서
봇 성격이 달라졌다 — CSI 횡보 게이트는 시장가에서 도움(-437→-273%)이었는데 지정가에서
해로웠다(+24.8→-41.6%). 같은 뒤집힘이 HA 에도 일어날 수 있으므로 재측정한다.

지금 라이브에 켜져 있는 설정이라 우선순위가 높다. HA 가 지정가에서 해롭다면 즉시
꺼야 하고, 유효하다면 근거를 현행 하니스 숫자로 갱신해야 한다(정합 규칙).

축: 진입 방식(지정가/시장가) × HA(on/off). 나머지는 라이브 정합 고정.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, "scripts")
import cursus_limit_battery as B  # noqa: E402
import dst_trend_bt_clamped as DST  # noqa: E402
from cursus_dev_changes import NEW_PAIRS, heikin_ashi  # noqa: E402

RNG = np.random.default_rng(20260802)


def simulate(sym: str, *, use_ha: bool, entry: str = "limit", ttl: int = 3,
             side_only: str | None = None):
    """B.simulate 와 동일 엔진 + HA on/off 토글."""
    df = DST._load_1h(sym)
    sig = DST._signals(heikin_ashi(df) if use_ha else df)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c_arr = df["close"].to_numpy(float)
    o_arr = df["open"].to_numpy(float)
    years = df.index.year.to_numpy()
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
    return trades, n_sig, n_fill


def run(pairs, **kw):
    allt: list[tuple[float, int, str]] = []
    s = f = 0
    for sym in pairs:
        t, ns, nf = simulate(sym, **kw)
        allt += t
        s += ns
        f += nf
    return allt, 100 * f / max(s, 1)


def main() -> int:
    print("=== Cursus 하이켄아시 재검증 (지정가 전환 후) ===", flush=True)
    print("  10x · size 10% · 4분할TP · SL 2% · maker/taker 분리", flush=True)

    B.head("① 진입방식 × HA (2×2)")
    cells = {}
    for entry in ("market", "limit"):
        for ha in (False, True):
            t, fr = run(NEW_PAIRS, use_ha=ha, entry=entry)
            cells[(entry, ha)] = t
            tag = ("시장가" if entry == "market" else "지정가") + (" HA on" if ha else " HA off")
            print(B.line(tag, B.stat(t), fr if entry == "limit" else None), flush=True)

    print("\n  → HA 효과(net 차이)", flush=True)
    for entry in ("market", "limit"):
        s0, s1 = B.stat(cells[(entry, False)]), B.stat(cells[(entry, True)])
        if s0 and s1:
            lab = "시장가" if entry == "market" else "지정가"
            d = s1["net"] - s0["net"]
            print(f"     {lab}: {s0['net']:+.1f}% → {s1['net']:+.1f}%  ({d:+.1f}%p)"
                  f"  거래 {s0['n']}→{s1['n']}", flush=True)

    base, ha_t = cells[("limit", False)], cells[("limit", True)]
    B.head("② 연도 일관성 (지정가 · HA off → on)")
    for y in sorted({x[1] for x in base}):
        a, b = B.stat([x for x in base if x[1] == y]), B.stat([x for x in ha_t if x[1] == y])
        sa = f"{a['net']:+.1f}%" if a else "n/a"
        sb = f"{b['net']:+.1f}%" if b else "n/a"
        mk = ("  개선" if (a and b and b["net"] > a["net"]) else "  ←악화") if (a and b) else ""
        print(f"  {y}   HA off {sa:>10}   HA on {sb:>10}{mk}", flush=True)

    B.head("③ 롱/숏 분리 (지정가)")
    for so in ("long", "short"):
        for ha in (False, True):
            t, fr = run(NEW_PAIRS, use_ha=ha, entry="limit", side_only=so)
            print(B.line(f"{so} HA {'on' if ha else 'off'}", B.stat(t), fr), flush=True)

    B.head("④ 페어별 (지정가 · off → on)")
    for sym in NEW_PAIRS:
        t0, _ = run([sym], use_ha=False, entry="limit")
        t1, _ = run([sym], use_ha=True, entry="limit")
        s0, s1 = B.stat(t0), B.stat(t1)
        if s0 is None or s1 is None:
            print(f"  {sym:<26} 표본부족", flush=True)
            continue
        print(f"  {sym:<26}{s0['net']:>+10.1f}% → {s1['net']:>+10.1f}%"
              f"   {'개선' if s1['net'] > s0['net'] else '←악화'}", flush=True)

    print("\n=== ⑤ 부트스트랩 (지정가, 복원추출 5000회) ===", flush=True)
    for tag, tr in (("HA off", base), ("HA on", ha_t)):
        v = np.array([x[0] for x in tr], float)
        boot = 100 * RNG.choice(v, size=(5000, len(v)), replace=True).sum(axis=1)
        p5, p50, p95 = np.percentile(boot, [5, 50, 95])
        print(f"  {tag:<8} 중앙 {p50:>+9.1f}%  5% {p5:>+9.1f}%  95% {p95:>+9.1f}%"
              f"   흑자확률 {100 * (boot > 0).mean():>5.1f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
