"""#AUTONOMOUS 2026-08-06: Cursus(DualST 추세) 를 주식에 적용 — 나스닥/코스피/코스닥.

파트너 요청: "주식에서도 통할지 궁금한데".

## 크립토 백테를 그대로 옮기면 안 되는 것들 (전부 반영)

① **오버나이트 갭** — 크립토는 24시간이라 SL 이 대개 그 가격에 체결되지만, 주식은
   장 마감 사이 갭이 뛴다. SL 을 건너뛴 갭은 **다음 봉 시가로 체결**시켰다.
   이걸 무시하면 손실이 과소평가되어 결과가 낙관적으로 부풀려진다.
② **레버리지 1배** — 주식 현물. 크립토 10~20배 설정을 그대로 쓰면 무의미하다.
③ **시장별 비용** — 미국은 제로커미션(스프레드·슬리피지만), 한국은 **거래세**가
   붙는다(매도 0.18% + 수수료). 한국 쪽이 왕복 5배 비싸다.
④ **공매도 제약** — 한국 개인은 사실상 공매도가 막혀 있다. 롱/숏을 분리해 보고하고,
   한국 시장은 **롱 성적만이 실현 가능**하다고 판단한다.

## 한계
- 1h 는 야후가 730일까지만 준다(실제 확보 ~2~3년). 일봉은 10년.
- 배당·액면분할은 auto_adjust=False 원가격 기준(분할 종목은 왜곡 가능).
- 유동성·호가단위·상하한가·거래정지 미반영.
"""

from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import dst_trend_bt_clamped as DST  # noqa: E402
from cursus_dev_changes import heikin_ashi  # noqa: E402
from stock_fetch import MARKETS, fetch  # noqa: E402

# 왕복 비용 — 미국: 제로커미션(스프레드·슬리피지 추정), 한국: 수수료+거래세(매도 0.18%)
COST = {"NASDAQ": 0.0004, "KOSPI": 0.0021, "KOSDAQ": 0.0021}
LEVERAGE, SIZE = 1.0, 1.0          # 주식 현물
SL_PCT = 0.02
TP_PCTS = (0.01, 0.02, 0.03, 0.04)
TP_FRAC = 0.25
RNG = np.random.default_rng(20260806)


def simulate(df: pd.DataFrame, cost: float, *, use_ha: bool = True,
             side_only: str | None = None, gap_fill: bool = True):
    """Cursus 원본 엔진(고정 SL 2% + 4분할 TP + 래더) — 주식용.

    Args:
        gap_fill: True 면 SL/TP 를 갭으로 건너뛴 경우 **시가**로 체결(정직).
                  False 면 크립토처럼 지정가 그대로 체결(비교용 — 갭 효과 측정).
    """
    sig = DST._signals(heikin_ashi(df) if use_ha else df)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    years = df.index.year.to_numpy()
    buy = np.concatenate([[False], sig["buy_sig"].to_numpy()[:-1]])
    sell = np.concatenate([[False], sig["sell_sig"].to_numpy()[:-1]])
    n = len(c)

    trades: list[tuple[float, int, str, int]] = []
    side = None
    e = stop = 0.0
    ent_i = 0
    tps: list[float] = []
    hits = 0
    remain = 1.0
    acc = 0.0
    i = 1
    while i < n:
        if side is not None:
            sgn = 1.0 if side == "long" else -1.0
            rev = bool(sell[i]) if side == "long" else bool(buy[i])
            sl_hit = (lo[i] <= stop) if side == "long" else (h[i] >= stop)
            if sl_hit:
                # ① 갭 — 시가가 이미 SL 너머면 그 가격에 못 팔았다. 시가로 체결.
                px = stop
                if gap_fill:
                    gapped = (o[i] < stop) if side == "long" else (o[i] > stop)
                    if gapped:
                        px = o[i]
                raw = (px - e) / e * sgn
                acc += (raw - cost) * SIZE * remain * LEVERAGE
                trades.append((acc, int(years[i]), side, i - ent_i))
                side = None; acc = 0.0; remain = 1.0; hits = 0
            else:
                done = False
                while hits < len(tps):
                    tp = tps[hits]
                    if not ((h[i] >= tp) if side == "long" else (lo[i] <= tp)):
                        break
                    hits += 1
                    px = tp
                    if gap_fill:   # 갭 상승/하락으로 TP 를 넘겨 열면 유리하게 체결
                        better = (o[i] > tp) if side == "long" else (o[i] < tp)
                        if better:
                            px = o[i]
                    raw = (px - e) / e * sgn
                    frac = remain if hits >= len(tps) else TP_FRAC
                    acc += (raw - cost) * SIZE * frac * LEVERAGE
                    remain = 0.0 if hits >= len(tps) else max(remain - TP_FRAC, 0.0)
                    if hits >= len(tps):
                        trades.append((acc, int(years[i]), side, i - ent_i))
                        side = None; acc = 0.0; remain = 1.0; hits = 0
                        done = True
                        break
                    if hits >= 2:
                        lad = tps[hits - 2]
                        if (side == "long" and lad > stop) or (side == "short" and lad < stop):
                            stop = lad
                if not done and side is not None and rev:
                    raw = (o[i] - e) / e * sgn
                    acc += (raw - cost) * SIZE * remain * LEVERAGE
                    trades.append((acc, int(years[i]), side, i - ent_i))
                    side = None; acc = 0.0; remain = 1.0; hits = 0

        if side is None and (buy[i] or sell[i]):
            d = 1 if buy[i] else -1
            want = "long" if d == 1 else "short"
            if side_only is not None and want != side_only:
                i += 1
                continue
            e = c[i - 1]          # 신호봉 종가 진입(시장가) — 다음 봉부터 관리
            ent_i = i
            side = want
            sgn = float(d)
            stop = e * (1 - sgn * SL_PCT)
            tps = [e * (1 + sgn * p) for p in TP_PCTS]
            hits = 0; remain = 1.0; acc = 0.0
            i += 1
            continue
        i += 1
    return trades


def stat(tr):
    if len(tr) < 10:
        return None
    v = np.array([t[0] for t in tr], float)
    win, loss = v[v > 0], v[v <= 0]
    dd = np.maximum.accumulate(np.cumsum(v)) - np.cumsum(v)
    return {"n": len(v), "net": 100 * v.sum(), "wr": 100 * len(win) / len(v),
            "rr": (win.mean() / abs(loss.mean())) if len(win) and len(loss) else float("nan"),
            "mdd": 100 * dd.max(), "per": 100 * v.mean()}


def line(tag, s):
    if s is None:
        return f"  {tag:<22} 표본부족"
    be = (100 - s["wr"]) / max(s["wr"], 1e-9)
    mark = "★" if s["net"] > 0 and s["rr"] > be else " "
    return (f"  {tag:<22}{s['n']:>6}{s['net']:>+10.1f}%{s['per']:>+7.2f}%"
            f"{s['wr']:>5.0f}%{s['rr']:>6.2f}{s['mdd']:>8.1f}% {mark}")


def run_market(mkt: str, tickers: list[str], interval: str, **kw):
    allt = []
    for t in tickers:
        try:
            df = fetch(t, interval=interval,
                       period="730d" if interval == "1h" else "10y")
        except Exception:  # noqa: BLE001
            continue
        if df is None or len(df) < 200:
            continue
        allt += simulate(df, COST[mkt], **kw)
    return allt


def main() -> int:
    print("=== Cursus(DualST) × 주식 ===", flush=True)
    print(f"  레버리지 {LEVERAGE:.0f}x · 비용 US {COST['NASDAQ']:.2%} / KR "
          f"{COST['KOSPI']:.2%}(거래세 포함) · 갭은 시가 체결", flush=True)
    hdr = f"  {'':<22}{'n':>6}{'net':>11}{'건당':>7}{'승률':>5}{'RR':>6}{'MDD':>9}"

    for interval, label in (("1d", "일봉 10년"), ("1h", "1시간봉 ~3년")):
        print(f"\n### {label}", flush=True)
        print(hdr, flush=True)
        for mkt, ts in MARKETS.items():
            tr = run_market(mkt, ts, interval)
            print(line(mkt, stat(tr)), flush=True)

        print(f"\n  [롱/숏 분리 — 한국은 개인 공매도 제약]", flush=True)
        print(hdr, flush=True)
        for mkt, ts in MARKETS.items():
            for so in ("long", "short"):
                tr = run_market(mkt, ts, interval, side_only=so)
                print(line(f"{mkt} {so}", stat(tr)), flush=True)

        print(f"\n  [갭 효과 — 갭 무시(크립토식) vs 시가 체결(정직)]", flush=True)
        print(hdr, flush=True)
        for mkt, ts in MARKETS.items():
            a = stat(run_market(mkt, ts, interval, gap_fill=False))
            b = stat(run_market(mkt, ts, interval, gap_fill=True))
            print(line(f"{mkt} 갭무시", a), flush=True)
            print(line(f"{mkt} 갭반영", b), flush=True)
            if a and b:
                print(f"     → 갭이 {b['net'] - a['net']:+.1f}%p 를 먹는다", flush=True)

        print(f"\n  [하이켄아시 on/off]", flush=True)
        print(hdr, flush=True)
        for mkt, ts in MARKETS.items():
            for ha in (False, True):
                tr = run_market(mkt, ts, interval, use_ha=ha)
                print(line(f"{mkt} HA {'on' if ha else 'off'}", stat(tr)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
