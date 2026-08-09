"""#AUTONOMOUS 2026-07-31: Pine 지표 정본 기준 Cursus 매수지점 + Ichimoku/DMI 필터.

개발자가 쓰는 지표 소스(Pine v6, "Dual SuperTrend, Ichimoku and DMI Color Weighted
by DGT") 수령 후 확인한 사실:

① **신호 조건이 우리 구현과 동일** — `ta.supertrend(2,14)` / `ta.supertrend(3,14)`,
   `stBull = close > ST1 and close > ST2`, `bullBreak = stBull and not stBull[1]`.
② **"매수 지점" = 브레이크아웃 라벨이고 위치가 신호봉 저점/고점**:
       if bullBreak → label.new(bar_index, **low**,  '▲', style_label_up)
       if bearBreak → label.new(bar_index, **high**, '▼', style_label_down)
   → 롱은 신호봉 저점, 숏은 고점. "눌림 가격대" 설명과 부합하고 실제 거래된
     가격이라 지정가 체결이 가능하다(HA_open 같은 계산값과 다름).
③ **우리가 안 쓰는 레이어 2종** — SuperTrend 라인을 Ichimoku 구름 위치와
   DMI(ADX+DI) 강도로 색칠한다. 개발자가 색을 보고 거른다면 추가 필터가 된다.
       Ichimoku: aboveCloud/belowCloud + Tenkan>=Kijun
       DMI: adx >= 25(strongTrend) + DI 방향, adx 17~25 는 회색(약한 추세)

검증: 매수지점 좌표 × 대기시간 × (필터 없음/Ichimoku/DMI/둘 다).
비용은 maker 왕복 0.02%(지정가 정합), 신호는 하이켄아시(배포된 설정).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import cursus_live_parity_bt as M  # noqa: E402
import dst_trend_bt_clamped as DST  # noqa: E402
from cursus_dev_changes import NEW_PAIRS, heikin_ashi  # noqa: E402

MAKER_RT = 0.0002
LEVERAGE, SIZE_PCT = 20.0, 0.9
SL_PCT = 0.02
TP_PCTS = (0.01, 0.02, 0.03, 0.04)
TP_FRAC = 0.25


def ichimoku_state(df: pd.DataFrame):
    """Pine 정합 — donchian 평균 기반 Tenkan/Kijun/Span. 반환 (aboveCloud, belowCloud, tkBull)."""
    h, lo = df["high"], df["low"]

    def donchian(n: int) -> pd.Series:
        return (lo.rolling(n).min() + h.rolling(n).max()) / 2.0

    conv, base = donchian(9), donchian(26)
    lead1 = (conv + base) / 2.0
    lead2 = donchian(52)
    span_a = lead1.shift(25)          # displacement 26 - 1
    span_b = lead2.shift(25)
    c = df["close"]
    return (c > span_a) & (c > span_b), (c < span_a) & (c < span_b), conv >= base


def dmi_state(df: pd.DataFrame, n: int = 14, adx_n: int = 14):
    """Wilder DMI — 반환 (adx, di_plus, di_minus)."""
    h = df["high"].to_numpy(float); lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    up = np.concatenate([[0.0], np.diff(h)])
    dn = np.concatenate([[0.0], -np.diff(lo)])
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = np.concatenate([[h[0] - lo[0]], np.maximum(
        h[1:] - lo[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(lo[1:] - c[:-1])))])
    a = 1.0 / n
    atr = pd.Series(tr).ewm(alpha=a, adjust=False).mean().to_numpy()
    pdi = 100 * pd.Series(plus).ewm(alpha=a, adjust=False).mean().to_numpy() / np.maximum(atr, 1e-12)
    mdi = 100 * pd.Series(minus).ewm(alpha=a, adjust=False).mean().to_numpy() / np.maximum(atr, 1e-12)
    dx = 100 * np.abs(pdi - mdi) / np.maximum(pdi + mdi, 1e-12)
    adx = pd.Series(dx).ewm(alpha=1.0 / adx_n, adjust=False).mean().to_numpy()
    return adx, pdi, mdi


def run(pairs, cand: str, ttl: int, use_ichi: bool, use_dmi: bool):
    allt: list[tuple[float, int]] = []
    n_sig = n_fill = 0
    for sym in pairs:
        df = DST._load_1h(sym)
        ha = heikin_ashi(df)
        sig = DST._signals(ha)                     # 신호·ST 는 HA(배포 설정)
        for c_ in ("open", "high", "low", "close"):
            sig[c_] = df[c_].to_numpy()            # 가격은 실제
        above, below, tk = ichimoku_state(df)
        adx, pdi, mdi = dmi_state(df)
        h = sig["high"].to_numpy(); lo = sig["low"].to_numpy()
        c_arr = sig["close"].to_numpy(); o_arr = sig["open"].to_numpy()
        years = sig.index.year.to_numpy()
        buy = np.concatenate([[False], sig["buy_sig"].to_numpy()[:-1]])
        sell = np.concatenate([[False], sig["sell_sig"].to_numpy()[:-1]])
        ab, bl, tkb = above.to_numpy(), below.to_numpy(), tk.to_numpy()
        n = len(c_arr)
        side = None; entry = stop = 0.0; tps: list[float] = []
        hits = 0; remain = 1.0; acc = 0.0
        i = 1
        while i < n:
            if side is not None:
                rev = bool(sell[i]) if side == "long" else bool(buy[i])
                sgn = 1.0 if side == "long" else -1.0
                if (lo[i] <= stop) if side == "long" else (h[i] >= stop):
                    raw = (stop - entry) / entry * sgn
                    acc += (raw - MAKER_RT) * SIZE_PCT * remain * LEVERAGE
                    allt.append((acc, int(years[i]))); side = None; acc = 0.0
                    remain = 1.0; hits = 0
                else:
                    done = False
                    while hits < len(tps):
                        tp = tps[hits]
                        if not ((h[i] >= tp) if side == "long" else (lo[i] <= tp)):
                            break
                        hits += 1
                        raw = (tp - entry) / entry * sgn
                        frac = remain if hits >= len(tps) else TP_FRAC
                        acc += (raw - MAKER_RT) * SIZE_PCT * frac * LEVERAGE
                        remain = 0.0 if hits >= len(tps) else max(remain - TP_FRAC, 0.0)
                        if hits >= len(tps):
                            allt.append((acc, int(years[i]))); side = None; acc = 0.0
                            remain = 1.0; hits = 0; done = True
                            break
                        if hits >= 2:
                            lad = tps[hits - 2]
                            if (side == "long" and lad > stop) or (side == "short" and lad < stop):
                                stop = lad
                    if not done and side is not None and rev:
                        raw = (o_arr[i] - entry) / entry * sgn
                        acc += (raw - MAKER_RT) * SIZE_PCT * remain * LEVERAGE
                        allt.append((acc, int(years[i]))); side = None; acc = 0.0
                        remain = 1.0; hits = 0
            if side is None and (buy[i] or sell[i]):
                d = 1 if buy[i] else -1
                # ── Pine 지표 색상 레이어를 필터로 ──
                if use_ichi and not (ab[i] and tkb[i] if d == 1 else bl[i] and not tkb[i]):
                    i += 1; continue
                if use_dmi and not (adx[i] >= 25 and (pdi[i] >= mdi[i] if d == 1 else pdi[i] < mdi[i])):
                    i += 1; continue
                n_sig += 1
                # ── 매수지점 좌표 ──
                # ⚠️ buy/sell 은 이미 1봉 지연된 배열이라 **신호봉은 i-1** 이다.
                #    Pine 라벨도 신호봉(bar_index)의 low/high 에 그려진다.
                #    i 를 쓰면 아직 끝나지 않은 봉의 저점을 참조 = look-ahead
                #    (7/31 1차 실행에서 체결률 100%·net +58,828% 로 부풀었다).
                sb = i - 1
                px = (lo[sb] if d == 1 else h[sb]) if cand == "라벨(신호봉 저/고)" else \
                     (c_arr[sb] if cand == "신호봉 종가" else (h[sb] + lo[sb]) / 2)
                if not np.isfinite(px) or px <= 0:
                    i += 1; continue
                # 체결은 신호를 인지한 **다음 봉(i)** 부터 — 신호봉 자신은 제외.
                fill = None
                for j in range(i, min(i + ttl + 1, n)):
                    if (d == 1 and lo[j] <= px) or (d == -1 and h[j] >= px):
                        fill = j; break
                if fill is None:
                    i += 1; continue
                n_fill += 1
                entry = px; side = "long" if d == 1 else "short"
                sgn = float(d)
                stop = entry * (1 - sgn * SL_PCT)
                tps = [entry * (1 + sgn * p) for p in TP_PCTS]
                hits = 0; remain = 1.0; acc = 0.0
                i = fill + 1
                continue
            i += 1
    return M.stat(allt), (100 * n_fill / max(n_sig, 1))


def main() -> int:
    print("=== Pine 정본 매수지점 + Ichimoku/DMI 필터 (HA 신호 · maker 비용) ===\n", flush=True)
    print(f"  {'매수지점':<18}{'필터':<12}{'TTL':>4}{'n':>6}{'체결':>6}{'net':>10}{'승률':>6}{'RR':>6}{'분기':>6}  판정",
          flush=True)
    for cand in ("라벨(신호봉 저/고)", "신호봉 종가"):
        for fname, ui, ud in (("없음", False, False), ("Ichimoku", True, False),
                              ("DMI", False, True), ("둘다", True, True)):
            for ttl in (1, 3, 6):
                s, fr = run(NEW_PAIRS, cand, ttl, ui, ud)
                if s is None:
                    print(f"  {cand:<18}{fname:<12}{ttl:>4}  표본부족", flush=True)
                    continue
                be = (100 - s["wr"]) / max(s["wr"], 1e-9)
                mark = "★흑자권" if s["rr"] > be else "적자"
                print(f"  {cand:<18}{fname:<12}{ttl:>4}{s['n']:>6}{fr:>5.0f}%"
                      f"{s['net']:>+10.0f}%{s['wr']:>5.0f}%{s['rr']:>6.2f}{be:>6.2f}  {mark}",
                      flush=True)
        print("", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
