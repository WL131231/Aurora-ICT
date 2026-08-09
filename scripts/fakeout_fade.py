"""#AUTONOMOUS 2026-07-29: 가짜돌파 페이드 — 즉시역진입 vs 피보 되돌림 진입 (파트너 지시).

분출 탐지 결과(7/29): 횡보 중 이탈을 **따라가면** 전 신호 큰 음수(B6 박스+볼륨 h6
-0.111%, 승률 37%, t-32). 통계가 가리키는 방향 = **반대(페이드)**.
파트너: "반대로 즉시 진입 / 반대로 가되 피보 0.618·0.707 되돌림에서 진입" 두 안 비교.

전제: CSI(1h,12봉) >= thr → **횡보 인식 상태에서만** 신호 탐색(파트너 구상 유지).
신호: 이탈봉(박스=최근24봉 Donchian 이탈, +볼륨 확인 옵션) → **이탈 반대 방향** 포지션.
진입 방식:
  I0 즉시     : 이탈 확정 다음봉 시가 (taker)
  I618/I707/I786 : 이탈 스윙(이탈봉 극단 ~ 박스 반대편) 되돌림 피보 레벨 지정가.
      롱(하단이탈 페이드) 기준 = 저점(이탈극단) + f×(박스상단 - 저점) 는 너무 멀어
      **이탈 임펄스 되돌림**으로 정의: 이탈봉 저점~고점 폭의 f 지점에 지정가.
      TTL 12봉 내 미체결이면 취소(라이브 정합).
청산: SL = 이탈 극단 ± 버퍼(0.1%), TP = 박스 중앙 / 반대편 / 1R / 2R
비용: 즉시=taker 0.11%, 지정가=maker 0.08%
판정: net>0 + 양반기 + 연도 다수 → ★ → 이후 검증배터리.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from bt_par import _load_full, _resample  # noqa: E402
from chop_state_index import csi_series, fit_csi, load_1h  # noqa: E402

PAIRS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "LINKUSDT", "HYPEUSDT"]
BOX_N = 24
TTL = 12
HOLD = 96
NOTIONAL = 18.0     # 시드% 환산 (size 0.9 × lev 20) — Origo 백테와 동일 잣대


def prep(sym: str, model):
    df = _resample(_load_full(sym))
    o = df["open"].to_numpy(); c = df["close"].to_numpy()
    h = df["high"].to_numpy(); lo = df["low"].to_numpy()
    v = df["volume"].to_numpy() if "volume" in df else np.ones(len(c))
    don_hi = pd.Series(h).rolling(BOX_N).max().shift(1).to_numpy()
    don_lo = pd.Series(lo).rolling(BOX_N).min().shift(1).to_numpy()
    volma = pd.Series(v).rolling(20).mean().to_numpy()
    csi = csi_series(load_1h(sym), model).reindex(df.index, method="ffill").to_numpy()
    return df, o, c, h, lo, v, volma, don_hi, don_lo, csi


def run(data, entry_mode: str, tp_kind: str, thr: float, need_vol: bool):
    """entry_mode: 'now' | 'f0.618' | 'f0.707' | 'f0.786'."""
    cost = 0.0011 if entry_mode == "now" else 0.0008
    out = []
    for sym, (df, o, c, h, lo, v, volma, don_hi, don_lo, csi) in data.items():
        n = len(c)
        i = BOX_N + 2
        while i < n - 2:
            if np.isnan(don_hi[i]) or np.isnan(csi[i]) or csi[i] < thr:
                i += 1; continue
            up_break = c[i] > don_hi[i]
            dn_break = c[i] < don_lo[i]
            if not (up_break or dn_break):
                i += 1; continue
            # 연속 이탈 구간의 첫 봉만 = 하나의 '분출 이벤트' (5m 남발 차단).
            prev_break = (c[i - 1] > don_hi[i - 1]) or (c[i - 1] < don_lo[i - 1])
            if prev_break:
                i += 1; continue
            if need_vol and not (v[i] > volma[i] * 2):
                i += 1; continue
            d = -1 if up_break else 1          # 페이드 = 이탈 반대
            bar_hi, bar_lo = h[i], lo[i]
            box_mid = (don_hi[i] + don_lo[i]) / 2
            # 진입가
            if entry_mode == "now":
                fill = i + 1
                entry = o[i + 1]
            else:
                f = float(entry_mode[1:])
                # 이탈 임펄스 되돌림: 상단이탈(숏 페이드)이면 고점에서 아래로 f 만큼,
                # 하단이탈(롱 페이드)이면 저점에서 위로 f 만큼.
                entry = (bar_hi - f * (bar_hi - bar_lo)) if d == -1 else \
                        (bar_lo + f * (bar_hi - bar_lo))
                fill = None
                for j in range(i + 1, min(i + 1 + TTL, n)):
                    if lo[j] <= entry <= h[j]:
                        fill = j
                        break
                if fill is None:
                    i += 1; continue
            sl = (bar_hi * 1.001) if d == -1 else (bar_lo * 0.999)
            risk = abs(sl - entry)
            # SL 최소폭 — 왕복비용의 3배(0.33%) 미만이면 손절 한 번이 비용에 먹힘.
            min_risk = entry * 0.0033
            if risk < min_risk:
                risk = min_risk
                sl = entry + d * (-risk)
            if risk <= 0:
                i += 1; continue
            tp = {"mid": box_mid,
                  "opp": don_lo[i] if d == -1 else don_hi[i],
                  "1R": entry + d * risk,
                  "2R": entry + d * 2 * risk}[tp_kind]
            if (d == 1 and tp <= entry) or (d == -1 and tp >= entry):
                i += 1; continue
            raw = 0.0
            for j in range(fill, min(fill + HOLD, n)):
                if d == 1:
                    if lo[j] <= sl:
                        raw = (sl - entry) / entry; break
                    if h[j] >= tp:
                        raw = (tp - entry) / entry; break
                else:
                    if h[j] >= sl:
                        raw = (entry - sl) / entry; break
                    if lo[j] <= tp:
                        raw = (entry - tp) / entry; break
            else:
                raw = ((c[min(fill + HOLD, n - 1)] - entry) / entry) * d
            out.append((df.index[fill], (raw - cost) * NOTIONAL * 100, sym))
            i = fill + 24   # 쿨다운 2h — 같은 국면 연타 방지
    return out


def stat(tr):
    if len(tr) < 30:
        return None
    tr = sorted(tr)
    nets = [p for _, p, _ in tr]
    net = sum(nets)
    w = sum(1 for p in nets if p > 0)
    half = len(tr) // 2
    h1 = sum(p for _, p, _ in tr[:half]); h2 = sum(p for _, p, _ in tr[half:])
    eq = pk = mdd = 0.0
    for _, p, _ in tr:
        eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    ys: dict[int, float] = {}
    for t, p, _ in tr:
        ys[t.year] = ys.get(t.year, 0.0) + p
    ypos = sum(1 for x in ys.values() if x > 0)
    return dict(n=len(tr), net=net, wr=100 * w / len(tr), h1=h1, h2=h2, mdd=mdd,
                nm=net / max(mdd, 1e-9), ys=ys,
                ok=net > 0 and h1 > 0 and h2 > 0 and ypos >= len(ys) - 1)


def line(s):
    if s is None:
        return "표본부족"
    y = " ".join(f"{k}:{v:+.0f}" for k, v in sorted(s["ys"].items()))
    return (f"n={s['n']:5d} net={s['net']:+8.1f}% 승률={s['wr']:3.0f}% H1={s['h1']:+7.1f} "
            f"H2={s['h2']:+7.1f} MDD={s['mdd']:6.1f} net/MDD={s['nm']:5.2f} [{y}]")


def main() -> int:
    model = fit_csi(PAIRS)
    data = {sym: prep(sym, model) for sym in PAIRS}
    winners = []
    for need_vol in (False, True):
        for thr in (0.5, 0.6):
            print(f"\n########## CSI>={thr} · 볼륨확인={'ON' if need_vol else 'OFF'} ##########",
                  flush=True)
            for em in ("now", "f0.618", "f0.707", "f0.786"):
                for tp in ("mid", "opp", "1R", "2R"):
                    s = stat(run(data, em, tp, thr, need_vol))
                    tag = f"{em:<7} tp={tp:<4}"
                    mark = "★" if s and s["ok"] else " "
                    print(f"  {mark}{tag} {line(s)}", flush=True)
                    if mark == "★":
                        winners.append((s["net"], thr, need_vol, em, tp, s))
    print("\n\n===== ★ 통과 요약 =====", flush=True)
    if not winners:
        print("  없음 — 전 조합 불합격", flush=True)
    for net, thr, nv, em, tp, s in sorted(winners, reverse=True, key=lambda x: x[0])[:10]:
        print(f"  CSI>={thr} vol={'ON' if nv else 'OFF'} {em} tp={tp}: {line(s)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
