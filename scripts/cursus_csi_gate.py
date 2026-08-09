"""#AUTONOMOUS 2026-07-30: Cursus + CSI 횡보 게이트 (파트너 지시 — Cursus 개발자 합의).

파트너: "CURSUS 는 횡보에 너무 취약해서 넣어야겠다. 횡보 탐지 → 진입 차단."

근거(라이브 실측 7/30): Cursus 1.0 실현 **RR 0.18** (손절 평균 -16.86 vs 익절 +3.11),
net **-865** — 전체 손실의 최대 처. 손절 한 번에 익절 5.4번이 날아가는 구조.

⚠️ Origo 와 정반대 예상인 이유: Origo(FVG 되돌림)는 **압축→확장 초입**을 먹어 CSI 가
찾아낸 횡보가 오히려 수익처였다(7/30 검증: CSI 홀로 잡은 55건 건당 +0.259, 승률 64%).
Cursus 는 **DualST 추세 추종** — 추세가 없으면 ST 라인이 계속 뒤집히며 휩쏘로 깎인다.
같은 게이트가 봇에 따라 반대로 작동할 수 있으므로 Cursus 에서 별도 검증한다.

하니스: dst_trend_bt_clamped.py (**유령체결 수정판** — PHANTOM-FIX/FIX2 적용본).
        원본 dst_trend_bt.py 는 ST 플립 시 체결 불가 스탑가로 익절을 기록해 5년 +218만
        이라는 가짜 결과를 냈다(7/27 무효 판정). 정직 하니스만 사용.
게이트: CSI(1h, 재료 8종 로지스틱) >= thr 이면 **신규 진입 skip** (보유분은 유지).
        직전 완결봉 기준(shift 1) — 인과.
판정: net 개선 + 페어 과반 개선 + 연도 일관 + RR 개선. 손절 평균 축소가 핵심 지표.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import dst_trend_bt_clamped as DST  # noqa: E402
from chop_state_index import csi_series, fit_csi, load_1h  # noqa: E402

from aurora.backtest.cost import apply_costs, apply_slippage, slip_pct  # noqa: E402

PAIRS = DST.PAIRS
FUNDING_PER_HOUR = getattr(DST, "FUNDING_PER_HOUR", 0.0)


def run_gated(df: pd.DataFrame, csi: np.ndarray, thr: float,
              trail_mult: float = 3.0) -> list[tuple[float, int]]:
    """dst_trend_bt_clamped._run 이식 + CSI 진입 게이트.

    thr >= 1.0 이면 게이트 비활성(기준선 재현).
    """
    sig = DST._signals(df)
    trail = DST._supertrend(sig, trail_mult, DST.ATR_PERIOD)
    h = sig["high"].values; low = sig["low"].values
    c = sig["close"].values; o = sig["open"].values
    stop_arr = trail.values
    years = sig.index.year.values
    buy = np.concatenate([[False], sig["buy_sig"].values[:-1]])
    sell = np.concatenate([[False], sig["sell_sig"].values[:-1]])
    trades: list[tuple[float, int]] = []
    side: str | None = None
    entry = stop = 0.0
    entry_i = 0
    skipped = 0
    for i in range(1, len(c)):
        s_now = stop_arr[i]
        if np.isnan(s_now):
            continue
        if side is not None:
            if side == "long":
                hit = low[i] <= stop
                rev = bool(sell[i])
            else:
                hit = h[i] >= stop
                rev = bool(buy[i])
            flip = (side == "long" and s_now > c[i]) or (side == "short" and s_now < c[i])
            if hit or rev or flip:
                exit_raw = stop if hit else o[i]
                slp = slip_pct(h[i], low[i], c[i])
                exit_px = apply_slippage(exit_raw, side, "exit", slp)
                raw = (exit_px - entry) / entry
                if side == "short":
                    raw = -raw
                net, _ = apply_costs(raw, DST.SIZE_PCT, DST.LEVERAGE)
                net -= (i - entry_i) * FUNDING_PER_HOUR * DST.SIZE_PCT * DST.LEVERAGE
                trades.append((net, int(years[i])))
                side = None
            elif side == "long":
                if s_now <= c[i]:
                    stop = max(stop, s_now)
            else:
                if s_now >= c[i]:
                    stop = min(stop, s_now)
        if side is None:
            # ★ CSI 게이트 — 횡보 인식 시 신규 진입 차단
            if (buy[i] or sell[i]) and thr < 1.0 and not np.isnan(csi[i]) and csi[i] >= thr:
                skipped += 1
                continue
            if buy[i]:
                slp = slip_pct(h[i], low[i], c[i])
                entry = apply_slippage(o[i], "long", "entry", slp)
                side = "long"
                stop = min(s_now, entry * 0.98) if not np.isnan(s_now) else entry * 0.98
                entry_i = i
            elif sell[i]:
                slp = slip_pct(h[i], low[i], c[i])
                entry = apply_slippage(o[i], "short", "entry", slp)
                side = "short"
                stop = max(s_now, entry * 1.02) if not np.isnan(s_now) else entry * 1.02
                entry_i = i
    return trades, skipped


def stat(tr):
    if not tr:
        return None
    nets = [n for n, _ in tr]
    net = sum(nets)
    w = [n for n in nets if n > 0]; l = [n for n in nets if n < 0]
    rr = (np.mean(w) / abs(np.mean(l))) if w and l else float("nan")
    eq = pk = mdd = 0.0
    for n in nets:
        eq += n; pk = max(pk, eq); mdd = max(mdd, pk - eq)
    ys: dict[int, float] = {}
    for n, y in tr:
        ys[y] = ys.get(y, 0.0) + n
    return dict(n=len(tr), net=net * 100, wr=100 * len(w) / len(tr), rr=rr,
                avgw=np.mean(w) * 100 if w else 0.0, avgl=np.mean(l) * 100 if l else 0.0,
                mdd=mdd * 100, ypos=sum(1 for v in ys.values() if v > 0), ytot=len(ys),
                ys=" ".join(f"{k}:{v * 100:+.0f}" for k, v in sorted(ys.items())))


def line(s):
    if s is None:
        return "거래 없음"
    return (f"n={s['n']:4d} net={s['net']:+8.1f}% 승률={s['wr']:3.0f}% RR={s['rr']:4.2f} "
            f"익평균={s['avgw']:+6.2f} 손평균={s['avgl']:+6.2f} MDD={s['mdd']:6.1f} "
            f"연도{s['ypos']}/{s['ytot']}")


def main() -> int:
    print("CSI 모델 학습(앞 70%)...", flush=True)
    model = fit_csi(PAIRS)
    THRS = [1.0, 0.45, 0.5, 0.55, 0.6, 0.65]
    agg: dict[float, list] = {t: [] for t in THRS}
    per_sym: dict[float, dict[str, float]] = {t: {} for t in THRS}
    for sym in PAIRS:
        df = DST._load_1h(sym)
        # CSI 를 1h 격자에 정렬 + 직전 완결봉 기준(shift 1 — 인과)
        cs = csi_series(load_1h(sym), model).reindex(df.index, method="ffill").shift(1)
        csi = cs.to_numpy()
        for thr in THRS:
            tr, sk = run_gated(df, csi, thr)
            agg[thr].extend(tr)
            per_sym[thr][sym] = sum(n for n, _ in tr) * 100
        print(f"  {sym} 완료", flush=True)

    print("\n===== Cursus × CSI 게이트 (7페어 5년 1h, 유령체결 수정판) =====", flush=True)
    base = stat(agg[1.0])
    print(f"  {'기준선(게이트 없음)':<20} {line(base)}", flush=True)
    for thr in THRS[1:]:
        s = stat(agg[thr])
        imp = sum(1 for sym in PAIRS if per_sym[thr][sym] > per_sym[1.0][sym])
        mark = "★" if s and base and s["net"] > base["net"] and imp >= 4 else " "
        print(f" {mark}{'CSI>=' + str(thr) + ' 차단':<20} {line(s)} 페어개선 {imp}/7", flush=True)
    print("\n  ※ 핵심 지표 = 손평균 축소(라이브 -16.86 이 문제) + net 개선 + 페어 과반", flush=True)
    print("\n  [연도별]", flush=True)
    for thr in THRS:
        s = stat(agg[thr])
        if s:
            nm = "기준선" if thr == 1.0 else f"CSI>={thr}"
            print(f"    {nm:<12} {s['ys']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
