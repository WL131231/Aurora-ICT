"""#AUTONOMOUS 2026-07-31: Cursus 지정가 진입 — 후보 좌표 × 대기시간 지도.

개발자: "차트는 하이켄아시 → 눌림 가격대에 들어갈 수 있음, 지정가",
        "매수 지점은 듀얼슈퍼트렌드 **지표**에 나온다".
원본 `매매기법.py`(163줄) 확인 결과 매수 지점 계산이 **없다** — `on_bar` 는
`entry = row["close"]`(신호봉 종가) 뿐이다. 즉 개발자가 쓰는 트뷰 지표 기능이며
계산식 회신 대기 중.

이 스크립트는 **답이 오면 바로 대조**할 수 있도록 유력 후보를 미리 격자로 돌린다.
후보 좌표(신호 발생 봉 기준, 롱이면 아래쪽 = 눌림):
    C0 신호봉 종가       — 대조군(사실상 현행 시장가와 동일 위치)
    C1 ST1 라인          — ATR14×2.0, 타이트한 지지
    C2 ST2 라인          — ATR14×3.0, 깊은 지지
    C3 HA 시가           — HA 몸통 하단(상승 시). ⚠️ 계산값이라 미체결 위험
    C4 신호봉 중간(hl2)  — 단순 절충
대기: 신호 후 N봉 안에 실제 low(롱)/high(숏)가 지정가에 닿으면 체결, 아니면 취소.

비용: maker 왕복 0.02% + 슬리피지 0 (지정가 정합). 미체결은 거래 자체가 없다.
★ 이 실험의 핵심은 **트레이드오프**다 — 깊은 좌표일수록 진입가는 유리하지만
   미체결이 늘어 신호를 놓친다. 어느 지점이 최적인지 지도를 만든다.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import cursus_live_parity_bt as M  # noqa: E402
import dst_trend_bt_clamped as DST  # noqa: E402
from cursus_dev_changes import NEW_PAIRS, heikin_ashi  # noqa: E402

MAKER_RT = 0.0002        # maker 왕복 0.01%×2
LEVERAGE, SIZE_PCT = 20.0, 0.9   # 상대 비교용 — 노출 변경은 net% 를 비례 스케일만 함
SL_PCT = 0.02
TP_PCTS = (0.01, 0.02, 0.03, 0.04)
TP_FRAC = 0.25
CANDS = ("C0 신호봉 종가", "C1 ST1 라인", "C2 ST2 라인", "C3 HA 시가", "C4 신호봉 hl2")


def limit_price(cand: str, i: int, d: int, sig, ha) -> float:
    """후보별 지정가. d=+1 롱(아래쪽), -1 숏(위쪽)."""
    if cand.startswith("C0"):
        return float(sig["close"].iloc[i])
    if cand.startswith("C1"):
        return float(sig["st1"].iloc[i])
    if cand.startswith("C2"):
        return float(sig["st2"].iloc[i])
    if cand.startswith("C3"):
        return float(ha["open"].iloc[i])
    return float((sig["high"].iloc[i] + sig["low"].iloc[i]) / 2.0)


def run(pairs, cand: str, ttl: int):
    """HA 신호 + 지정가 진입. 반환 (stat, 체결률)."""
    allt: list[tuple[float, int]] = []
    n_sig = n_fill = 0
    for sym in pairs:
        df = DST._load_1h(sym)
        ha = heikin_ashi(df)
        sig = DST._signals(ha)                     # 신호·ST 는 HA 기준
        for c in ("open", "high", "low", "close"):
            sig[c] = df[c].to_numpy()              # 가격은 실제로 교체
        st1 = DST._supertrend(ha, 2.0, 14).to_numpy()
        st2 = DST._supertrend(ha, 3.0, 14).to_numpy()
        sig["st1"], sig["st2"] = st1, st2
        h, lo = sig["high"].to_numpy(), sig["low"].to_numpy()
        c_arr, o_arr = sig["close"].to_numpy(), sig["open"].to_numpy()
        years = sig.index.year.to_numpy()
        buy = np.concatenate([[False], sig["buy_sig"].to_numpy()[:-1]])
        sell = np.concatenate([[False], sig["sell_sig"].to_numpy()[:-1]])
        n = len(c_arr)
        side = None; entry = stop = 0.0; tps: list[float] = []
        hits = 0; remain = 1.0; acc = 0.0; ei = 0
        i = 1
        while i < n:
            if side is not None:
                rev = bool(sell[i]) if side == "long" else bool(buy[i])
                sgn = 1.0 if side == "long" else -1.0
                sl_hit = (lo[i] <= stop) if side == "long" else (h[i] >= stop)
                if sl_hit:
                    raw = (stop - entry) / entry * sgn
                    acc += raw * SIZE_PCT * remain * LEVERAGE - MAKER_RT * SIZE_PCT * remain * LEVERAGE
                    allt.append((acc, int(years[i]))); side = None; acc = 0.0; remain = 1.0; hits = 0
                else:
                    done = False
                    while hits < len(tps):
                        tp = tps[hits]
                        got = (h[i] >= tp) if side == "long" else (lo[i] <= tp)
                        if not got:
                            break
                        hits += 1
                        raw = (tp - entry) / entry * sgn
                        frac = remain if hits >= len(tps) else TP_FRAC
                        acc += raw * SIZE_PCT * frac * LEVERAGE - MAKER_RT * SIZE_PCT * frac * LEVERAGE
                        remain = 0.0 if hits >= len(tps) else max(remain - TP_FRAC, 0.0)
                        if hits >= len(tps):
                            allt.append((acc, int(years[i]))); side = None; acc = 0.0
                            remain = 1.0; hits = 0; done = True
                            break
                        if hits >= 2:
                            ladder = tps[hits - 2]
                            if (side == "long" and ladder > stop) or (side == "short" and ladder < stop):
                                stop = ladder
                    if not done and side is not None and rev:
                        raw = (o_arr[i] - entry) / entry * sgn
                        acc += raw * SIZE_PCT * remain * LEVERAGE - MAKER_RT * SIZE_PCT * remain * LEVERAGE
                        allt.append((acc, int(years[i]))); side = None; acc = 0.0; remain = 1.0; hits = 0
            if side is None and (buy[i] or sell[i]):
                n_sig += 1
                d = 1 if buy[i] else -1
                px = limit_price(cand, i, d, sig, ha)
                if not np.isfinite(px) or px <= 0:
                    i += 1; continue
                # TTL 내 체결 판정 — 롱이면 저가가 지정가 이하로, 숏이면 고가가 이상으로
                fill = None
                for j in range(i, min(i + ttl + 1, n)):
                    if (d == 1 and lo[j] <= px) or (d == -1 and h[j] >= px):
                        fill = j; break
                if fill is None:
                    i += 1; continue
                n_fill += 1
                entry = px
                side = "long" if d == 1 else "short"
                sgn = float(d)
                stop = entry * (1 - sgn * SL_PCT)
                tps = [entry * (1 + sgn * p) for p in TP_PCTS]
                hits = 0; remain = 1.0; acc = 0.0; ei = fill
                i = fill + 1
                continue
            i += 1
    return M.stat(allt), (100 * n_fill / max(n_sig, 1))


def main() -> int:
    print("=== Cursus 지정가 후보 지도 (HA 신호 + maker 비용, 개발자안 페어) ===", flush=True)
    print("  ※ 매수 지점 회신 전 사전 탐색 — 답 오면 해당 좌표를 채택한다.\n", flush=True)
    print(f"  {'후보':<14}{'TTL':>4}{'n':>6}{'체결률':>7}{'net':>10}{'승률':>6}{'RR':>6}{'분기':>6}  판정",
          flush=True)
    for cand in CANDS:
        for ttl in (1, 3, 6, 12):
            s, fr = run(NEW_PAIRS, cand, ttl)
            if s is None:
                print(f"  {cand:<14}{ttl:>4}  거래 없음", flush=True)
                continue
            be = (100 - s["wr"]) / max(s["wr"], 1e-9)
            mark = "★흑자권" if s["rr"] > be else "적자"
            print(f"  {cand:<14}{ttl:>4}{s['n']:>6}{fr:>6.0f}%{s['net']:>+10.0f}%"
                  f"{s['wr']:>5.0f}%{s['rr']:>6.2f}{be:>6.2f}  {mark}", flush=True)
        print("", flush=True)
    print("  ※ 깊은 좌표일수록 진입가는 유리하나 체결률이 떨어져 신호를 놓친다 —", flush=True)
    print("     net 과 체결률을 함께 볼 것.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
