"""flip_close 반사실 검증 — 실제 시세로 "안 잘랐으면?" 추적 (FST #2, 2026-07-02).

라이브 진단: Origo 청산의 37%가 flip_close(HTF FVG flip watcher 조기청산),
평균 +0.39R 에서 승자 절단 (설계 TP 2.5R 의 15%). 백테스트(+124)엔 flip 이 없어
라이브 적자와 괴리. 그러나 flip 이 자른 판이 "이후 TP 로 갔을지 SL 로 뒤집혔을지"
는 데이터로만 답할 수 있다 → flip 청산 시점 이후 실제 5m 시세(Bybit 공개 API)로
TP-first vs SL-first 전수 추적.

방법 (근사 명시):
    - entry 복원: exit_price 와 pnl/qty 로 역산 (수수료 포함돼 소폭 오차).
    - SL 거리%: 같은 심볼 sl_hit 들의 |pnl/qty|/price 중앙값 (심볼별 추정).
    - TP = entry ± 2.5×SL거리 (구독제 min_rr 2.5), SL = entry ∓ 1×SL거리.
    - flip 청산 이후 최대 72h 5m 봉으로 어느 쪽 먼저 touch 했나 판정.
      (같은 봉에서 둘 다 touch 시 보수적으로 SL-first 처리)
사용: PYTHONIOENCODING=utf-8 python scripts/flip_counterfactual.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request

import pandas as pd

P = "data/live/fst_snapshots/fst_2026-07-02_all_users.csv"
RR = 2.5           # 구독제 min_rr — TP 배수
HOLD_HOURS = 72    # 반사실 추적 한도
API = "https://api.bybit.com/v5/market/kline"


def fetch_5m(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    """Bybit 공개 kline (5m). 반환: [[ts,o,h,l,c,...], ...] 오름차순."""
    sym = symbol.replace("/", "").replace(":USDT", "")
    out: list[list] = []
    cur = start_ms
    while cur < end_ms:
        url = (f"{API}?category=linear&symbol={sym}&interval=5"
               f"&start={cur}&end={end_ms}&limit=1000")
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read())
        rows = data.get("result", {}).get("list", [])
        if not rows:
            break
        rows = sorted(rows, key=lambda x: int(x[0]))
        out.extend(rows)
        nxt = int(rows[-1][0]) + 300_000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.15)  # rate limit 여유
    return out


def main() -> int:
    df = pd.read_csv(P)
    df["pnl_usdt"] = pd.to_numeric(df["pnl_usdt"], errors="coerce")
    df["qty"] = pd.to_numeric(df["qty"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    lines = [f"===== flip_close 반사실 (보유 지속 시 TP{RR}R vs SL, 실시세 {HOLD_HOURS}h) ====="]
    for model in ("Origo 1.2", "Origo 1.1"):
        m = df[(df["model"] == model) & (df["mode"] == "live")]
        sl = m[(m["event_type"] == "sl_hit") & m["qty"].gt(0) & m["price"].gt(0)]
        # 심볼별 SL 거리% 추정 — |pnl/qty|/price 중앙값
        dist = (sl["pnl_usdt"].abs() / sl["qty"] / sl["price"]).groupby(sl["symbol"]).median()
        glob = float((sl["pnl_usdt"].abs() / sl["qty"] / sl["price"]).median())
        fl = m[(m["event_type"] == "flip_close") & m["qty"].gt(0) & m["price"].gt(0)]
        res = {"tp": 0, "sl": 0, "neither": 0, "skip": 0}
        gain_tp = 0.0   # 반사실 net (R 단위)
        real_r = 0.0
        details = []
        by_tf: dict[str, dict[str, float]] = {}  # tf -> {real, hold, n}
        for _, t in fl.iterrows():
            d = float(dist.get(t["symbol"], glob))
            if not d or pd.isna(d):
                res["skip"] += 1
                continue
            exit_px = float(t["price"])
            side = 1 if str(t["direction"]).lower().startswith("l") else -1
            entry = exit_px - side * float(t["pnl_usdt"]) / float(t["qty"])
            sl_px = entry * (1 - side * d)
            tp_px = entry * (1 + side * d * RR)
            realized = side * (exit_px - entry) / (entry * d)  # 실현 R
            real_r += realized
            kl = fetch_5m(t["symbol"], int(t["ts_ms"]), int(t["ts_ms"]) + HOLD_HOURS * 3600_000)
            verdict = "neither"
            for row in kl:
                h, low = float(row[2]), float(row[3])
                hit_sl = low <= sl_px if side == 1 else h >= sl_px
                hit_tp = h >= tp_px if side == 1 else low <= tp_px
                if hit_sl:            # 동시 touch 은 보수적으로 SL 우선
                    verdict = "sl"
                    break
                if hit_tp:
                    verdict = "tp"
                    break
            res[verdict] += 1
            hold_r = {"tp": RR, "sl": -1.0, "neither": realized}[verdict]
            gain_tp += hold_r
            tf = (re.search(r"@(\w+)", str(t.get("reason", ""))) or [None, "?"])[1]
            agg = by_tf.setdefault(tf, {"real": 0.0, "hold": 0.0, "n": 0, "tp": 0, "sl": 0})
            agg["real"] += realized
            agg["hold"] += hold_r
            agg["n"] += 1
            if verdict in ("tp", "sl"):
                agg[verdict] += 1
            details.append((t["symbol"], tf, verdict, round(realized, 2)))
            print(f"  {model} {t['symbol']} @{tf} {verdict} (실현 {realized:+.2f}R)", flush=True)

        n = len(fl) - res["skip"]
        if n <= 0:
            continue
        lines.append(
            f"\n[{model}] flip 절단 {n}건 반사실: TP도달 {res['tp']}건 / SL반전 {res['sl']}건 "
            f"/ 미도달 {res['neither']}건")
        lines.append(
            f"  실현 합 {real_r:+.1f}R  vs  보유지속 합 {gain_tp:+.1f}R  "
            f"→ flip 이 {'깎은' if gain_tp > real_r else '지킨'} 가치 {abs(gain_tp - real_r):.1f}R")
        lines.append("  --- flip TF 별 (실현R합 vs 보유R합 / TP·SL 반사실) ---")
        for tf, a in sorted(by_tf.items(), key=lambda kv: kv[1]["hold"] - kv[1]["real"],
                            reverse=True):
            lines.append(
                f"  @{tf:<5} n={a['n']:>3}  실현 {a['real']:>+7.1f}R  보유 {a['hold']:>+7.1f}R"
                f"  Δ={a['hold'] - a['real']:>+6.1f}R  (TP {a['tp']} / SL {a['sl']})")

    txt = "\n".join(lines)
    with open("flip_counterfactual_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
