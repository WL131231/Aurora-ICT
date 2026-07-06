"""손절 트레이드 MFE(최대 유리 이동) 전수 분석 — 파트너 가설 검증 (2026-07-07).

가설: "분명 수익 구간인데 TP 가 너무 멀어서 손절 터지는 게 너무 많다."
검증: Origo 1.2/1.4 의 sl_hit 트레이드마다 [진입→청산] 구간 실시세(Bybit 1m)로
MFE%(진입가 대비 최대 유리 이동)를 재고, 20배 ROI 환산으로
    - +20% ROI(가격 1%) 이상 갔다가 죽은 비율
    - +40% ROI(가격 2%) 이상 갔다가 죽은 비율
을 집계. 높으면 "이익 잠금(조기 트레일/BE/TP 캡)" 처방의 근거.

사용: PYTHONIOENCODING=utf-8 python scripts/sl_mfe_analysis.py
담당: 지영민 (FST 자율연구).
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request

import pandas as pd

P = "data/live/fst_snapshots/fst_2026-07-07_all_users.csv"
API = "https://api.bybit.com/v5/market/kline"
LEV = 20  # ROI 환산 레버리지 (파트너 기준)


def fetch_1m(symbol: str, start_ms: int, end_ms: int) -> list[list]:
    sym = symbol.replace("/", "").replace(":USDT", "")
    out: list[list] = []
    cur = start_ms
    while cur < end_ms:
        url = (f"{API}?category=linear&symbol={sym}&interval=1"
               f"&start={cur}&end={min(end_ms, cur + 999 * 60_000)}&limit=1000")
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                rows = json.loads(r.read()).get("result", {}).get("list", [])
        except Exception:  # noqa: BLE001
            return out
        if not rows:
            break
        rows = sorted(rows, key=lambda x: int(x[0]))
        out.extend(rows)
        nxt = int(rows[-1][0]) + 60_000
        if nxt <= cur:
            break
        cur = nxt
        time.sleep(0.12)
    return out


def main() -> int:
    df = pd.read_csv(P)
    df["pnl_usdt"] = pd.to_numeric(df["pnl_usdt"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["mode"] == "live"].sort_values("ts_ms").reset_index(drop=True)

    lines = [f"===== sl_hit MFE 분석 (Bybit 1m 실시세, ROI={LEV}x 환산) ====="]
    for model in ("Origo 1.4", "Origo 1.2"):
        sl = df[(df["model"] == model) & (df["event_type"] == "sl_hit")
                & (~df["reason"].fillna("").str.contains("trail_stop"))]
        mfes = []
        seen_sig = set()
        for _, t in sl.iterrows():
            # 같은 (symbol, 분단위 청산시각) = 같은 신호(여러 유저) — 1회만 (신호 단위 표본)
            sig = (t["symbol"], int(t["ts_ms"]) // 60_000)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            ent = df[(df["user_code"] == t["user_code"]) & (df["symbol"] == t["symbol"])
                     & (df["event_type"].isin(["entry", "recovered"]))
                     & (df["ts_ms"] < t["ts_ms"])]
            if not len(ent):
                continue
            e = ent.iloc[-1]
            entry_px = float(e["price"])
            if not entry_px or pd.isna(entry_px):
                continue
            kl = fetch_1m(t["symbol"], int(e["ts_ms"]), int(t["ts_ms"]))
            if not kl:
                continue
            is_long = str(t["direction"]).lower().startswith("l")
            if is_long:
                best = max(float(r[2]) for r in kl)   # high
                mfe = (best - entry_px) / entry_px
            else:
                best = min(float(r[3]) for r in kl)   # low
                mfe = (entry_px - best) / entry_px
            mfes.append(mfe * 100)
            print(f"  {model} {t['symbol']} {'L' if is_long else 'S'} "
                  f"MFE={mfe*100:.2f}% (ROI {mfe*100*LEV:.0f}%)", flush=True)

        if not mfes:
            continue
        s = pd.Series(mfes)
        n = len(s)
        lines.append(
            f"\n[{model}] 손절 신호 {n}건 (유저 중복 제거)"
            f"\n  MFE 중앙값 {s.median():.2f}% (ROI {s.median()*LEV:.0f}%) / "
            f"평균 {s.mean():.2f}%"
            f"\n  +20% ROI(가격1%) 이상 갔다 죽음: {(s >= 1.0).sum()}건 ({(s >= 1.0).mean()*100:.0f}%)"
            f"\n  +40% ROI(가격2%) 이상 갔다 죽음: {(s >= 2.0).sum()}건 ({(s >= 2.0).mean()*100:.0f}%)"
            f"\n  +60% ROI(가격3%) 이상: {(s >= 3.0).sum()}건 ({(s >= 3.0).mean()*100:.0f}%)")

    txt = "\n".join(lines)
    with open("sl_mfe_result.txt", "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print("\n" + txt + "\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
